#!/usr/bin/env python3
"""Non-invasive telemetry hook for HuggingFace transformer models.

Attaches PyTorch forward hooks to a live model *without modifying its code* and
streams per-submodule traces (shapes, latency, activation stats), attention
matrices and anomalies to llm-trace over TCP (line-delimited JSON, see
docs/protocol.md).

Example:
    # terminal 1 — start the TUI listening for telemetry
    ./build/llm-trace

    # terminal 2 — stream a real GPT-2 forward/generate pass into it
    python examples/hook.py --model gpt2 --steps 20

Requires: torch, transformers  (pip install torch transformers)
"""
import argparse
import json
import socket
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


class TelemetrySink:
    """Line-delimited JSON sender. Degrades to a no-op if nothing is listening."""

    def __init__(self, host, port):
        self.sock = None
        try:
            self.sock = socket.create_connection((host, port), timeout=2.0)
            print(f"[hook] connected to llm-trace at {host}:{port}")
        except OSError as e:
            print(f"[hook] no collector at {host}:{port} ({e}); running dry")

    def send(self, event_type, payload):
        if not self.sock:
            return
        msg = json.dumps(
            {"event_type": event_type,
             "timestamp": int(time.time() * 1000),
             "payload": payload},
            separators=(",", ":"),
        ) + "\n"
        try:
            self.sock.sendall(msg.encode("utf-8"))
        except OSError:
            self.sock = None

    def close(self):
        if self.sock:
            self.sock.close()


def tensor_stats(t: torch.Tensor) -> dict:
    """Cheap numerical summary computed in-process; the raw tensor never leaves."""
    f = t.detach().float()
    n = f.numel()
    if n == 0:
        return {"mean": 0, "variance": 0, "min": 0, "max": 0, "sparsity": 0}
    near_zero = (f.abs() < 1e-6).sum().item()
    return {
        "mean": f.mean().item(),
        "variance": f.var(unbiased=False).item(),
        "min": f.min().item(),
        "max": f.max().item(),
        "sparsity": 100.0 * near_zero / n,
    }


def primary_output(out):
    """Modules may return a tensor or a tuple; pull the first tensor out."""
    if isinstance(out, torch.Tensor):
        return out
    if isinstance(out, (tuple, list)):
        for x in out:
            if isinstance(x, torch.Tensor):
                return x
    return None


def classify(name: str) -> str:
    n = name.lower()
    if "wte" in n or "embed" in n:
        return "Embedding"
    if "attn" in n or "attention" in n:
        return "SelfAttention"
    if "mlp" in n or "fc" in n or "proj" in n:
        return "MLP"
    if "ln" in n or "norm" in n:
        return "LayerNorm"
    if "lm_head" in n:
        return "LMHead"
    return "Module"


def normalize(name: str) -> str:
    """Map framework-specific module paths onto the canonical scheme used by the
    TUI topology (embed_tokens, layers.N.self_attn, layers.N.mlp, norm, lm_head)."""
    n = name
    if n in ("transformer.wte", "model.embed_tokens", "transformer.word_embeddings"):
        return "embed_tokens"
    if n in ("lm_head",):
        return "lm_head"
    if n in ("transformer.ln_f", "model.norm"):
        return "norm"
    # transformer.h.5.attn -> layers.5.self_attn, transformer.h.5.mlp -> layers.5.mlp
    parts = n.split(".")
    for tag in ("h", "layers", "layer"):
        if tag in parts:
            i = parts.index(tag)
            if i + 1 < len(parts) and parts[i + 1].isdigit():
                idx = parts[i + 1]
                tail = ".".join(parts[i + 2:])
                if "attn" in tail or "attention" in tail:
                    return f"layers.{idx}.self_attn"
                if "mlp" in tail:
                    return f"layers.{idx}.mlp"
                if "ln_1" in tail or "input_layernorm" in tail:
                    return f"layers.{idx}.input_layernorm"
                if "ln_2" in tail or "post_attention" in tail:
                    return f"layers.{idx}.post_attention_layernorm"
                return f"layers.{idx}"
    return n


def attach_hooks(model, sink, event_id, device):
    """Register forward hooks on the leaf submodules we care about."""
    starts = {}

    targets = ("wte", "embed_tokens", "attn", "attention", "mlp", "ln_1",
               "ln_2", "ln_f", "norm", "lm_head")

    def pre_hook(module, _inp):
        starts[id(module)] = time.perf_counter()

    def make_post_hook(name):
        canon = normalize(name)
        ltype = classify(name)

        def post_hook(module, inp, out):
            t0 = starts.get(id(module), time.perf_counter())
            latency_ms = (time.perf_counter() - t0) * 1000.0
            tensor = primary_output(out)
            if tensor is None:
                return
            shape = list(tensor.shape)
            nbytes = tensor.element_size() * tensor.numel()
            meta = {"shape": shape, "dtype": str(tensor.dtype).replace("torch.", ""),
                    "size_bytes": nbytes, "device": device}
            event_id[0] += 1
            sink.send("layer_trace", {
                "event_id": event_id[0],
                "layer_name": canon,
                "layer_type": ltype,
                "device": device,
                "latency_ms": latency_ms,
                "input": meta,
                "output": meta,
                "stats": tensor_stats(tensor),
            })
            stats = tensor_stats(tensor)
            if abs(stats["max"]) > 30.0:
                sink.send("anomaly", {
                    "severity": "WARNING",
                    "layer_name": canon,
                    "description": f"Exploding activations: max {stats['max']:.1f}",
                })
        return post_hook

    handles = []
    for name, module in model.named_modules():
        if any(t in name for t in targets) and len(list(module.children())) == 0 \
                or name.endswith("attn") or name.endswith("mlp"):
            handles.append(module.register_forward_pre_hook(pre_hook))
            handles.append(module.register_forward_hook(make_post_hook(name)))
    return handles


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="gpt2", help="HF model id (default gpt2)")
    ap.add_argument("--prompt", default="I want it to be keyboard driven")
    ap.add_argument("--steps", type=int, default=20, help="tokens to generate")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5005)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[hook] loading {args.model} on {device} …")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, attn_implementation="eager")
    model.to(device).eval()

    sink = TelemetrySink(args.host, args.port)
    cfg = model.config
    sink.send("model_info", {
        "name": args.model,
        "layers": getattr(cfg, "num_hidden_layers", getattr(cfg, "n_layer", 0)),
        "hidden_size": getattr(cfg, "hidden_size", getattr(cfg, "n_embd", 0)),
        "num_heads": getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 0)),
        "vocab_size": getattr(cfg, "vocab_size", 0),
        "quantization": str(next(model.parameters()).dtype).replace("torch.", ""),
    })

    event_id = [100]
    attach_hooks(model, sink, event_id, device)

    input_ids = tok(args.prompt, return_tensors="pt").input_ids.to(device)
    print(f"[hook] generating {args.steps} tokens …")
    with torch.no_grad():
        for _ in range(args.steps):
            out = model(input_ids, output_attentions=True)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)

            # Stream attention matrices (averaged over the batch) for each layer.
            tokens = [tok.decode([i]).strip() or "·"
                      for i in input_ids[0].tolist()]
            for li, attn in enumerate(out.attentions or []):
                a = attn[0]  # [heads, q, k]
                heads = min(3, a.shape[0])
                sink.send("attention_weights", {
                    "layer_name": f"layers.{li}.self_attn",
                    "num_heads": heads,
                    "token_count": len(tokens),
                    "tokens": tokens,
                    "matrices": a[:heads].cpu().tolist(),
                })

            input_ids = torch.cat([input_ids, next_id], dim=1)
            time.sleep(0.05)

    print("[hook] done. text:",
          tok.decode(input_ids[0], skip_special_tokens=True))
    sink.close()


if __name__ == "__main__":
    main()
