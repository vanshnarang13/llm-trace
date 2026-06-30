# Telemetry Protocol (v1)

`llm-trace` ingests **line-delimited JSON** over TCP (default port `5005`). Every
packet is a single-line JSON object terminated by `\n`. A packet that fails to
parse is dropped with a warning; the connection is not closed.

## Envelope

```json
{ "event_type": "<type>", "timestamp": 1717800000000, "payload": { } }
```

| field        | type    | notes                                            |
|--------------|---------|--------------------------------------------------|
| `event_type` | string  | `model_info` \| `layer_trace` \| `attention_weights` \| `anomaly` |
| `timestamp`  | integer | unix epoch **milliseconds** (optional; defaults to 0) |
| `payload`    | object  | shape depends on `event_type`                    |

## `model_info`
Sent once at the start of a session to register topology.

```json
{ "name": "llama-3-8b", "layers": 32, "hidden_size": 4096,
  "num_heads": 32, "vocab_size": 128256, "quantization": "Q4_K_M" }
```

## `layer_trace`
Emitted after a submodule finishes its forward pass.

```json
{ "event_id": 102, "layer_name": "layers.1.self_attn", "layer_type": "SelfAttention",
  "device": "CUDA:0", "latency_ms": 1.142,
  "input":  { "shape": [1, 32, 4096], "dtype": "float16", "size_bytes": 262144, "device": "CUDA:0" },
  "output": { "shape": [1, 32, 4096], "dtype": "float16", "size_bytes": 262144, "device": "CUDA:0" },
  "stats":  { "mean": 0.031, "variance": 0.533, "min": -4.12, "max": 5.73, "sparsity": 54.2 } }
```

`layer_name` follows the canonical scheme the TUI's topology tree expects:
`embed_tokens`, `layers.<i>.input_layernorm`, `layers.<i>.self_attn`,
`layers.<i>.post_attention_layernorm`, `layers.<i>.mlp`, `norm`, `lm_head`.
`stats` accepts either `min`/`max` or `min_val`/`max_val`.

## `attention_weights`
Sent alongside an attention `layer_trace`. `matrices` is `[head][query][key]`.
Senders should downsample large contexts before transmitting.

```json
{ "layer_name": "layers.1.self_attn", "num_heads": 3, "token_count": 4,
  "tokens": ["I", "want", "it", "."],
  "matrices": [ [ [1,0,0,0], [0.1,0.9,0,0], [0.05,0.15,0.8,0], [0.1,0.2,0.3,0.4] ] ] }
```

## `anomaly`
Sender-flagged instability (e.g. a CUDA OOM → CPU fallback). The detector also
generates these internally from `layer_trace` stats. `timestamp` is taken from
the envelope.

```json
{ "severity": "ERROR", "layer_name": "layers.22.mlp",
  "description": "CUDA OOM fallback: MLP executed on CPU host memory" }
```
