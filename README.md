# llm-trace

A small terminal tool for watching how local transformer models actually run. It
hooks into a model's forward pass without touching the model code, grabs what is
happening at each layer (tensor shapes, activation stats, latency, attention
weights) as tokens move through, and shows it live in a keyboard driven TUI. The
look is borrowed from tools like btop and lazygit.

It is written in C++17 and uses [FTXUI](https://github.com/ArthurSonzogni/FTXUI)
for the interface. A Python script attaches PyTorch hooks to any HuggingFace
model and streams the data over TCP. There is also a built in simulator so you
can try the UI without setting up a model first.

## What it looks like

Running `./build/llm-trace --sim`. In a real terminal the panels are colored
(teal weights, yellow stats, red anomalies); the captures below are plain text.

### Dashboard

```
 LLM-TRACE  local transformer telemetry  mode: SIM [mock tracer]                                         [1]Dashboard [2]Attention [3]CallTree [4]Help
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
╭ 1. MODEL TOPOLOGY ◀ ───────╮╭ 2. LIVE PACKET STREAM ───────────────────────────────────────────╮╭ 4. RUNTIME METRICS ──────────────────────────────╮
│Model: llama-3-8b           ││ID     TIMESTAMP    TYPE          LAYER       DEVICE   LATENCY    ││path   embed_tokens                               │
│layers 32  heads 32         │├──────────────────────────────────────────────────────────────────┤│type   Embedding                                  │
│hidden 4096  quant Q4_K_M   ││1671   20:52:20.535 LMHead        lm_head     CUDA:0   0.781 ms   ││device CUDA:0                                     │
├────────────────────────────┤│1670   20:52:20.535 RMSNorm       norm        CUDA:0   0.068 ms   │├──────────────────────────────────────────────────┤
│▪ embed_tokens  [capture]   ││1669   20:52:20.535 MLP           layers.31.mlCUDA:0   3.791 ms   ││TENSORS                                           │
│ ▸ layers.0                 ││1668   20:52:20.535 RMSNorm       layers.31.poCUDA:0   0.057 ms   ││  in   [1, 4, 4096]                               │
│   ● self_attn              ││1667   20:52:20.535 SelfAttention layers.31.seCUDA:0   2.019 ms   ││  out  [1, 4, 4096]  float16                      │
│   ● mlp                    ││1666   20:52:20.535 RMSNorm       layers.31.inCUDA:0   0.050 ms   ││  size 0.031 MB                                   │
│ ▸ layers.1                 ││1665   20:52:20.535 MLP           layers.30.mlCUDA:0   3.649 ms   │├──────────────────────────────────────────────────┤
│   ● self_attn              ││1664   20:52:20.535 RMSNorm       layers.30.poCUDA:0   0.044 ms   ││ACTIVATIONS                                       │
│   ● mlp                    ││1663   20:52:20.535 SelfAttention layers.30.seCUDA:0   1.919 ms   ││  mean 0.01000                                    │
│ ▸ layers.2                 ││1662   20:52:20.535 RMSNorm       layers.30.inCUDA:0   0.055 ms   ││  var  0.63532                                    │
│   ● self_attn              ││1661   20:52:20.535 MLP           layers.29.mlCUDA:0   3.758 ms   ││  min  -1.4349  max 1.8000                        │
│   ● mlp                    ││1660   20:52:20.535 RMSNorm       layers.29.poCUDA:0   0.043 ms   ││  spars ░░░░░░░░░░░░░░ 2.0%                       │
│ ▸ layers.3                 ││1659   20:52:20.535 SelfAttention layers.29.seCUDA:0   1.883 ms   │╰──────────────────────────────────────────────────╯
│   ● self_attn              ││1658   20:52:20.535 RMSNorm       layers.29.inCUDA:0   0.047 ms   │╭ 5. NUMERICAL ANOMALY LEDGER ─────────────────────╮
│   ● mlp                    ││1657   20:52:20.535 MLP           layers.28.mlCUDA:0   3.235 ms   ││20:52:20⚠ layers.15Exploding activations: |max| 12│
│ ▸ layers.4                 ││1656   20:52:20.535 RMSNorm       layers.28.poCUDA:0   0.051 ms   ││20:52:20⚠ layers.1.Exploding activations: |max| 12│
│   ● self_attn              ││1655   20:52:20.535 SelfAttention layers.28.seCUDA:0   2.029 ms   ││20:52:20⚠ layers.23Exploding activations: |max| 12│
│   ● mlp                    ││1654   20:52:20.535 RMSNorm       layers.28.inCUDA:0   0.049 ms   ││20:52:20.✖ layers.9.CUDA OOM fallback: MLP execute│
│ ▸ layers.5                 ││1653   20:52:20.535 MLP           layers.27.mlCUDA:0   3.548 ms   ││20:52:20.✖ layers.17CUDA OOM fallback: MLP execute│
│   ● self_attn              ││1652   20:52:20.535 RMSNorm       layers.27.poCUDA:0   0.051 ms   ││20:52:20.✖ layers.11CUDA OOM fallback: MLP execute│
│   ● mlp                    ││1651   20:52:20.535 SelfAttention layers.27.seCUDA:0   1.598 ms   ││20:52:20⚠ layers.14Exploding activations: |max| 12│
│ ▸ layers.6                 ││1650   20:52:20.535 RMSNorm       layers.27.inCUDA:0   0.045 ms   ││20:52:20✖ layers.14.CUDA OOM fallback: MLP execute│
│   ● self_attn              ││1649   20:52:20.535 MLP           layers.26.mlCUDA:0   3.374 ms   ││20:52:19.✖ layers.11CUDA OOM fallback: MLP execute│
│   ● mlp                    ││1648   20:52:20.535 RMSNorm       layers.26.poCUDA:0   0.056 ms   │╰──────────────────────────────────────────────────╯
│ ▸ layers.7                 ││1647   20:52:20.535 SelfAttention layers.26.seCUDA:0   1.716 ms   │╭ 6. PERFORMANCE ──────────────────────────────────╮
│   ● self_attn              ││1646   20:52:20.535 RMSNorm       layers.26.inCUDA:0   0.050 ms   ││CPU  ░░░░░░░░░░░░░░ 0%                            │
│   ● mlp                    ││1645   20:52:20.535 MLP           layers.25.mlCUDA:0   3.476 ms   ││RAM  ██████████████ 15.9/16.0 GB                  │
│ ▸ layers.8                 ││1644   20:52:20.535 RMSNorm       layers.25.poCUDA:0   0.052 ms   ││GPU  Apple GPU (counters unavailable)             │
│   ● self_attn              ││1643   20:52:20.535 SelfAttention layers.25.seCUDA:0   1.972 ms   │├──────────────────────────────────────────────────┤
│   ● mlp                    ││1642   20:52:20.535 RMSNorm       layers.25.inCUDA:0   0.057 ms   ││speed 8.58 tok/s  avg 1.17 ms  events 2125        │
│ ▸ layers.9                 ││1641   20:52:20.535 MLP           layers.24.mlCUDA:0   3.894 ms   │├──────────────────────────────────────────────────┤
│   ● self_attn              ││1640   20:52:20.535 RMSNorm       layers.24.poCUDA:0   0.056 ms   ││SLOWEST LAYERS                                    │
│   ● mlp                    ││1639   20:52:20.535 SelfAttention layers.24.seCUDA:0   1.862 ms   ││layers.27.mlp                    █████████ 3.949 m│
│ ▸ layers.10                ││1638   20:52:20.535 RMSNorm       layers.24.inCUDA:0   0.047 ms   ││layers.31.mlp                    █████████ 3.937 m│
│   ● self_attn              ││1637   20:52:20.535 MLP           layers.23.mlCUDA:0   3.348 ms   ││layers.30.mlp                    █████████ 3.824 m│
│   ● mlp                    ││1636   20:52:20.535 RMSNorm       layers.23.poCUDA:0   0.043 ms   ││layers.29.mlp                    █████████ 3.770 m│
╰────────────────────────────╯╰──────────────────────────────────────────────────────────────────╯╰──────────────────────────────────────────────────╯
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 [Tab] focus  [1-4] tabs  [j/k] nav  [p] sim  [q] quit
```

<details>
<summary>Attention tab (heatmap + token journey)</summary>

```
 LLM-TRACE  local transformer telemetry  mode: SIM [mock tracer]                                         [1]Dashboard [2]Attention [3]CallTree [4]Help
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
╭ 3. ATTENTION · layers.0.self_attn ◀ ───────────────────────────────────────────────────────╮╭ 2. TOKEN ACTIVATION JOURNEY ─────────────────────────╮
│head 0  hjkl pan · +/- contrast · </> head                                view [0-7] x [0-7]││token cycle #0 of 13  (j/k to step)                   │
├────────────┬───────────────────────────────────────────────────────────────────────────────┤├──────────────────────────────────────────────────────┤
│query \ key │I     want  it    to    be    keyboadrivenand                                  ││s0 → embed_tokens            μ 0.0100 max 1.800spars 2│
├────────────┼───────────────────────────────────────────────────────────────────────────────┤│s1 → layers.0.input_layernormμ 0.0000 max 1.200spars 0│
│I           │██                                                                             ││s2 → layers.0.self_attn      μ 0.0300 max 4.500spars 5│
│want        │▓▓    ██                                                                       ││s3 → layers.0.post_attention_μ 0.0000 max 1.200spars 0│
│it          │▒▒    ▓▓    ██                                                                 ││s4 → layers.0.mlp            μ 0.0500 max 5.500spars 5│
│to          │░░    ▒▒    ▓▓    ██                                                           ││s5 → layers.1.input_layernormμ 0.0000 max 1.200spars 0│
│be          │      ░░    ▒▒    ▓▓    ██                                                     ││s6 → layers.1.self_attn      μ 0.0300 max 4.550spars 5│
│keyboard    │            ░░    ▒▒    ▓▓    ██                                               ││s7 → layers.1.post_attention_μ 0.0000 max 1.200spars 0│
│driven      │                  ░░    ▒▒    ▓▓    ██                                         ││s8 → layers.1.mlp            μ 0.0500 max 5.560spars 5│
│and         │                        ░░    ▒▒    ▓▓    ██                                   ││s9 → layers.2.input_layernormμ 0.0000 max 1.200spars 0│
├────────────┴───────────────────────────────────────────────────────────────────────────────┤│s10→ layers.2.self_attn      μ 0.0300 max 4.600spars 5│
│avg entropy 1.446 bits · contrast 1.0x                                                      ││s11→ layers.2.post_attention_μ 0.0000 max 1.200spars 0│
│                                                                                            ││s12→ layers.2.mlp            μ 0.0500 max 5.620spars 4│
│                                                                                            ││s13→ layers.3.input_layernormμ 0.0000 max 1.200spars 0│
│                                                                                            ││s14→ layers.3.self_attn      μ 0.0300 max 4.650spars 5│
│                                                                                            ││s15→ layers.3.post_attention_μ 0.0000 max 1.200spars 0│
│                                                                                            ││s16→ layers.3.mlp            μ 0.0500 max 5.680spars 5│
│                                                                                            ││s17→ layers.4.input_layernormμ 0.0000 max 1.200spars 0│
│                                                                                            ││s18→ layers.4.self_attn      μ 0.0300 max 4.700spars 5│
│                                                                                            ││s19→ layers.4.post_attention_μ 0.0000 max 1.200spars 0│
│                                                                                            ││s20→ layers.4.mlp            μ 0.0500 max 5.740spars 4│
│                                                                                            ││s21→ layers.5.input_layernormμ 0.0000 max 1.200spars 0│
│                                                                                            ││s22→ layers.5.self_attn      μ 0.0300 max 4.750spars 5│
│                                                                                            ││s23→ layers.5.post_attention_μ 0.0000 max 1.200spars 0│
│                                                                                            ││s24→ layers.5.mlp            μ 0.0500 max 5.800spars 5│
│                                                                                            ││s25→ layers.6.input_layernormμ 0.0000 max 1.200spars 0│
│                                                                                            ││s26→ layers.6.self_attn      μ 0.0300 max 4.800spars 5│
│                                                                                            ││s27→ layers.6.post_attention_μ 0.0000 max 1.200spars 0│
│                                                                                            ││s28→ layers.6.mlp            μ 0.0500 max 5.860spars 5│
│                                                                                            ││s29→ layers.7.input_layernormμ 0.0000 max 1.200spars 0│
│                                                                                            ││s30→ layers.7.self_attn      μ 0.0300 max 4.850spars 5│
│                                                                                            ││s31→ layers.7.post_attention_μ 0.0000 max 1.200spars 0│
│                                                                                            ││s32→ layers.7.mlp            μ 0.0500 max 5.920spars 5│
│                                                                                            ││s33→ layers.8.input_layernormμ 0.0000 max 1.200spars 0│
│                                                                                            ││s34→ layers.8.self_attn      μ 0.0300 max 4.900spars 5│
│                                                                                            ││s35→ layers.8.post_attention_μ 0.0000 max 1.200spars 0│
╰────────────────────────────────────────────────────────────────────────────────────────────╯╰──────────────────────────────────────────────────────╯
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 [Tab] focus  [1-4] tabs  [j/k] nav  [p] sim  [q] quit
```
</details>

<details>
<summary>Call Tree tab (flame profile)</summary>

```
 LLM-TRACE  local transformer telemetry  mode: SIM [mock tracer]                                         [1]Dashboard [2]Attention [3]CallTree [4]Help
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
╭ 1. CALL TREE / FLAME PROFILE ◀ ──────────────────────────────────────────────────────────────╮╭ 6. PERFORMANCE ────────────────────────────────────╮
│▼ Inference loop                                                ██████████ 152.70 ms (100.0%) ││CPU  ░░░░░░░░░░░░░░ 0%                              │
│  ● embed_tokens                                                           0.20 ms (0.1%)     ││RAM  ██████████████ 15.9/16.0 GB                    │
│  ▼ transformer blocks                                          █████████  151.54 ms (99.2%)  ││GPU  Apple GPU (counters unavailable)               │
│    ▶ layer 0                                                              3.44 ms (2.3%)     │├────────────────────────────────────────────────────┤
│    ▶ layer 1                                                              3.61 ms (2.4%)     ││speed 8.63 tok/s  avg 1.17 ms  events 2127          │
│    ▶ layer 2                                                              3.69 ms (2.4%)     │├────────────────────────────────────────────────────┤
│    ▶ layer 3                                                              3.68 ms (2.4%)     ││SLOWEST LAYERS                                      │
│    ▶ layer 4                                                              3.77 ms (2.5%)     ││layers.31.mlp                     ██████████ 4.054 m│
│    ▶ layer 5                                                              3.90 ms (2.6%)     ││layers.29.mlp                     ██████████ 4.053 m│
│    ▶ layer 6                                                              3.87 ms (2.5%)     ││layers.30.mlp                     ██████████ 3.920 m│
│    ▶ layer 7                                                              4.14 ms (2.7%)     ││layers.27.mlp                     ██████████ 3.919 m│
│    ▶ layer 8                                                              3.91 ms (2.6%)     ││layers.28.mlp                     █████████░ 3.761 m│
│    ▶ layer 9                                                              4.29 ms (2.8%)     ││                                                    │
│    ▶ layer 10                                                             4.24 ms (2.8%)     ││                                                    │
│    ▶ layer 11                                                             4.35 ms (2.8%)     ││                                                    │
│    ▶ layer 12                                                             4.49 ms (2.9%)     ││                                                    │
│    ▶ layer 13                                                             4.53 ms (3.0%)     ││                                                    │
│    ▶ layer 14                                                             4.55 ms (3.0%)     ││                                                    │
│    ▶ layer 15                                                             4.56 ms (3.0%)     ││                                                    │
│    ▶ layer 16                                                             4.77 ms (3.1%)     ││                                                    │
│    ▶ layer 17                                                             4.92 ms (3.2%)     ││                                                    │
│    ▶ layer 18                                                             4.94 ms (3.2%)     ││                                                    │
│    ▶ layer 19                                                             4.93 ms (3.2%)     ││                                                    │
│    ▶ layer 20                                                             5.17 ms (3.4%)     ││                                                    │
│    ▶ layer 21                                                             5.13 ms (3.4%)     ││                                                    │
│    ▶ layer 22                                                             5.24 ms (3.4%)     ││                                                    │
│    ▶ layer 23                                                             5.33 ms (3.5%)     ││                                                    │
│    ▶ layer 24                                                             5.42 ms (3.5%)     ││                                                    │
│    ▶ layer 25                                                             5.47 ms (3.6%)     ││                                                    │
│    ▶ layer 26                                                             5.58 ms (3.7%)     ││                                                    │
│    ▶ layer 27                                                             5.82 ms (3.8%)     ││                                                    │
│    ▶ layer 28                                                             5.65 ms (3.7%)     ││                                                    │
│    ▶ layer 29                                                             6.06 ms (4.0%)     ││                                                    │
│    ▶ layer 30                                                             5.97 ms (3.9%)     ││                                                    │
│    ▶ layer 31                                                             6.13 ms (4.0%)     ││                                                    │
│  ● output_norm                                                            0.06 ms (0.0%)     ││                                                    │
│  ● lm_head                                                                0.90 ms (0.6%)     ││                                                    │
│                                                                                              ││                                                    │
╰──────────────────────────────────────────────────────────────────────────────────────────────╯╰────────────────────────────────────────────────────╯
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
 [Tab] focus  [1-4] tabs  [j/k] nav  [p] sim  [q] quit
```
</details>

## How it fits together

```
                 TCP :5005 (line delimited JSON)
  model + hooks ─────────────►┐
  (examples/hook.py)          │        ┌──► RingBuffer (fixed size, flat memory)
                              ▼        │
  MockTracer ──────────► EventBus ─────┼──► TelemetryAggregator (latency, tok/s)
  (synthetic)                          │
                                       ├──► AnomalyDetector (NaN/Inf, exploding,
                                       │                     dead layer, sparsity)
                                       └──► Dashboard (FTXUI) ──► your terminal
                              DeviceMonitor (CPU/RAM) ─────────────┘
```

The pieces:

* **EventBus** is a small thread safe pub/sub hub for telemetry events.
* **TelemetryServer** listens on a TCP port and decodes the JSON protocol.
* **RingBuffer** keeps a fixed amount of history so memory stays flat no matter
  how long you run. Attention matrices are large, so only the latest one per
  layer is kept rather than every frame.
* **TelemetryAggregator** tracks rolling per layer latency and tokens per second.
* **AnomalyDetector** flags numerical trouble and CPU or OOM fallbacks.
* **Dashboard** and **Panels** are the TUI itself.

The wire format lives in [docs/protocol.md](docs/protocol.md).

## Building

You need CMake 3.16 or newer and a C++17 compiler. FTXUI and nlohmann/json are
fetched automatically during configure.

```sh
cmake -B build
cmake --build build -j
```

## Running

Try the UI straight away with the built in simulator. No Python needed.

```sh
./build/llm-trace --sim
```

Or listen for a real model and stream telemetry into it from another shell.

```sh
./build/llm-trace                                  # terminal 1
python examples/hook.py --model gpt2 --steps 20    # terminal 2
```

Other flags: `--headless` prints rolling stats instead of drawing the UI, which
is handy for piping or CI. `--snapshot` renders one populated frame and exits.
`--port N` changes the TCP port. `--help` lists everything.

### Real models

```sh
pip install torch transformers
python examples/hook.py --model gpt2 --steps 30
```

`hook.py` registers forward hooks on the embedding, attention, MLP and norm
modules. It computes the activation stats inside the model process, so the raw
tensors never leave that process, and sends a compact summary to llm-trace.

## Keys

| key             | action                                         |
|-----------------|------------------------------------------------|
| `1` to `4`      | switch tab (Dashboard, Attention, Call Tree, Help) |
| `Tab` / `S-Tab` | cycle focus between panels                      |
| `j` / `k`       | move down / up in lists                         |
| `Enter` / `Space` | select a layer, or expand a flame node        |
| `h j k l`       | pan the attention matrix                        |
| `+` / `-`       | attention heatmap contrast                      |
| `<` / `>`       | previous / next attention head                  |
| `[` / `]`       | shrink / grow the attention viewport            |
| `p`             | pause or resume the simulator                   |
| `q` / `Esc`     | quit                                            |

## What each tab shows

1. **Dashboard** has the model tree, the live packet stream, a metrics inspector
   for the selected layer, the anomaly ledger, and a panel with CPU and RAM plus
   the slowest layers so you can see which block dominates compute.
2. **Attention** has a heatmap you can pan and zoom, with per head Shannon
   entropy, next to a token by token view of the activations.
3. **Call Tree** is a flame style breakdown of where inference time goes.
4. **Help** is the key reference.
