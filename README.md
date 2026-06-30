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
