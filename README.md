# llama-autotune

A cross-platform benchmarking and optimization tool for `llama.cpp` that automatically discovers the fastest stable configuration for any combination of:

- CPU-only, NVIDIA (CUDA), AMD (ROCm/Vulkan), Intel, and Apple Silicon systems
- Single-GPU and multi-GPU setups
- Dense and Mixture-of-Experts (MoE) models
- Any GGUF quantization

```bash
llama-autotune doctor                 # check this PC and the installed llama.cpp
llama-autotune inspect model.gguf     # what the model needs and what fits right now
llama-autotune launch model.gguf      # start llama-server with settings that fit
```

## Features

- **Memory-safe by design** — plans against the RAM that is *free right now*, not total RAM, and keeps a safety margin so Windows stays responsive. llama.cpp's own `--fit` compares with total RAM; with default settings an 8 GB test PC fell to 6 MiB free
- **Exact memory math from the model file** — weights, KV cache (grouped-query, sliding-window, shared and hybrid/recurrent layers) and the hidden private copy made by weight repacking, which llama.cpp's own estimate leaves out on CPU (up to 3.2 GB for an 8B model). Matches llama.cpp's numbers on every tested model
- **RAM watchdog** — every benchmark is stopped before the PC starts swapping, and it tells whether the model or another program used the memory
- **Works with any llama.cpp build** — reads each build's `--help` and never sends a flag it does not accept; older spellings (`--mlock`, `--no-mmap`, `-mmp`) are used automatically on older builds
- **Honest model inspection** — the quantization comes from the actual tensors, because file metadata and filenames can both be wrong
- **Safe llama-server defaults** — explicit context size, one chat slot, no 8 GiB prompt cache on 8 GB PCs, no vision projector unless asked
- **Hardware detection without admin rights** — Win32 APIs on Windows (free RAM, commit, power, performance vs efficiency cores); llama.cpp's own device list for GPUs and VRAM
- **Auto-Scaling Search** — a tiny speed probe first, then benchmark size and trial count scale to the hardware
- **3-Stage Optimization** — heuristic baseline → local grid search → Bayesian (Optuna) tuning, only over settings llama-bench can measure
- **SQLite history and launch profiles** — results keyed by machine, model fingerprint and llama.cpp build

## Installation

### Prerequisites

- Python 3.12+
- [llama.cpp](https://github.com/ggml-org/llama.cpp) built binaries (`llama-bench.exe`, `llama-server.exe`)
- `uv` (recommended) or `pip`

### Option 1: Install from source (recommended)

```bash
git clone https://github.com/Najafu/llama-autotune.git
cd llama-autotune
uv sync
```

### Option 2: Install directly from GitHub

```bash
pip install git+https://github.com/Najafu/llama-autotune.git
```

### Setting up llama.cpp

The tool needs the `llama-bench` and `llama-server` binaries (`llama-bench.exe` / `llama-server.exe` on Windows). It looks for them in this order:

1. The directory set in the `LLAMA_CPP_DIR` environment variable
2. Directories next to the installed package
3. Your system `PATH`

```bash
# Linux / macOS
export LLAMA_CPP_DIR=/path/to/llama.cpp/build/bin

# Windows (PowerShell)
$env:LLAMA_CPP_DIR = "C:\path\to\llamacpp"
```

### Verify installation

```bash
llama-autotune --version
llama-autotune inspect path/to/model.gguf
```

## Usage

### doctor

Check this PC and the installed llama.cpp in one step. Shows CPU cores (performance and efficiency), the instruction set llama.cpp actually uses, free RAM, power source, memory bandwidth, the llama.cpp build and its devices, and warns about anything that would distort results, such as `LLAMA_ARG_*` environment variables or a stray llama.cpp `config.ini`. Paste its output into bug reports.

```bash
llama-autotune doctor
```

### inspect

Show hardware, model metadata and a memory plan: for each context size, how much RAM the model needs and whether it fits in the memory free right now, or which savings make it fit.

```bash
llama-autotune inspect model.gguf
```

Real output for Llama 3.2 1B on an 8 GB PC with a browser open:

```text
       Memory plan (free now 2,134 MiB, keeping 810 MiB spare)
│        Context │ Needs (defaults) │ Verdict                        │
│          2,048 │        1,585 MiB │ Won't fit (short by 37 MiB)    │
│          4,096 │        1,649 MiB │ Won't fit (short by 71 MiB)    │
│ 131,072 (full) │        5,617 MiB │ Won't fit (short by 2,179 MiB) │
```

Closing a browser tab or two would make the first rows fit. The last row is what llama-server's defaults would have reserved.

If the file's metadata disagrees with its tensors, both are shown, for example `Q6_K (file metadata says Q4_K_S; tensors say Q6_K)`. MoE models display both total and active parameter counts.

### benchmark

Run a benchmark with optional custom parameters.

```bash
llama-autotune benchmark model.gguf -r 3 -t 8 -b 2048
```

### search

Run the full 3-stage optimizer to find the best configuration. The optimizer
automatically detects hardware speed with a minimal probe (`-p 64 -n 32 -r 1`)
and scales all benchmarks accordingly:

| Speed tier | Gen TPS | Trial budget | Benchmark size |
|---|---|---|---|
| `very_slow` | < 1 or timeout | Heuristic only | — |
| `slow` | 1 – 4 | 10 | 64 prompt, 32 gen, 1 rep |
| `medium` | 4 – 15 | 25 | 256 prompt, 64 gen, 2 reps |
| `fast` | > 15 | 55 | 512 prompt, 128 gen, 3 reps |

No configuration needed — works on anything from a Raspberry Pi to a Threadripper.

```bash
llama-autotune search model.gguf --objective balanced
llama-autotune search model.gguf --objective max_generation_tps --profile best.json
```

| Objective | Optimizes for |
|---|---|
| `max_generation_tps` | Fastest token generation |
| `max_prompt_tps` | Fastest prompt processing |
| `min_latency` | Shortest startup time |
| `max_context` | Largest context size that fits in memory (generation speed as tiebreaker) |
| `max_efficiency` | Generation speed per MB of memory |
| `balanced` | Generation speed (default) |

MoE models (e.g. OLMoE, Qwen-MoE, DeepSeek, Mixtral) are fully supported — the inspector automatically detects `expert_count` and `expert_used_count` from GGUF headers and reports active vs total parameters.

### launch

Start `llama-server` with settings that fit in the memory free right now, then open the printed address in your browser to chat.

```bash
llama-autotune launch model.gguf                  # 4,096-token context, reduced if needed
llama-autotune launch model.gguf -c 8192 --images # longer context, keep the vision projector
llama-autotune launch model.gguf --dry-run        # print the command only
llama-autotune launch model.gguf --profile best.json
```

When the model does not fit, `launch` refuses with the exact shortfall instead of letting Windows swap. To make it fit, it first shrinks the batch (no speed cost), then turns weight repacking off, then compresses the KV cache, and only then shortens the context. Profiles remember the llama.cpp build they were tuned on and warn after an upgrade.

### export / import

Save and load launch profiles.

```bash
llama-autotune export profile.json --model model.gguf --hardware "RTX 4090" --score 102
llama-autotune import profile.json
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CPP_DIR` | parent of tool directory | Path to the `llama-bench` / `llama-server` binaries |

## Data Storage

Benchmark history and launch profiles are stored in a SQLite database at:

```
~/.llama-autotune/benchmarks.db        # Linux / macOS
C:\Users\<you>\.llama-autotune\benchmarks.db   # Windows
```

Delete this file at any time to reset the benchmark history — it is recreated automatically.

## Dependencies

| Package | Purpose |
|---------|---------|
| typer | CLI framework |
| pydantic | Data validation |
| rich | Terminal output |
| optuna | Bayesian optimization |
| psutil | Cross-platform system probes |

GGUF files are parsed with a built-in reader, the database uses Python's `sqlite3`, and Windows probes use the Win32 API through `ctypes`, so no other libraries are needed.

## Project Structure

```
src/llama_autotune/
├── cli.py              # Typer CLI (doctor, inspect, benchmark, search, launch, export, import)
├── compat.py           # llama.cpp compatibility layer: build probing, flag registry, rendering
├── model_inspector.py  # GGUF reader: metadata, tensor table, KV geometry, real quantization
├── memory.py           # Memory model: weights, KV cache, repack copy, compute buffers
├── constraints.py      # Memory gate (free RAM, safety margin, savings) + OOM detection
├── resources.py        # Free RAM, commit, power, CPU topology, bandwidth (Win32 + psutil)
├── hardware.py         # Hardware summary; llama.cpp devices are the authority
├── benchmark.py        # llama-bench runner with RAM watchdog
├── heuristics.py       # Baseline config + safe llama-server defaults
├── search_space.py     # Tunable parameters (only what llama-bench measures is searched)
├── optimizer.py        # 3-stage search engine
├── multi_gpu.py        # Multi-GPU tensor-split candidates
├── profiles.py         # JSON profile export/import
└── database.py         # SQLite storage (stdlib sqlite3, versioned schema)
```

## Architecture

```
llama.cpp build probe ┐
Hardware + free RAM  ─┼→ Memory gate → Speed probe → Benchmarks (watchdog) → Optimizer → History → Launcher
GGUF inspector       ┘
```

## Troubleshooting

### "llama-bench not found"

The most common issue. Set `LLAMA_CPP_DIR` to the directory containing the llama.cpp binaries and verify:

```bash
# Windows (PowerShell)
$env:LLAMA_CPP_DIR = "C:\path\to\llamacpp"

# Linux / macOS
export LLAMA_CPP_DIR=/path/to/llama.cpp/build/bin

llama-autotune benchmark model.gguf -r 1
```

On Linux/macOS make sure the binaries are executable (`chmod +x llama-bench`).

### "This model does not fit in the memory free right now"

The tool keeps 10% of your RAM (at least 512 MiB) spare so Windows stays responsive, and plans against what is free *now*. Close programs you don't need (browsers use the most), or pick a smaller model or quantization. `llama-autotune inspect model.gguf` shows exactly how much each context size needs.

### Benchmark stopped with `memory_guard`

The watchdog stopped llama-bench before your PC started swapping. `memory_guard` means the model needed more RAM than was free; `memory_guard_external` means another program took the memory during the run, and the search retries once automatically.

### Search finishes instantly with "using heuristic"

Your machine was classified as `very_slow` (under 1 token/sec on the probe). The tool returns the safe heuristic config instead of spending hours benchmarking. Try a smaller model or a lower quantization (e.g. Q4 instead of Q8).

### Benchmark fails or times out

- Check the model runs at all: `llama-bench -m model.gguf -p 16 -n 8 -r 1`
- Very large models on low-RAM machines can hit the OOM detector — the constraint engine rejects configs it estimates won't fit, but estimates can be off for unusual architectures.
- Run with `-v` (`llama-autotune -v search ...`) to see per-trial logs.

### Where did my results go?

Every benchmark is saved to the SQLite database (see [Data Storage](#data-storage)). The best config is also printed at the end of `search`, and `--profile best.json` writes a reusable profile file.

## FAQ

**Does it work without a GPU?**
Yes — CPU-only systems are fully supported. The search space automatically adjusts (thread count instead of GPU layers).

**How long does a search take?**
The tool probes your hardware speed first and scales the workload: a few minutes on fast machines, and it degrades gracefully to heuristics-only on very slow ones (target: under 10–20 minutes on any machine).

**Are MoE models supported?**
Yes. `expert_count` / `expert_used_count` are read from GGUF headers, and both total and active parameters are reported and used in heuristics.

**Can I share profiles between machines?**
Yes — `export` writes a JSON profile that `import` reads on another machine. Profiles are hardware-specific, so treat an imported profile as a starting point, not a guarantee.

## Testing

```bash
uv run pytest -v
```

The suite (about 150 tests) covers flag parsing and rendering against recorded llama.cpp b11347 output and a synthetic older build, the GGUF reader on synthetic files, the memory model and gate, the RAM watchdog, system probes, the optimizer on a simulated machine, the CLI, and the database including upgrades of v0.3.0 databases. Memory figures are also checked against real models when they are present. Tests that need a real GGUF model or the llama.cpp binaries skip automatically, so the suite runs anywhere. CI runs on Linux, Windows, and macOS.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

PRs welcome. Please run tests (`uv run pytest`) before submitting and match the existing code style (Google-style docstrings, full type annotations).

## Project Status

Current version: **0.4.0**, phase 1 (Foundation) of the next-generation roadmap: correct and memory-safe on current llama.cpp.

| Phase | Status |
|---|---|
| 1. Foundation: compatibility layer, memory model and gate, watchdog, Windows probes | ✓ (v0.4.0) |
| 2. Measured CPU tuning: bandwidth-calibrated prediction, one-process sweeps, plain-language verdicts | next |
| 3. One-click Windows release: guided setup, automatic llama.cpp download, desktop shortcut | planned |
| 4. Model recommendations, GPU and hybrid tuning | planned |
| 5. Community benchmark sharing | later |

> Note: developed and battle-tested on Windows; Linux and macOS support is implemented and covered by CI, but has seen less real-world use. Bug reports welcome.
