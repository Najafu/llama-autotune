# llama-autotune

A cross-platform benchmarking and optimization tool for `llama.cpp` that automatically discovers the fastest stable configuration for any combination of:

- CPU-only, NVIDIA (CUDA), AMD (ROCm/Vulkan), Intel, and Apple Silicon systems
- Single-GPU and multi-GPU setups
- Dense and Mixture-of-Experts (MoE) models
- Any GGUF quantization

```bash
llama-autotune search model.gguf --objective max_generation_tps
```

## Features

- **Hardware Detection** — CPU cores, RAM, GPU vendor/model/VRAM across Windows, Linux, macOS
- **GGUF Inspection** — reads model architecture, parameter count, quantization, layer count, context length, MoE status (total vs active parameters, expert count) directly from GGUF headers
- **Auto-Scaling Search** — runs a tiny speed probe first, then scales benchmark size and trial count to match the hardware (works on Raspberry Pi and Threadripper alike)
- **3-Stage Optimization** — heuristic baseline → local grid search → Bayesian (Optuna) tuning
- **Constraint Engine** — VRAM estimation, OOM detection, plausibility checks (rejects bad configs before benchmarking)
- **Multi-GPU Support** — tensor-split ratio search for 2+ GPU systems
- **SQLite Database** — persistent benchmark history and launch profiles
- **Launch Profiles** — export/import reusable JSON configurations
- **Zero Third-Party GPU Libs** — hardware detection uses stdlib + WMI/subprocess

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

The tool needs `llama-bench.exe` and `llama-server.exe` in your PATH, or set the `LLAMA_CPP_DIR` environment variable:

```bash
# Linux / macOS
export LLAMA_CPP_DIR=/path/to/llama.cpp/build

# Windows (PowerShell)
$env:LLAMA_CPP_DIR = "C:\path\to\llamacpp"
```

### Verify installation

```bash
llama-autotune --help
llama-autotune inspect path/to/model.gguf
```

## Usage

### inspect

Show hardware and model metadata. MoE models display both total and active parameter counts.

```bash
llama-autotune inspect model.gguf

# MoE models show Active Params:
# Parameters     7,000,000,000
# MoE            Yes
# Active Params  1,000,000,000
```

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

Objectives: `max_generation_tps`, `max_prompt_tps`, `min_latency`, `max_context`, `max_efficiency`, `balanced`

MoE models (e.g. OLMoE, Qwen-MoE, DeepSeek, Mixtral) are fully supported — the inspector automatically detects `expert_count` and `expert_used_count` from GGUF headers and reports active vs total parameters.

### launch

Start `llama-server` with the optimal profile.

```bash
llama-autotune launch model.gguf
llama-autotune launch model.gguf --profile best.json
```

### export / import

Save and load launch profiles.

```bash
llama-autotune export profile.json --model model.gguf --hardware "RTX 4090" --score 102
llama-autotune import profile.json
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CPP_DIR` | parent of tool directory | Path to `llama-bench.exe` / `llama-server.exe` |

## Dependencies

| Package | Purpose |
|---------|---------|
| typer | CLI framework |
| pydantic | Data validation |
| rich | Terminal output |
| sqlalchemy | Database ORM |
| optuna | Bayesian optimization |
| psutil | Hardware detection |
| structlog | Logging |

> **Note:** GGUF files are parsed with a fast custom header reader (no external library needed).

## Project Structure

```
src/llama_autotune/
├── cli.py              # Typer CLI (6 commands)
├── models.py           # Pydantic data classes
├── hardware.py         # CPU/RAM/GPU detection
├── model_inspector.py  # GGUF header parser
├── benchmark.py        # llama-bench subprocess wrapper
├── heuristics.py       # Baseline config generator
├── search_space.py     # Tunable parameter definitions
├── optimizer.py        # 3-stage search engine
├── constraints.py      # VRAM estimator + OOM detection
├── multi_gpu.py        # Multi-GPU tensor-split tuning
├── profiles.py         # JSON profile export/import
└── database.py         # SQLite storage
```

## Architecture

```
Hardware Probe → GGUF Inspector → Config Generator → Speed Probe → Benchmark Engine → Optimizer → Profile Database → Launcher
```

## Testing

```bash
uv run pytest -v
```

74 tests cover parser variants, MoE detection, constraint logic, hardware detection, search space, and database operations.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

PRs welcome. Please run tests (`uv run pytest`) before submitting and match the existing code style (Google-style docstrings, full type annotations).

## Project Status

All milestones 1–4 are complete. MoE detection verified with real models (OLMoE-1B-7B). The tool is ready for daily use.

| Milestone | Status |
|---|---|
| 1. Hardware Detection + GGUF Inspection + Benchmark | ✓ |
| 2. Config Generator + SQLite Storage + Result Parsing | ✓ |
| 3. Optimization Engine + OOM Detection + Auto-Scaling | ✓ |
| 4. Launch Profiles + Multi-GPU + CLI Commands | ✓ |

**Future (no timeline):** Community benchmark sharing, cloud database, web dashboard.
