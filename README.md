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

The suite (90+ tests) covers the CLI commands, benchmark output parsing, MoE detection, objective scoring, constraint logic, hardware detection, search space, profiles, and database operations. Tests that need a real GGUF model or the llama.cpp binaries skip automatically when those are not present, so the suite runs anywhere. CI runs on Linux, Windows, and macOS.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

PRs welcome. Please run tests (`uv run pytest`) before submitting and match the existing code style (Google-style docstrings, full type annotations).

## Project Status

Current version: **0.3.0**. All milestones 1–4 are complete. MoE detection verified with real models (OLMoE-1B-7B). The tool is ready for daily use.

| Milestone | Status |
|---|---|
| 1. Hardware Detection + GGUF Inspection + Benchmark | ✓ |
| 2. Config Generator + SQLite Storage + Result Parsing | ✓ |
| 3. Optimization Engine + OOM Detection + Auto-Scaling | ✓ |
| 4. Launch Profiles + Multi-GPU + CLI Commands | ✓ |

**Future (no timeline):** Community benchmark sharing, cloud database, web dashboard.

> Note: developed and battle-tested on Windows; Linux and macOS support is implemented and covered by CI, but has seen less real-world use. Bug reports welcome.
