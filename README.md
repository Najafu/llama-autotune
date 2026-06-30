# llama-autotune

A cross-platform benchmarking and optimization tool for `llama.cpp` that automatically discovers the fastest stable configuration for any model + hardware combination.

```bash
llama-autotune search model.gguf --objective max_generation_tps
```

## Features

- **Hardware Detection** — CPU cores, RAM, GPU vendor/model/VRAM across Windows, Linux, macOS
- **GGUF Inspection** — reads model architecture, parameter count, quantization, layer count, context length, MoE info directly from GGUF headers
- **3-Stage Optimization** — rule-based baseline → local grid search → Bayesian (Optuna) tuning
- **Constraint Engine** — VRAM estimation, OOM detection, plausibility checks
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

Show hardware and model metadata.

```bash
llama-autotune inspect model.gguf
```

### benchmark

Run a benchmark with optional custom parameters.

```bash
llama-autotune benchmark model.gguf -r 3 -t 8 -b 2048
```

### search

Run the full 3-stage optimizer to find the best configuration.

```bash
llama-autotune search model.gguf --objective balanced
llama-autotune search model.gguf --objective max_generation_tps --profile best.json
```

Objectives: `max_generation_tps`, `max_prompt_tps`, `min_latency`, `max_context`, `max_efficiency`, `balanced`

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
| gguf | GGUF model parser |
| optuna | Bayesian optimization |
| psutil | Hardware detection |
| structlog | Logging |
| pyyaml | Config files |

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
Hardware Probe → GGUF Inspector → Config Generator → Benchmark Engine → Optimizer → Profile Database → Launcher
```

## Testing

```bash
uv run pytest -v
```

## License

MIT. See [LICENSE](LICENSE).

## Contributing

PRs welcome. Please run tests (`uv run pytest`) before submitting and match the existing code style (Google-style docstrings, full type annotations).

## Roadmap

- Milestone 1: Hardware Detection + GGUF Inspection + Benchmark ✓
- Milestone 2: Config Generator + SQLite Storage + Result Parsing ✓
- Milestone 3: Optimization Engine + OOM Detection + Retry Logic ✓
- Milestone 4: Launch Profiles + Multi-GPU + CLI Commands ✓
- Milestone 5: Community Benchmark Sharing + Cloud Database + Web Dashboard
