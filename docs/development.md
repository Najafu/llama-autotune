# Development

## Setting up

```bash
git clone https://github.com/Najafu/llama-autotune.git
cd llama-autotune
uv sync
uv run llama-autotune --help
```

Python 3.12 or newer. The dependencies are typer (command line), pydantic (data models), rich (terminal output), optuna (`search`) and psutil (system probes). GGUF files are read with a built-in reader, the database uses Python's `sqlite3`, and Windows probes use the Win32 API through `ctypes`.

## Tests

```bash
uv run pytest            # about 260 tests, under 30 seconds
uv run pytest -v         # with test names
```

The suite needs neither llama.cpp nor a model:
- **The tuner** on simulated machines: 2-core to 64-core, hybrid and multi-socket, busy and quiet.
- **llama.cpp compatibility:** option parsing and rendering against recorded `--help` output.
- **The GGUF reader** on synthetic files.
- **Memory:** the memory model and plan, checked against figures measured with llama.cpp b12228.
- **Safety and probes:** the memory watchdog and the system probes.
- **Commands and database:** the commands, and the database including upgrades of old ones.

Tests that use real models or llama.cpp skip automatically when those are missing. CI runs the suite on Windows, Linux and macOS with Python 3.12, 3.13 and 3.14. Tests never touch your own `~/.llama-autotune` folder: the database and the cache are redirected to a temporary folder.

Please run the tests before a pull request, and keep the style: Google-style docstrings and full type annotations.

## Project structure

```text
src/llama_autotune/
├── cli.py              Commands: setup, doctor, inspect, tune, models, launch, benchmark, search, export, import
├── tuner.py            tune: one method per stage (prediction, screening, confirmation, duel, long-chat check), verdicts
├── benchmark.py        Running llama-bench: one-process sweeps, JSONL results, memory watchdog
├── memory.py           Memory model: weights, repacked copy, KV cache, compute buffer, runtime
├── constraints.py      Memory plan: free RAM, safety margin, savings in order of least harm
├── compat.py           llama.cpp compatibility: build probing, option list, rendering, architecture check, profile option check
├── model_inspector.py  GGUF reader: metadata, tensor table, KV layout, real quantization, MoE
├── vision.py           Vision files (mmproj): matching by width, memory
├── extra_args.py       Options passed to launch: names, checks, remembering
├── resources.py        Free RAM, commit, power, CPU topology, memory speed (Win32 and psutil)
├── hardware.py         Hardware summary; llama.cpp's own device list for graphics cards
├── heuristics.py       Starting configuration and safe llama-server defaults
├── database.py         SQLite storage with a versioned schema
├── optimizer.py        search: three-stage search engine
├── search_space.py     search: settings to explore
└── profiles.py         Reading and writing launch profile files
```

## How the decisions were made

The numbers in these docs come from experiments on the test PC, a 2-core Intel i3-6100 with 8 GB of DDR3 and no graphics card, with llama.cpp b11347 and b12228. The [benchmark protocol](benchmark-protocol.md) explains each measuring rule and the experiment behind it. The [changelog](../CHANGELOG.md) lists what changed in each version and why.

## Status

Version 0.5.0 has measured tuning on processors, which was the goal.

**Not tested on real hardware:**
- graphics cards
- PCs with several graphics cards
- processors with many cores or several sockets (covered by simulated tests only)

**Real-world use:** Linux and macOS pass CI but have seen little real use.

**Next:** tuning for graphics cards may come in a later version. Nothing else is planned.
