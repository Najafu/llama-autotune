# llama-autotune

A cross-platform benchmarking and optimization tool for `llama.cpp` that automatically discovers the fastest stable configuration for any combination of:

- CPU-only, NVIDIA (CUDA), AMD (ROCm/Vulkan), Intel, and Apple Silicon systems
- Single-GPU and multi-GPU setups
- Dense and Mixture-of-Experts (MoE) models
- Any GGUF quantization

```bash
llama-autotune setup C:\llama         # once: where you unzipped llama.cpp
llama-autotune doctor                 # check this PC and the installed llama.cpp
llama-autotune inspect model.gguf     # what the model needs and what fits right now
llama-autotune tune model.gguf        # measure the fastest settings (up to 10 minutes, once)
llama-autotune launch model.gguf      # start llama-server with the tuned settings
```

## Features

- **Memory-safe by design** — plans against the RAM that is *free right now*, not total RAM, and keeps a safety margin so Windows stays responsive. llama.cpp's own `--fit` compares with total RAM; with default settings an 8 GB test PC fell to 6 MiB free
- **Exact memory math from the model file** — weights, KV cache (grouped-query, sliding-window, shared and hybrid/recurrent layers) and the hidden private copy made by weight repacking, which llama.cpp's own estimate leaves out on CPU (up to 3.2 GB for an 8B model). Matches llama.cpp's numbers on every tested model
- **RAM watchdog** — every benchmark is stopped before the PC starts swapping, and it tells whether the model or another program used the memory
- **Works with any llama.cpp build** — reads each build's `--help` and never sends a flag it does not accept; older spellings (`--mlock`, `--no-mmap`, `-mmp`) are used automatically on older builds
- **Honest model inspection** — the quantization comes from the actual tensors, because file metadata and filenames can both be wrong
- **Safe llama-server defaults** — explicit context size, one chat slot, no 8 GiB prompt cache on 8 GB PCs, no vision projector unless asked
- **Hardware detection without admin rights** — Win32 APIs on Windows (free RAM, commit, power, performance vs efficiency cores); llama.cpp's own device list for GPUs and VRAM
- **Measured CPU tuning in bounded time** — `tune` predicts speed from memory bandwidth, calibrates on llama.cpp's defaults, then tests threads, flash attention and weight repacking in one llama-bench process and confirms the best third with more repetitions. A setting only replaces the default when it wins beyond the measured noise; see the [benchmark protocol](docs/benchmark-protocol.md)
- **Plain verdicts** — Fast, Usable, Slow or Won't fit, with one reason, instead of tokens/s jargon
- **Small and big CPUs** — every thread count on small CPUs (3 threads beat the default 2 by 22% on a 2-core i3), a coarse-then-fine search on many-core CPUs, efficiency cores left out, processor groups above 64 threads on Windows, and NUMA placement on multi-socket machines
- **Advanced search** — heuristic baseline → local grid search → Bayesian (Optuna) tuning, for GPUs and experiments
- **SQLite history and launch profiles** — results keyed by machine, model fingerprint and llama.cpp build

## Getting started

Everything runs as commands in a terminal or PowerShell. There is no installer.

**One-time setup (about 10 minutes):**

1. Install [uv](https://docs.astral.sh/uv/), which also takes care of Python:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Install llama-autotune:
   ```bash
   uv tool install git+https://github.com/Najafu/llama-autotune
   ```
3. Download llama.cpp from its [releases page](https://github.com/ggml-org/llama.cpp/releases). For a PC without a GPU, take the file ending in `bin-win-cpu-x64.zip`, and unzip it, for example to `C:\llama`.
4. Tell llama-autotune where it is (once):
   ```bash
   llama-autotune setup C:\llama
   ```
5. Check everything:
   ```bash
   llama-autotune doctor
   ```

**For each model** (a `.gguf` file, for example from Hugging Face):

```bash
llama-autotune inspect model.gguf   # optional: does it fit, and at which context sizes
llama-autotune tune model.gguf      # once: measures the fastest settings, up to 10 minutes
llama-autotune launch model.gguf    # every time: starts the chat at http://127.0.0.1:8080
```

Press Ctrl+C in the terminal to stop the chat. After a llama.cpp update, `launch` reminds you to run `tune` again.

### Other ways to install

```bash
git clone https://github.com/Najafu/llama-autotune.git   # from source, for development
cd llama-autotune
uv sync                                                  # then: uv run llama-autotune ...

pip install git+https://github.com/Najafu/llama-autotune.git   # with an existing Python 3.12+
```

### Where llama-autotune looks for llama.cpp

In this order: the `LLAMA_CPP_DIR` environment variable, the folder saved by `llama-autotune setup`, folders next to the installed package, the system `PATH`, and finally common places where people unzip it (the current folder, `C:\llama`, `C:\llama.cpp`, and the newest `llama-b…-bin-…` folder in Downloads). When it is not found, every command says how to get it.

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

### tune

Measure the fastest settings for a model on this PC, once. It takes up to 10 minutes (`--minutes` changes the limit). The result is saved, and `launch` uses it from then on.

```bash
llama-autotune tune model.gguf
llama-autotune tune model.gguf -c 8192 --minutes 5 --profile chat.json
```

The output starts with a verdict, then shows llama.cpp's default next to the tuned value for each setting, and the speed of both:

```text
Verdict: Fast: about 13.0 tokens per second; text appears faster than you can read it.
                          Settings
| Setting                      | llama.cpp default | Tuned |
| Threads (writing)            | 2                 | 4     |
| Threads (reading the prompt) | 2                 | 4     |
| Flash attention              | auto              | off   |
| Weight repacking             | on                | on    |
                  Speed (tokens per second)
|                                    | Default |       Tuned |
| Writing (generation)               |     9.6 | 13.0 (+35%) |
| Reading the prompt                 |    60.8 |        86.0 |
| Writing after 2,048 tokens of chat |         |        11.8 |
Memory speed 15.1 GB/s; predicted 9.1 tokens/s before measuring, this model
reaches 78% of the bandwidth limit. 21 tests in 3.0 min.
```

That is Llama 3.2 1B on a 2-core/4-thread i3-6100 with 8 GB RAM. On the same PC, Qwen3.5 2B gained 10% (3 threads, flash attention on, repacking off to save memory) and Gemma 4 E2B gained 11%.

What it does, in order (details in [docs/benchmark-protocol.md](docs/benchmark-protocol.md)):

1. Checks that the model fits in the memory free right now, and never tests settings that would not fit.
2. Measures memory speed and predicts the generation speed.
3. Runs llama.cpp's defaults: the bar to beat.
4. Tests every combination of thread count, flash attention and weight repacking in one llama-bench process.
5. Re-tests the best third with three repetitions, plus untested thread counts next to the best.
6. Picks prompt threads and the prompt batch size, and on multi-socket PCs tests `--numa distribute`.
7. Measures the speed after a few thousand tokens of chat, and re-runs the defaults to detect disturbances.

Close other programs before tuning. A busy CPU or running on battery gives lower and noisier numbers, and the tool warns when it sees either.

### benchmark

Run a benchmark with optional custom parameters.

```bash
llama-autotune benchmark model.gguf -r 3 -t 8 -b 2048
```

### search

Advanced: run the 3-stage Optuna optimizer over many settings. On a CPU, `tune` is faster and more reliable; `search` is meant for GPUs and experiments. The optimizer
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

MoE models (e.g. Granite MoE, OLMoE, Qwen-MoE, DeepSeek, Mixtral) are supported: the inspector reads `expert_count` and `expert_used_count`, reports active vs total parameters, counts every expert for memory and only the experts used per token for the speed prediction. Tested with Granite 3.1 1B-A400M (32 experts, 8 used): 0.27 GB read per token of a 0.77 GB file, predicted 53.9 and measured 47.8 tokens/s, memory estimate 43 MiB above the real working set.

### launch

Start `llama-server` with settings that fit in the memory free right now, then open the printed address in your browser to chat.

```bash
llama-autotune launch model.gguf                  # 4,096-token context, reduced if needed
llama-autotune launch model.gguf -c 8192 --images # longer context, and read images if they fit
llama-autotune launch model.gguf --dry-run        # print the command only
llama-autotune launch model.gguf --editing        # faster when rewriting text you paste in
llama-autotune launch model.gguf --profile best.json
llama-autotune launch model.gguf -c max           # the largest context that fits in free RAM now
```

After `tune`, `launch` uses the measured settings for that model automatically. If llama.cpp has been upgraded since, it says so and suggests running `tune` again.

`--editing` turns on llama.cpp's n-gram speculative decoding (`--spec-type ngram-mod`). It speeds up answers that repeat text from the prompt, such as fixing or rewriting a pasted document: 1.7 to 2.6 times faster in testing. On brand-new text it is about 7% slower, so it is off by default.

**Your own llama-server options**, such as sampling settings, can follow the model. Both llama.cpp spellings and the names used on model cards work, and they are remembered for that model until you change or forget them:

```bash
llama-autotune launch model.gguf --temp 0.1 --top-k 50 --repeat-penalty 1.1
llama-autotune launch model.gguf temperature=0.1 top_k=50 repetition_penalty=1.1   # same thing
llama-autotune launch model.gguf                   # uses the remembered options again
llama-autotune launch model.gguf --forget-settings
```

Each option is checked against the installed llama.cpp first, so a typo gets a suggestion (`--topk` → "Did you mean --top-k?"). An option you give replaces the one llama-autotune would have set (for example `-t`). Options that change memory use, such as `-c` or `-ub`, are not covered by the memory plan, and `launch` says so. These options become the server's defaults: the built-in web chat sends no sampling values of its own (checked on b12228), so they apply there; another chat app may send its own values per message.

`-c max` searches, in steps of 1,024 tokens, for the largest context that fits in the RAM free right now, applying the memory savings below as needed. Expect it to be slower than the default.

**Images are optional.** A vision model comes with a second file, the vision projector (`mmproj-*.gguf`). `launch --images` finds the one that belongs to the model in the same folder, by its metadata rather than its name: its output width must equal the model's embedding width, so the 2B, 4B and 9B projectors in one folder never get mixed up. `--mmproj file.gguf` names one explicitly. The vision file and the work of reading one image are part of the memory plan: about the file size plus half again; the Qwen3.5 2B projector (637 MiB) needed 912 MiB in testing. If the model fits at the wanted context only without images, `launch` leaves images out, says so, and starts the text chat. With `-c max` it reports the largest context with and without images. Reading an image is slow on a CPU: a 1280x960 screenshot took 94 seconds and 1,221 tokens of context on the test PC, so smaller images help (see the next section).

When the model does not fit, `launch` refuses with the exact shortfall instead of letting Windows swap. To make it fit, it first shrinks the batch (no speed cost), then turns weight repacking off, then compresses the KV cache, and only then shortens the context. Profiles remember the llama.cpp build they were tuned on and warn after an upgrade.

### Using it from a coding agent or another app

`launch` starts an OpenAI-compatible server and prints what an app needs: the base URL (`http://127.0.0.1:8080/v1`), the model name (the file name without `.gguf`, set with llama-server's `--alias`), and the context size. Tool calling works out of the box (llama-server uses the model's own chat template).

For opencode, add a provider to `opencode.json` (in the project folder, or `~/.config/opencode/opencode.json`). The field names below follow opencode's documentation for OpenAI-compatible servers at the time of writing; check its [providers page](https://opencode.ai/docs/providers/) if your version differs:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "model": "local/qwen",
  "providers": {
    "local": {
      "name": "llama-autotune",
      "package": "@opencode/ai/providers/openai-compatible",
      "settings": { "baseURL": "http://127.0.0.1:8080/v1" },
      "models": {
        "qwen": {
          "modelID": "Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": 20480, "output": 4096 }
        }
      }
    }
  }
}
```

- **`limit.context`:** set it to the context `launch` printed. Without it, opencode assumes 200,000 tokens and sends more than the server holds.
- **Images:** add `"image"` to `input` only when `launch` says images are on. opencode's `media.image` settings can shrink screenshots, which makes them much faster to read on a CPU.
- **Sampling settings** (temperature, top_p and so on): give them to `launch`, which makes them the server's defaults.

Expect the first request to be slow on a CPU. A coding agent sends thousands of tokens of instructions and tool descriptions first: Qwen3.5 4B on the test PC read prompts at about 16 tokens/s, so 10,000 tokens take about 10 minutes. Later requests are quick, because llama-server reuses what it has already read. A smaller model reads prompts several times faster; `tune` shows the prompt speed.

### export / import

Save and load launch profiles.

```bash
llama-autotune export profile.json --model model.gguf --hardware "RTX 4090" --score 102
llama-autotune import profile.json
```

## Environment

| Variable | Default | Description |
|----------|---------|-------------|
| `LLAMA_CPP_DIR` | not set | Path to the `llama-bench` / `llama-server` binaries; overrides `llama-autotune setup` |

## Data Storage

Benchmark history and launch profiles are stored in a SQLite database at:

```
~/.llama-autotune/benchmarks.db        # Linux / macOS
C:\Users\<you>\.llama-autotune\benchmarks.db   # Windows
```

Delete this file at any time to reset the benchmark history — it is recreated automatically. The same folder holds `config.json` (the llama.cpp folder saved by `setup`) and a cache of what each llama.cpp build supports.

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
├── cli.py              # Typer CLI (doctor, inspect, tune, launch, benchmark, search, export, import)
├── compat.py           # llama.cpp compatibility layer: build probing, flag registry, rendering
├── model_inspector.py  # GGUF reader: metadata, tensor table, KV geometry, real quantization
├── memory.py           # Memory model: weights, KV cache, repack copy, compute buffers
├── constraints.py      # Memory gate (free RAM, safety margin, savings) + OOM detection
├── extra_args.py       # Your llama-server options for launch: names, checks, remembering
├── vision.py           # Vision files (mmproj): matching by width, memory
├── resources.py        # Free RAM, commit, power, CPU topology, bandwidth (Win32 + psutil)
├── hardware.py         # Hardware summary; llama.cpp devices are the authority
├── tuner.py            # Measured CPU tuning: predict, calibrate, screen, confirm, verdict
├── benchmark.py        # llama-bench runner: one-process sweeps, JSONL records, RAM watchdog
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
Hardware + free RAM  ─┼→ Memory gate → Prediction → Baseline → Screen → Confirm → Verdict → History → Launcher
GGUF inspector       ┘          (bandwidth)   (calibrate)  (one llama-bench process each, RAM watchdog)
```

## Troubleshooting

### "llama.cpp was not found"

Download and unzip llama.cpp (see [Getting started](#getting-started)), then tell llama-autotune where it is:

```bash
llama-autotune setup C:\llama
llama-autotune doctor
```

`setup` also finds the programs in a subfolder of the folder you give. If `LLAMA_CPP_DIR` is set, it wins over `setup`.

On Linux/macOS make sure the binaries are executable (`chmod +x llama-bench`).

### "This model does not fit in the memory free right now"

The tool keeps 10% of your RAM (at least 512 MiB) spare so Windows stays responsive, and plans against what is free *now*. Close programs you don't need (browsers use the most), or pick a smaller model or quantization. `llama-autotune inspect model.gguf` shows exactly how much each context size needs.

### Benchmark stopped with `memory_guard`

The watchdog stopped llama-bench before your PC started swapping. `memory_guard` means the model needed more RAM than was free; `memory_guard_external` means another program took the memory during the run, and the search retries once automatically.

### tune says "Speed changed during the test"

llama.cpp's defaults are measured at the start and again at the end. A large difference means another program or heat slowed the PC down during the test. Close other programs, plug in a laptop, and run `tune` again.

### tune keeps llama.cpp's defaults

That is a valid result: no tested setting was faster beyond the measurement noise, so nothing is changed for nothing. On many PCs the defaults are already close to the best.

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

**How long does tuning take?**
At most 10 minutes by default (`--minutes` changes it). Test sizes scale with the measured speed, and work is cut when time runs short. On a PC so slow that even one test would take too long, `tune` gives a predicted verdict instead of measuring.

**Does it use all my CPU cores?**
It tests. Writing text is limited by memory speed, so the fastest setting often uses fewer threads than the CPU has, and the tool prefers fewer threads when the difference is within noise, leaving the PC responsive. Efficiency cores on Intel hybrid CPUs are never used.

**Are MoE models supported?**
Yes. `expert_count` / `expert_used_count` are read from GGUF headers, and both total and active parameters are reported and used in heuristics.

**Can I share profiles between machines?**
Yes — `export` writes a JSON profile that `import` reads on another machine. Profiles are hardware-specific, so treat an imported profile as a starting point, not a guarantee.

## Testing

```bash
uv run pytest -v
```

The suite (about 190 tests) covers the tuner on simulated machines (2-core to 64-core, hybrid and multi-socket), processor-group parsing, flag parsing and rendering against recorded llama.cpp b11347 output and a synthetic older build, the GGUF reader on synthetic files, the memory model and gate, the RAM watchdog, system probes, the optimizer on a simulated machine, the CLI, and the database including upgrades of v0.3.0 databases. Memory figures are also checked against real models when they are present. Tests that need a real GGUF model or the llama.cpp binaries skip automatically, so the suite runs anywhere. CI runs on Linux, Windows, and macOS.

## Changelog

See [CHANGELOG.md](CHANGELOG.md) for release history.

## License

MIT. See [LICENSE](LICENSE).

## Contributing

PRs welcome. Please run tests (`uv run pytest`) before submitting and match the existing code style (Google-style docstrings, full type annotations).

## Project Status

Current version: **0.5.0**, phase 2 (CPU tuning) of the next-generation roadmap: measured, bounded-time CPU tuning with plain verdicts, on top of phase 1's memory safety and llama.cpp compatibility. Next: phase 3, model recommendations and GPU tuning.

llama-autotune stays a command-line tool: install it with one command, then run `doctor`, `tune` and `launch` in a terminal or PowerShell. There is no installer or setup wizard.

| Phase | Status |
|---|---|
| 1. Foundation: compatibility layer, memory model and gate, watchdog, Windows probes | ✓ (v0.4.0) |
| 2. Measured CPU tuning: bandwidth-calibrated prediction, one-process sweeps, plain-language verdicts | ✓ (v0.5.0) |
| 3. Model recommendations, GPU, multi-GPU and hybrid tuning | next (v1.0) |
| 4. Community benchmark sharing | later |

> Note: developed and battle-tested on Windows; Linux and macOS support is implemented and covered by CI, but has seen less real-world use. Bug reports welcome.
