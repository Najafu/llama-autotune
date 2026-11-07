# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-11-07

Phase 1 (Foundation) of the next-generation design: correct and memory-safe
on current llama.cpp. Every memory figure below was checked against
llama.cpp b11347 on real models.

### Fixed

- Flags rejected by current llama.cpp are no longer generated: `--mlock`,
  `--no-mmap` and `-mmp` (replaced by `--load-mode` in b10875) and
  `-nkvo 0/1` (now a bare switch in llama-server). `-fa` now uses `on/off`.
- KV-cache estimate was about 16x too low (it ignored grouped-query
  attention and head size). It now matches llama.cpp exactly, including
  sliding-window, shared-cache and hybrid recurrent models.
- Memory checks used 80% of *total* RAM; they now use the RAM available
  right now minus a safety margin.
- Quantization is read from the actual tensors. File metadata and filenames
  can both be wrong (two local "GPTQ-Q6_K" files are Q6_K but declare
  Q4_K_S). The `general.file_type` table was also wrong (15 is Q4_K_M).
- The search could only try as many threads as physical cores, so it could
  never find that 3 threads beat 2 on a 2-core/4-thread CPU (+22% measured).
- Context size is no longer searched by Optuna: llama-bench does not use it,
  so those trials repeated identical benchmarks.
- OOM detection no longer matches words such as "room".
- llama-bench output is collected through files, so a long log can no
  longer stall a benchmark.

### Added

- `compat` module: probes the installed llama.cpp (`--version`, `--help`,
  `--list-devices`), caches a flag registry per build, renders settings in
  the spelling each build accepts, and skips settings a build cannot express.
- Memory model and gate: weights, KV cache, recurrent state, compute buffers
  (from `llama-fit-params` when installed) and the private copy made by
  weight repacking, which llama.cpp's own estimate omits on CPU (up to
  3.2 GB for an 8B model). When a setup does not fit, savings are applied in
  order of least harm: smaller ubatch, repacking off, q8_0 KV cache, then a
  shorter context.
- RAM watchdog on every benchmark; it reports whether the model or another
  program used the memory, and the search retries once in the second case.
- `doctor` command: hardware, instruction-set tier actually used by
  llama.cpp, free RAM, power, bandwidth, llama.cpp build and devices, and
  warnings for `LLAMA_ARG_*` variables or a stray llama.cpp `config.ini`.
- `inspect` shows a memory plan per context size against free RAM.
- `launch` plans memory, refuses with the exact shortfall when a model does
  not fit, and writes safe llama-server defaults: explicit context, one
  slot, prompt cache off on 8 GB PCs, few context checkpoints for
  sliding-window and hybrid models, no vision projector unless `--images`.
  New `--dry-run`, `-c` and `--images` options.
- Windows probes through the Win32 API (no administrator rights): available
  RAM, commit headroom, battery and battery saver, and performance vs
  efficiency cores, used to keep threads off efficiency cores.
- GPUs, VRAM and backend come from llama.cpp's own `--list-devices`.
- Profiles and benchmark history record the llama.cpp build; `launch` warns
  when a profile was tuned on a different build. History is keyed by
  machine fingerprint and model fingerprint instead of the architecture name.

### Changed

- Database moved from SQLAlchemy to the standard-library `sqlite3`, with a
  versioned schema that upgrades v0.3.0 databases in place.
- Removed dependencies: `sqlalchemy`, `structlog`.
- `LLAMA_ARG_*` environment variables are removed before running llama.cpp.

## [0.3.0] - 2026-07-03

### Fixed

- The profile import command is now exposed as `llama-autotune import` (it was
  accidentally registered as `import-cmd`, contradicting the documentation).
- Binary resolution is now truly cross-platform: `llama-bench` / `llama-server`
  are resolved without the `.exe` suffix on Linux and macOS, and the system
  `PATH` is searched as a final fallback.
- `min_latency` optimization no longer lets failed configurations outrank
  working ones. Failed and implausible trials are pruned from the Bayesian
  stage instead of receiving a `-1.0` sentinel score that beat legitimate
  negative latency scores.
- `launch` on Windows now runs `llama-server` as a child process instead of
  `os.execvp`, which does not quote arguments correctly on Windows (paths
  containing spaces would break).

### Added

- Real memory measurement: benchmark runs now record the peak RAM of the
  llama-bench process (via psutil), since llama-bench does not report memory
  in its JSON output. This makes the `max_efficiency` objective meaningful —
  previously memory was always 0 and the objective degraded to generation
  speed.
- The `max_context` objective is now implemented: the search maximizes context
  size (validated by the constraint engine and benchmark success), with
  generation throughput as a tiebreaker. Previously it was a placeholder that
  optimized generation speed.
- CLI test suite (`tests/test_cli.py`) and objective scoring test suite
  (`tests/test_optimizer_scoring.py`).
- Continuous integration on Linux, Windows, and macOS via GitHub Actions.
- This changelog.

### Changed

- `--help` output is cleaner: usage examples moved to a single epilog instead
  of being appended to every command description.
- Tests that inspect real GGUF models now skip gracefully when the model file
  is not present on the machine.

## [0.2.0] - 2026-06

### Added

- Auto-scaling speed probe: benchmark size and trial budget scale to the
  measured hardware speed tier (`very_slow` / `slow` / `medium` / `fast`).
- MoE detection from GGUF headers (`expert_count`, `expert_used_count`),
  reporting active vs total parameters. Verified with OLMoE-1B-7B.
- Multi-GPU tensor-split search.
- SQLite persistence for benchmarks and launch profiles.

## [0.1.0] - 2026-06

### Added

- Initial release: hardware detection, GGUF inspection, llama-bench
  integration, heuristic config generation, 3-stage optimizer
  (baseline → grid search → Optuna), constraint engine with VRAM
  estimation and OOM detection, launch profiles, Typer CLI.
