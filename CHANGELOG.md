# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
