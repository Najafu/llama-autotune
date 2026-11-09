# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.5.0] - 2026-11-09

Phase 2 (CPU tuning) of the next-generation design: measured tuning that
finishes in bounded time, never recommends settings slower than llama.cpp's
defaults, and answers in plain words.

### Added

- `tune` command: measures the fastest settings for a model on this PC in
  at most 10 minutes (`--minutes`). It checks free memory, predicts the speed
  from memory bandwidth, runs llama.cpp's defaults as the bar to beat, tests
  threads, flash attention and weight repacking in one llama-bench process,
  and re-tests the best third with three repetitions (successive halving).
  It then picks prompt threads and ubatch, measures the speed after 2,048
  tokens of chat, and lets the winner and the defaults take turns in a final
  duel, so a program starting halfway cannot decide the result. The
  protocol is documented in `docs/benchmark-protocol.md`.
- Plain verdicts: Fast (10 tokens/s or more), Usable (4 to 10), Slow, or
  Won't fit, each with one reason.
- `launch` applies the settings saved by `tune` for that model and machine,
  and warns when llama.cpp was upgraded since.
- `launch --editing` turns on n-gram speculative decoding
  (`--spec-type ngram-mod`): 1.7 to 2.6 times faster when rewriting pasted
  text, and about 7% slower on new text, so it is off by default.
- Speed prediction: efficiency x bandwidth / bytes read per token, including
  the KV cache at a given depth. The efficiency is calibrated per model.
- One-process llama-bench sweeps (`run_sweep`): comma lists, JSONL
  records, and results finished before a watchdog stop are kept.
- NUMA node count in `doctor`. On multi-socket PCs, `tune` also tests
  `--numa distribute`.
- `launch` passes your own llama-server options through, such as sampling
  settings: `--temp 0.1 --top-k 50 --repeat-penalty 1.1`, or the names used on
  model cards (`temperature=0.1 top_k=50 repetition_penalty=1.1`). They are
  checked against the installed build (a typo gets a suggestion), replace the
  matching generated option, and are remembered per model until
  `--forget-settings`. The built-in web chat sends no sampling values of its
  own, so these apply there.
- Images with vision models: `launch --images` finds the vision file
  (`mmproj-*.gguf`) that belongs to the model by its metadata (projection width
  equals the model's embedding width), passes it with `--mmproj`, and counts it
  in the memory plan (file size plus half again; 912 MiB measured for a 637 MiB
  projector). Before, `--images` never told llama-server where the file was, so
  images were rejected. Images are optional: when they do not fit at the wanted
  context they are left out with a message and the text chat starts. `--mmproj`
  names a file explicitly; `inspect` shows the vision file and its cost.
- `launch` names the model with `--alias` (the file name without `.gguf`) and
  prints the base URL, model name and context an app such as a coding agent
  needs. The README shows an opencode configuration.
- `launch -c max`: the largest context that fits in the RAM free right now
  (binary search in 1,024-token steps).
- `setup <folder>` saves where llama.cpp is, so no environment variable is
  needed. llama.cpp is also found in common unzip places (current folder,
  `C:\llama`, the newest download), and every command explains how to get
  it when it is missing.
- MoE speed prediction: only the experts used per token are counted as read
  (Granite 3.1 1B-A400M: 0.27 of 0.77 GB; predicted 53.9, measured 47.8
  tokens/s). Before, every expert was counted, so MoE predictions were several
  times too low.
- Above-normal priority for `tune` measurements and for `launch`
  (`--prio 1 --prio-batch 1`): +13% and much steadier speed while other
  programs keep half the CPU busy. High priority gave nothing more.

### Fixed

Found by experiments on the test PC (details in `docs/benchmark-protocol.md`):

- The memory check could underestimate long contexts with flash attention off
  by about 800 MiB (Llama 3.2 1B at 16K: llama.cpp needs 1,082 MiB of
  compute buffer, the estimate said 290). The compute buffer is now modelled
  from logits, a core area and attention scores that grow with the context,
  and never fell below llama.cpp's figure in 28 checks. When such a context
  does not fit, turning flash attention back on is tried before cutting the
  context.
- `launch` and `inspect` overestimated llama-server's memory by 190 to 450 MiB
  on b12228, which reserves output logits for one token only. On the 8 GB test
  PC this decided whether a model could start (Llama 3.2 1B was refused, short
  by 51 MiB). Estimates now land 40 to 70 MiB above the measured working set.
- A small runtime allowance (tokenizer and the like, 11 to 85 MiB measured) is
  now part of the estimate.
- The memory-speed test ran while other programs competed for memory, so a
  busy PC measured 12.8 instead of 18 GB/s and reported an impossible 121%
  efficiency. `tune` now uses the best bandwidth ever measured on this PC.
- The flash-attention choice compared speeds from two different measurements
  and could keep it on (7.0 instead of 10.3 tokens/s after 2,048 tokens of
  chat). Each setting now scores its empty-chat and long-chat speed together,
  from like-for-like measurements.
- On a busy PC the final check against the defaults could reach the wrong
  verdict (the same +8-21% gain measured as -3% when a program started
  halfway). Winner and defaults now take turns, judged by the median of paired
  ratios.

Found by testing on llama.cpp b12228 (0.9.0-dev):

- Flash attention was chosen from an empty context only. On b12228 it ties
  there on the test PC but is 37% slower after 2,048 tokens of chat, and
  llama.cpp's "auto" turns it on, so the tuned Llama 3.2 1B wrote only
  6.4 tokens/s in a longer chat. `tune` now also compares it after a long
  chat (11.0 tokens/s), and sets that time aside before the ubatch test.
- `launch` ignored a setting that `tune` had kept at llama.cpp's default and
  used its own guess instead (flash attention off instead of auto). It now
  uses exactly what was measured.
- When the defaults won for writing, `tune` also discarded a faster prompt
  batch size it had just measured (+29% on prompts), while still reporting it.
- LFM2/LFM3 short-convolution state was missing from the memory estimate
  (0.61 MiB for LFM3 1.7B; KV cache and repacking already matched llama.cpp).

Found earlier:

- A model whose architecture is newer than the installed llama.cpp (for
  example LFM3 on b11347) made `tune` fail with "Tuning failed (failed)"
  after measuring memory speed, and `inspect` said it fits. Every command now
  checks the model type against the architecture table inside the installed
  llama.cpp library before doing anything, and says plainly to update
  llama.cpp and where to get it.
- Windows PCs with more than 64 logical processors: threads outside
  processor group 0 were not counted.
- Thread search on many-core CPUs started at the core count minus one, so it
  never tried the lower counts where bandwidth-bound generation often peaks.
  It now spreads from a quarter of the performance cores and refines around
  the best.
- Memory bandwidth was measured with one thread, which cannot saturate
  workstation and server memory. It now also measures with one thread per
  performance core and reports the higher figure.

### Changed

- `search` uses the same memory model as `tune` and `launch`, so its
  `max_context` objective no longer picks contexts that only fit on paper
  (flash attention off at long contexts was underestimated).
- When flash attention off wins in a long chat but loses in an empty one, the
  reported speed is that of the saved settings, not of the defaults.
- `search` is now described as the advanced option for GPUs and experiments;
  `tune` is the default for CPUs.
- Tests no longer touch the user's real database.

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
