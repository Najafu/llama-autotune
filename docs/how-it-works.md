# How it works

What llama-autotune does behind the commands, and why. For the exact measuring procedure of `tune`, see the [benchmark protocol](benchmark-protocol.md).

## Overview

```text
llama.cpp build check ─┐
PC and free RAM        ├─> Memory plan ─> Prediction ─> Measurements ─> Verdict ─> Saved settings ─> launch
Model file (GGUF)      ┘                (memory speed)  (llama-bench, RAM watchdog)
```

1. **Read the parts:** the installed llama.cpp build, the PC (processor, free RAM, memory speed) and the model file are read. Nothing is assumed.
2. **Memory plan:** decides what fits in the RAM that is free now.
3. **Prediction:** estimates the speed from the memory speed.
4. **Measurements:** llama-bench runs under a memory watchdog and decide the settings.
5. **Save and use:** the result is saved for this PC and model, together with the llama.cpp build it was measured on, and `launch` uses it.

## Memory planning

Most crashes and frozen PCs with local models come from running out of memory. llama.cpp's own memory fitting compares with the PC's *total* RAM; with default settings, an 8 GB test PC fell to 6 MiB free. llama-autotune plans against the RAM that is **free right now**, and keeps 10% of total RAM (at least 512 MiB) spare.

The memory a model needs is computed from its file, in parts:

| Part | What it is | How it is known |
|---|---|---|
| Weights | The model itself | The model's tensor table |
| Repacked copy | A private, faster-to-read copy of some weights that llama.cpp makes on the processor | Which tensor types the processor's instruction set repacks; up to 3.2 GB for an 8B model, and llama.cpp's own estimate leaves it out |
| Context memory (KV cache) | What the model keeps of the conversation | The model's attention layout: grouped-query, sliding-window, shared and recurrent layers; matched llama.cpp exactly on every tested model |
| Recurrent state | Fixed-size state of hybrid models (Qwen3.5, LFM) | The model's metadata |
| Compute buffer | Working memory while reading and writing | Measured once with llama.cpp's `llama-fit-params`, then modelled (below) |
| Runtime | Tokenizer and similar, not listed by llama.cpp | Vocabulary size x 384 bytes, at least 32 MiB |
| Vision file | Only with images | File size plus half again (at least 256 MiB more) |

**Compute buffer:** this is modelled from three parts.
- **Output scores:** vocabulary x batch x 4 bytes in llama-bench. The llama-server chat needs them for one token only, since llama.cpp b12228.
- **A fixed working area:** 60 to 85 MiB on the tested models.
- **Attention scores, with flash attention off:** context x batch x heads x 4 bytes. These grow with the context: 1 GiB for Llama 3.2 1B at 16,384 tokens.

Checked against llama.cpp on five models, both flash-attention settings, at 4K and 16K contexts, the model was never below llama.cpp's own figure. Real llama-server memory on the test PC came out 40 to 70 MiB under the estimate.

**When a model does not fit,** savings are applied in order of least harm:
1. smaller batch
2. weight repacking off
3. flash attention on (if it was off)
4. compressed context memory (q8_0)
5. shorter context

If nothing works, the command refuses with the exact amount missing.

**Memory watchdog:** every benchmark runs under it. It checks free RAM four times a second and stops the test before the PC starts swapping. It also tells whether the model or another program took the memory.

**Model files are checked while they are read.** A file that ends too early, or whose header claims impossible sizes, is refused in plain words before anything is planned or run.

## Tuning

`tune` measures instead of guessing, within a time limit:

1. **Predict.** Writing one token reads the model's active weights once, so the speed is roughly *efficiency x memory speed / bytes read per token*. The memory speed is the best ever measured on the PC, because other programs can only make it look slower. The efficiency is calibrated per model from the first measurement (66 to 98% on the test PC).
2. **Measure the defaults** with llama.cpp's own settings: the bar to beat.
3. **Screen.** Thread counts, flash attention and weight repacking, all in one llama-bench process (the model loads once per repacking setting).
4. **Confirm.** The best third and anything close to the leader run again with more repetitions, plus untested thread counts next to the best.
5. **Duel.** The winner and the defaults take turns, three or more times, and the median of the paired results decides. When time is too short for that, the defaults are measured once more and averaged with their first run. On a busy PC, measuring one after the other gave anywhere from -3% to +22% for the same real gain; taking turns gave +8% to +21%.
6. **Prompt settings.** Threads and batch size for reading prompts are chosen by prompt speed. A different thread count for prompts is confirmed with three repetitions when there is time.
7. **Long chat.** The speed after 2,048 tokens of conversation, with flash attention both off and on. On the test PC flash attention tied in an empty chat but was 37% slower later on, so both are scored together.

All measurements run at above-normal priority, as `launch` does. That gave 13% more speed and much steadier numbers while other programs used half the processor.

A setting replaces llama.cpp's default only when it wins beyond the measured noise, which is at least 4%.

## Compatibility with llama.cpp

llama.cpp changes its options often. llama-autotune reads each installed build's `--help` and `--version`, and keeps a list of what that build accepts. It never sends an option the build does not understand, and uses older spellings on older builds (for example `--mlock` and `--no-mmap` before the `--load-mode` option existed). The list is cached per build, so an update is noticed automatically.

**Model types:** before running a model, llama-autotune checks that the installed llama.cpp knows its architecture. It reads the table of architecture names inside llama.cpp's library, so a too-old llama.cpp gets a plain "update llama.cpp" message instead of a failed load.

**Environment variables:** `LLAMA_ARG_*` variables silently change llama.cpp's settings, so they are removed before llama.cpp runs, and `doctor` warns about them.

## What is run, and what is refused

llama-autotune starts programs and passes options on, so it is careful about where both come from:
- **llama.cpp** is taken from the places listed in [Getting started](getting-started.md#how-llamacpp-is-found). The folder you are standing in is not one of them.
- **Profile files** can come from anyone. `launch --profile` only runs the options llama-autotune itself writes, and on a processor it checks them against the RAM free now.
- **Extra options** for `launch` cannot load a different model, and passwords among them are neither saved nor shown.

## Hardware detection

No administrator rights are needed. On Windows the Win32 API gives:
- free RAM and commit headroom
- battery and battery saver
- performance and efficiency cores
- processor groups (for PCs with more than 64 threads)
- NUMA nodes

On Linux and macOS, `psutil` and system files give the same. Graphics devices and the processor's instruction set come from llama.cpp itself, because that is what it will use.

## Where data is kept

Everything is in one folder: `~/.llama-autotune` (on Windows `C:\Users\<you>\.llama-autotune`).

| File | Contents |
|---|---|
| `benchmarks.db` | Tuned settings, benchmark history, and options you gave `launch` per model |
| `config.json` | The llama.cpp folder saved by `setup` |
| `cache\` | What each llama.cpp build accepts, and the compute buffer llama.cpp reported for each model |

Results are keyed by a fingerprint of the PC and a fingerprint of the model file, and the llama.cpp build is stored with them. A moved or renamed model file therefore still gets its tuned settings in `launch` (`models` shows the path it was tuned at). The PC's fingerprint is its processor, its amount of RAM, its graphics devices and the kind of operating system; it does not change with system updates, but it does when you add RAM or a graphics card. Delete the folder at any time to start fresh; it is recreated automatically. Nothing is sent anywhere.

## Environment variables

| Variable | Meaning |
|---|---|
| `LLAMA_CPP_DIR` | Folder with llama.cpp's programs; overrides `setup` |
