# Troubleshooting and FAQ

## Problems

### "llama.cpp was not found"

llama-autotune needs llama.cpp's programs. Download and unzip llama.cpp (see [Getting started](getting-started.md#3-get-llamacpp)), then tell llama-autotune where it is:

```bash
llama-autotune setup C:\llama
llama-autotune doctor
```

If you set the `LLAMA_CPP_DIR` environment variable at some point, it wins over `setup`. Remove it or point it at the right folder. On Linux and macOS, make sure the programs are executable (`chmod +x llama-server llama-bench`).

### "Cannot read this model"

The file starts like a model but ends too early or is damaged. Almost always the download did not finish. Compare the file's size with the size on the download page, and download it again.

### "This model does not fit in the memory free right now"

llama-autotune plans against the RAM that is free *now* and keeps 10% of your total RAM (at least 512 MiB) spare, so Windows stays responsive. The message says how much is missing. You can:
- **Free memory:** close programs you don't need. Web browsers use the most.
- **Ask for less:** a shorter context, for example `-c 2048`.
- **Choose a smaller model or quantization:** 4B instead of 8B, `Q4_K_M` instead of `Q8_0`.

`llama-autotune inspect model.gguf` shows what each context size needs. If even the smallest size is short by more than a few hundred MiB, the model is too large for the PC.

### "cannot load '...' models: this model type is newer than your llama.cpp"

The model uses an architecture your llama.cpp does not know yet; for example, LFM3 needs llama.cpp b11960 or newer. Download the newest llama.cpp release, unzip it over the old one, and run the command again. llama-autotune checks this before doing anything, so no time is wasted.

### tune says "Speed changed by ...% during the test"

The speed of llama.cpp's defaults changed a lot between the start and the end, because another program started or the PC got hot. The result is still usable: when there is time, the final comparison lets the winner and the defaults take turns, so both meet the same disturbance. It is less precise, though. Close other programs, plug in a laptop, and run `tune` again for cleaner numbers.

### tune says "Other programs are using ...% of the CPU"

Other programs were busy while measuring. llama-autotune runs at above-normal priority to limit the effect, but a quiet PC gives the most reliable result.

### tune says "A test was stopped because free memory ran too low"

The memory watchdog stopped a test before the PC started swapping. Results finished before the stop are kept; if the stop came before any setting could be compared, llama.cpp's defaults are kept. If it happens often, close other programs before tuning.

### tune keeps llama.cpp's defaults

That is a valid result. No tested setting was faster beyond the measurement noise, so nothing changes for nothing. On many PCs llama.cpp's defaults are already close to the best.

### Images are "left out"

The vision file did not fit in the free RAM next to the model at the context you asked for. The text chat still works. Free some memory, use a shorter context (`-c`), or check what images need with `inspect`. See [Images](chat-and-apps.md#images).

### "No vision file for this model"

`--images` needs the model's vision projector (`mmproj-...gguf`) in the same folder as the model. Download the one made for your model, or name it with `--mmproj`.

### An option I gave launch is refused

`launch` checks every extra option against your llama.cpp. The message suggests the correct spelling, for example `--topk` → `--top-k`. Options from newer llama.cpp releases are refused by older ones; update llama.cpp if you need them. Options that would load a different model, such as `-m` or `--hf-repo`, are always refused: name the model you want as the first argument instead.

### "Context must be a number of tokens (256 or more) or 'max'"

`-c 0` and other very small values are refused. llama.cpp reads 0 as "the model's full context", which is the size most likely to fill the memory. Give a number of 256 or more, or `-c max` for the longest context that fits now.

### "This profile contains options llama-autotune never writes"

`launch --profile` only runs the options llama-autotune itself puts into a profile. The file was edited by hand or made by something else. Remove the options named in the message from the file, or, if you trust them, give them yourself after the model name. `llama-autotune import profile.json` shows what a profile contains.

### models says "file missing"

The model is no longer at the path it was tuned at. `launch` still finds its tuned settings, because they belong to the file, not the path. Run `tune` on the model again to update the path, or ignore the line.

### My tuned results are gone

Results belong to the PC as it was when they were measured: its processor, its amount of RAM and its graphics devices. After adding RAM or a graphics card, tune again. Results made by version 0.5.0 or earlier are carried over automatically.

### A coding agent is very slow at first

Coding agents send thousands of tokens of instructions with the first request, and a processor reads them at tens of tokens per second. Later requests are quick. See [What to expect](chat-and-apps.md#what-to-expect-on-a-pc-without-a-graphics-card).

### search finishes at once with "using heuristic"

Either the first quick test found the PC too slow (under 1 token per second) for this model, or llama-bench could not run it at all; the message says which. `search` then returns a safe starting configuration instead of spending hours. Try a smaller model or quantization, or see "A benchmark fails or times out" below.

### search says max_context cannot be searched

llama-bench does not measure a context size, so there is nothing to search. Use `llama-autotune launch model.gguf -c max` for the longest context that fits now.

### A benchmark fails or times out

- Check that the model runs at all: `llama-bench -m model.gguf -p 16 -n 8 -r 1` in the llama.cpp folder.
- Run with `-v` (for example `llama-autotune -v tune model.gguf`) to see detailed logs.
- Include the output of `llama-autotune doctor` when you report the problem.

### Where are my results?

In `~/.llama-autotune/benchmarks.db` (on Windows `C:\Users\<you>\.llama-autotune\benchmarks.db`). `llama-autotune models` lists them. See [Where data is kept](how-it-works.md#where-data-is-kept).

## FAQ

**Does it work without a graphics card?**
Yes. That is what it was made and tested for.

**Does it tune graphics cards?**
Not yet. On a PC with a graphics card, `launch` lets llama.cpp place the model on it, and `search` can explore settings, but neither has been verified on real graphics cards.

**How long does tune take?**
At most 10 minutes by default (`--minutes` changes it), usually 3 to 6 on small models. The size of each test follows the measured speed, and work is trimmed when time runs short. On a PC so slow that even one test would take too long, `tune` gives a predicted verdict without measuring.

**Does it use all my processor cores?**
It tests different thread counts. Writing text is limited by memory speed, so the fastest setting often uses fewer threads than the processor has; when two settings are within noise, the one with fewer threads wins, leaving the PC more responsive. On Intel hybrid processors, efficiency cores are not used.

**Are MoE models supported?**
Yes. Memory counts every expert, because all of them stay in RAM, while the speed prediction counts only the experts used per token. Tested with Granite 3.1 1B-A400M.

**Do I need to tune again after updating llama.cpp?**
It is a good idea: new llama.cpp versions change speeds and defaults. `launch` and `models` tell you when results come from an older build.

**Can I share settings between PCs?**
Yes: `export settings.json --model model.gguf` writes a model's tuned settings to a file, and `launch model.gguf --profile settings.json` on the other PC uses them. But settings are specific to a PC, so run `tune` on each PC for the best result.

**Is anything sent over the internet?**
No. llama-autotune runs everything locally. Only the installation (`uv tool install`) and your own downloads of llama.cpp and models use the internet.
