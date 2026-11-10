# Commands

Every command and option. Run any command with `--help` for the same information in the terminal.

| Command | What it does |
|---|---|
| [`setup`](#setup) | Remember where llama.cpp is |
| [`doctor`](#doctor) | Check this PC and the installed llama.cpp |
| [`inspect`](#inspect) | Show a model's details and whether it fits |
| [`tune`](#tune) | Measure the fastest settings for a model |
| [`models`](#models) | List every model tuned on this PC |
| [`launch`](#launch) | Start the chat server |
| [`benchmark`](#benchmark) | Run one benchmark with settings you choose |
| [`search`](#search) | Advanced: explore many settings |
| [`export` / `import`](#export-and-import) | Save and load launch profiles |

**Options for every command:** `-v` / `--verbose` shows detailed logs, and `--version` prints the version.

**Model paths:** commands take the path to a `.gguf` file. If the path contains spaces, put it in quotes: `"C:\My Models\model.gguf"`. A file that is damaged or was not downloaded completely is refused with "Cannot read this model".

---

## setup

Remember the folder where llama.cpp is, so no environment variable is needed.

```bash
llama-autotune setup C:\llama
```

It checks that the folder (or one of its subfolders) contains `llama-server` and `llama-bench`, saves the location, and prints the llama.cpp build. If the `LLAMA_CPP_DIR` environment variable is set to a different folder, that variable wins, and `setup` tells you.

## doctor

Check the PC and the installed llama.cpp in one step.

```bash
llama-autotune doctor
llama-autotune doctor --no-bandwidth    # skip the 1-second memory speed test
```

It shows:
- **Processor:** cores and threads (performance and efficiency cores on hybrid CPUs), the instruction set llama.cpp uses on it, and NUMA nodes on multi-socket machines.
- **Memory:** total RAM, free RAM, and commit headroom (what can still be reserved before the page file is full).
- **Power:** mains or battery, and whether battery saver is on.
- **Memory speed** in GB/s.
- **llama.cpp:** where it is, the build number, devices, and features (memory fitting, load modes, weight repacking, prompt cache, speculative decoding).
- **Warnings** for anything that distorts results: running on battery, `LLAMA_ARG_*` environment variables, or a llama.cpp `config.ini` left on the PC.

Include its output when you report a bug.

## inspect

Show a model's details and how much memory it needs.

```bash
llama-autotune inspect model.gguf
```

**Model details:** architecture, parameters (total and active for MoE models), quantization, layers, context length, file size, bytes read per generated token, and KV cache size per 1,024 tokens of context.

The quantization is read from the model's tensors, not its name or metadata, because both can be wrong. When they disagree, both are shown, for example `Q6_K (file metadata says Q4_K_S; tensors say Q6_K)`.

**Vision file:** if the model has one in the same folder, `inspect` names it and says how much extra memory images need.

**Memory plan:** for each context size, the RAM needed with llama.cpp's defaults and whether it fits in the RAM free right now. Real output for Llama 3.2 1B on an 8 GB PC with many programs open:

```text
       Memory plan (free now 1,463 MiB, keeping 810 MiB spare)
|        Context | Needs (defaults) | Verdict                        |
|          2,048 |        1,461 MiB | Won't fit (short by 712 MiB)   |
|          4,096 |        1,525 MiB | Won't fit (short by 746 MiB)   |
|         32,768 |        2,477 MiB | Won't fit (short by 1,236 MiB) |
| 131,072 (full) |        5,741 MiB | Won't fit (short by 2,916 MiB) |
```

When a size fits only with memory savings, the verdict lists them, for example "Fits with: ubatch 256, weight repacking off". The last row is what llama-server would reserve by default: the model's full context.

## tune

Measure the fastest settings for a model on this PC. Run it once per model, and again after updating llama.cpp.

```bash
llama-autotune tune model.gguf
llama-autotune tune model.gguf -c 8192            # plan memory for an 8,192-token chat
llama-autotune tune model.gguf --minutes 5        # shorter time limit
llama-autotune tune model.gguf --profile chat.json   # also write a launch profile file
```

| Option | Default | Meaning |
|---|---|---|
| `-c`, `--ctx-size` | 4096 | Context size you want to chat with (256 or more); used for the memory check |
| `--minutes` | 10 | Time limit for the measurements (0.5 or more) |
| `-p`, `--profile` | | Also save a profile file for `launch --profile` |

Close other programs first. The tool warns when the PC runs on battery or other programs keep the CPU busy, because both lower and disturb the numbers. Ctrl+C stops the measurement; nothing is saved then.

`tune` measures processor settings only. On a PC where llama.cpp lists a graphics card it stops and says so; `launch` still works there.

Real output for Llama 3.2 1B on a 2-core Intel i3-6100 with 8 GB of RAM (llama.cpp b12228):

```text
Verdict: Fast: about 12.9 tokens per second; text appears faster than you can read it.
                          Settings
| Setting                      | llama.cpp default | Tuned        |
| Threads (writing)            | 2                 | 4            |
| Threads (reading the prompt) | 2                 | 4            |
| Flash attention              | auto              | off          |
| Weight repacking             | on                | on           |
| Prompt batch (ubatch)        | 512               | 512          |
| Priority                     | normal            | above normal |
                  Speed (tokens per second)
|                                    | Default |      Tuned |
| Writing (generation)               |    12.3 | 12.9 (+5%) |
| Reading the prompt                 |    59.2 |       95.4 |
| Writing after 2,048 tokens of chat |         |       10.3 |
...
  Flash attention off: 27% faster after 2,048 tokens of chat, -3% in an empty one.
```

What the lines mean:
- **Writing:** how fast new text appears. The percentage is the gain over llama.cpp's defaults, measured with both taking turns (when time is short, the defaults are measured a second time instead).
- **Reading the prompt:** how fast your message (and, in apps, their instructions) is read before the answer starts.
- **Writing after 2,048 tokens of chat:** the speed later in a conversation, when the model also has to read everything said so far.

Settings are only changed when they win beyond the measurement noise; otherwise "llama.cpp's default settings are already the fastest on this PC" is a valid result. The result is saved, and `launch` uses it from then on. How the measurements work: [Benchmark protocol](benchmark-protocol.md).

## models

List every model tuned on this PC, fastest first, with everything measured.

```bash
llama-autotune models
llama-autotune models --fits-now    # only models that fit in the RAM free right now
```

For each model:
- the verdict
- writing speed and its gain over the defaults
- prompt speed and long-chat speed
- the tuned settings
- size, and active size for MoE models
- whether it can read images
- whether it fits in the RAM free now (or how much is missing)
- the date and llama.cpp build of the tuning

```text
1. granite-3.1-1b-a400m-instruct-Q4_K_M  Fast
   1.0B MoE (0.43B active), 0.8 GB
   Writing 47.8 (+0%); reading prompts 157.9; long chat: run tune again to measure
   Settings: default threads, 4 for prompts, flash attention off
   Fits now: no, 174 MiB short; tuned 2026-11-09 on llama.cpp b12228
```

At the end it names the fastest and the largest model that fit now, with the command to start one, and says which results come from an older llama.cpp. The fastest model is not always the best one: larger models are usually more capable but slower.

"Fits now" says "file missing" when the model is no longer at the path it was tuned at, and "file unreadable" when the file is damaged. A moved model keeps its tuned settings in `launch`; run `tune` on it again to update the path shown here.

## launch

Start llama-server with settings that fit in the RAM free right now, then open the printed address in your browser.

```bash
llama-autotune launch model.gguf
```

| Option | Default | Meaning |
|---|---|---|
| `-c`, `--ctx-size` | 4096 | Context size in tokens (256 or more), or `max` for the largest that fits now |
| `--images` | off | Also read images, using the model's vision file |
| `--mmproj FILE` | | Use this vision file (implies `--images`) |
| `--editing` | off | Faster when rewriting text you paste in |
| `--host` | 127.0.0.1 | Address to listen on |
| `--port` | 8080 | Port to listen on |
| `-p`, `--profile FILE` | | Start with the options in a profile file, after checking them (see [export and import](#export-and-import)) |
| `--dry-run` | | Print the llama-server command without starting it |
| `--forget-settings` | | Forget the extra options remembered for this model |
| *any other llama-server option* | | Passed through and remembered, e.g. `--temp 0.7`. Options that load a different model are refused; passwords are not remembered |

What `launch` does, in order:
1. **Settings:** uses the settings `tune` saved for this model and PC. If llama.cpp was updated since, it says so.
2. **Memory:** plans the memory. If the model does not fit, it applies savings in order of least harm (smaller batch, weight repacking off, flash attention on, compressed KV cache, shorter context). If it still does not fit, it refuses with the exact amount missing instead of letting the PC swap.
3. **Safe server defaults for a desktop:** one chat slot, no large prompt cache on 8 GB PCs, few context checkpoints, above-normal priority, and the vision file only when asked for.
4. **Your options:** adds the options you gave or saved earlier.
5. **Start:** starts llama-server and prints the chat address, plus the base URL, model name and context for apps.

On a PC with a graphics card, step 2 is skipped: llama.cpp places the model on the card itself, and llama-autotune does not check the memory.

Examples:

```bash
llama-autotune launch model.gguf -c 16384                        # longer conversations
llama-autotune launch model.gguf -c max                          # the longest that fits now
llama-autotune launch model.gguf --images                        # read images too, if they fit
llama-autotune launch model.gguf temperature=0.7 top_p=0.8       # the model's recommended settings
llama-autotune launch model.gguf --editing                       # rewriting pasted text
llama-autotune launch model.gguf --dry-run                       # only show the command
```

Details on context size, your own options, images, editing and apps: [Chat and apps](chat-and-apps.md).

## benchmark

Run one llama-bench measurement with the settings you choose, under the same memory watchdog as `tune`. Useful for checking a single setting by hand.

```bash
llama-autotune benchmark model.gguf -t 3 -r 3
llama-autotune benchmark model.gguf -t 4 --no-flash-attn -d 2048    # after 2,048 tokens of chat
```

| Option | Meaning |
|---|---|
| `-t`, `--threads` | CPU threads |
| `-b`, `--batch-size` | Batch size |
| `-ub`, `--ubatch-size` | Physical batch size |
| `-ngl`, `--n-gpu-layers` | Layers on the graphics card |
| `--flash-attn` / `--no-flash-attn` | Flash attention on or off |
| `--repack` / `--no-repack` | Weight repacking on or off |
| `-d`, `--depth` | Tokens already in the context |
| `-r`, `--repetitions` | Repetitions (default 3) |
| `-o`, `--output FILE` | Save the result as JSON |

Without any settings it uses a reasonable starting configuration. The result is saved to the history.

## search

Advanced: explore many settings with a three-stage search (a starting configuration, a local grid, then Bayesian optimization with Optuna). At the end the winner and the starting configuration are measured again, taking turns; the winner is kept only when it still leads by more than 4%. On a CPU, `tune` is faster and more reliable; `search` is kept for experiments. It is experimental on graphics cards: its results there have not been verified on real hardware.

```bash
llama-autotune search model.gguf
llama-autotune search model.gguf --objective max_prompt_tps --profile best.json
```

| Option | Default | Meaning |
|---|---|---|
| `-O`, `--objective` | balanced | What to optimize (table below) |
| `--trials-b` | 15 | Trials in the grid stage |
| `--trials-c` | 40 | Trials in the Bayesian stage |
| `-p`, `--profile FILE` | | Save the best result as a profile |

| Objective | Optimizes for |
|---|---|
| `balanced` | Writing and prompt reading speed together (their geometric mean) |
| `max_generation_tps` | Writing speed |
| `max_prompt_tps` | Prompt reading speed |
| `min_latency` | Shortest wait for the first word (reading the test prompt, then one token) |
| `max_efficiency` | Writing speed per MB of memory |

`max_context` is not searched: llama-bench cannot measure a context size. For the largest context that fits in free RAM, use `launch -c max`.

A first quick test sorts the PC into a speed class and sizes each benchmark to it. The number of trials does not change with the class: it is what `--trials-b` and `--trials-c` say, up to 55 by default, plus four runs for the final comparison. On a PC too slow to test (under 1 token per second), or when the quick test fails, `search` returns a safe starting configuration without measuring.

| Speed class | Writing speed | Benchmark size |
|---|---|---|
| very slow | under 1 token/s | not measured |
| slow | 1 to 4 | 64 prompt, 32 written, 1 repetition |
| medium | 4 to 15 | 256 prompt, 64 written, 2 repetitions |
| fast | over 15 | 512 prompt, 128 written, 3 repetitions |

With 55 trials, a search takes much longer than `tune`.

## export and import

Save and load launch profiles: a model path, the llama-server options, and the llama.cpp build they were made for. `export` writes the settings `tune` measured for a model on this PC (run `tune` first); `import` shows a profile file before you use it.

```bash
llama-autotune export profile.json --model model.gguf
llama-autotune import profile.json
llama-autotune launch model.gguf --profile profile.json
```

A profile can come from anyone, so `launch --profile` only accepts the options llama-autotune itself writes (threads, context, batch sizes, flash attention, cache types and the like). A profile that carries anything else, such as a network address or a file to write, is refused. On a CPU the profile is also checked against the RAM free right now, and is not started when it does not fit.

Profiles are specific to a PC. Treat one from another PC as a starting point and run `tune` on your own. `launch --profile` warns when the profile was made for a different llama.cpp build, and refuses options the installed build does not accept.
