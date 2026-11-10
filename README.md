# llama-autotune

Run local AI models with [llama.cpp](https://github.com/ggml-org/llama.cpp) at their best speed on the PC you have, without filling up its memory.

## What is llama-autotune?

You can run AI chat models, like the ones behind ChatGPT but smaller, on your own PC: free, offline and private. One of the most widely used programs for this, and the engine inside many other AI apps, is **llama.cpp**. It is powerful, but it has dozens of settings: how many processor threads, how much conversation to keep, how to store the model in memory. The wrong ones make it much slower than it could be, or fill up the memory until the whole PC freezes.

**llama-autotune is a helper for llama.cpp that chooses those settings for you.** You give it a model file, and it:

1. checks your PC and tells you whether the model fits in the memory you have free;
2. measures which settings run fastest on *your* PC (once per model, up to 10 minutes);
3. tells you in plain words what to expect: **Fast**, **Usable**, **Slow** or **Won't fit**;
4. starts the model with those settings, so you can chat in your browser or connect an app such as a coding agent.

**It is:**
- a small command-line tool: you type short commands in PowerShell or a terminal;
- made for ordinary PCs, and tested on an older 2-core desktop processor (Intel i3-6100) with 8 GB of RAM and no graphics card;
- completely local: nothing is uploaded, and no account is needed.

**It is not:**
- an AI model: you download a model yourself, for example from Hugging Face;
- a chat app with its own window: the chat runs in your browser, using the page built into llama.cpp;
- a replacement for llama.cpp: you download llama.cpp once, and llama-autotune drives it;
- a way to make a small PC run big models: it gets the most out of your PC and tells you honestly when a model is too big.

**What to expect:** setting up takes about 10 minutes, and each new model takes up to 10 more to measure. On a PC without a graphics card, small models (1 to 4 billion parameters) write about as fast as you read or faster. Larger ones need more RAM, or a graphics card.

## Why use it instead of llama.cpp alone

- **Keeps your memory from filling up.** It plans against the RAM that is free right now and keeps a safety margin, so Windows stays responsive. A model that does not fit is refused with the exact amount missing.
- **Finds the fastest settings by measuring.** It takes up to 10 minutes per model and keeps a setting only when it measurably beats llama.cpp's own defaults.
- **Starts the chat** in your browser, or for apps such as coding agents, with the measured settings.
- **Works with your llama.cpp,** old or new, and never passes it an option it does not understand.

## Install

You need Windows 10 or 11 (Linux and macOS also work) and about 10 minutes.

1. Install [uv](https://docs.astral.sh/uv/). It also installs Python for you. In PowerShell:
   ```powershell
   powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
   ```
2. Install llama-autotune:
   ```bash
   uv tool install git+https://github.com/Najafu/llama-autotune
   ```
3. Download llama.cpp from its [releases page](https://github.com/ggml-org/llama.cpp/releases). For a PC without a graphics card, take the file ending in `bin-win-cpu-x64.zip`, and unzip it, for example to `C:\llama`.
4. Tell llama-autotune where llama.cpp is, and check your PC:
   ```bash
   llama-autotune setup C:\llama
   llama-autotune doctor
   ```

More detail: [Getting started](docs/getting-started.md).

## Use

Get a model as a `.gguf` file, for example from [Hugging Face](https://huggingface.co/models?library=gguf). Then:

```bash
llama-autotune tune model.gguf      # once per model: measures the fastest settings (up to 10 minutes)
llama-autotune launch model.gguf    # every time: starts the chat at http://127.0.0.1:8080
```

Press Ctrl+C in the terminal to stop the chat.

`tune` ends with a verdict, followed by two tables: the settings it chose next to llama.cpp's defaults, and the speeds it measured. For example:

```text
Verdict: Fast: about 12.9 tokens per second; text appears faster than you can read it.
                  Speed (tokens per second)
|                                    | Default |      Tuned |
| Writing (generation)               |    12.3 | 12.9 (+5%) |
| Reading the prompt                 |    59.2 |       95.4 |
| Writing after 2,048 tokens of chat |         |       10.3 |
```

## How do I...

| I want to... | Run |
|---|---|
| See if a model fits before trying it | `llama-autotune inspect model.gguf` |
| See every model I have tuned, and which fit now | `llama-autotune models` |
| Chat with a longer context | `llama-autotune launch model.gguf -c 16384` |
| Get the longest context that fits right now | `llama-autotune launch model.gguf -c max` |
| Let the model read images | `llama-autotune launch model.gguf --images` |
| Use the settings from the model's web page | `llama-autotune launch model.gguf temperature=0.7 top_p=0.8 top_k=20` |
| Rewrite or fix text I paste in, faster | `llama-autotune launch model.gguf --editing` |
| Use it from a coding agent such as opencode | See [Chat and apps](docs/chat-and-apps.md#coding-agents-and-other-apps) |
| Save a model's tuned settings to a file | `llama-autotune export settings.json --model model.gguf` |

Settings you give `launch` are remembered for that model (passwords such as `--api-key` are not).

## Documentation

- [Getting started](docs/getting-started.md): installing, finding llama.cpp, your first model
- [Commands](docs/commands.md): every command and option, with examples
- [Chat and apps](docs/chat-and-apps.md): context size, your own settings, images, coding agents
- [Troubleshooting and FAQ](docs/troubleshooting.md)
- [How it works](docs/how-it-works.md): memory planning, tuning, compatibility, where data is kept
- [Benchmark protocol](docs/benchmark-protocol.md): exactly how `tune` measures
- [Development](docs/development.md): tests, project structure, status
- [Changelog](CHANGELOG.md)

## Status

Version 0.5.0. Tuning is measured and tested on CPUs, on a 2-core Intel i3 with 8 GB of RAM, with llama.cpp b11347 and b12228. On PCs with a graphics card, `launch` lets llama.cpp place the model on it, but llama-autotune does not tune graphics cards yet.

MIT license. Bug reports and pull requests are welcome.
