# Getting started

This page takes you from nothing to a running chat. It takes about 10 minutes, plus the time to download a model.

## What you need

- **A PC** with Windows 10 or 11, Linux or macOS. llama-autotune was developed on Windows; Linux and macOS are tested automatically but have seen less real use.
- **RAM:** 8 GB is enough for small models (1 to 4 billion parameters). More RAM lets you run larger models and longer conversations.
- **No administrator rights** are needed.

## 1. Install uv

[uv](https://docs.astral.sh/uv/) installs llama-autotune and the Python it needs. You do not need to install Python yourself.

On Windows, open PowerShell and run:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

On Linux or macOS:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Close and reopen the terminal afterwards, so the `uv` command is found.

## 2. Install llama-autotune

```bash
uv tool install git+https://github.com/Najafu/llama-autotune
```

This puts the `llama-autotune` command on your PATH. To update later, run `uv tool upgrade llama-autotune`.

### Other ways to install

If you already have Python 3.12 or newer:

```bash
pip install git+https://github.com/Najafu/llama-autotune.git
```

From source, for development:

```bash
git clone https://github.com/Najafu/llama-autotune.git
cd llama-autotune
uv sync
uv run llama-autotune --help
```

## 3. Get llama.cpp

llama-autotune uses llama.cpp's own programs (`llama-server`, `llama-bench`) and does not download them for you.

1. Open the [llama.cpp releases page](https://github.com/ggml-org/llama.cpp/releases).
2. Under the newest release, open "Assets" and download the file for your PC:
   - **Windows PC without a graphics card:** the file ending in `bin-win-cpu-x64.zip`.
   - **Windows PC with an NVIDIA card:** a file with `cuda` in its name. **AMD or Intel graphics:** a file with `vulkan` in its name.
   - **Linux or macOS:** the file for your system, or build llama.cpp yourself.
3. Unzip it into a folder of its own, for example `C:\llama`.

The CPU zip contains versions for every processor generation, and llama.cpp picks the right one when it starts.

## 4. Tell llama-autotune where llama.cpp is

```bash
llama-autotune setup C:\llama
```

`setup` checks that the folder contains `llama-server` and `llama-bench` (it also looks in its subfolders) and saves the location. You only do this once, and again only if you move llama.cpp.

### How llama.cpp is found

llama-autotune looks in this order:

1. The `LLAMA_CPP_DIR` environment variable, if you set one.
2. The folder saved by `llama-autotune setup`.
3. Folders next to the installed package.
4. Your system `PATH`.
5. Common unzip places: `C:\llama`, `C:\llama.cpp`, `C:\llamacpp`, the same names in your user folder, and the newest `llama-b...-bin-...` folder in Downloads.

The folder you are standing in is not searched, so a program that only happens to be named `llama-server` is never run by accident. If none of these places has llama.cpp, the commands that need it say how to get it.

## 5. Check your PC

```bash
llama-autotune doctor
```

`doctor` shows your processor, the instruction set llama.cpp uses on it, total and free RAM, power source, memory speed, the llama.cpp build and any graphics devices. It ends with "No problems found." or a list of warnings, such as running on battery or llama.cpp settings left in environment variables.

## 6. Get a model

Models are `.gguf` files. A good place to find them is [Hugging Face](https://huggingface.co/models?library=gguf).

- **Size:** on an 8 GB PC, start with a model of 1 to 4 billion parameters ("1B" to "4B" in the name).
- **Quantization:** pick a file marked `Q4_K_M`; it is small and loses little quality.
- **Images:** vision models come with a second file named `mmproj-...gguf`. Put it in the same folder as the model.

Keep each model family in its own folder, for example `models\qwen`, `models\llama`.

## 7. Check that it fits

```bash
llama-autotune inspect model.gguf
```

`inspect` shows the model's details and a memory plan: for each context size, how much RAM it needs and whether that fits in the RAM free right now. If it says "Cannot read this model", the download did not finish: download the file again. If nothing fits, close programs you do not need (web browsers use the most) or choose a smaller model.

## 8. Tune it

```bash
llama-autotune tune model.gguf
```

This takes up to 10 minutes and only has to be done once per model (and again after you update llama.cpp). Close other programs first; the PC should be as quiet as possible while it measures. At the end you get a verdict:

| Verdict | Writing speed | What it feels like |
|---|---|---|
| Fast | 10 tokens per second or more | Text appears faster than you can read it |
| Usable | 4 to 10 tokens per second | About reading speed |
| Slow | under 4 tokens per second | Slower than reading; try a smaller model |
| Won't fit | | The model does not fit in the RAM free right now |

## 9. Chat

```bash
llama-autotune launch model.gguf
```

Open http://127.0.0.1:8080 in your browser and start chatting. Press Ctrl+C in the terminal to stop.

From now on, `llama-autotune launch` uses the settings `tune` measured. `llama-autotune models` lists every model you have tuned.

## Next steps

- [Chat and apps](chat-and-apps.md): longer conversations, the model's recommended settings, images, coding agents.
- [Commands](commands.md): everything each command can do.
- [Troubleshooting](troubleshooting.md): if something does not work.
