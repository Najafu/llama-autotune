# Chat and apps

Everything about `llama-autotune launch`: how long a conversation can be, using the settings a model's page recommends, images, faster editing, and connecting apps such as coding agents.

## Context size: how long a conversation can be

The context is everything the model keeps in view: your messages, its answers, and any instructions an app sends. It is counted in tokens (roughly three quarters of a word each). When a conversation outgrows the context, the oldest part falls out of view.

```bash
llama-autotune launch model.gguf               # 4,096 tokens: a normal chat
llama-autotune launch model.gguf -c 16384      # longer conversations and documents
llama-autotune launch model.gguf -c max        # the longest that fits in the RAM free right now
```

The smallest context `launch` accepts is 256 tokens, and never more than the model was trained for.

A longer context needs more memory. If the one you ask for does not fit, `launch` makes room in this order:
1. **Smaller batch:** no speed cost.
2. **Weight repacking off:** somewhat slower.
3. **Flash attention on** (only when `tune` had turned it off): slower late in a conversation.
4. **Compressed context memory** (q8_0 KV cache).
5. **Shorter context.**

It prints every saving it applies.

`-c max` searches, in steps of 1,024 tokens, for the longest context that fits now, and applies those savings as needed. Expect it to be slower than the default, and to change from day to day with the RAM that is free.

`llama-autotune inspect model.gguf` shows the memory each context size needs before you start.

## Your own settings (temperature, top_p and so on)

Model pages usually recommend sampling settings, such as `temperature=0.7, top_p=0.8, top_k=20`. Give them to `launch` after the model:

```bash
llama-autotune launch model.gguf temperature=0.7 top_p=0.8 top_k=20 repetition_penalty=1.0 presence_penalty=1.5
```

- **Spellings:** both the names used on model pages and llama.cpp's own spellings work, written as `--temp 0.7`, `--temp=0.7` or `temperature=0.7`:

  | On model pages | llama.cpp option |
  |---|---|
  | `temperature` | `--temp` |
  | `top_p` | `--top-p` |
  | `top_k` | `--top-k` |
  | `min_p` | `--min-p` |
  | `typical_p` | `--typical` |
  | `repetition_penalty` | `--repeat-penalty` |
  | `presence_penalty` | `--presence-penalty` |
  | `frequency_penalty` | `--frequency-penalty` |
  | `max_tokens`, `max_new_tokens` | `-n` |
  | `seed` | `--seed` |

- **Any other llama-server option** works too, for example `--reasoning-budget 0`.
- **Checked before starting.** A typo gets a suggestion instead of a server that stops: `--topk` gives "Did you mean --top-k?". Values with a fixed list of choices are checked too.
- **Remembered per model.** The next `launch` of that model uses them again and prints them. Giving a setting again replaces its old value. `--forget-settings` clears them all.
- **Yours win.** If you give an option that llama-autotune would also set, such as `-t` for threads, yours is used and `launch` says which ones it replaced.
- **Memory options are your responsibility.** Options that change memory use, such as `-b`, `-ub` or `-ctk`, are not covered by the memory plan, and `launch` warns about them.
- **One model per launch.** Options that load a different model (`-m`, `--hf-repo` and similar) are refused, because the memory plan is for the model you named.
- **Passwords are not kept.** The value of `--api-key` or `--hf-token` is used for that start only. It is never saved, and it is shown as `***` in the printed command, so give it each time.

**Where they apply:** these settings become the server's defaults. The built-in web chat sends no sampling values of its own, so they apply there (checked on llama.cpp b12228). An app that sends its own values with each message, as some chat apps do, overrides them for those messages.

## Images

Vision models can read images. They come with a second file, the vision projector, usually named `mmproj-...gguf`. Keep it in the same folder as the model.

```bash
llama-autotune launch model.gguf --images
llama-autotune launch model.gguf --mmproj C:\models\qwen\mmproj-F16.gguf    # name it yourself
```

- **Matching:** `--images` finds the vision file that belongs to the model by its metadata, not its name. A vision file's output width equals the embedding width of the model it was made for. So a folder holding vision files for the 2B, 4B and 9B versions of a model never gets them mixed up.
- **Memory:** about the vision file's size plus half again. A 637 MiB vision file needed 912 MiB in testing, including reading one image. It is part of the memory plan.
- **Images are optional.** If the model fits at the context you asked for only *without* images, `launch` leaves images out, says how much was missing, and starts the text chat. A smaller `-c` can make room. With `-c max`, it reports the longest context with and without images.
- **Images are slow on a processor.** On the test PC, a 1280x960 screenshot took 94 seconds to read and used 1,221 tokens of context. Smaller images are much faster.

`inspect` shows whether a model has a vision file and what it costs.

## Faster editing of pasted text

```bash
llama-autotune launch model.gguf --editing
```

This turns on llama.cpp's n-gram speculative decoding (`--spec-type ngram-mod`). When an answer repeats a lot of your text, as when fixing or rewriting a pasted document, it was 1.7 to 2.6 times faster in testing. On new text it is about 7% slower, so it is off by default.

## Coding agents and other apps

`launch` starts an OpenAI-compatible server. When it starts, it prints what an app needs:

```text
For apps such as coding agents: base URL http://127.0.0.1:8080/v1, model name Qwen3.5-4B-Q4_K_M, context 20480 tokens.
```

- **Model name:** the file name without `.gguf`.
- **Tool calls**, which coding agents rely on, work out of the box.

### opencode

Add a provider to `opencode.json`, either in your project folder or in `~/.config/opencode/opencode.json`. The field names follow opencode's documentation for OpenAI-compatible servers at the time of writing; check its [providers page](https://opencode.ai/docs/providers/) if your version differs.

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
          "modelID": "Qwen3.5-4B-Q4_K_M",
          "capabilities": { "tools": true, "input": ["text"], "output": ["text"] },
          "limit": { "context": 20480, "output": 4096 }
        }
      }
    }
  }
}
```

- **`modelID`:** the model name `launch` printed.
- **`limit.context`:** the context `launch` printed. Without it, opencode assumes 200,000 tokens and sends more than the server holds.
- **Images:** add `"image"` to `input` only when `launch` says images are on. opencode's `media.image` settings can make screenshots smaller, which makes them much faster to read.
- **Sampling settings:** give them to `launch`, as shown above.

### What to expect on a PC without a graphics card

A coding agent sends thousands of tokens of instructions and tool descriptions before your first question.
- **The first request is slow.** On the test PC (2-core i3, 8 GB), Qwen3.5 4B read prompts at about 16 tokens per second, so 10,000 tokens take about 10 minutes.
- **Later requests are quick,** because llama-server keeps what it has already read.
- **Smaller models are faster:** they read prompts several times faster. `tune` and `models` show each model's prompt speed.
- **For heavy agent work with images,** plan on 16 GB of RAM or a graphics card.
