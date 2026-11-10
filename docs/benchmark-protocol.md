# Benchmark protocol

This is how `llama-autotune tune` measures a model on a CPU. Its goals are to finish within a time limit (10 minutes by default), to never run anything that does not fit in free memory, and to never recommend settings that are slower than llama.cpp's own defaults.

## 1. Memory gate

Before anything runs, the model's memory need is computed from its file: the weights, the KV cache for the chosen context, llama.cpp's compute buffer, the private copy made by weight repacking, and a small runtime allowance (vocabulary x 384 bytes, at least 32 MiB) for memory llama.cpp's own breakdown does not list. That total is compared with the RAM free right now, minus a safety margin of 10% of total RAM (at least 512 MiB).

The compute buffer starts from one `llama-fit-params` answer (4,096 tokens, flash attention on) and is modelled from three parts: output logits (vocabulary x ubatch x 4 bytes in llama-bench; one token in llama-server from b12228), a core working area, and, with flash attention off, attention scores of context x ubatch x heads x 4 bytes, which grow with the context (1 GiB for Llama 3.2 1B at 16K). Checked against llama-server and llama-fit-params for five models, both flash-attention settings and 4K and 16K contexts, it is never below llama.cpp's figure. Real llama-server working sets on the test PC came out 40 to 70 MiB under the estimate.

If it does not fit, savings are applied in this order: a smaller ubatch, repacking off, a q8_0 KV cache, and then a shorter context. If it still does not fit, nothing runs and the tool reports the shortfall. Settings the gate turned off, such as repacking, are never tested.

## 2. Pre-flight

The tool warns when the PC is on battery, battery saver is on, or other programs use more than 25% of the CPU. Each of these lowers or destabilizes the speed.

## 3. Prediction

Generating a token reads every active weight once, plus the KV cache filled so far. On a CPU this is limited by memory bandwidth, so:

```
predicted tokens/s = efficiency x bandwidth / bytes read per token
```

- **Bandwidth** is a memory-copy test counting read plus write bytes. It runs one copy thread, then one thread per performance core (up to 16), and takes the higher result, because one core cannot saturate the memory of workstations and servers. Its buffers never take more than a quarter of the RAM free at that moment. The highest bandwidth ever measured on the PC is used, because other programs can only make the test slower.
- **Bytes read per token** come from the model's tensor table. Lookup tables such as token embeddings are excluded because only one row is read per token.
- **Efficiency** starts at 0.8. Measured models on the test PC reached 0.66 to 0.98 of the bandwidth limit, depending on architecture.

If the prediction is below 0.3 tokens/s, even one test would take too long. The tool then gives a predicted verdict without measuring.

## 4. Baseline and calibration

llama.cpp's defaults run first, with the gate's mandatory savings applied and two repetitions. This run serves three purposes:

- **Bar to beat:** the final settings must beat it by more than the noise.
- **Calibration:** the measured speed divided by the bandwidth limit gives this model's efficiency on this PC.
- **Test sizes:** generation tests are set to about 3 seconds for screening and 6 seconds for confirming, between 8 and 128 tokens.

## 5. Screening

All candidates run in **one llama-bench process**, using its comma lists (`-t 2,3,4 -fa off,on --repack 0,1`). The model loads once per repacking value, not once per candidate. Each candidate gets one generation test and one prompt test, one repetition each.

| Setting | Values |
|---|---|
| Threads, CPU with up to 8 threads | Every count from one below the performance-core count up to all logical threads. On the 2-core/4-thread test PC, 3 threads were 22% faster than the default 2. |
| Threads, larger CPUs | A quarter, half and three quarters of the performance cores, all of them, and all their threads. On multi-socket PCs, one socket's cores are also tried. Generation is bandwidth-bound and often peaks well below the core count. |
| Efficiency cores | Never used. |
| Flash attention | off, on |
| Weight repacking | off, on; only when its extra memory fits and the model has weights that can be repacked |

If the estimated screening time exceeds 40% of the time left, the tool cuts work in this order: smaller tests, then no repacking test, then the thread counts furthest from the default.

If the screening is stopped (time limit or memory watchdog) before any candidate has a generation speed, there is nothing to compare, and llama.cpp's defaults are kept.

## 6. Confirmation (successive halving)

The best third of the screened candidates (at least 2), and any candidate within 8% of the leader, run again with three repetitions and longer tests: at most 6 in all. Untested thread counts halfway between the best one and its neighbors are added. For example, if 32 threads won between 16 and 48, then 24 and 40 are tried.

## 7. Prompt settings

- **Prompt threads (`-tb`):** chosen from the prompt tests, because llama-bench has no `-tb`. Reading a prompt is compute-bound and often wants more threads than writing. The screen measures each prompt speed only once, so a different thread count for prompts is measured again with three repetitions when time allows, and kept only if it still wins by more than 6%.
- **ubatch:** 128, 256 and 512 (up to the gate's limit) are tried on a 512-token prompt when time allows, after the time for section 10 has been set aside. A better ubatch is kept even when the defaults win for writing, because it only affects prompts.

## 8. Multi-socket PCs

On PCs with more than one NUMA node, the winner also runs with `--numa distribute`. It is kept only if it wins beyond the noise.

## 9. Decision rules

- **Noise:** the larger of 4% and twice the measured relative standard deviation. On the test PC, with 3 repetitions, the spread was 2.6 to 6.8% for generation and 2.7 to 10.9% for prompt processing.
- **Ties:** among candidates within the noise of the fastest, the one with repacking off (less memory) and then fewer threads (a more responsive PC) is kept.
- **Defaults:** if the winner does not beat llama.cpp's defaults beyond the noise, the defaults are kept, and the tool says so.

## 10. Speed with a filled context

The winner runs once with half the context already filled (at most 2,048 tokens). Every generated token then also reads that KV cache. The speed is shown next to the start-of-chat speed. In the earlier experiments, 4,096 tokens of context cost 27%.

When flash attention was screened, this run tries it both off and on. On a CPU it can tie in an empty context and still be much slower once the chat grows: on llama.cpp b12228, Llama 3.2 1B wrote 11.4 tokens/s either way in an empty context, but 9.3 (off) against 5.8 (on) after 2,048 tokens. Each setting scores the geometric mean of its speed in an empty chat and after the long chat, because a conversation spends its time in between. The empty-chat speeds of both settings come from the same measurement (the confirmation, else the screen), and the setting switches only when its score wins beyond the noise. An earlier rule compared the empty-chat speed of one setting from the screen with the winner's speed from the duel; on the busy test PC that mismatch kept flash attention on and left a long chat at 7.0 tokens/s. This matters because llama.cpp's default, "auto", turns flash attention on for the CPU.

When this test changes the flash-attention setting, the reported start-of-chat speed is adjusted to the setting that is saved. When the defaults had won and "auto" already meant the winning setting, the setting is only written out explicitly and the speed stands.

## 11. Final duel against the defaults

Right after the confirmation (before sections 7 to 10), the winner and llama.cpp's defaults take turns, one short llama-bench process each: defaults, winner, defaults, winner, three pairs. The gain is the median of the three winner/default ratios. The noise is twice the standard error of that median, from the median absolute deviation, between 4% and 10%, so one disturbed run does not decide. When the gain is positive but within the noise, two more pairs run if time allows.

Why: on the busy test PC, the same comparison (3 threads against the default 2) measured sequentially (all default runs, then all winner runs) came out anywhere from -3% to +22% when another program started halfway, twice reaching the wrong verdict. Taking turns, it came out +8% to +21%, always the right way, because both sides meet the same disturbance.

If the speed of the defaults changed by more than twice the noise (at least 8%) since the first measurement, another program or heat disturbed the test, and the tool suggests running it again on an idle PC. Without time for the duel, the defaults run once more and are averaged with the first run, as before.

## Priority

Every llama-bench run uses above-normal priority (`--prio 1`), and `launch` starts llama-server the same way (`--prio 1 --prio-batch 1`). With two other programs keeping half of the test PC's CPU busy, this gave 13% more speed (11.6 against 10.3 tokens/s) and much steadier runs (10.9 to 11.6 against 7.8 to 11.2). High priority (`--prio 2`) gave nothing more and makes the rest of Windows less responsive, so it is not used.

## 12. Safety

Every llama-bench process runs under the RAM watchdog, which polls 4 times a second. It stops the process when free RAM falls below 400 MiB, and it tells whether the model or another program used the memory. Results finished before a stop are kept. Stopping `tune` with Ctrl+C stops the running llama-bench too.

## 13. What is stored

The winning settings, the speeds of the defaults and the winner, the speed in a long chat, the verdict, the calibrated efficiency, the bandwidth, the context, the model's full path and the llama.cpp build are stored in `~/.llama-autotune/benchmarks.db`. They are keyed by a hardware fingerprint and the model file's fingerprint. `launch` applies them automatically and warns when llama.cpp has been upgraded since.

## Verdicts

| Verdict | Generation speed | Meaning |
|---|---|---|
| Fast | 10 tokens/s or more | Text appears faster than you can read it |
| Usable | 4 to 10 tokens/s | About reading speed (a typical reader manages about 5 tokens/s) |
| Slow | under 4 tokens/s | Slower than reading; a smaller model or quantization will feel better |
| Won't fit | — | The model does not fit in the memory free right now |
