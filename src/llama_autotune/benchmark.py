"""Subprocess wrapper around llama-bench.

Runs benchmarks with arguments rendered by the compatibility layer,
guards every run with a RAM watchdog, and parses llama-bench's JSON
output into structured results.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

import psutil

from .compat import FlagRegistry, binary_name, clean_env, find_llama_binary, render_args
from .models import BenchmarkResult, SearchConfig

__all__ = ["binary_name", "find_llama_binary", "find_llama_bench", "run_benchmark"]

DEFAULT_MIN_AVAILABLE_MIB = 400


def find_llama_bench() -> str:
    """Locate the llama-bench executable.

    Returns:
        Absolute path to llama-bench (see :func:`find_llama_binary`).
    """
    return find_llama_binary("llama-bench")


def run_benchmark(
    model_path: str | Path,
    config: SearchConfig | None = None,
    repetitions: int = 3,
    timeout: int = 300,
    n_prompt: int = 512,
    n_gen: int = 128,
    no_warmup: bool = False,
    n_depth: int = 0,
    registry: FlagRegistry | None = None,
    min_available_mib: int = DEFAULT_MIN_AVAILABLE_MIB,
) -> BenchmarkResult:
    """Execute llama-bench for a given model and return the results.

    Arguments come from the compatibility layer, so only flags the
    installed build accepts are used. ``LLAMA_ARG_*`` environment
    variables are removed so they cannot silently change the result.
    A watchdog stops the run when available RAM falls below
    ``min_available_mib``.

    Args:
        model_path: Path to the model file to benchmark.
        config: Settings to benchmark; None uses llama-bench defaults.
        repetitions: Number of repetitions (``-r``).
        timeout: Maximum seconds to wait for the subprocess.
        n_prompt: Prompt tokens (``-p``).
        n_gen: Generated tokens (``-n``).
        no_warmup: Skip llama-bench's warmup run.
        n_depth: Tokens already in the KV cache (``-d``); 0 omits the flag.
        registry: Flags of the installed llama-bench; None renders the
            current llama.cpp spelling.
        min_available_mib: RAM floor for the watchdog.

    Returns:
        A :class:`BenchmarkResult`. On failure ``success`` is False and
        ``error`` says why (``timeout``, ``memory_guard``,
        ``memory_guard_external``, ``not_found`` or ``failed``).
    """
    result = BenchmarkResult()
    bench_path = find_llama_bench()
    cmd = [bench_path, "-m", str(model_path), "-r", str(repetitions), "-o", "json",
           "-p", str(n_prompt), "-n", str(n_gen)]
    if n_depth:
        cmd += ["-d", str(n_depth)]
    if no_warmup:
        cmd.append("--no-warmup")
    if config is not None:
        cmd.extend(render_args(config, "bench", registry).args)

    available_mib = psutil.virtual_memory().available / (1024 * 1024)
    if available_mib < min_available_mib:
        result.error = "memory_guard_external"
        result.min_available_mb = round(available_mib, 1)
        result.raw_output = f"[MEMORY_GUARD_EXTERNAL] only {available_mib:.0f} MiB free before starting"
        return result

    try:
        with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as out, \
             tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:
            start = time.time()
            proc = subprocess.Popen(
                cmd, stdout=out, stderr=err, env=clean_env(),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            watch = watch_process(proc, timeout, min_available_mib)
            result.startup_time = time.time() - start
            out.seek(0)
            err.seek(0)
            stdout, stderr = out.read(), err.read()
    except FileNotFoundError:
        result.error = "not_found"
        result.raw_output = f"[ERROR] llama-bench not found at: {bench_path}"
        return result
    except OSError as e:
        result.error = "failed"
        result.raw_output = f"[ERROR] could not run llama-bench: {e}"
        return result

    result.raw_output = stdout
    result.memory_usage = watch.peak_rss_mib
    result.min_available_mb = watch.min_available_mib
    if watch.stopped_by:
        result.error = watch.stopped_by
        result.raw_output = f"[{watch.stopped_by.upper()}] {watch.detail}\n{stderr[-2000:]}"
        return result
    if proc.returncode != 0:
        result.error = "failed"
        result.raw_output += f"\nSTDERR: {stderr[-4000:]}"
        return result

    parsed = _parse_benchmark_output(stdout)
    if parsed:
        result.prompt_tps = parsed.get("prompt_tps", 0.0)
        result.generation_tps = parsed.get("generation_tps", 0.0)
        result.success = True
    else:
        result.error = "failed"
    return result


@dataclass
class WatchReport:
    """What the watchdog observed while a process ran."""

    peak_rss_mib: float = 0.0
    min_available_mib: float = 0.0
    stopped_by: str = ""
    detail: str = ""


def watch_process(proc: subprocess.Popen, timeout: float, min_available_mib: int) -> WatchReport:
    """Watch a process until it exits; stop it on timeout or low RAM.

    When available RAM falls below the floor the process is killed. The
    stop is blamed on the process (``memory_guard``) only when available
    RAM fell by a meaningful amount and the process itself accounts for at
    least half of that drop. Otherwise another program took the memory, or
    it was already low, and the stop is reported as
    ``memory_guard_external`` so the caller can wait and retry instead of
    rejecting the configuration.

    Args:
        proc: The running process.
        timeout: Seconds before the process is killed.
        min_available_mib: Available-RAM floor in MiB.

    Returns:
        A :class:`WatchReport`.
    """
    report = WatchReport()
    mib = 1024 * 1024
    try:
        ps = psutil.Process(proc.pid)
    except psutil.Error:
        ps = None
    start = time.time()
    start_avail = psutil.virtual_memory().available
    low = start_avail
    peak = 0
    children: list[psutil.Process] = []
    polls = 0

    while proc.poll() is None:
        if time.time() - start > timeout:
            report.stopped_by, report.detail = "timeout", f"no result after {timeout:.0f} s"
            break
        rss = 0
        if ps is not None:
            try:
                if polls % 8 == 0:
                    children = ps.children(recursive=True)
                rss = ps.memory_info().rss
                for c in children:
                    try:
                        rss += c.memory_info().rss
                    except psutil.Error:
                        pass
            except psutil.Error:
                pass
        polls += 1
        peak = max(peak, rss)
        avail = psutil.virtual_memory().available
        low = min(low, avail)
        if avail < min_available_mib * mib:
            drop = start_avail - avail
            own = drop >= 64 * mib and rss >= drop * 0.5
            report.stopped_by = "memory_guard" if own else "memory_guard_external"
            report.detail = (f"available RAM fell to {avail // mib} MiB "
                             f"(process used {rss // mib} MiB, system lost {max(drop, 0) // mib} MiB)")
            break
        time.sleep(0.25)

    if report.stopped_by:
        proc.kill()
    try:
        proc.wait(timeout=15)
    except subprocess.TimeoutExpired:
        pass
    report.peak_rss_mib = round(peak / mib, 1)
    report.min_available_mib = round(low / mib, 1)
    return report


def _parse_benchmark_output(output: str) -> dict | None:
    """Parse the JSON output from llama-bench into a flat dictionary.

    Handles JSON arrays, single JSON objects, and JSONL (one object per
    line).  Works with both the current llama-bench format (using
    ``avg_ts``) and legacy format (``pp_avg`` / ``tg_avg``).

    When ``--no-warmup`` is used, llama-bench outputs two entries: a
    prompt-processing entry (``n_gen=0``) and a generation entry
    (``n_gen>0``).  The function identifies each by the ``n_gen`` field.

    Args:
        output: The raw stdout text from llama-bench.

    Returns:
        A dictionary with keys ``prompt_tps``, ``generation_tps``, and
        ``memory_usage``, or ``None`` if no valid data could be
        extracted.
    """
    entries: list[dict] = []

    # Try top-level JSON parse first (array or single object)
    try:
        data = json.loads(output)
        if isinstance(data, list):
            entries = data
        elif isinstance(data, dict):
            entries = [data]
    except (json.JSONDecodeError, ValueError, TypeError):
        # Fall back to line-by-line JSONL
        for line in output.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
                if isinstance(d, dict):
                    entries.append(d)
            except (json.JSONDecodeError, ValueError, TypeError):
                continue

    if not entries:
        return None

    prompt_tps = 0.0
    gen_tps = 0.0
    memory = 0.0
    is_split = len(entries) > 1

    for entry in entries:
        n_gen = entry.get("n_gen", 0)
        n_prompt = entry.get("n_prompt", 0)
        avg_ts = entry.get("avg_ts")

        if avg_ts is not None:
            ftps = float(avg_ts)
            if n_gen and n_gen > 0:
                gen_tps = ftps
            elif n_prompt and n_prompt > 0 and is_split:
                prompt_tps = ftps
            elif not is_split:
                gen_tps = ftps

        prompt_tps = max(prompt_tps,
                         float(entry.get("pp_avg", entry.get("prompt_tps", 0))))
        gen_tps = max(gen_tps,
                      float(entry.get("tg_avg", entry.get("generation_tps", 0))))
        memory = max(memory,
                     float(entry.get("mem_usage", entry.get("memory_usage", 0))))

    return {
        "prompt_tps": prompt_tps,
        "generation_tps": gen_tps,
        "memory_usage": memory,
    }
