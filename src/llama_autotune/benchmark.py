"""Subprocess wrapper around llama-bench.

Runs benchmarks with arguments rendered by the compatibility layer,
guards every run with a RAM watchdog, and parses llama-bench's JSON
output into structured results. :func:`run_sweep` measures many
candidate settings in one llama-bench process through its comma lists,
which avoids paying the start-up cost once per candidate.
"""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import median

import psutil

from .compat import FlagRegistry, binary_name, clean_env, find_llama_binary, render_args, unknown_architecture
from .models import BenchmarkResult, SearchConfig

__all__ = ["BenchRecord", "SweepResult", "binary_name", "find_llama_binary", "find_llama_bench",
           "parse_records", "run_benchmark", "run_sweep", "summarize"]

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
    cmd = [bench_path, "-m", str(model_path), "-r", str(repetitions), "-o", "jsonl",
           "-p", str(n_prompt), "-n", str(n_gen)]
    if n_depth:
        cmd += ["-d", str(n_depth)]
    if no_warmup:
        cmd.append("--no-warmup")
    if config is not None:
        cmd.extend(render_args(config, "bench", registry).args)

    run = _execute(cmd, timeout, min_available_mib)
    result.startup_time = run.elapsed
    result.memory_usage = run.peak_rss_mib
    result.min_available_mb = run.min_available_mib
    result.raw_output = run.stdout
    if run.error:
        result.error = run.error
        result.raw_output = run.detail
        return result
    records = parse_records(run.stdout)
    if records:
        result.prompt_tps, result.generation_tps = summarize(records)
        result.success = True
    else:
        result.error = "failed"
    return result


def summarize(records: list[BenchRecord]) -> tuple[float, float]:
    """``(prompt, generation)`` tokens per second of one llama-bench run.

    A test that generates counts as generation (also the combined prompt and
    generation test of older builds); a prompt-only test counts as prompt.
    """
    prompt = max((r.avg_ts for r in records if r.kind == "pp"), default=0.0)
    generation = max((r.avg_ts for r in records if r.kind != "pp"), default=0.0)
    return prompt, generation


@dataclass
class _Run:
    """Outcome of one guarded llama-bench process."""

    stdout: str = ""
    stderr: str = ""
    elapsed: float = 0.0
    peak_rss_mib: float = 0.0
    min_available_mib: float = 0.0
    error: str = ""
    detail: str = ""


def _execute(cmd: list[str], timeout: float, min_available_mib: int) -> _Run:
    """Run llama-bench under the RAM watchdog, keeping whatever it printed."""
    run = _Run()
    available_mib = psutil.virtual_memory().available / (1024 * 1024)
    run.min_available_mib = round(available_mib, 1)
    if available_mib < min_available_mib:
        run.error = "memory_guard_external"
        run.detail = f"[MEMORY_GUARD_EXTERNAL] only {available_mib:.0f} MiB free before starting"
        return run
    try:
        with tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as out, \
             tempfile.TemporaryFile("w+", encoding="utf-8", errors="replace") as err:
            start = time.monotonic()
            proc = subprocess.Popen(
                cmd, stdout=out, stderr=err, env=clean_env(),
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                watch = watch_process(proc, timeout, min_available_mib)
            finally:
                # Ctrl+C lands here: on Windows the hidden llama-bench never sees it
                # and would keep running at raised priority after we are gone.
                if proc.poll() is None:
                    proc.kill()
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        pass
            run.elapsed = time.monotonic() - start
            out.seek(0)
            err.seek(0)
            run.stdout, run.stderr = out.read(), err.read()
    except FileNotFoundError:
        run.error, run.detail = "not_found", f"[ERROR] llama-bench not found at: {cmd[0]}"
        return run
    except OSError as e:
        run.error, run.detail = "failed", f"[ERROR] could not run llama-bench: {e}"
        return run
    run.peak_rss_mib = watch.peak_rss_mib
    run.min_available_mib = watch.min_available_mib
    arch = unknown_architecture(run.stderr)
    if watch.stopped_by:
        run.error = watch.stopped_by
        run.detail = f"[{watch.stopped_by.upper()}] {watch.detail}\n{run.stderr[-2000:]}"
    elif arch:
        run.error, run.detail = "unsupported_arch", arch
    elif proc.returncode != 0:
        run.error = "failed"
        run.detail = f"{run.stdout}\nSTDERR: {run.stderr[-4000:]}"
    return run


@dataclass
class BenchRecord:
    """One llama-bench test result (one line of its JSONL output)."""

    threads: int = 0
    flash_attn: bool | None = None
    repack: bool | None = None
    ubatch: int = 0
    n_prompt: int = 0
    n_gen: int = 0
    n_depth: int = 0
    avg_ts: float = 0.0
    stddev_ts: float = 0.0
    samples_ts: list[float] = field(default_factory=list)

    @property
    def kind(self) -> str:
        """``"pp"`` (prompt only), ``"tg"`` (generation only) or ``"pg"``."""
        if self.n_gen == 0:
            return "pp"
        return "tg" if self.n_prompt == 0 else "pg"

    @property
    def median_ts(self) -> float:
        """Median tokens per second over the repetitions."""
        return median(self.samples_ts) if self.samples_ts else self.avg_ts

    @property
    def spread(self) -> float:
        """Relative standard deviation (0.03 = 3%)."""
        return self.stddev_ts / self.avg_ts if self.avg_ts else 0.0


def _flag_bool(value) -> bool | None:
    if isinstance(value, bool):
        return value
    if value in (0, 1, "0", "1"):
        return bool(int(value))
    return None  # -1 or "auto"


def parse_records(output: str) -> list[BenchRecord]:
    """Parse llama-bench JSONL output, skipping lines that are not results.

    A run stopped by the watchdog or a timeout leaves its finished tests
    in the output, so partial output still yields records.
    """
    records = []
    for line in output.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            d = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(d, dict) or "avg_ts" not in d:
            continue
        records.append(BenchRecord(
            threads=int(d.get("n_threads", 0)),
            flash_attn=_flag_bool(d.get("flash_attn")),
            repack=_flag_bool(d.get("repack")),
            ubatch=int(d.get("n_ubatch", 0)),
            n_prompt=int(d.get("n_prompt", 0)),
            n_gen=int(d.get("n_gen", 0)),
            n_depth=int(d.get("n_depth", 0)),
            avg_ts=float(d.get("avg_ts", 0.0)),
            stddev_ts=float(d.get("stddev_ts", 0.0)),
            samples_ts=[float(x) for x in d.get("samples_ts") or []],
        ))
    return records


@dataclass
class SweepResult:
    """Records and run statistics of one llama-bench sweep."""

    records: list[BenchRecord] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    elapsed: float = 0.0
    peak_rss_mib: float = 0.0
    min_available_mib: float = 0.0
    error: str = ""
    detail: str = ""


def render_lists(lists: dict[str, list], registry: FlagRegistry | None) -> tuple[list[str], list[str]]:
    """Render setting value lists as llama-bench comma-list arguments.

    Each value is rendered through the compatibility layer and the values
    of one setting are joined with commas (``-t 2,3,4``). A setting the
    build does not accept is left out and named in the second list.
    """
    args: list[str] = []
    skipped: list[str] = []
    for name, values in lists.items():
        rendered = []
        for v in values:
            r = render_args(SearchConfig(**{name: v}), "bench", registry)
            rendered.append(r.args if not r.skipped and len(r.args) == 2 else None)
        if not rendered or any(r is None for r in rendered) or len({r[0] for r in rendered}) != 1:
            skipped.append(name)
            continue
        args += [rendered[0][0], ",".join(dict.fromkeys(r[1] for r in rendered))]
    return args, skipped


def run_sweep(
    model_path: str | Path,
    base: SearchConfig | None = None,
    lists: dict[str, list] | None = None,
    *,
    n_prompt: int = 0,
    n_gen: int = 0,
    repetitions: int = 1,
    n_depth: int = 0,
    numa: str | None = None,
    delay: int = 0,
    registry: FlagRegistry | None = None,
    timeout: float = 600,
    min_available_mib: int = DEFAULT_MIN_AVAILABLE_MIB,
) -> SweepResult:
    """Benchmark every combination of ``lists`` in one llama-bench process.

    Args:
        model_path: GGUF model.
        base: Settings shared by every candidate.
        lists: Setting name to candidate values, e.g. ``{"threads": [2, 3, 4]}``.
        n_prompt: Prompt-processing test size; 0 skips the prompt test.
        n_gen: Generation test size; 0 skips the generation test.
        repetitions: Repetitions per test (``-r``).
        n_depth: Tokens already in the context (``-d``).
        numa: llama-bench ``--numa`` mode for multi-socket machines.
        delay: Seconds of pause between tests, against heat build-up.
        registry: Flags of the installed llama-bench.
        timeout: Seconds before the process is stopped.
        min_available_mib: RAM floor for the watchdog.

    Returns:
        A :class:`SweepResult`; on failure ``error`` is set and
        ``records`` holds the tests finished before it.
    """
    out = SweepResult()
    cmd = [find_llama_bench(), "-m", str(model_path), "-r", str(repetitions), "-o", "jsonl",
           "-p", str(n_prompt), "-n", str(n_gen)]
    if n_depth:
        cmd += ["-d", str(n_depth)]
    if numa:
        cmd += ["--numa", numa]
    if delay and (registry is None or registry.has("--delay")):
        cmd += ["--delay", str(delay)]
    lists = lists or {}
    if base is not None:
        fixed = base.model_copy(update={k: None for k in lists})
        cmd += render_args(fixed, "bench", registry).args
    list_args, out.skipped = render_lists(lists, registry)
    cmd += list_args
    out.command = cmd
    run = _execute(cmd, timeout, min_available_mib)
    out.records = parse_records(run.stdout)
    out.elapsed, out.peak_rss_mib, out.min_available_mib = run.elapsed, run.peak_rss_mib, run.min_available_mib
    out.error, out.detail = run.error, run.detail
    if not out.error and not out.records:
        out.error, out.detail = "failed", run.stderr[-4000:]
    return out


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
    start = time.monotonic()
    start_avail = psutil.virtual_memory().available
    low = start_avail
    peak = 0
    children: list[psutil.Process] = []
    polls = 0

    while proc.poll() is None:
        if time.monotonic() - start > timeout:
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
