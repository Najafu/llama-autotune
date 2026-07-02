"""Subprocess wrapper around llama-bench.

Provides functions to locate the llama-bench executable, run benchmarks
against a given model, and parse the JSON output into structured results.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .models import BenchmarkResult, SearchConfig


def find_llama_bench() -> str:
    """Locate the llama-bench.exe executable.

    Checks the ``LLAMA_CPP_DIR`` environment variable first, then falls
    back to several relative paths derived from the current file's
    location.  Returns the first existing file found.

    Returns:
        Absolute path to llama-bench.exe, or ``"llama-bench.exe"`` as a
        last resort if none of the candidates exist.
    """
    candidates = [
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "llama-bench.exe"),
        os.path.join(os.path.dirname(__file__), "..", "..", "llama-bench.exe"),
    ]
    env_val = os.environ.get("LLAMA_CPP_DIR")
    if env_val:
        candidates.insert(0, os.path.join(env_val, "llama-bench.exe"))
    for c in candidates:
        resolved = os.path.abspath(c)
        if os.path.isfile(resolved):
            return resolved
    return "llama-bench.exe"


def run_benchmark(
    model_path: str | Path,
    config: SearchConfig | None = None,
    repetitions: int = 3,
    timeout: int = 300,
    n_prompt: int = 512,
    n_gen: int = 128,
    no_warmup: bool = False,
) -> BenchmarkResult:
    """Execute llama-bench for a given model and return the results.

    Runs the benchmark subprocess with the specified model, optional
    configuration, and number of repetitions.  The output is parsed from
    JSON and written into a ``BenchmarkResult`` instance.

    Args:
        model_path: Path to the model file to benchmark.
        config: Optional ``SearchConfig`` whose extra arguments are
            appended to the command line.
        repetitions: Number of benchmark repetitions (passed via ``-r``).
        timeout: Maximum time in seconds to wait for the subprocess.
        n_prompt: Number of prompt tokens (passed via ``-p``).
        n_gen: Number of generation tokens (passed via ``-n``).
        no_warmup: If True, pass ``--no-warmup`` to skip warmup runs.

    Returns:
        A ``BenchmarkResult`` with ``success`` set to ``True`` on
        success, or ``False`` with error details in ``raw_output`` on
        failure.
    """
    result = BenchmarkResult()

    bench_path = find_llama_bench()
    model_path = str(model_path)

    cmd = [bench_path, "-m", model_path, "-r", str(repetitions), "-o", "json",
           "-p", str(n_prompt), "-n", str(n_gen)]
    if no_warmup:
        cmd.append("--no-warmup")
    if config is not None:
        cmd.extend(config.to_bench_args())

    try:
        start = time.time()
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
        elapsed = time.time() - start
        result.startup_time = elapsed
        result.raw_output = proc.stdout

        if proc.returncode != 0:
            result.success = False
            result.raw_output += f"\nSTDERR: {proc.stderr}"
            return result

        parsed = _parse_benchmark_output(proc.stdout)
        if parsed:
            result.prompt_tps = parsed.get("prompt_tps", 0.0)
            result.generation_tps = parsed.get("generation_tps", 0.0)
            result.memory_usage = parsed.get("memory_usage", 0.0)
            result.success = True
        else:
            result.success = False

    except subprocess.TimeoutExpired:
        result.success = False
        result.raw_output = "[TIMEOUT]"
    except FileNotFoundError:
        result.success = False
        result.raw_output = f"[ERROR] llama-bench not found at: {bench_path}"
    except Exception as e:
        result.success = False
        result.raw_output = f"[ERROR] {e}"

    return result


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
