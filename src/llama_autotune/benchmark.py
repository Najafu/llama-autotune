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

    Returns:
        A ``BenchmarkResult`` with ``success`` set to ``True`` on
        success, or ``False`` with error details in ``raw_output`` on
        failure.
    """
    result = BenchmarkResult()

    bench_path = find_llama_bench()
    model_path = str(model_path)

    cmd = [bench_path, "-m", model_path, "-r", str(repetitions), "-o", "json"]
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

    Handles both a single JSON object and a JSON array (uses the last
    entry).  Falls back to line-by-line JSONL parsing if top-level
    parsing fails.

    Args:
        output: The raw stdout text from llama-bench.

    Returns:
        A dictionary with keys ``prompt_tps``, ``generation_tps``, and
        ``memory_usage``, or ``None`` if no valid data could be
        extracted.
    """
    try:
        data = json.loads(output)
        if isinstance(data, list) and len(data) > 0:
            entry = data[-1]
        elif isinstance(data, dict):
            entry = data
        else:
            return None

        return {
            "prompt_tps": float(entry.get("pp_avg", entry.get("prompt_tps", 0))),
            "generation_tps": float(entry.get("tg_avg", entry.get("generation_tps", 0))),
            "memory_usage": float(entry.get("mem_usage", entry.get("memory_usage", 0))),
        }
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    last_result = None
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
            if isinstance(data, dict):
                last_result = {
                    "prompt_tps": float(data.get("pp_avg", data.get("prompt_tps", 0))),
                    "generation_tps": float(data.get("tg_avg", data.get("generation_tps", 0))),
                    "memory_usage": float(data.get("mem_usage", data.get("memory_usage", 0))),
                }
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    return last_result
