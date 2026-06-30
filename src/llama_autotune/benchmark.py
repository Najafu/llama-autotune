from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from .models import BenchmarkResult, SearchConfig


def find_llama_bench() -> str:
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
    result = BenchmarkResult()

    bench_path = find_llama_bench()
    model_path = str(model_path)

    cmd = [bench_path, "-m", model_path, "-r", str(repetitions), "-o", "json"]
    if config is not None:
        cmd.extend(config.to_llama_args())

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
