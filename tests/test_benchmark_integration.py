import os
import sys

from llama_autotune.benchmark import find_llama_bench, run_benchmark
from llama_autotune.models import SearchConfig


def test_find_llama_bench():
    path = find_llama_bench()
    assert path != ""
    assert os.path.isfile(path)
    assert path.endswith(".exe") or path.endswith(".py")


def test_run_benchmark_list_devices():
    model = os.path.join(
        os.path.dirname(__file__), "..", "..", "llamacpp",
        "models", "qwen", "Qwen2.5-0.5B-OBLITERATED.IQ4_XS.gguf",
    )
    if not os.path.isfile(model):
        model = os.path.join(
            os.path.dirname(__file__), "..", "..",
            "models", "qwen", "Qwen2.5-0.5B-OBLITERATED.IQ4_XS.gguf",
        )
    if not os.path.isfile(model):
        return

    cfg = SearchConfig(threads=2, n_gpu_layers=0, flash_attn=False)
    result = run_benchmark(model, cfg, repetitions=1, timeout=120)
    print(f"Prompt TPS: {result.prompt_tps}")
    print(f"Generation TPS: {result.generation_tps}")
    print(f"Success: {result.success}")
    print(f"Raw: {result.raw_output[:200]}")
