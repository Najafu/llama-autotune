import os
import sys

from llama_autotune.benchmark import (
    binary_name,
    find_llama_bench,
    find_llama_binary,
    run_benchmark,
)
from llama_autotune.models import SearchConfig


def test_binary_name_is_platform_aware():
    expected = "llama-bench.exe" if os.name == "nt" else "llama-bench"
    assert binary_name("llama-bench") == expected


def test_find_llama_bench():
    path = find_llama_bench()
    assert path != ""
    if not os.path.isfile(path):
        import pytest
        pytest.skip("llama-bench not installed on this machine")
    assert os.path.basename(path) == binary_name("llama-bench")


def test_find_llama_binary_prefers_env_dir(tmp_path, monkeypatch):
    exe = tmp_path / binary_name("llama-bench")
    exe.write_bytes(b"")
    monkeypatch.setenv("LLAMA_CPP_DIR", str(tmp_path))
    assert find_llama_binary("llama-bench") == str(exe)


def test_find_llama_binary_missing_returns_bare_name(monkeypatch):
    monkeypatch.delenv("LLAMA_CPP_DIR", raising=False)
    result = find_llama_binary("llama-definitely-not-a-tool")
    assert result == binary_name("llama-definitely-not-a-tool")


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
