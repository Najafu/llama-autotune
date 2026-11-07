"""Tests for compat.py — parsing llama.cpp output and rendering flags per build."""

from pathlib import Path

import pytest

from llama_autotune.compat import (
    clean_env,
    llama_env_overrides,
    parse_cpu_variant,
    parse_devices,
    parse_help,
    parse_version,
    render_args,
)
from llama_autotune.models import LoadMode, SearchConfig, SplitMode

FIXTURES = Path(__file__).parent / "fixtures"

# A reduced help listing in the style of builds before July 2026, when
# mmap and mlock had their own flags and llama-bench used -mmp.
LEGACY_SERVER_HELP = """
-t,    --threads N                      number of threads to use during generation
-c,    --ctx-size N                     size of the prompt context
-fa,   --flash-attn [on|off|auto]       set Flash Attention use
--mlock                                 force system to keep model in RAM rather than swapping
--no-mmap                               do not memory-map model (slower load but may reduce pageouts)
-nkvo, --no-kv-offload                  disable KV offload
"""
LEGACY_BENCH_HELP = """
  -t, --threads <n>                         (default: 2)
  -fa, --flash-attn <0|1>                   (default: 0)
  -mmp, --mmap <0|1>                        (default: 1)
"""


def _read(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8", errors="replace")


@pytest.fixture(scope="module")
def server():
    return parse_help(_read("llama-server-b11347-help.txt"))


@pytest.fixture(scope="module")
def bench():
    return parse_help(_read("llama-bench-b11347-help.txt"))


def test_parse_help_counts_and_value_kinds(server, bench):
    assert len(server.specs) > 200
    assert server.get("-t").takes_value
    assert not server.get("-nkvo").takes_value            # bare switch in llama-server
    assert bench.get("-nkvo").choices == ["0", "1"]       # takes 0|1 in llama-bench
    assert server.get("-fa").choices == ["on", "off", "auto"]
    assert "mmap+mlock" in bench.get("-lm").choices


def test_parse_help_has_no_removed_load_flags(server):
    assert not server.has("--mlock")       # removed in b10875
    assert not server.has("--no-mmap")
    assert server.has("--fit") and server.has("-fitt")


def test_b11347_build_has_no_legacy_or_tb_bench_flags(bench):
    assert not bench.has("-mmp")
    assert not bench.has("-tb")
    assert bench.has("--repack")


def test_parse_version_and_cpu_variant():
    text = _read("llama-bench-b11347-version.txt")
    build = parse_version(text)
    assert (build.version, build.build_number, build.commit) == ("0.5.0-dev", 11347, "5fc4f3c8c")
    assert parse_cpu_variant(text) == ("haswell", "avx2")


@pytest.mark.parametrize("dll,expected", [
    ("C:\\llama\\ggml-cpu-sandybridge.dll", ("sandybridge", "avx")),
    ("/usr/lib/libggml-cpu-zen4.so", ("zen4", "avx512")),
    ("C:\\llama\\ggml-cpu-x64.dll", ("x64", "no-avx")),
    ("/opt/libggml-cpu-armv8.2_2.so", ("armv8.2_2", "arm")),
])
def test_parse_cpu_variant_tiers(dll, expected):
    assert parse_cpu_variant(f"load_backend: loaded CPU backend from {dll}") == expected


def test_parse_devices():
    assert parse_devices(_read("llama-bench-b11347-list-devices.txt")) == []
    text = "Available devices:\n  CUDA0: NVIDIA GeForce RTX 4060 (8188 MiB, 7300 MiB free)\n"
    (dev,) = parse_devices(text)
    assert (dev.name, dev.total_mib, dev.free_mib) == ("CUDA0", 8188, 7300)


def test_render_server_current_build(server):
    cfg = SearchConfig(threads=3, threads_batch=4, ctx_size=4096, flash_attn=False,
                       load_mode=LoadMode.MLOCK, repack=False, no_kv_offload=True,
                       cache_ram_mib=0, parallel=1, no_mmproj=True)
    result = render_args(cfg, "server", server)
    assert result.skipped == []
    assert result.args == ["-t", "3", "-tb", "4", "-c", "4096", "-fa", "off", "-lm", "mlock",
                           "--no-repack", "-np", "1", "-nkvo", "-cram", "0", "--no-mmproj"]


def test_render_bench_current_build(bench):
    cfg = SearchConfig(threads=3, threads_batch=4, ctx_size=4096, flash_attn=True, repack=False,
                       no_kv_offload=False, tensor_split="3,1", split_mode=SplitMode.LAYER)
    args = render_args(cfg, "bench", bench).args
    assert args == ["-t", "3", "-fa", "on", "-ts", "3/1", "-sm", "layer", "--repack", "0", "-nkvo", "0"]
    assert "-c" not in args and "-tb" not in args


def test_render_falls_back_to_legacy_spellings():
    server, bench = parse_help(LEGACY_SERVER_HELP), parse_help(LEGACY_BENCH_HELP)
    cfg = SearchConfig(threads=2, load_mode=LoadMode.MLOCK, no_kv_offload=True, flash_attn=True)
    assert render_args(cfg, "server", server).args == ["-t", "2", "-fa", "on", "--mlock", "-nkvo"]
    bench_cfg = SearchConfig(threads=2, load_mode=LoadMode.NONE, flash_attn=True)
    assert render_args(bench_cfg, "bench", bench).args == ["-t", "2", "-fa", "1", "-mmp", "0"]


def test_render_never_emits_unknown_flags():
    server = parse_help(LEGACY_SERVER_HELP)
    cfg = SearchConfig(threads=2, repack=False, cache_ram_mib=0, load_mode=LoadMode.DIRECT_IO)
    result = render_args(cfg, "server", server)
    assert result.args == ["-t", "2"]
    assert len(result.skipped) == 3
    assert all("no matching flag" in s for s in result.skipped)


def test_render_without_registry_uses_current_spelling():
    args = SearchConfig(load_mode=LoadMode.NONE, flash_attn=False, no_kv_offload=True).to_llama_args()
    assert args == ["-fa", "off", "-lm", "none", "-nkvo"]


def test_clean_env_removes_llama_arg_variables(monkeypatch):
    monkeypatch.setenv("LLAMA_ARG_CTX_SIZE", "131072")
    assert llama_env_overrides()["LLAMA_ARG_CTX_SIZE"] == "131072"
    assert "LLAMA_ARG_CTX_SIZE" not in clean_env()
