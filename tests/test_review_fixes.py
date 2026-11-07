"""Regression tests for issues found in review of the phase 1 changes."""

import json

import llama_autotune.benchmark as bench_mod
import llama_autotune.compat as compat_mod
from llama_autotune.compat import Capabilities, detect_capabilities
from llama_autotune.constraints import is_plausible, plan_memory
from llama_autotune.hardware import detect_hardware
from llama_autotune.models import Backend, HardwareInfo, KVLayout, ModelInfo, OptimizeObjective, SearchConfig
from llama_autotune.search_space import get_search_space

MIB = 1024 * 1024


def _model() -> ModelInfo:
    return ModelInfo(architecture="llama", bytes_per_token=1000 * MIB, tensor_bytes=1000 * MIB,
                     vocab_size=128_256, n_embd=2048, n_layers=16, tensor_types={"Q4_K": 600 * MIB},
                     kv_layout=KVLayout(k_elems_full=8192, v_elems_full=8192))


def test_hybrid_cpu_searches_performance_core_hyperthreads():
    hw = HardwareInfo(physical_cores=12, logical_cores=20, performance_cores=8, efficiency_cores=4,
                      performance_threads=16, backend=Backend.CPU)
    threads = get_search_space(hw, _model(), OptimizeObjective.BALANCED)["threads"]
    assert (threads.low, threads.high) == (7, 16)


def test_gate_keeps_the_heuristic_ubatch():
    d = plan_memory(_model(), 10_000 * MIB, ubatch=256, compute_override=258 * MIB)
    assert d.fits and d.ubatch == 256 and d.savings == []
    tight = plan_memory(_model(), 1_000 * MIB, ubatch=256, compute_override=258 * MIB)
    assert not any(s.startswith("ubatch 256") for s in tight.savings)


def test_optimizer_check_accepts_what_the_gate_accepted():
    hw = HardwareInfo(logical_cores=4, physical_cores=2, isa_tier="avx2", backend=Backend.CPU)
    kw = {"tensor_sizes": None, "compute_override": 258 * MIB}
    d = plan_memory(_model(), 1_400 * MIB, ubatch=256, **kw)
    assert d.fits
    cfg = d.apply(SearchConfig(threads=2, batch_size=512, ubatch_size=256))
    assert is_plausible(cfg, _model(), hw, budget=d.budget, **kw)
    bigger = cfg.model_copy(update={"repack": True})
    assert not is_plausible(bigger, _model(), hw, budget=d.budget, **kw)


def test_benchmark_not_started_when_ram_is_already_low(monkeypatch):
    started = []
    monkeypatch.setattr(bench_mod.subprocess, "Popen", lambda *a, **k: started.append(1))
    result = bench_mod.run_benchmark("m.gguf", SearchConfig(threads=2), min_available_mib=10**9)
    assert result.error == "memory_guard_external" and started == []


def test_benchmark_os_error_is_a_failed_trial_not_a_crash(monkeypatch):
    def deny(*a, **k):
        raise PermissionError(5, "Access is denied")
    monkeypatch.setattr(bench_mod.subprocess, "Popen", deny)
    result = bench_mod.run_benchmark("m.gguf", SearchConfig(threads=2), min_available_mib=0)
    assert not result.success and result.error == "failed"


def test_empty_device_list_means_cpu_even_with_an_os_gpu():
    hw = detect_hardware(Capabilities())
    assert hw.backend == Backend.CPU and hw.gpu_count == 0


def test_failed_help_probe_is_not_cached(monkeypatch, tmp_path):
    cache = tmp_path / "compat.json"
    monkeypatch.setattr(compat_mod, "_cache_path", lambda paths: cache)
    outputs = {"--version": "version: 1.0 (build 42, commit abc)", "--help": "", "--list-devices": ""}
    monkeypatch.setattr(compat_mod, "_run", lambda cmd, timeout=30: outputs[cmd[-1]])
    caps = detect_capabilities()
    assert caps.build.build_number == 42 and caps.server is None
    assert not cache.exists()


def test_malformed_cache_triggers_a_fresh_probe(monkeypatch, tmp_path):
    cache = tmp_path / "compat.json"
    cache.write_text(json.dumps({"build": {"version": "x", "build_number": 1, "commit": "c"}, "server": None}))
    monkeypatch.setattr(compat_mod, "_cache_path", lambda paths: cache)
    help_text = "-t,    --threads N      number of threads\n"
    outputs = {"--version": "version: 2.0 (build 7, commit def)", "--help": help_text, "--list-devices": ""}
    monkeypatch.setattr(compat_mod, "_run", lambda cmd, timeout=30: outputs[cmd[-1]])
    caps = detect_capabilities()
    assert caps.build.build_number == 7 and caps.server.has("-t")
