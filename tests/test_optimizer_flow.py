"""Optimizer integration with the memory gate, compat layer and benchmark, on a simulated machine."""

import pytest

from llama_autotune.memory import ComputeModel

import llama_autotune.optimizer as opt_mod
from llama_autotune.compat import Capabilities, parse_help
from llama_autotune.models import Backend, BenchmarkResult, HardwareInfo, KVLayout, ModelInfo
from llama_autotune.resources import MemoryStatus

MIB = 1024 * 1024


@pytest.fixture
def simulated(monkeypatch, tmp_path):
    """Patch probes and llama-bench; record every benchmarked config."""
    calls = []
    bench_help = (tmp_path / "h.txt")
    bench_help.write_text("  -t, --threads <n>   (default: 2)\n  --repack <0|1>   (default: 1)\n"
                          "  -ub, --ubatch-size <n>   (default: 512)\n  -b, --batch-size <n>  (default: 2048)\n")
    caps = Capabilities(bench=parse_help(bench_help.read_text()))
    hw = HardwareInfo(cpu_name="Sim CPU", physical_cores=2, logical_cores=4, performance_cores=2,
                      ram_gb=8.0, ram_available_gb=3.0, backend=Backend.CPU, isa_tier="avx2")
    model = ModelInfo(path="m.gguf", architecture="llama", n_layers=16, n_heads=32, training_context=131072,
                      bytes_per_token=1300 * MIB, tensor_bytes=1300 * MIB, vocab_size=128256, n_embd=2048,
                      tensor_types={"Q4_K": 800 * MIB}, kv_layout=KVLayout(k_elems_full=8192, v_elems_full=8192))

    def fake_bench(model_path, config, **kwargs):
        calls.append((config.model_copy(), kwargs.get("registry")))
        threads = config.threads or 2
        return BenchmarkResult(success=True, generation_tps={1: 8.0, 2: 11.5, 3: 13.9, 4: 13.7}.get(threads, 5.0),
                               prompt_tps=60.0)

    monkeypatch.setattr(opt_mod, "detect_capabilities", lambda: caps)
    monkeypatch.setattr(opt_mod, "detect_hardware", lambda c=None: hw)
    monkeypatch.setattr(opt_mod, "inspect_model", lambda p: model)
    monkeypatch.setattr(opt_mod, "tensor_sizes_of", lambda p: None)
    monkeypatch.setattr(opt_mod, "compute_model_for",
                        lambda p, m, server_build=0: ComputeModel(258 * MIB, 128256, 2048, 32))
    monkeypatch.setattr(opt_mod, "run_benchmark", fake_bench)
    return calls, caps, monkeypatch


def test_gate_choices_reach_every_benchmark(simulated):
    calls, caps, mp = simulated
    mp.setattr(opt_mod, "memory_status", lambda: MemoryStatus(total=8 * 1024**3, available=2600 * MIB))
    opt = opt_mod.Optimizer("m.gguf", n_trials_stage_b=4, n_trials_stage_c=3)
    assert opt.gate.fits
    assert opt.gate.repack is False                 # 800 MiB repack copy did not fit the budget
    best = opt.run()
    assert calls, "nothing was benchmarked"
    assert all(cfg.repack is False for cfg, _ in calls)
    assert all(reg is caps.bench for _, reg in calls)
    assert best.threads == 3
    assert best.ctx_size == opt.gate.ctx            # context comes from the gate, never searched


def test_model_that_does_not_fit_is_never_benchmarked(simulated):
    calls, _, mp = simulated
    mp.setattr(opt_mod, "memory_status", lambda: MemoryStatus(total=8 * 1024**3, available=900 * MIB))
    opt = opt_mod.Optimizer("m.gguf", n_trials_stage_b=4, n_trials_stage_c=3)
    assert not opt.gate.fits
    opt.run()
    assert calls == []
