"""Tests for memory.py, the KV layout in model_inspector.py, and the memory gate."""

import os

import pytest

from llama_autotune.constraints import detect_oom_in_output, memory_budget, plan_memory, safety_margin
from llama_autotune.memory import (
    MIB,
    compute_buffer_bytes,
    estimate_memory,
    kv_cache_bytes,
    repack_copy_bytes,
    tensor_sizes_of,
)
from llama_autotune.model_inspector import inspect_model, kv_layout
from llama_autotune.models import KVLayout, ModelInfo

MODELS = r"C:\sandbox\llama\models"


def _llama_like_kv(**extra) -> dict:
    return {"general.architecture": "llama", "llama.block_count": 16, "llama.embedding_length": 2048,
            "llama.attention.head_count": 32, "llama.attention.head_count_kv": 8,
            "llama.attention.key_length": 64, "llama.attention.value_length": 64, **extra}


def test_kv_layout_gqa():
    layout = kv_layout(_llama_like_kv(), "llama")
    assert (layout.k_elems_full, layout.v_elems_full) == (16 * 8 * 64, 16 * 8 * 64)
    assert kv_cache_bytes(layout, 4096) == 128 * MIB          # Llama 3.2 1B, matches llama.cpp


def test_kv_layout_sliding_window_and_shared_layers():
    kv = {"general.architecture": "g", "g.block_count": 10, "g.embedding_length": 1536,
          "g.attention.head_count": 8, "g.attention.head_count_kv": 1,
          "g.attention.key_length": 512, "g.attention.value_length": 512,
          "g.attention.key_length_swa": 256, "g.attention.value_length_swa": 256,
          "g.attention.sliding_window": 512, "g.attention.shared_kv_layers": 4,
          "g.attention.sliding_window_pattern": [True, True, False] * 3 + [True]}
    layout = kv_layout(kv, "g")
    # first 6 layers own a cache: 4 sliding-window, 2 full
    assert layout.k_elems_full == 2 * 512 and layout.k_elems_swa == 4 * 256
    assert layout.swa_window == 512
    small, large = kv_cache_bytes(layout, 4096), kv_cache_bytes(layout, 32768)
    swa_part = 1024 * 4 * 512 * 2                            # window + ubatch cells
    assert small == 4096 * 2 * 1024 * 2 + swa_part
    assert large - small == (32768 - 4096) * 2 * 1024 * 2    # only full layers grow


def test_kv_layout_hybrid_recurrent():
    kv = {"general.architecture": "q", "q.block_count": 32, "q.embedding_length": 2560,
          "q.attention.head_count": 16, "q.attention.head_count_kv": 4,
          "q.attention.key_length": 256, "q.attention.value_length": 256,
          "q.full_attention_interval": 4, "q.ssm.inner_size": 4096, "q.ssm.state_size": 128,
          "q.ssm.conv_kernel": 4, "q.ssm.group_count": 16}
    layout = kv_layout(kv, "q")
    assert layout.k_elems_full == 8 * 4 * 256                 # every 4th layer has attention
    assert round(layout.recurrent_bytes / MIB, 2) == 50.25    # Qwen3.5 4B, llama.cpp log
    assert kv_cache_bytes(layout, 4096) == 128 * MIB


def test_kv_layout_short_convolution():
    head_kv = [4 if il % 2 == 0 and 2 <= il <= 22 else 0 for il in range(24)]   # 11 attention layers
    kv = {"general.architecture": "lfm3", "lfm3.block_count": 24, "lfm3.embedding_length": 2048,
          "lfm3.attention.head_count": 16, "lfm3.attention.head_count_kv": head_kv,
          "lfm3.shortconv.l_cache": 7}
    layout = kv_layout(kv, "lfm3")
    assert layout.recurrent_bytes == 13 * 4 * 6 * 2048                  # 0.61 MiB, llama.cpp b12228 log


LFM3 = os.path.join(MODELS, "uncensored", "lfm", "LFM3-1.7B-Uncensored-AD-Q4_K_M.gguf")


@pytest.mark.skipif(not os.path.isfile(LFM3), reason="LFM3 model not present on this machine")
def test_lfm3_memory_matches_llama_cpp():
    m = inspect_model(LFM3)
    # llama-server b12228, -c 4096: KV 48 MiB full + 7.5 MiB sliding window, RS 0.61 MiB
    assert kv_cache_bytes(m.kv_layout, 4096) == int(55.5 * MIB)
    assert round(m.kv_layout.recurrent_bytes / MIB, 2) == 0.61
    assert round(repack_copy_bytes(m, "avx2", tensor_sizes_of(LFM3)) / MIB) == 496


def test_kv_cache_quantized_types():
    layout = KVLayout(k_elems_full=1024, v_elems_full=1024)
    f16 = kv_cache_bytes(layout, 4096)
    q8 = kv_cache_bytes(layout, 4096, "q8_0", "q8_0")
    assert q8 == int(f16 * 34 / 64)


def _model(**kw) -> ModelInfo:
    base = dict(architecture="llama", bytes_per_token=1000 * MIB, tensor_bytes=1000 * MIB,
                vocab_size=128_256, n_embd=2048, tensor_types={"Q4_K": 600 * MIB, "Q6_K": 400 * MIB},
                kv_layout=KVLayout(k_elems_full=16 * 8 * 64, v_elems_full=16 * 8 * 64))
    base.update(kw)
    return ModelInfo(**base)


def test_repack_copy_depends_on_isa_tier():
    m = _model()
    assert repack_copy_bytes(m, "avx2") == 600 * MIB
    assert repack_copy_bytes(m, "arm") == 1000 * MIB
    assert repack_copy_bytes(m, "avx") == 0
    assert repack_copy_bytes(m, "no-avx") == 0
    assert repack_copy_bytes(m, "unknown") == 600 * MIB     # self-built: assume the worst


def test_repack_copy_excludes_embedding_tables():
    sizes = {"token_embd.weight": ("Q4_K", 100), "blk.0.ffn_up.weight": ("Q4_K", 50),
             "blk.0.attn_q.weight": ("Q6_K", 30)}
    assert repack_copy_bytes(_model(), "avx2", sizes) == 50


def test_compute_buffer_scales_with_ubatch():
    m = _model()
    assert compute_buffer_bytes(m, 128) < compute_buffer_bytes(m, 512)
    assert compute_buffer_bytes(m, 512) >= 258 * MIB          # llama.cpp reported 258 MiB


def test_estimate_memory_parts():
    est = estimate_memory(_model(), 4096, repack=True, isa_tier="avx2", compute_override=258 * MIB)
    assert est.weights == 1000 * MIB
    assert est.repack_copy == 600 * MIB
    assert est.kv_cache == 128 * MIB
    assert est.runtime == 128_256 * 384                     # tokenizer and the like: 47 MiB
    assert est.total_mib == 1000 + 600 + 128 + 258 + 47
    assert estimate_memory(_model(), 4096, repack=False).repack_copy == 0
    assert estimate_memory(_model(vocab_size=0), 4096).runtime == 32 * MIB


def test_safety_margin_and_budget():
    gib = 1024**3
    assert safety_margin(8 * gib) == int(0.8 * gib)
    assert safety_margin(2 * gib) == 512 * MIB
    assert memory_budget(3 * gib, 8 * gib) == 3 * gib - int(0.8 * gib)
    assert memory_budget(100, 8 * gib) == 0


def test_gate_fits_without_savings():
    d = plan_memory(_model(), 4000 * MIB, compute_override=258 * MIB)
    assert d.fits and d.savings == [] and d.ctx == 4096 and d.repack


def test_gate_applies_savings_in_order_of_least_harm():
    need_default = estimate_memory(_model(), 4096, compute_override=258 * MIB).total
    d = plan_memory(_model(), need_default - 100 * MIB, compute_override=258 * MIB)
    assert d.fits and d.ubatch == 256 and d.repack            # halving ubatch saves 129 MiB
    d = plan_memory(_model(), need_default - 500 * MIB, compute_override=258 * MIB)
    assert d.fits and d.ubatch == 128 and not d.repack        # then repacking off
    assert d.savings[0].startswith("ubatch") and "repacking" in d.savings[-1]


def test_gate_reduces_context_last_and_reports_shortfall():
    d = plan_memory(_model(), 1150 * MIB, ctx=8192, compute_override=258 * MIB)
    assert d.fits and d.ctx < 8192 and d.cache_type == "q8_0"
    d = plan_memory(_model(), 300 * MIB, compute_override=258 * MIB)
    assert not d.fits and d.ctx == 2048 and d.shortfall > 0


def test_gate_apply_sets_flash_attention_for_quantized_cache():
    from llama_autotune.models import SearchConfig

    d = plan_memory(_model(), 1150 * MIB, ctx=8192, compute_override=258 * MIB)
    cfg = d.apply(SearchConfig(batch_size=2048))
    assert cfg.cache_type_v == "q8_0" and cfg.flash_attn is True and cfg.ctx_size == d.ctx


def test_oom_detection_has_no_false_positives():
    assert detect_oom_in_output("ggml_backend_cpu_buffer_type_alloc_buffer: failed to allocate buffer")
    assert detect_oom_in_output("CUDA error: out of memory")
    assert not detect_oom_in_output("plenty of room in the context")


# ── checks against real models, values measured with llama.cpp b11347 ─────


REAL = [
    # file, KV MiB at 4096 (f16), repack copy MiB on AVX2 (llama.cpp omits it)
    (r"lama\Llama-3.2-1B-Instruct-GPTQ-Q6_K.gguf", 128, 0),
    (r"gemma4\gemma-4-E2B_q4_0-it.gguf", 36, 999),
    (r"uncensored\qwen\Qwen3.5-4B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf", 178, 1435),
    (r"krikri\llama-krikri-8b-instruct-q4_k_m.gguf", 512, 3204),
]


@pytest.mark.parametrize("rel,kv_mib,repack_mib", REAL)
def test_real_models_match_llama_cpp(rel, kv_mib, repack_mib):
    path = os.path.join(MODELS, rel)
    if not os.path.exists(path):
        pytest.skip("model not present on this machine")
    m = inspect_model(path)
    layout = m.kv_layout
    assert round((kv_cache_bytes(layout, 4096) + layout.recurrent_bytes) / MIB) == kv_mib
    assert round(repack_copy_bytes(m, "avx2", tensor_sizes_of(path)) / MIB) == repack_mib


# ── compute buffer model (llama.cpp b12228, experiment E1b) ──────────

def _cm(base_mib, vocab, n_embd, heads, server=False):
    from llama_autotune.memory import ComputeModel
    return ComputeModel(base_mib * MIB, vocab, n_embd, heads, single_output=server)


# (base at 4K fa on, vocab, n_embd, heads, ctx, ubatch, fa, llama-bench MiB, llama-server MiB)
MEASURED = [
    (274, 128256, 2048, 32, 4096, 512, True, 274, 72.02),    # Llama 3.2 1B
    (274, 128256, 2048, 32, 4096, 512, False, 290, 290.01),
    (274, 128256, 2048, 32, 16384, 512, False, 1082, 1082.01),
    (274, 128256, 2048, 32, 16384, 512, True, 290, 84.02),
    (274, 128256, 2048, 32, 4096, 256, False, 145, 145.01),
    (274, 128000, 2048, 16, 4096, 512, False, 283, 165.52),  # LFM3 1.7B
    (513, 248320, 2048, 8, 16384, 512, False, 539, 318.02),  # Qwen3.5 2B
    (566, 262144, 1536, 8, 4096, 512, True, 566, 84.53),     # Gemma 4 E2B
    (514, 248320, 2560, 16, 4096, 512, False, 527, 189.02),  # Qwen3.5 4B
]


@pytest.mark.parametrize("base,vocab,embd,heads,ctx,ub,fa,bench,server", MEASURED)
def test_compute_model_never_below_llama_cpp(base, vocab, embd, heads, ctx, ub, fa, bench, server):
    b = _cm(base, vocab, embd, heads).bytes(ctx, ub, fa) / MIB
    s = _cm(base, vocab, embd, heads, server=True).bytes(ctx, ub, fa) / MIB
    assert bench <= b <= bench + 70
    assert server <= s <= server + 60


def test_gate_counts_attention_scores_and_turns_flash_attention_on():
    """Llama 3.2 1B, 16K context, flash attention off: the scores alone are 1 GiB."""
    cm = _cm(274, 128256, 2048, 32, server=True)
    model = _model(kv_layout=KVLayout(k_elems_full=16 * 8 * 64, v_elems_full=16 * 8 * 64), tensor_types={})
    need_off = plan_memory(model, 10**12, ctx=16384, compute_model=cm, flash_attn=False).estimate.total
    need_on = plan_memory(model, 10**12, ctx=16384, compute_model=cm, flash_attn=True).estimate.total
    assert need_off - need_on >= 1000 * MIB                  # 1,024 MiB of scores, minus the mask term
    gate = plan_memory(model, need_on + 50 * MIB, ctx=16384, min_ctx=16384, compute_model=cm, flash_attn=False)
    assert gate.fits and gate.flash_attn is True and gate.ctx == 16384
    assert any("flash attention on" in s for s in gate.savings)
    from llama_autotune.models import SearchConfig
    assert gate.apply(SearchConfig(flash_attn=False)).flash_attn is True
