"""Tests for model_inspector.py — GGUF header parsing and MoE detection."""

from llama_autotune.model_inspector import (
    _resolve_param_count,
    _resolve_active_param_count,
    _resolve_file_type,
    _read_gguf_header,
    inspect_model,
)


def _kv(**kwargs) -> dict:
    """Build a minimal GGUF key-value store."""
    return {
        "general.architecture": "test",
        "test.block_count": 10,
        "test.attention.head_count": 8,
        "test.context_length": 4096,
        **kwargs,
    }


# ── _resolve_param_count ─────────────────────────────────────────────


def test_resolve_param_count_uses_exact():
    kv = _kv(**{"general.parameter_count": 7_000_000_000})
    assert _resolve_param_count(kv) == 7_000_000_000


def test_resolve_param_count_label_simple():
    kv = _kv(**{"general.size_label": "7B"})
    assert _resolve_param_count(kv) == 7_000_000_000


def test_resolve_param_count_label_hyphenated():
    """'1B-7B' should return the total (second value)."""
    kv = _kv(**{"general.size_label": "1B-7B"})
    assert _resolve_param_count(kv) == 7_000_000_000


def test_resolve_param_count_label_millions():
    kv = _kv(**{"general.size_label": "70M"})
    assert _resolve_param_count(kv) == 70_000_000


def test_resolve_param_count_label_trillions():
    kv = _kv(**{"general.size_label": "1.5T"})
    assert _resolve_param_count(kv) == 1_500_000_000_000


def test_resolve_param_count_label_unknown():
    kv = _kv()
    assert _resolve_param_count(kv) == 0


def test_resolve_param_count_empty_label():
    kv = _kv(**{"general.size_label": ""})
    assert _resolve_param_count(kv) == 0


# ── _resolve_active_param_count ──────────────────────────────────────


def test_resolve_active_count_from_label():
    """Should extract active params from '1B-7B' label."""
    kv = _kv(**{"general.size_label": "1B-7B"})
    assert _resolve_active_param_count(kv, 7_000_000_000, 64, 8) == 1_000_000_000


def test_resolve_active_count_via_ratio():
    """Fall back to expert ratio when no label available."""
    kv = _kv()
    assert _resolve_active_param_count(kv, 7_000_000_000, 64, 8) == 875_000_000


def test_resolve_active_count_zero_total():
    kv = _kv()
    assert _resolve_active_param_count(kv, 0, 64, 8) == 0


def test_resolve_active_count_no_moe():
    kv = _kv()
    assert _resolve_active_param_count(kv, 7_000_000_000, 0, 0) == 0


def test_resolve_active_count_label_millions():
    kv = _kv(**{"general.size_label": "500M-3B"})
    assert _resolve_active_param_count(kv, 3_000_000_000, 8, 2) == 500_000_000


# ── _resolve_file_type ────────────────────────────────────────────────


def test_file_type_from_filename():
    kv = _kv()
    result = _resolve_file_type(kv, model_path="/models/foo-Q4_K_M.gguf")
    assert result == "Q4_K_M"


def test_file_type_from_general_name():
    kv = _kv(**{"general.name": "MyModel Q2_K"})
    result = _resolve_file_type(kv, model_path="")
    assert result == "Q2_K"


def test_file_type_from_known_file_type():
    kv = _kv(**{"general.file_type": 2})  # Q4_0
    result = _resolve_file_type(kv, model_path="")
    assert result == "Q4_0"


def test_file_type_unknown():
    kv = _kv()
    result = _resolve_file_type(kv, model_path="")
    assert result == "unknown"


# ── inspect_model ────────────────────────────────────────────────────


import os

import pytest

OLMOE_PATH = r"C:\llamacpp\models\olmoe\OLMoE-1B-7B-0125-Instruct.Q2_K.gguf"
DENSE_PATH = r"C:\sandbox\llama\models\qwen\Qwen2.5-3B-Instruct-Q4_K_M.gguf"


@pytest.mark.skipif(not os.path.exists(OLMOE_PATH), reason="OLMoE model not present on this machine")
def test_inspect_olmoe():
    """Real OLMoE model must show correct MoE metadata."""
    info = inspect_model(OLMOE_PATH)
    assert info.architecture == "olmoe"
    assert info.is_moe is True
    assert info.parameters == 7_000_000_000
    assert info.active_parameters == 1_000_000_000
    assert info.quantization == "Q2_K"
    assert info.n_layers == 16
    assert info.n_heads == 16
    assert info.training_context == 4096


@pytest.mark.skipif(not os.path.exists(DENSE_PATH), reason="Qwen dense model not present on this machine")
def test_inspect_dense_model():
    """Real dense Qwen model must keep showing MoE=False."""
    info = inspect_model(DENSE_PATH)
    assert info.architecture == "qwen2"
    assert info.is_moe is False
    assert info.active_parameters == info.parameters
    assert info.parameters == 3_000_000_000
    assert info.quantization == "Q4_K_M"


def test_inspect_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        inspect_model(r"C:\nonexistent\model.gguf")
