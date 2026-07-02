from llama_autotune.benchmark import _parse_benchmark_output


def test_parse_combined_avg_ts():
    """Single combined entry (no --no-warmup) — avg_ts is gen_tps."""
    sample = '[{"n_prompt": 512, "n_gen": 0, "avg_ns": 123456, "avg_ts": 98.76, "mem_usage": 2048.0}]'
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 0.0
    assert result["generation_tps"] == 98.76
    assert result["memory_usage"] == 2048.0


def test_parse_split_avg_ts():
    """Split entries with --no-warmup — prompt entry + gen entry."""
    sample = (
        '[{"n_prompt": 16, "n_gen": 0, "avg_ts": 5123.45},'
        ' {"n_prompt": 0, "n_gen": 8, "avg_ts": 98.76}]'
    )
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 5123.45
    assert result["generation_tps"] == 98.76


def test_parse_jsonl_avg_ts():
    """JSONL format with split entries."""
    sample = (
        '{"n_prompt": 64, "n_gen": 0, "avg_ts": 1000.0}\n'
        '{"n_prompt": 0, "n_gen": 32, "avg_ts": 50.0}'
    )
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 1000.0
    assert result["generation_tps"] == 50.0


def test_parse_legacy_pp_tg():
    """Legacy format with pp_avg/tg_avg still works."""
    sample = '{"pp_avg": 5000, "tg_avg": 150, "mem_usage": 1024}'
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 5000
    assert result["generation_tps"] == 150
    assert result["memory_usage"] == 1024


def test_parse_legacy_array():
    """Legacy array — last entry wins."""
    sample = (
        '[{"model": "a", "pp_avg": 100, "tg_avg": 10}, '
        '{"model": "b", "pp_avg": 200, "tg_avg": 20}]'
    )
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 200
    assert result["generation_tps"] == 20


def test_parse_legacy_json_lines():
    sample = (
        '{"pp_avg": 1000.0, "tg_avg": 50.0}\n'
        '{"pp_avg": 2000.0, "tg_avg": 100.0}'
    )
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 2000.0
    assert result["generation_tps"] == 100.0


def test_parse_empty():
    assert _parse_benchmark_output("") is None


def test_parse_garbage():
    assert _parse_benchmark_output("not json at all") is None


def test_parse_single_object():
    sample = '{"pp_avg": 5000, "tg_avg": 150, "mem_usage": 1024}'
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 5000
    assert result["generation_tps"] == 150


def test_parse_combined_object_avg_ts():
    """Single dict (not array) with avg_ts."""
    sample = '{"n_prompt": 64, "n_gen": 0, "avg_ts": 42.5, "mem_usage": 512.0}'
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["generation_tps"] == 42.5
    assert result["memory_usage"] == 512.0


def test_parse_no_json():
    """Non-JSON output (e.g. error message) returns None."""
    assert _parse_benchmark_output("ggml_init: error loading model") is None


def test_parse_memory_from_avg_ts_entry():
    """Memory parsed from gen entry when present."""
    sample = (
        '[{"n_prompt": 16, "n_gen": 0, "avg_ts": 500.0},'
        ' {"n_prompt": 0, "n_gen": 8, "avg_ts": 30.0, "mem_usage": 4096.0}]'
    )
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 500.0
    assert result["generation_tps"] == 30.0
    assert result["memory_usage"] == 4096.0
