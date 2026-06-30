from llama_autotune.benchmark import _parse_benchmark_output


def test_parse_json_array():
    sample = (
        '[{"model": "test", "pp_avg": 5123.45, "tg_avg": 98.76, "mem_usage": 2048.0}]'
    )
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 5123.45
    assert result["generation_tps"] == 98.76
    assert result["memory_usage"] == 2048.0


def test_parse_last_entry():
    sample = (
        '[{"model": "a", "pp_avg": 100, "tg_avg": 10}, '
        '{"model": "b", "pp_avg": 200, "tg_avg": 20}]'
    )
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 200


def test_parse_json_lines():
    sample = (
        '{"pp_avg": 1000.0, "tg_avg": 50.0}\n'
        '{"pp_avg": 2000.0, "tg_avg": 100.0}'
    )
    result = _parse_benchmark_output(sample)
    assert result is not None
    assert result["prompt_tps"] == 2000.0


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
