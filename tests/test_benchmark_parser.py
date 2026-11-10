"""llama-bench output: one parser (JSONL records) and one summary for single runs."""

from llama_autotune.benchmark import parse_records, summarize


def test_prompt_and_generation_tests_are_told_apart():
    out = ('{"n_prompt": 64, "n_gen": 0, "avg_ts": 1000.0}\n'
           '{"n_prompt": 0, "n_gen": 32, "avg_ts": 50.0}')
    assert summarize(parse_records(out)) == (1000.0, 50.0)


def test_combined_test_of_older_builds_counts_as_generation():
    out = '{"n_prompt": 512, "n_gen": 128, "avg_ts": 42.5}'
    assert summarize(parse_records(out)) == (0.0, 42.5)


def test_prompt_only_run_has_no_generation_speed():
    out = '{"n_prompt": 512, "n_gen": 0, "avg_ts": 98.76}'
    assert summarize(parse_records(out)) == (98.76, 0.0)


def test_log_lines_around_the_results_are_ignored():
    out = ('ggml_init: something\n'
           '{"n_prompt": 0, "n_gen": 8, "avg_ts": 30.0}\n'
           'not json at all\n'
           '{"unrelated": true}')
    records = parse_records(out)
    assert len(records) == 1 and summarize(records) == (0.0, 30.0)


def test_no_results_means_no_records():
    assert parse_records("") == []
    assert parse_records("ggml_init: error loading model") == []
    assert summarize([]) == (0.0, 0.0)
