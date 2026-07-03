"""Tests for Optimizer._score — objective-specific scoring."""

from llama_autotune.models import BenchmarkResult, OptimizeObjective, SearchConfig
from llama_autotune.optimizer import Optimizer


def _make_optimizer(objective: OptimizeObjective) -> Optimizer:
    """Build an Optimizer without hardware detection or model inspection."""
    opt = Optimizer.__new__(Optimizer)
    opt.objective = objective
    return opt


def _result(gen=10.0, prompt=50.0, startup=5.0, memory=2000.0) -> BenchmarkResult:
    return BenchmarkResult(
        generation_tps=gen,
        prompt_tps=prompt,
        startup_time=startup,
        memory_usage=memory,
        success=True,
    )


def test_score_max_generation_tps():
    opt = _make_optimizer(OptimizeObjective.MAX_GENERATION_TPS)
    assert opt._score(_result(gen=42.0), SearchConfig()) == 42.0


def test_score_max_prompt_tps():
    opt = _make_optimizer(OptimizeObjective.MAX_PROMPT_TPS)
    assert opt._score(_result(prompt=123.0), SearchConfig()) == 123.0


def test_score_min_latency_is_negative_startup():
    opt = _make_optimizer(OptimizeObjective.MIN_LATENCY)
    assert opt._score(_result(startup=5.0), SearchConfig()) == -5.0


def test_score_min_latency_faster_beats_slower():
    """A 2s startup must outscore a 5s startup (higher is better)."""
    opt = _make_optimizer(OptimizeObjective.MIN_LATENCY)
    fast = opt._score(_result(startup=2.0), SearchConfig())
    slow = opt._score(_result(startup=5.0), SearchConfig())
    assert fast > slow


def test_score_min_latency_real_config_beats_failure_sentinel():
    """Any working config must outrank the old -1.0 failure sentinel.

    Regression test: stage C previously returned -1.0 for failed configs,
    which outranked legitimate min_latency scores like -5.0.
    """
    opt = _make_optimizer(OptimizeObjective.MIN_LATENCY)
    score = opt._score(_result(startup=0.5), SearchConfig())
    assert score > -1.0


def test_score_max_context_scales_with_ctx():
    opt = _make_optimizer(OptimizeObjective.MAX_CONTEXT)
    small = opt._score(_result(gen=100.0), SearchConfig(ctx_size=4096))
    large = opt._score(_result(gen=1.0), SearchConfig(ctx_size=32768))
    assert large > small


def test_score_max_context_gen_tps_breaks_ties():
    opt = _make_optimizer(OptimizeObjective.MAX_CONTEXT)
    slower = opt._score(_result(gen=5.0), SearchConfig(ctx_size=8192))
    faster = opt._score(_result(gen=10.0), SearchConfig(ctx_size=8192))
    assert faster > slower


def test_score_max_context_none_ctx():
    opt = _make_optimizer(OptimizeObjective.MAX_CONTEXT)
    score = opt._score(_result(gen=10.0), SearchConfig())
    assert score == 10.0 / 1000.0


def test_score_max_efficiency():
    opt = _make_optimizer(OptimizeObjective.MAX_EFFICIENCY)
    assert opt._score(_result(gen=10.0, memory=2000.0), SearchConfig()) == 10.0 / 2000.0


def test_score_balanced_defaults_to_generation():
    opt = _make_optimizer(OptimizeObjective.BALANCED)
    assert opt._score(_result(gen=7.0), SearchConfig()) == 7.0
