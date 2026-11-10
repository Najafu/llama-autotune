"""Tests for the search optimizer's scoring, grid and final confirmation."""

import math

import pytest

from llama_autotune.models import BenchmarkResult, OptimizeObjective, SearchConfig
from llama_autotune.optimizer import Optimizer
from llama_autotune.search_space import ParamDef


def _make_optimizer(objective: OptimizeObjective = OptimizeObjective.BALANCED) -> Optimizer:
    """Build an Optimizer without hardware detection or model inspection."""
    opt = Optimizer.__new__(Optimizer)
    opt.objective = objective
    opt._n_prompt = 512
    return opt


def _result(gen=10.0, prompt=50.0, startup=5.0, memory=2000.0) -> BenchmarkResult:
    return BenchmarkResult(generation_tps=gen, prompt_tps=prompt, startup_time=startup, memory_usage=memory,
                           success=True)


def test_score_max_generation_tps():
    opt = _make_optimizer(OptimizeObjective.MAX_GENERATION_TPS)
    assert opt._score(_result(gen=42.0), SearchConfig()) == 42.0


def test_score_max_prompt_tps():
    opt = _make_optimizer(OptimizeObjective.MAX_PROMPT_TPS)
    assert opt._score(_result(prompt=123.0), SearchConfig()) == 123.0


def test_score_min_latency_is_the_time_to_the_first_word():
    opt = _make_optimizer(OptimizeObjective.MIN_LATENCY)
    # 512 prompt tokens at 64 tokens/s, then one token at 4 tokens/s
    assert opt._score(_result(gen=4.0, prompt=64.0), SearchConfig()) == pytest.approx(-(8.0 + 0.25))


def test_score_min_latency_faster_prompt_reading_wins():
    opt = _make_optimizer(OptimizeObjective.MIN_LATENCY)
    assert opt._score(_result(prompt=100.0), SearchConfig()) > opt._score(_result(prompt=50.0), SearchConfig())


def test_score_min_latency_without_speeds_falls_back_to_run_time():
    opt = _make_optimizer(OptimizeObjective.MIN_LATENCY)
    assert opt._score(_result(gen=0.0, prompt=0.0, startup=5.0), SearchConfig()) == -5.0


def test_score_max_context_is_writing_speed_because_context_is_not_searched():
    opt = _make_optimizer(OptimizeObjective.MAX_CONTEXT)
    assert opt._score(_result(gen=10.0), SearchConfig(ctx_size=32768)) == 10.0


def test_score_max_efficiency():
    opt = _make_optimizer(OptimizeObjective.MAX_EFFICIENCY)
    assert opt._score(_result(gen=10.0, memory=2000.0), SearchConfig()) == 10.0 / 2000.0


def test_score_balanced_weighs_writing_and_prompt_reading():
    opt = _make_optimizer(OptimizeObjective.BALANCED)
    assert opt._score(_result(gen=8.0, prompt=50.0), SearchConfig()) == pytest.approx(math.sqrt(400.0))
    assert opt._score(_result(gen=7.0, prompt=0.0), SearchConfig()) == 7.0   # no prompt test: writing speed


# ── grid ─────────────────────────────────────────────────────────────


def test_grid_reaches_both_ends_of_the_range():
    opt = _make_optimizer()
    assert opt._grid_values(ParamDef("batch_size", 128, 4096, step=128), 5) == [128, 1152, 2176, 3072, 4096]
    assert opt._grid_values(ParamDef("threads", 2, 4, step=1), 5) == [2, 3, 4]
    assert opt._grid_values(ParamDef("threads", 3, 3, step=1), 5) == [3]


def test_grid_can_put_every_layer_on_the_gpu():
    from llama_autotune.models import Backend, HardwareInfo, ModelInfo
    from llama_autotune.search_space import get_search_space

    for layers in (32, 30, 3):
        space = get_search_space(HardwareInfo(backend=Backend.CUDA, gpu_count=1), ModelInfo(n_layers=layers),
                                 OptimizeObjective.BALANCED)
        values = _make_optimizer()._grid_values(space["n_gpu_layers"], 5)
        assert values[-1] == layers and (space["n_gpu_layers"].high - space["n_gpu_layers"].low) % \
            space["n_gpu_layers"].step == 0


# ── confirming the winner ────────────────────────────────────────────


def _confirming(speeds: dict[int, float]) -> Optimizer:
    opt = _make_optimizer(OptimizeObjective.MAX_GENERATION_TPS)
    opt._baseline_config = SearchConfig(threads=2)
    opt._best_config = SearchConfig(threads=3)
    opt._best_score = 14.0  # the lucky single measurement from the search
    opt.measure = lambda cfg: _result(gen=speeds[cfg.threads])
    return opt


def test_a_lucky_search_result_does_not_replace_the_baseline():
    opt = _confirming({2: 11.5, 3: 11.6})   # re-measured: within noise
    opt._confirm_winner()
    assert opt.best_config.threads == 2 and opt.best_score == pytest.approx(11.5)


def test_a_real_win_is_kept_with_its_confirmed_score():
    opt = _confirming({2: 11.5, 3: 13.9})
    opt._confirm_winner()
    assert opt.best_config.threads == 3 and opt.best_score == pytest.approx(13.9)
