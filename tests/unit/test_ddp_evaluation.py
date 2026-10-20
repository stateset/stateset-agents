"""Global evaluation means must retain population weights and failure coverage."""

import asyncio
import copy
import json
import math
import random
from unittest.mock import AsyncMock, patch

import numpy as np
import pytest

from stateset_agents.training.ddp_evaluation import aggregate_evaluation_metrics


def _metrics(rewards, attempted=None, length=2.0):
    attempted = len(rewards) if attempted is None else attempted
    scale = max(map(abs, rewards), default=0.0)
    return {
        "eval_reward": math.fsum(r / len(rewards) for r in rewards),
        "eval_reward_std": (
            float(np.std(np.array(rewards) / scale)) * scale if scale else 0.0
        ),
        "eval_episode_length": length if rewards else 0.0,
        "eval_success_rate": (
            sum(r > 0.5 for r in rewards) / attempted if attempted else 0.0
        ),
        "eval_num_episodes": float(len(rewards)),
        "eval_attempted_episodes": float(attempted),
        "eval_failed_episodes": float(attempted - len(rewards)),
        "eval_error_rate": (attempted - len(rewards)) / attempted if attempted else 0.0,
        "eval_complete": float(len(rewards) == attempted and attempted > 0),
    }


@pytest.mark.parametrize(
    "rows",
    [
        [[0.0], [1.0, 1.0, 1.0]],
        [[], [0.0, 2.0]],
        [[], []],
        [[-1e308], [1e308]],
        [[1e308, 1e308], [1e308]],
    ],
)
def test_global_population_moments_use_sample_counts(rows):
    actual = aggregate_evaluation_metrics([_metrics(row) for row in rows])
    expected = _metrics([r for row in rows for r in row])
    for name in expected:
        assert actual[name] == pytest.approx(expected[name], rel=1e-14)


def test_failed_attempts_remain_in_global_denominators():
    result = aggregate_evaluation_metrics([_metrics([1.0], 3), _metrics([0.0, 1.0])])
    assert result["eval_reward"] == pytest.approx(2 / 3)
    assert result["eval_success_rate"] == 2 / 5
    assert result["eval_error_rate"] == 2 / 5
    assert result["eval_complete"] == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("eval_num_episodes", 1.5),
        ("eval_attempted_episodes", 0),
        ("eval_success_rate", 0.1),
        ("eval_failed_episodes", -1),
        ("eval_reward", float("nan")),
        ("eval_reward_std", -1),
        ("eval_episode_length", -1),
        ("eval_complete", 0),
        ("eval_num_episodes", True),
        ("eval_error_rate", 0.1),
    ],
)
def test_invalid_local_statistics_reject(field, value):
    row = _metrics([0.0, 1.0])
    row[field] = value
    with pytest.raises(ValueError):
        aggregate_evaluation_metrics([row])


@pytest.mark.parametrize("row", [{}, {"unknown": 1}, None])
def test_metric_schema_is_explicit(row):
    with pytest.raises(ValueError):
        aggregate_evaluation_metrics([row])


def test_empty_and_overflowing_global_populations_reject():
    from stateset_agents.training.evaluation import _EvaluationMetrics

    with pytest.raises(ValueError):
        aggregate_evaluation_metrics([])
    row = _metrics([])
    row.update(
        eval_attempted_episodes=float(2**53 - 1),
        eval_failed_episodes=float(2**53 - 1),
        eval_error_rate=1.0,
    )
    with pytest.raises(ValueError):
        aggregate_evaluation_metrics([_EvaluationMetrics(row, successes=0)] * 2)


def test_large_native_count_receipt_is_exact_and_stays_out_of_json():
    from stateset_agents.training.ddp_evaluation import _pack
    from stateset_agents.training.evaluation import _EvaluationMetrics

    count, successes = 2**53 - 1, 2**52 + 1
    rate = successes / count
    row = _metrics([0.0, 1.0])
    row.update(
        eval_num_episodes=float(count),
        eval_attempted_episodes=float(count),
        eval_success_rate=rate,
        eval_reward=rate,
    )
    with pytest.raises(ValueError, match="exact native count"):
        aggregate_evaluation_metrics([row])
    receipt = _EvaluationMetrics(row, successes=successes)
    result = aggregate_evaluation_metrics([receipt])
    assert _pack(result)[0][3] == successes
    assert set(json.loads(json.dumps(result))) == set(row)


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [RuntimeError, asyncio.CancelledError])
async def test_best_save_failure_preserves_previous_selection(tmp_path, error):
    from tests.integration.test_replica_consensus import _make

    trainer = _make(tmp_path)
    trainer.best_eval_metric = 0.2
    trainer.steps_without_improvement = 3
    trainer.save_checkpoint = AsyncMock(side_effect=error("save failed"))
    with pytest.raises(error):
        await trainer._update_best_checkpoint(_metrics([1.0]))
    assert trainer.best_eval_metric == 0.2
    assert trainer.steps_without_improvement == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", [False, True])
async def test_local_evaluation_preserves_rngs_modes_and_gradients(tmp_path, failure):
    torch = pytest.importorskip("torch")
    from stateset_agents.training.runtime_state import capture_rng
    from tests.integration.test_learner_preflight import _same
    from tests.integration.test_replica_consensus import _make

    trainer = _make(tmp_path)
    model = trainer.agent.model
    model.train()
    model.lookup.eval()
    model.config.use_cache = False
    for p in model.parameters():
        p.grad = torch.ones_like(p)
    gradients = [p.grad.clone() for p in model.parameters()]
    modes = [m.training for m in model.modules()]
    before = copy.deepcopy(capture_rng(torch))

    async def local(**kwargs):
        assert not model.training and not torch.is_grad_enabled()
        assert model.config.use_cache is True
        random.random(), np.random.rand(), torch.rand(1)
        if failure:
            raise RuntimeError("evaluation failed")
        return _metrics([0.0, 1.0])

    with patch(
        "stateset_agents.training.multi_turn_trainer.evaluate_agent", side_effect=local
    ):
        if failure:
            with pytest.raises(RuntimeError):
                await trainer._evaluate_local([{}], 1)
        else:
            await trainer._evaluate_local([{}], 1)
    _same(capture_rng(torch), before, torch)
    assert modes == [m.training for m in model.modules()]
    assert model.config.use_cache is False
    _same([p.grad for p in model.parameters()], gradients, torch)


@pytest.mark.asyncio
@pytest.mark.parametrize("metrics", [_metrics([]), _metrics([1.0], attempted=2)])
async def test_incomplete_evaluation_does_not_select_or_consume_patience(
    tmp_path, metrics
):
    from tests.integration.test_replica_consensus import _make

    trainer = _make(tmp_path)
    trainer.best_eval_metric = 0.2
    trainer.steps_without_improvement = 3
    trainer.save_checkpoint = AsyncMock()
    await trainer._update_best_checkpoint(metrics)
    trainer.save_checkpoint.assert_not_awaited()
    assert (trainer.best_eval_metric, trainer.steps_without_improvement) == (0.2, 3)
