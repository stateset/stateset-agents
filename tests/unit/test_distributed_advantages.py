"""Reward batches and rollout reuse preserve the estimator's population."""

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import loss_computation as lc  # noqa: E402
from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from stateset_agents.training.single_turn_trainer import (  # noqa: E402
    SingleTurnGRPOTrainer,
)
from tests.unit.test_online_accumulation_weighting import _run, _trainer  # noqa: E402
from tests.unit.test_token_group_aggregation import _config, _groups  # noqa: E402


@pytest.mark.parametrize("clip", [None, 3.0, 0.0])
def test_batch_norm_uses_all_rewards_and_complete_prompt_means(clip):
    config = _config("reinforce_pp_baseline", reward_clip=clip)
    groups = _groups()
    rewards = [
        torch.tensor([t.total_reward for t in g.trajectories], dtype=torch.float32)
        for g in groups
    ]
    if clip is not None:
        rewards = [r.clamp(-clip, clip) for r in rewards]
    population_std = torch.cat(rewards).std(correction=0)
    expected = [
        (
            (r - r.mean()) / (population_std + 1e-8)
            if population_std > 1e-8
            else torch.zeros_like(r)
        )
        for r in rewards
    ]
    actual = lc._token_batch_advantages(
        groups, config, lc._token_objective(config), "cpu"
    )
    for observed, wanted in zip(actual, expected, strict=True):
        torch.testing.assert_close(observed, wanted)


def test_running_mean_observes_one_clipped_batch_and_uses_previous_state():
    config = _config("grpo", reward_clip=3.0)
    config.objective = None
    config.baseline_type = "global_mean"
    config.advantage_normalization = False
    seen = []
    actual = lc._token_batch_advantages(
        _groups(),
        config,
        lc._token_objective(config),
        "cpu",
        global_reward_mean=0.5,
        global_reward_count=3,
        global_reward_sum=1.5,
        update_global_stats=lambda mean, count: seen.append((mean, count)),
    )
    assert seen == [(0.4, 5)]
    assert [a.tolist() for a in actual] == [[-2.5, 2.5], [-3.5, 0.5, 2.5]]


@pytest.mark.parametrize(
    "fault", ["nan", "inf_clipped", "fp32_overflow", "clip", "state", "count", "sum"]
)
def test_invalid_reward_batches_do_not_mutate_baselines(fault):
    groups = _groups()
    config = _config("grpo")
    config.objective = None
    config.baseline_type = "global_mean"
    kwargs = {
        "global_reward_mean": 0.5,
        "global_reward_count": 3,
        "global_reward_sum": 1.5,
    }
    if fault == "nan":
        groups[0].trajectories[0].total_reward = float("nan")
    elif fault == "inf_clipped":
        groups[0].trajectories[0].total_reward = float("inf")
        config.reward_clip = 3.0
    elif fault == "fp32_overflow":
        groups[0].trajectories[0].total_reward = 1e50
    elif fault == "clip":
        config.reward_clip = -1
    elif fault == "state":
        kwargs["global_reward_mean"] = float("nan")
    elif fault == "count":
        kwargs["global_reward_count"] = 2**63 - 1
        kwargs["global_reward_sum"] = None
    elif fault == "sum":
        kwargs["global_reward_sum"] = 99.0
    seen = []
    with pytest.raises(ValueError):
        lc._token_batch_advantages(
            groups,
            config,
            lc._token_objective(config),
            "cpu",
            **kwargs,
            update_global_stats=lambda *values: seen.append(values),
        )
    assert not seen


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
@pytest.mark.parametrize("inner", [1, 3])
@pytest.mark.parametrize("enhanced", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
async def test_rollout_baseline_is_frozen_and_rewards_counted_once(
    tmp_path, monkeypatch, kind, inner, enhanced, normalize
):
    trainer = _trainer(tmp_path, kind, "grpo", 0.0, 1)
    trainer.config.objective = None
    trainer.config.baseline_type = "global_mean"
    trainer.config.advantage_normalization = normalize
    trainer.config.num_gradient_updates = inner
    trainer.config.use_reference_model = enhanced
    trainer.config.reward_clip = 3.0
    trainer._update_global_stats(0.5, 3)
    seen = []
    original = lc._token_batch_advantages

    def inspect(*args, **kwargs):
        seen.append(
            (
                kwargs["global_reward_mean"],
                kwargs["global_reward_count"],
                kwargs["update_global_stats"] is not None,
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(lc, "_token_batch_advantages", inspect)
    await _run(trainer, _groups())
    assert trainer._global_reward_count == 8
    assert trainer._global_reward_mean == pytest.approx((1.5 + 2.0) / 8)
    assert seen[:inner] == [(0.5, 3, index == 0) for index in range(inner)]
    assert seen[inner:] == [(0.5, 5, index == 0) for index in range(inner)]
    assert getattr(trainer, "_loss_reward_baseline", None) is None


def test_frozen_baseline_context_restores_after_loss_failure(tmp_path):
    trainer = _trainer(tmp_path, MultiTurnGRPOTrainer, "grpo", 0, 1)
    snapshot = trainer._reward_baseline_kwargs()
    with pytest.raises(RuntimeError, match="loss failed"):
        with trainer._frozen_reward_baseline(snapshot, update=False):
            assert trainer._reward_baseline_kwargs()["update_global_stats"] is None
            raise RuntimeError("loss failed")
    assert trainer._reward_baseline_kwargs() == snapshot


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
@pytest.mark.parametrize("enhanced", [False, True])
async def test_pending_flush_does_not_reobserve_inner_rollout(
    tmp_path, monkeypatch, kind, enhanced
):
    from stateset_agents.training.grpo_accumulation import backward_accumulated_loss

    trainer = _trainer(tmp_path, kind, "grpo", 0, 2)
    trainer.config.objective = None
    trainer.config.baseline_type = "global_mean"
    trainer.config.advantage_normalization = False
    trainer.config.reward_clip = 3.0
    trainer._update_global_stats(0.5, 3)
    pending = lc.compute_grpo_loss(
        [_groups()[0]],
        trainer.config,
        trainer.agent,
        0.5,
        3,
        trainer._update_global_stats,
    )
    backward_accumulated_loss(trainer, pending["total_loss"], pending, torch)
    trainer._grad_accum_step += 1
    assert trainer._global_reward_count == 5
    trainer.config.num_gradient_updates = 3
    trainer.config.use_reference_model = enhanced
    seen = []
    original = lc._token_batch_advantages

    def inspect(*args, **kwargs):
        seen.append(
            (
                kwargs["global_reward_mean"],
                kwargs["global_reward_count"],
                kwargs["update_global_stats"] is not None,
            )
        )
        return original(*args, **kwargs)

    monkeypatch.setattr(lc, "_token_batch_advantages", inspect)
    await _run(trainer, _groups()[1:])
    assert trainer._global_reward_count == 8
    assert trainer._global_reward_mean == pytest.approx(3.5 / 8)
    # The old-policy snapshot precedes the flush; the streamed learner then
    # scores exactly one batch for each inner update, with no discarded pass.
    assert len(seen) == 3
    assert all((mean, count) == (0.5, 5) for mean, count, _ in seen)
    assert sum(updating for _, _, updating in seen) == 1
    assert trainer.global_step == 4  # one pending flush, then three inner steps


def test_distributed_advantage_helpers_import_without_torch():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import stateset_agents.training.distributed_advantages; assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
