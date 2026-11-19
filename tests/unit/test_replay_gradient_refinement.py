"""Compare native replay gradients with exact finite group expectations."""

import itertools
import json
import math
from fractions import Fraction
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.trajectory import (  # noqa: E402
    ConversationTurn,
    MultiTurnTrajectory,
    TrajectoryGroup,
)
from stateset_agents.training import loss_computation as lc  # noqa: E402
from stateset_agents.training.continual_learning import (  # noqa: E402
    ContinualLearningConfig,
    ContinualLearningManager,
)
from stateset_agents.training.multi_turn_trainer import (  # noqa: E402
    MultiTurnGRPOTrainer,
)
from stateset_agents.training.single_turn_trainer import (  # noqa: E402
    SingleTurnGRPOTrainer,
)
from tests.unit.test_online_accumulation_weighting import _run, _trainer  # noqa: E402
from tests.unit.test_token_group_aggregation import _config  # noqa: E402

CURRENT = (Fraction(1, 3),) * 3
BEHAVIOR = (Fraction(1, 2), Fraction(1, 4), Fraction(1, 4))


class ThreeActionPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.logits = torch.nn.Parameter(torch.zeros(3))

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.logits.expand(*input_ids.shape, -1))


def group(pair, probabilities):
    trajectories = [
        MultiTurnTrajectory(
            total_reward=float(action),
            turns=[
                ConversationTurn(
                    role="assistant",
                    content=str(action),
                    metadata={
                        "prompt_token_ids": [0],
                        "token_ids": [action],
                        "sampler_log_probs": [math.log(probabilities[action])],
                    },
                )
            ],
        )
        for action in pair
    ]
    return TrajectoryGroup(scenario_id="fixed prompt", trajectories=trajectories)


def pair_ascent(pair, p, old, mode, epsilon=Fraction(0)):
    gradient = [0, 0, 0]
    for action, peer in (pair, pair[::-1]):
        gap = abs(action - peer)
        advantage = (action - peer) / (gap + 2 * epsilon) if gap > 2 * epsilon else 0
        ratio = p[action] / old[action] if mode != "uncorrected" else 1
        if mode == "clipped" and (
            (advantage > 0 and ratio > Fraction(6, 5))
            or (advantage < 0 and ratio < Fraction(4, 5))
        ):
            continue
        if mode == "joint":
            ratio *= p[peer] / old[peer]
        for k in range(3):
            gradient[k] += ratio * advantage * (int(action == k) - p[k]) / 2
    return gradient


@pytest.mark.parametrize("mode", ["fresh", "uncorrected", "row", "clipped", "joint"])
@pytest.mark.parametrize("epsilon", [0.0, 1e-8, 0.5, 1.0])
def test_exact_expected_group_gradient_through_native_autograd(mode, epsilon):
    model = ThreeActionPolicy()
    law = CURRENT if mode == "fresh" else BEHAVIOR
    native = torch.zeros(3, dtype=torch.float64)
    oracle = [Fraction(0) for _ in range(3)]
    cfg = _config(
        old_logprobs_source="sampler" if mode in ("row", "clipped") else "recompute",
        objective_overrides={
            "advantage_eps": epsilon,
            "clip": "clipped" if mode == "clipped" else "none",
        },
    )
    for pair in itertools.product(range(3), repeat=2):
        a, b = pair
        probability = law[a] * law[b]
        values = lc.compute_grpo_loss(
            [group(pair, law)], cfg, SimpleNamespace(model=model), 0.0, 0, None
        )
        (loss_grad,) = torch.autograd.grad(values["total_loss"], model.logits)
        # Joint importance weighting is an oracle applied to the unweighted
        # native score gradient. Both group factors are held fixed.
        weight = CURRENT[a] * CURRENT[b] / probability if mode == "joint" else 1
        native -= float(probability * weight) * loss_grad.double()
        analytic = pair_ascent(pair, CURRENT, law, mode, Fraction(epsilon))
        oracle = [v + probability * g for v, g in zip(oracle, analytic, strict=True)]
    eps = Fraction(epsilon)
    near = 1 / (1 + 2 * eps) if eps < Fraction(1, 2) else 0
    far = 1 / (1 + eps) if eps < 1 else 0
    middle = {
        "uncorrected": near / 16,
        "row": (2 * near - far) / 36,
        "clipped": -near / 18,
    }
    assert oracle[1] == middle.get(mode, 0)
    torch.testing.assert_close(
        native,
        torch.tensor(list(map(float, oracle)), dtype=native.dtype),
        atol=1e-7,
        rtol=1e-6,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", [MultiTurnGRPOTrainer, SingleTurnGRPOTrainer])
@pytest.mark.parametrize("inner", [1, 3])
async def test_native_trainers_use_each_groups_behavior_at_every_step(
    tmp_path, kind, inner
):
    trainer = _trainer(tmp_path, kind, "grpo", 0.0, 1)
    trainer.config.num_gradient_updates = inner
    trainer.config.old_logprobs_source = "sampler"
    trainer.config.objective_overrides = {"advantage_eps": 0.0}
    model = ThreeActionPolicy()
    trainer.agent.model = model
    lr = 0.15
    trainer.optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    manager = ContinualLearningManager(
        ContinualLearningConfig(strategy="replay", replay_ratio=1, replay_min_size=1),
        seed=48,
    )
    old_pair, new_pair = (0, 1), (0, 2)
    manager.add_trajectory_groups([group(old_pair, BEHAVIOR)], task_id="old")
    if inner > 1:
        saved = json.loads(json.dumps(manager.state_dict()))
        manager = ContinualLearningManager(manager.config, seed=99)
        manager.load_state_dict(saved)
    trainer.continual_manager = manager
    fresh = group(new_pair, CURRENT)
    observed = []
    trainer.optimizer.register_step_pre_hook(
        lambda *args: observed.append((-model.logits.grad).detach().clone())
    )
    if kind == MultiTurnGRPOTrainer:
        mixed, replay = trainer._maybe_mix_replay([fresh], "new")
        assert len(replay) == 1
        out = await trainer.training_step(mixed)
        assert out["old_logprobs_source"] == "sampler"
    else:
        await _run(trainer, [fresh])
    logits = [0.0] * 3
    assert len(observed) == inner
    for actual in observed:
        exp = list(map(math.exp, logits))
        p = [v / sum(exp) for v in exp]
        fresh_grad = pair_ascent(new_pair, p, CURRENT, "clipped")
        replay_grad = pair_ascent(old_pair, p, BEHAVIOR, "clipped")
        expected = [(a + b) / 2 for a, b in zip(fresh_grad, replay_grad, strict=True)]
        torch.testing.assert_close(actual, torch.tensor(expected), rtol=1e-6, atol=1e-7)
        logits = [value + lr * g for value, g in zip(logits, expected, strict=True)]
    torch.testing.assert_close(model.logits, torch.tensor(logits), rtol=1e-6, atol=1e-7)


def test_missing_old_scores_require_opt_in_and_freeze_one_fallback():
    batch = [group((0, 1), BEHAVIOR)]
    batch[0].trajectories[1].turns[0].metadata.pop("sampler_log_probs")
    cfg = _config(old_logprobs_source="sampler")
    model = ThreeActionPolicy()
    agent = SimpleNamespace(model=model)
    with pytest.raises(ValueError, match="sampler log probabilities"):
        lc.freeze_rollout_logprobs(batch, cfg, agent)
    cfg.allow_missing_sampler_logprobs = True
    snapshot = lc.freeze_rollout_logprobs(batch, cfg, agent)
    with torch.no_grad():
        model.logits[0] += 2
    result = lc.compute_grpo_loss(
        batch, cfg, agent, 0.0, 0, None, old_logprobs=snapshot
    )
    assert result["old_logprobs_source"] == "recompute_fallback"
    assert result["ratio_mean"] != pytest.approx(1.0)


@pytest.mark.parametrize(
    "estimator,factor", [("group_mean", Fraction(2, 3)), ("leave_one_out", 1)]
)
def test_three_response_raw_baselines_cancel_peer_shift(estimator, factor):
    model = ThreeActionPolicy()
    cfg = _config(
        old_logprobs_source="sampler",
        objective_overrides={"advantage": estimator, "clip": "none"},
    )
    ascent = torch.zeros(3, dtype=torch.float64)
    for triple in itertools.product(range(3), repeat=3):
        batch = group(triple, BEHAVIOR)
        loss = lc.compute_grpo_loss(
            [batch], cfg, SimpleNamespace(model=model), 0.0, 0, None
        )
        (gradient,) = torch.autograd.grad(loss["total_loss"], model.logits)
        ascent -= float(math.prod(BEHAVIOR[i] for i in triple)) * gradient.double()
    expected = [float(factor * (i - 1) * CURRENT[i]) for i in range(3)]
    torch.testing.assert_close(
        ascent, torch.tensor(expected, dtype=ascent.dtype), atol=1e-7, rtol=1e-6
    )


class PrefixPolicy(torch.nn.Module):
    """One logit controls the second token only after a zero first token."""

    def __init__(self):
        super().__init__()
        self.coordinate = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, input_ids, attention_mask=None):
        one = self.coordinate * input_ids.eq(0)
        return SimpleNamespace(logits=torch.stack([torch.zeros_like(one), one], dim=-1))


@pytest.mark.parametrize(
    "first_zero,expected",
    [(Fraction(1, 2), Fraction(1, 16)), (Fraction(3, 4), Fraction(3, 32))],
)
def test_conditional_token_ratios_leave_historical_prefix_visitation(
    first_zero, expected
):
    model = PrefixPolicy()
    cfg = _config(
        old_logprobs_source="sampler", objective_overrides={"advantage_eps": 0.0}
    )
    paths = {}
    for first, second in itertools.product(range(2), repeat=2):
        prefix_mass = first_zero if first == 0 else 1 - first_zero
        row = MultiTurnTrajectory(
            total_reward=float(second),
            turns=[
                ConversationTurn(
                    role="assistant",
                    content="two tokens",
                    metadata={
                        "prompt_token_ids": [2],
                        "token_ids": [first, second],
                        "sampler_log_probs": [math.log(prefix_mass), math.log(0.5)],
                    },
                )
            ],
        )
        paths[first, second] = (row, prefix_mass / 2)
    ascent = 0.0
    for (left, pl), (right, pr) in itertools.product(paths.values(), repeat=2):
        batch = TrajectoryGroup(scenario_id="fixed prompt", trajectories=[left, right])
        values = lc.compute_grpo_loss(
            [batch], cfg, SimpleNamespace(model=model), 0.0, 0, None
        )
        (grad,) = torch.autograd.grad(values["total_loss"], model.coordinate)
        ascent -= float(pl * pr) * grad.item()
    assert ascent == pytest.approx(float(expected), abs=1e-7)
