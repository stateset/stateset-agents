"""Prompt-group boundaries must preserve the configured objective's measure."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.trajectory import (  # noqa: E402
    ConversationTurn,
    MultiTurnTrajectory,
    TrajectoryGroup,
)
from stateset_agents.training import loss_computation as lc  # noqa: E402
from stateset_agents.training.objectives import OBJECTIVES  # noqa: E402


class TinyPolicy(torch.nn.Module):
    def __init__(self, seed=7):
        super().__init__()
        weights = torch.randn(16, 16, generator=torch.Generator().manual_seed(seed))
        self.lookup = torch.nn.Embedding.from_pretrained(weights, freeze=False)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.lookup(input_ids))


def _trajectory(reward, *responses):
    return MultiTurnTrajectory(
        total_reward=reward,
        turns=[
            ConversationTurn(
                role="assistant",
                content="logged action",
                metadata={"prompt_token_ids": [1, 2], "token_ids": response},
            )
            for response in responses
        ],
    )


def _groups(zero_rewards=False):
    groups = [
        TrajectoryGroup(
            scenario_id="first",
            trajectories=[_trajectory(-2, [2, 3], [4]), _trajectory(5, [5, 6, 7, 8])],
        ),
        TrajectoryGroup(
            scenario_id="second",
            trajectories=[
                _trajectory(-10, [2]),
                _trajectory(1, [5, 4], [3, 6], [7, 8]),
                _trajectory(9, [9, 10, 11]),
            ],
        ),
    ]
    if zero_rewards:
        for group in groups:
            for trajectory in group.trajectories:
                trajectory.total_reward = 0.0
    return groups


def _config(name="grpo", **overrides):
    values = {
        "objective": name,
        "objective_overrides": None,
        "max_completion_length": 8,
        "generation_batch_size": 3,
        "entropy_coef": 0.0,
        "reward_clip": None,
        "clip_ratio": 0.2,
        "advantage_normalization": True,
        "baseline_type": "group_mean",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _row_terms(model, turn):
    """Independent causal-token calculation, without trainer loss helpers."""
    prompt = turn.metadata["prompt_token_ids"]
    response = turn.metadata["token_ids"]
    ids = torch.tensor([prompt + response])
    logits = model(ids).logits[0, len(prompt) - 1 : -1].float()
    log_probs = logits.log_softmax(-1)
    selected = log_probs.gather(-1, torch.tensor(response).unsqueeze(-1)).squeeze(-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)
    return selected, entropy


def _reference(groups, model, config, *, beta=0.0, reference_model=None):
    """Explicit sums over tokens/turns/trajectories with the named denominator."""
    objective = OBJECTIVES[config.objective]
    rewards = torch.tensor(
        [
            trajectory.total_reward
            for group in groups
            for trajectory in group.trajectories
        ],
        dtype=torch.float32,
    )
    if config.reward_clip is not None:
        rewards = rewards.clamp(-config.reward_clip, config.reward_clip)
    terms, entropies, kls = [], [], []
    for group in groups:
        values = torch.tensor(
            [trajectory.total_reward for trajectory in group.trajectories],
            dtype=torch.float32,
        )
        if config.reward_clip is not None:
            values = values.clamp(-config.reward_clip, config.reward_clip)
        centered = values - values.mean()
        if objective.advantage == "leave_one_out":
            advantages = values - (values.sum() - values) / (len(values) - 1)
        elif objective.advantage == "group_mean":
            advantages = centered
        else:
            scale = (
                rewards.std(correction=0)
                if objective.advantage == "batch_norm"
                else values.std(correction=0)
            )
            advantages = (
                centered / (scale + objective.advantage_eps)
                if scale > objective.advantage_eps
                else torch.zeros_like(centered)
            )
        for trajectory, advantage in zip(group.trajectories, advantages, strict=True):
            for turn in trajectory.turns:
                selected, entropy = _row_terms(model, turn)
                # The tests use the on-policy ratio 1, preserving its gradient.
                ratio = (selected - selected.detach()).exp()
                surrogate = (
                    -advantage * selected
                    if objective.clip == "cispo"
                    else -advantage * ratio
                )
                kl = torch.zeros_like(selected)
                if reference_model is not None:
                    with torch.no_grad():
                        reference, _ = _row_terms(reference_model, turn)
                    difference = reference - selected
                    kl = difference.exp() - difference - 1
                terms.append(surrogate + beta * kl)
                entropies.append(entropy)
                kls.append(kl)
    trajectories = sum(len(group.trajectories) for group in groups)
    tokens = sum(term.numel() for term in terms)
    if objective.aggregate == "seq_mean":
        policy = sum(term.mean() for term in terms) / trajectories
    elif objective.aggregate == "seq_sum_const":
        policy = sum(term.sum() for term in terms) / (
            trajectories * config.max_completion_length
        )
    else:
        policy = sum(term.sum() for term in terms) / tokens
    entropy = sum(value.sum() for value in entropies) / tokens
    return policy - config.entropy_coef * entropy, {
        "kl": float((sum(value.sum() for value in kls) / tokens).detach()),
        "entropy": float(entropy.detach()),
    }


def _check_reference(groups, config, *, beta=0.0):
    model = TinyPolicy()
    reference_model = TinyPolicy(seed=13) if beta else None
    agent = SimpleNamespace(model=model)
    actual = lc.compute_enhanced_grpo_loss(
        groups, beta, config, agent, reference_model=reference_model
    )
    expected, metrics = _reference(
        groups, model, config, beta=beta, reference_model=reference_model
    )
    torch.testing.assert_close(actual["total_loss"], expected, rtol=1e-5, atol=1e-6)
    parameters = list(model.parameters())
    actual_grad = torch.autograd.grad(actual["total_loss"], parameters)
    expected_grad = torch.autograd.grad(expected, parameters)
    for observed, wanted in zip(actual_grad, expected_grad, strict=True):
        torch.testing.assert_close(observed, wanted, rtol=1e-5, atol=1e-6)
    if beta:
        assert float(actual["kl_penalty"]) == pytest.approx(metrics["kl"], abs=1e-6)
    if config.entropy_coef:
        assert actual["entropy"] == pytest.approx(metrics["entropy"], abs=1e-6)


@pytest.mark.parametrize(
    "name",
    ["grpo", "dr_grpo", "dapo", "bnpo", "cispo", "rloo", "reinforce_pp_baseline"],
)
@pytest.mark.parametrize("chunk_size", [1, 16])
def test_uneven_groups_match_global_objective_and_gradient(name, chunk_size):
    _check_reference(_groups(), _config(name, generation_batch_size=chunk_size))


@pytest.mark.parametrize("name", ["grpo", "dapo", "dr_grpo"])
def test_kl_and_entropy_use_their_own_global_denominators(name):
    _check_reference(_groups(), _config(name, entropy_coef=0.07), beta=0.2)


@pytest.mark.parametrize("name", ["grpo", "dapo"])
def test_entropy_is_not_multiplied_by_assistant_turn_count(name):
    _check_reference(_groups(zero_rewards=True), _config(name, entropy_coef=0.1))


@pytest.mark.parametrize("name", ["grpo", "dapo"])
def test_equal_groups_remain_a_valid_control(name):
    groups = [
        TrajectoryGroup(
            scenario_id=str(i),
            trajectories=[_trajectory(-1, [2, 3]), _trajectory(1, [4, 5])],
        )
        for i in range(2)
    ]
    _check_reference(groups, _config(name))


def test_batch_standard_deviation_uses_clipped_rewards_across_groups():
    _check_reference(_groups(), _config("reinforce_pp_baseline", reward_clip=2.0))


def test_singleton_group_preserves_zero_advantage_and_entropy_weight():
    groups = _groups()
    groups[1].trajectories = groups[1].trajectories[:1]
    _check_reference(groups, _config(entropy_coef=0.1))


@pytest.mark.parametrize("coefficient", [-0.1, float("nan"), float("inf")])
def test_invalid_entropy_coefficient_fails_before_forward(monkeypatch, coefficient):
    model = TinyPolicy()

    def unexpected_forward(*args, **kwargs):
        raise AssertionError("Invalid objective reached model forward")

    monkeypatch.setattr(model, "forward", unexpected_forward)
    with pytest.raises(ValueError, match="entropy_coef"):
        lc.compute_grpo_loss(
            _groups(),
            _config(entropy_coef=coefficient),
            SimpleNamespace(model=model),
            0.0,
            0,
            lambda *args: None,
        )


def test_ratio_and_clip_metrics_weight_actual_sampled_tokens():
    groups = _groups()
    model = TinyPolicy()
    shifts = [0.3, -0.1]
    counts = []
    for group, shift in zip(groups, shifts, strict=True):
        count = 0
        for trajectory in group.trajectories:
            for turn in trajectory.turns:
                selected, _ = _row_terms(model, turn)
                turn.metadata["sampler_log_probs"] = (
                    selected.detach() - shift
                ).tolist()
                count += selected.numel()
        counts.append(count)
    actual = lc.compute_grpo_loss(
        groups,
        _config(old_logprobs_source="sampler"),
        SimpleNamespace(model=model),
        0.0,
        0,
        lambda *args: None,
    )
    expected_ratio = sum(
        count * torch.tensor(shift).exp().item()
        for count, shift in zip(counts, shifts, strict=True)
    ) / sum(counts)
    assert actual["ratio_mean"] == pytest.approx(expected_ratio, abs=1e-6)
    assert actual["clip_fraction"] == pytest.approx(counts[0] / sum(counts), abs=1e-6)


def test_legacy_running_baseline_receives_one_observation_per_batch():
    observed = []
    lc.compute_grpo_loss(
        _groups(),
        _config(None, baseline_type="global_mean", advantage_normalization=False),
        SimpleNamespace(model=TinyPolicy()),
        2.0,
        1,
        lambda mean, count: observed.append((mean, count)),
    )
    assert observed == [(0.6, 5)]
