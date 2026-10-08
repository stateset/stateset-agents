"""Measure policy advantages per group, not apparent pooled reward variation."""

from types import SimpleNamespace

import pytest

from stateset_agents.training.callbacks import ZeroSignalGuard, has_no_policy_signal


@pytest.mark.parametrize(
    "metrics",
    [
        {},
        {"loss": 0.0},
        {"average_reward": 0.0},
        {"reward_std": 0.0},
        {"average_reward": None, "reward_std": 0.0},
        {"average_reward": False, "reward_std": 0.0},
        {"average_reward": "0", "reward_std": 0.0},
        {"average_reward": float("nan"), "reward_std": 0.0},
        {"average_reward": 10**400, "reward_std": 0.0},
    ],
)
def test_unknown_reward_evidence_breaks_consecutive_zero_streak(metrics):
    guard = ZeroSignalGuard(max_zero_steps=2)
    zero = {"average_reward": 0.0, "reward_std": 0.0}
    guard.on_step_end(0, zero)
    guard.on_step_end(1, metrics)
    assert guard.zero_steps == 0
    guard.on_step_end(2, zero)
    assert not guard.should_abort


@pytest.mark.parametrize(
    "fraction", [None, False, "0", float("nan"), float("inf"), -1, 2]
)
def test_invalid_advantage_metric_does_not_fall_back_to_zero_rewards(fraction):
    assert not has_no_policy_signal(
        {
            "nonzero_advantage_fraction": fraction,
            "average_reward": 0.0,
            "reward_std": 0.0,
        }
    )


def test_measured_advantages_override_rewards_and_abort_is_latched():
    guard = ZeroSignalGuard(max_zero_steps=2)
    zero = {"nonzero_advantage_fraction": 0.0, "average_reward": 1.0, "reward_std": 0.5}
    guard.on_step_end(0, zero)
    guard.on_step_end(1, {"nonzero_advantage_fraction": 0.1})
    assert guard.zero_steps == 0
    guard.on_step_end(2, zero)
    guard.on_step_end(3, zero)
    assert guard.should_abort
    reason = guard.abort_reason
    assert "2 consecutive steps" in reason
    guard.on_step_end(4, {})
    assert guard.should_abort and guard.abort_reason == reason
    assert not has_no_policy_signal(
        {"nonzero_advantage_fraction": 0.5, "average_reward": 0.0, "reward_std": 0.0}
    )


@pytest.mark.parametrize("patience", [0, -1, True, 1.5, "2", None])
def test_guard_rejects_invalid_patience(patience):
    with pytest.raises(ValueError, match="positive integer"):
        ZeroSignalGuard(max_zero_steps=patience)


@pytest.mark.asyncio
@pytest.mark.parametrize("token_level", [False, True], ids=["gspo", "gspo_token"])
@pytest.mark.parametrize(
    "groups,active_fraction,zero_group_fraction",
    [
        ([[0.0, 0.0], [1.0, 1.0]], 0.0, 1.0),
        ([[1.0, 1.0]], 0.0, 1.0),
        ([[2.0]], 0.0, 1.0),
        ([[0.0, 0.0, 0.0], [0.0, 2.0]], 0.4, 0.5),
        ([[0.0, 1.0, 2.0]], 2 / 3, 0.0),
    ],
)
async def test_real_trainers_record_group_advantages(
    monkeypatch, token_level, groups, active_fraction, zero_group_fraction
):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from stateset_agents.training.gspo_config import GSPOConfig
    from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer
    from stateset_agents.training.gspo_trainer import GSPOTrainer
    from tests._tiny_tokenizer import tiny_tokenizer

    torch.manual_seed(0)
    model = transformers.GPT2LMHeadModel(
        transformers.GPT2Config(
            n_embd=16,
            n_layer=1,
            n_head=2,
            vocab_size=256,
            n_positions=64,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )

    class Reward:
        async def compute_turn_reward(self, turn, context):
            return SimpleNamespace(
                total_reward=groups[int(context["user_query"])][int(turn.content)]
            )

        async def compute_reward(self, turns, context):
            return await self.compute_turn_reward(turns[0], context)

    trainer_class = GSPOTokenTrainer if token_level else GSPOTrainer
    trainer = trainer_class(
        config=GSPOConfig(
            model_name="gpt2",
            num_generations=2,
            num_outer_iterations=1,
            max_prompt_length=16,
            max_completion_length=16,
        ),
        model=model,
        tokenizer=tiny_tokenizer(),
        agent=None,
        environment=None,
        reward_model=Reward(),
        ref_model=None,
    )

    async def generate(prompt, count):
        return [(str(i), -5.0) for i in range(len(groups[int(prompt)]))]

    monkeypatch.setattr(trainer.generator, "generate_group_responses", generate)
    train_step = trainer.train_step_token_level if token_level else trainer.train_step
    metrics = await train_step(
        [str(i) for i in range(len(groups))], num_groups=len(groups)
    )
    assert metrics["nonzero_advantage_fraction"] == pytest.approx(active_fraction)
    assert metrics["zero_advantage_group_fraction"] == pytest.approx(
        zero_group_fraction
    )
    assert metrics["advantage_group_count"] == len(groups)
    for key in (
        "nonzero_advantage_fraction",
        "zero_advantage_group_fraction",
        "advantage_group_count",
    ):
        assert trainer.training_metrics[key] == [metrics[key]]
    guard = ZeroSignalGuard(max_zero_steps=1)
    guard.on_step_end(0, metrics)
    assert guard.should_abort == (active_fraction == 0.0)
    if len(groups) > 1:
        assert metrics["reward_std"] > 0.0
