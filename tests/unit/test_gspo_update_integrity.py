"""Invalid data or backward results must not advance a native GSPO update."""

import copy
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from stateset_agents.training.gspo_config import GSPOConfig  # noqa: E402
from stateset_agents.training.gspo_token_trainer import GSPOTokenTrainer  # noqa: E402
from stateset_agents.training.gspo_trainer import GSPOTrainer  # noqa: E402
from tests._tiny_tokenizer import tiny_tokenizer  # noqa: E402


@pytest.fixture(params=[GSPOTrainer, GSPOTokenTrainer], ids=["gspo", "gspo_token"])
def trainer(request, monkeypatch):
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
            return SimpleNamespace(total_reward=float(turn.content == "ok"))

        async def compute_reward(self, turns, context):
            return await self.compute_turn_reward(turns[0], context)

    result = request.param(
        config=GSPOConfig(
            model_name="gpt2",
            num_generations=2,
            num_outer_iterations=100,
            warmup_ratio=0.0,
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
        return [("ok", -5.0), ("nope", -5.0)]

    monkeypatch.setattr(result.generator, "generate_group_responses", generate)
    return result


async def _train(trainer, prompts):
    method = (
        trainer.train_step_token_level
        if isinstance(trainer, GSPOTokenTrainer)
        else trainer.train_step
    )
    return await method(prompts, num_groups=len(prompts))


def _assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        assert torch.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for item, wanted in zip(actual, expected, strict=True):
            _assert_same(item, wanted)
    else:
        assert actual == expected


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure",
    [
        "reward_nan",
        "reward_inf",
        "reward_overflow",
        "empty_group",
        "logprob_nan",
        "loss_nan",
        "loss_inf",
        "gradient_nan",
        "gradient_inf",
        "gradient_norm_overflow",
        "backward_exception",
        "max_norm_nan",
        "max_norm_inf",
    ],
)
async def test_failed_update_preserves_committed_state_and_can_recover(
    trainer, monkeypatch, failure
):
    await _train(trainer, ["good"])
    assert trainer.optimizer.state
    before_model = copy.deepcopy(trainer.model.state_dict())
    before_optimizer = copy.deepcopy(trainer.optimizer.state_dict())
    before_scheduler = copy.deepcopy(trainer.scheduler.state_dict())
    before_metrics = copy.deepcopy(trainer.training_metrics)

    hook = None
    with monkeypatch.context() as patch:
        if failure.startswith("reward_"):
            original = trainer.reward_model.compute_turn_reward
            bad_reward = {
                "reward_nan": float("nan"),
                "reward_inf": float("inf"),
                "reward_overflow": 1e100,
            }[failure]

            async def reward(turn, context):
                if context["user_query"] == "bad":
                    return SimpleNamespace(total_reward=bad_reward)
                return await original(turn, context)

            patch.setattr(trainer.reward_model, "compute_turn_reward", reward)
        elif failure in ("empty_group", "logprob_nan"):
            original_generate = trainer.generator.generate_group_responses

            async def generate(prompt, count):
                if prompt == "bad":
                    return [] if failure == "empty_group" else [("ok", float("nan"))]
                return await original_generate(prompt, count)

            patch.setattr(trainer.generator, "generate_group_responses", generate)
        elif failure.startswith("loss_"):
            name = (
                "compute_gspo_token_loss"
                if isinstance(trainer, GSPOTokenTrainer)
                else "compute_gspo_loss"
            )
            original_loss = getattr(trainer, name)

            def bad_loss(*args):
                return original_loss(*args) + float(failure.removeprefix("loss_"))

            patch.setattr(trainer, name, bad_loss)
        elif failure.startswith("max_norm_"):
            patch.setattr(
                trainer.config,
                "max_grad_norm",
                float(failure.removeprefix("max_norm_")),
            )
        else:

            def bad_gradient(gradient):
                if failure == "backward_exception":
                    raise RuntimeError("injected backward failure")
                value = {
                    "gradient_nan": float("nan"),
                    "gradient_inf": float("inf"),
                    "gradient_norm_overflow": 1e30,
                }[failure]
                return torch.full_like(gradient, value)

            hook = next(trainer.model.parameters()).register_hook(bad_gradient)

        try:
            with pytest.raises(
                (ValueError, RuntimeError), match="finite|nonempty|backward failure"
            ):
                await _train(trainer, ["good", "bad"])
        finally:
            if hook is not None:
                hook.remove()

    _assert_same(trainer.model.state_dict(), before_model)
    _assert_same(trainer.optimizer.state_dict(), before_optimizer)
    _assert_same(trainer.scheduler.state_dict(), before_scheduler)
    assert trainer.training_metrics == before_metrics
    assert all(parameter.grad is None for parameter in trainer.model.parameters())

    await _train(trainer, ["good"])
    assert trainer.scheduler.last_epoch == before_scheduler["last_epoch"] + 1
    assert len(trainer.training_metrics["policy_loss"]) == 2


@pytest.mark.parametrize(
    "rewards",
    [
        [],
        1.0,
        [[0.0, 1.0]],
        [0.0, float("nan")],
        [float("inf")],
        [float("-inf"), 0.0],
        [0.0, 1e100],
        [3e38, 3e38],
    ],
)
def test_group_advantages_reject_invalid_rewards_instead_of_hiding_them(rewards):
    with pytest.raises(ValueError, match="finite|nonempty"):
        GSPOTrainer.compute_group_advantages(None, rewards)


def test_nonfinite_estimator_result_is_rejected(monkeypatch):
    from stateset_agents.training import objectives

    monkeypatch.setattr(
        objectives,
        "compute_advantages",
        lambda *args: torch.tensor([float("nan"), 0.0]),
    )
    with pytest.raises(ValueError, match="advantages.*finite"):
        GSPOTrainer.compute_group_advantages(None, [0.0, 1.0])


def test_large_finite_rewards_with_representable_statistics_remain_supported():
    advantages, stats = GSPOTrainer.compute_group_advantages(None, [-3e38, 3e38])
    assert torch.allclose(advantages, torch.tensor([-1.0, 1.0]))
    assert stats["mean_reward"] == 0.0
    assert stats["std_reward"] == pytest.approx(3e38)
