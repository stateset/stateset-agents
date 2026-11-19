"""Invalid data or backward results must not advance a native GSPO update."""

import copy
import math
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
    return build_trainer(request.param, monkeypatch)


def build_trainer(trainer_type, monkeypatch, *, dtype=torch.float32, **options):
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

    result = trainer_type(
        config=GSPOConfig(
            model_name="gpt2",
            num_generations=2,
            num_outer_iterations=100,
            warmup_ratio=0.0,
            max_prompt_length=16,
            max_completion_length=16,
            **options,
        ),
        model=model.to(dtype),
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


@pytest.mark.asyncio
async def test_large_finite_gradient_commits_expected_adam_moments(trainer):
    await _train(trainer, ["good"])
    p = next(trainer.model.parameters())
    before = copy.deepcopy(trainer.optimizer.state[p])
    epoch = trainer.scheduler.last_epoch
    handle = p.register_hook(lambda grad: torch.full_like(grad, 1e30))
    try:
        await _train(trainer, ["good", "bad"])
    finally:
        handle.remove()

    # All other gradients are negligible compared with this whole embedding.
    # Each coordinate takes an equal share of its group's L2 norm.
    g = torch.full_like(p, trainer.config.max_grad_norm / math.sqrt(p.numel()))
    beta1, beta2 = trainer.optimizer.param_groups[0]["betas"]
    after = trainer.optimizer.state[p]
    torch.testing.assert_close(
        after["exp_avg"], beta1 * before["exp_avg"] + (1 - beta1) * g
    )
    torch.testing.assert_close(
        after["exp_avg_sq"], beta2 * before["exp_avg_sq"] + (1 - beta2) * g.square()
    )
    assert after["step"].item() == before["step"].item() + 1
    assert trainer.scheduler.last_epoch == epoch + 1
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


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["token_mean", "seq_sum_const"])
async def test_population_override_matches_one_flat_batch(trainer, monkeypatch, mode):
    """Compare public updates against independent policy and row-KL sums."""
    groups = {"one": ["ok", ""], "two": ["ok", "bad", "longer"]}
    trainer.config.max_grad_norm = 1e6
    trainer.ref_model = copy.deepcopy(trainer.model).eval()
    with torch.no_grad():
        trainer.ref_model.transformer.wte.weight[
            trainer.tokenizer.convert_tokens_to_ids("o")
        ].add_(0.4)
    trainer._objective = trainer._objective.with_(
        aggregate=mode, advantage="batch_norm", kl_coef=0.7
    )

    async def generate(prompt, count):
        return [(answer, 0.0) for answer in groups[prompt]]

    monkeypatch.setattr(trainer.generator, "generate_group_responses", generate)
    reward = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0])
    advantage = (reward - torch.tensor([0.5, 0.5, 1 / 3, 1 / 3, 1 / 3])) / reward.std(
        correction=0
    )

    def scores(model, ids, start):
        # The character tokenizer places the first response action at start.
        logits = model(ids).logits[0, start - 1 : -1]
        return logits.log_softmax(-1).gather(-1, ids[0, start:, None]).flatten()

    current, reference = [], []
    for prompt, responses in groups.items():
        for response in responses:
            ids = trainer.tokenizer(
                prompt + response, return_tensors="pt", add_special_tokens=False
            )["input_ids"]

            current.append(scores(trainer.model, ids, len(prompt)))
            with torch.no_grad():
                reference.append(scores(trainer.ref_model, ids, len(prompt)))
    lengths = [row.numel() for row in current]
    denom = (
        sum(lengths)
        if mode == "token_mean"
        else len(lengths) * trainer.config.max_completion_length
    )
    policy = (
        sum(
            -a * (row - row.detach()).exp().sum()
            for a, row in zip(advantage, current, strict=True)
        )
        / denom
    )
    gaps = torch.stack(
        [
            (ref.sum() - row.sum()) / max(row.numel(), 1)
            for row, ref in zip(current, reference, strict=True)
        ]
    )
    expected = policy + 0.7 * (gaps.exp() - gaps - 1).mean()
    params = tuple(trainer.model.parameters())
    grads = torch.autograd.grad(expected, params)
    metrics = await _train(trainer, list(groups))
    assert metrics["policy_loss"] == pytest.approx(expected.item(), abs=3e-6)
    for param, grad in zip(params, grads, strict=True):
        torch.testing.assert_close(param.grad, grad, atol=3e-6, rtol=1e-4)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "clip,expected_fraction", [("none", 0), ("clipped", 0), ("cispo", 1)]
)
async def test_diagnostics_follow_the_resolved_ratio_and_clip(
    trainer, monkeypatch, clip, expected_fraction
):
    trainer.config.rescore_old_log_probs = False
    trainer._objective = trainer._objective.with_(
        ratio_clamp=0.1, clip=clip, clip_low=0.2, clip_high=0.2, is_cap=1.05
    )

    async def generate(prompt, count):
        return [("ok", -1000.0), ("nope", -1000.0)]

    monkeypatch.setattr(trainer.generator, "generate_group_responses", generate)
    metrics = await _train(trainer, ["hello"])
    assert metrics["sequence_importance_ratio"] == pytest.approx(math.exp(0.1))
    assert metrics["clipping_fraction"] == expected_fraction
