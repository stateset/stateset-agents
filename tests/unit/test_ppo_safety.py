"""PPO must reject fabricated runs and invalid updates before optimizer commits."""

import copy
import math
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental import ppo  # noqa: E402


class TinyPolicy(torch.nn.Module):
    """Small causal-shaped policy; no network, model download or HF dependency."""

    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=4)
        self.embedding = torch.nn.Embedding(32, 4)
        self.projection = torch.nn.Linear(4, 32)
        self.nonfinite_logits = False

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        hidden = self.embedding(input_ids)
        logits = self.projection(hidden)
        if self.nonfinite_logits:
            logits = logits * float("nan")
        return SimpleNamespace(logits=logits, hidden_states=(hidden,))


class CharacterTokenizer:
    """Deterministic characters give unambiguous boundaries on either pad side."""

    padding_side = "right"

    def __call__(self, texts, *, return_tensors, padding, truncation, max_length):
        rows = [[1 + ord(char) % 31 for char in text][:max_length] for text in texts]
        width = max(map(len, rows))
        ids, masks = [], []
        for row in rows:
            pad = width - len(row)
            if self.padding_side == "left":
                ids.append([0] * pad + row)
                masks.append([0] * pad + [1] * len(row))
            else:
                ids.append(row + [0] * pad)
                masks.append([1] * len(row) + [0] * pad)
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }


@pytest.fixture
def trainer(monkeypatch):
    torch.manual_seed(17)
    monkeypatch.setattr(ppo, "_load_transformers_ppo", lambda: True)
    monkeypatch.setattr(ppo, "get_cosine_schedule_with_warmup", None)
    return ppo.PPOTrainer(
        config=ppo.PPOConfig(
            num_episodes=2,
            num_ppo_epochs=1,
            max_prompt_length=4,
            max_completion_length=4,
            value_hidden_size=4,
            report_to="none",
            entropy_coef=0.0,
        ),
        model=TinyPolicy(),
        tokenizer=CharacterTokenizer(),
        reward_fn=Mock(side_effect=AssertionError("unexpected reward callback")),
    )


def snapshot(trainer):
    return copy.deepcopy(
        {
            "policy": trainer.model.state_dict(),
            "critic": trainer.value_head.state_dict(),
            "policy_optimizer": trainer.policy_optimizer.state_dict(),
            "value_optimizer": trainer.value_optimizer.state_dict(),
            "policy_scheduler": trainer.policy_scheduler.state_dict(),
            "value_scheduler": trainer.value_scheduler.state_dict(),
            "global_step": trainer.global_step,
            "metrics_history": trainer.metrics_history,
            "kl_controller": (
                None if trainer.kl_controller is None else vars(trainer.kl_controller)
            ),
        }
    )


def assert_same(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            assert_same(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for got, want in zip(actual, expected, strict=True):
            assert_same(got, want)
    else:
        assert actual == expected


@pytest.mark.parametrize("episodes", [None, 0, 2])
async def test_high_level_train_rejects_before_callbacks_or_updates(trainer, episodes):
    before = snapshot(trainer)
    trainer.train_step = AsyncMock(side_effect=AssertionError("unexpected update"))
    with pytest.raises(ValueError, match="max_prompt_length|total update budget"):
        await trainer.train(["prompt"], num_episodes=episodes)
    trainer.reward_fn.assert_not_called()
    trainer.train_step.assert_not_awaited()
    assert_same(snapshot(trainer), before)


async def test_convenience_train_rejects_before_model_loading(monkeypatch):
    load = Mock(side_effect=AssertionError("unexpected model stack import"))
    reward = Mock(side_effect=AssertionError("unexpected reward callback"))
    monkeypatch.setattr(ppo, "_load_transformers_ppo", load)
    with pytest.raises(ValueError, match="top_p=1"):
        await ppo.train_ppo("unused/no-download", ["prompt"], reward, top_p=0.9)
    load.assert_not_called()
    reward.assert_not_called()


@pytest.mark.parametrize(
    "prompts,completions,rewards,match",
    [
        ([], [], [], "equal nonempty"),
        (["a"], [], [1.0], "equal nonempty"),
        (["a"], ["b"], [], "equal nonempty"),
        (["a"], ["b"], [float("nan")], "finite numbers"),
        (["a"], ["b"], [float("inf")], "finite numbers"),
        (["a"], ["b"], [-float("inf")], "finite numbers"),
        (["a"], ["b"], [True], "finite numbers"),
        (["a"], ["b"], [1e100], "learner dtype"),
        (["a"], [""], [1.0], "response action"),
    ],
)
async def test_invalid_batch_preserves_learner(
    trainer, prompts, completions, rewards, match
):
    before = snapshot(trainer)
    trainer.model.eval()
    trainer.value_head.eval()
    with pytest.raises(ValueError, match=match):
        await trainer.train_step(prompts, completions, rewards)
    assert_same(snapshot(trainer), before)
    assert not trainer.model.training
    assert not trainer.value_head.training


@pytest.mark.parametrize("side", ["left", "right"])
async def test_long_prompt_is_rejected_before_forward_or_update(trainer, side):
    trainer.tokenizer.padding_side = side
    before = snapshot(trainer)
    trainer.model.forward = Mock(side_effect=AssertionError("unexpected forward"))
    with pytest.raises(ValueError, match="prompts exceed max_prompt_length"):
        await trainer.train_step(["abcde", "a"], ["f", "bc"], [1.0, 0.0])
    trainer.model.forward.assert_not_called()
    assert_same(snapshot(trainer), before)


async def test_nonfinite_loss_cannot_commit_either_optimizer(trainer):
    before = snapshot(trainer)
    trainer.model.nonfinite_logits = True
    with pytest.raises(FloatingPointError, match="loss must be a finite scalar"):
        await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    assert_same(snapshot(trainer), before)


@pytest.mark.parametrize("component", ["policy", "critic"])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf")])
async def test_invalid_gradient_cannot_commit_either_optimizer(
    trainer, component, invalid
):
    before = snapshot(trainer)
    target = trainer.model if component == "policy" else trainer.value_head
    parameter = next(target.parameters())
    handle = parameter.register_hook(lambda grad: torch.full_like(grad, invalid))
    try:
        with pytest.raises(FloatingPointError, match="gradients must remain finite"):
            await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    finally:
        handle.remove()
    assert_same(snapshot(trainer), before)
    assert all(parameter.grad is None for parameter in trainer.model.parameters())
    assert all(parameter.grad is None for parameter in trainer.value_head.parameters())


@pytest.mark.parametrize("side", ["left", "right"])
async def test_finite_external_batch_still_updates_policy_and_critic(trainer, side):
    trainer.tokenizer.padding_side = side
    before = snapshot(trainer)
    metrics = await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    assert trainer.global_step == 1
    assert metrics["average_reward"] == 0.5
    assert all(
        math.isfinite(metrics[key]) for key in ("policy_loss", "value_loss", "entropy")
    )
    assert any(
        not torch.equal(value, before["policy"][key])
        for key, value in trainer.model.state_dict().items()
    )
    assert any(
        not torch.equal(value, before["critic"][key])
        for key, value in trainer.value_head.state_dict().items()
    )
    assert trainer.policy_optimizer.state
    assert trainer.value_optimizer.state


@pytest.mark.parametrize("epochs", [0, -1, True])
async def test_invalid_inner_epoch_count_rejects_before_mutation(trainer, epochs):
    before = snapshot(trainer)
    trainer.config.num_ppo_epochs = epochs
    with pytest.raises(ValueError, match="num_ppo_epochs must be a positive integer"):
        await trainer.train_step(["a"], ["b"], [1.0])
    assert_same(snapshot(trainer), before)


def enable_adaptive_kl(trainer):
    """Use an identical frozen reference and a visibly changing KL coefficient."""
    trainer.config.beta = 0.2
    trainer.config.use_adaptive_kl = True
    trainer.config.target_kl = 0.01
    trainer.ref_model = copy.deepcopy(trainer.model).eval()
    trainer.ref_model.requires_grad_(False)
    trainer.kl_controller = ppo.AdaptiveKLController(
        init_kl_coef=0.2, target_kl=0.01, horizon=10
    )


async def test_positive_kl_coefficient_requires_reference_before_tokenizer_or_forward(
    trainer,
):
    trainer.config.beta = 0.2
    before = snapshot(trainer)
    trainer.tokenizer = Mock(side_effect=AssertionError("unexpected tokenization"))
    trainer.model.forward = Mock(side_effect=AssertionError("unexpected forward"))
    with pytest.raises(ValueError, match="beta > 0 requires a reference model"):
        await trainer.train_step(["a"], ["b"], [1.0])
    trainer.tokenizer.assert_not_called()
    trainer.model.forward.assert_not_called()
    assert_same(snapshot(trainer), before)


def test_adaptive_kl_proposal_is_pure_and_update_remains_supported():
    controller = ppo.AdaptiveKLController(init_kl_coef=0.2, target_kl=0.01, horizon=10)
    proposed = controller.propose(0.0, 1)
    assert proposed == pytest.approx(0.18)
    assert controller.kl_coef == 0.2
    assert controller.update(0.0, 1) == proposed
    assert controller.kl_coef == proposed


@pytest.mark.parametrize("rejection", ["loss", "policy_gradient", "critic_gradient"])
async def test_adaptive_kl_rejected_update_preserves_controller(
    trainer, monkeypatch, rejection
):
    enable_adaptive_kl(trainer)
    before = snapshot(trainer)
    handle = None
    if rejection == "loss":
        original = trainer.value_loss

        def nonfinite_value_loss(*args, **kwargs):
            return original(*args, **kwargs) * float("nan")

        monkeypatch.setattr(trainer, "value_loss", nonfinite_value_loss)
        match = "loss must be a finite scalar"
    else:
        target = trainer.model if rejection == "policy_gradient" else trainer.value_head
        handle = next(target.parameters()).register_hook(
            lambda grad: torch.full_like(grad, float("nan"))
        )
        match = "gradients must remain finite"
    try:
        with pytest.raises(FloatingPointError, match=match):
            await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    finally:
        if handle is not None:
            handle.remove()
    assert_same(snapshot(trainer), before)


async def test_successful_inner_update_commits_adaptive_kl(trainer):
    enable_adaptive_kl(trainer)
    before = trainer.kl_controller.kl_coef
    expected = trainer.kl_controller.propose(0.0, 1)
    await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    assert trainer.kl_controller.kl_coef == pytest.approx(expected)
    assert trainer.kl_controller.kl_coef != before
    assert trainer.policy_optimizer.state
    assert trainer.value_optimizer.state
    assert trainer.global_step == 1


async def test_optimizer_exception_does_not_commit_adaptive_kl(trainer, monkeypatch):
    enable_adaptive_kl(trainer)
    before = trainer.kl_controller.kl_coef

    def fail_value_step():
        raise RuntimeError("injected value optimizer failure")

    monkeypatch.setattr(trainer.value_optimizer, "step", fail_value_step)
    with pytest.raises(RuntimeError, match="injected value optimizer failure"):
        await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    assert trainer.kl_controller.kl_coef == before
    # Policy AdamW already committed; delaying adaptation does not roll it back.
    assert trainer.policy_optimizer.state
    assert not trainer.value_optimizer.state
    assert trainer.global_step == 0
    assert trainer._runtime_checkpoint_poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        await trainer.train_step(["ab"], ["cd"], [1.0])
    with pytest.raises(RuntimeError, match="quarantined"):
        trainer.save_checkpoint("must-not-publish")


@pytest.mark.parametrize("failure", [False, True])
async def test_scoring_disables_dropout_preserves_gradients_and_restores_modes(
    trainer, monkeypatch, failure
):
    trainer.model.embedding = torch.nn.Sequential(
        trainer.model.embedding, torch.nn.Dropout(0.8)
    )
    trainer.model.train()
    trainer.model.projection.eval()  # Mixed child modes must survive the update.
    trainer.value_head.train()
    enable_adaptive_kl(trainer)
    trainer.ref_model.train()  # A caller may supply a reference in train mode.
    models = (trainer.model, trainer.value_head, trainer.ref_model)
    modes = {m: m.training for model in models for m in model.modules()}
    original_policy_loss = trainer.ppo_loss
    original_value_loss = trainer.value_loss
    calls = []

    def policy_loss(current, old, *args):
        assert all(not m.training for m in modes)
        assert current.requires_grad and not old.requires_grad
        torch.testing.assert_close(current, old, rtol=0, atol=0)
        calls.append("policy")
        return original_policy_loss(current, old, *args)

    def value_loss(current, old, *args):
        assert current.requires_grad and not old.requires_grad
        torch.testing.assert_close(current, old, rtol=0, atol=0)
        calls.append("critic")
        if failure:
            raise RuntimeError("injected scoring failure")
        return original_value_loss(current, old, *args)

    monkeypatch.setattr(trainer, "ppo_loss", policy_loss)
    monkeypatch.setattr(trainer, "value_loss", value_loss)
    if failure:
        with pytest.raises(RuntimeError, match="scoring failure"):
            await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
        assert not getattr(trainer, "_runtime_checkpoint_poisoned", False)
    else:
        await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
        assert next(trainer.model.parameters()).grad is not None
    assert calls == ["policy", "critic"]
    assert {m: m.training for m in modes} == modes


@pytest.mark.parametrize("side", ["left", "right"])
async def test_merged_prompt_boundary_rejected_before_forward(trainer, side):
    class MergingTokenizer(CharacterTokenizer):
        def __call__(self, texts, **kwargs):
            encoded = super().__call__(texts, **kwargs)
            for i, text in enumerate(texts):
                if text == "abcd":
                    # Same length, different tokenization at the join boundary.
                    positions = encoded["attention_mask"][i].nonzero().flatten()
                    encoded["input_ids"][i, positions[1]] = 30
            return encoded

    trainer.tokenizer = MergingTokenizer()
    trainer.tokenizer.padding_side = side
    trainer.model.forward = Mock(side_effect=AssertionError("unexpected forward"))
    before = snapshot(trainer)
    with pytest.raises(ValueError, match="token boundary"):
        await trainer.train_step(["ab", "a"], ["cd", "e"], [1.0, 0.0])
    assert_same(snapshot(trainer), before)
    trainer.model.forward.assert_not_called()


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("completion", ["bcdef", "bcdefghijklmnopqrstuvwxyz"])
async def test_truncated_completion_reward_rejected(trainer, side, completion):
    trainer.tokenizer.padding_side = side
    before = snapshot(trainer)
    with pytest.raises(ValueError, match="completions exceed max_completion_length"):
        await trainer.train_step(["a", "ab"], [completion, "c"], [1.0, 0.0])
    assert_same(snapshot(trainer), before)


async def test_failure_after_first_inner_epoch_quarantines_partial_batch(
    trainer, monkeypatch
):
    trainer.config.num_ppo_epochs = 2
    original = trainer.value_loss
    count = 0

    def fail_second(*args):
        nonlocal count
        count += 1
        loss = original(*args)
        return loss if count == 1 else loss * float("nan")

    monkeypatch.setattr(trainer, "value_loss", fail_second)
    with pytest.raises(FloatingPointError, match="finite scalar"):
        await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    assert trainer.policy_optimizer.state
    assert trainer.global_step == 0
    assert trainer._runtime_checkpoint_poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        await trainer.train_step(["ab"], ["cd"], [1.0])


async def test_complete_checkpoint_reload_recovers_partial_ppo_update(
    trainer, monkeypatch, tmp_path
):
    monkeypatch.setattr(
        trainer.model,
        "save_pretrained",
        lambda path: torch.save(
            trainer.model.state_dict(), Path(path) / "pytorch_model.bin"
        ),
        raising=False,
    )
    monkeypatch.setattr(
        trainer.tokenizer, "save_pretrained", lambda _: None, raising=False
    )
    await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    checkpoint = trainer.save_checkpoint(str(tmp_path / "complete"))
    before = snapshot(trainer)
    original = trainer.value_optimizer.step

    def fail_step():
        raise RuntimeError("injected partial commit")

    monkeypatch.setattr(trainer.value_optimizer, "step", fail_step)
    with pytest.raises(RuntimeError, match="partial commit"):
        await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    monkeypatch.setattr(trainer.value_optimizer, "step", original)
    assert trainer.load_runtime_checkpoint(checkpoint, trusted=True)
    assert not trainer._runtime_checkpoint_poisoned
    assert_same(snapshot(trainer), before)
    await trainer.train_step(["ab", "a"], ["cd", "ef"], [1.0, 0.0])
    assert trainer.global_step == 2
