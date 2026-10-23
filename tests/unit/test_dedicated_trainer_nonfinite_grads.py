"""DAPO, GEPO and VAPO never step on non-finite gradients.

Before the unified RL spine's ``OptimizerStep`` these trainers clipped and
stepped whatever gradients a pass produced, so one NaN/Inf gradient silently
turned the weights (and AdamW state) into NaN. They now follow the shared
``nonfinite_grad_policy``: ``"raise"`` (the default) fails the step with every
weight, optimizer and scheduler untouched; ``"skip"`` discards that pass's
update, logs it and keeps training (``global_step`` still counts the
``train_step`` call).
"""

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers import GPT2Config, GPT2LMHeadModel  # noqa: E402

from stateset_agents.training.dapo_trainer import DAPOConfig, DAPOTrainer  # noqa: E402
from stateset_agents.training.gepo_trainer import GEPOConfig, GEPOTrainer  # noqa: E402
from stateset_agents.training.vapo_trainer import VAPOConfig, VAPOTrainer  # noqa: E402

VOCAB = 64
PROMPT = [5, 6, 7, 8]
COMPLETIONS = [("1a", [17, 33]), ("ab", [33, 34]), ("9", [25])]


def _model() -> Any:
    torch.manual_seed(0)
    return GPT2LMHeadModel(
        GPT2Config(
            n_embd=16,
            n_layer=1,
            n_head=2,
            vocab_size=VOCAB,
            n_positions=32,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    )


def _reward(prompt: str, response: str) -> float:
    return sum(ch.isdigit() for ch in response) / max(len(response), 1)


def _token_rollouts(prompt: str) -> list[dict[str, Any]]:
    out = []
    for text, completion in COMPLETIONS:
        ids = torch.tensor(PROMPT + completion)
        mask = torch.zeros(len(ids))
        mask[len(PROMPT) :] = 1.0
        out.append(
            {
                "response": text,
                "input_ids": ids,
                "attention_mask": torch.ones(len(ids), dtype=torch.long),
                "response_mask": mask,
                "sequence_length": len(completion),
            }
        )
    return out


def _gepo_rollouts(prompt: str, group_size: int) -> list[dict[str, Any]]:
    width = len(PROMPT) + max(len(c) for _, c in COMPLETIONS)
    out = []
    for text, completion in COMPLETIONS:
        ids = PROMPT + completion
        pad = width - len(ids)
        out.append(
            {
                "response": text,
                "input_ids": torch.tensor(ids + [0] * pad),
                "attention_mask": torch.tensor([1] * len(ids) + [0] * pad),
                "response_start_idx": len(PROMPT),
            }
        )
    return out


def _dapo(**config: Any) -> Any:
    trainer = DAPOTrainer(
        config=DAPOConfig(
            model_name="gpt2", group_size=3, bf16=False, report_to="none", **config
        ),
        model=_model(),
        tokenizer=None,
        reward_fn=_reward,
    )

    async def generate(prompt: str) -> list[dict[str, Any]]:
        return _token_rollouts(prompt)

    trainer.generate_group_responses = generate
    return trainer


def _gepo(**config: Any) -> Any:
    trainer = GEPOTrainer(
        config=GEPOConfig(
            model_name="gpt2",
            group_size=3,
            warmup_ratio=0.0,
            bf16=False,
            report_to="none",
            **config,
        ),
        model=_model(),
        tokenizer=None,
        reward_fn=_reward,
    )

    async def generate(prompt: str, group_size: int) -> list[dict[str, Any]]:
        return _gepo_rollouts(prompt, group_size)

    trainer.generate_group_responses = generate
    return trainer


def _vapo(**config: Any) -> Any:
    trainer = VAPOTrainer(
        config=VAPOConfig(
            model_name="gpt2",
            group_size=3,
            per_device_train_batch_size=1,
            warmup_ratio=0.0,
            bf16=False,
            report_to="none",
            **config,
        ),
        model=_model(),
        tokenizer=None,
        reward_fn=_reward,
    )
    trainer.value_warmup_complete = True

    async def generate(prompt: str) -> list[dict[str, Any]]:
        return _token_rollouts(prompt)

    trainer.generate_group_responses = generate
    return trainer


FACTORIES: dict[str, Callable[..., Any]] = {
    "dapo": _dapo,
    "gepo": _gepo,
    "vapo": _vapo,
}


def _optimizers(trainer: Any) -> list[Any]:
    if isinstance(trainer, VAPOTrainer):
        return [trainer.actor_optimizer, trainer.critic_optimizer]
    return [trainer.optimizer]


def _schedulers(trainer: Any) -> list[Any]:
    if isinstance(trainer, VAPOTrainer):
        return [trainer.actor_scheduler, trainer.critic_scheduler]
    return [trainer.scheduler]


def _modules(trainer: Any) -> list[Any]:
    if isinstance(trainer, VAPOTrainer):
        return [trainer.model, trainer.value_head]
    return [trainer.model]


def _snapshot(trainer: Any) -> dict[str, Any]:
    return {
        "weights": [copy.deepcopy(m.state_dict()) for m in _modules(trainer)],
        "optimizers": [copy.deepcopy(o.state_dict()) for o in _optimizers(trainer)],
        "schedulers": [s.last_epoch for s in _schedulers(trainer)],
    }


def _assert_unchanged(trainer: Any, before: dict[str, Any]) -> None:
    after = _snapshot(trainer)
    for want, got in zip(before["weights"], after["weights"], strict=True):
        for key in want:
            assert torch.equal(want[key], got[key]), key
    assert [len(o["state"]) for o in after["optimizers"]] == [
        len(o["state"]) for o in before["optimizers"]
    ]
    assert after["schedulers"] == before["schedulers"]


def _poison(module: Any, *, after_backwards: int = 0) -> Any:
    """Make one parameter's gradient NaN on every backward after the first
    ``after_backwards`` ones."""
    parameter = next(module.parameters())
    seen = {"backwards": 0}

    def hook(gradient: Any) -> Any:
        seen["backwards"] += 1
        if seen["backwards"] > after_backwards:
            return torch.full_like(gradient, float("nan"))
        return gradient

    return parameter.register_hook(hook)


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(FACTORIES))
async def test_default_policy_raises_without_touching_state(name: str) -> None:
    trainer = FACTORIES[name]()
    before = _snapshot(trainer)
    _poison(trainer.model)
    with pytest.raises(ValueError, match="gradients must be finite"):
        await trainer.train_step(["q"])
    _assert_unchanged(trainer, before)
    assert trainer.global_step == 0
    for module in _modules(trainer):
        assert all(p.grad is None for p in module.parameters())
    for module in _modules(trainer):
        assert all(bool(torch.isfinite(p).all()) for p in module.parameters())


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(FACTORIES))
async def test_skip_policy_discards_every_nonfinite_pass(name: str) -> None:
    trainer = FACTORIES[name](nonfinite_grad_policy="skip", num_gradient_updates=2)
    before = _snapshot(trainer)
    _poison(trainer.model)
    metrics = await trainer.train_step(["q"])
    _assert_unchanged(trainer, before)
    assert metrics["optimizer_updates"] == 0
    assert trainer._nonfinite_skips == 2
    # global_step counts train_step calls, as before.
    assert trainer.global_step == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(FACTORIES))
async def test_skip_policy_keeps_the_finite_pass(name: str) -> None:
    trainer = FACTORIES[name](nonfinite_grad_policy="skip", num_gradient_updates=2)
    before = _snapshot(trainer)
    # One backward per pass for DAPO/GEPO/VAPO with a single prompt group.
    _poison(trainer.model, after_backwards=1)
    metrics = await trainer.train_step(["q"])
    assert metrics["optimizer_updates"] == 1
    assert trainer._nonfinite_skips == 1
    assert [s.last_epoch for s in _schedulers(trainer)] == [
        epoch + 1 for epoch in before["schedulers"]
    ]
    for module in _modules(trainer):
        assert all(bool(torch.isfinite(p).all()) for p in module.parameters())


@pytest.mark.asyncio
async def test_vapo_critic_nonfinite_gradient_skips_the_actor_too() -> None:
    trainer = _vapo(nonfinite_grad_policy="skip")
    before = _snapshot(trainer)
    _poison(trainer.value_head)
    metrics = await trainer.train_step(["q"])
    assert metrics["optimizer_updates"] == 0
    _assert_unchanged(trainer, before)


@pytest.mark.asyncio
async def test_vapo_value_warmup_rejects_nonfinite_critic_gradients() -> None:
    trainer = _vapo(value_warmup_steps=1)
    trainer.value_warmup_complete = False
    before = _snapshot(trainer)
    _poison(trainer.value_head)
    with pytest.raises(ValueError, match="gradients must be finite"):
        await trainer.train_step(["q"])
    _assert_unchanged(trainer, before)
    assert trainer.value_warmup_complete is False


@pytest.mark.asyncio
@pytest.mark.parametrize("name", sorted(FACTORIES))
async def test_finite_steps_report_their_updates(name: str) -> None:
    trainer = FACTORIES[name](num_gradient_updates=2)
    metrics = await trainer.train_step(["q"])
    assert metrics["optimizer_updates"] == 2
    assert trainer.global_step == 1
