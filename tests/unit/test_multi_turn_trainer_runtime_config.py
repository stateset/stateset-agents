"""MultiTurnGRPOTrainer runtime wiring: chunked token-path backward inside
``training_step``, the schedule built by ``initialize``, gradient
checkpointing, ignored-config warnings, and fp16-only loss scaling."""

from __future__ import annotations

import asyncio
import logging
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.training import multi_turn_trainer as mtt  # noqa: E402
from stateset_agents.training.config import TrainingConfig  # noqa: E402
from tests.unit.test_token_path_chunked_backward import (  # noqa: E402
    _config,
    _grads,
    _groups,
    _max_abs_diff,
    _single_backward_reference,
    _tiny_model,
)


def _training_config(**over):
    base = {
        "num_episodes": 8,
        "gradient_accumulation_steps": 2,
        "generation_batch_size": 2,
        "entropy_coef": 0.05,
        "bf16": False,
        "fp16": False,
        "report_to": "none",
        "reward_clip": None,
        "max_completion_length": 16,
        "seed": None,
        "lr_scheduler_type": "constant",
    }
    base.update(over)
    config = TrainingConfig(**base)
    return config


def _trainer(config, model=None):
    model = model or _tiny_model(0)
    agent = SimpleNamespace(model=model, tokenizer=None)
    trainer = mtt.MultiTurnGRPOTrainer(agent, environment=object(), config=config)
    asyncio.run(trainer.initialize())
    return trainer


@pytest.mark.parametrize("objective", [None, "gspo", "dapo"])
def test_training_step_backpropagates_chunks_scaled_for_accumulation(objective):
    config = _training_config(objective=objective)
    trainer = _trainer(config)
    groups = _groups()
    metrics = asyncio.run(trainer.training_step(groups))
    assert metrics["path"] == "token"
    assert metrics["backward_done"] is True
    assert metrics["optimizer_step"] is False  # first of two accumulation steps

    reference = _tiny_model(0)
    ref_config = _config(
        objective=objective, entropy_coef=0.05, generation_batch_size=2
    )
    want_loss, _ = _single_backward_reference(
        groups, ref_config, reference, beta=0.0, ref=None, old_logprobs=None
    )
    want = [g / 2 for g in _grads(reference)]
    assert _max_abs_diff(_grads(trainer.agent.model), want) < 1e-10
    assert metrics["total_loss"] == pytest.approx(float(want_loss), abs=1e-10)


def test_training_step_backpropagates_the_ewc_penalty_separately():
    config = _training_config()
    trainer = _trainer(config)
    model = trainer.agent.model
    first = next(model.parameters())
    trainer.continual_manager = SimpleNamespace(
        reference_model=None,
        get_effective_beta=lambda beta: beta,
        compute_ewc_penalty=lambda agent: first.square().sum(),
    )
    groups = _groups()
    asyncio.run(trainer.training_step(groups))

    reference = _tiny_model(0)
    _single_backward_reference(
        groups,
        _config(entropy_coef=0.05, generation_batch_size=2),
        reference,
        beta=0.0,
        ref=None,
        old_logprobs=None,
    )
    ref_first = next(reference.parameters())
    ref_first.square().sum().backward()
    want = [g / 2 for g in _grads(reference)]
    assert _max_abs_diff(_grads(model), want) < 1e-10


def test_inner_updates_use_chunked_backward_and_step():
    config = _training_config(num_gradient_updates=2, gradient_accumulation_steps=1)
    trainer = _trainer(config)
    before = [p.detach().clone() for p in trainer.agent.model.parameters()]
    metrics = asyncio.run(trainer.training_step(_groups()))
    assert metrics["backward_done"] is True
    assert metrics["optimizer_updates"] == 2
    after = list(trainer.agent.model.parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))


def test_direct_loss_calls_outside_training_step_keep_a_differentiable_loss():
    trainer = _trainer(_training_config())
    out = trainer.compute_grpo_loss(_groups())
    assert out["backward_done"] is False
    assert out["total_loss"].requires_grad


# --- scheduler ------------------------------------------------------------------


def test_initialize_builds_the_warmup_cosine_schedule_for_manual_loops():
    config = _training_config(
        lr_scheduler_type="cosine",
        warmup_ratio=0.25,
        num_episodes=16,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,
    )
    trainer = _trainer(config)
    assert trainer.lr_scheduler is not None
    # 16 planned optimizer steps, 4 warmup steps: lr starts at 0 and ramps.
    assert trainer.optimizer.param_groups[0]["lr"] == 0.0
    metrics = asyncio.run(trainer.training_step(_groups()))
    assert metrics["optimizer_step"] is True
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(1e-3 / 4)


def test_initialize_keeps_an_existing_scheduler():
    config = _training_config(lr_scheduler_type="linear")
    model = _tiny_model(0)
    agent = SimpleNamespace(model=model, tokenizer=None)
    trainer = mtt.MultiTurnGRPOTrainer(agent, environment=object(), config=config)
    sentinel = object()
    trainer.lr_scheduler = sentinel
    asyncio.run(trainer.initialize())
    assert trainer.lr_scheduler is sentinel


def test_constant_schedule_leaves_no_scheduler():
    trainer = _trainer(_training_config(lr_scheduler_type="constant"))
    assert trainer.lr_scheduler is None


# --- gradient checkpointing ------------------------------------------------------


def test_gradient_checkpointing_is_enabled_non_reentrant():
    trainer = _trainer(_training_config(gradient_checkpointing=True))
    model = trainer.agent.model
    assert model.is_gradient_checkpointing
    # the step still trains (non-reentrant checkpointing with chunked backward)
    asyncio.run(trainer.training_step(_groups()))
    assert any(p.grad is not None for p in model.parameters())


def test_gradient_checkpointing_off_by_default():
    trainer = _trainer(_training_config())
    assert not trainer.agent.model.is_gradient_checkpointing


class _FakePeftModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)
        self.peft_config = {"default": object()}
        self.enable_calls = []
        self.require_grads = 0

    def gradient_checkpointing_enable(self, **kwargs):
        self.enable_calls.append(kwargs)

    def enable_input_require_grads(self):
        self.require_grads += 1


def test_peft_policy_gets_input_require_grads():
    trainer = mtt.MultiTurnGRPOTrainer(
        SimpleNamespace(model=_FakePeftModel(), tokenizer=None),
        environment=object(),
        config=_training_config(gradient_checkpointing=True),
    )
    assert trainer._apply_gradient_checkpointing() is True
    model = trainer.agent.model
    assert model.enable_calls == [
        {"gradient_checkpointing_kwargs": {"use_reentrant": False}}
    ]
    assert model.require_grads == 1


def test_model_without_checkpointing_support_warns(caplog):
    trainer = mtt.MultiTurnGRPOTrainer(
        SimpleNamespace(model=torch.nn.Linear(2, 2), tokenizer=None),
        environment=object(),
        config=_training_config(gradient_checkpointing=True),
    )
    with caplog.at_level(logging.WARNING, logger=mtt.logger.name):
        assert trainer._apply_gradient_checkpointing() is False
    assert "does not support it" in caplog.text


# --- ignored config fields ---------------------------------------------------------


def test_ignored_config_fields_warn_once(caplog):
    config = _training_config(num_epochs=3, lora_r=32, temperature=0.2)
    model = _tiny_model(0)
    agent = SimpleNamespace(model=model, tokenizer=None)
    trainer = mtt.MultiTurnGRPOTrainer(agent, environment=object(), config=config)
    with caplog.at_level(logging.WARNING, logger=mtt.logger.name):
        asyncio.run(trainer.initialize())
        asyncio.run(trainer.initialize())
    warnings = [r for r in caplog.records if "ignores TrainingConfig" in r.message]
    assert len(warnings) == 1
    for name in ("num_epochs", "lora_r", "temperature"):
        assert name in warnings[0].getMessage()
    assert "gradient_checkpointing" not in warnings[0].getMessage()


def test_default_config_does_not_warn_about_ignored_fields(caplog):
    with caplog.at_level(logging.WARNING, logger=mtt.logger.name):
        _trainer(_training_config())
    assert "ignores TrainingConfig" not in caplog.text


def test_non_dataclass_configs_are_not_inspected():
    trainer = mtt.MultiTurnGRPOTrainer(
        SimpleNamespace(model=None), environment=object(), config=MagicMock()
    )
    assert trainer._warn_ignored_config_fields() == []


# --- loss scaling -------------------------------------------------------------------


@pytest.mark.parametrize(
    ("bf16", "fp16", "expect_scaler"),
    [(True, False, False), (False, True, True), (False, False, False)],
)
def test_grad_scaler_only_for_fp16(monkeypatch, bf16, fp16, expect_scaler):
    created = []

    class _Scaler:
        def __init__(self, device):
            created.append(device)

    fake_amp = SimpleNamespace(GradScaler=_Scaler, autocast=torch.amp.autocast)
    monkeypatch.setattr(mtt, "get_amp", lambda: fake_amp)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    trainer = _trainer(_training_config(bf16=bf16, fp16=fp16))
    assert (trainer.scaler is not None) is expect_scaler
    assert created == (["cuda"] if expect_scaler else [])
