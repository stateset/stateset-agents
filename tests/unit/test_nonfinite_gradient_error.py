"""Non-finite gradients raise a typed ``NonFiniteGradientError``.

MultiTurn GRPO decided whether to skip an update by matching the message
string ``"gradients must be finite"`` of the spine's ``ValueError``: any
``ValueError`` that happened to carry that text was skipped, and rewording
the spine's message would have silently disabled ``nonfinite_grad_policy=
"skip"``. The spine now raises :class:`NonFiniteGradientError`, a
``TrainingError`` that is still a ``ValueError`` (``except ValueError``
handlers and the message are unchanged), and the trainer catches the type.
"""

from __future__ import annotations

import pickle

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.core.errors import ErrorCode, TrainingError  # noqa: E402
from stateset_agents.training import multi_turn_trainer  # noqa: E402
from stateset_agents.training.spine.update import (  # noqa: E402
    NonFiniteGradientError,
    OptimizerStep,
)
from tests.unit.test_rollout_sync import (  # noqa: E402
    _fake_grad,
    _model,
    _multi_turn_trainer,
)


def test_is_a_training_error_and_still_a_value_error() -> None:
    error = NonFiniteGradientError("GRPO gradients must be finite")
    assert isinstance(error, TrainingError)
    assert isinstance(error, ValueError)
    assert str(error) == "GRPO gradients must be finite"
    assert error.code is ErrorCode.TRN_GRADIENT_EXPLODED
    restored = pickle.loads(pickle.dumps(error))
    assert type(restored) is NonFiniteGradientError


def test_the_spine_commit_raises_the_typed_error() -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, float("nan"))
    owner = type("Owner", (), {"global_step": 0})()
    step = OptimizerStep(owner, torch, optimizers=[optimizer], max_grad_norm=1.0)
    with pytest.raises(NonFiniteGradientError, match="Policy gradients must be finite"):
        step()


def _skipping_trainer() -> object:
    agent = type("Agent", (), {})()
    agent.model = _model()
    agent.tokenizer = None
    return _multi_turn_trainer(agent, nonfinite_grad_policy="skip")


def test_skip_policy_skips_the_typed_error(monkeypatch: pytest.MonkeyPatch) -> None:
    trainer = _skipping_trainer()

    def nonfinite(*args: object, **kwargs: object) -> bool:
        raise NonFiniteGradientError("GRPO gradients must be finite")

    monkeypatch.setattr(multi_turn_trainer, "safe_optimizer_step", nonfinite)
    _fake_grad(trainer.agent.model)
    assert trainer._apply_optimizer_step(torch) is False
    assert trainer.nonfinite_skips == 1


def test_an_untyped_value_error_with_the_same_text_is_not_skipped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Control flow no longer depends on the message string."""
    trainer = _skipping_trainer()

    def unrelated(*args: object, **kwargs: object) -> bool:
        raise ValueError("reward gradients must be finite (user check)")

    monkeypatch.setattr(multi_turn_trainer, "safe_optimizer_step", unrelated)
    _fake_grad(trainer.agent.model)
    with pytest.raises(ValueError, match="user check"):
        trainer._apply_optimizer_step(torch)
    assert trainer.nonfinite_skips == 0


def test_too_many_consecutive_skips_raise_the_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trainer = _skipping_trainer()
    trainer.config.max_consecutive_nonfinite_skips = 1

    def nonfinite(*args: object, **kwargs: object) -> bool:
        raise NonFiniteGradientError("GRPO gradients must be finite")

    monkeypatch.setattr(multi_turn_trainer, "safe_optimizer_step", nonfinite)
    assert trainer._apply_optimizer_step(torch) is False
    with pytest.raises(NonFiniteGradientError, match="consecutive non-finite"):
        trainer._apply_optimizer_step(torch)
