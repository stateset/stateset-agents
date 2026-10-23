"""``spine.update.OptimizerStep``: the shared, validated optimizer commit."""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.spine import (  # noqa: E402
    NONFINITE_GRAD_POLICIES,
    STEP_COUNTERS,
    OptimizerStep,
)


def _setup(n: int = 1) -> tuple[SimpleNamespace, list[Any], list[Any], list[Any]]:
    torch.manual_seed(0)
    modules = [torch.nn.Linear(3, 2) for _ in range(n)]
    optimizers = [torch.optim.SGD(m.parameters(), lr=0.1) for m in modules]
    schedulers = [
        torch.optim.lr_scheduler.LambdaLR(o, lambda step: 0.5**step) for o in optimizers
    ]
    owner = SimpleNamespace(global_step=0)
    return owner, modules, optimizers, schedulers


def _grads(module: Any, value: float = 1.0) -> None:
    for parameter in module.parameters():
        parameter.grad = torch.full_like(parameter, value)


def _state(modules: list[Any]) -> list[Any]:
    return [p.detach().clone() for m in modules for p in m.parameters()]


def _same(left: list[Any], right: list[Any]) -> bool:
    return all(torch.equal(a, b) for a, b in zip(left, right, strict=True))


def test_policy_and_counter_vocabularies() -> None:
    assert STEP_COUNTERS == ("commits", "train_steps", "none")
    assert NONFINITE_GRAD_POLICIES == ("raise", "skip")
    owner, _, optimizers, _ = _setup()
    with pytest.raises(ValueError, match="counter"):
        OptimizerStep(owner, torch, optimizers=optimizers, max_grad_norm=1.0, counter="steps")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="nonfinite_grad_policy"):
        OptimizerStep(
            owner,
            torch,
            optimizers=optimizers,
            max_grad_norm=1.0,
            nonfinite_grad_policy="ignore",  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("counter", "per_pass", "per_train_step"),
    [("commits", 1, 0), ("train_steps", 0, 1), ("none", 0, 0)],
)
def test_step_counters(counter: str, per_pass: int, per_train_step: int) -> None:
    owner, (module,), optimizers, schedulers = _setup()
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        schedulers=schedulers,
        max_grad_norm=10.0,
        counter=counter,  # type: ignore[arg-type]
    )
    for _ in range(2):
        _grads(module)
        assert step() is True
    assert owner.global_step == 2 * per_pass
    step.end_train_step()
    assert owner.global_step == 2 * per_pass + per_train_step
    assert schedulers[0].last_epoch == 2


def test_counter_none_needs_no_global_step() -> None:
    _, (module,), optimizers, _ = _setup()
    owner = SimpleNamespace()
    step = OptimizerStep(owner, torch, optimizers=optimizers, max_grad_norm=1.0)
    _grads(module)
    assert step() is True
    step.end_train_step()
    assert not hasattr(owner, "global_step")


def test_commit_steps_every_optimizer_and_scheduler_exactly_once() -> None:
    owner, modules, optimizers, schedulers = _setup(2)
    calls: list[str] = []
    for name, optimizer in zip("ab", optimizers, strict=True):
        optimizer.register_step_pre_hook(lambda *args, _name=name: calls.append(_name))
    events: list[str] = []
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        schedulers=schedulers,
        max_grad_norm=10.0,
        on_commit=lambda: events.append(
            f"commit lr={optimizers[0].param_groups[0]['lr']}"
        ),
    )
    for module in modules:
        _grads(module)
    before = _state(modules)
    assert step() is True
    assert calls == ["a", "b"]
    assert [s.last_epoch for s in schedulers] == [1, 1]
    # on_commit runs after the schedulers advanced.
    assert events == ["commit lr=0.05"]
    assert not _same(before, _state(modules))


def test_clip_groups_are_clipped_independently() -> None:
    owner, modules, optimizers, _ = _setup(2)
    _grads(modules[0], 3.0)
    _grads(modules[1], 0.01)
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        max_grad_norm=1.0,
        clip_groups=[list(m.parameters()) for m in modules],
    )
    assert step() is True
    norms = [
        float(torch.cat([p.grad.reshape(-1) for p in m.parameters()]).norm())
        for m in modules
    ]
    assert norms[0] == pytest.approx(1.0, rel=1e-5)
    # The small group is not pulled down by the large one's norm.
    assert norms[1] == pytest.approx(0.01 * 8**0.5, rel=1e-5)


def test_clip_return_value_is_never_used(monkeypatch: pytest.MonkeyPatch) -> None:
    owner, (module,), optimizers, _ = _setup()
    seen: list[dict[str, Any]] = []
    monkeypatch.setattr(
        torch.nn.utils,
        "clip_grad_norm_",
        lambda *args, **kwargs: seen.append(kwargs),
    )
    _grads(module, 5.0)
    step = OptimizerStep(owner, torch, optimizers=optimizers, max_grad_norm=1.0)
    assert step() is True
    assert seen == [{"error_if_nonfinite": True}]


@pytest.mark.parametrize("clear", [False, True])
def test_gradients_after_commit(clear: bool) -> None:
    owner, (module,), optimizers, _ = _setup()
    _grads(module)
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        max_grad_norm=10.0,
        zero_grad_after_commit=clear,
    )
    assert step() is True
    assert all((p.grad is None) == clear for p in module.parameters())


def test_no_gradients_is_not_a_commit() -> None:
    owner, (module,), optimizers, schedulers = _setup()
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        schedulers=schedulers,
        max_grad_norm=1.0,
        counter="commits",
    )
    assert step() is False
    assert owner.global_step == 0
    assert schedulers[0].last_epoch == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_raise_policy_rejects_nonfinite_gradients_before_stepping(
    value: float,
) -> None:
    owner, modules, optimizers, schedulers = _setup(2)
    _grads(modules[0])
    _grads(modules[1], value)
    before = _state(modules)
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        schedulers=schedulers,
        max_grad_norm=1.0,
        counter="commits",
        label="DAPO",
    )
    with pytest.raises(ValueError, match="DAPO gradients must be finite"):
        step()
    assert _same(before, _state(modules))
    assert all(p.grad is None for m in modules for p in m.parameters())
    assert all(not o.state for o in optimizers)
    assert [s.last_epoch for s in schedulers] == [0, 0]
    assert owner.global_step == 0


def test_skip_policy_discards_logs_and_limits_consecutive_skips(
    caplog: pytest.LogCaptureFixture,
) -> None:
    owner, (module,), optimizers, schedulers = _setup()
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        schedulers=schedulers,
        max_grad_norm=1.0,
        counter="train_steps",
        nonfinite_grad_policy="skip",
        max_consecutive_nonfinite_skips=1,
        named_modules=[module],
        label="GEPO",
    )
    before = _state([module])
    _grads(module, float("nan"))
    with caplog.at_level(logging.WARNING):
        assert step() is False
    assert "GEPO" in caplog.text and "weight" in caplog.text
    assert _same(before, _state([module]))
    assert all(p.grad is None for p in module.parameters())
    assert schedulers[0].last_epoch == 0
    assert owner._nonfinite_skips == 1

    _grads(module)  # a finite commit resets the consecutive count
    assert step() is True
    assert owner._consecutive_nonfinite == 0

    _grads(module, float("inf"))
    assert step() is False
    _grads(module, float("nan"))
    with pytest.raises(ValueError, match="2 consecutive non-finite updates"):
        step()
    assert owner._nonfinite_skips == 3
    assert all(p.grad is None for p in module.parameters())


def test_total_norm_overflow_raises_even_when_skipping() -> None:
    owner, (module,), optimizers, _ = _setup()
    _grads(module, 1e30)
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        max_grad_norm=1.0,
        nonfinite_grad_policy="skip",
    )
    with pytest.raises(RuntimeError, match="non-finite"):
        step()
    assert all(p.grad is None for p in module.parameters())


def test_failed_optimizer_step_discards_gradients() -> None:
    owner, (module,), optimizers, schedulers = _setup()
    discarded: list[bool] = []

    def boom(*args: Any, **kwargs: Any) -> None:
        raise RuntimeError("injected step failure")

    optimizers[0].step = boom  # type: ignore[method-assign]
    _grads(module)
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        schedulers=schedulers,
        max_grad_norm=1.0,
        counter="commits",
        discard=lambda: discarded.append(True),
    )
    with pytest.raises(RuntimeError, match="injected"):
        step()
    assert discarded == [True]
    assert owner.global_step == 0
    assert schedulers[0].last_epoch == 0


@pytest.mark.parametrize(
    ("gradient_scale", "max_grad_norm", "match"),
    [
        (0.0, 1.0, "gradient_scale"),
        (float("nan"), 1.0, "gradient_scale"),
        (1.0, float("inf"), "max_grad_norm"),
        (1.0, -1.0, "max_grad_norm"),
    ],
)
def test_invalid_scale_or_norm_is_rejected(
    gradient_scale: float, max_grad_norm: float, match: str
) -> None:
    owner, (module,), optimizers, _ = _setup()
    _grads(module)
    step = OptimizerStep(
        owner, torch, optimizers=optimizers, max_grad_norm=max_grad_norm
    )
    with pytest.raises(ValueError, match=match):
        step(gradient_scale=gradient_scale)
    assert all(p.grad is None for p in module.parameters())


def test_gradient_scale_corrects_before_clipping() -> None:
    owner, (module,), optimizers, _ = _setup()
    _grads(module, 0.1)
    optimizers[0].step = lambda *a, **k: None  # type: ignore[method-assign]
    step = OptimizerStep(owner, torch, optimizers=optimizers, max_grad_norm=100.0)
    assert step(gradient_scale=4.0) is True
    assert all(
        torch.allclose(p.grad, torch.full_like(p, 0.4)) for p in module.parameters()
    )


def test_finalize_result_cancels_gradient_scale() -> None:
    owner, (module,), optimizers, _ = _setup()
    _grads(module, 0.1)
    optimizers[0].step = lambda *a, **k: None  # type: ignore[method-assign]
    step = OptimizerStep(
        owner, torch, optimizers=optimizers, max_grad_norm=100.0, finalize=lambda: True
    )
    assert step(gradient_scale=4.0) is True
    assert all(
        torch.allclose(p.grad, torch.full_like(p, 0.1)) for p in module.parameters()
    )


def test_grad_scaler_overflow_is_a_skip_not_an_error() -> None:
    owner, modules, optimizers, schedulers = _setup(2)
    scaler = torch.amp.GradScaler("cpu", init_scale=64.0)
    scaler.scale(torch.zeros(()))  # initialise the scale, as a backward would
    _grads(modules[0])
    _grads(modules[1], float("inf"))
    before = _state(modules)
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        schedulers=schedulers,
        scaler=scaler,
        max_grad_norm=1.0,
        counter="commits",
    )
    assert step() is False
    assert scaler.get_scale() < 64.0
    assert _same(before, _state(modules))
    assert owner.global_step == 0
    assert [s.last_epoch for s in schedulers] == [0, 0]


def test_grad_scaler_overflow_with_one_optimizer_uses_scaler_step() -> None:
    owner, (module,), optimizers, schedulers = _setup()
    scaler = torch.amp.GradScaler("cpu", init_scale=64.0)
    scaler.scale(torch.zeros(()))
    stepped: list[Any] = []
    original = scaler.step
    scaler.step = lambda optimizer: stepped.append(optimizer) or original(optimizer)  # type: ignore[method-assign]
    _grads(module, float("nan"))
    before = _state([module])
    step = OptimizerStep(
        owner,
        torch,
        optimizers=optimizers,
        schedulers=schedulers,
        scaler=scaler,
        max_grad_norm=1.0,
    )
    assert step() is False
    assert stepped == optimizers
    assert scaler.get_scale() < 64.0
    assert _same(before, _state([module]))
    assert schedulers[0].last_epoch == 0
