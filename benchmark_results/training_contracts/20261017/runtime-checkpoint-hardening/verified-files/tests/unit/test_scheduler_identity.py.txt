"""Future LR behavior must be bound even when state_dict values are identical."""

import functools
import json

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.scheduler_identity import (  # noqa: E402
    callable_identity,
    scheduler_identity,
)

GLOBAL_DECAY = 0.9


def _global_decay(step):
    return GLOBAL_DECAY**step


def _closure(gamma):
    return lambda step: gamma**step


def _partial_decay(step, *, warmup):
    return 1.0 if step <= warmup else 0.5


def _scheduler(function):
    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    return torch.optim.lr_scheduler.LambdaLR(optimizer, function)


def test_equal_saved_state_can_hide_different_future_learning_rates():
    first, second = _scheduler(_closure(0.5)), _scheduler(_closure(0.9))
    assert first.state_dict() == second.state_dict()
    assert scheduler_identity(first) != scheduler_identity(second)
    for scheduler in (first, second):
        scheduler.optimizer.step()
        scheduler.step()
    assert first.get_last_lr() != second.get_last_lr()


def test_partial_arguments_and_mutable_global_bindings_are_captured(monkeypatch):
    first = _scheduler(functools.partial(_partial_decay, warmup=2))
    second = _scheduler(functools.partial(_partial_decay, warmup=3))
    assert first.state_dict() == second.state_dict()
    assert scheduler_identity(first) != scheduler_identity(second)
    identity = callable_identity(_global_decay)
    monkeypatch.setitem(globals(), "GLOBAL_DECAY", 0.5)
    assert callable_identity(_global_decay) != identity


def test_relocated_code_is_the_same_implementation():
    functions = []
    for filename, prefix in [
        ("/node-a/source.py", ""),
        ("/node-b/source.py", "\n\n\n"),
    ]:
        namespace = {}
        exec(
            compile(
                prefix + "def decay(step):\n    return 0.9 ** step\n", filename, "exec"
            ),
            namespace,
        )
        functions.append(namespace["decay"])
    assert callable_identity(functions[0]) == callable_identity(functions[1])


def test_composite_schedulers_include_child_lambda_bindings():
    def composite(gamma):
        base = _scheduler(_closure(gamma))
        return torch.optim.lr_scheduler.ChainedScheduler(
            [base], optimizer=base.optimizer
        )

    first, second = composite(0.5), composite(0.9)
    assert first.state_dict() == second.state_dict()
    assert scheduler_identity(first) != scheduler_identity(second)
    json.dumps(scheduler_identity(first), allow_nan=False)


def test_transformers_linear_schedule_partial_is_supported():
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.1)
    scheduler = get_linear_schedule_with_warmup(optimizer, 2, 10)
    assert scheduler_identity(scheduler) == scheduler_identity(scheduler)


def test_opaque_callable_bindings_fail_closed():
    opaque = object()
    with pytest.raises(ValueError, match="Unsupported"):
        callable_identity(lambda step: opaque)


def test_callable_objects_bind_instance_and_class_values(monkeypatch):
    class Decay:
        multiplier = 0.9

        def __init__(self, gamma):
            self.gamma = gamma

        def __call__(self, step):
            return self.multiplier * self.gamma**step

    identity = callable_identity(Decay(0.5))
    assert callable_identity(Decay(0.5)) == identity
    assert callable_identity(Decay(0.8)) != identity
    monkeypatch.setattr(Decay, "multiplier", 0.8)
    assert callable_identity(Decay(0.5)) != identity


def test_cyclic_callable_bindings_reject():
    values = []
    values.append(values)
    with pytest.raises(ValueError, match="Cyclic"):
        callable_identity(lambda step: values)
