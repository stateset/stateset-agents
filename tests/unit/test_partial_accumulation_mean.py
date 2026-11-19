"""Final partial GRPO windows must match the mean of their actual batches."""

import copy
import math
import random
from decimal import Decimal, localcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.single_turn_trainer import SingleTurnGRPOTrainer
from stateset_agents.training.trainer_utils import (
    _clip_gradient_norm,
    backward_training_loss,
    safe_optimizer_step,
)


class TinyModel(torch.nn.Linear):
    def __init__(self):
        super().__init__(2, 1, bias=False, dtype=torch.float64)
        with torch.no_grad():
            self.weight.fill_(0.75)

    def save_pretrained(self, path):
        torch.save(self.state_dict(), path / "pytorch_model.bin")


def make_trainer(path, trainer_type, *, amp, max_norm):
    model = TinyModel()
    value = trainer_type(
        SimpleNamespace(model=model, tokenizer=None),
        object(),
        config=SimpleNamespace(
            output_dir=str(path),
            num_episodes=1,
            gradient_accumulation_steps=4,
            max_grad_norm=max_norm,
            fp16=amp,
        ),
    )
    value.optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    value.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        value.optimizer, lambda step: 0.9**step
    )
    value.scaler = (
        torch.amp.GradScaler("cpu", init_scale=64, growth_interval=2) if amp else None
    )
    value._setup_scheduler = MagicMock()
    value._get_training_scenarios = lambda: [{}]
    value._get_eval_scenarios = lambda: []
    return value


def batch_loss(model, values):
    return model(torch.tensor([values], dtype=torch.float64)).square().mean()


def accumulate(value, values):
    loss = (
        batch_loss(value.agent.model, values) / value.config.gradient_accumulation_steps
    )
    backward_training_loss(value, loss, torch, scaler=value.scaler)
    value._grad_accum_step += 1
    if value._grad_accum_step % value.config.gradient_accumulation_steps == 0:
        if isinstance(value, SingleTurnGRPOTrainer):
            value._apply_optimizer_step(
                torch, value.scaler is not None, value.config.max_grad_norm
            )
        else:
            value._apply_optimizer_step(torch)


def close(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right, rtol=1e-12, atol=1e-12)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            close(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            close(first, second)
    else:
        assert left == right


def _decimal_clipped(values, cap, dtype):
    with localcontext() as context:
        context.prec = 110
        coordinates = [Decimal.from_float(v) for v in values]
        norm = sum(v * v for v in coordinates).sqrt()
        scale = min(Decimal(1), Decimal.from_float(cap) / (norm + Decimal("0.000001")))
        return torch.tensor([float(v * scale) for v in coordinates], dtype=dtype)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "trainer_type",
    [SingleTurnGRPOTrainer, MultiTurnGRPOTrainer],
    ids=["single", "multi"],
)
@pytest.mark.parametrize("amp", [False, True])
@pytest.mark.parametrize("pending", [1, 2, 3])
@pytest.mark.parametrize("max_norm", [10.0, 0.15])
async def test_completed_resume_flush_matches_direct_partial_mean(
    tmp_path, trainer_type, amp, pending, max_norm
):
    original = make_trainer(tmp_path, trainer_type, amp=amp, max_norm=max_norm)
    # Warm optimizer momentum, scheduler, and AMP growth state with a full window.
    for values in ([0.1, 0.5], [0.2, 0.4], [0.3, 0.7], [0.4, 0.2]):
        accumulate(original, values)
    assert original.global_step == 1
    reference = copy.deepcopy(original.agent.model)
    optimizer = torch.optim.SGD(reference.parameters(), lr=0.1, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 0.9**step)
    optimizer.load_state_dict(copy.deepcopy(original.optimizer.state_dict()))
    scheduler.load_state_dict(copy.deepcopy(original.lr_scheduler.state_dict()))
    scaler = (
        torch.amp.GradScaler("cpu", init_scale=64, growth_interval=2) if amp else None
    )
    if amp:
        scaler.load_state_dict(original.scaler.state_dict())
    batches = [[0.3, 0.8], [0.2, 0.6], [0.9, 0.1]][:pending]
    for values in batches:
        accumulate(original, values)
    original.current_epoch = 0
    await original.save_checkpoint(checkpoint_name="pending")
    restored = make_trainer(
        tmp_path / "restored", trainer_type, amp=amp, max_norm=max_norm
    )
    restored.config.resume_from_checkpoint = str(tmp_path / "pending")
    # Actual checkpoint recovery and the trainer's final flush, with no episodes left.
    assert await restored.train() is restored.agent

    # Independent reference: one mean loss, without accumulation correction.
    loss = sum(batch_loss(reference, values) for values in batches) / pending
    if scaler is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
    else:
        loss.backward()
    torch.nn.utils.clip_grad_norm_(
        reference.parameters(), max_norm, error_if_nonfinite=True
    )
    if scaler is not None:
        scaler.step(optimizer)
        scaler.update()
    else:
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    scheduler.step()
    close(reference.state_dict(), restored.agent.model.state_dict())
    close(optimizer.state_dict(), restored.optimizer.state_dict())
    close(scheduler.state_dict(), restored.lr_scheduler.state_dict())
    if amp:
        close(scaler.state_dict(), restored.scaler.state_dict())
    assert restored.global_step == 2
    assert restored._grad_accum_step == 0
    assert all(
        parameter.grad is None for parameter in restored.agent.model.parameters()
    )

    # A second resume must not repeat the flushed update (including momentum).
    await restored.save_checkpoint(checkpoint_name="completed")
    completed = make_trainer(
        tmp_path / "completed-run", trainer_type, amp=amp, max_norm=max_norm
    )
    completed.config.resume_from_checkpoint = str(tmp_path / "restored" / "completed")
    assert await completed.train() is completed.agent
    close(restored.agent.model.state_dict(), completed.agent.model.state_dict())
    close(restored.optimizer.state_dict(), completed.optimizer.state_dict())
    close(restored.lr_scheduler.state_dict(), completed.lr_scheduler.state_dict())
    if amp:
        close(restored.scaler.state_dict(), completed.scaler.state_dict())
    assert completed.global_step == 2
    assert completed._grad_accum_step == 0
    assert all(
        parameter.grad is None for parameter in completed.agent.model.parameters()
    )


@pytest.mark.parametrize("correction", [0, -1, float("nan"), float("inf")])
def test_invalid_correction_cannot_mutate_optimizer(tmp_path, correction):
    value = make_trainer(tmp_path, SingleTurnGRPOTrainer, amp=False, max_norm=1)
    accumulate(value, [0.2, 0.7])
    weights = copy.deepcopy(value.agent.model.state_dict())
    optimizer = copy.deepcopy(value.optimizer.state_dict())
    with pytest.raises(ValueError, match="gradient_scale"):
        safe_optimizer_step(value, torch, max_grad_norm=1, gradient_scale=correction)
    close(weights, value.agent.model.state_dict())
    close(optimizer, value.optimizer.state_dict())
    assert value.global_step == 0 and value._grad_accum_step == 0
    assert value.agent.model.weight.grad is None


def test_overflow_from_correction_is_not_an_amp_skip(tmp_path):
    value = make_trainer(tmp_path, SingleTurnGRPOTrainer, amp=True, max_norm=1)
    accumulate(value, [2.0, 4.0])
    unscaled = value.agent.model.weight.grad / value.scaler.get_scale()
    assert not torch.isfinite(unscaled * 1e308).all()
    weights = copy.deepcopy(value.agent.model.state_dict())
    scaler = copy.deepcopy(value.scaler.state_dict())
    with pytest.raises(RuntimeError, match="non-finite"):
        safe_optimizer_step(
            value, torch, max_grad_norm=1, scaler=value.scaler, gradient_scale=1e308
        )
    close(weights, value.agent.model.state_dict())
    close(scaler, value.scaler.state_dict())
    assert value.global_step == 0 and value._grad_accum_step == 0
    assert value.agent.model.weight.grad is None


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("cap_size", ["zero", "unit", "minimum", "maximum"])
def test_clipping_large_finite_gradients_commits_correct_sgd_update(dtype, cap_size):
    """The norm may exceed the dtype limit while every clipped coordinate fits."""
    model = torch.nn.Linear(2, 1, bias=True, dtype=dtype)
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    trainer = SimpleNamespace(
        agent=SimpleNamespace(model=model),
        optimizer=optimizer,
        lr_scheduler=scheduler,
        global_step=0,
        _sync_rollout_backend=MagicMock(),
    )
    info = torch.finfo(dtype)
    cap = {
        "zero": 0.0,
        "unit": 1.0,
        "minimum": info.tiny * info.eps,
        "maximum": info.max,
    }[cap_size]
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
        model.weight.grad = torch.tensor(
            [[0.75 * info.max, info.tiny * info.eps]], dtype=dtype
        )
        model.bias.grad = torch.tensor([-0.75 * info.max], dtype=dtype)

    values = [*model.weight.grad.flatten().tolist(), model.bias.grad.item()]
    expected = -_decimal_clipped(values, cap, dtype)
    assert safe_optimizer_step(trainer, torch, max_grad_norm=cap)
    actual = torch.cat([model.weight.flatten(), model.bias])
    torch.testing.assert_close(actual, expected, rtol=3 * info.eps, atol=0)
    assert trainer.global_step == scheduler.last_epoch == 1
    assert optimizer.param_groups[0]["lr"] == 0.5
    trainer._sync_rollout_backend.assert_called_once()


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
@pytest.mark.parametrize("size", ["large", "subnormal"])
def test_clipping_with_finite_norm_and_tiny_coefficient(dtype, size):
    info = torch.finfo(dtype)
    unit = info.tiny * info.eps
    gradient = torch.tensor([info.max / 4 if size == "large" else unit], dtype=dtype)
    assert torch.linalg.vector_norm(gradient).isfinite()
    with localcontext() as ctx:
        ctx.prec = 90
        value = Decimal.from_float(gradient.item())
        expected = float(
            value * Decimal.from_float(unit) / (abs(value) + Decimal("0.000001"))
        )
    _clip_gradient_norm([gradient], unit, torch)
    torch.testing.assert_close(
        gradient, torch.tensor([expected], dtype=dtype), rtol=0, atol=0
    )


def test_clipping_old_pytorch_norm_api_and_complex_parameters(monkeypatch):
    monkeypatch.delattr(torch.nn.utils, "get_total_norm", raising=False)
    grad = torch.tensor([3 + 4j], dtype=torch.complex128)
    _clip_gradient_norm([grad], 2.0, torch)
    torch.testing.assert_close(
        grad, torch.tensor([(3 + 4j) * 2 / (5 + 1e-6)], dtype=grad.dtype)
    )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_clipping_mixed_exponents_against_decimal(dtype):
    rng = random.Random(74102)
    info = torch.finfo(dtype)
    unit = info.tiny * info.eps
    low = math.frexp(unit)[1] - 1
    high = math.frexp(info.max)[1] - 1

    def coordinate():
        exponent = rng.randint(low, high)
        return math.ldexp(rng.uniform(-1, 1), exponent)

    for case in range(400):
        gradient = torch.tensor([coordinate() for _ in range(12)], dtype=dtype)
        cap = abs(coordinate()) if case % 16 else unit
        reference = _decimal_clipped(gradient.tolist(), cap, dtype)
        _clip_gradient_norm([gradient[:2], gradient[2:7], gradient[7:]], cap, torch)
        for value, expected in zip(gradient.tolist(), reference.tolist(), strict=True):
            spacing = unit
            if expected:
                exponent = math.frexp(abs(expected))[1] - 1
                spacing = max(unit, math.ldexp(info.eps, exponent))
            assert abs(value - expected) <= 3 * spacing


@pytest.mark.parametrize("partition", [False, True])
@pytest.mark.parametrize("small", [False, True])
def test_dense_strided_norm_and_clip_match_analytic_reference(partition, small):
    # More than one workspace chunk. The even columns are gradient coordinates,
    # the odd ones are adjacent storage that clipping must never touch.
    n = 1024 * 1025
    storage = torch.ones(n, 2, dtype=torch.float32)
    g = storage[:, 0]
    g.fill_(2e-23 if small else 0.01)
    g[0] = 1.2e-19 if small else -3.14
    first, rest = g[:2].tolist()
    norm = math.hypot(first, math.sqrt(n - 1) * rest)
    cap = 1e-7 if small else 1.0
    groups = list(g.split(63001)) if partition else [g]
    measured = _clip_gradient_norm(groups, cap, torch)
    assert measured == pytest.approx(norm, rel=4e-15, abs=0)
    scale = min(1, cap / (norm + 1e-6))
    expected = torch.tensor([first * scale, rest * scale], dtype=g.dtype)
    assert g[0] == expected[0]
    assert (g[1:] == expected[1]).all()
    assert (storage[:, 1] == 1).all()


def test_independent_head_clipping_refines_integer_model():
    # The model uses one unit of epsilon. Scale coordinates and caps together
    # to exercise all 125 tables with the production epsilon.
    for a in range(-2, 3):
        for b in range(-2, 3):
            for cap in (0, 4, 6, 14, 26):
                for m, pair, length in ((a, (3, -4), 5), (b, (5, 12), 13)):
                    coords = [m * x * 1e-6 for x in pair]
                    expected_norm = abs(m) * length * 1e-6
                    expected = _decimal_clipped(coords, cap * 1e-6, torch.float64)
                    for split in (False, True):
                        g = torch.tensor(coords, dtype=torch.float64)
                        measured = _clip_gradient_norm(
                            list(reversed(g.split(1))) if split else [g],
                            cap * 1e-6,
                            torch,
                        )
                        assert measured == pytest.approx(
                            expected_norm, rel=4e-16, abs=0
                        )
                        torch.testing.assert_close(g, expected, rtol=5e-16, atol=0)


def test_parameter_clipping_counts_tied_parameter_once_and_skips_unused():
    from stateset_agents.training.trainer_utils import clip_parameter_gradients

    shared = torch.nn.Parameter(torch.zeros(2))
    shared.grad = torch.tensor([3.0, -4.0])
    unused = torch.nn.Parameter(torch.ones(1))
    assert clip_parameter_gradients(iter([unused, shared, shared]), 2.0) == 5
    assert unused.grad is None
    torch.testing.assert_close(shared.grad, torch.tensor([6, -8]) / (5 + 1e-6))
    assert clip_parameter_gradients(iter([unused]), 2.0) == 0


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_invalid_late_chunk_preserves_all_coordinates(dtype):
    healthy = torch.tensor([3.0, 4.0], dtype=dtype)
    invalid = torch.tensor([0.0, math.nan], dtype=dtype)
    with pytest.raises(RuntimeError, match="non-finite"):
        _clip_gradient_norm([healthy, invalid], 0.0, torch)
    assert healthy.tolist() == [3.0, 4.0]
    assert invalid[0] == 0 and torch.isnan(invalid[1])


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_stored_half_coordinates_realize_lean_rounding_witness(dtype):
    g = torch.tensor([1, 2], dtype=dtype)
    norm = _clip_gradient_norm([g], 1.0, torch)
    assert norm == math.sqrt(5)
    assert g.tolist() == [229 / 512, 229 / 256]
    assert math.hypot(*g.tolist()) > 1.0
