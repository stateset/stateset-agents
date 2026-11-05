"""``spine.update.UpdateLoop``: K update passes over one frozen rollout batch."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training.spine import (  # noqa: E402
    UPDATE_MODES,
    OptimizerStep,
    UpdateLoop,
    batch_shares,
    share_size,
)


def _setup(counter: str = "train_steps") -> tuple[Any, Any, OptimizerStep, list[int]]:
    torch.manual_seed(0)
    model = torch.nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    owner = SimpleNamespace(global_step=0)
    steps: list[int] = []
    optimizer.register_step_pre_hook(lambda *args: steps.append(1))
    step = OptimizerStep(
        owner,
        torch,
        optimizers=[optimizer],
        max_grad_norm=100.0,
        counter=counter,  # type: ignore[arg-type]
    )
    return model, owner, step, steps


ITEMS = [torch.tensor([[1.0, 0.0, 2.0]]), torch.tensor([[0.5, -1.0, 0.0]])]


def _item_loss(model: Any) -> Any:
    return lambda _pass, _index, item: model(item).square().mean()


def test_vocabularies_and_validation() -> None:
    assert UPDATE_MODES == ("chunked", "single_graph")
    _, _, step, _ = _setup()
    for passes in (0, -1, 1.0, True):
        with pytest.raises(ValueError, match="passes"):
            UpdateLoop(step, passes=passes)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mode"):
        UpdateLoop(step, passes=1, mode="graph")  # type: ignore[arg-type]
    loop = UpdateLoop(step, passes=1)
    with pytest.raises(ValueError, match="shares"):
        loop.run(ITEMS, lambda *a: None, shares=[1.0])
    with pytest.raises(ValueError, match="initial_loss"):
        UpdateLoop(step, passes=1, mode="single_graph").run(ITEMS, lambda *a: None)


def test_batch_shares_and_share_size() -> None:
    assert batch_shares([1.0, 3.0]) == [0.25, 0.75]
    assert batch_shares([0.0, 0.0]) == [0.0, 0.0]
    mask = torch.tensor([[1.0, 1.0, 1.0, 0.0], [0.0, 0.0, 1.0, 1.0]])
    assert share_size(mask, "token_mean") == 4.0  # action frame: mask[:, 1:]
    assert share_size(mask, "seq_mean") == 2.0


def test_chunked_passes_commit_once_each_and_count_the_train_step() -> None:
    model, owner, step, steps = _setup()
    calls: list[tuple[int, int]] = []
    starts: list[int] = []

    def loss_fn(pass_index: int, index: int, item: Any) -> Any:
        calls.append((pass_index, index))
        return model(item).square().mean()

    result = UpdateLoop(step, passes=3).run(ITEMS, loss_fn, on_pass_start=starts.append)
    assert starts == [0, 1, 2]
    assert calls == [(p, i) for p in range(3) for i in range(2)]
    assert len(steps) == 3 and result.committed == 3
    assert len(result.losses) == 3
    assert owner.global_step == 1


def test_chunked_shares_reproduce_the_batch_objective_gradient() -> None:
    model, _, step, _ = _setup()
    sizes = [1.0, 3.0]
    shares = batch_shares(sizes)

    expected_loss = sum(
        share * model(item).square().mean()
        for share, item in zip(shares, ITEMS, strict=True)
    )
    expected_loss.backward()
    expected = [p.grad.clone() for p in model.parameters()]
    model.zero_grad()

    seen: list[list[Any]] = []
    step.optimizers[0].register_step_pre_hook(
        lambda *args: seen.append([p.grad.clone() for p in model.parameters()])
    )
    result = UpdateLoop(step, passes=1).run(ITEMS, _item_loss(model), shares=shares)
    for got, want in zip(seen[0], expected, strict=True):
        torch.testing.assert_close(got, want)
    assert result.losses[0] == pytest.approx(expected_loss.item(), rel=1e-6)


def test_chunked_skips_zero_share_items_and_none_losses() -> None:
    model, _, step, steps = _setup()
    called: list[int] = []

    def loss_fn(_pass: int, index: int, item: Any) -> Any:
        called.append(index)
        return None if index == 1 else model(item).sum()

    result = UpdateLoop(step, passes=1).run(
        ITEMS + ITEMS, loss_fn, shares=[0.0, 0.5, 0.5, 0.0]
    )
    assert called == [1, 2]  # zero-share items never run their forward
    assert result.committed == 1 and len(steps) == 1


def test_each_pass_starts_from_clean_gradients() -> None:
    model, _, step, _ = _setup()
    norms: list[float] = []
    step.optimizers[0].register_step_pre_hook(
        lambda *args: norms.append(float(model.weight.grad.norm()))
    )
    # A constant gradient: if a pass leaked into the next, the norm would grow.
    UpdateLoop(step, passes=3).run(ITEMS[:1], lambda *a: model.weight.sum())
    assert norms == pytest.approx([norms[0]] * 3)


def test_single_graph_reduces_one_graph_and_uses_the_default_commit() -> None:
    model, owner, step, steps = _setup()
    reduced: list[float] = []

    def reduce(total: Any) -> Any:
        reduced.append(total.item())
        return total / 2

    result = UpdateLoop(step, passes=2, mode="single_graph").run(
        ITEMS,
        _item_loss(model),
        initial_loss=lambda: torch.tensor(0.0, requires_grad=True),
        reduce=reduce,
    )
    assert len(steps) == 2 and result.committed == 2
    assert result.losses == pytest.approx([value / 2 for value in reduced])
    assert owner.global_step == 1


def test_single_graph_custom_commit_reports_commits() -> None:
    model, _, step, steps = _setup()
    outcomes = iter([False, True])
    seen: list[float] = []

    def commit(loss: Any) -> bool:
        seen.append(loss.item())
        return next(outcomes)

    result = UpdateLoop(step, passes=2, mode="single_graph").run(
        ITEMS,
        _item_loss(model),
        initial_loss=lambda: torch.tensor(0.0),
        commit=commit,
    )
    assert result.committed == 1
    assert result.losses == pytest.approx(seen)
    assert steps == []  # the custom commit owns backward and stepping


@pytest.mark.parametrize("mode", UPDATE_MODES)
def test_empty_batch_never_steps_stale_gradients(mode: str) -> None:
    model, owner, step, steps = _setup()
    for parameter in model.parameters():
        parameter.grad = torch.ones_like(parameter)
    result = UpdateLoop(step, passes=2, mode=mode).run(  # type: ignore[arg-type]
        [], _item_loss(model), initial_loss=lambda: torch.tensor(0.0)
    )
    assert result.losses == [0.0, 0.0] and result.committed == 0
    assert steps == []
    assert owner.global_step == 1


@pytest.mark.parametrize("mode", UPDATE_MODES)
def test_failed_pass_discards_gradients_and_propagates(mode: str) -> None:
    model, owner, step, steps = _setup()

    def loss_fn(_pass: int, index: int, item: Any) -> Any:
        if index == 1:
            raise RuntimeError("injected forward failure")
        return model(item).sum()

    with pytest.raises(RuntimeError, match="injected"):
        UpdateLoop(step, passes=2, mode=mode).run(  # type: ignore[arg-type]
            ITEMS, loss_fn, initial_loss=lambda: torch.tensor(0.0)
        )
    assert all(p.grad is None for p in model.parameters())
    assert steps == [] and owner.global_step == 0
