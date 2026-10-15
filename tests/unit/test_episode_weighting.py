"""Per-episode weighting of multi-turn GSPO rows (GSPO eq. 5)."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import objectives  # noqa: E402
from stateset_agents.training.loss_computation import (  # noqa: E402
    EPISODE_WEIGHTINGS,
    episode_row_weights,
)


def test_row_weights_per_turn_and_per_episode() -> None:
    assert EPISODE_WEIGHTINGS == ("per_turn", "per_episode")
    assert episode_row_weights([0, 0, 1], "per_turn") is None
    assert episode_row_weights([0, 0, 1], "per_episode") == [0.5, 0.5, 1.0]
    with pytest.raises(ValueError):
        episode_row_weights([0], "per_token")


def _loss(row_weights, total_rows):
    gspo = objectives.OBJECTIVES["gspo"]
    # Three rows: episode A has 2 turns (A=+1), episode B has 1 turn (A=-1).
    logp = torch.zeros(3, 4, requires_grad=True)
    mask = torch.ones(3, 4)
    adv = torch.tensor([1.0, 1.0, -1.0])
    result = objectives.policy_loss(
        logp_cur=logp,
        mask=mask,
        advantages=adv,
        objective=gspo,
        total_rows=total_rows,
        total_tokens=float(mask.sum()) if row_weights is None else None,
        row_weights=row_weights,
    )
    result.loss.backward()
    return float(result.loss), logp.grad.sum(-1)


def test_per_episode_weighting_balances_episodes() -> None:
    weights = torch.tensor(episode_row_weights([0, 0, 1], "per_episode"))
    loss, grad = _loss(weights, float(weights.sum()))
    assert loss == pytest.approx(0.0, abs=1e-6)
    # Each episode carries the same total gradient magnitude.
    assert float(grad[:2].sum()).__abs__() == pytest.approx(float(grad[2]).__abs__())
    # Per-turn (default) lets the 2-turn episode dominate.
    loss_turn, grad_turn = _loss(None, 3.0)
    assert loss_turn == pytest.approx(-1.0 / 3.0, abs=1e-6)
    assert abs(float(grad_turn[:2].sum())) == pytest.approx(
        2 * abs(float(grad_turn[2]))
    )


def test_row_weights_shape_is_checked() -> None:
    with pytest.raises(ValueError, match="row_weights"):
        objectives.aggregate(
            objectives.OBJECTIVES["gspo"],
            torch.zeros(2, 3),
            torch.ones(2, 3),
            row_weights=torch.ones(3),
        )
