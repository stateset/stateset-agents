"""Round 6: the symbolic physics reward does not pay the trivial solution.

A constant ("0", "1", "x - x") satisfies symmetry, parity and zero constraints
trivially, and every parseable expression used to earn full validity, so the
constant scored 1.0 against constraint-only tasks. A task without declared
variables used to score 0 instead of failing.
"""

from __future__ import annotations

from typing import Any

import pytest

from stateset_agents.core.errors import ConfigurationError
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.environments.symbolic_physics import SymbolicPhysicsEnvironment
from stateset_agents.rewards.symbolic_physics_reward import (
    SymbolicPhysicsRewardFunction,
    is_constant_expression,
)

#: An even, symmetric relation in x and y: x**2 + y**2 keeps every constraint,
#: and so does any constant (that is the defect).
EVEN_SYMMETRIC: dict[str, Any] = {
    "variables": ["x", "y"],
    "constraints": [
        {"type": "symmetry", "pairs": [["x", "y"]]},
        {"type": "sign_flip", "variables": ["x", "y"], "parity": "even"},
    ],
}
#: Adds a zero at the origin and quadratic scaling: "0" kept all four.
QUADRATIC: dict[str, Any] = {
    "variables": ["x", "y"],
    "constraints": [
        *EVEN_SYMMETRIC["constraints"],
        {"type": "zeros", "points": [{"x": 0.0, "y": 0.0}]},
        {"type": "scale", "power": 2},
    ],
}


async def _score(expression: str, context: dict[str, Any] | None) -> float:
    reward = SymbolicPhysicsRewardFunction()
    turn = ConversationTurn(role="assistant", content=expression)
    return (await reward.compute_reward([turn], context)).score


@pytest.mark.parametrize("context", [EVEN_SYMMETRIC, QUADRATIC], ids=["even", "quad"])
@pytest.mark.parametrize("constant", ["0", "1", "x - x", "sin(x)**2 + cos(x)**2"])
async def test_constants_score_zero_against_constraints_they_satisfy(
    constant: str, context: dict[str, Any]
) -> None:
    # Before round 6: "0" scored 1.0 on both tasks, "1" 1.0 and 0.75.
    assert await _score(constant, context) < 0.1


@pytest.mark.parametrize("context", [EVEN_SYMMETRIC, QUADRATIC], ids=["even", "quad"])
async def test_the_real_relation_still_scores_high(context: dict[str, Any]) -> None:
    assert await _score("x**2 + y**2", context) > 0.9


async def test_constraints_and_equivalence_cannot_lift_a_constant() -> None:
    # The loose equivalence tolerance (rel_tol 1e-2) matches "1000" to
    # 1000 + x on the sample range; the constant still earns nothing.
    context = {"variables": ["x"], "target_expression": "1000 + x"}
    assert await _score("1000", context) == 0.0
    assert await _score("1000 + x", context) > 0.9
    result = await SymbolicPhysicsRewardFunction().compute_reward(
        [ConversationTurn(role="assistant", content="1")], dict(EVEN_SYMMETRIC)
    )
    assert result.metadata["error"] == "trivial_expression"


async def test_a_constant_target_accepts_an_equivalent_constant() -> None:
    context = {"variables": ["x", "y"], "target_expression": "1"}
    assert await _score("1", context) > 0.9
    assert await _score("sin(x)**2 + cos(x)**2", context) > 0.9
    assert await _score("0", context) < 0.5


@pytest.mark.parametrize("context", [None, {}, {"target_expression": "x"}])
async def test_missing_variables_raise_a_configuration_error(
    context: dict[str, Any] | None,
) -> None:
    reward = SymbolicPhysicsRewardFunction()
    assert reward.fail_on_error is True
    with pytest.raises(ConfigurationError, match="variables"):
        await _score("x + y", context)


async def test_the_environment_stops_on_a_task_without_variables() -> None:
    env = SymbolicPhysicsEnvironment(
        tasks=[{"id": "bad", "prompt": "?", "variables": [], "constraints": []}],
        max_turns=1,
    )
    state = await env.reset()
    with pytest.raises(ConfigurationError):
        await env.step(state, ConversationTurn(role="assistant", content="x"))


def test_constancy_is_read_from_names_and_values() -> None:
    assert is_constant_expression("pi", [3.14159] * 4, ["x"])
    assert is_constant_expression("x - x", [0.0, 0.0, 0.0], ["x"])
    assert not is_constant_expression("x", [0.5, -1.0], ["x"])
    # One sample cannot show constancy; the names still can.
    assert not is_constant_expression("x", [0.5], ["x"])
    assert is_constant_expression("2", [2.0], ["x"])
    # A derived variable counts as a declared name.
    assert not is_constant_expression("d", [0.1, 0.7], ["x", "y", "d"])


async def test_near_misses_score_well_below_a_target_relation() -> None:
    context = {
        "variables": ["x", "y"],
        "target_expression": "x**2 + y**2",
        "constraints": QUADRATIC["constraints"],
    }
    assert await _score("x**2 + y**2", context) == 1.0
    for near_miss in ("2*(x**2 + y**2)", "-(x**2 + y**2)", "(x**2 + y**2) + 1"):
        assert await _score(near_miss, context) <= 0.4 + 1e-9


async def test_constraint_only_tasks_do_not_fix_scale_or_sign() -> None:
    """Known limit: constraints admit a family of relations. Without a
    ``target_expression``, any non-constant member scores like the intended
    one; give a target where the exact relation matters."""
    for member in ("2*(x**2 + y**2)", "-(x**2 + y**2)"):
        assert await _score(member, QUADRATIC) == await _score("x**2 + y**2", QUADRATIC)
