"""BFCL environment: worker protocol, scoring, splits and fail-closed errors."""

from __future__ import annotations

import os
import sys
import textwrap
from pathlib import Path
from typing import Any

import pytest

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.training.bfcl import (
    CATEGORIES,
    BfclChecker,
    BfclEnvironment,
    bfcl_split,
)
from stateset_agents.training.tau_retail import EvaluationUnavailable

FAKE_WORKER = textwrap.dedent("""
    import json, sys
    print(json.dumps({"ready": True, "version": "fake", "stood_in": []}), flush=True)
    for line in sys.stdin:
        request = json.loads(line)
        if request["op"] == "tasks":
            reply = {"prompt": "Functions:\\n{functions}", "tasks": [
                {"id": "t0", "question": [[{"role": "user", "content": "area?"}]],
                 "function": [{"name": "area"}]},
                {"id": "t1", "question": [[{"role": "user", "content": "sum?"}]],
                 "function": [{"name": "sum"}]}]}
        elif request["output"] == "crash":
            sys.exit(3)
        elif request["output"] == "garbage":
            print("not json", flush=True)
            continue
        else:
            reply = {"valid": request["output"] == "[" + request["id"] + "()]"}
        print(json.dumps(reply), flush=True)
    """)


@pytest.fixture
def environment(tmp_path: Path) -> Any:
    worker = tmp_path / "worker.py"
    worker.write_text(FAKE_WORKER)
    checker = BfclChecker(sys.executable, worker=worker)
    yield BfclEnvironment(checker, "simple_python")
    checker.close()


def reply(text: str) -> Any:
    async def policy(history: Any, context: dict[str, Any]) -> ConversationTurn:
        return ConversationTurn(role="assistant", content=text)

    return policy


@pytest.mark.asyncio
async def test_checker_verdict_is_the_reward(environment: BfclEnvironment) -> None:
    assert environment.task_ids() == ["t0", "t1"]
    right = await environment.run_episode(reply("[t0()]"), {"id": "t0"})
    wrong = await environment.run_episode(reply("[t1()]"), {"id": "t0"})
    assert right.total_reward == 1.0 and right.final_state["task_success"] is True
    assert wrong.total_reward == 0.0 and wrong.final_state["task_success"] is False
    assert [t.role for t in right.turns] == ["system", "user", "assistant"]
    assert '"name": "area"' in right.turns[0].content


@pytest.mark.asyncio
async def test_policy_context_and_bad_scenarios(environment: BfclEnvironment) -> None:
    seen: list[dict[str, Any]] = []

    async def policy(history: Any, context: dict[str, Any]) -> ConversationTurn:
        seen.append(context)
        return ConversationTurn(role="assistant", content="x")

    await environment.run_episode(policy, {"id": "t1"})
    assert seen == [{"task_id": "t1"}]
    with pytest.raises(ValueError):
        await environment.reset({"id": "missing"})


@pytest.mark.asyncio
@pytest.mark.parametrize("output", ["garbage", "crash"])
async def test_a_broken_checker_is_not_a_wrong_answer(
    environment: BfclEnvironment, output: str
) -> None:
    with pytest.raises(EvaluationUnavailable):
        await environment.run_episode(reply(output), {"id": "t0"})


def test_split_is_disjoint_deterministic_and_seeded() -> None:
    ids = [f"task_{i}" for i in range(40)]
    train, test = bfcl_split(ids, seed=1, test_fraction=0.25)
    assert len(test) == 10 and not set(train) & set(test)
    assert set(train) | set(test) == set(ids)
    assert (train, test) == bfcl_split(list(reversed(ids)), seed=1, test_fraction=0.25)
    assert test != bfcl_split(ids, seed=2, test_fraction=0.25)[1]
    with pytest.raises(ValueError):
        bfcl_split(ids, seed=1, test_fraction=1.0)
    with pytest.raises(ValueError):
        bfcl_split(["a", "a"], seed=1)


def bfcl_python() -> str | None:
    default = Path.home() / ".cache/stateset-agents/bfcl-venv/bin/python"
    path = os.environ.get("STATESET_BFCL_PYTHON") or str(default)
    return path if Path(path).exists() else None


def render(answer: list[dict[str, dict[str, list[Any]]]]) -> str:
    """Write a reference answer in BFCL's call syntax, first option per argument."""
    calls = []
    for call in answer:
        for name, arguments in call.items():
            chosen = [
                f"{key}={options[0]!r}"
                for key, options in arguments.items()
                if options[0] != ""
            ]
            calls.append(f"{name}({', '.join(chosen)})")
    return "[" + ", ".join(calls) + "]"


@pytest.mark.asyncio
@pytest.mark.parametrize("category", CATEGORIES)
async def test_official_checker_accepts_reference_answers(category: str) -> None:
    python = bfcl_python()
    if python is None:
        pytest.skip("isolated BFCL interpreter not installed")
    import json

    checker = BfclChecker(python)
    try:
        environment = BfclEnvironment(checker, category)
        data = Path(python).parent.parent.glob(
            f"lib/python*/site-packages/bfcl_eval/data/possible_answer/BFCL_v4_{category}.json"
        )
        answers = [json.loads(line) for line in next(data).read_text().splitlines()]
        assert len(answers) == len(environment.task_ids()) >= 200
        accepted = 0
        for row in answers[:60]:
            good = await environment.run_episode(
                reply(render(row["ground_truth"])), {"id": row["id"]}
            )
            idle = await environment.run_episode(
                reply("I cannot help with that."), {"id": row["id"]}
            )
            accepted += int(good.final_state["task_success"])
            assert idle.final_state["task_success"] is False
        assert accepted >= 57
    finally:
        checker.close()
