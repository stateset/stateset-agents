"""Declared lifecycles: sound machines, one definition, no undeclared statuses."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents.lifecycles import (
    FANOUT_RUN,
    FINETUNE_RUN,
    GATEWAY_TRAINING_JOB,
    LIFECYCLES,
    REMOTE_JOB,
    Lifecycle,
    get_lifecycle,
)

PACKAGE = Path(__file__).resolve().parents[2] / "stateset_agents"


@pytest.mark.parametrize("machine", LIFECYCLES.values(), ids=lambda m: m.name)
def test_every_lifecycle_is_sound(machine: Lifecycle) -> None:
    assert machine.initial in machine.active
    assert machine.terminal and machine.terminal <= machine.all_states
    assert machine.settled <= machine.terminal
    assert machine.settled, "some outcome must be permanent"
    for state in machine.active:
        assert machine.next_states(state), f"{state} is a dead end but not terminal"
    for state, meaning in machine.states.items():
        assert meaning.endswith("."), state
    assert json.loads(json.dumps(machine.describe()))["name"] == machine.name


def test_the_modules_that_own_a_status_use_the_declaration() -> None:
    from stateset_agents import finetune_workspace
    from stateset_agents.api import training_queue
    from stateset_agents.api.services import training_service
    from stateset_agents.fanout import journal
    from stateset_agents.remote.job import JobStatus

    assert journal.TERMINAL_STATUSES is FANOUT_RUN.terminal
    assert finetune_workspace.TERMINAL_STATES is FINETUNE_RUN.terminal
    assert training_service.TERMINAL_TRAINING_STATUSES is GATEWAY_TRAINING_JOB.terminal
    assert training_queue.TERMINAL is GATEWAY_TRAINING_JOB.terminal
    assert {s.value for s in JobStatus} == REMOTE_JOB.all_states
    assert {s.value for s in JobStatus if s.is_terminal} == REMOTE_JOB.terminal


def test_declarations_keep_the_values_that_are_already_on_disk() -> None:
    # These strings are in state.json files, job records and API responses.
    assert FANOUT_RUN.terminal == {"succeeded", "failed", "cancelled"}
    assert FINETUNE_RUN.terminal == {"succeeded", "failed", "interrupted", "cancelled"}
    assert GATEWAY_TRAINING_JOB.terminal == {"completed", "failed", "cancelled"}
    assert REMOTE_JOB.terminal == {"succeeded", "failed", "cancelled"}
    assert FINETUNE_RUN.all_states == {
        "prepared",
        "starting",
        "queued",
        "running",
        "succeeded",
        "failed",
        "interrupted",
        "cancelled",
    }


def test_finetune_declaration_matches_the_writer_that_enforces_it(tmp_path) -> None:
    from stateset_agents import finetune_workspace as workspace

    assert workspace.FINAL_STATUSES == FINETUNE_RUN.terminal - {"interrupted"}
    assert FINETUNE_RUN.settled == {"failed", "cancelled"}
    for current in sorted(FINETUNE_RUN.all_states):
        for new in sorted(FINETUNE_RUN.all_states):
            run = tmp_path / f"{current}-{new}"
            run.mkdir()
            workspace.write_json(run / "state.json", {"status": current})
            moved = workspace.transition_state(run, status=new)
            assert moved == FINETUNE_RUN.can_transition(current, new), (current, new)


def _written_statuses(path: Path) -> set[str]:
    """String literals a module writes as a status."""
    found: set[str] = set()

    def literal(node: ast.AST) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            found.add(node.value)
        elif isinstance(node, ast.IfExp):
            literal(node.body)
            literal(node.orelse)

    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.keyword) and node.arg == "status":
            literal(node.value)
        elif isinstance(node, ast.Dict):
            for key, value in zip(node.keys, node.values, strict=True):
                if isinstance(key, ast.Constant) and key.value == "status":
                    literal(value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                named = isinstance(target, ast.Attribute) and target.attr == "status"
                keyed = (
                    isinstance(target, ast.Subscript)
                    and isinstance(target.slice, ast.Constant)
                    and target.slice.value == "status"
                )
                if named or keyed:
                    literal(node.value)
    return found


#: Statuses a module writes that belong to a different record, with the reason.
OTHER_RECORDS = {
    "finetune_workspace.py": {"read_only"},  # the workspace summary, not a run
    "fanout/orchestrator.py": {"timed_out"},  # a process state, not a run status
}


@pytest.mark.parametrize(
    ("module", "machine"),
    [
        ("fanout/orchestrator.py", FANOUT_RUN),
        ("finetune_worker.py", FINETUNE_RUN),
        ("finetune_workspace.py", FINETUNE_RUN),
        ("api/services/training_service.py", GATEWAY_TRAINING_JOB),
        ("api/training_queue.py", GATEWAY_TRAINING_JOB),
    ],
)
def test_no_module_writes_a_status_its_lifecycle_does_not_declare(
    module: str, machine: Lifecycle
) -> None:
    written = _written_statuses(PACKAGE / module) - OTHER_RECORDS.get(module, set())
    assert written, f"{module} writes no status literal; is the scan still right?"
    undeclared = written - machine.all_states
    assert not undeclared, (
        f"{module} writes {sorted(undeclared)}, which {machine.name} does not "
        "declare. Add the status to stateset_agents/lifecycles.py (with its "
        "meaning and transitions) or list it in OTHER_RECORDS with a reason."
    )


def test_require_names_the_legal_moves() -> None:
    REMOTE_JOB.require("pending", "running")
    REMOTE_JOB.require("running", "running")
    with pytest.raises(ValueError, match="it can become: nothing"):
        REMOTE_JOB.require("cancelled", "failed")
    with pytest.raises(ValueError, match="cancelled, failed, succeeded"):
        REMOTE_JOB.require("running", "pending")
    with pytest.raises(ValueError, match="unknown status 'paused'"):
        REMOTE_JOB.require("paused", "running")
    assert FANOUT_RUN.can_transition("running", "pending")  # retry on preemption
    assert not GATEWAY_TRAINING_JOB.can_transition("completed", "running")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"initial": "nowhere"}, "initial state"),
        ({"terminal": frozenset({"gone"})}, "undeclared states"),
        ({"transitions": {"a": frozenset({"c"})}}, "undeclared states"),
        ({"transitions": {}}, "unreachable states"),
        ({"terminal": frozenset()}, "needs a terminal state"),
    ],
)
def test_a_malformed_lifecycle_is_rejected(changes, message) -> None:
    base = {
        "name": "t",
        "summary": "s",
        "states": {"a": "A.", "b": "B."},
        "initial": "a",
        "terminal": frozenset({"b"}),
        "transitions": {"a": frozenset({"b"})},
        "owner": "o",
        "stored_in": "x",
    }
    with pytest.raises(ValueError, match=message):
        Lifecycle(**{**base, **changes})


def test_cli_lists_lifecycles_for_people_and_agents() -> None:
    from stateset_agents.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["lifecycles", "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert [m["name"] for m in data] == list(LIFECYCLES)
    remote = data[0]
    running = next(s for s in remote["states"] if s["status"] == "running")
    assert running == {
        "status": "running",
        "meaning": "The provider is running it.",
        "terminal": False,
        "settled": False,
        "next": ["cancelled", "failed", "succeeded"],
    }
    text = runner.invoke(app, ["lifecycles", "fanout-run"])
    assert text.exit_code == 0 and "next: failed, pending, succeeded" not in text.output
    assert "running      active" in text.output
    missing = runner.invoke(app, ["lifecycles", "nope"])
    assert missing.exit_code == 2 and "known:" in missing.output
    assert get_lifecycle("remote-job") is REMOTE_JOB
