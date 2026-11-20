"""BFCL function-calling tasks scored by the official checker.

``bfcl-eval`` cannot share an interpreter with this package (it pins older
numpy and newer httpx), so the official decoder and AST checker run in a
separate interpreter through :mod:`bfcl_checker_worker`. A reply that is
missing or malformed raises :class:`EvaluationUnavailable`; it is never
scored as a wrong answer.

BFCL publishes no training split. :func:`bfcl_split` divides a category
deterministically so a policy is never scored on tasks it trained on, which
also means results are not comparable with leaderboard numbers.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import threading
import uuid
from pathlib import Path
from typing import Any

from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.trajectory import ConversationTurn

from .tau_retail import EvaluationUnavailable

CATEGORIES = ("simple_python", "multiple", "parallel", "parallel_multiple")
WORKER = Path(__file__).with_name("bfcl_checker_worker.py")


class BfclChecker:
    """A long-lived worker process running the official BFCL checker."""

    def __init__(self, python: str | Path, *, worker: Path = WORKER) -> None:
        self._lock = threading.Lock()
        self._process = subprocess.Popen(
            [str(python), str(worker)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self._read()
        if ready.get("ready") is not True:
            raise EvaluationUnavailable("BFCL checker worker did not start")
        self.version: str = ready["version"]
        self.stood_in: list[str] = list(ready.get("stood_in", []))

    def _read(self) -> dict[str, Any]:
        assert self._process.stdout is not None
        line = self._process.stdout.readline()
        if not line:
            raise EvaluationUnavailable("BFCL checker worker exited")
        try:
            reply = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvaluationUnavailable(
                "BFCL checker sent a malformed reply"
            ) from error
        if not isinstance(reply, dict):
            raise EvaluationUnavailable("BFCL checker sent a malformed reply")
        return reply

    def request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """Send one request and return its reply."""
        with self._lock:
            assert self._process.stdin is not None
            try:
                self._process.stdin.write(json.dumps(payload) + "\n")
                self._process.stdin.flush()
            except (BrokenPipeError, OSError) as error:
                raise EvaluationUnavailable("BFCL checker worker exited") from error
            return self._read()

    def tasks(self, category: str) -> tuple[str, list[dict[str, Any]]]:
        """Return the reference system-prompt template and a category's tasks."""
        if category not in CATEGORIES:
            raise ValueError(f"category must be one of {CATEGORIES}")
        reply = self.request({"op": "tasks", "category": category})
        return reply["prompt"], reply["tasks"]

    def check(self, category: str, task_id: str, output: str) -> bool:
        """Return whether the official checker accepts ``output``."""
        reply = self.request(
            {"op": "check", "category": category, "id": task_id, "output": output}
        )
        if type(reply.get("valid")) is not bool:
            raise EvaluationUnavailable("BFCL checker reply has no verdict")
        return bool(reply["valid"])

    def close(self) -> None:
        """Stop the worker."""
        if self._process.poll() is None:
            self._process.terminate()
            self._process.wait(timeout=10)


def bfcl_split(
    task_ids: list[str], *, seed: int, test_fraction: float = 0.5
) -> tuple[list[str], list[str]]:
    """Split task IDs into disjoint train and test lists by a seeded hash."""
    if not 0 < test_fraction < 1:
        raise ValueError("test_fraction must be between 0 and 1")
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("task_ids must be unique")

    def rank(task_id: str) -> str:
        return hashlib.sha256(f"{seed}:{task_id}".encode()).hexdigest()

    ordered = sorted(task_ids, key=rank)
    cut = round(len(ordered) * test_fraction)
    return sorted(ordered[cut:]), sorted(ordered[:cut])


class BfclEnvironment(Environment):
    """One single-turn function-calling task per episode."""

    def __init__(self, checker: BfclChecker, category: str) -> None:
        super().__init__(max_turns=1, strict_rewards=True)
        self.checker = checker
        self.category = category
        self.prompt_template, tasks = checker.tasks(category)
        self.tasks = {task["id"]: task for task in tasks}

    def task_ids(self) -> list[str]:
        """Return every task ID in this category."""
        return sorted(self.tasks)

    def clone(self) -> BfclEnvironment:
        """Return an environment sharing the same checker worker."""
        return BfclEnvironment(self.checker, self.category)

    def observation_context(self, state: EnvironmentState) -> dict[str, Any]:
        """Expose only the task ID; answers stay in the checker process."""
        return {"task_id": state.context["task_id"]}

    def _task(self, scenario: dict[str, Any] | None) -> dict[str, Any]:
        if not scenario or scenario.get("id") not in self.tasks:
            raise ValueError("scenario needs the 'id' of a task in this category")
        return self.tasks[scenario["id"]]

    async def get_initial_prompt(self, scenario: dict[str, Any] | None = None) -> str:
        """Fill BFCL's reference system prompt with the task's functions."""
        functions = json.dumps(self._task(scenario)["function"], indent=4)
        return self.prompt_template.format(functions=functions)

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        """Start the task named by ``scenario['id']``."""
        task = self._task(scenario)
        return EnvironmentState(
            f"{task['id']}-{uuid.uuid4().hex}",
            0,
            EpisodeStatus.ONGOING,
            {
                "task_id": task["id"],
                "messages": [dict(message) for message in task["question"][0]],
                "task_success": False,
            },
        )

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        """Score the agent's single reply with the official checker."""
        if state.is_done:
            raise ValueError("Episode already ended")
        state.turn_count += 1
        success = await asyncio.to_thread(
            self.checker.check,
            self.category,
            state.context["task_id"],
            action.content or "",
        )
        state.status = EpisodeStatus.COMPLETED if success else EpisodeStatus.FAILED
        state.context["task_success"] = success
        return state, float(success), True, {}


def make_environment(
    *, python: str | Path, category: str = "simple_python"
) -> BfclEnvironment:
    """Start a checker worker under ``python`` and build the environment."""
    return BfclEnvironment(BfclChecker(python), category)
