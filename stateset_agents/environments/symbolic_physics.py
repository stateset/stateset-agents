"""
Symbolic physics environment for discovering analytic relations.
"""

from __future__ import annotations

import json
import logging
import random
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

from stateset_agents.core.environment import Environment, EnvironmentState, EpisodeStatus
from stateset_agents.core.reward_base import RewardFunction
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.rewards.symbolic_physics_reward import (
    SymbolicPhysicsRewardFunction,
    SymbolicRewardConfig,
)

logger = logging.getLogger(__name__)


@dataclass
class SymbolicPhysicsTask:
    """Task spec for symbolic physics relation discovery."""

    task_id: str
    prompt: str
    variables: List[str]
    target_expression: Optional[str] = None
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    derived_variables: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SymbolicPhysicsTask":
        return cls(
            task_id=str(data.get("id") or data.get("task_id") or data.get("name") or "task"),
            prompt=str(data.get("prompt") or data.get("context") or ""),
            variables=list(data.get("variables") or []),
            target_expression=data.get("target_expression"),
            constraints=list(data.get("constraints") or []),
            derived_variables=dict(data.get("derived_variables") or {}),
            metadata=dict(data.get("metadata") or {}),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.task_id,
            "prompt": self.prompt,
            "variables": list(self.variables),
            "target_expression": self.target_expression,
            "constraints": list(self.constraints),
            "derived_variables": dict(self.derived_variables),
            "metadata": dict(self.metadata),
        }

    def to_scenario(self) -> Dict[str, Any]:
        return {"id": self.task_id, "context": self.prompt, "task": self.to_dict()}


def load_symbolic_tasks(path: Union[str, Path]) -> List[SymbolicPhysicsTask]:
    """Load symbolic physics tasks from a JSON or JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Task file not found: {path}")
    if path.suffix.lower() == ".jsonl":
        tasks = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                tasks.append(SymbolicPhysicsTask.from_dict(json.loads(line)))
        return tasks
    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [SymbolicPhysicsTask.from_dict(item) for item in data]
        return [SymbolicPhysicsTask.from_dict(data)]
    raise ValueError("Expected a .json or .jsonl task file")


class SymbolicPhysicsEnvironment(Environment):
    """Environment that rewards symbolic expressions against task constraints."""

    def __init__(
        self,
        tasks: Sequence[Union[SymbolicPhysicsTask, Dict[str, Any]]],
        max_turns: int = 2,
        reward_fn: Optional[RewardFunction] = None,
        reward_config: Optional[SymbolicRewardConfig] = None,
        success_threshold: float = 0.95,
        seed: Optional[int] = None,
        task_sampling: str = "random",
    ) -> None:
        if not tasks:
            raise ValueError("SymbolicPhysicsEnvironment requires at least one task.")
        self.tasks = [
            task if isinstance(task, SymbolicPhysicsTask) else SymbolicPhysicsTask.from_dict(task)
            for task in tasks
        ]
        self.task_sampling = task_sampling
        self._rng = random.Random(seed)
        self._task_index = 0
        self.success_threshold = success_threshold
        self.scenarios = [task.to_scenario() for task in self.tasks]
        reward_fn = reward_fn or SymbolicPhysicsRewardFunction(config=reward_config)
        super().__init__(max_turns=max_turns, reward_fn=reward_fn)

    def _select_task(self) -> SymbolicPhysicsTask:
        if self.task_sampling == "sequential":
            task = self.tasks[self._task_index % len(self.tasks)]
            self._task_index += 1
            return task
        return self._rng.choice(self.tasks)

    async def reset(self, scenario: Optional[Dict[str, Any]] = None) -> EnvironmentState:
        if scenario:
            task_data = scenario.get("task", scenario)
            task = (
                task_data
                if isinstance(task_data, SymbolicPhysicsTask)
                else SymbolicPhysicsTask.from_dict(task_data)
            )
        else:
            task = self._select_task()

        episode_id = str(uuid.uuid4())
        state = EnvironmentState(
            episode_id=episode_id,
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "task_id": task.task_id,
                "prompt": task.prompt,
                "variables": task.variables,
                "target_expression": task.target_expression,
                "constraints": task.constraints,
                "derived_variables": task.derived_variables,
                "turns": [],
            },
            metadata={"task": task.to_dict()},
        )
        self.active_episodes[episode_id] = state
        self._last_state = state
        return state

    async def get_initial_prompt(self, scenario: Optional[Dict[str, Any]] = None) -> str:
        base = (
            "You are a symbolic physics assistant. "
            "Return a single expression using explicit * and **."
        )
        if scenario and "context" in scenario:
            return f"{base}\n\nTask:\n{scenario['context']}"
        return base

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> Tuple[EnvironmentState, float, bool, Dict[str, Any]]:
        new_state = state.copy()
        new_state.turn_count += 1
        turns = list(new_state.context.get("turns", []))
        turns.append(action)
        new_state.context["turns"] = turns

        reward = 0.0
        if self.reward_fn:
            reward_result = await self.reward_fn.compute_reward(
                turns=turns, context=new_state.context
            )
            reward = float(getattr(reward_result, "score", 0.0))

        done = reward >= self.success_threshold
        if not done and new_state.turn_count >= self.max_turns:
            done = True
            new_state.status = EpisodeStatus.TIMEOUT
        elif done:
            new_state.status = EpisodeStatus.COMPLETED
        else:
            new_state.status = EpisodeStatus.ONGOING

        if done:
            user_message = "Acknowledged."
        else:
            user_message = "Constraint violation. Try a concise expression."

        env_turn = ConversationTurn(role="user", content=user_message)
        info = {"env_response": env_turn, "reward": reward}
        self._last_state = new_state
        return new_state, reward, done, info

    def clone(self) -> "SymbolicPhysicsEnvironment":
        return SymbolicPhysicsEnvironment(
            tasks=self.tasks,
            max_turns=self.max_turns,
            reward_fn=self.reward_fn,
            success_threshold=self.success_threshold,
            task_sampling=self.task_sampling,
        )
