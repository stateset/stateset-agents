"""
Long-term planning utilities for conversational agents.

Provides a lightweight planning manager that can generate and maintain
multi-step plans, then inject them into the prompt as system context.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class PlanStatus(str, Enum):
    """Status for individual plan steps."""

    PENDING = "pending"
    ACTIVE = "active"
    DONE = "done"
    BLOCKED = "blocked"


@dataclass
class PlanStep:
    """Single step in a plan."""

    description: str
    status: PlanStatus = PlanStatus.PENDING
    step_id: str = ""
    notes: Optional[str] = None
    updated_at: datetime = field(default_factory=datetime.now)

    def mark(self, status: PlanStatus, notes: Optional[str] = None) -> None:
        self.status = status
        if notes:
            self.notes = notes
        self.updated_at = datetime.now()


@dataclass
class Plan:
    """Plan with ordered steps for a goal."""

    goal: str
    steps: List[PlanStep] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def ensure_active(self) -> None:
        for step in self.steps:
            if step.status == PlanStatus.ACTIVE:
                return
        for step in self.steps:
            if step.status == PlanStatus.PENDING:
                step.status = PlanStatus.ACTIVE
                return

    def advance(self) -> None:
        """Mark active step done and advance to the next pending step."""
        for step in self.steps:
            if step.status == PlanStatus.ACTIVE:
                step.mark(PlanStatus.DONE)
                break
        for step in self.steps:
            if step.status == PlanStatus.PENDING:
                step.mark(PlanStatus.ACTIVE)
                break
        self.updated_at = datetime.now()

    def update_step(
        self,
        step_id: Optional[str] = None,
        step_index: Optional[int] = None,
        status: Optional[PlanStatus] = None,
        notes: Optional[str] = None,
    ) -> None:
        step = None
        if step_id:
            for candidate in self.steps:
                if candidate.step_id == step_id:
                    step = candidate
                    break
        elif step_index is not None and 0 <= step_index < len(self.steps):
            step = self.steps[step_index]

        if step is None:
            return

        if status is not None:
            step.mark(status, notes=notes)
        elif notes:
            step.notes = notes
        self.updated_at = datetime.now()

    def progress(self) -> float:
        if not self.steps:
            return 0.0
        done = sum(1 for step in self.steps if step.status == PlanStatus.DONE)
        return float(done) / float(len(self.steps))

    def summarize(
        self,
        max_steps: int = 6,
        keep_completed: int = 2,
        include_goal: bool = True,
    ) -> str:
        lines = []
        if include_goal:
            lines.append(f"Goal: {self.goal}")

        completed = [s for s in self.steps if s.status == PlanStatus.DONE]
        active_or_pending = [s for s in self.steps if s.status != PlanStatus.DONE]

        display_steps: List[PlanStep] = []
        if keep_completed > 0:
            display_steps.extend(completed[-keep_completed:])
        display_steps.extend(active_or_pending)
        display_steps = display_steps[:max_steps]

        if display_steps:
            lines.append("Steps:")
            for idx, step in enumerate(display_steps, 1):
                status_tag = {
                    PlanStatus.DONE: "[x]",
                    PlanStatus.ACTIVE: "[>]",
                    PlanStatus.BLOCKED: "[!]",
                    PlanStatus.PENDING: "[ ]",
                }.get(step.status, "[ ]")
                lines.append(f"{idx}. {status_tag} {step.description}")

        return "\n".join(lines).strip()


@dataclass
class PlanningConfig:
    """Configuration for long-term planning."""

    enabled: bool = False
    max_steps: int = 6
    update_interval: int = 1
    include_plan_in_context: bool = True
    allow_goal_update: bool = True
    goal_min_chars: int = 8
    keep_completed: int = 2
    plan_prefix: str = "Long-term plan"


class Planner:
    """Planner interface for generating and revising plans."""

    def create_plan(self, goal: str, config: PlanningConfig) -> Plan:
        raise NotImplementedError

    def revise_plan(self, plan: Plan, goal: str, config: PlanningConfig) -> Plan:
        return self.create_plan(goal, config)


class HeuristicPlanner:
    """Simple deterministic planner that splits goals into steps."""

    def create_plan(self, goal: str, config: PlanningConfig) -> Plan:
        steps = _split_goal_into_steps(goal, config.max_steps)
        plan_steps = [
            PlanStep(description=step, step_id=f"step_{idx + 1}")
            for idx, step in enumerate(steps)
        ]
        plan = Plan(goal=goal, steps=plan_steps)
        plan.ensure_active()
        return plan

    def revise_plan(self, plan: Plan, goal: str, config: PlanningConfig) -> Plan:
        if goal != plan.goal:
            return self.create_plan(goal, config)
        return plan


class PlanningManager:
    """Manages long-term plans across conversations."""

    def __init__(
        self,
        config: Optional[PlanningConfig] = None,
        planner: Optional[Any] = None,
        goal_extractor: Optional[Callable[[List[Dict[str, str]], Dict[str, Any]], Optional[str]]] = None,
    ):
        self.config = config or PlanningConfig()
        self.planner = planner or HeuristicPlanner()
        self.goal_extractor = goal_extractor or _default_goal_extractor
        self._plans: Dict[str, Plan] = {}
        self._turn_counters: Dict[str, int] = {}
        self._last_updates: Dict[str, int] = {}

    def get_plan(self, conversation_id: str) -> Optional[Plan]:
        return self._plans.get(conversation_id)

    def clear_conversation(self, conversation_id: str) -> None:
        """Remove planning state for a conversation."""
        self._plans.pop(conversation_id, None)
        self._turn_counters.pop(conversation_id, None)
        self._last_updates.pop(conversation_id, None)

    def build_plan_message(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, str]]:
        if not self.config.enabled or not self.config.include_plan_in_context:
            return None

        context = context or {}
        conversation_id = str(context.get("conversation_id") or "default")
        turn_count = self._turn_counters.get(conversation_id, 0) + 1
        self._turn_counters[conversation_id] = turn_count

        plan = self._plans.get(conversation_id)
        has_plan_update = isinstance(context.get("plan_update"), dict)
        goal_override = context.get("plan_goal")
        has_goal_override = isinstance(goal_override, str) and goal_override.strip()
        update_needed = (
            plan is None
            or turn_count - self._last_updates.get(conversation_id, 0)
            >= self.config.update_interval
            or has_plan_update
            or has_goal_override
        )

        if update_needed:
            plan = self._ensure_plan(conversation_id, messages, context, plan)
            self._last_updates[conversation_id] = turn_count

        if plan is None:
            return None

        summary = plan.summarize(
            max_steps=self.config.max_steps,
            keep_completed=self.config.keep_completed,
        )
        if not summary:
            return None

        content = f"{self.config.plan_prefix}:\n{summary}"
        return {"role": "system", "content": content}

    def apply_update(
        self,
        conversation_id: str,
        update: Dict[str, Any],
    ) -> None:
        plan = self._plans.get(conversation_id)
        if plan is None:
            return

        action = str(update.get("action") or "").lower()
        if action == "advance":
            plan.advance()
            return

        status = update.get("status")
        if isinstance(status, str):
            try:
                status_value = PlanStatus(status)
            except ValueError:
                status_value = None
        else:
            status_value = status

        step_index = update.get("step_index")
        if isinstance(step_index, int) and step_index > 0:
            step_index = step_index - 1

        plan.update_step(
            step_id=update.get("step_id"),
            step_index=step_index,
            status=status_value if isinstance(status_value, PlanStatus) else None,
            notes=update.get("notes"),
        )

    def _ensure_plan(
        self,
        conversation_id: str,
        messages: List[Dict[str, str]],
        context: Dict[str, Any],
        current_plan: Optional[Plan],
    ) -> Optional[Plan]:
        update = context.get("plan_update")
        if isinstance(update, dict):
            self.apply_update(conversation_id, update)

        explicit_goal = _extract_explicit_goal(context)
        if explicit_goal:
            goal = explicit_goal
        elif current_plan is None:
            goal = self.goal_extractor(messages, context)
        else:
            goal = None

        if not goal or len(goal) < self.config.goal_min_chars:
            return current_plan

        if current_plan is None:
            current_plan = self.planner.create_plan(goal, self.config)
        elif self.config.allow_goal_update and explicit_goal:
            current_plan = self.planner.revise_plan(current_plan, goal, self.config)

        self._plans[conversation_id] = current_plan
        return current_plan


def _default_goal_extractor(
    messages: List[Dict[str, str]],
    context: Dict[str, Any],
) -> Optional[str]:
    explicit = _extract_explicit_goal(context)
    if explicit:
        return explicit

    for msg in reversed(messages):
        if msg.get("role") == "user":
            content = str(msg.get("content") or "").strip()
            if content:
                return content
    return None


def _extract_explicit_goal(context: Dict[str, Any]) -> Optional[str]:
    explicit = context.get("plan_goal")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    explicit = context.get("goal")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    user_goal = context.get("user_goal")
    if isinstance(user_goal, str) and user_goal.strip():
        return user_goal.strip()
    scenario = context.get("scenario")
    if isinstance(scenario, dict):
        for key in ("goal", "user_goal", "context"):
            value = scenario.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
    return None


def _split_goal_into_steps(goal: str, max_steps: int) -> List[str]:
    cleaned = goal.strip()
    if not cleaned:
        return []

    parts = re.split(r"\.\s+|;|,|\band\b|\bthen\b|\bnext\b", cleaned, flags=re.IGNORECASE)
    steps = [part.strip() for part in parts if part.strip()]

    if not steps:
        steps = [cleaned]

    return steps[:max_steps]
