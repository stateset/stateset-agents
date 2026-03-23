"""State and scenario helpers for the single-turn trainer."""

from __future__ import annotations

from typing import Any

from stateset_agents.core.environment import EnvironmentState


def get_environment_scenarios(environment: Any, config: Any) -> list[dict[str, Any]]:
    """Return sanitized environment scenarios for trainer consumption."""
    scenarios = getattr(environment, "scenarios", None)
    if not isinstance(scenarios, list):
        return []

    cleaned: list[dict[str, Any]] = []
    for idx, scenario in enumerate(scenarios):
        if scenario is None:
            continue
        if isinstance(scenario, dict):
            cleaned.append(dict(scenario))
        else:
            cleaned.append({"id": f"scenario_{idx}", "context": str(scenario)})

    max_examples = getattr(config, "max_examples", None)
    if isinstance(max_examples, int) and max_examples > 0:
        cleaned = cleaned[:max_examples]
    return cleaned


def apply_task_schedule(config: Any, scenario: dict[str, Any], episode: int) -> None:
    """Inject scheduled task ids into a scenario when configured."""
    if config is None:
        return

    task_schedule = getattr(config, "task_schedule", None)
    task_switch_steps = int(getattr(config, "task_switch_steps", 0) or 0)
    task_key = getattr(config, "task_id_key", "task_id")
    if not task_schedule or task_switch_steps <= 0 or task_key in scenario:
        return

    task_idx = min(episode // task_switch_steps, len(task_schedule) - 1)
    scenario[task_key] = task_schedule[task_idx]


def get_episode_scenario(
    environment: Any,
    config: Any,
    episode: int,
) -> dict[str, Any] | None:
    """Resolve the scenario for a specific training episode."""
    scenarios = get_environment_scenarios(environment, config)
    scenario: dict[str, Any] | None = None
    if scenarios:
        scenario = dict(scenarios[episode % len(scenarios)])
    else:
        task_schedule = getattr(config, "task_schedule", None)
        task_switch_steps = int(getattr(config, "task_switch_steps", 0) or 0)
        task_key = getattr(config, "task_id_key", "task_id")
        if task_schedule and task_switch_steps > 0:
            task_idx = min(episode // task_switch_steps, len(task_schedule) - 1)
            scenario = {task_key: task_schedule[task_idx]}

    if scenario is not None:
        apply_task_schedule(config, scenario, episode)
    return scenario


def resolve_task_id(config: Any, context: dict[str, Any] | None) -> str | None:
    """Resolve the active task id from state or scenario context."""
    if context is None or config is None:
        return None

    task_key = getattr(config, "task_id_key", "task_id")
    if task_key and task_key in context:
        value = context.get(task_key)
        return str(value) if value is not None else None

    scenario = context.get("scenario")
    if isinstance(scenario, dict) and task_key in scenario:
        value = scenario.get(task_key)
        return str(value) if value is not None else None

    return None


def extract_context(state: Any) -> dict[str, Any] | None:
    """Extract a context mapping from the environment state."""
    if isinstance(state, EnvironmentState):
        return state.context
    if isinstance(state, dict):
        return state
    return None


def extract_prompt(state: Any, context: dict[str, Any] | None) -> str:
    """Derive the current prompt from state or scenario context."""
    if isinstance(state, dict):
        if "prompt" in state:
            return str(state["prompt"])
        nested_state = state.get("state")
        if isinstance(nested_state, dict) and "prompt" in nested_state:
            return str(nested_state["prompt"])

    if context:
        prompt = context.get("prompt")
        if prompt:
            return str(prompt)
        scenario = context.get("scenario")
        if isinstance(scenario, dict):
            if scenario.get("prompt"):
                return str(scenario["prompt"])
            if scenario.get("context"):
                return str(scenario["context"])
        task = context.get("task")
        if isinstance(task, dict) and task.get("description"):
            return str(task["description"])

    return "Hello"


def merge_scenario_into_state(
    state: Any,
    scenario: dict[str, Any] | None,
    config: Any,
) -> Any:
    """Ensure selected scenario metadata survives across dict-based env states."""
    if scenario is None:
        return state

    task_key = getattr(config, "task_id_key", "task_id") if config is not None else "task_id"
    scenario_copy = dict(scenario)
    task_value = scenario_copy.get(task_key)

    if isinstance(state, EnvironmentState):
        state.context.setdefault("scenario", scenario_copy)
        if task_value is not None:
            state.context.setdefault(task_key, task_value)
        if scenario_copy.get("prompt") is not None:
            state.context.setdefault("prompt", scenario_copy.get("prompt"))
        elif scenario_copy.get("context") is not None:
            state.context.setdefault("prompt", scenario_copy.get("context"))
        return state

    if isinstance(state, dict):
        state.setdefault("scenario", scenario_copy)
        if task_value is not None:
            state.setdefault(task_key, task_value)
        if scenario_copy.get("prompt") is not None:
            state.setdefault("prompt", scenario_copy.get("prompt"))
        elif scenario_copy.get("context") is not None:
            state.setdefault("prompt", scenario_copy.get("context"))
        return state

    return state


__all__ = [
    "apply_task_schedule",
    "extract_context",
    "extract_prompt",
    "get_environment_scenarios",
    "get_episode_scenario",
    "merge_scenario_into_state",
    "resolve_task_id",
]
