"""
Scenario selection helpers for the multi-turn trainer.
"""

from __future__ import annotations

from typing import Any


def get_environment_scenarios(
    environment: Any, config: Any | None
) -> list[dict[str, Any]]:
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


def split_scenarios(
    scenarios: list[dict[str, Any]],
    config: Any | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if not scenarios:
        return [], []

    eval_split = float(getattr(config, "eval_split_size", 0.1) or 0.0)
    if eval_split <= 0.0 or len(scenarios) < 2:
        return scenarios, scenarios

    stratify = bool(getattr(config, "stratify_by_task", True))
    task_key = getattr(config, "task_id_key", "task_id")
    if stratify:
        by_task: dict[str | None, list[dict[str, Any]]] = {}
        for scenario in scenarios:
            task_value = None
            if isinstance(scenario, dict) and task_key in scenario:
                raw_value = scenario.get(task_key)
                task_value = str(raw_value) if raw_value is not None else None
            by_task.setdefault(task_value, []).append(scenario)

        if len(by_task) > 1:
            train: list[dict[str, Any]] = []
            eval_set: list[dict[str, Any]] = []
            for group in by_task.values():
                if len(group) <= 1:
                    train.extend(group)
                    continue
                eval_count = max(1, int(len(group) * eval_split))
                eval_count = min(eval_count, len(group) - 1)
                train.extend(group[:-eval_count])
                eval_set.extend(group[-eval_count:])
            if train and eval_set:
                return train, eval_set

    eval_count = max(1, int(len(scenarios) * eval_split))
    eval_count = min(eval_count, len(scenarios) - 1)
    return scenarios[:-eval_count], scenarios[-eval_count:]


def apply_task_schedule(config: Any | None, scenario: dict[str, Any], index: int) -> None:
    if config is None:
        return

    task_schedule = getattr(config, "task_schedule", None)
    task_switch_steps = int(getattr(config, "task_switch_steps", 0) or 0)
    task_key = getattr(config, "task_id_key", "task_id")
    if not task_schedule or task_switch_steps <= 0:
        return
    if task_key in scenario:
        return

    task_idx = min(index // task_switch_steps, len(task_schedule) - 1)
    scenario[task_key] = task_schedule[task_idx]


def expand_scenarios(
    config: Any | None,
    scenarios: list[dict[str, Any]],
    count: int,
    prefix: str,
) -> list[dict[str, Any]]:
    if not scenarios or count <= 0:
        return []

    expanded: list[dict[str, Any]] = []
    for idx in range(count):
        base = scenarios[idx % len(scenarios)]
        scenario = dict(base)
        if "id" not in scenario:
            scenario["id"] = f"{prefix}_{idx}"
        apply_task_schedule(config, scenario, idx)
        expanded.append(scenario)
    return expanded


def build_training_scenarios(
    environment: Any,
    config: Any | None,
) -> list[dict[str, Any]]:
    num_episodes = getattr(config, "num_episodes", 100)
    env_scenarios = get_environment_scenarios(environment, config)
    if env_scenarios:
        train_base, _ = split_scenarios(env_scenarios, config)
        if not train_base:
            train_base = env_scenarios
        return expand_scenarios(config, train_base, num_episodes, "train")

    scenarios: list[dict[str, Any]] = []
    for i in range(num_episodes):
        scenario = {"id": f"train_{i}", "context": f"Training scenario {i}"}
        apply_task_schedule(config, scenario, i)
        scenarios.append(scenario)
    return scenarios


def build_eval_scenarios(
    environment: Any,
    config: Any | None,
) -> list[dict[str, Any]]:
    env_scenarios = get_environment_scenarios(environment, config)
    if env_scenarios:
        _, eval_base = split_scenarios(env_scenarios, config)
        if not eval_base:
            eval_base = env_scenarios
        eval_scenarios = []
        for idx, base in enumerate(eval_base):
            scenario = dict(base)
            if "id" not in scenario:
                scenario["id"] = f"eval_{idx}"
            apply_task_schedule(config, scenario, idx)
            eval_scenarios.append(scenario)
        return eval_scenarios

    scenarios: list[dict[str, Any]] = []
    for i in range(20):
        scenario = {"id": f"eval_{i}", "context": f"Evaluation scenario {i}"}
        apply_task_schedule(config, scenario, i)
        scenarios.append(scenario)
    return scenarios


__all__ = [
    "apply_task_schedule",
    "build_eval_scenarios",
    "build_training_scenarios",
    "expand_scenarios",
    "get_environment_scenarios",
    "split_scenarios",
]
