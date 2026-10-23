"""``spine.loop.run_dedicated_loop``: the dedicated entry points' outer loop."""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.training.spine.loop import run_dedicated_loop


def _recorder() -> tuple[list[str], dict[str, Any]]:
    events: list[str] = []

    async def step(batch: Any) -> dict[str, Any]:
        events.append(f"step {batch}")
        return {"batch": batch}

    def report(iteration: int, metrics: dict[str, Any]) -> None:
        events.append(f"report {iteration}")

    def save(path: str) -> None:
        events.append(f"save {os.path.basename(path)}")

    return events, {"step": step, "report": report, "save": save}


async def test_iteration_order_save_cadence_and_final_save() -> None:
    events, hooks = _recorder()
    run = await run_dedicated_loop(
        iterations=4,
        batch=lambda i: f"b{i}",
        output_dir="/out",
        save_steps=2,
        final_name="final",
        **hooks,
    )
    assert events == [
        "step b0",
        "report 0",
        "step b1",
        "report 1",
        "save checkpoint-2",
        "step b2",
        "report 2",
        "step b3",
        "report 3",
        "save checkpoint-4",
        "save final",
    ]
    assert (run.iterations, run.aborted, run.metrics) == (4, False, {"batch": "b3"})


async def test_callbacks_run_after_the_save_and_may_abort() -> None:
    events, hooks = _recorder()

    async def after_save(iteration: int, metrics: dict[str, Any]) -> bool:
        events.append(f"callbacks {iteration}")
        return iteration == 1

    run = await run_dedicated_loop(
        iterations=5,
        batch=lambda i: i,
        output_dir="/out",
        save_steps=2,
        final_name="final_model",
        after_save=after_save,
        **hooks,
    )
    assert events[-4:] == [
        "report 1",
        "save checkpoint-2",
        "callbacks 1",
        "save final_model",
    ]
    assert run.aborted and run.iterations == 2


async def test_a_missing_batch_ends_training_with_a_final_save() -> None:
    events, hooks = _recorder()
    run = await run_dedicated_loop(
        iterations=3,
        batch=lambda i: None,
        output_dir="/out",
        save_steps=1,
        final_name="final",
        **hooks,
    )
    assert events == ["save final"] and run.iterations == 0


async def test_a_failed_step_propagates_without_a_final_save() -> None:
    saved: list[str] = []

    async def step(batch: Any) -> dict[str, Any]:
        raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        await run_dedicated_loop(
            iterations=2,
            batch=lambda i: i,
            step=step,
            save=saved.append,
            output_dir="/out",
            save_steps=1,
            final_name="final",
        )
    assert saved == []


async def test_train_with_gspo_honours_use_gspo_token(monkeypatch, tmp_path) -> None:
    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
    from stateset_agents.core.environment import ConversationEnvironment
    from stateset_agents.training import gspo_entrypoints as ep
    from stateset_agents.training import gspo_token_trainer as tok
    from stateset_agents.training.gspo_config import GSPOConfig

    calls: list[str] = []

    class FakeTokenTrainer:
        def __init__(self, **kwargs: Any) -> None:
            self.training_metrics: dict[str, list[float]] = {}
            self.generator = SimpleNamespace()

        async def train_step(self, queries: Any, num_groups: int = 1) -> dict:
            calls.append("train_step")
            return {}

        async def train_step_token_level(
            self, queries: Any, num_groups: int = 1
        ) -> dict:
            calls.append("train_step_token_level")
            return {}

        def save_model(self, path: str) -> None:
            calls.append(f"save {os.path.basename(path)}")

    monkeypatch.setattr(tok, "GSPOTokenTrainer", FakeTokenTrainer)
    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://x", use_stub_model=True, stub_responses=["x"])
    )
    await agent.initialize()
    config = GSPOConfig(
        model_name="stub://x",
        output_dir=str(tmp_path),
        report_to="none",
        num_outer_iterations=2,
        save_steps=100,
        use_gspo_token=True,
    )
    await ep.train_with_gspo(
        config=config,
        agent=agent,
        environment=ConversationEnvironment(scenarios=[{"user_query": "Q"}]),
        reward_model=SimpleNamespace(),
    )
    assert calls == [
        "train_step_token_level",
        "train_step_token_level",
        "save final_model",
    ]
