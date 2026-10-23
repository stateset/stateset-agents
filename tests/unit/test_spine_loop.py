"""``spine.loop.SpineLoopMixin``: the GRPO outer loop moved out of MultiTurn.

The phase names ``train()`` passes to ``_training_phase`` -- and their order --
are the protocol ``DistributedTrainer`` uses to coordinate ranks, so a run's
phase sequence is pinned here. The ``TrainingRuntime.__init_subclass__``
wrappers must still apply exactly once to the inherited ``train``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
from stateset_agents.training.spine import SpineLoopMixin

#: Episodes 0-1 run task "a", episode 2 task "b" (a task boundary flush);
#: logging every update, evaluation every 2 updates, a final flush.
EXPECTED_PHASES = [
    "scheduler",
    "evaluation-scenarios",
    "training-start",
    # episode 0
    "task-id",
    "replay",
    "episode-end",
    "optimizer-step",
    "training-log",
    # episode 1
    "task-id",
    "replay",
    "episode-end",
    "optimizer-step",
    "training-log",
    "evaluation-log",
    "evaluation-end",
    # episode 2: the switch to task "b" commits a boundary update first
    "task-id",
    "optimizer-step",
    "training-log",
    "replay",
    "episode-end",
    "optimizer-step",
    "training-log",
    "evaluation-log",
    "evaluation-end",
    # finalization
    "optimizer-step",
    "training-log",
    "training-end",
    "tracking-finish",
]


def _trainer() -> tuple[MultiTurnGRPOTrainer, list[str]]:
    trainer = MultiTurnGRPOTrainer(
        agent=SimpleNamespace(model=None),
        environment=MagicMock(),
        config=TrainingConfig(
            report_to="none",
            num_episodes=3,
            gradient_accumulation_steps=1,
            logging_steps=1,
            eval_steps=2,
            save_steps=100,
        ),
    )
    tasks = {0: "a", 1: "a", 2: "b"}

    def bump() -> None:
        trainer.global_step += 1

    def switch(task_id: str | None) -> None:
        if trainer._current_task_id not in (None, task_id):
            bump()  # flushing the outgoing task commits an update
        trainer._current_task_id = task_id

    async def training_step(groups: list[Any]) -> dict[str, Any]:
        bump()
        return {"optimizer_step": True, "optimizer_updates": 1, "policy_loss": 0.5}

    trainer._setup_scheduler = MagicMock()  # type: ignore[method-assign]
    trainer._get_training_scenarios = MagicMock(  # type: ignore[method-assign]
        return_value=[{"id": i} for i in range(3)]
    )
    trainer._get_eval_scenarios = MagicMock(return_value=[])  # type: ignore[method-assign]
    trainer._resolve_task_id = lambda scenario: tasks[scenario["id"]]  # type: ignore[method-assign]
    trainer._maybe_handle_task_switch = switch  # type: ignore[method-assign]
    trainer.generate_trajectories = AsyncMock(  # type: ignore[method-assign]
        return_value=[SimpleNamespace(trajectories=[])]
    )
    trainer._maybe_mix_replay = lambda groups, task: (groups, [])  # type: ignore[method-assign]
    trainer.training_step = training_step  # type: ignore[method-assign]
    trainer.evaluate = AsyncMock(return_value={"eval_reward": 0.0})  # type: ignore[method-assign]
    trainer._finish_current_task = bump  # type: ignore[method-assign]
    trainer.save_checkpoint = AsyncMock()  # type: ignore[method-assign]

    phases: list[str] = []

    async def record(name: str, function: Any, *args: Any, **kwargs: Any) -> Any:
        phases.append(name)
        return await SpineLoopMixin._training_phase(
            trainer, name, function, *args, **kwargs
        )

    trainer._training_phase = record  # type: ignore[method-assign]
    return trainer, phases


@pytest.mark.asyncio
async def test_training_phase_protocol_is_unchanged() -> None:
    trainer, phases = _trainer()
    assert await trainer.train() is trainer.agent
    assert phases == EXPECTED_PHASES
    assert trainer.global_step == 5  # 3 batches + boundary + final flush
    trainer.save_checkpoint.assert_awaited_once_with()  # final save only


def test_multi_turn_inherits_the_spine_loop() -> None:
    assert issubclass(MultiTurnGRPOTrainer, SpineLoopMixin)
    for name in ("train", "_training_phase", "_optimizer_update_receipt"):
        assert name not in MultiTurnGRPOTrainer.__dict__
        assert getattr(MultiTurnGRPOTrainer, name) is getattr(SpineLoopMixin, name)


def _wrap_depth(function: Any) -> int:
    depth = 0
    while hasattr(function, "__wrapped__"):
        function = function.__wrapped__
        depth += 1
    return depth


def test_runtime_wrappers_apply_once() -> None:
    """``TrainingRuntime.__init_subclass__`` guards ``train`` where it is
    defined; inheriting it must not wrap it again."""
    assert _wrap_depth(SpineLoopMixin.__dict__["train"]) == 1
    assert _wrap_depth(MultiTurnGRPOTrainer.train) == 1
    from stateset_agents.training.distributed import DistributedTrainer

    assert _wrap_depth(DistributedTrainer.__dict__["train"]) == 1
    # training_step: profile_learner_step plus the runtime phase profiler
    assert _wrap_depth(MultiTurnGRPOTrainer.__dict__["training_step"]) == 2
