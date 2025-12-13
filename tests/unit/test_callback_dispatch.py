from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from stateset_agents.training.callbacks import (
    notify_episode_end,
    notify_training_end,
    notify_training_start,
)
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.diagnostics import DiagnosticsMonitor


@dataclass
class EpisodeRecorder:
    calls: List[Tuple[int, Dict[str, Any]]]

    def on_episode_end(self, episode: int, metrics: Dict[str, Any]) -> None:
        self.calls.append((episode, dict(metrics)))


async def test_notify_episode_end_dispatches_callable_and_method_callbacks() -> None:
    cfg = TrainingConfig(num_episodes=1)
    diagnostics = DiagnosticsMonitor(cfg)
    recorder = EpisodeRecorder(calls=[])
    callbacks: List[Any] = [diagnostics, recorder]

    await notify_training_start(callbacks, trainer="trainer", config=cfg)
    await notify_episode_end(callbacks, episode=0, metrics={"total_reward": 1.23})
    await notify_training_end(callbacks, metrics={"final_step": 1})

    assert diagnostics.episode_count == 1
    assert diagnostics.total_rewards == [1.23]
    assert recorder.calls == [(0, {"total_reward": 1.23})]

