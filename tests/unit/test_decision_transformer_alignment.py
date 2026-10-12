"""Each Decision Transformer action must be paired with the full history that
preceded it and with the return-to-go at its own turn."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.data.conversation_dataset import (  # noqa: E402
    ConversationTurnData,
    TrajectoryData,
)
from stateset_agents.training.decision_transformer import (  # noqa: E402
    DecisionTransformerTrainer,
)


class _RecordingEmbedder:
    def __init__(self):
        self.texts: list[str] = []

    def forward(self, texts):
        self.texts.append(texts[0])
        return torch.full((1, 2), float(len(self.texts)))


def test_states_and_returns_align_with_assistant_turns():
    trainer = DecisionTransformerTrainer.__new__(DecisionTransformerTrainer)
    trainer.embedder = _RecordingEmbedder()
    turns = [
        ConversationTurnData(role="system", content="S"),
        ConversationTurnData(role="user", content="u1"),
        ConversationTurnData(role="assistant", content="a1"),
        ConversationTurnData(role="user", content="u2"),
        ConversationTurnData(role="assistant", content="a2"),
    ]
    traj = TrajectoryData(
        trajectory_id="t",
        turns=turns,
        total_reward=3.0,
        turn_rewards=[0.0, 0.0, 1.0, 0.0, 2.0],
        returns_to_go=[10.0, 10.0, 10.0, 9.0, 9.0],
    )
    [data] = trainer._prepare_dataset([traj])
    texts = trainer.embedder.texts
    # state for a1, a1, state for a2, a2
    assert texts == [
        "system: S\nuser: u1\n",
        "a1",
        "system: S\nuser: u1\nassistant: a1\nuser: u2\n",
        "a2",
    ]
    assert data["returns_to_go"].tolist() == [10.0, 9.0]
    assert data["states"].shape[0] == data["actions"].shape[0] == 2
