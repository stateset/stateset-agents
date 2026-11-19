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
    DecisionTransformerConfig,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("lengths,capacity", [([1, 3, 5], 2), ([1, 1], 4)])
async def test_offline_epoch_losses_count_action_and_successor_targets(
    lengths, capacity
):
    import math

    config = DecisionTransformerConfig(
        n_layer=1,
        n_head=1,
        n_embd=4,
        state_dim=2,
        action_dim=2,
        max_context_length=3,
        batch_size=capacity,
        use_conversation_embeddings=False,
        dropout=0,
        warmup_steps=0,
    )
    trainer = DecisionTransformerTrainer(config, device="cpu")
    rows = [
        TrajectoryData(
            str(i),
            [],
            0.0,
            [0.0] * n,
            metadata={
                "state_embeddings": [[i + 1.0, j / 10] for j in range(n)],
                "action_embeddings": [[j + 1.0, -i] for j in range(n)],
            },
        )
        for i, n in enumerate(lengths)
    ]
    actions, successors, visits = [], [], []

    def observe(_model, args, kwargs, prediction):
        states, targets = args[:2]
        masks = kwargs["attention_mask"]
        for i, length in enumerate(masks.sum(1).int().tolist()):
            visits.append(states[i, 0, 0].item())
            actions.extend(
                (prediction[0][i, :length] - targets[i, :length])
                .detach()
                .square()
                .sum(-1)
                .tolist()
            )
            successors.extend(
                (prediction[1][i, : length - 1] - states[i, 1:length])
                .detach()
                .square()
                .sum(-1)
                .tolist()
            )

    with trainer.model.register_forward_hook(observe, with_kwargs=True):
        result = await trainer.train(rows, num_epochs=1)
    (epoch,) = result["training_metrics"]
    assert sorted(visits) == list(range(1, len(rows) + 1))
    assert (
        epoch["training_step"]
        == trainer.training_step
        == math.ceil(len(rows) / capacity)
    )
    assert epoch["lr"] == trainer.scheduler.get_last_lr()[0]
    expected_action = math.fsum(actions) / len(actions)
    expected_state = math.fsum(successors) / max(len(successors), 1)
    assert epoch["action_loss"] == pytest.approx(expected_action)
    assert epoch["state_loss"] == pytest.approx(expected_state)
    assert result["final_loss"] == pytest.approx(expected_action + 0.1 * expected_state)
