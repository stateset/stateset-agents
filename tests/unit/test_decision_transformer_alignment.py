"""Each Decision Transformer action must be paired with the full history that
preceded it and with the return-to-go at its own turn."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.data.conversation_dataset import (  # noqa: E402
    ConversationTurnData,
    TrajectoryData,
)
from stateset_agents.experimental.offline_rl.decision_transformer import (  # noqa: E402
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
    trainer.config = DecisionTransformerConfig()
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


def _tiny_trainer(**overrides):
    config = DecisionTransformerConfig(
        n_layer=1,
        n_head=1,
        n_embd=8,
        dropout=0.0,
        state_dim=2,
        action_dim=2,
        use_conversation_embeddings=False,
        warmup_steps=1,
        **overrides,
    )
    trainer = DecisionTransformerTrainer(config, device="cpu")
    trainer.embedder = _RecordingEmbedder()
    return trainer


class _CapturingModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = []

    def get_action(self, states, actions, returns_to_go, timesteps):
        self.calls.append((states, actions, returns_to_go, timesteps))
        return torch.zeros(1, actions.shape[-1])


def test_generation_context_matches_training_states():
    trainer = _tiny_trainer()
    history = [
        {"role": "user", "content": "u1"},
        {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "u2"},
    ]
    turns = [ConversationTurnData(**t) for t in history] + [
        ConversationTurnData(role="assistant", content="a2")
    ]
    traj = TrajectoryData(
        trajectory_id="t",
        turns=turns,
        total_reward=1.0,
        turn_rewards=[0.0] * 4,
        returns_to_go=[1.0] * 4,
    )
    trainer._prepare_dataset([traj])
    training_texts = list(trainer.embedder.texts)

    trainer.embedder = _RecordingEmbedder()
    trainer.model = _CapturingModel()
    trainer.generate_response(history, target_return=1.0)

    # Same text for the prior state, prior action and current state as training.
    assert trainer.embedder.texts == training_texts[:3]
    states, actions, rtg, timesteps = trainer.model.calls[0]
    assert states.shape[1] == 2
    # Past state is the real embedding of the pre-a1 history, not zeros.
    assert torch.all(states[0, 0] == 1.0)
    assert torch.all(actions[0, 0] == 2.0)
    assert torch.all(states[0, 1] == 3.0)
    assert timesteps.tolist() == [[0, 1]]
    assert rtg[0, :, 0].tolist() == [1.0, 1.0]


def test_generation_keeps_last_context_window_and_valid_timesteps():
    trainer = _tiny_trainer(max_context_length=2, max_episode_length=3)
    history = []
    for i in range(5):
        history += [
            {"role": "user", "content": f"u{i}"},
            {"role": "assistant", "content": f"a{i}"},
        ]
    history.append({"role": "user", "content": "last"})
    trainer.model = _CapturingModel()
    trainer.generate_response(history, target_return=0.5)
    states, _, _, timesteps = trainer.model.calls[0]
    assert states.shape[1] == 2
    assert int(timesteps.max()) < 3


def test_generation_rejects_misaligned_past_actions():
    trainer = _tiny_trainer()
    history = [{"role": "user", "content": "hi"}]
    with pytest.raises(ValueError, match="past_actions"):
        trainer.generate_response(
            history, target_return=1.0, past_actions=[torch.zeros(2)]
        )


def test_long_trajectories_are_split_into_context_windows():
    trainer = _tiny_trainer(max_context_length=2, max_episode_length=10)
    turns = []
    for i in range(5):
        turns += [
            ConversationTurnData(role="user", content=f"u{i}"),
            ConversationTurnData(role="assistant", content=f"a{i}"),
        ]
    traj = TrajectoryData(
        trajectory_id="t",
        turns=turns,
        total_reward=1.0,
        turn_rewards=[0.0] * 10,
        returns_to_go=[float(10 - i) for i in range(10)],
    )
    windows = trainer._prepare_dataset([traj])
    assert [w["timesteps"].tolist() for w in windows] == [[0, 1], [2, 3], [4]]
    assert windows[1]["returns_to_go"].tolist() == [5.0, 3.0]


@pytest.mark.asyncio
async def test_train_runs_when_dataset_is_smaller_than_a_batch():
    trainer = _tiny_trainer(batch_size=64)
    data = [
        {
            "states": torch.ones(2, 2).numpy(),
            "actions": torch.ones(2, 2).numpy(),
            "returns_to_go": torch.ones(2).numpy(),
            "timesteps": torch.arange(2).numpy(),
        }
    ] * 3
    trainer._prepare_dataset = lambda dataset: data
    result = await trainer.train(data, num_epochs=1)
    assert trainer.training_step == 1
    assert result["final_loss"] >= 0.0


def test_checkpoint_restores_embedding_projection(tmp_path):
    trainer = _tiny_trainer()
    projection = torch.nn.Linear(2, 2)
    trainer.embedder = torch.nn.Module()
    trainer.embedder.projection = projection
    path = tmp_path / "dt.pt"
    trainer.save(str(path))

    restored = _tiny_trainer()
    restored.embedder = torch.nn.Module()
    restored.embedder.projection = torch.nn.Linear(2, 2)
    restored.load(str(path))
    assert torch.equal(restored.embedder.projection.weight, projection.weight)
