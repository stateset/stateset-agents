"""ConversationDataset.to_offline_rl_format builds per-decision transitions."""

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.data.conversation_dataset import (  # noqa: E402
    ConversationDataset,
    ConversationDatasetConfig,
    ConversationTurnData,
    TrajectoryData,
)


class _TextCache:
    """Embeds a text as [len(text)] so tests can see which prefix was used."""

    def get_or_compute(self, key, text):
        return torch.tensor([float(len(text))])


def _dataset(trajs):
    config = ConversationDatasetConfig(normalize_rewards=False, compute_returns=False)
    return ConversationDataset(trajs, config=config, embedding_cache=_TextCache())


def _turns(*pairs):
    return [ConversationTurnData(role=r, content=c) for r, c in pairs]


def test_terminal_flag_and_reward_when_conversation_ends_on_user_turn():
    turns = _turns(("user", "u1"), ("assistant", "a1"), ("user", "thanks"))
    traj = TrajectoryData("t", turns, total_reward=1.0, turn_rewards=[0.0, 0.0, 1.0])
    data = _dataset([traj]).to_offline_rl_format()
    assert data["dones"].tolist() == [1.0]
    assert data["rewards"].tolist() == [1.0]


def test_next_state_is_the_next_decision_state():
    turns = _turns(
        ("user", "u1"), ("assistant", "a1"), ("user", "u2"), ("assistant", "a2")
    )
    traj = TrajectoryData("t", turns, total_reward=0.0, turn_rewards=[0.0] * 4)
    data = _dataset([traj]).to_offline_rl_format()
    # next state of decision 0 is exactly the state of decision 1
    assert data["next_states"][0].tolist() == data["states"][1].tolist()
    assert data["dones"].tolist() == [0.0, 1.0]


def test_assistant_first_trajectory_is_rejected_without_an_observation():
    first = TrajectoryData(
        "a", _turns(("user", "long prompt text"), ("assistant", "x")), 0.0, [0.0, 0.0]
    )
    second = TrajectoryData("b", _turns(("assistant", "hello")), 0.0, [0.0])
    with pytest.raises(ValueError, match="preceding observation"):
        _dataset([first, second]).to_offline_rl_format()


def test_sharegpt_roles_are_canonicalised():
    item = {
        "conversations": [
            {"from": "human", "value": "hi"},
            {"from": "gpt", "value": "hello"},
        ],
        "reward": 1.0,
    }
    traj = ConversationDataset._parse_hf_item(item, 0, "conversations", "reward")
    assert [t.role for t in traj.turns] == ["user", "assistant"]
