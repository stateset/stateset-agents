"""State/action features and supervised token rows must describe the same actions."""

from types import SimpleNamespace

import pytest
import torch

from stateset_agents.data.conversation_dataset import (
    ConversationDataset,
    ConversationDatasetConfig,
)
from stateset_agents.training.sampled_workflow import workflow_tokenizer


def dataset():
    return ConversationDataset.from_dict(
        {
            "trajectories": [
                {
                    "trajectory_id": "trace",
                    "turns": [
                        {"role": "user", "content": "request"},
                        {"role": "assistant", "content": "lookup", "reward": 0},
                        {"role": "tool", "content": "eligible", "reward": 0},
                        {"role": "assistant", "content": "refund", "reward": 1},
                        {"role": "user", "content": "resolved", "reward": 2},
                    ],
                }
            ]
        },
        config=ConversationDatasetConfig(
            normalize_rewards=False, compute_returns=False
        ),
    )


def test_next_state_includes_tool_observations_and_terminal_reward():
    data = dataset()
    rows = data.policy_transitions()
    assert rows[0]["next_history"][-1] == {"role": "tool", "content": "eligible"}
    assert rows[1]["reward"] == 3 and rows[1]["done"] is True
    texts = []

    def embed(key, text):
        texts.append(text)
        return torch.ones(2)

    data.embedding_cache = SimpleNamespace(get_or_compute=embed)
    features = data.to_offline_rl_format()
    assert features["states"].shape == (2, 2)
    assert features["dones"].tolist() == [0, 1]
    assert "tool: eligible" in texts[2] and "tool: eligible" in texts[3]


def test_only_current_action_is_supervised_in_each_token_row():
    tokenizer = workflow_tokenizer()
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )
    rows = dataset().to_policy_training_format(tokenizer, 16)
    assert [token for token in rows[0]["labels"] if token != -100] == tokenizer.encode(
        "lookup", add_special_tokens=False
    )
    assert [token for token in rows[1]["labels"] if token != -100] == tokenizer.encode(
        "refund", add_special_tokens=False
    )
    assert rows[0]["transition_id"] != rows[1]["transition_id"]


def test_truncating_a_logged_action_is_rejected():
    tokenizer = workflow_tokenizer()
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )
    with pytest.raises(ValueError, match="max_length"):
        dataset().to_policy_training_format(tokenizer, 2)
