"""Decision Transformer learns return-conditioned behaviour.

Logged single-turn episodes take either the GOOD action (return 1) or the BAD
action (return 0) in random states. Conditioned on a high target return the
trained model must emit (close to) the GOOD action; conditioned on a low one,
the BAD action. A model that ignores the return-to-go collapses to the mean
action (0, 0) and fails both directions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tests.learning._offline_toy import (
    ACTION_DIM,
    BAD_ACTION,
    GOOD_ACTION,
    STATE_DIM,
    heldout_states,
)
from tests.learning._toy import seeds

pytestmark = pytest.mark.learning


def _dataset(seed: int, num_states: int = 64) -> list:
    from stateset_agents.data.conversation_dataset import (
        ConversationTurnData,
        TrajectoryData,
    )

    rng = np.random.default_rng(seed)
    data = []
    for i in range(num_states):
        state = rng.normal(size=STATE_DIM).astype(np.float32)
        for action, ret in ((GOOD_ACTION, 1.0), (BAD_ACTION, 0.0)):
            data.append(
                TrajectoryData(
                    trajectory_id=f"{i}-{ret}",
                    turns=[
                        ConversationTurnData(role="user", content="q"),
                        ConversationTurnData(role="assistant", content="a"),
                    ],
                    total_reward=ret,
                    turn_rewards=[ret],
                    returns_to_go=[ret],
                    metadata={
                        "state_embeddings": [state],
                        "action_embeddings": [action],
                    },
                )
            )
    return data


def _train(seed: int):
    import asyncio

    from stateset_agents.experimental.offline_rl.decision_transformer import (
        DecisionTransformerConfig,
        DecisionTransformerTrainer,
    )

    torch.manual_seed(seed)
    np.random.seed(seed)
    config = DecisionTransformerConfig(
        n_layer=1,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        max_context_length=4,
        max_episode_length=8,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        use_conversation_embeddings=False,
        learning_rate=1e-3,
        warmup_steps=10,
        batch_size=32,
    )
    trainer = DecisionTransformerTrainer(config, device="cpu")
    asyncio.run(trainer.train(_dataset(seed), num_epochs=40))
    return trainer.model


@torch.no_grad()
def _conditioned_actions(model, states: torch.Tensor, target: float) -> torch.Tensor:
    model.eval()
    n = states.shape[0]
    return model.get_action(
        states.view(n, 1, STATE_DIM),
        torch.zeros(n, 1, ACTION_DIM),
        torch.full((n, 1, 1), target),
        torch.zeros(n, 1, dtype=torch.long),
    )


def _side(actions: torch.Tensor) -> float:
    """Mean of (distance to BAD - distance to GOOD); +1.41 is exactly GOOD.

    Observed after training: +1.40 (target 1) and -1.40 (target 0).
    """
    good = torch.as_tensor(GOOD_ACTION)
    bad = torch.as_tensor(BAD_ACTION)
    return float(((actions - bad).norm(dim=-1) - (actions - good).norm(dim=-1)).mean())


@pytest.mark.parametrize("seed", seeds())
def test_decision_transformer_follows_target_return(seed: int) -> None:
    model = _train(seed)
    states = heldout_states(seed)
    high = _side(_conditioned_actions(model, states, 1.0))
    low = _side(_conditioned_actions(model, states, 0.0))
    assert high > 1.0, (high, low)
    assert low < -1.0, (high, low)
