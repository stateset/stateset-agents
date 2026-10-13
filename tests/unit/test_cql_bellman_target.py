"""CQL's Bellman target bootstraps from next-state actions, never the current
dataset action (``Q(s', a_t)`` is the value of repeating a_t in s')."""

import inspect

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.experimental.offline_rl.algorithms import (  # noqa: E402
    ConservativeQLearning,
    CQLConfig,
)


def _cql(**kw):
    torch.manual_seed(0)
    return ConservativeQLearning(state_dim=3, action_dim=2, config=CQLConfig(**kw))


def test_target_does_not_depend_on_current_action():
    cql = _cql()
    rewards = torch.zeros(4, 1)
    next_states = torch.randn(4, 3)
    dones = torch.zeros(4, 1)
    torch.manual_seed(1)
    a = cql._bellman_target(rewards, next_states, dones)
    torch.manual_seed(1)
    b = cql._bellman_target(rewards, next_states, dones)
    torch.testing.assert_close(a, b)
    params = inspect.signature(cql._bellman_target).parameters
    assert "actions" not in params  # only next-state actions can enter


def test_target_uses_dataset_next_actions_when_given():
    cql = _cql(discount_factor=0.5)
    next_states = torch.randn(2, 3)
    next_actions = torch.rand(2, 2) * 2 - 1
    target = cql._bellman_target(
        torch.ones(2, 1), next_states, torch.tensor([[0.0], [1.0]]), next_actions
    )
    with torch.no_grad():
        q = torch.min(
            cql.q1_target(next_states, next_actions),
            cql.q2_target(next_states, next_actions),
        )
    torch.testing.assert_close(target[0], 1.0 + 0.5 * q[0])
    torch.testing.assert_close(target[1], torch.tensor([1.0]))


def test_cql_penalty_scales_logsumexp_by_temperature():
    """CQL(H): tau * logsumexp(Q / tau); without the tau factor the penalty
    explodes as tau shrinks."""
    cql = _cql(temperature=0.1, num_random_actions=4)
    states = torch.randn(3, 3)
    actions = torch.zeros(3, 2)
    q1 = cql.q1(states, actions)
    q2 = cql.q2(states, actions)
    torch.manual_seed(2)
    l1, _ = cql._compute_cql_loss(states, actions, q1, q2)
    torch.manual_seed(2)
    rand = torch.empty(3 * 4, 2).uniform_(-1, 1)
    rep = states.unsqueeze(1).repeat(1, 4, 1).view(12, -1)
    qr = cql.q1(rep, rand).view(3, 4, 1)
    expected = (0.1 * torch.logsumexp(qr / 0.1, dim=1) - q1).mean()
    torch.testing.assert_close(l1, expected)
