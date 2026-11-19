"""Independent finite-trajectory, shared-gradient and RMSprop checks for A2C."""

import asyncio
import itertools
import math
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stateset_agents.core.enhanced import advanced_rl_algorithms as rl


class Inputs(dict):
    def to(self, device):
        return Inputs({name: x.to(device) for name, x in self.items()})


class Tokenizer:
    def __len__(self):
        return 3

    def decode(self, tokens):
        return f"action{tokens[0]}"

    def __call__(self, text, **kwargs):
        return Inputs(input_ids=torch.tensor([[int(text.split()[-1])]]))


class Encoder(torch.nn.Module):
    config = SimpleNamespace(hidden_size=2)

    def forward(self, input_ids, **kwargs):
        x = input_ids.float()
        return SimpleNamespace(
            hidden_states=(torch.stack((x / 10, 1 - x / 8), dim=-1),)
        )


class Episode:
    def __init__(self, start, rewards, terminal):
        self.start, self.rewards, self.terminal = start, rewards, terminal

    async def reset(self):
        return self.start

    def observation_context(self, state):
        return state

    async def step(self, state, turn):
        offset = state - self.start
        return (
            state + 1,
            self.rewards[offset],
            self.terminal and offset + 1 == len(self.rewards),
            {},
        )


def trainer(monkeypatch, decay, *, algorithm="a2c", **kwargs):
    network = rl.ActorCriticNetwork
    monkeypatch.setattr(
        rl,
        "ActorCriticNetwork",
        lambda input_size, hidden_size, output_size: network(
            input_size, 4, output_size
        ),
    )
    torch.manual_seed(72)
    agent = SimpleNamespace(tokenizer=Tokenizer(), model=Encoder())
    if algorithm == "ppo":
        config = rl.PPOConfig(
            gamma=0.5, gae_lambda=decay, learning_rate=0.01, ppo_epochs=1, **kwargs
        )
        return rl.PPOTrainer(agent, config, "cpu")
    return rl.A2CTrainer(
        agent,
        rl.A2CConfig(gamma=0.5, gae_lambda=decay, learning_rate=0.01, **kwargs),
        "cpu",
    )


def forward(w, x):
    h1 = np.maximum(0, x @ w["shared_layers.0.weight"].T + w["shared_layers.0.bias"])
    h2 = np.maximum(0, h1 @ w["shared_layers.2.weight"].T + w["shared_layers.2.bias"])
    z = h2 @ w["actor.weight"].T + w["actor.bias"]
    centered = z - z.max(axis=-1, keepdims=True)
    logp = centered - np.log(np.exp(centered).sum(axis=-1, keepdims=True))
    v = (h2 @ w["critic.weight"].T + w["critic.bias"]).reshape(-1)
    return h1, h2, logp, v


def lambda_targets(rewards, values, bootstrap, gamma, decay):
    """Mix every n-step discounted return explicitly, including the final cutoff."""
    result = []
    for i in range(len(rewards)):
        target = 0.0
        payoff = 0.0
        for end in range(i, len(rewards)):
            offset = end - i
            payoff += gamma**offset * rewards[end]
            last = end + 1 == len(rewards)
            successor = bootstrap if last else values[end + 1]
            weight = decay**offset * (1.0 if last else 1.0 - decay)
            target += weight * (payoff + gamma ** (offset + 1) * successor)
        result.append(target)
    return np.array(result)


def oracle_update(w, squares, x, actions, returns, config):
    h1, h2, logp, v = forward(w, x)
    n = len(x)
    advantage = returns - v
    if n > 1:
        advantage = (advantage - advantage.mean()) / (advantage.std(ddof=1) + 1e-8)
    prob = np.exp(logp)
    entropy = -(prob * logp).sum(axis=1)
    loss_pi = -np.mean(advantage * logp[np.arange(n), actions])
    loss_v = np.mean((v - returns) ** 2)
    metrics = {
        "policy_loss": loss_pi,
        "value_loss": loss_v,
        "entropy": entropy.mean(),
        "total_loss": loss_pi
        + config.value_loss_coef * loss_v
        - config.entropy_coef * entropy.mean(),
    }

    actor = (
        advantage[:, None] * (prob - np.eye(prob.shape[1])[actions])
        + config.entropy_coef * prob * (logp + entropy[:, None])
    ) / n
    critic = (2 * config.value_loss_coef / n * (v - returns))[:, None]
    shared2 = (actor @ w["actor.weight"] + critic @ w["critic.weight"]) * (h2 > 0)
    shared1 = (shared2 @ w["shared_layers.2.weight"]) * (h1 > 0)
    grads = {}
    for name, upstream, features in [
        ("actor", actor, h2),
        ("critic", critic, h2),
        ("shared_layers.2", shared2, h1),
        ("shared_layers.0", shared1, x),
    ]:
        grads[name + ".weight"] = upstream.T @ features
        grads[name + ".bias"] = upstream.sum(axis=0)
    norm = math.sqrt(sum(np.square(g).sum() for g in grads.values()))
    clip = min(1.0, config.max_grad_norm / (norm + 1e-6))
    for name in w:
        grads[name] *= clip
        squares[name] = 0.99 * squares[name] + 0.01 * grads[name] ** 2
        w[name] = (
            (
                w[name]
                - config.learning_rate * grads[name] / (np.sqrt(squares[name]) + 1e-5)
            )
            .astype(np.float32)
            .astype(float)
        )
    return metrics, grads


@pytest.mark.asyncio
@pytest.mark.parametrize("decay", [0.0, 0.5, 0.95, 1.0])
async def test_complete_updates_against_n_step_mixture_and_analytic_rmsprop(
    monkeypatch, decay
):
    t = trainer(monkeypatch, decay, entropy_coef=0.13, max_grad_norm=0.4)
    w = {
        k: v.detach().numpy().astype(float)
        for k, v in t.actor_critic.named_parameters()
    }
    squares = {k: np.zeros_like(v) for k, v in w.items()}
    for batch in [
        [Episode(2, [1.0, -2.0], False), Episode(8, [3.0, -1.0], True)],
        [Episode(3, [1.5], True)],
    ]:
        expected_returns = []
        for episode in batch:
            rewards = await t.collect_experience(episode, len(episode.rewards))
            states = np.arange(episode.start, episode.start + len(rewards) + 1)
            features = (
                np.stack((states / 10, 1 - states / 8), axis=-1)
                .astype(np.float32)
                .astype(float)
            )
            _, _, lp, values = forward(w, features)
            end = len(t.actions)
            for j in range(len(rewards)):
                row = end - len(rewards) + j
                assert t.log_probs[row] == pytest.approx(
                    lp[j, t.actions[row]], abs=2e-6
                )
                assert t.values[row] == pytest.approx(values[j], abs=2e-6)
            bootstrap = 0.0 if episode.terminal else values[-1]
            expected_returns.extend(
                lambda_targets(rewards, values, bootstrap, t.config.gamma, decay)
            )
        metrics, grads = oracle_update(
            w,
            squares,
            np.asarray(t.states, dtype=float),
            np.array(t.actions),
            np.asarray(expected_returns),
            t.config,
        )
        actual = await t.train_step()
        assert actual == pytest.approx(metrics, abs=1e-5)
        for k, p in t.actor_critic.named_parameters():
            np.testing.assert_allclose(p.grad.numpy(), grads[k], rtol=3e-4, atol=1e-6)
            np.testing.assert_allclose(
                t.optimizer.state[p]["square_avg"].numpy(),
                squares[k],
                rtol=3e-4,
                atol=2e-8,
            )
            np.testing.assert_allclose(p.detach().numpy(), w[k], rtol=2e-5, atol=2e-5)
    assert not any(
        (t.states, t.actions, t.values, t.rewards, t.dones, t.log_probs, t.bootstraps)
    )
    assert await t.train_step() == {"loss": 0.0}


@pytest.mark.asyncio
@pytest.mark.parametrize("algorithm", ["a2c", "ppo"])
async def test_large_logits_and_close_rewards_keep_the_policy_gradient(
    monkeypatch, algorithm
):
    t = trainer(
        monkeypatch,
        0.0,
        algorithm=algorithm,
        value_loss_coef=0.0,
        entropy_coef=0.0,
        max_grad_norm=1e10,
    )
    with torch.no_grad():
        for p in t.actor_critic.parameters():
            p.zero_()
        t.actor_critic.actor.bias.fill_(1e20)
    collect = t.collect_experience if algorithm == "a2c" else t.collect_trajectory
    await collect(Episode(0, [1e8 + 1], True), 1)
    await collect(Episode(0, [1e8 - 1], True), 1)
    assert t.log_probs == pytest.approx([-math.log(3)] * 2)
    # Select two distinct possible actions at the uniform policy.
    t.actions[:] = [0, 2]
    metrics = await t.train_step()
    assert metrics["entropy"] == pytest.approx(math.log(3))
    a = 1 / (math.sqrt(2) + 1e-8)
    assert t.actor_critic.actor.bias.grad.tolist() == pytest.approx(
        [-a / 2, 0.0, a / 2], abs=1e-7
    )


@pytest.mark.parametrize("decay", [0.0, 0.5, 1.0])
def test_all_segmented_returns_in_the_tla_domain(decay):
    # Each successful segment's critic snapshot belongs to the collecting policy.
    for lengths, terminals, versions in itertools.product(
        itertools.product((1, 2), repeat=2),
        itertools.product((False, True), repeat=2),
        [(0, 0), (0, 1)],
    ):
        expected, advantages, bootstrap, dones, rewards, values = [], [], {}, [], [], []
        for k, (length, term, v) in enumerate(
            zip(lengths, terminals, versions, strict=True), 1
        ):
            r = [2 - 2 * j if k == 1 else 3 * j - 4 for j in range(1, length + 1)]
            predictions = np.array(
                [(4 * k + j) * (1 + v) for j in range(1, length + 1)]
            )
            tail = 0 if term else (4 * k + length + 1) * (1 + v)
            targets = lambda_targets(r, predictions, tail, 0.5, decay)
            expected.extend(targets)
            advantages.extend(targets - predictions)
            values.extend(predictions.tolist())
            rewards.extend(r)
            dones.extend([False] * (length - 1) + [term])
            if not term:
                bootstrap[len(rewards) - 1] = tail
        actual = rl._rollout_targets(rewards, values, dones, 0.5, decay, bootstrap)
        np.testing.assert_array_equal(actual, (advantages, expected))


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["success", "error", "cancel"])
async def test_update_waits_for_complete_segment_and_discards_failed_prefix(
    monkeypatch, outcome
):
    t = trainer(monkeypatch, 0.5)
    await t.collect_experience(Episode(1, [1.0], True), 1)
    entered, resume = asyncio.Event(), asyncio.Event()
    ordinary_embedding = t._get_state_embedding

    async def embedding(messages):
        if messages[-1]["content"] == "7":
            entered.set()
            await resume.wait()
            if outcome == "error":
                raise ValueError("failed cutoff value")
        return await ordinary_embedding(messages)

    monkeypatch.setattr(t, "_get_state_embedding", embedding)
    collecting = asyncio.create_task(
        t.collect_experience(Episode(5, [2.0, 3.0], False), 2)
    )
    await asyncio.wait_for(entered.wait(), 2)
    observed = []
    update = t._update

    def capture():
        observed.append((t.rewards.copy(), t.bootstraps.copy()))
        return update()

    monkeypatch.setattr(t, "_update", capture)
    updating = asyncio.create_task(t.train_step())
    await asyncio.sleep(0)
    assert not updating.done()
    assert t.rewards == [1.0] and not t.bootstraps
    if outcome == "cancel":
        collecting.cancel()
    resume.set()
    if outcome == "success":
        assert await collecting == [2.0, 3.0]
    else:
        with pytest.raises(
            asyncio.CancelledError if outcome == "cancel" else ValueError
        ):
            await collecting
    await updating
    rewards, cutoffs = observed[0]
    assert rewards == ([1.0, 2.0, 3.0] if outcome == "success" else [1.0])
    assert list(cutoffs) == ([2] if outcome == "success" else [])
    assert not t.rewards and not t.bootstraps


@pytest.mark.asyncio
@pytest.mark.parametrize(("reward", "cap"), [(1e20, 0.5), (1.6e19, math.inf)])
async def test_finite_large_actor_and_critic_gradients_reach_rmsprop(
    monkeypatch, reward, cap
):
    t = trainer(
        monkeypatch, 1.0, value_loss_coef=0.5, entropy_coef=0.0, max_grad_norm=cap
    )
    with torch.no_grad():
        for p in t.actor_critic.parameters():
            p.zero_()
    await t.collect_experience(Episode(1, [reward], True), 1)
    action = t.actions[0]
    actual = await t.train_step()
    assert actual["value_loss"] == pytest.approx(reward**2)
    direction = np.array([1 / 3, 1 / 3, 1 / 3, -1.0])
    direction[action] -= 1.0
    clipped = direction * min(reward, cap / np.linalg.norm(direction))
    grad = torch.cat([t.actor_critic.actor.bias.grad, t.actor_critic.critic.bias.grad])
    np.testing.assert_allclose(grad.numpy(), clipped, rtol=2e-6)
    expected = -0.01 * clipped / (0.1 * np.abs(clipped) + 1e-5)
    updated = torch.cat([t.actor_critic.actor.bias, t.actor_critic.critic.bias])
    np.testing.assert_allclose(updated.detach().numpy(), expected, rtol=2e-6)


@pytest.mark.asyncio
@pytest.mark.parametrize("cap", [-1.0, math.nan])
async def test_invalid_clip_cap_preserves_policy_and_rollout(monkeypatch, cap):
    t = trainer(monkeypatch, 0.5, max_grad_norm=cap)
    await t.collect_experience(Episode(2, [1.0], True), 1)
    before = [p.detach().clone() for p in t.actor_critic.parameters()]
    with pytest.raises(ValueError, match="max_grad_norm"):
        await t.train_step()
    assert not t.optimizer.state and t.rewards == [1.0]
    assert all(
        torch.equal(a, b)
        for a, b in zip(before, t.actor_critic.parameters(), strict=True)
    )
