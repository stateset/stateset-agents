"""Decision Transformer conditioning from conversation rewards to a causal action."""

import itertools
import math
import random
from copy import deepcopy
from fractions import Fraction

import pytest
import torch

from stateset_agents.data.conversation_dataset import (
    ConversationDataset,
    ConversationDatasetConfig,
    ConversationTurnData,
    TrajectoryData,
)
from stateset_agents.training.decision_transformer import (
    DecisionTransformerConfig,
    DecisionTransformerTrainer,
)


def trajectory(rewards, roles=()):
    return TrajectoryData(
        "t",
        [ConversationTurnData(r, str(i)) for i, r in enumerate(roles)],
        0.0,
        list(rewards),
    )


def exact_suffixes(rewards, gamma):
    g = Fraction(gamma)
    return [
        float(sum(Fraction(x) * g**j for j, x in enumerate(rewards[i:])))
        for i in range(len(rewards))
    ]


@pytest.mark.parametrize(
    "rewards,gamma",
    [
        ([1.0, -1e16, 1e16, 1.0], 1.0),
        ([-1e16, 2e16, 2.0], 0.5),
        ([2.0**-1074, 2.0**-1074, -(2.0**-1074)], 0.5),
        ([1.0, 2.0**-52, 2.0**-1074], 0.5),
        ([-1.0, -(2.0**-52), -(2.0**-1074)], 0.5),
        ([1.25, -3.5, 8.0, -1.5, 0.25] * 20, 0.99),
    ],
)
def test_each_discounted_suffix_rounds_independently(rewards, gamma):
    expected = exact_suffixes(rewards, gamma)
    actual = trajectory(rewards).compute_returns_to_go(gamma)
    assert list(map(float.hex, actual)) == list(map(float.hex, expected))


def test_signed_discounts_and_widely_separated_reward_scales():
    rng = random.Random(517)
    for gamma in [-1.0, -0.99, -0.5, 0.0, 0.5, 0.99, 1.0, 1.1]:
        for _ in range(8):
            rewards = [
                math.ldexp(rng.uniform(-1, 1), rng.randint(-1050, 900))
                for _ in range(17)
            ]
            actual = trajectory(rewards).compute_returns_to_go(gamma)
            assert list(map(float.hex, actual)) == list(
                map(float.hex, exact_suffixes(rewards, gamma))
            )


def test_preembedded_policy_reads_its_assistant_turn_return():
    t = DecisionTransformerTrainer.__new__(DecisionTransformerTrainer)
    t.embedder = None
    tr = trajectory([8, 4, 2, 1], ["system", "assistant", "tool", "assistant"])
    tr.metadata.update(state_embeddings=[[0], [1]], action_embeddings=[[1], [0]])
    tr.compute_returns_to_go(0.5)
    [row] = t._prepare_dataset([tr])
    assert row["returns_to_go"].tolist() == [5.25, 1.0]


def test_manual_reward_normalization_refreshes_computed_conditioning():
    tr = trajectory([1.0, 3.0], ["user", "assistant"])
    dataset = ConversationDataset(
        [tr], ConversationDatasetConfig(normalize_rewards=False, discount_factor=0.5)
    )
    dataset.compute_all_returns(0.25)
    dataset.normalize_rewards("standard")
    assert tr.returns_to_go == exact_suffixes(tr.turn_rewards, 0.25)


def test_loaded_action_labels_and_dataset_discount_reach_training():
    config = ConversationDatasetConfig(
        normalize_rewards=False, compute_returns=False, discount_factor=0.25
    )
    data = {
        "trajectories": [
            {
                "turns": [
                    {"role": "assistant", "content": "a", "reward": 1},
                    {"role": "tool", "content": "t", "reward": 2},
                    {"role": "assistant", "content": "b", "reward": 4},
                ],
                "metadata": {
                    "state_embeddings": [[0], [1]],
                    "action_embeddings": [[1], [0]],
                },
                "returns_to_go": [8, 6],
            }
        ]
    }
    dataset = ConversationDataset.from_dict(data, config)
    dt = DecisionTransformerTrainer.__new__(DecisionTransformerTrainer)
    dt.embedder = None
    assert dt._prepare_dataset(dataset)[0]["returns_to_go"].tolist() == [8.0, 6.0]
    dataset[0].returns_to_go = None
    assert dt._prepare_dataset(dataset)[0]["returns_to_go"].tolist() == [1.75, 4.0]


def test_failed_recomputation_keeps_previously_published_returns():
    a, b = trajectory([1.0, 1.0]), trajectory([2.0, 1.0])
    dataset = ConversationDataset(
        [a, b], ConversationDatasetConfig(normalize_rewards=False, discount_factor=0.5)
    )
    old = [a.returns_to_go, b.returns_to_go]
    b.turn_rewards = [1.7e308, 1.7e308]
    with pytest.raises(ValueError, match="exceeds"):
        dataset.compute_all_returns(1.0)
    assert [a.returns_to_go, b.returns_to_go] == old
    assert a.returns_discount == b.returns_discount == 0.5


class HistoryEmbedder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.factor = torch.nn.Parameter(torch.tensor(0.125))

    def forward(self, texts):
        return self.factor * self.factor.new_tensor(
            [[len(s), sum(s.encode()) % 31] for s in texts]
        )


def trainer():
    config = DecisionTransformerConfig(
        n_layer=2,
        n_head=2,
        n_embd=8,
        dropout=0,
        state_dim=2,
        action_dim=2,
        max_context_length=2,
        max_episode_length=16,
        use_conversation_embeddings=False,
        warmup_steps=0,
    )
    t = DecisionTransformerTrainer(config, device="cpu")
    t.embedder = HistoryEmbedder()
    return t


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_generated_conditioning_matches_teacher_forced_history_and_checkpoint(
    tmp_path, dtype
):
    torch.manual_seed(5)
    t = trainer()
    t.model.to(dtype)
    roles = [
        "user",
        "assistant",
        "tool",
        "assistant",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    tr = trajectory([0, 1, 2, 3, 0, -1, 0, 2], roles)
    tr.compute_returns_to_go(0.5)
    [row] = t._prepare_dataset([tr])
    states, actions, returns = [
        torch.from_numpy(row[key]).to(dtype)[None]
        for key in ("states", "actions", "returns_to_go")
    ]
    returns = returns.unsqueeze(-1)
    steps = torch.arange(4)[None]
    with torch.no_grad():
        expected = t.model(
            states[:, -2:], actions[:, -2:], returns[:, -2:], steps[:, -2:]
        )[0][0, -1]
        torch.testing.assert_close(
            t.model.get_action(states, actions, returns, steps)[0], expected
        )
    history = [{"role": v.role, "content": v.content} for v in tr.turns[:-1]]
    args = (
        history,
        returns[0, -1, 0].item(),
        list(actions[0, :-1]),
        returns[0, :-1, 0].tolist(),
    )
    torch.testing.assert_close(t.generate_response(*args), expected)
    path = tmp_path / "dt.pt"
    t.save(str(path))
    other = trainer()
    other.model.to(dtype)
    with torch.no_grad():
        other.embedder.factor.fill_(2)
    other.load(str(path))
    torch.testing.assert_close(other.generate_response(*args), expected)
    with pytest.raises(ValueError, match="align"):
        t.generate_response(history, 2.0, args[2], [1.0])


def test_optional_past_returns_keep_actions_and_zero_only_missing_budgets():
    t = trainer()
    recorded = []
    t.model.get_action = lambda *args: recorded.append(args) or args[1][:, -1]
    history = [
        {"role": "user", "content": "u"},
        {"role": "assistant", "content": "a"},
        {"role": "user", "content": "v"},
    ]
    t.generate_response(history, 3.0, [torch.tensor([1.0, 2.0])])
    states, actions, returns, steps = recorded[0]
    torch.testing.assert_close(actions[0, 0], torch.tensor([1.0, 2.0]))
    torch.testing.assert_close(states[0, 0], t.embedder(["user: u\n"])[0])
    assert returns.flatten().tolist() == [0.0, 3.0] and steps.flatten().tolist() == [
        0,
        1,
    ]


def test_all_turn_tables_select_the_independent_discounted_action_suffix():
    dt = DecisionTransformerTrainer.__new__(DecisionTransformerTrainer)
    dt.embedder = None
    for roles in itertools.product(["tool", "assistant"], repeat=4):
        indices = [i for i, r in enumerate(roles) if r == "assistant"]
        if not indices:
            continue
        for discount in [-0.5, 0.0, 0.5, 1.0]:
            for rewards in itertools.product([-1.0, 0.0, 1.0], repeat=4):
                tr = trajectory(rewards, roles)
                tr.metadata.update(
                    state_embeddings=[[i] for i in range(4)],
                    action_embeddings=[[-i] for i in range(4)],
                )
                tr.compute_returns_to_go(discount)
                [row] = dt._prepare_dataset([tr])
                oracle = exact_suffixes(rewards, discount)
                assert row["states"][:, 0].tolist() == indices
                assert row["returns_to_go"].tolist() == [oracle[i] for i in indices]


def test_masked_states_and_timestep_sentinels_cannot_affect_an_action():
    torch.manual_seed(8)
    model = trainer().model.double().eval()
    states = torch.randn(1, 2, 2, dtype=torch.float64, requires_grad=True)
    actions = torch.randn_like(states)
    returns = torch.ones_like(states[:, :, :1])
    steps = torch.tensor([[1, 2]])
    expected = model(states[:, 1:], actions[:, 1:], returns[:, 1:], steps[:, 1:])[0]
    states.data[:, 0] = math.nan
    actions[:, 0] = math.nan
    returns[:, 0] = math.nan
    steps[:, 0] = -100
    actual = model(states, actions, returns, steps, torch.tensor([[0, 1]]))[0][:, 1:]
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert (states.grad[:, 0] == 0).all()
    assert all(p.grad is None or p.grad.isfinite().all() for p in model.parameters())


@pytest.mark.parametrize("method", ["standard", "percentile", "minmax"])
def test_finite_training_labels_under_large_reward_scale(method):
    tr = trajectory([-1e308, 1e308], ["user", "assistant"])
    ConversationDataset(
        [tr],
        ConversationDatasetConfig(
            reward_normalization_method=method, discount_factor=0.5
        ),
    )
    expected = [0.0, 1.0] if method == "minmax" else [-1.0, 1.0]
    assert tr.turn_rewards == pytest.approx(expected, abs=1e-15)
    assert tr.returns_to_go == pytest.approx(exact_suffixes(expected, 0.5), abs=1e-15)


def test_refined_grid_decides_which_side_of_a_tie_a_distant_reward_reaches():
    # The exact first return is 1 + 2^-53 + 2^-1201. Its first rounded
    # enclosing endpoints disagree at the binary64 midpoint above one.
    rs = [1.0, 2.0**-52] + [0.0] * 1199 + [1.0]
    assert trajectory(rs).compute_returns_to_go(0.5)[0] == math.nextafter(1.0, math.inf)
    assert trajectory([-x for x in rs]).compute_returns_to_go(0.5)[0] == math.nextafter(
        -1.0, -math.inf
    )


@pytest.mark.parametrize(
    "field,values",
    [
        ("actions", [[1.0, 2.0]]),
        ("returns_to_go", [3.0]),
        ("timesteps", [0.5, 1.5]),
        ("states", [[1.0, 2.0], [math.nan, 3.0]]),
    ],
)
def test_invalid_retained_rows_fail_before_weights_or_scheduler_change(field, values):
    t = trainer()
    before = deepcopy(t.model.state_dict())
    row = {
        "states": [[1.0, 2.0], [3.0, 4.0]],
        "actions": [[2.0, 1.0], [4.0, 3.0]],
        "timesteps": [0, 1],
        "returns_to_go": [3.0, 1.0],
    }
    row[field] = values
    with pytest.raises(ValueError):
        t.train_step([row])
    assert t.training_step == 0 and not t.optimizer.state
    torch.testing.assert_close(t.model.state_dict(), before, atol=0, rtol=0)


def test_two_joint_updates_match_independent_unpadded_predictions():
    torch.manual_seed(71)
    t = trainer()
    t.model.double()
    ref = deepcopy(t.model)
    optimizer = torch.optim.AdamW(
        ref.parameters(), lr=t.config.learning_rate, weight_decay=t.config.weight_decay
    )
    rows = [
        {
            "states": [[0.2, 1.1]],
            "actions": [[-0.25, 0.75]],
            "returns_to_go": [1.0],
            "timesteps": [0],
        },
        {
            "states": [[1.2, 0.2], [0.6, 1.4]],
            "actions": [[0.25, 0.5], [0.125, -0.75]],
            "returns_to_go": [1.25, -0.5],
            "timesteps": [2, 3],
        },
    ]
    for batch in [rows, rows[::-1]]:
        action_errors, state_errors = [], []
        for row in batch:
            s, a, r = [
                torch.tensor(row[key], dtype=torch.float64)[None]
                for key in ("states", "actions", "returns_to_go")
            ]
            pa, ps, _ = ref(s, a, r.unsqueeze(-1), torch.tensor([row["timesteps"]]))
            action_errors.extend((pa - a).square().sum(-1).flatten())
            state_errors.extend((ps[:, :-1] - s[:, 1:]).square().sum(-1).flatten())
        expected = (
            torch.stack(action_errors).mean() + 0.1 * torch.stack(state_errors).mean()
        )
        optimizer.zero_grad()
        expected.backward()
        torch.nn.utils.clip_grad_norm_(ref.parameters(), t.config.max_grad_norm)
        optimizer.step()
        result = t.train_step(batch)
        assert result["loss"] == pytest.approx(expected.item(), abs=2e-15)
        for p, q in zip(t.model.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(p, q, atol=2e-12, rtol=2e-10)
            if q.grad is None:
                assert p.grad is None
            else:
                torch.testing.assert_close(p.grad, q.grad, atol=1e-14, rtol=1e-11)
                for key in ["exp_avg", "exp_avg_sq", "step"]:
                    torch.testing.assert_close(
                        t.optimizer.state[p][key],
                        optimizer.state[q][key],
                        atol=1e-15,
                        rtol=1e-10,
                    )


def test_large_finite_action_error_preserves_the_clipped_update():
    torch.manual_seed(63)
    t = trainer()
    with torch.no_grad():
        t.model.embed_action.weight.zero_()
    row = {
        "states": [[0.0, 0.0]],
        "actions": [[1e20, -1e20]],
        "returns_to_go": [1.0],
        "timesteps": [0],
    }
    bias = t.model.predict_action[0].bias
    before = bias.detach().clone()
    result = t.train_step([row])
    assert result["action_loss"] == pytest.approx(
        2 * torch.tensor(1e20).item() ** 2, rel=1e-13
    )
    assert result["state_loss"] == 0
    assert bias[0] > before[0] and bias[1] < before[1]
    for p in t.model.parameters():
        assert p.isfinite().all() and (p.grad is None or p.grad.isfinite().all())
