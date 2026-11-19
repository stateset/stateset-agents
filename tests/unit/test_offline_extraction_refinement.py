"""Check logged-action IQL updates against independent conditional probabilities."""

import itertools
import math
from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

from stateset_agents.training.offline_grpo_trainer import (
    OfflineGRPOConfig,
    OfflineGRPOTrainer,
)
from stateset_agents.training.offline_rl_algorithms import ImplicitQLearning


class Table(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.scores = torch.nn.Parameter(torch.tensor(values, dtype=torch.float64))

    def forward(self, input_ids, attention_mask):
        return SimpleNamespace(logits=self.scores[input_ids])


def trainer(model, *, use_adamw=False, **config):
    instance = OfflineGRPOTrainer(
        OfflineGRPOConfig(
            state_dim=2,
            action_dim=1,
            value_hidden_size=2,
            value_num_layers=1,
            policy_temperature=1.0,
            learning_rate=0.01,
            **config,
        ),
        model=model,
        device="cpu",
    )
    instance.value_pretrained = True
    instance._offline_learner = SimpleNamespace(
        q1=lambda s, a: s[:, :1], q2=lambda s, a: s[:, :1], value_net=lambda s: s[:, 1:]
    )
    instance._setup_policy_optimizer()
    if not use_adamw:
        instance.policy_optimizer = torch.optim.SGD(model.parameters(), lr=0.125)
    return instance


def batch(ids, labels, values):
    return {
        "input_ids": ids,
        "labels": labels,
        "states": values,
        "actions": [[0]] * len(ids),
    }


def test_scalar_and_batched_iql_agree_on_extreme_opposite_values():
    instance = object.__new__(ImplicitQLearning)
    instance.q1 = instance.q2 = lambda s, a: s[:, :1]
    instance.value_net = lambda s: s[:, 1:]
    instance.config = SimpleNamespace(temperature=1.0, clip_score=7.0)
    state = torch.tensor([3e38, -3e38])
    action = torch.tensor([0.0])
    assert instance.get_advantage(state, action) == state[0].item() - state[1].item()
    assert instance.get_policy_weight(state, action) == 7


async def test_stationary_prefix_fit_counts_half_of_each_long_action():
    model = Table([[0, 0, -math.inf]] * 3)
    t = trainer(model, batch_size=3, gradient_accumulation_steps=1, max_grad_norm=10)
    data = batch(
        [[2, 0, 2], [2, 1, 0], [2, 1, 1]],
        [[-100, 0, -100], [-100, 1, 0], [-100, 1, 1]],
        [[0, 0]] * 3,
    )
    metrics = await t.train_step(data)
    assert metrics["policy_loss"] == pytest.approx(math.log(2))
    assert metrics["grad_norm"] < 1e-15
    # Fitting whole-action frequencies would instead put 2/3 on B.
    with torch.no_grad():
        model.scores[2, 1] = math.log(2)
    moved = await t.train_step(data)
    assert moved["policy_loss"] > metrics["policy_loss"]
    assert moved["grad_norm"] > 0.1


async def test_representable_mean_with_unrepresentable_single_token_gap():
    model = Table([[1e308, -1e308, 0], [0, 0, 0], [0, 0, 0]])
    t = trainer(model, batch_size=2, gradient_accumulation_steps=1)
    result = await t.train_step(
        batch([[0, 1, 0]] * 2, [[-100, 1, 0]] * 2, [[0, 0]] * 2)
    )
    assert result["nll"] == result["policy_loss"] == 1e308
    assert math.isfinite(result["grad_norm"]) and result["grad_norm"] > 0


@pytest.mark.parametrize(
    "case", ["masked", "shifted", "small_weight", "dominant", "opposite_values"]
)
async def test_offline_selected_token_loss_and_sgd(case):
    logits = [[0.0] * 3 for _ in range(3)]
    values = [0.0, 0.0]
    gap, weight = 0.0, 1.0
    if case == "masked":
        logits[0] = [math.nan] * 3
    elif case == "shifted":
        gap = 2.0
        logits[1] = [1e9, 1e9, 1e9 + gap]
    elif case == "small_weight":
        values[0], weight = -120.0, math.exp(-120)
    elif case == "dominant":
        gap = 60.0
        logits[1] = [-gap, -gap, 0.0]
    else:
        values, weight = [3e38, -3e38], 2.0
    model = Table(logits)
    t = trainer(
        model, gradient_accumulation_steps=1, policy_max_weight=2, max_grad_norm=1000
    )
    start = model.scores.detach().clone()
    metrics = await t.train_step(batch([[0, 1, 2]], [[-100, -100, 2]], [values]))
    assert metrics["policy_weight_mean"] == pytest.approx(weight, rel=2e-14, abs=0)
    assert metrics["policy_loss"] == pytest.approx(
        weight * math.log1p(2 * math.exp(-gap)), rel=2e-14, abs=0
    )
    g = weight * math.exp(-gap) / (1 + 2 * math.exp(-gap))
    grad = torch.zeros_like(start)
    grad[1] = grad.new_tensor([g, g, -2 * g])
    expected = start - 0.125 * grad
    torch.testing.assert_close(
        model.scores, expected, atol=0, rtol=2e-14, equal_nan=True
    )
    assert metrics["grad_norm"] == pytest.approx(math.sqrt(6) * g, rel=2e-14, abs=0)


@pytest.mark.parametrize("sizes", [[3], [2, 1], [1, 2]])
async def test_length_normalized_conditional_counts_and_microbatch_gradients(sizes):
    # Three logged actions: short A, long B-A and long B-B, all from the same prompt.
    # The unnormalized action weights are 1, 2 and 3. At the root the observed
    # masses are therefore 1 and 5/2; at B they are 1 and 3/2.
    rows = [[2, 0, 2], [2, 1, 0], [2, 1, 1]]
    labels = [[-100, 0, -100], [-100, 1, 0], [-100, 1, 1]]
    model = Table([[0, 0, -math.inf], [0, 0, -math.inf], [0, 0, -math.inf]])
    t = trainer(
        model, batch_size=3, gradient_accumulation_steps=len(sizes), max_grad_norm=10
    )
    # The values reported by the critic are float32. Use their represented
    # values when independently constructing the exact training target.
    states = torch.tensor([[0.0, 0.0], [math.log(2), 0.0], [math.log(3), 0.0]])
    w1, w2, w3 = map(math.exp, states[:, 0].tolist())
    offset = 0
    for size in sizes:
        stop = offset + size
        await t.train_step(
            batch(rows[offset:stop], labels[offset:stop], states[offset:stop])
        )
        offset = stop
    root = ((w2 + w3) / 2 - w1) / 6
    later = (w3 - w2) / 12
    expected = (
        torch.tensor([[0.0, 0.0], [-later, later], [-root, root]], dtype=torch.float64)
        * 0.125
    )
    torch.testing.assert_close(model.scores[:, :2], expected, atol=1e-16, rtol=2e-14)
    assert t.optimizer_updates == 1


@pytest.mark.parametrize("head", [1, 2])
async def test_conditional_gradients_on_all_bounded_path_triples(head):
    # Token 2 is the prompt. In this fixture token 0 is A and token 1 is B.
    paths = [(0,), (1,), (1, 0), (1, 1)]
    model = Table(
        [
            [0, 0, -math.inf],
            [math.log(3 - head), math.log(head), -math.inf],
            [math.log(head), math.log(3 - head), -math.inf],
        ]
    )
    t = trainer(
        model,
        batch_size=2,
        gradient_accumulation_steps=2,
        max_grad_norm=10,
        policy_max_weight=1,
    )
    optimizer = t.policy_optimizer
    optimizer.param_groups[0]["lr"] = 0.0
    recorded = []
    optimizer.register_step_pre_hook(
        lambda opt, args, kwargs: recorded.append(model.scores.grad.clone())
    )
    log_two = torch.tensor(math.log(2)).item()
    for k, triple in enumerate(itertools.product(paths, repeat=3)):
        values = [((k + i) % 3 - 1) * log_two for i in range(3)]
        ids = [[2, *p, *([2] if len(p) == 1 else [])] for p in triple]
        labels = [[-100, *p, *([-100] if len(p) == 1 else [])] for p in triple]
        states = [[v, 0] for v in values]
        boundary = 1 + k % 2
        for sl in [slice(None, boundary), slice(boundary, None)]:
            await t.train_step(batch(ids[sl], labels[sl], states[sl]))
        reference = torch.zeros_like(model.scores)
        for path, value in zip(triple, values, strict=True):
            prefix = 2
            for token in path:
                pa = (head if prefix == 2 else 3 - head) / 3
                slope = min(1, math.exp(value)) * (pa - (token == 0)) / (3 * len(path))
                reference[prefix, 0] += slope
                reference[prefix, 1] -= slope
                prefix = token
        torch.testing.assert_close(recorded.pop(), reference, atol=2e-16, rtol=1e-14)


async def test_two_causal_transformer_updates_match_teacher_forced_adamw():
    from transformers import GPT2Config, GPT2LMHeadModel

    torch.manual_seed(12)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=8,
            n_positions=8,
            n_embd=8,
            n_layer=1,
            n_head=2,
            resid_pdrop=0,
            attn_pdrop=0,
            embd_pdrop=0,
        )
    ).double()
    t = trainer(
        model,
        use_adamw=True,
        batch_size=2,
        gradient_accumulation_steps=2,
        policy_max_weight=1.5,
        max_grad_norm=100,
    )
    ref = deepcopy(model)
    opt = torch.optim.AdamW(
        ref.parameters(),
        lr=t.config.learning_rate,
        betas=(t.config.adam_beta1, t.config.adam_beta2),
        weight_decay=t.config.weight_decay,
    )
    rows = [[2, 3, 1, 0], [2, 1, 4, 3], [2, 3, 5, 6]]
    labels = [[-100, -100, 1, -100], [-100, 1, 4, 3], [-100, -100, 5, 6]]
    states = [[1.5, 0], [-0.5, 0], [0, 0.25]]
    for ordering in [[0, 1, 2], [2, 0, 1]]:
        opt.zero_grad()
        losses = []
        for i in ordering:
            prediction = ref(torch.tensor([rows[i]])).logits[0].log_softmax(-1)
            active = [pos for pos in range(1, len(rows[i])) if labels[i][pos] != -100]
            weight = min(1.5, math.exp(states[i][0] - states[i][1]))
            losses.append(
                -weight
                * sum(prediction[pos - 1, rows[i][pos]] for pos in active)
                / len(active)
            )
        sum(losses).div(3).backward()
        opt.step()
        for indices in [ordering[:2], ordering[2:]]:
            values = await t.train_step(
                batch(
                    [rows[i] for i in indices],
                    [labels[i] for i in indices],
                    [states[i] for i in indices],
                )
            )
            expected = sum(losses[ordering.index(i)].item() for i in indices) / len(
                indices
            )
            assert values["policy_loss"] == pytest.approx(expected, abs=2e-13)
        for p, q in zip(model.parameters(), ref.parameters(), strict=True):
            torch.testing.assert_close(p, q, rtol=1e-9, atol=3e-11)
            for k in ["exp_avg", "exp_avg_sq"]:
                torch.testing.assert_close(
                    t.policy_optimizer.state[p][k],
                    opt.state[q][k],
                    rtol=2e-9,
                    atol=5e-14,
                )
