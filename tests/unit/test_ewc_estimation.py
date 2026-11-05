"""Empirical Fisher estimates must use valid examples and preserve live state."""

import asyncio
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from stateset_agents.training import continual_learning as module  # noqa: E402


class BadGradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value):
        return value.clone()

    @staticmethod
    def backward(ctx, gradient):
        return gradient * float("nan")


class ScoreModel(torch.nn.Module):
    def __init__(self, dtype=torch.float32):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0, dtype=dtype))
        self.other = torch.nn.Parameter(torch.tensor(3.0, dtype=dtype))
        self.dropout = torch.nn.Dropout(0.5)
        self.forward_calls = 0

    def forward(self, x, kind, labels):
        self.forward_calls += 1
        if kind == "sample_error":
            raise ValueError("invalid sample")
        if kind == "oom":
            raise torch.OutOfMemoryError("synthetic OOM")
        if kind == "interrupt":
            raise KeyboardInterrupt()
        if kind == "cancel":
            raise asyncio.CancelledError()
        if kind == "nanloss":
            loss = self.weight * float("nan")
        elif kind == "gradnan":
            loss = BadGradient.apply(self.weight) + self.other * x
        elif kind == "disconnected":
            loss = torch.ones((), requires_grad=True)
        elif kind in {"linear", "linearoverflow"}:
            loss = self.weight * x
        else:
            loss = 0.5 * (self.weight * x).square() + self.other * 0
        return SimpleNamespace(loss=loss)


@pytest.fixture
def setup(monkeypatch):
    def create(dtype=torch.float32):
        model = ScoreModel(dtype)
        model.weight.grad = torch.full_like(model.weight, 11)
        model.other.grad = torch.full_like(model.other, 13)
        manager = module.ContinualLearningManager(
            module.ContinualLearningConfig(
                strategy="ewc", ewc_lambda=0.2, ewc_num_samples=10
            )
        )

        def prepare(sample, agent, config):
            kind, x = sample
            if kind == "prepare_error":
                raise ValueError("invalid inputs")
            labels = torch.tensor([[-100, 1]])
            if kind == "all_masked":
                labels.fill_(-100)
            elif kind == "only_first_label":
                labels = torch.tensor([[1, -100]])
            return {"x": torch.tensor(x, dtype=dtype), "kind": kind}, labels

        monkeypatch.setattr(module, "_prepare_inputs_and_labels", prepare)
        return manager, SimpleNamespace(model=model)

    return create


INVALID = [
    "prepare_error",
    "sample_error",
    "nanloss",
    "gradnan",
    "disconnected",
    "all_masked",
    "only_first_label",
    "linearoverflow",
]


@pytest.mark.parametrize("kind", INVALID)
def test_failed_samples_do_not_dilute_or_contaminate_fisher(setup, kind):
    manager, agent = setup()
    x = 1e30 if kind == "linearoverflow" else 3.0
    fisher, _ = manager._estimate_fisher(agent, [("valid", 2.0), (kind, x)])
    # d[0.5*(w*x)^2]/dw = w*x^2 = 8, so one accepted sample has Fisher 64.
    torch.testing.assert_close(fisher["weight"], torch.tensor(64.0))
    torch.testing.assert_close(fisher["other"], torch.tensor(0.0))
    assert agent.model.weight.grad.item() == 11
    assert agent.model.other.grad.item() == 13


@pytest.mark.parametrize("kind", INVALID)
def test_no_valid_samples_preserve_previous_task_memory(setup, kind):
    manager, agent = setup()
    manager._ewc_fisher = {"weight": torch.tensor(0.7)}
    manager._ewc_params = {"weight": torch.tensor(1.3)}
    x = 1e30 if kind == "linearoverflow" else 3.0
    group = SimpleNamespace(trajectories=[(kind, x)])
    manager._update_ewc(agent, recent_groups=[group])
    torch.testing.assert_close(manager._ewc_fisher["weight"], torch.tensor(0.7))
    torch.testing.assert_close(manager._ewc_params["weight"], torch.tensor(1.3))


@pytest.mark.parametrize("root,child", [(True, False), (False, True)])
def test_estimation_restores_mixed_module_modes(setup, root, child):
    manager, agent = setup()
    agent.model.train(root)
    agent.model.dropout.train(child)
    manager._estimate_fisher(agent, [("valid", 2.0)])
    assert agent.model.training is root
    assert agent.model.dropout.training is child


def test_frozen_model_retains_training_mode(setup):
    manager, agent = setup()
    agent.model.requires_grad_(False)
    assert manager._estimate_fisher(agent, [("valid", 2.0)]) == ({}, {})
    assert agent.model.training is True


@pytest.mark.parametrize(
    "kind,error",
    [
        ("oom", torch.OutOfMemoryError),
        ("interrupt", KeyboardInterrupt),
        ("cancel", asyncio.CancelledError),
    ],
)
def test_resource_failures_and_interrupts_propagate_and_restore_mode(
    setup, kind, error
):
    manager, agent = setup()
    with pytest.raises(error):
        manager._estimate_fisher(agent, [(kind, 2.0)])
    assert agent.model.training is True


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_gradients_are_squared_in_float32(setup, dtype):
    manager, agent = setup(dtype)
    fisher, means = manager._estimate_fisher(agent, [("linear", 1000.0)])
    assert fisher["weight"].dtype == means["weight"].dtype == torch.float32
    torch.testing.assert_close(fisher["weight"], torch.tensor(1_000_000.0))


def test_finite_mean_does_not_overflow_when_sum_would(setup):
    manager, agent = setup()
    fisher, _ = manager._estimate_fisher(agent, [("linear", 1e19)] * 4)
    expected = torch.tensor(1e19).double().square().float()
    assert torch.isfinite(fisher["weight"])
    torch.testing.assert_close(fisher["weight"], expected)


def test_all_valid_samples_match_analytic_reference(setup):
    manager, agent = setup()
    fisher, _ = manager._estimate_fisher(agent, [("valid", 2.0), ("valid", 3.0)])
    torch.testing.assert_close(fisher["weight"], torch.tensor((64.0 + 324.0) / 2))


def test_attempt_cap_does_not_scan_past_failed_examples(setup):
    manager, agent = setup()
    manager.config.ewc_num_samples = 1
    assert manager._estimate_fisher(agent, [("sample_error", 1), ("valid", 2)]) == (
        {},
        {},
    )
    assert agent.model.forward_calls == 1
    assert manager.last_fisher_estimate["attempted"] == 1
    assert manager.last_fisher_estimate["accepted"] == 0
    assert manager.last_fisher_estimate["sample_errors"] == 1


@pytest.mark.parametrize("count", [0, 1])
def test_zero_budget_or_empty_data_preserves_modes(setup, count):
    manager, agent = setup()
    manager.config.ewc_num_samples = count
    agent.model.dropout.eval()
    assert manager._estimate_fisher(agent, []) == ({}, {})
    assert agent.model.training and not agent.model.dropout.training


def test_connected_zero_gradients_are_valid_observations(setup):
    manager, agent = setup()
    fisher, _ = manager._estimate_fisher(agent, [("valid", 0)])
    assert manager.last_fisher_estimate["accepted"] == 1
    assert fisher["weight"].item() == 0


@pytest.mark.parametrize("kind", ["all_masked", "only_first_label"])
def test_unscored_trajectories_do_not_run_model_forward(setup, kind):
    manager, agent = setup()
    assert manager._estimate_fisher(agent, [(kind, 2)]) == ({}, {})
    assert agent.model.forward_calls == 0
    assert manager.last_fisher_estimate["no_supervised_tokens"] == 1


def test_valid_float64_statistics_keep_precision(setup):
    manager, agent = setup(torch.float64)
    fisher, means = manager._estimate_fisher(agent, [("valid", 2)])
    assert fisher["weight"].dtype == means["weight"].dtype == torch.float64
    torch.testing.assert_close(
        fisher["weight"], torch.tensor(64.0, dtype=torch.float64)
    )


def test_nonfinite_policy_parameters_fail_before_forward(setup):
    manager, agent = setup()
    with torch.no_grad():
        agent.model.other.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite policy parameters"):
        manager._estimate_fisher(agent, [("linear", 2)])
    assert agent.model.training is True
    assert agent.model.forward_calls == 0


def test_consolidation_overflow_does_not_partially_change_task_memory(
    setup, monkeypatch
):
    manager, agent = setup()
    manager._ewc_fisher = {"weight": torch.tensor(2e38), "other": torch.tensor(0.5)}
    manager._ewc_params = {"weight": torch.tensor(1.3), "other": torch.tensor(2.3)}
    before = {name: value.clone() for name, value in manager._ewc_fisher.items()}
    means = {name: value.clone() for name, value in manager._ewc_params.items()}
    monkeypatch.setattr(
        manager,
        "_estimate_fisher",
        lambda *args: (
            {"other": torch.tensor(1.0), "weight": torch.tensor(2e38)},
            {"other": torch.tensor(3.0), "weight": torch.tensor(2.0)},
        ),
    )
    with pytest.raises(ValueError, match="nonfinite task memory"):
        manager._update_ewc(
            agent, recent_groups=[SimpleNamespace(trajectories=[object()])]
        )
    for name in before:
        torch.testing.assert_close(
            manager._ewc_fisher[name], before[name], rtol=0, atol=0
        )
        torch.testing.assert_close(
            manager._ewc_params[name], means[name], rtol=0, atol=0
        )


def test_transformer_fisher_matches_independent_causal_loss_gradient():
    from transformers import GPT2Config, GPT2LMHeadModel

    from stateset_agents.core.trajectory import ConversationTurn, MultiTurnTrajectory
    from stateset_agents.training.sampled_workflow import workflow_tokenizer

    tokenizer = workflow_tokenizer()
    tokenizer.chat_template = (
        "{% for message in messages %}{% if message['role'] == 'assistant' %}"
        "{% generation %}{{ message['content'] }} {{ eos_token }}{% endgeneration %}"
        "{% else %}{{ message['content'] }} {{ eos_token }}{% endif %} {% endfor %}"
    )
    torch.manual_seed(7)
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=32,
            n_embd=8,
            n_layer=1,
            n_head=1,
        )
    )
    samples = [
        [("user", "request"), ("assistant", "lookup")],
        [
            ("user", "eligible"),
            ("assistant", "refund"),
            ("tool", "resolved"),
            ("assistant", "finish"),
        ],
        [("user", "request")],
    ]
    trajectories = [
        MultiTurnTrajectory(
            turns=[
                ConversationTurn(role=role, content=content) for role, content in turns
            ]
        )
        for turns in samples
    ]
    words = [
        ["request", "<eos>", "lookup", "<eos>"],
        [
            "eligible",
            "<eos>",
            "refund",
            "<eos>",
            "resolved",
            "<eos>",
            "finish",
            "<eos>",
        ],
    ]
    masks = [
        [False, False, True, True],
        [False, False, True, True, False, False, True, True],
    ]
    parameters = dict(model.named_parameters())
    expected = {name: torch.zeros_like(p) for name, p in parameters.items()}
    model.eval()
    for row, mask in zip(words, masks, strict=True):
        ids = torch.tensor([tokenizer.convert_tokens_to_ids(row)])
        labels = ids.masked_fill(~torch.tensor([mask]), -100)
        logits = model(ids).logits[:, :-1].float()
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, len(tokenizer)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
        )
        grads = torch.autograd.grad(loss, list(parameters.values()), allow_unused=True)
        for (name, _), grad in zip(parameters.items(), grads, strict=True):
            if grad is not None:
                expected[name] += grad.square() / 2
    model.train()
    model.transformer.h[0].attn.attn_dropout.eval()
    modes = [child.training for child in model.modules()]
    manager = module.ContinualLearningManager(
        module.ContinualLearningConfig(
            strategy="ewc", ewc_lambda=0.2, ewc_num_samples=3
        )
    )
    actual, _ = manager._estimate_fisher(
        SimpleNamespace(model=model, tokenizer=tokenizer), trajectories
    )
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], rtol=1e-5, atol=1e-7)
    assert [child.training for child in model.modules()] == modes
    assert manager.last_fisher_estimate["accepted"] == 2
    assert manager.last_fisher_estimate["no_supervised_tokens"] == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("api", ["async", "runtime"])
async def test_promoted_fisher_survives_checkpoint_recovery(
    tmp_path, setup, dtype, api
):
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
    from tests.unit.test_online_accumulation_weighting import _trainer

    manager, agent = setup(dtype)
    manager._update_ewc(
        agent, recent_groups=[SimpleNamespace(trajectories=[("linear", 1000)])]
    )
    agent.model.zero_grad(set_to_none=True)
    trainer = _trainer(tmp_path, MultiTurnGRPOTrainer, "grpo", 0.0, 1)
    trainer.agent = agent
    trainer.optimizer = torch.optim.SGD(agent.model.parameters(), lr=0.1)
    trainer.continual_manager = manager
    agent.model.save_pretrained = lambda path: torch.save(
        agent.model.state_dict(), path / "pytorch_model.bin"
    )
    path = tmp_path / "saved"
    if api == "runtime":
        trainer.save_runtime_checkpoint(str(path))
    else:
        await trainer.save_checkpoint(checkpoint_name="saved")
    restored = _trainer(tmp_path / "restored", MultiTurnGRPOTrainer, "grpo", 0.0, 1)
    restored.agent.model = ScoreModel(dtype)
    restored.optimizer = torch.optim.SGD(restored.agent.model.parameters(), lr=0.1)
    restored.continual_manager = module.ContinualLearningManager(manager.config)
    if api == "runtime":
        assert restored.load_runtime_checkpoint(str(path))
    else:
        assert restored.load_checkpoint(path)
    assert restored.continual_manager._ewc_fisher["weight"].dtype == torch.float32
    with torch.no_grad():
        restored.agent.model.weight.add_(1)
    penalty = restored.continual_manager.compute_ewc_penalty(restored.agent)
    assert penalty.dtype == torch.float32
    torch.testing.assert_close(penalty, torch.tensor(100_000.0))


def test_estimation_enables_autograd_inside_no_grad_without_changing_caller(setup):
    manager, agent = setup()
    with torch.no_grad():
        fisher, _ = manager._estimate_fisher(agent, [("valid", 2.0)])
        assert not torch.is_grad_enabled()
    torch.testing.assert_close(fisher["weight"], torch.tensor(64.0))
