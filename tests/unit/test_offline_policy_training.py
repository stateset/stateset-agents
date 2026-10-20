"""Offline policy extraction must train logged tokens and reload identical weights."""

from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from stateset_agents.training.offline_grpo_trainer import (
    OfflineGRPOConfig,
    OfflineGRPOTrainer,
)


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(8, 8)
        self.head = torch.nn.Linear(8, 8)

    def forward(self, input_ids, attention_mask=None):
        return SimpleNamespace(logits=self.head(self.embedding(input_ids)))


class LoggedDataset:
    def to_offline_rl_format(self):
        return {
            "states": np.ones((3, 2)),
            "actions": np.ones((3, 2)),
            "next_states": np.zeros((3, 2)),
            "rewards": np.ones(3),
            "dones": np.ones(3),
        }

    def to_policy_training_format(self, tokenizer, max_length):
        return [{"input_ids": [1, 2, 3], "labels": [-100, -100, 3]} for _ in range(3)]


def trainer(model=None):
    return OfflineGRPOTrainer(
        OfflineGRPOConfig(
            state_dim=2,
            action_dim=2,
            value_hidden_size=4,
            value_num_layers=2,
            pretrain_value_epochs=1,
            pretrain_value_batch_size=4,
            batch_size=2,
            learning_rate=0.05,
            num_epochs=1,
            gradient_accumulation_steps=1,
        ),
        model=model if model is not None else TinyPolicy(),
        tokenizer=SimpleNamespace(pad_token_id=0),
        device="cpu",
    )


async def test_real_logged_actions_update_policy_and_include_partial_batch(tmp_path):
    torch.manual_seed(3)
    instance = trainer()
    instance.config.adam_beta1, instance.config.adam_beta2 = 0.5, 0.8
    instance.config.weight_decay = 0.2
    original = {
        name: p.detach().clone() for name, p in instance.model.named_parameters()
    }
    events = []
    instance.callbacks = [lambda event, data: events.append(event)]
    result = await instance.train(dataset=LoggedDataset())
    assert result["optimizer_updates"] == 2
    assert any(
        not torch.equal(original[name], p)
        for name, p in instance.model.named_parameters()
    )
    assert events.count("step_end") == 2 and events.count("episode_end") == 1
    path = tmp_path / "checkpoint"
    instance.save_checkpoint(str(path))
    restored = trainer()
    restored.load(str(path / "training_state.pt"))
    assert restored.training_step == 2 and restored.current_epoch == 1
    assert restored.policy_optimizer.param_groups[0]["betas"] == (0.5, 0.8)
    assert restored.policy_optimizer.param_groups[0]["weight_decay"] == 0.2
    assert all(
        torch.equal(p, restored.model.state_dict()[name])
        for name, p in instance.model.state_dict().items()
    )
    assert restored._rng.bit_generator.state == instance._rng.bit_generator.state
    await instance.train(num_epochs=2, dataset=LoggedDataset())
    await restored.train(num_epochs=2, dataset=LoggedDataset())
    assert all(
        torch.equal(p, restored.model.state_dict()[name])
        for name, p in instance.model.state_dict().items()
    )


async def test_assistant_only_loss_and_no_value_gradient_during_policy_step():
    instance = trainer()
    await instance.pretrain_value_functions(LoggedDataset())
    ids = [[1, 2, 3]]
    batch = {
        "input_ids": ids,
        "labels": [[-100, -100, 3]],
        "states": [[1.0, 1.0]],
        "actions": [[1.0, 1.0]],
    }
    instance.model.eval()
    expected = torch.nn.functional.cross_entropy(
        instance.model(torch.tensor(ids)).logits[:, 1, :], torch.tensor([3])
    )
    result = await instance.train_step(batch)
    assert result["nll"] == pytest.approx(float(expected.detach()))
    assert result["supervised_tokens"] == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("labels", [[-100, -100, -100]]),
        ("labels", [[-100, -100, 4]]),
        ("input_ids", [[True, False, True]]),
    ],
)
async def test_bad_logged_tokens_fail_before_policy_changes(field, value):
    instance = trainer()
    await instance.pretrain_value_functions(LoggedDataset())
    original = {
        name: p.detach().clone() for name, p in instance.model.named_parameters()
    }
    batch = {
        "input_ids": [[1, 2, 3]],
        "labels": [[-100, -100, 3]],
        "states": [[1.0, 1.0]],
        "actions": [[1.0, 1.0]],
        field: value,
    }
    with pytest.raises(ValueError):
        await instance.train_step(batch)
    assert instance.training_step == 0
    assert all(
        torch.equal(original[name], p) for name, p in instance.model.named_parameters()
    )


async def test_missing_language_model_cannot_report_policy_training():
    instance = trainer()
    instance.model = None
    with pytest.raises(ValueError, match="language model"):
        await instance.train(dataset=LoggedDataset())


async def test_resuming_cannot_silently_change_the_logged_dataset(tmp_path):
    instance = trainer()
    await instance.train(dataset=LoggedDataset())
    path = tmp_path / "state.pt"
    instance.save(str(path))
    restored = trainer()
    restored.load(str(path), require_policy=True)

    class ChangedDataset(LoggedDataset):
        def to_offline_rl_format(self):
            features = super().to_offline_rl_format()
            features["rewards"][0] = -1
            return features

    with pytest.raises(ValueError, match="dataset differs"):
        await restored.train(num_epochs=2, dataset=ChangedDataset())


async def test_partial_accumulation_commits_a_real_final_update():
    instance = trainer()
    instance.config.gradient_accumulation_steps = 4
    result = await instance.train(dataset=LoggedDataset())
    assert result["optimizer_updates"] == 1 and instance._grad_accum_step == 0
    assert result["final_metrics"]["optimizer_updates"] == 1
    assert result["final_metrics"]["grad_norm"] > 0
    assert sum(row["optimizer_updates"] for row in result["training_metrics"]) == 1


def varied_policy_batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3], [1, 4, 5], [2, 6, 7]]),
        "labels": torch.tensor([[-100, -100, 3], [-100, 4, 5], [-100, -100, 7]]),
        "states": torch.tensor([[1.0, 0.0], [0.0, 1.0], [-1.0, 0.5]]),
        "actions": torch.tensor([[0.0, 1.0], [1.0, 0.0], [0.5, -1.0]]),
    }


@pytest.mark.parametrize(
    "sizes,accumulation", [([2, 1], 2), ([1, 2], 2), ([2, 1], 4), ([1, 1, 1], 4)]
)
async def test_accumulation_matches_combined_action_objective(sizes, accumulation):
    """Changing microbatch boundaries must preserve each action's weight."""
    torch.manual_seed(19)
    reference = trainer()
    reference.config.max_grad_norm = 1000.0
    await reference.pretrain_value_functions(LoggedDataset())
    accumulated = deepcopy(reference)
    accumulated.config.gradient_accumulation_steps = accumulation
    # SGD makes the parameter delta directly measure the gradient, without
    # Adam's adaptive normalization concealing an incorrect gradient magnitude.
    for instance in (reference, accumulated):
        instance.policy_optimizer = torch.optim.SGD(
            instance.model.parameters(), lr=0.05
        )
    batch = varied_policy_batch()
    await reference.train_step(batch)
    start = 0
    for size in sizes:
        await accumulated.train_step(
            {name: values[start : start + size] for name, values in batch.items()}
        )
        start += size
    if accumulated._grad_accum_step:
        accumulated._commit_policy_update()
    assert reference.optimizer_updates == accumulated.optimizer_updates == 1
    for name, parameter in reference.model.named_parameters():
        torch.testing.assert_close(
            parameter, accumulated.model.state_dict()[name], rtol=1e-6, atol=1e-7
        )


async def test_uneven_pending_gradient_resume_matches_combined_batch(tmp_path):
    torch.manual_seed(23)
    instance = trainer()
    instance.config.gradient_accumulation_steps = 4
    await instance.pretrain_value_functions(LoggedDataset())
    reference = deepcopy(instance)
    reference.config.gradient_accumulation_steps = 1
    batch = varied_policy_batch()
    combined = {name: torch.cat([value, value[:1]]) for name, value in batch.items()}
    expected = await reference.train_step(combined)
    await instance.train_step({name: value[:2] for name, value in batch.items()})
    await instance.train_step({name: value[2:] for name, value in batch.items()})
    checkpoint = tmp_path / "pending.pt"
    instance.save(str(checkpoint))
    restored = trainer()
    restored.load(str(checkpoint), require_policy=True)
    assert restored._grad_accum_step == 2 and restored._grad_accum_actions == 3
    assert restored._grad_accum_normalizer == 8
    await restored.train_step({name: value[:1] for name, value in batch.items()})
    norm = restored._commit_policy_update()
    assert norm == pytest.approx(expected["grad_norm"], abs=1e-6)
    assert restored._grad_accum_actions == restored._grad_accum_normalizer == 0
    for name, parameter in reference.model.named_parameters():
        torch.testing.assert_close(
            parameter, restored.model.state_dict()[name], rtol=1e-6, atol=1e-7
        )


@pytest.mark.parametrize("pending", [False, True])
async def test_legacy_checkpoint_requires_completed_accumulation(tmp_path, pending):
    instance = trainer()
    instance.config.gradient_accumulation_steps = 4
    await instance.pretrain_value_functions(LoggedDataset())
    await instance.train_step(varied_policy_batch())
    if not pending:
        instance._commit_policy_update()
    checkpoint = tmp_path / "legacy.pt"
    instance.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=True)
    del payload["policy_accumulation"]
    torch.save(payload, checkpoint)
    restored = trainer()
    original = deepcopy(restored.model.state_dict())
    if pending:
        with pytest.raises(ValueError, match="pending gradients without action counts"):
            restored.load(str(checkpoint), require_policy=True)
        for name, value in restored.model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)
    else:
        restored.load(str(checkpoint), require_policy=True)
        assert restored._grad_accum_step == restored._grad_accum_actions == 0
        for name, value in restored.model.state_dict().items():
            torch.testing.assert_close(value, instance.model.state_dict()[name])


@pytest.mark.parametrize(
    "changes",
    [
        {"action_count": 0},
        {"normalizer": 0},
        {"schema_version": 2},
        {"action_count": True},
    ],
)
async def test_invalid_pending_action_state_fails_before_policy_changes(
    tmp_path, changes
):
    instance = trainer()
    instance.config.gradient_accumulation_steps = 4
    await instance.pretrain_value_functions(LoggedDataset())
    await instance.train_step(varied_policy_batch())
    checkpoint = tmp_path / "invalid.pt"
    instance.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=True)
    payload["policy_accumulation"].update(changes)
    torch.save(payload, checkpoint)
    restored = trainer()
    original = deepcopy(restored.model.state_dict())
    with pytest.raises(ValueError, match="offline checkpoint.*accumulation"):
        restored.load(str(checkpoint), require_policy=True)
    for name, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


async def test_pending_counts_cannot_hide_missing_gradients(tmp_path):
    instance = trainer()
    instance.config.gradient_accumulation_steps = 4
    await instance.pretrain_value_functions(LoggedDataset())
    await instance.train_step(varied_policy_batch())
    checkpoint = tmp_path / "missing-gradients.pt"
    instance.save(str(checkpoint))
    payload = torch.load(checkpoint, weights_only=True)
    payload["gradients"] = {}
    torch.save(payload, checkpoint)
    restored = trainer()
    original = deepcopy(restored.model.state_dict())
    with pytest.raises(ValueError, match="missing pending policy gradients"):
        restored.load(str(checkpoint), require_policy=True)
    for name, value in restored.model.state_dict().items():
        torch.testing.assert_close(value, original[name], rtol=0, atol=0)


async def test_mid_epoch_resume_preserves_cursor_and_pending_gradients(tmp_path):
    torch.manual_seed(7)
    instance = trainer()
    instance.config.gradient_accumulation_steps = 4
    checkpoint = tmp_path / "interrupted.pt"

    class Interrupted(RuntimeError):
        pass

    async def interrupt(event, metrics):
        if event == "step_end":
            instance.save(str(checkpoint))
            raise Interrupted("simulate interrupted caller")

    interrupt.fail_on_error = True
    instance.callbacks = [interrupt]
    with pytest.raises(Interrupted):
        await instance.train(dataset=LoggedDataset())
    assert instance._batch_offset == 2 and instance._grad_accum_step == 1
    restored = trainer()
    restored.config.gradient_accumulation_steps = 4
    restored.load(str(checkpoint), require_policy=True)
    assert restored._grad_accum_actions == 2
    instance.callbacks = []
    await instance.train(dataset=LoggedDataset())
    await restored.train(dataset=LoggedDataset())
    assert instance.training_step == restored.training_step == 2
    assert all(
        torch.equal(value, restored.model.state_dict()[name])
        for name, value in instance.model.state_dict().items()
    )


async def test_public_offline_entrypoint_embeds_logs_updates_and_publishes(
    tmp_path, monkeypatch
):
    from stateset_agents.training.train import train

    class LocalFeatures:
        embedding_dim = 2

        def __init__(self, **kwargs):
            pass

        def get_or_compute(self, key, text):
            return torch.tensor([float("eligible" in text), 1.0])

    monkeypatch.setattr(
        "stateset_agents.data.conversation_dataset.EmbeddingCache", LocalFeatures
    )
    path = tmp_path / "logs.jsonl"
    path.write_text(
        '{"trajectory_id":"one","turns":[{"role":"user","content":"eligible"},'
        '{"role":"assistant","content":"refund","reward":1}]}\n'
    )
    from stateset_agents.training.sampled_workflow import workflow_tokenizer

    tokenizer = workflow_tokenizer()
    tokenizer.chat_template = (
        "{% for message in messages %}{{ message['content'] }} {% endfor %}"
    )
    policy = TinyPolicy()
    # The fixture tokenizer includes tokens beyond TinyPolicy's eight-token vocabulary.
    policy.embedding = torch.nn.Embedding(len(tokenizer), 8)
    policy.head = torch.nn.Linear(8, len(tokenizer))
    agent = SimpleNamespace(
        model=policy,
        tokenizer=tokenizer,
        config=SimpleNamespace(model_name="local-policy", model_revision="a" * 40),
    )
    original = policy.head.weight.detach().clone()
    checkpoint = tmp_path / "published"
    completed = []
    kwargs = {
        "agent": agent,
        "environment": SimpleNamespace(reward_fn=None),
        "training_mode": "offline",
        "dataset_path": str(path),
        "num_epochs": 1,
        "gradient_accumulation_steps": 1,
        "learning_rate": 0.01,
    }
    await train(
        **kwargs,
        save_path=str(checkpoint),
        callbacks=[lambda event, data: completed.append((event, data))],
    )
    assert not torch.equal(original, policy.head.weight)
    assert (checkpoint / "training_state.pt").is_file()
    assert (checkpoint / "pytorch_model.bin").is_file()
    assert completed[-1][0] == "training_end"
    assert completed[-1][1]["optimizer_updates"] == 1
    kwargs["num_epochs"] = 2
    await train(**kwargs, resume_from_checkpoint=str(checkpoint))


async def test_public_offline_training_rejects_online_objectives_before_loading():
    from stateset_agents.training.train import train

    with pytest.raises(ValueError, match="IQL-weighted logged-action extraction"):
        await train(
            agent=SimpleNamespace(config=SimpleNamespace(model_name="local")),
            environment=SimpleNamespace(reward_fn=None),
            training_mode="offline",
            objective="gspo",
            dataset_path="must-not-be-loaded.jsonl",
        )
