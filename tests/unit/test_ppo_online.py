"""Actual categorical sampling, retained likelihoods and exact PPO continuation."""

import copy
import json
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.experimental import ppo, ppo_online
from tests.unit.test_ppo_safety import (
    CharacterTokenizer,
    assert_same,
    snapshot,
)
from tests.unit.test_ppo_safety import trainer as trainer_fixture


class NonRoundTripTokenizer(CharacterTokenizer):
    eos_token_id = None

    def decode(self, ids, *, skip_special_tokens):
        # Re-encoding this text cannot recover sampled token IDs.
        return "tokens:" + ",".join(map(str, ids))

    def save_pretrained(self, path):
        (Path(path) / "tokenizer_config.json").write_text("{}")


@pytest.fixture
def online(monkeypatch, tmp_path):
    trainer = trainer_fixture.__wrapped__(monkeypatch)
    trainer.config.output_dir = str(tmp_path)
    trainer.config.num_generations = 2
    trainer.config.mini_batch_size = 2
    trainer.config.max_completion_length = 3
    trainer.tokenizer = NonRoundTripTokenizer()
    trainer.reward_fn = lambda prompt, completion: 1.0 if completion else 0.0
    monkeypatch.setattr(
        trainer.model,
        "save_pretrained",
        lambda path: torch.save(
            trainer.model.state_dict(), Path(path) / "pytorch_model.bin"
        ),
        raising=False,
    )
    return trainer


def receipt(result, step):
    return json.loads(
        (Path(result["rollout_directory"]) / f"rollout-{step}.json").read_text()
    )


@pytest.mark.parametrize("temperature", [0.5, 1.0, 1.8])
async def test_actual_sampling_retains_exact_behavior_scores_and_commits(
    online, temperature
):
    online.config.temperature = temperature
    behavior = copy.deepcopy(online.model).eval()
    before = snapshot(online)
    result = await online.train(["ab", "c"], num_episodes=1)
    record = receipt(result, 0)
    assert result["updates"] == online.global_step == 1
    assert len(record["rows"]) == 4
    for row in record["rows"]:
        assert len(row["action_ids"]) == 3
        assert row["termination"] == "horizon"
        assert row["completion"] == online.tokenizer.decode(
            row["action_ids"], skip_special_tokens=True
        )
        prefix = list(row["prompt_ids"])
        for action, likelihood in zip(
            row["action_ids"], row["behavior_log_probs"], strict=True
        ):
            ids = torch.tensor([prefix])
            logits = behavior(ids, torch.ones_like(ids)).logits[0, -1]
            expected = torch.log_softmax(logits / temperature, dim=-1)[action].item()
            assert likelihood == pytest.approx(expected, abs=1e-6)
            prefix.append(action)
    commit = json.loads(
        (Path(result["rollout_directory"]) / "commit-0.json").read_text()
    )
    assert commit["rollout_sha256"] == content_hash(record)
    assert commit["global_step"] == 1
    assert any(
        not torch.equal(before["policy"][k], v)
        for k, v in online.model.state_dict().items()
    )
    assert any(
        not torch.equal(before["critic"][k], v)
        for k, v in online.value_head.state_dict().items()
    )
    assert online.model.training and online.value_head.training


async def test_eos_action_is_included_once_and_keeps_its_likelihood(online):
    online.tokenizer.eos_token_id = 7
    with torch.no_grad():
        online.model.projection.weight.zero_()
        online.model.projection.bias.fill_(-1000)
        online.model.projection.bias[7] = 0
    result = await online.train(["a"], num_episodes=1)
    assert all(
        row["action_ids"] == [7] and row["termination"] == "eos"
        for row in receipt(result, 0)["rows"]
    )


async def test_checkpoint_reload_reproduces_next_sampling_and_update(online, tmp_path):
    await online.train(["ab", "c"], num_episodes=1)
    checkpoint = online.save_checkpoint(str(tmp_path / "checkpoint"))
    first = await online.train(["ab", "c"], num_episodes=2)
    expected = snapshot(online)
    online.load_runtime_checkpoint(checkpoint, trusted=True)
    second = await online.train(["ab", "c"], num_episodes=2)
    assert receipt(first, 1) == receipt(second, 1)
    assert_same(snapshot(online), expected)
    assert (await online.train(["ab", "c"], num_episodes=2))["updates"] == 0


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, 1e100])
async def test_invalid_reward_preserves_learner_and_sampling_rng(online, invalid):
    before, rng = snapshot(online), torch.get_rng_state().clone()
    online.reward_fn = lambda *_: invalid
    with pytest.raises(ValueError, match="rewards must"):
        await online.train(["ab"], num_episodes=1)
    assert_same(snapshot(online), before)
    torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
    assert not getattr(online, "_runtime_checkpoint_poisoned", False)
    assert not online._ppo_online_active


async def test_async_reward_failure_can_retry_same_samples(online):
    seen = []

    async def fail(prompt, completion):
        seen.append(completion)
        raise RuntimeError("reward transport failure")

    online.reward_fn = fail
    with pytest.raises(RuntimeError, match="reward transport"):
        await online.train(["a"], num_episodes=1)

    async def succeed(prompt, completion):
        seen.append(completion)
        return 1.0

    online.reward_fn = succeed
    await online.train(["a"], num_episodes=1)
    assert seen[0] == seen[1]


@pytest.mark.parametrize(
    "mutation", ["weights", "nonfinite", "config", "optimizer", "gradients", "metrics"]
)
async def test_reward_callback_cannot_change_behavior_learner(online, mutation):
    def reward(*_):
        if mutation in {"weights", "nonfinite"}:
            with torch.no_grad():
                next(online.model.parameters()).add_(
                    1.0 if mutation == "weights" else float("nan")
                )
        elif mutation == "config":
            online.config.temperature *= 0.9
        elif mutation == "optimizer":
            online.policy_optimizer.param_groups[0]["lr"] *= 2
        elif mutation == "gradients":
            parameter = next(online.model.parameters())
            parameter.grad = torch.ones_like(parameter)
        else:
            online.metrics_history["policy_loss"].append(123.0)
        return 1.0

    online.reward_fn = reward
    with pytest.raises((RuntimeError, ValueError, FloatingPointError)):
        await online.train(["a"], num_episodes=1)
    assert online.global_step == 0
    assert online._runtime_checkpoint_poisoned
    with pytest.raises(RuntimeError, match="quarantined"):
        await online.train(["a"], num_episodes=1)


@pytest.mark.parametrize("operation", ["train", "train_step", "save", "reload"])
async def test_reentrant_training_and_checkpoint_operations_are_rejected(
    online, operation
):
    async def reward(*_):
        with pytest.raises(RuntimeError, match="active"):
            if operation == "train":
                await online.train(["a"], num_episodes=1)
            elif operation == "train_step":
                await online.train_step(["a"], ["b"], [1.0])
            elif operation == "save":
                online.save_checkpoint()
            else:
                online.load_runtime_checkpoint("unused")
        return 1.0

    online.reward_fn = reward
    await online.train(["a"], num_episodes=1)
    assert online.global_step == 1


@pytest.mark.parametrize("drift", ["prompts", "tokenizer", "eos"])
async def test_bound_prompt_population_and_tokenizer_cannot_drift(online, drift):
    await online.train(["a"], num_episodes=1)
    if drift == "tokenizer":

        class ChangedTokenizer(NonRoundTripTokenizer):
            def __call__(self, *args, **kwargs):
                encoded = super().__call__(*args, **kwargs)
                encoded["input_ids"] += 1
                return encoded

        online.tokenizer = ChangedTokenizer()
    elif drift == "eos":
        online.tokenizer.eos_token_id = 7
    with pytest.raises(ValueError, match="prompt population"):
        await online.train(["b"] if drift == "prompts" else ["a"], num_episodes=2)
    assert online.global_step == 1


async def test_commit_receipt_failure_quarantines_updated_learner(online, monkeypatch):
    write = ppo_online._write

    def fail(path, data):
        if path.name.startswith("commit"):
            raise OSError("journal disk full")
        write(path, data)

    monkeypatch.setattr(ppo_online, "_write", fail)
    with pytest.raises(OSError, match="disk full"):
        await online.train(["a"], num_episodes=1)
    assert online.global_step == 1
    assert online._runtime_checkpoint_poisoned


async def test_real_causal_transformer_online_update_without_download(online):
    from transformers import GPT2Config, GPT2LMHeadModel

    config = GPT2Config(
        vocab_size=32,
        n_embd=8,
        n_layer=1,
        n_head=2,
        n_positions=16,
        resid_pdrop=0.2,
        embd_pdrop=0.2,
        attn_pdrop=0.2,
    )
    trainer = ppo.PPOTrainer(
        online.config, GPT2LMHeadModel(config), online.tokenizer, online.reward_fn
    )
    result = await trainer.train(["ab", "c"], num_episodes=1)
    assert result["updates"] == 1
    assert len(receipt(result, 0)["rows"]) == 4


async def test_convenience_loader_pins_both_assets_and_saves_checkpoint(
    online, monkeypatch
):
    model_loader, tokenizer_loader = Mock(), Mock()
    model_loader.from_pretrained.return_value = online.model
    tokenizer_loader.from_pretrained.return_value = online.tokenizer
    monkeypatch.setattr(ppo, "AutoModelForCausalLM", model_loader)
    monkeypatch.setattr(ppo, "AutoTokenizer", tokenizer_loader)
    revision = "a" * 40
    result = await ppo.train_ppo(
        "test/causal",
        ["a"],
        online.reward_fn,
        num_episodes=1,
        use_lora=False,
        output_dir=online.config.output_dir,
        model_revision=revision,
        max_prompt_length=4,
        max_completion_length=2,
        mini_batch_size=1,
        num_generations=2,
        num_ppo_epochs=1,
        value_hidden_size=4,
        report_to="none",
    )
    for loader in (model_loader, tokenizer_loader):
        assert loader.from_pretrained.call_args.kwargs["revision"] == revision
        assert loader.from_pretrained.call_args.kwargs["trust_remote_code"] is False
    assert (Path(result["checkpoint"]) / "training_state.pt").is_file()


@pytest.mark.parametrize(
    "field,value",
    [
        ("top_p", 0.9),
        ("top_k", 5),
        ("gradient_accumulation_steps", 2),
        ("num_generations", 0),
    ],
)
async def test_unsupported_online_settings_reject_before_scoring(online, field, value):
    setattr(online.config, field, value)
    online.model.forward = Mock(side_effect=AssertionError("unexpected scoring"))
    with pytest.raises(ValueError):
        await online.train(["a"], num_episodes=1)
    online.model.forward.assert_not_called()
    assert online.global_step == 0


async def test_online_reference_kl_uses_frozen_model_without_changing_it(online):
    online.config.beta = 0.1
    online.ref_model = copy.deepcopy(online.model).eval().requires_grad_(False)
    before = copy.deepcopy(online.ref_model.state_dict())
    await online.train(["ab", "c"], num_episodes=1)
    assert_same(online.ref_model.state_dict(), before)


async def test_fresh_trainer_resumes_bound_prompt_cursor_and_rng(
    online, monkeypatch, tmp_path
):
    await online.train(["ab", "c"], num_episodes=1)
    checkpoint = online.save_checkpoint(str(tmp_path / "resume"))
    expected_run = await online.train(["ab", "c"], num_episodes=2)
    expected = snapshot(online)
    config = copy.deepcopy(online.config)
    config.resume_from_checkpoint = checkpoint
    config.ppo_prompt_sha256 = None  # Rebind from the actual supplied population.
    restored = ppo.PPOTrainer(
        config, copy.deepcopy(online.model), online.tokenizer, online.reward_fn
    )
    # Local fixture writes a trusted pickle; normal save_pretrained uses safetensors.
    load = restored.load_runtime_checkpoint
    monkeypatch.setattr(
        restored, "load_runtime_checkpoint", lambda path: load(path, trusted=True)
    )
    actual_run = await restored.train(["ab", "c"], num_episodes=2)
    assert receipt(actual_run, 1)["rows"] == receipt(expected_run, 1)["rows"]
    assert_same(snapshot(restored), expected)


async def test_partial_online_optimizer_failure_restores_from_complete_checkpoint(
    online, monkeypatch, tmp_path
):
    await online.train(["a"], num_episodes=1)
    checkpoint = online.save_checkpoint(str(tmp_path / "before-failure"))
    before = snapshot(online)
    step = online.value_optimizer.step

    def fail():
        raise RuntimeError("critic commit failed")

    monkeypatch.setattr(online.value_optimizer, "step", fail)
    with pytest.raises(RuntimeError, match="critic commit"):
        await online.train(["a"], num_episodes=2)
    assert online._runtime_checkpoint_poisoned
    monkeypatch.setattr(online.value_optimizer, "step", step)
    online.load_runtime_checkpoint(checkpoint, trusted=True)
    assert_same(snapshot(online), before)
    assert (await online.train(["a"], num_episodes=2))["updates"] == 1


async def test_sampled_bandit_reward_increases_target_action_probability(online):
    """A real optimizer must improve a known synthetic objective, not just run."""
    online.config.max_completion_length = 1
    online.config.num_episodes = 20
    online.config.mini_batch_size = 4
    online.config.num_generations = 4
    with torch.no_grad():
        online.model.projection.weight.zero_()
        online.model.projection.bias.fill_(-10)
        online.model.projection.bias[7:9] = 0
    for group in online.policy_optimizer.param_groups:
        group["lr"] = group["initial_lr"] = 0.03
    online.policy_scheduler.base_lrs = [0.03]
    online.reward_fn = lambda _, completion: float(completion == "tokens:7")
    ids = torch.tensor([[1 + ord("a") % 31]])

    def probability():
        with torch.no_grad():
            logits = online.model(ids, torch.ones_like(ids)).logits[0, -1]
            return torch.softmax(logits / online.config.temperature, -1)[7].item()

    before = probability()
    result = await online.train(["a"])
    assert result["updates"] == 20
    assert before < 0.51
    assert probability() > 0.8


@pytest.mark.parametrize(
    "field,value",
    [
        ("gamma", -0.1),
        ("gamma", 1.1),
        ("gae_lambda", float("nan")),
        ("value_coef", -1),
        ("value_clip_eps", float("inf")),
        ("value_hidden_size", 0),
        ("target_kl", -1),
        ("use_adaptive_kl", True),
    ],
)
def test_invalid_ppo_estimator_settings_reject(field, value):
    config = ppo.PPOConfig(**{field: value})
    with pytest.raises(ValueError):
        config.validate_for_training()
