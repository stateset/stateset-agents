"""PreferenceTrainer / RewardModelTrainer mechanics on a tiny GPT-2."""

from __future__ import annotations

import asyncio
import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.reward_base import RewardResult  # noqa: E402
from stateset_agents.core.trajectory import ConversationTurn  # noqa: E402
from stateset_agents.training.preference import (  # noqa: E402
    PairwiseRewardFunction,
    PreferenceConfig,
    PreferenceDataset,
    PreferenceTrainer,
    RewardModelConfig,
    RewardModelTrainer,
    load_reward_function,
    train_preference,
    train_reward_model,
)
from stateset_agents.training.preference import trainer as trainer_module  # noqa: E402
from tests.learning._toy import make_model, make_tokenizer  # noqa: E402


@pytest.fixture(autouse=True)
def _single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return make_tokenizer()


def _pairs(prompts: list[str], **extra: Any) -> PreferenceDataset:
    return PreferenceDataset.from_records(
        [
            {
                "prompt": [{"role": "user", "content": prompt}],
                "chosen": f"{index}{index}",
                "rejected": "ab" if index % 2 else "xyz",
                **extra,
            }
            for index, prompt in enumerate(prompts)
        ]
    )


TRAIN = ["hello", "count", "say a number", "abc def"]
HELD = ["xyz", "ok go"]


def _config(**overrides: Any) -> PreferenceConfig:
    values: dict[str, Any] = {
        "learning_rate": 1e-2,
        "batch_size": 4,
        "max_steps": 2,
        "device": "cpu",
        "max_length": 48,
        "max_prompt_length": None,
        "lr_scheduler": "constant",
    }
    values.update(overrides)
    return PreferenceConfig(**values)


def _trainer(
    tokenizer: Any, seed: int = 0, train: Any = None, **overrides: Any
) -> PreferenceTrainer:
    return PreferenceTrainer(
        make_model(seed, n_embd=16),
        tokenizer,
        _config(**overrides),
        train if train is not None else _pairs(TRAIN),
        _pairs(HELD),
    )


def _params(model: Any) -> list[torch.Tensor]:
    return [p.detach().clone() for p in model.parameters()]


def _max_diff(a: list[torch.Tensor], b: list[torch.Tensor]) -> float:
    return max(float((x - y).abs().max()) for x, y in zip(a, b, strict=True))


# A large Adam epsilon makes the first update linear in the gradient, so these
# exactness checks are not dominated by Adam's sign-like first step.
LINEAR_ADAM = {"adam_epsilon": 1.0}


def test_accumulation_equals_one_large_batch(tokenizer: Any) -> None:
    weighted = PreferenceDataset(
        tuple(
            type(pair)(pair.prompt, pair.chosen, pair.rejected, weight=1.0 + index)
            for index, pair in enumerate(_pairs(TRAIN).pairs)
        )
    )
    big = _trainer(tokenizer, train=weighted, max_steps=1, **LINEAR_ADAM)
    small = _trainer(
        tokenizer,
        train=weighted,
        max_steps=1,
        batch_size=2,
        gradient_accumulation_steps=2,
        **LINEAR_ADAM,
    )
    big.train()
    small.train()
    assert big.global_step == small.global_step == 1
    assert _max_diff(_params(big.model), _params(small.model)) < 1e-6


def test_reference_is_a_frozen_copy(tokenizer: Any) -> None:
    trainer = _trainer(tokenizer)
    before = _params(trainer.model)
    trainer.train()
    assert all(not p.requires_grad for p in trainer.ref_model.parameters())
    assert _max_diff(before, _params(trainer.ref_model)) == 0.0
    assert _max_diff(before, _params(trainer.model)) > 0.0
    assert not trainer.ref_model.training


def test_precomputed_reference_matches_online_reference(tokenizer: Any) -> None:
    online = _trainer(tokenizer, **LINEAR_ADAM)
    offline = _trainer(tokenizer, precompute_reference_logps=True, **LINEAR_ADAM)
    assert offline.ref_model is None  # released after scoring
    online_summary, offline_summary = online.train(), offline.train()
    assert _max_diff(_params(online.model), _params(offline.model)) < 1e-6
    assert online_summary.eval_metrics == pytest.approx(
        offline_summary.eval_metrics, rel=1e-5, abs=1e-6
    )
    with pytest.raises(RuntimeError, match="released"):
        offline.evaluate(_pairs(["fresh prompt"]))


@pytest.mark.parametrize("method", ["simpo", "orpo"])
def test_reference_free_methods_have_no_reference(tokenizer: Any, method: str) -> None:
    trainer = _trainer(tokenizer, method=method)
    assert trainer.ref_model is None
    assert trainer.train().global_step == 2
    with pytest.raises(ValueError, match="reference-free"):
        PreferenceTrainer(
            make_model(0, n_embd=16),
            tokenizer,
            _config(method=method),
            _pairs(TRAIN),
            ref_model=make_model(1, n_embd=16),
        )


def test_lora_reference_is_the_adapter_disabled_policy(tokenizer: Any) -> None:
    pytest.importorskip("peft")
    base = make_model(0, n_embd=16)
    reference = copy.deepcopy(base)
    trainer = PreferenceTrainer(
        base,
        tokenizer,
        _config(use_lora=True, lora_target_modules=["c_attn"]),
        _pairs(TRAIN),
    )
    assert trainer.ref_model is None and hasattr(trainer.model, "peft_config")
    trainer.train()
    sequences = [item.chosen for item in trainer._train_pairs]
    with torch.no_grad():
        ours, _ = trainer._reference_logps(sequences)
        reference.eval()
        expected, _ = trainer._logps(reference, sequences)
        policy, _ = trainer._logps(trainer.model, sequences)
    assert torch.allclose(ours, expected, atol=1e-5)
    assert not torch.allclose(policy, ours, atol=1e-4)  # the adapter trained


def test_kto_pairs_each_prompt_with_another_completion(tokenizer: Any) -> None:
    trainer = _trainer(tokenizer, method="kto", batch_size=2)
    items = trainer._train_unpaired
    assert len(items) == 2 * len(TRAIN) and trainer._train_pairs == []
    assert all(i.kl_sequence.prompt_ids == i.sequence.prompt_ids for i in items)
    # the KL completions are a permutation of the completions, never all own
    assert sorted(i.kl_sequence.response_ids for i in items) == sorted(
        i.sequence.response_ids for i in items
    )
    assert any(i.kl_sequence.response_ids != i.sequence.response_ids for i in items)
    summary = trainer.train()
    assert "kl" in summary.history[-1] and "eval/kl" in summary.eval_metrics


def test_kto_kl_partner_is_never_the_example_itself(tokenizer: Any) -> None:
    data = PreferenceDataset.from_records(
        [
            {"prompt": f"p{i}", "completion": str(i) * 2, "label": i % 2 == 0}
            for i in range(7)
        ]
    )
    trainer = _trainer(tokenizer, method="kto", train=data)
    assert all(
        i.kl_sequence.response_ids != i.sequence.response_ids
        for i in trainer._train_unpaired
    )


def test_unpaired_records_need_kto(tokenizer: Any) -> None:
    data = PreferenceDataset.from_records(
        [{"prompt": "Q", "completion": "a", "label": True}]
    )
    with pytest.raises(ValueError, match="kto"):
        _trainer(tokenizer, train=data)


def test_eval_data_must_not_share_prompts(tokenizer: Any) -> None:
    with pytest.raises(ValueError, match="overlap"):
        PreferenceTrainer(
            make_model(0, n_embd=16),
            tokenizer,
            _config(),
            _pairs(TRAIN),
            _pairs(["HELLO"]),
        )


def test_nonfinite_loss_raises_or_skips(
    tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    real = trainer_module.sequence_logprobs

    def poisoned(*args: Any, **kwargs: Any) -> Any:
        sums, counts = real(*args, **kwargs)
        return sums * math.nan, counts

    monkeypatch.setattr(trainer_module, "sequence_logprobs", poisoned)
    raising = _trainer(tokenizer)
    with pytest.raises(ValueError, match="non-finite"):
        raising.train()
    skipping = _trainer(tokenizer, nonfinite_policy="skip")
    before = _params(skipping.model)
    monkeypatch.setattr(skipping, "evaluate", lambda *a, **k: {})
    summary = skipping.train()
    assert (summary.global_step, summary.skipped_steps, summary.update_attempts) == (
        0,
        2,
        2,
    )
    assert _max_diff(before, _params(skipping.model)) == 0.0


def test_nonfinite_gradients_raise_or_skip(tokenizer: Any) -> None:
    for policy in ("raise", "skip"):
        trainer = _trainer(tokenizer, nonfinite_policy=policy, max_steps=1)
        parameter = next(trainer.model.parameters())
        parameter.register_hook(lambda grad: grad * math.inf)
        before = _params(trainer.model)
        if policy == "raise":
            with pytest.raises(ValueError, match="finite"):
                trainer.train()
        else:
            assert trainer.train().skipped_steps == 1
        assert _max_diff(before, _params(trainer.model)) == 0.0


def test_checkpoint_resume_is_exact(tokenizer: Any, tmp_path: Path) -> None:
    straight = _trainer(tokenizer, max_steps=4, batch_size=2, lr_scheduler="linear")
    straight.train()
    interrupted = _trainer(tokenizer, max_steps=4, batch_size=2, lr_scheduler="linear")
    interrupted.config.max_steps = 2  # stop a 4-step run halfway
    interrupted.train()
    checkpoint = interrupted.save_checkpoint(tmp_path / "ckpt")
    meta = json.loads((checkpoint / "preference_checkpoint.json").read_text())
    assert (
        meta["schema"] == "stateset-preference-checkpoint-v1"
        and meta["global_step"] == 2
    )
    resumed = _trainer(tokenizer, max_steps=4, batch_size=2, lr_scheduler="linear")
    summary = resumed.train(resume_from=checkpoint)
    assert summary.global_step == 4 and resumed.epoch == straight.epoch
    assert _max_diff(_params(straight.model), _params(resumed.model)) == 0.0
    with pytest.raises(ValueError, match="method"):
        _trainer(tokenizer, method="ipo").load_checkpoint(checkpoint)


def test_train_preference_api_writes_artifacts(tokenizer: Any, tmp_path: Path) -> None:
    rows = [
        {
            "prompt": [{"role": "user", "content": f"question {i}"}],
            "chosen": "11",
            "rejected": "aa",
        }
        for i in range(10)
    ]
    result = train_preference(
        make_model(0, n_embd=16),
        rows,
        tokenizer=tokenizer,
        config=_config(output_dir=str(tmp_path), validation_fraction=0.2),
        method="ipo",
    )
    assert result.summary.method == "ipo" and result.summary.eval_metrics
    assert result.summary.data["eval_pairs"] == 2
    metrics = json.loads((tmp_path / "preference_metrics.json").read_text())
    assert metrics["global_step"] == 2
    assert (tmp_path / "final" / "model" / "model.safetensors").exists()


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"method": "ppo"}, "method"),
        ({"beta": 0.0}, "beta"),
        ({"method": "ipo", "label_smoothing": 0.1}, "label_smoothing"),
        ({"method": "simpo", "precompute_reference_logps": True}, "reference-free"),
        ({"save_steps": 1}, "output_dir"),
        ({"max_prompt_length": 48}, "smaller"),
        ({"nonfinite_policy": "ignore"}, "nonfinite_policy"),
        ({"lora_target_modules": []}, "lora_target_modules"),
    ],
)
def test_config_validation(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _config(**overrides).validate()


def test_config_from_dict_rejects_unknown_keys() -> None:
    with pytest.raises(ValueError, match="unknown"):
        PreferenceConfig.from_dict({"betaa": 0.1})
    assert PreferenceConfig.from_dict({"method": "kto"}).method == "kto"
    assert RewardModelConfig.from_dict({}).learning_rate == 1e-5


# ------------------------------------------------------------ reward models


def _rm_trainer(tokenizer: Any, **overrides: Any) -> RewardModelTrainer:
    values: dict[str, Any] = {
        "learning_rate": 1e-2,
        "batch_size": 4,
        "max_steps": 3,
        "device": "cpu",
        "max_length": 48,
        "max_prompt_length": None,
    }
    values.update(overrides)
    return RewardModelTrainer(
        make_model(0, n_embd=16).transformer,
        tokenizer,
        RewardModelConfig(**values),
        _pairs(TRAIN),
        _pairs(HELD),
    )


def test_reward_model_trains_and_reports_auc(tokenizer: Any) -> None:
    trainer = _rm_trainer(tokenizer)
    summary = trainer.train()
    assert summary.method == "rm" and summary.global_step == 3
    assert set(summary.eval_metrics) >= {
        "eval/loss",
        "eval/accuracy",
        "eval/auc",
        "eval/margin",
    }
    assert {"accuracy", "margin", "loss"} <= set(summary.history[-1])


def test_reward_model_uses_pair_margins(tokenizer: Any) -> None:
    plain = _rm_trainer(tokenizer)
    with_margin = RewardModelTrainer(
        make_model(0, n_embd=16).transformer,
        tokenizer,
        plain.config,
        _pairs(TRAIN, margin=3.0),
    )
    items = with_margin._train_pairs
    margin_loss, _ = with_margin._pair_loss(items)
    no_margin_loss, _ = plain._pair_loss(plain._train_pairs)
    assert torch.all(margin_loss.losses > no_margin_loss.losses)


CONTRACT_INPUTS = {
    "empty": [],
    "single_assistant": [ConversationTurn(role="assistant", content="I can help.")],
    "multi_turn": [
        ConversationTurn(role="user", content="Where is my order?"),
        ConversationTurn(role="assistant", content="Let me check."),
        ConversationTurn(role="user", content="Thanks"),
        ConversationTurn(role="assistant", content="It ships tomorrow."),
    ],
    "blank_reply": [
        ConversationTurn(role="user", content="Hello?"),
        ConversationTurn(role="assistant", content=""),
    ],
    "very_long": [
        ConversationTurn(role="user", content="Status?"),
        ConversationTurn(role="assistant", content="Your refund is on its way. " * 400),
    ],
    "dicts": [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "42"},
    ],
}


@pytest.mark.parametrize("name", sorted(CONTRACT_INPUTS))
@pytest.mark.parametrize("context", [None, {}, {"reward_scope": "latest_turn"}])
def test_reward_function_keeps_the_reward_contract(
    tokenizer: Any, name: str, context: Any
) -> None:
    reward = _rm_trainer(tokenizer).reward_function(temperature=2.0)
    turns = copy.deepcopy(CONTRACT_INPUTS[name])
    snapshot, context_before = copy.deepcopy(turns), copy.deepcopy(context)
    first = asyncio.run(reward.compute_reward(turns, context))
    second = asyncio.run(
        reward.compute_reward(copy.deepcopy(turns), copy.deepcopy(context))
    )
    assert isinstance(first, RewardResult)
    assert math.isfinite(first.score) and 0.0 <= first.score <= 1.0
    assert first.score == second.score
    assert first.score == pytest.approx(
        1 / (1 + math.exp(-first.metadata["raw_reward"] / 2.0))
    )
    assert turns == snapshot and context == context_before


def test_reward_function_scores_the_newest_reply_in_context(tokenizer: Any) -> None:
    trainer = _rm_trainer(tokenizer)
    reward = trainer.reward_function()
    turns = CONTRACT_INPUTS["multi_turn"]
    result = asyncio.run(reward.compute_reward(turns, {"reward_scope": "latest_turn"}))
    context = [{"role": t.role, "content": t.content} for t in turns[:3]]
    assert result.metadata["raw_reward"] == pytest.approx(
        trainer.score(context, turns[3].content)
    )


def test_reward_function_round_trips_through_disk(
    tokenizer: Any, tmp_path: Path
) -> None:
    trainer = _rm_trainer(tokenizer)
    trainer.train()
    checkpoint = trainer.save_checkpoint(tmp_path / "rm")
    loaded = load_reward_function(checkpoint)
    live = trainer.reward_function()
    for turns in CONTRACT_INPUTS.values():
        assert asyncio.run(loaded.compute_reward(turns)).score == pytest.approx(
            asyncio.run(live.compute_reward(turns)).score, abs=1e-6
        )
    resumed = _rm_trainer(tokenizer)
    resumed.load_checkpoint(checkpoint)
    assert resumed.global_step == 3
    assert _max_diff(_params(resumed.model), _params(trainer.model)) == 0.0


def test_reward_function_rejects_bad_settings_and_nonfinite_output(
    tokenizer: Any,
) -> None:
    trainer = _rm_trainer(tokenizer)
    with pytest.raises(ValueError, match="temperature"):
        PairwiseRewardFunction(trainer.model, tokenizer, temperature=0.0)
    reward = trainer.reward_function()
    with torch.no_grad():
        trainer.model.head.bias.fill_(math.nan)
    with pytest.raises(ValueError, match="non-finite"):
        asyncio.run(reward.compute_reward(CONTRACT_INPUTS["dicts"]))


def test_train_reward_model_api(tokenizer: Any, tmp_path: Path) -> None:
    result = train_reward_model(
        make_model(0, n_embd=16).transformer,
        _pairs(TRAIN),
        tokenizer=tokenizer,
        eval_data=_pairs(HELD),
        config=RewardModelConfig(device="cpu", max_steps=1, output_dir=str(tmp_path)),
    )
    assert result.checkpoint is not None
    assert (result.checkpoint / "reward_model" / "reward_model.json").exists()
    assert (tmp_path / "reward_model_metrics.json").exists()
    assert isinstance(result.reward_function(), PairwiseRewardFunction)
