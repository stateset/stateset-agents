"""Checkpoint loads must reject changes to the training setup before mutation."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from stateset_agents.core.errors import ConfigurationError  # noqa: E402
from stateset_agents.training.distillation import (  # noqa: E402
    DistillationConfig,
    DistillationDataset,
    DistillationTrainer,
)
from stateset_agents.training.preference import (  # noqa: E402
    OnlineDPOConfig,
    OnlineDPOTrainer,
    PreferenceConfig,
    PreferenceDataset,
    PreferenceTrainer,
    RewardModelConfig,
    RewardModelTrainer,
)
from tests.learning._toy import DigitReward, make_model, make_tokenizer  # noqa: E402

_METADATA = {
    "preference": "preference_checkpoint.json",
    "online": "online_dpo_checkpoint.json",
    "distillation": "distillation_checkpoint.json",
    "reward_model": "reward_model_checkpoint.json",
}


def _trainer(
    kind: str,
    *,
    prompts: list[Any] | None = None,
    tokenizer: Any = None,
    record_updates: dict[str, Any] | None = None,
    eval_prompts: list[str] | None = None,
    **overrides: Any,
) -> Any:
    settings = {
        "learning_rate": 1e-2,
        "batch_size": 2,
        "max_steps": 1,
        "device": "cpu",
        "max_length": 48,
        "max_prompt_length": None,
        "lr_scheduler": "linear",
    }
    settings.update(overrides)
    tokenizer = make_tokenizer() if tokenizer is None else tokenizer
    model = make_model(0, n_embd=16)
    prompts = ["hello", "count", "abc def"] if prompts is None else prompts
    updates = record_updates or {}
    if kind == "online":
        config = OnlineDPOConfig(**{"rounds": 1, "max_new_tokens": 4, **settings})
        return OnlineDPOTrainer(
            model, tokenizer, DigitReward(), prompts, config, eval_prompts=eval_prompts
        )
    if kind == "distillation":
        config = DistillationConfig(**{"lmbda": 0.5, "max_new_tokens": 4, **settings})
        data = DistillationDataset.from_records(
            {"prompt": prompt, "completion": "77ab", **updates} for prompt in prompts
        )
        evaluation = (
            DistillationDataset.from_records(
                {"prompt": prompt, "completion": "77ab"} for prompt in eval_prompts
            )
            if eval_prompts is not None
            else None
        )
        return DistillationTrainer(
            model,
            tokenizer,
            config,
            data,
            evaluation,
            teacher=make_model(50, n_embd=16),
        )
    pairs = PreferenceDataset.from_records(
        {"prompt": prompt, "chosen": "77", "rejected": "abc", **updates}
        for prompt in prompts
    )
    evaluation = (
        PreferenceDataset.from_records(
            {"prompt": prompt, "chosen": "77", "rejected": "abc"}
            for prompt in eval_prompts
        )
        if eval_prompts is not None
        else None
    )
    if kind == "reward_model":
        return RewardModelTrainer(
            model.transformer,
            tokenizer,
            RewardModelConfig(**settings),
            pairs,
            evaluation,
        )
    return PreferenceTrainer(
        model, tokenizer, PreferenceConfig(**settings), pairs, evaluation
    )


@pytest.fixture(scope="module", params=tuple(_METADATA))
def checkpoint(request: Any, tmp_path_factory: Any) -> tuple[str, Path]:
    trainer = _trainer(request.param)
    trainer.train()
    path = trainer.save_checkpoint(tmp_path_factory.mktemp("resume-contract"))
    return request.param, path


def _assert_rejected_without_mutation(trainer: Any, path: Path, match: str) -> None:
    weights = [p.detach().clone() for p in trainer.model.parameters()]
    history = copy.deepcopy(trainer.history)
    if hasattr(trainer, "optimizer"):
        optimizer = copy.deepcopy(trainer.optimizer.state_dict())
        scheduler = copy.deepcopy(trainer.lr_scheduler.state_dict())
        progress = trainer._progress()
    else:
        progress = trainer.round
    rng = torch.get_rng_state().clone()

    with pytest.raises(ConfigurationError, match=match):
        trainer.load_checkpoint(path)

    assert all(
        torch.equal(before, after)
        for before, after in zip(weights, trainer.model.parameters(), strict=True)
    )
    assert trainer.history == history
    assert torch.equal(torch.get_rng_state(), rng)
    if hasattr(trainer, "optimizer"):
        assert trainer.optimizer.state_dict() == optimizer
        assert trainer.lr_scheduler.state_dict() == scheduler
        assert trainer._progress() == progress
    else:
        assert trainer.round == progress


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 17),
        ("batch_size", 1),
        ("gradient_accumulation_steps", 2),
        ("learning_rate", 0.002),
        ("lr_scheduler", "constant"),
        ("max_length", 40),
        ("append_eos", False),
    ],
)
def test_resume_rejects_changed_training_settings(
    checkpoint: tuple[str, Path], field: str, value: Any
) -> None:
    kind, path = checkpoint
    _assert_rejected_without_mutation(_trainer(kind, **{field: value}), path, field)


def test_resume_rejects_changed_objective(checkpoint: tuple[str, Path]) -> None:
    kind, path = checkpoint
    field, value = {
        "preference": ("beta", 0.2),
        "online": ("reference_update", "previous_round"),
        "distillation": ("on_policy_loss", "sampled"),
        "reward_model": ("center_rewards_coefficient", 0.1),
    }[kind]
    _assert_rejected_without_mutation(_trainer(kind, **{field: value}), path, field)


@pytest.mark.parametrize("field", ["max_steps", "num_epochs"])
def test_online_resume_rejects_changed_per_round_budget(
    tmp_path: Path, field: str
) -> None:
    path = _trainer("online").save_checkpoint(tmp_path)
    _assert_rejected_without_mutation(_trainer("online", **{field: 2}), path, field)


@pytest.mark.parametrize("damage", ["not_object", "missing", "invalid", "unknown"])
def test_resume_rejects_invalid_saved_configuration_before_loading_weights(
    checkpoint: tuple[str, Path], tmp_path: Path, damage: str
) -> None:
    kind, path = checkpoint
    meta = json.loads((path / _METADATA[kind]).read_text())
    if damage == "not_object":
        meta["config"] = []
        match = "config"
    elif damage == "missing":
        del meta["config"]["seed"]
        match = "seed"
    elif damage == "invalid":
        # True compares equal to the valid integer 1, but is not a step count.
        meta["config"]["gradient_accumulation_steps"] = True
        match = "gradient_accumulation_steps"
    else:
        meta["config"]["future_objective"] = "unknown"
        match = "future_objective"
    (tmp_path / _METADATA[kind]).write_text(json.dumps(meta))
    # No binary files: configuration must be checked before loading any state.
    _assert_rejected_without_mutation(_trainer(kind), tmp_path, match)


def test_resume_allows_operational_changes_and_larger_run_budget(
    checkpoint: tuple[str, Path], tmp_path: Path
) -> None:
    kind, path = checkpoint
    overrides = {
        "device": "cpu:0",
        "output_dir": str(tmp_path / "continued"),
        "logging_steps": 3,
        "eval_steps": 2,
        "save_steps": 2,
        **({"rounds": 2} if kind == "online" else {"max_steps": 2, "num_epochs": 2}),
    }
    resumed = _trainer(kind, **overrides)
    resumed.load_checkpoint(path)
    if kind == "online":
        assert resumed.round == 1
    else:
        assert resumed.global_step == 1
        # Extending the loop budget must retain the saved scheduler horizon.
        assert resumed.lr_scheduler.lr_lambdas[0].total == 1
    resumed.train()
    assert (resumed.round if kind == "online" else resumed.global_step) == 2


def test_reward_model_resume_preserves_training_trajectory(tmp_path: Path) -> None:
    straight = _trainer("reward_model", max_steps=4)
    straight.train()
    interrupted = _trainer("reward_model", max_steps=4)
    interrupted.config.max_steps = 2
    interrupted.train()
    path = interrupted.save_checkpoint(tmp_path)
    resumed = _trainer("reward_model", max_steps=4)
    resumed.train(resume_from=path)
    assert resumed.global_step == straight.global_step == 4
    assert resumed.history[-1] == straight.history[-1]
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            straight.model.parameters(), resumed.model.parameters(), strict=True
        )
    )


@pytest.mark.parametrize(
    "prompts",
    [
        ["count", "hello", "abc def"],  # Same rows, different sampler indices.
        ["hello", "count", "xyz"],
        ["hello", "count"],
        ["hello", "count", "abc def", "hello"],
    ],
)
def test_resume_rejects_changed_training_data(
    checkpoint: tuple[str, Path], prompts: list[str]
) -> None:
    kind, path = checkpoint
    _assert_rejected_without_mutation(
        _trainer(kind, prompts=prompts), path, "train_data"
    )


def test_resume_rejects_changed_evaluation_data(checkpoint: tuple[str, Path]) -> None:
    kind, path = checkpoint
    _assert_rejected_without_mutation(
        _trainer(kind, eval_prompts=["xyz"]), path, "eval_data"
    )


@pytest.mark.parametrize("change", ["vocabulary", "padding", "decoder", "cleanup"])
def test_resume_rejects_changed_tokenizer(
    checkpoint: tuple[str, Path], change: str
) -> None:
    from tokenizers.decoders import WordPiece
    from tokenizers.models import WordLevel

    kind, path = checkpoint
    tokenizer = make_tokenizer()
    if change == "vocabulary":
        vocab = tokenizer.get_vocab()
        # Neither symbol appears in the encoded data: hashing rows alone misses this.
        vocab["!"], vocab["?"] = vocab["?"], vocab["!"]
        tokenizer.backend_tokenizer.model = WordLevel(vocab, unk_token="<unk>")
    elif change == "padding":
        tokenizer.pad_token = "X"
    elif change == "decoder":
        tokenizer.backend_tokenizer.decoder = WordPiece(prefix="##", cleanup=False)
    else:
        tokenizer.clean_up_tokenization_spaces = (
            not tokenizer.clean_up_tokenization_spaces
        )
    _assert_rejected_without_mutation(
        _trainer(kind, tokenizer=tokenizer), path, "tokenizer"
    )


@pytest.mark.parametrize("damage", ["missing", "not_object", "unknown", "digest"])
def test_v2_resume_requires_valid_input_fingerprints(
    checkpoint: tuple[str, Path], tmp_path: Path, damage: str
) -> None:
    kind, path = checkpoint
    meta = json.loads((path / _METADATA[kind]).read_text())
    assert meta["schema"].endswith("-v2")
    if damage == "missing":
        del meta["inputs"]
    elif damage == "not_object":
        meta["inputs"] = []
    elif damage == "unknown":
        meta["inputs"]["new_kind_of_input"] = "0" * 64
    else:
        meta["inputs"]["train_data"] = "not-a-digest"
    (tmp_path / _METADATA[kind]).write_text(json.dumps(meta))
    _assert_rejected_without_mutation(_trainer(kind), tmp_path, "input")


def test_legacy_resume_warns_that_input_identity_is_unverified(
    checkpoint: tuple[str, Path], tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    kind, original = checkpoint
    path = Path(shutil.copytree(original, tmp_path / "legacy"))
    meta_path = path / _METADATA[kind]
    meta = json.loads(meta_path.read_text())
    meta["schema"] = meta["schema"].removesuffix("-v2") + "-v1"
    del meta["inputs"]
    meta_path.write_text(json.dumps(meta))
    trainer = _trainer(kind)
    trainer.load_checkpoint(path)
    assert "legacy checkpoint without input fingerprints" in caplog.text
    assert (trainer.round if kind == "online" else trainer.global_step) == 1


def test_resume_accepts_relocated_tokenizer_and_transient_backend_settings(
    checkpoint: tuple[str, Path], tmp_path: Path
) -> None:
    from transformers import AutoTokenizer

    kind, path = checkpoint
    tokenizer_dir = path / ("reward_model" if kind == "reward_model" else "model")
    relocated = shutil.copytree(tokenizer_dir, tmp_path / "relocated")
    tokenizer = AutoTokenizer.from_pretrained(relocated, local_files_only=True)
    trainer = _trainer(kind, tokenizer=tokenizer)
    # Encoding calls modify these; they are not the trainer's collation policy.
    tokenizer(["hello", "ab"], padding=True, truncation=True, max_length=4)
    trainer.load_checkpoint(path)
    assert (trainer.round if kind == "online" else trainer.global_step) == 1


@pytest.mark.parametrize(
    ("kind", "updates"),
    [
        ("preference", {"weight": 2.0}),
        ("preference", {"chosen": "88"}),
        ("reward_model", {"margin": 0.5}),
        ("reward_model", {"weight": 2.0}),
        ("distillation", {"completion": "88ab"}),
        ("distillation", {"weight": 2.0}),
    ],
)
def test_resume_rejects_changed_training_targets(
    tmp_path: Path, kind: str, updates: dict[str, Any]
) -> None:
    path = _trainer(kind).save_checkpoint(tmp_path)
    _assert_rejected_without_mutation(
        _trainer(kind, record_updates=updates), path, "train_data"
    )


def test_kto_resume_rejects_changed_labels(tmp_path: Path) -> None:
    def make(labels: list[bool]) -> Any:
        data = PreferenceDataset.from_records(
            {"prompt": prompt, "completion": "77", "label": label}
            for prompt, label in zip(["hello", "count", "abc def"], labels, strict=True)
        )
        return PreferenceTrainer(
            make_model(0, n_embd=16),
            make_tokenizer(),
            PreferenceConfig(method="kto", device="cpu"),
            data,
        )

    path = make([True, False, True]).save_checkpoint(tmp_path)
    _assert_rejected_without_mutation(make([False, False, True]), path, "train_data")


def test_online_resume_rejects_changed_reward_scenarios(tmp_path: Path) -> None:
    source = _trainer(
        "online", prompts=[{"prompt": "hello", "scenario": {"ground_truth": "7"}}]
    )
    path = source.save_checkpoint(tmp_path)
    changed = _trainer(
        "online", prompts=[{"prompt": "hello", "scenario": {"ground_truth": "8"}}]
    )
    _assert_rejected_without_mutation(changed, path, "train_data")


def test_online_checkpoint_rejects_nonserializable_scenario_before_writing(
    tmp_path: Path,
) -> None:
    trainer = _trainer(
        "online", prompts=[{"prompt": "hello", "scenario": {"callback": object()}}]
    )
    with pytest.raises(ConfigurationError, match="scenarios.*finite JSON"):
        trainer.save_checkpoint(tmp_path / "checkpoint")
    assert not (tmp_path / "checkpoint").exists()


@pytest.mark.parametrize("kind", ["online", "distillation"])
def test_resume_rejects_changed_stop_tokens(tmp_path: Path, kind: str) -> None:
    path = _trainer(kind).save_checkpoint(tmp_path)
    changed = _trainer(kind)
    changed._stop_ids.append(changed.tokenizer.convert_tokens_to_ids("7"))
    _assert_rejected_without_mutation(changed, path, "sampling")


@pytest.mark.parametrize("kind", ["preference", "reward_model", "distillation"])
def test_resume_ignores_annotation_only_changes(tmp_path: Path, kind: str) -> None:
    path = _trainer(kind).save_checkpoint(tmp_path)
    _trainer(
        kind, record_updates={"metadata": {"annotation": "reviewed"}}
    ).load_checkpoint(path)
