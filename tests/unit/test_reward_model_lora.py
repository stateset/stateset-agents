"""LoRA reward models: adapters-only training, persistence, resume."""

from __future__ import annotations

import builtins
import json
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("peft")

from stateset_agents.training.preference import (  # noqa: E402
    PreferenceDataset,
    RewardModelConfig,
    RewardModelTrainer,
    load_reward_function,
    save_reward_model,
    train_reward_model,
)
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


def _pairs(prompts: list[str]) -> PreferenceDataset:
    return PreferenceDataset.from_records(
        {
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": f"{index}{index}",
            "rejected": "ab" if index % 2 else "xyz",
        }
        for index, prompt in enumerate(prompts)
    )


TRAIN = ["hello", "count", "say a number", "abc def"]
HELD = ["xyz", "ok go"]
PROMPT = [{"role": "user", "content": "xyz"}]


def _config(**overrides: Any) -> RewardModelConfig:
    values: dict[str, Any] = {
        "learning_rate": 1e-2,
        "batch_size": 4,
        "max_steps": 3,
        "device": "cpu",
        "max_length": 48,
        "max_prompt_length": None,
        "use_lora": True,
        "lora_r": 4,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "lora_target_modules": ["c_attn"],
    }
    values.update(overrides)
    return RewardModelConfig(**values)


def _trainer(tokenizer: Any, **overrides: Any) -> RewardModelTrainer:
    return RewardModelTrainer(
        make_model(0, n_embd=16).transformer,
        tokenizer,
        _config(**overrides),
        _pairs(TRAIN),
        _pairs(HELD),
    )


def test_only_adapters_and_head_train(tokenizer: Any) -> None:
    trainer = _trainer(tokenizer)
    backbone = trainer.model.backbone
    assert hasattr(backbone, "peft_config")
    frozen = {
        name: p.detach().clone()
        for name, p in trainer.model.named_parameters()
        if not p.requires_grad
    }
    trainable = {n for n, p in trainer.model.named_parameters() if p.requires_grad}
    assert {"head.weight", "head.bias"} <= trainable
    assert all("lora_" in name for name in trainable - {"head.weight", "head.bias"})
    head = trainer.model.head.weight.detach().clone()
    trainer.train()
    for name, parameter in trainer.model.named_parameters():
        if name in frozen:
            assert torch.equal(frozen[name], parameter.detach()), name
    assert not torch.equal(head, trainer.model.head.weight.detach())


def test_adapter_save_and_load_reproduce_scores(tokenizer: Any, tmp_path: Path) -> None:
    trainer = _trainer(tokenizer)
    trainer.train()
    checkpoint = trainer.save_checkpoint(tmp_path / "ckpt")
    saved = checkpoint / "reward_model"
    meta = json.loads((saved / "reward_model.json").read_text())
    assert meta["lora"] is True and (saved / "adapter_model.safetensors").exists()
    assert not (saved / "model.safetensors").exists()  # adapter only
    with pytest.raises(ValueError, match="base_model"):
        load_reward_function(checkpoint)  # a config-built toy records no base
    reward = load_reward_function(
        checkpoint, base_model=make_model(0, n_embd=16).transformer
    )
    expected = trainer.score(PROMPT, "1234")
    assert reward.raw_score(PROMPT, "1234") == pytest.approx(expected, abs=1e-5)


def test_merged_save_is_standalone(tokenizer: Any, tmp_path: Path) -> None:
    trainer = _trainer(tokenizer)
    trainer.train()
    directory = save_reward_model(
        trainer.model, tokenizer, tmp_path / "merged", trainer.encoder, merge_lora=True
    )
    meta = json.loads((directory / "reward_model.json").read_text())
    assert meta["lora"] is False and not (directory / "adapter_config.json").exists()
    reward = load_reward_function(directory)
    assert reward.raw_score(PROMPT, "1234") == pytest.approx(
        trainer.score(PROMPT, "1234"), abs=1e-5
    )
    assert hasattr(trainer.model.backbone, "peft_config")  # the original is intact
    with pytest.raises(ValueError, match="LoRA reward models only"):
        load_reward_function(directory, base_model="anything")


def test_lora_checkpoint_resume_is_exact(tokenizer: Any, tmp_path: Path) -> None:
    straight = _trainer(tokenizer, max_steps=4, batch_size=2)
    straight.train()
    interrupted = _trainer(tokenizer, max_steps=4, batch_size=2)
    interrupted.config.max_steps = 2
    interrupted.train()
    checkpoint = interrupted.save_checkpoint(tmp_path / "ckpt")
    resumed = _trainer(tokenizer, max_steps=4, batch_size=2)
    resumed.train(resume_from=checkpoint)
    for a, b in zip(
        straight.model.parameters(), resumed.model.parameters(), strict=True
    ):
        assert torch.equal(a.detach(), b.detach())
    with pytest.raises(ValueError, match="use_lora"):
        _trainer(tokenizer, use_lora=False).load_checkpoint(checkpoint)


def test_train_reward_model_api_with_lora(tokenizer: Any, tmp_path: Path) -> None:
    result = train_reward_model(
        make_model(0, n_embd=16).transformer,
        _pairs(TRAIN),
        tokenizer=tokenizer,
        eval_data=_pairs(HELD),
        config=_config(),
        output_dir=str(tmp_path / "rm"),
    )
    assert result.summary.eval_metrics is not None
    assert (tmp_path / "rm" / "final" / "reward_model" / "adapter_config.json").exists()


def test_missing_peft_is_a_clear_import_error(
    tokenizer: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = builtins.__import__

    def no_peft(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "peft" or name.startswith("peft."):
            raise ImportError("No module named 'peft'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_peft)
    with pytest.raises(ImportError, match="stateset-agents\\[training\\]"):
        _trainer(tokenizer)


def test_lora_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="lora_r"):
        _config(lora_r=0).validate()
    with pytest.raises(ValueError, match="use_lora"):
        _config(use_lora="yes").validate()
