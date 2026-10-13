"""How ``TRLGRPOTrainerWrapper`` maps StateSet settings onto TRL's GRPOConfig."""

from __future__ import annotations

import dataclasses
from unittest.mock import MagicMock

import pytest

pytest.importorskip("torch")
pytest.importorskip("trl")

from stateset_agents.training.config import get_config_for_task  # noqa: E402
from stateset_agents.training.trl_grpo_config import TRLGRPOConfig  # noqa: E402
from stateset_agents.training.trl_grpo_trainer import (  # noqa: E402
    TRLGRPOTrainerWrapper,
)


def _wrapper(config: TRLGRPOConfig) -> TRLGRPOTrainerWrapper:
    wrapper = object.__new__(TRLGRPOTrainerWrapper)
    wrapper.config = config
    return wrapper


def _config(tmp_path, **overrides) -> TRLGRPOConfig:
    base = {
        "output_dir": str(tmp_path),
        "report_to": "none",
        "bf16": False,
        "fp16": False,
    }
    base.update(overrides)
    return TRLGRPOConfig(**base)


def test_customer_service_profile_builds_a_valid_trl_config(tmp_path):
    # num_generations=12 with one prompt per device and 4 accumulation steps:
    # TRL's default generation batch (4) is not a multiple of the group size.
    base = get_config_for_task("customer_service", model_name="gpt2")
    config = TRLGRPOConfig.from_training_config(
        base,
        num_episodes=10,
        output_dir=str(tmp_path),
        bf16=False,
        fp16=False,
        report_to="none",
    )
    grpo = _wrapper(config)._create_grpo_config()
    assert grpo.num_generations == config.num_generations
    assert grpo.generation_batch_size % grpo.num_generations == 0


def test_divisible_generation_batch_keeps_trl_default(tmp_path):
    config = _config(
        tmp_path,
        num_generations=4,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
    )
    grpo = _wrapper(config)._create_grpo_config()
    assert grpo.steps_per_generation == config.gradient_accumulation_steps
    assert grpo.generation_batch_size == 4


def test_warmup_follows_the_configured_ratio(tmp_path):
    config = _config(tmp_path, warmup_ratio=0.1, num_episodes=100)
    grpo = _wrapper(config)._create_grpo_config()
    fields = {f.name: f for f in dataclasses.fields(type(grpo))}
    if "float" in str(fields["warmup_steps"].type):
        # transformers >= 5: a fractional warmup_steps is a ratio of TRL's
        # real step count.
        assert grpo.warmup_steps == pytest.approx(0.1)
    else:
        assert grpo.warmup_ratio == pytest.approx(0.1)
        assert grpo.warmup_steps == 0


def test_train_resumes_from_configured_checkpoint(tmp_path):
    wrapper = _wrapper(_config(tmp_path, resume_from_checkpoint=str(tmp_path)))
    wrapper.trainer = MagicMock()
    wrapper.train()
    wrapper.trainer.train.assert_called_once_with(resume_from_checkpoint=str(tmp_path))


def test_train_without_checkpoint_starts_fresh(tmp_path):
    wrapper = _wrapper(_config(tmp_path))
    wrapper.trainer = MagicMock()
    wrapper.train()
    wrapper.trainer.train.assert_called_once_with(resume_from_checkpoint=None)
