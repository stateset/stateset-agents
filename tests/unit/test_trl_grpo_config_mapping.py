"""How ``TRLGRPOTrainerWrapper`` maps StateSet settings onto TRL's GRPOConfig."""

from __future__ import annotations

import dataclasses
import os
import subprocess
import sys
from pathlib import Path
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


# The child cold-imports torch/transformers/trl: seconds on Linux, but it has
# exceeded 30 s on loaded Windows runners (and locally under heavy load).
@pytest.mark.timeout(150)
def test_config_preview_survives_unavailable_optional_trainer(tmp_path):
    script = """
import builtins
real_import = builtins.__import__
def without_trainer(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "trl" and "GRPOTrainer" in fromlist:
        raise RuntimeError("optional trainer backend unavailable")
    return real_import(name, globals, locals, fromlist, level)
builtins.__import__ = without_trainer
from stateset_agents.training import trl_grpo_trainer as module
from stateset_agents.training.trl_grpo_config import TRLGRPOConfig
assert module.GRPOConfig is not None
assert module.TRLGRPOTrainer is None and module.TRL_GRPO_AVAILABLE is False
wrapper = object.__new__(module.TRLGRPOTrainerWrapper)
wrapper.config = TRLGRPOConfig(output_dir="preview", bf16=False, fp16=False, report_to="none")
assert wrapper._create_grpo_config().output_dir == "preview"
try:
    module._require_trl_grpo()
except ImportError:
    pass
else:
    raise AssertionError("training must still reject an unavailable backend")
"""
    subprocess.run(
        [sys.executable, "-c", script],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        check=True,
        timeout=120,
    )
