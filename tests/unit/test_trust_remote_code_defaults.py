"""``trust_remote_code`` is off unless a config or curated preset turns it on.

Loading a Hugging Face repo with ``trust_remote_code=True`` executes Python
shipped in that repo. The trainers, serving helpers and generic presets used to
hard-code it on, so pointing any of them at an untrusted model name ran that
repo's code. One keyword-only ``TrainingConfig.trust_remote_code`` (default
False) now drives every loader; only model families whose architecture is not
native to transformers keep it on, explicitly.
"""

from __future__ import annotations

import dataclasses
import inspect

import pytest

from stateset_agents.core.model_presets import PRESETS, ModelPreset
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.gspo_config import GSPOConfig

# Families without a native transformers implementation (they ship custom
# modelling code), so their curated presets opt in explicitly.
_REMOTE_CODE_FAMILIES = ("moonshotai/Kimi-", "zai-org/GLM-5", "meta-models/Muse-")


def test_training_config_defaults_off_and_is_keyword_only():
    field = {f.name: f for f in dataclasses.fields(TrainingConfig)}["trust_remote_code"]
    assert field.default is False and field.kw_only
    assert TrainingConfig(trust_remote_code=True).trust_remote_code is True


def test_gspo_config_defaults_off():
    assert GSPOConfig().trust_remote_code is False


def test_model_preset_dataclass_defaults_off():
    field = {f.name: f for f in dataclasses.fields(ModelPreset)}["trust_remote_code"]
    assert field.default is False


@pytest.mark.parametrize("name", sorted(PRESETS))
def test_only_non_native_preset_families_trust_remote_code(name):
    preset = PRESETS[name]
    needs_remote = preset.model_id.startswith(_REMOTE_CODE_FAMILIES)
    assert preset.trust_remote_code is needs_remote, preset.model_id


def test_sft_does_not_trust_unknown_models():
    from stateset_agents.training.sft import _trust_remote_code

    assert _trust_remote_code("someone/unvetted-model") is False


def test_sft_model_load_kwargs_do_not_default_to_remote_code():
    from stateset_agents.training.sft import model_load_kwargs

    assert model_load_kwargs().get("trust_remote_code", False) is False


def test_vllm_config_defaults_off():
    from stateset_agents.training.vllm_backend import VLLMConfig

    assert VLLMConfig().trust_remote_code is False


def test_serving_export_defaults_off():
    from stateset_agents.training.serving_artifacts import (
        export_merged_model_for_serving,
    )

    default = (
        inspect.signature(export_merged_model_for_serving)
        .parameters["trust_remote_code"]
        .default
    )
    assert default is False


@pytest.mark.parametrize("enabled", [False, True])
def test_base_model_manager_follows_the_config(enabled):
    from stateset_agents.training.base_trainer import (
        BaseModelManager,
        BaseTrainerConfig,
    )

    manager = BaseModelManager(BaseTrainerConfig(trust_remote_code=enabled))
    assert manager._build_model_kwargs()["trust_remote_code"] is enabled


def test_base_trainer_config_defaults_off():
    from stateset_agents.training.base_trainer import BaseTrainerConfig

    assert BaseTrainerConfig().trust_remote_code is False


def test_shared_runtime_does_not_assume_remote_code():
    from stateset_agents.training import trainer_runtime

    source = inspect.getsource(trainer_runtime)
    assert '"trust_remote_code", True' not in source


def test_trl_grpo_loader_reads_the_config():
    from stateset_agents.training import trl_grpo_trainer

    source = inspect.getsource(trl_grpo_trainer)
    assert "trust_remote_code=True" not in source
    assert '"trust_remote_code": True' not in source


def test_starter_default_is_off_and_curated_families_opt_in():
    from stateset_agents.training import starter_factory

    assert starter_factory._COMMON_DEFAULTS["trust_remote_code"] is False
    from stateset_agents.training.kimi_k3_starter import KimiK3Config
    from stateset_agents.training.qwen3_5_starter import Qwen35Config

    assert KimiK3Config().trust_remote_code is True
    assert Qwen35Config().trust_remote_code is False
