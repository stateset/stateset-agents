"""Unit tests for the DeepSeek V4 Flash starter configuration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from stateset_agents.training.config import get_config_for_task
from stateset_agents.training.deepseek_v4_starter import (
    DEEPSEEK_V4_BASE_MODEL,
    DEEPSEEK_V4_LORA_TARGET_MODULES,
    DEEPSEEK_V4_STARTER_PROFILE_CHOICES,
    DEEPSEEK_V4_STARTER_PROFILE_DESCRIPTIONS,
    DeepseekV4Config,
    create_deepseek_v4_agent_config,
    create_deepseek_v4_preview,
    describe_deepseek_v4_starter_profiles,
    get_deepseek_v4_config,
    get_deepseek_v4_gspo_config,
    get_deepseek_v4_profile_description,
    get_deepseek_v4_profile_overrides,
    load_deepseek_v4_config_file,
    summarize_deepseek_v4_config,
    validate_deepseek_v4_config,
    write_deepseek_v4_config_file,
)
from tests.unit.forwarder_asserts import assert_forwards_to_driver


class TestDeepseekV4Config:
    """Test suite for the DeepSeek V4 Flash helper config."""

    def test_default_config_creation(self):
        config = DeepseekV4Config()

        assert config.model_name == DEEPSEEK_V4_BASE_MODEL
        assert config.starter_profile == "balanced"
        assert config.use_lora is True
        assert config.use_wandb is False
        assert config.report_to == "none"
        assert config.use_4bit is True
        assert config.use_vllm is True
        assert config.trust_remote_code is True
        assert config.attn_implementation == "sdpa"

    def test_balanced_profile_defaults(self):
        config = get_deepseek_v4_config()

        assert config.starter_profile == "balanced"
        assert config.use_4bit is True
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 16

    def test_memory_profile_defaults(self):
        config = get_deepseek_v4_config(starter_profile="memory")

        assert config.starter_profile == "memory"
        assert config.use_4bit is True
        assert config.gradient_accumulation_steps == 32
        assert config.num_generations == 2
        assert config.max_prompt_length == 4096

    def test_quality_profile_defaults(self):
        config = get_deepseek_v4_config(starter_profile="quality")
        profile = get_deepseek_v4_profile_overrides("quality")

        assert config.starter_profile == "quality"
        assert config.max_prompt_length == profile["max_prompt_length"]
        assert config.max_completion_length == profile["max_completion_length"]
        assert config.num_outer_iterations == profile["num_outer_iterations"]
        assert config.num_generations == profile["num_generations"]

    def test_profile_description_lookup(self):
        description = get_deepseek_v4_profile_description("memory")

        assert description == DEEPSEEK_V4_STARTER_PROFILE_DESCRIPTIONS["memory"]
        assert "Lower-memory" in description

    def test_profile_catalog_lists_all_profiles(self):
        payload = describe_deepseek_v4_starter_profiles(task="sales")

        assert payload["task"] == "sales"
        assert payload["default_profile"] == "balanced"
        assert set(payload["profiles"]) == set(DEEPSEEK_V4_STARTER_PROFILE_CHOICES)
        assert payload["profiles"]["memory"]["summary"]["quantization_mode"] == "4bit"

    def test_explicit_overrides_beat_profile_defaults(self):
        config = get_deepseek_v4_config(
            starter_profile="memory",
            use_4bit=False,
            per_device_train_batch_size=2,
            num_outer_iterations=9,
        )

        assert config.starter_profile == "memory"
        assert config.use_4bit is False
        assert config.per_device_train_batch_size == 2
        assert config.num_outer_iterations == 9

    def test_config_with_custom_task(self):
        config = get_deepseek_v4_config(
            task="technical_support", num_outer_iterations=20
        )

        assert config.task == "technical_support"
        assert config.num_outer_iterations == 20
        assert "technical support" in config.system_prompt.lower()

    def test_config_with_lora_disabled(self):
        config = get_deepseek_v4_config(use_lora=False)

        assert config.use_lora is False
        assert config.lora_r is None
        assert config.lora_alpha is None

    def test_quantization_flags(self):
        config = get_deepseek_v4_config(use_4bit=True, use_8bit=True)

        assert config.use_4bit is True
        assert config.use_8bit is False

    def test_agent_config_creation(self):
        config = get_deepseek_v4_config()
        agent_config = create_deepseek_v4_agent_config(config)

        assert agent_config.model_name == DEEPSEEK_V4_BASE_MODEL
        assert agent_config.trust_remote_code is True
        assert agent_config.attn_implementation == "sdpa"
        assert agent_config.max_new_tokens == 1536

    def test_config_summary(self):
        config = get_deepseek_v4_config(starter_profile="memory")
        summary = summarize_deepseek_v4_config(config)

        assert summary["starter_profile"] == "memory"
        assert summary["quantization_mode"] == "4bit"
        assert summary["effective_batch_size"] == 32
        assert summary["uses_quantization"] is True

    def test_gspo_config_generation(self):
        base_config = get_config_for_task(
            "customer_service", model_name=DEEPSEEK_V4_BASE_MODEL
        )
        config = get_deepseek_v4_config(task="customer_service")

        with patch(
            "stateset_agents.training.deepseek_v4_starter.get_config_for_task",
            return_value=base_config,
        ):
            gspo_config = get_deepseek_v4_gspo_config(config)

        assert gspo_config.model_name == DEEPSEEK_V4_BASE_MODEL
        assert gspo_config.use_lora is True
        assert gspo_config.num_generations == 4
        assert gspo_config.max_prompt_length == 8192
        assert gspo_config.max_completion_length == 1536
        assert gspo_config.lora_target_modules == DEEPSEEK_V4_LORA_TARGET_MODULES

    def test_validation_warnings(self):
        config = get_deepseek_v4_config(
            use_lora=False,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
        )
        warnings = validate_deepseek_v4_config(config)

        assert any("learning rate" in warning.lower() for warning in warnings)
        assert any("oom" in warning.lower() for warning in warnings)
        assert any("lora" in warning.lower() for warning in warnings)

    def test_validation_warns_on_unknown_starter_profile(self):
        config = DeepseekV4Config(starter_profile="custom-profile")
        warnings = validate_deepseek_v4_config(config)

        assert any("starter_profile" in warning for warning in warnings)

    def test_validation_warns_on_non_deepseek_model(self):
        config = DeepseekV4Config(model_name="meta-llama/Llama-3.1-8B-Instruct")
        warnings = validate_deepseek_v4_config(config)

        assert any("deepseek checkpoint" in warning.lower() for warning in warnings)

    def test_validation_warns_on_inference_only_repos(self):
        config = DeepseekV4Config(model_name="deepseek-ai/DeepSeek-V4-Flash-FP8")
        warnings = validate_deepseek_v4_config(config)

        assert any("inference-only" in warning.lower() for warning in warnings)

    def test_validation_warns_when_quantization_disabled(self):
        config = get_deepseek_v4_config(use_4bit=False)
        warnings = validate_deepseek_v4_config(config)

        assert any("4-bit" in warning.lower() for warning in warnings)

    def test_json_config_file_roundtrip(self, tmp_path):
        config = get_deepseek_v4_config(
            task="technical_support",
            use_lora=False,
            learning_rate=1e-5,
            output_dir="./outputs/deepseek_v4_roundtrip",
        )
        config_path = write_deepseek_v4_config_file(
            config, tmp_path / "deepseek_v4.json"
        )
        loaded = load_deepseek_v4_config_file(config_path)

        assert loaded.task == "technical_support"
        assert loaded.use_lora is False
        assert loaded.learning_rate == 1e-5
        assert loaded.output_dir == "./outputs/deepseek_v4_roundtrip"

    def test_preview_payload_can_be_loaded_as_config(self, tmp_path):
        config = get_deepseek_v4_config(
            task="sales", output_dir="./outputs/deepseek_v4_preview"
        )
        preview_path = write_deepseek_v4_config_file(
            config,
            tmp_path / "deepseek_v4_preview.json",
            include_preview=True,
        )
        loaded = load_deepseek_v4_config_file(preview_path)
        preview = create_deepseek_v4_preview(loaded)

        assert loaded.task == "sales"
        assert preview["config"]["output_dir"] == "./outputs/deepseek_v4_preview"


class TestDeepseekV4StarterScript:
    """Test the dedicated starter script surface."""

    def test_script_is_a_forwarder(self):
        """finetune_deepseek_v4_gspo.py is a thin forwarder onto
        examples/finetune_gspo.py --model deepseek-v4.

        Checked structurally (AST) rather than by spawning an interpreter --
        see tests/unit/forwarder_asserts.py for why, and
        test_example_model_presets.py for the one real subprocess that still
        proves the driver runs end to end."""
        assert_forwards_to_driver(
            "finetune_deepseek_v4_gspo.py",
            model="deepseek-v4",
        )

    def test_dry_run_preview_via_starter_profile(self):
        """The starter's own config resolution is reachable through the
        unified driver's --starter-profile flag."""
        from examples import finetune_gspo

        exit_code = finetune_gspo.main(
            ["--model", "deepseek-v4", "--starter-profile", "balanced", "--dry-run"]
        )
        assert exit_code == 0

    def test_cli_dry_run_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_deepseek_v4_gspo.py",
                "--starter-profile",
                "balanced",
                "--dry-run",
            ],
            cwd=repo_root,
            text=True,
        )
        payload = json.loads(output)

        assert payload["config"]["model_name"] == DEEPSEEK_V4_BASE_MODEL
        assert payload["config"]["starter_profile"] == "balanced"
        assert payload["agent_config"]["attn_implementation"] == "sdpa"
        assert payload["gspo_overrides"]["use_lora"] is True

    @pytest.mark.slow
    # One real interpreter spawn per file is enough to prove the script is
    # runnable end to end; that job belongs to this file's kept
    # test_cli_dry_run_subprocess. This variant only re-checks profile values
    # the in-process config tests already assert directly, and costs ~9 s of
    # wall time to do it. Run it with `-m slow`.
    def test_cli_memory_profile_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_deepseek_v4_gspo.py",
                "--starter-profile",
                "memory",
                "--dry-run",
            ],
            cwd=repo_root,
            text=True,
        )
        payload = json.loads(output)

        assert payload["config"]["starter_profile"] == "memory"
        assert payload["summary"]["quantization_mode"] == "4bit"
        assert payload["gspo_overrides"]["use_4bit"] is True
        assert payload["gspo_overrides"]["num_generations"] == 2

    @pytest.mark.slow
    # One real interpreter spawn per file is enough to prove the script is
    # runnable end to end; that job belongs to this file's kept
    # test_cli_dry_run_subprocess. This variant only re-checks profile values
    # the in-process config tests already assert directly, and costs ~9 s of
    # wall time to do it. Run it with `-m slow`.
    def test_cli_list_profiles_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_deepseek_v4_gspo.py",
                "--task",
                "sales",
                "--list-profiles",
            ],
            cwd=repo_root,
            text=True,
        )
        payload = json.loads(output)

        assert payload["task"] == "sales"
        assert payload["profiles"]["quality"]["summary"]["num_generations"] == 8
        assert "memory" in payload["profiles"]
