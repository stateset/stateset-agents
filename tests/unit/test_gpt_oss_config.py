"""Unit tests for the gpt-oss starter configuration."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from stateset_agents.training.config import get_config_for_task
from stateset_agents.training.gpt_oss_starter import (
    GPT_OSS_BASE_MODEL,
    GPT_OSS_LORA_TARGET_MODULES,
    GPT_OSS_STARTER_PROFILE_CHOICES,
    GPT_OSS_STARTER_PROFILE_DESCRIPTIONS,
    GptOssConfig,
    create_gpt_oss_agent_config,
    create_gpt_oss_preview,
    describe_gpt_oss_starter_profiles,
    get_gpt_oss_config,
    get_gpt_oss_gspo_config,
    get_gpt_oss_profile_description,
    get_gpt_oss_profile_overrides,
    load_gpt_oss_config_file,
    summarize_gpt_oss_config,
    validate_gpt_oss_config,
    write_gpt_oss_config_file,
)


class TestGptOssConfig:
    """Test suite for the gpt-oss helper config."""

    def test_default_config_creation(self):
        config = GptOssConfig()

        assert config.model_name == GPT_OSS_BASE_MODEL
        assert config.starter_profile == "balanced"
        assert config.use_lora is True
        assert config.use_wandb is False
        assert config.report_to == "none"
        assert config.trust_remote_code is True
        assert config.attn_implementation == "sdpa"

    def test_balanced_profile_defaults(self):
        config = get_gpt_oss_config()

        assert config.starter_profile == "balanced"
        assert config.use_4bit is True
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 16

    def test_memory_profile_defaults(self):
        config = get_gpt_oss_config(starter_profile="memory")

        assert config.starter_profile == "memory"
        assert config.use_4bit is True
        assert config.num_generations == 2
        assert config.max_prompt_length == 2048

    def test_quality_profile_defaults(self):
        config = get_gpt_oss_config(starter_profile="quality")
        profile = get_gpt_oss_profile_overrides("quality")

        assert config.starter_profile == "quality"
        assert config.max_prompt_length == profile["max_prompt_length"]
        assert config.max_completion_length == profile["max_completion_length"]
        assert config.num_outer_iterations == profile["num_outer_iterations"]
        assert config.num_generations == profile["num_generations"]

    def test_profile_description_lookup(self):
        description = get_gpt_oss_profile_description("memory")

        assert description == GPT_OSS_STARTER_PROFILE_DESCRIPTIONS["memory"]
        assert "Lower-memory" in description

    def test_profile_catalog_lists_all_profiles(self):
        payload = describe_gpt_oss_starter_profiles(task="sales")

        assert payload["task"] == "sales"
        assert payload["default_profile"] == "balanced"
        assert set(payload["profiles"]) == set(GPT_OSS_STARTER_PROFILE_CHOICES)
        assert payload["profiles"]["memory"]["summary"]["quantization_mode"] == "4bit"

    def test_explicit_overrides_beat_profile_defaults(self):
        config = get_gpt_oss_config(
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
        config = get_gpt_oss_config(task="technical_support", num_outer_iterations=20)

        assert config.task == "technical_support"
        assert config.num_outer_iterations == 20
        assert "technical support" in config.system_prompt.lower()

    def test_config_with_lora_disabled(self):
        config = get_gpt_oss_config(use_lora=False)

        assert config.use_lora is False
        assert config.lora_r is None
        assert config.lora_alpha is None

    def test_quantization_flags(self):
        config = get_gpt_oss_config(use_4bit=True, use_8bit=True)

        assert config.use_4bit is True
        assert config.use_8bit is False

    def test_agent_config_creation(self):
        config = get_gpt_oss_config()
        agent_config = create_gpt_oss_agent_config(config)

        assert agent_config.model_name == GPT_OSS_BASE_MODEL
        assert agent_config.trust_remote_code is True
        assert agent_config.attn_implementation == "sdpa"
        assert agent_config.max_new_tokens == 1024

    def test_config_summary(self):
        config = get_gpt_oss_config(starter_profile="memory")
        summary = summarize_gpt_oss_config(config)

        assert summary["starter_profile"] == "memory"
        assert summary["quantization_mode"] == "4bit"
        assert summary["effective_batch_size"] == 24
        assert summary["uses_quantization"] is True

    def test_gspo_config_generation(self):
        base_config = get_config_for_task(
            "customer_service", model_name=GPT_OSS_BASE_MODEL
        )
        config = get_gpt_oss_config(task="customer_service")

        with patch(
            "stateset_agents.training.gpt_oss_starter.get_config_for_task",
            return_value=base_config,
        ):
            gspo_config = get_gpt_oss_gspo_config(config)

        assert gspo_config.model_name == GPT_OSS_BASE_MODEL
        assert gspo_config.use_lora is True
        assert gspo_config.num_generations == 4
        assert gspo_config.max_prompt_length == 4096
        assert gspo_config.max_completion_length == 1024
        assert gspo_config.lora_target_modules == GPT_OSS_LORA_TARGET_MODULES

    def test_validation_warnings(self):
        config = get_gpt_oss_config(
            use_lora=False,
            learning_rate=2e-5,
            per_device_train_batch_size=4,
        )
        warnings = validate_gpt_oss_config(config)

        assert any("learning rate" in warning.lower() for warning in warnings)
        assert any("oom" in warning.lower() for warning in warnings)
        assert any("lora" in warning.lower() for warning in warnings)

    def test_validation_warns_on_unknown_starter_profile(self):
        config = GptOssConfig(starter_profile="custom-profile")
        warnings = validate_gpt_oss_config(config)

        assert any("starter_profile" in warning for warning in warnings)

    def test_validation_warns_on_non_gpt_oss_model(self):
        config = GptOssConfig(model_name="meta-llama/Llama-3.1-8B-Instruct")
        warnings = validate_gpt_oss_config(config)

        assert any("gpt-oss checkpoint" in warning.lower() for warning in warnings)

    def test_validation_warns_on_120b_variant(self):
        config = GptOssConfig(model_name="openai/gpt-oss-120b")
        warnings = validate_gpt_oss_config(config)

        assert any("multi-gpu" in warning.lower() for warning in warnings)

    def test_json_config_file_roundtrip(self, tmp_path):
        config = get_gpt_oss_config(
            task="technical_support",
            use_lora=False,
            learning_rate=1e-5,
            output_dir="./outputs/gpt_oss_roundtrip",
        )
        config_path = write_gpt_oss_config_file(config, tmp_path / "gpt_oss.json")
        loaded = load_gpt_oss_config_file(config_path)

        assert loaded.task == "technical_support"
        assert loaded.use_lora is False
        assert loaded.learning_rate == 1e-5
        assert loaded.output_dir == "./outputs/gpt_oss_roundtrip"

    def test_preview_payload_can_be_loaded_as_config(self, tmp_path):
        config = get_gpt_oss_config(
            task="sales", output_dir="./outputs/gpt_oss_preview"
        )
        preview_path = write_gpt_oss_config_file(
            config,
            tmp_path / "gpt_oss_preview.json",
            include_preview=True,
        )
        loaded = load_gpt_oss_config_file(preview_path)
        preview = create_gpt_oss_preview(loaded)

        assert loaded.task == "sales"
        assert preview["config"]["output_dir"] == "./outputs/gpt_oss_preview"


class TestGptOssStarterScript:
    """Test the dedicated starter script surface."""

    def test_script_is_a_forwarder(self):
        """finetune_gpt_oss_gspo.py is a thin forwarder onto
        examples/finetune_gspo.py --model gpt-oss. It prints a notice
        to stderr and exits 0 under --dry-run."""
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [sys.executable, "examples/finetune_gpt_oss_gspo.py", "--dry-run"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        assert result.returncode == 0, result.stderr
        assert "forwarder" in result.stderr.lower()
        assert "gpt-oss" in result.stderr

    def test_dry_run_preview_via_starter_profile(self):
        """The starter's own config resolution is reachable through the
        unified driver's --starter-profile flag."""
        from examples import finetune_gspo

        exit_code = finetune_gspo.main(
            ["--model", "gpt-oss", "--starter-profile", "balanced", "--dry-run"]
        )
        assert exit_code == 0

    def test_cli_dry_run_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_gpt_oss_gspo.py",
                "--starter-profile",
                "balanced",
                "--dry-run",
            ],
            cwd=repo_root,
            text=True,
        )
        payload = json.loads(output)

        assert payload["config"]["model_name"] == GPT_OSS_BASE_MODEL
        assert payload["config"]["starter_profile"] == "balanced"
        assert payload["agent_config"]["attn_implementation"] == "sdpa"
        assert payload["gspo_overrides"]["use_lora"] is True

    def test_cli_memory_profile_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_gpt_oss_gspo.py",
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

    def test_cli_list_profiles_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_gpt_oss_gspo.py",
                "--task",
                "sales",
                "--list-profiles",
            ],
            cwd=repo_root,
            text=True,
        )
        payload = json.loads(output)

        assert payload["task"] == "sales"
        assert payload["profiles"]["quality"]["summary"]["num_generations"] == 6
        assert "memory" in payload["profiles"]
