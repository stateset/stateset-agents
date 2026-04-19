"""Unit tests for the GLM 5.1 starter configuration."""

from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from stateset_agents.training.config import get_config_for_task
from stateset_agents.training.glm5_1_starter import (
    GLM5_1_BASE_MODEL,
    GLM5_1_FP8_MODEL,
    GLM5_1_LORA_TARGET_MODULES,
    GLM5_1_STARTER_PROFILE_CHOICES,
    GLM5_1_STARTER_PROFILE_DESCRIPTIONS,
    Glm51Config,
    create_glm5_1_agent_config,
    create_glm5_1_preview,
    describe_glm5_1_starter_profiles,
    get_glm5_1_config,
    get_glm5_1_gspo_config,
    get_glm5_1_profile_description,
    get_glm5_1_profile_overrides,
    get_glm5_1_serving_recommendations,
    load_glm5_1_config_file,
    summarize_glm5_1_config,
    validate_glm5_1_config,
    write_glm5_1_config_file,
)


class TestGlm51Config:
    """Test suite for the GLM 5.1 helper config."""

    def test_default_config_creation(self):
        config = Glm51Config()

        assert config.model_name == GLM5_1_BASE_MODEL
        assert config.starter_profile == "balanced"
        assert config.use_lora is True
        assert config.use_4bit is True
        assert config.use_vllm is True
        assert config.use_wandb is False
        assert config.report_to == "none"
        assert config.trust_remote_code is True

    def test_memory_profile_defaults(self):
        config = get_glm5_1_config(starter_profile="memory")

        assert config.starter_profile == "memory"
        assert config.use_4bit is True
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 32
        assert config.num_generations == 2
        assert config.max_prompt_length == 4096

    def test_quality_profile_defaults(self):
        config = get_glm5_1_config(starter_profile="quality")
        profile = get_glm5_1_profile_overrides("quality")

        assert config.starter_profile == "quality"
        assert config.max_prompt_length == profile["max_prompt_length"]
        assert config.max_completion_length == profile["max_completion_length"]
        assert config.num_outer_iterations == profile["num_outer_iterations"]
        assert config.num_generations == profile["num_generations"]

    def test_profile_description_lookup(self):
        description = get_glm5_1_profile_description("memory")

        assert description == GLM5_1_STARTER_PROFILE_DESCRIPTIONS["memory"]
        assert "memory" in description.lower()

    def test_profile_catalog_lists_all_profiles(self):
        payload = describe_glm5_1_starter_profiles(task="sales")

        assert payload["task"] == "sales"
        assert payload["default_profile"] == "balanced"
        assert set(payload["profiles"]) == set(GLM5_1_STARTER_PROFILE_CHOICES)
        assert payload["profiles"]["memory"]["summary"]["quantization_mode"] == "4bit"

    def test_explicit_overrides_beat_profile_defaults(self):
        config = get_glm5_1_config(
            starter_profile="memory",
            use_4bit=False,
            num_outer_iterations=7,
        )

        assert config.starter_profile == "memory"
        assert config.use_4bit is False
        assert config.num_outer_iterations == 7

    def test_config_with_custom_task(self):
        config = get_glm5_1_config(task="technical_support", num_outer_iterations=15)

        assert config.task == "technical_support"
        assert config.num_outer_iterations == 15
        assert "technical support" in config.system_prompt.lower()

    def test_config_with_lora_disabled(self):
        config = get_glm5_1_config(use_lora=False)

        assert config.use_lora is False
        assert config.lora_r is None
        assert config.lora_alpha is None

    def test_quantization_flags(self):
        config = get_glm5_1_config(use_4bit=True, use_8bit=True)

        assert config.use_4bit is True
        assert config.use_8bit is False

    def test_agent_config_creation(self):
        config = get_glm5_1_config()
        agent_config = create_glm5_1_agent_config(config)

        assert agent_config.model_name == GLM5_1_BASE_MODEL
        assert agent_config.trust_remote_code is True
        assert agent_config.max_new_tokens == 1536

    def test_config_summary(self):
        config = get_glm5_1_config(starter_profile="memory")
        summary = summarize_glm5_1_config(config)

        assert summary["starter_profile"] == "memory"
        assert summary["quantization_mode"] == "4bit"
        assert summary["effective_batch_size"] == 32
        assert summary["uses_quantization"] is True

    def test_gspo_config_generation(self):
        base_config = get_config_for_task(
            "customer_service", model_name=GLM5_1_BASE_MODEL
        )
        config = get_glm5_1_config(task="customer_service")

        with patch(
            "stateset_agents.training.glm5_1_starter.get_config_for_task",
            return_value=base_config,
        ):
            gspo_config = get_glm5_1_gspo_config(config)

        assert gspo_config.model_name == GLM5_1_BASE_MODEL
        assert gspo_config.use_lora is True
        assert gspo_config.use_vllm is True
        assert gspo_config.use_4bit is True
        assert gspo_config.lora_target_modules == GLM5_1_LORA_TARGET_MODULES

    def test_validation_warnings(self):
        config = get_glm5_1_config(
            use_lora=False,
            use_4bit=False,
            learning_rate=5e-5,
            per_device_train_batch_size=4,
        )
        warnings = validate_glm5_1_config(config)

        assert any("learning rate" in warning.lower() for warning in warnings)
        assert any("oom" in warning.lower() for warning in warnings)
        assert any("lora" in warning.lower() for warning in warnings)
        assert any("quantization" in warning.lower() for warning in warnings)

    def test_validation_warns_on_unknown_starter_profile(self):
        config = Glm51Config(starter_profile="custom-profile")
        warnings = validate_glm5_1_config(config)

        assert any("starter_profile" in warning for warning in warnings)

    def test_serving_recommendations_bf16_default(self):
        rec = get_glm5_1_serving_recommendations()

        assert rec["tensor_parallel_size"] == 8
        assert rec["pipeline_parallel_size"] == 2
        assert rec["reasoning_parser"] == "glm45"
        assert rec["tool_call_parser"] == "glm45"
        assert rec["enable_auto_tool_choice"] is True
        assert rec["quantization"] is None

    def test_serving_recommendations_fp8(self):
        rec = get_glm5_1_serving_recommendations(use_fp8=True)

        assert rec["tensor_parallel_size"] == 8
        assert rec["pipeline_parallel_size"] == 1
        assert rec["quantization"] == "fp8"

    def test_serving_recommendations_disable_tool_choice(self):
        rec = get_glm5_1_serving_recommendations(enable_auto_tool_choice=False)

        assert rec["enable_auto_tool_choice"] is False
        assert rec["tool_call_parser"] is None

    def test_json_config_file_roundtrip(self, tmp_path):
        config = get_glm5_1_config(
            task="technical_support",
            use_lora=True,
            learning_rate=2e-6,
            output_dir="./outputs/glm5_1_roundtrip",
        )
        config_path = write_glm5_1_config_file(config, tmp_path / "glm5_1.json")
        loaded = load_glm5_1_config_file(config_path)

        assert loaded.task == "technical_support"
        assert loaded.use_lora is True
        assert loaded.learning_rate == 2e-6
        assert loaded.output_dir == "./outputs/glm5_1_roundtrip"

    def test_preview_payload_can_be_loaded_as_config(self, tmp_path):
        config = get_glm5_1_config(task="sales", output_dir="./outputs/glm5_1_preview")
        preview_path = write_glm5_1_config_file(
            config,
            tmp_path / "glm5_1_preview.json",
            include_preview=True,
        )
        loaded = load_glm5_1_config_file(preview_path)
        preview = create_glm5_1_preview(loaded)

        assert loaded.task == "sales"
        assert preview["config"]["output_dir"] == "./outputs/glm5_1_preview"

    def test_examples_helper_reexports_packaged_symbols(self):
        from examples import glm5_1_config as example_config_module

        assert example_config_module.Glm51Config is Glm51Config
        assert example_config_module.GLM5_1_BASE_MODEL == GLM5_1_BASE_MODEL
        assert example_config_module.GLM5_1_FP8_MODEL == GLM5_1_FP8_MODEL
        assert (
            example_config_module.GLM5_1_STARTER_PROFILE_CHOICES
            == GLM5_1_STARTER_PROFILE_CHOICES
        )
        assert (
            example_config_module.describe_glm5_1_starter_profiles
            is describe_glm5_1_starter_profiles
        )
        assert (
            example_config_module.get_glm5_1_serving_recommendations
            is get_glm5_1_serving_recommendations
        )


class TestGlm51StarterScript:
    """Test the dedicated GLM 5.1 starter script surface."""

    def test_training_function_signature(self):
        from examples import finetune_glm5_1_gspo as training_module

        sig = inspect.signature(training_module.finetune_glm5_1)
        params = list(sig.parameters.keys())

        assert "task" in params
        assert "starter_profile" in params
        assert "use_4bit" in params
        assert "dry_run" in params

    def test_dry_run_preview_payload(self):
        from examples.finetune_glm5_1_gspo import create_glm5_1_dry_run_payload

        preview = create_glm5_1_dry_run_payload()

        assert preview["model_name"] == GLM5_1_BASE_MODEL
        assert preview["summary"]["starter_profile"] == "balanced"
        assert preview["agent_config"]["trust_remote_code"] is True
        assert preview["serving_manifest"]["recommended"]["reasoning_parser"] == "glm45"
        assert preview["serving_manifest"]["recommended"]["tool_call_parser"] == "glm45"

    def test_dry_run_preview_payload_fp8(self):
        from examples.finetune_glm5_1_gspo import create_glm5_1_dry_run_payload

        preview = create_glm5_1_dry_run_payload(
            model_name=GLM5_1_FP8_MODEL, use_fp8_serving=True
        )

        assert preview["model_name"] == GLM5_1_FP8_MODEL
        assert preview["serving_manifest"]["recommended"]["quantization"] == "fp8"
        assert preview["serving_manifest"]["recommended"]["pipeline_parallel_size"] == 1

    def test_async_dry_run(self):
        from stateset_agents.training.glm5_1_starter import finetune_glm5_1

        preview = asyncio.run(finetune_glm5_1(dry_run=True))

        assert preview["config"]["model_name"] == GLM5_1_BASE_MODEL
        assert preview["summary"]["starter_profile"] == "balanced"
        assert preview["gspo_overrides"]["use_lora"] is True
        assert preview["gspo_overrides"]["use_vllm"] is True

    def test_cli_dry_run_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_glm5_1_gspo.py",
                "--dry-run",
            ],
            cwd=repo_root,
            text=True,
        )
        payload = json.loads(output)

        assert payload["model_name"] == GLM5_1_BASE_MODEL
        assert payload["config"]["starter_profile"] == "balanced"
        assert payload["gspo_overrides"]["use_lora"] is True
        assert payload["serving_manifest"]["recommended"]["reasoning_parser"] == "glm45"

    def test_cli_memory_profile_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_glm5_1_gspo.py",
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
                "examples/finetune_glm5_1_gspo.py",
                "--task",
                "sales",
                "--list-profiles",
            ],
            cwd=repo_root,
            text=True,
        )
        payload = json.loads(output)

        assert payload["task"] == "sales"
        assert "memory" in payload["profiles"]
        assert payload["profiles"]["quality"]["summary"]["num_generations"] == 8
