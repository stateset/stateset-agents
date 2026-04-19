"""Unit tests for the Qwen3.5-0.8B starter configuration."""

from __future__ import annotations

import asyncio
import inspect
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

from stateset_agents.training.qwen3_5_starter import (
    QWEN35_08B_BASE_MODEL,
    QWEN35_08B_LORA_TARGET_MODULES,
    QWEN35_08B_STARTER_PROFILE_CHOICES,
    QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS,
    Qwen35Config,
    create_qwen3_5_agent_config,
    create_qwen3_5_preview,
    describe_qwen3_5_starter_profiles,
    get_qwen3_5_config,
    get_qwen3_5_gspo_config,
    get_qwen3_5_profile_description,
    get_qwen3_5_profile_overrides,
    summarize_qwen3_5_config,
    load_qwen3_5_config_file,
    validate_qwen3_5_config,
    write_qwen3_5_config_file,
)
from stateset_agents.training.config import get_config_for_task


class TestQwen35Config:
    """Test suite for the Qwen3.5-0.8B helper config."""

    def test_default_config_creation(self):
        config = Qwen35Config()

        assert config.model_name == QWEN35_08B_BASE_MODEL
        assert config.starter_profile == "balanced"
        assert config.use_lora is True
        assert config.use_wandb is False
        assert config.report_to == "none"
        assert config.trust_remote_code is True
        assert config.attn_implementation == "sdpa"

    def test_memory_profile_defaults(self):
        config = get_qwen3_5_config(starter_profile="memory")

        assert config.starter_profile == "memory"
        assert config.use_4bit is True
        assert config.per_device_train_batch_size == 1
        assert config.gradient_accumulation_steps == 8
        assert config.num_generations == 2
        assert config.max_prompt_length == 768

    def test_quality_profile_defaults(self):
        config = get_qwen3_5_config(starter_profile="quality")
        profile = get_qwen3_5_profile_overrides("quality")

        assert config.starter_profile == "quality"
        assert config.max_prompt_length == profile["max_prompt_length"]
        assert config.max_completion_length == profile["max_completion_length"]
        assert config.num_outer_iterations == profile["num_outer_iterations"]
        assert config.num_generations == profile["num_generations"]

    def test_profile_description_lookup(self):
        description = get_qwen3_5_profile_description("memory")

        assert description == QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS["memory"]
        assert "4-bit" in description

    def test_profile_catalog_lists_all_profiles(self):
        payload = describe_qwen3_5_starter_profiles(task="sales")

        assert payload["task"] == "sales"
        assert payload["default_profile"] == "balanced"
        assert set(payload["profiles"]) == set(QWEN35_08B_STARTER_PROFILE_CHOICES)
        assert payload["profiles"]["memory"]["summary"]["quantization_mode"] == "4bit"

    def test_explicit_overrides_beat_profile_defaults(self):
        config = get_qwen3_5_config(
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
        config = get_qwen3_5_config(task="technical_support", num_outer_iterations=40)

        assert config.task == "technical_support"
        assert config.num_outer_iterations == 40
        assert "technical support" in config.system_prompt.lower()

    def test_config_with_lora_disabled(self):
        config = get_qwen3_5_config(use_lora=False)

        assert config.use_lora is False
        assert config.lora_r is None
        assert config.lora_alpha is None

    def test_quantization_flags(self):
        config = get_qwen3_5_config(use_4bit=True, use_8bit=True)

        assert config.use_4bit is True
        assert config.use_8bit is False

    def test_agent_config_creation(self):
        config = get_qwen3_5_config()
        agent_config = create_qwen3_5_agent_config(config)

        assert agent_config.model_name == QWEN35_08B_BASE_MODEL
        assert agent_config.trust_remote_code is True
        assert agent_config.attn_implementation == "sdpa"
        assert agent_config.max_new_tokens == 768

    def test_config_summary(self):
        config = get_qwen3_5_config(starter_profile="memory")
        summary = summarize_qwen3_5_config(config)

        assert summary["starter_profile"] == "memory"
        assert summary["quantization_mode"] == "4bit"
        assert summary["effective_batch_size"] == 8
        assert summary["uses_quantization"] is True

    def test_gspo_config_generation(self):
        base_config = get_config_for_task(
            "customer_service", model_name=QWEN35_08B_BASE_MODEL
        )
        config = get_qwen3_5_config(task="customer_service")

        with patch(
            "stateset_agents.training.qwen3_5_starter.get_config_for_task", return_value=base_config
        ):
            gspo_config = get_qwen3_5_gspo_config(config)

        assert gspo_config.model_name == QWEN35_08B_BASE_MODEL
        assert gspo_config.use_lora is True
        assert gspo_config.num_generations == 4
        assert gspo_config.max_prompt_length == 1024
        assert gspo_config.max_completion_length == 768
        assert gspo_config.lora_target_modules == QWEN35_08B_LORA_TARGET_MODULES

    def test_validation_warnings(self):
        config = get_qwen3_5_config(
            use_lora=False,
            learning_rate=5e-5,
            per_device_train_batch_size=8,
        )
        warnings = validate_qwen3_5_config(config)

        assert any("learning rate" in warning.lower() for warning in warnings)
        assert any("oom" in warning.lower() for warning in warnings)
        assert any("lora" in warning.lower() for warning in warnings)

    def test_validation_warns_on_unknown_starter_profile(self):
        config = Qwen35Config(starter_profile="custom-profile")
        warnings = validate_qwen3_5_config(config)

        assert any("starter_profile" in warning for warning in warnings)

    def test_json_config_file_roundtrip(self, tmp_path):
        config = get_qwen3_5_config(
            task="technical_support",
            use_lora=False,
            learning_rate=1e-5,
            output_dir="./outputs/qwen3_5_roundtrip",
        )
        config_path = write_qwen3_5_config_file(config, tmp_path / "qwen3_5_0_8b.json")
        loaded = load_qwen3_5_config_file(config_path)

        assert loaded.task == "technical_support"
        assert loaded.use_lora is False
        assert loaded.learning_rate == 1e-5
        assert loaded.output_dir == "./outputs/qwen3_5_roundtrip"

    def test_preview_payload_can_be_loaded_as_config(self, tmp_path):
        config = get_qwen3_5_config(task="sales", output_dir="./outputs/qwen3_5_preview")
        preview_path = write_qwen3_5_config_file(
            config,
            tmp_path / "qwen3_5_preview.json",
            include_preview=True,
        )
        loaded = load_qwen3_5_config_file(preview_path)
        preview = create_qwen3_5_preview(loaded)

        assert loaded.task == "sales"
        assert preview["config"]["output_dir"] == "./outputs/qwen3_5_preview"

    def test_examples_helper_reexports_packaged_symbols(self):
        from examples import qwen3_5_config as example_config_module

        assert example_config_module.Qwen35Config is Qwen35Config
        assert example_config_module.QWEN35_08B_BASE_MODEL == QWEN35_08B_BASE_MODEL
        assert example_config_module.QWEN35_08B_STARTER_PROFILE_CHOICES == QWEN35_08B_STARTER_PROFILE_CHOICES
        assert example_config_module.QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS == QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS
        assert example_config_module.describe_qwen3_5_starter_profiles is describe_qwen3_5_starter_profiles
        assert example_config_module.get_qwen3_5_profile_overrides is get_qwen3_5_profile_overrides
        assert example_config_module.summarize_qwen3_5_config is summarize_qwen3_5_config

    def test_generic_family_script_reuses_shared_0_8b_defaults(self):
        from examples import finetune_qwen3_gspo as family_module

        shared_config = get_qwen3_5_config(
            task="customer_service",
            output_dir="./outputs/qwen3_5_shared_test",
            num_outer_iterations=100,
            generations_per_iteration=40,
            save_steps_every=10,
        )
        base_config = get_config_for_task(
            "customer_service", model_name=QWEN35_08B_BASE_MODEL
        )
        expected = get_qwen3_5_gspo_config(shared_config, base_config=base_config)
        actual = family_module.get_qwen3_config(
            model_name=QWEN35_08B_BASE_MODEL,
            task="customer_service",
            output_dir="./outputs/qwen3_5_shared_test",
        )

        assert actual.model_name == expected.model_name
        assert actual.learning_rate == expected.learning_rate
        assert actual.num_generations == expected.num_generations
        assert actual.generations_per_iteration == expected.generations_per_iteration
        assert actual.lora_target_modules == expected.lora_target_modules
        assert actual.save_steps == expected.save_steps

    def test_generic_family_script_has_qwen3_5_27b_profile(self):
        from examples import finetune_qwen3_gspo as family_module

        config = family_module.get_qwen3_config(
            model_name="Qwen/Qwen3.5-27B",
            task="customer_service",
            use_vllm=True,
            output_dir="./outputs/qwen3_5_27b_shared_test",
        )

        assert config.use_vllm is True
        assert config.use_reference_model is True
        assert config.temperature == 1.0
        assert config.top_p == 0.95
        assert config.num_generations == 8
        assert config.max_prompt_length == 4096


class TestQwen35StarterScript:
    """Test the dedicated starter script surface."""

    def test_training_function_signature(self):
        from examples import finetune_qwen3_5_0_8b_gspo as training_module

        sig = inspect.signature(training_module.finetune_qwen3_5_0_8b)
        params = list(sig.parameters.keys())

        assert "task" in params
        assert "starter_profile" in params
        assert "use_4bit" in params
        assert "dry_run" in params

    def test_dry_run_preview(self):
        from examples import finetune_qwen3_5_0_8b_gspo as training_module

        preview = asyncio.run(training_module.finetune_qwen3_5_0_8b(dry_run=True))

        assert preview["config"]["model_name"] == QWEN35_08B_BASE_MODEL
        assert preview["summary"]["starter_profile"] == "balanced"
        assert preview["agent_config"]["trust_remote_code"] is True
        assert preview["gspo_overrides"]["num_generations"] == 4

    def test_cli_dry_run_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_qwen3_5_0_8b_gspo.py",
                "--dry-run",
            ],
            cwd=repo_root,
            text=True,
        )
        payload = json.loads(output)

        assert payload["config"]["model_name"] == QWEN35_08B_BASE_MODEL
        assert payload["config"]["starter_profile"] == "balanced"
        assert payload["agent_config"]["attn_implementation"] == "sdpa"
        assert payload["gspo_overrides"]["use_lora"] is True

    def test_cli_memory_profile_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_qwen3_5_0_8b_gspo.py",
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
                "examples/finetune_qwen3_5_0_8b_gspo.py",
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


class TestQwen3527BStarterScript:
    """Test the dedicated Qwen3.5-27B starter surface."""

    def test_training_function_signature(self):
        from examples import finetune_qwen3_5_27b_gspo as training_module

        sig = inspect.signature(training_module.finetune_qwen3_5_27b)
        params = list(sig.parameters.keys())

        assert "use_vllm" in params
        assert "export_merged" in params
        assert "language_model_only" in params
        assert "dry_run" in params

    def test_dry_run_preview(self):
        from examples import finetune_qwen3_5_27b_gspo as training_module

        preview = asyncio.run(training_module.finetune_qwen3_5_27b(dry_run=True))

        assert preview["model_name"] == "Qwen/Qwen3.5-27B"
        assert preview["gspo_overrides"]["use_vllm"] is True
        assert preview["gspo_overrides"]["num_generations"] == 8
        assert preview["serving_manifest"]["recommended"]["reasoning_parser"] == "qwen3"
        assert (
            preview["serving_manifest"]["recommended"]["language_model_only"] is True
        )

    def test_cli_dry_run_subprocess(self):
        repo_root = Path(__file__).resolve().parents[2]
        output = subprocess.check_output(
            [
                sys.executable,
                "examples/finetune_qwen3_5_27b_gspo.py",
                "--dry-run",
            ],
            cwd=repo_root,
            text=True,
        )
        payload = json.loads(output)

        assert payload["model_name"] == "Qwen/Qwen3.5-27B"
        assert payload["gspo_overrides"]["use_vllm"] is True
        assert payload["serving_manifest"]["recommended"]["tool_call_parser"] == "qwen3_coder"
        assert payload["serving_manifest"]["recommended"]["language_model_only"] is True
