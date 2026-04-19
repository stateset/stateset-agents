"""Backward-compatible re-export for the packaged Qwen3.5-0.8B starter helpers."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateset_agents.training.qwen3_5_starter import (
    QWEN35_08B_BASE_MODEL,
    QWEN35_08B_CONFIG_SUFFIXES,
    QWEN35_08B_DEFAULT_OUTPUT_DIR,
    QWEN35_08B_LORA_TARGET_MODULES,
    QWEN35_08B_POST_TRAINED_MODEL,
    QWEN35_08B_STARTER_PROFILE_CHOICES,
    QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS,
    QWEN35_08B_SUPPORTED_VARIANTS,
    QWEN35_08B_TASK_CHOICES,
    Qwen35Config,
    create_qwen3_5_agent_config,
    create_qwen3_5_preview,
    describe_qwen3_5_starter_profiles,
    finetune_qwen3_5_0_8b,
    get_qwen3_5_config,
    get_qwen3_5_gspo_config,
    get_qwen3_5_gspo_overrides,
    get_qwen3_5_profile_description,
    get_qwen3_5_profile_overrides,
    get_qwen3_5_system_prompt,
    load_qwen3_5_config_file,
    run_qwen3_5_0_8b_config,
    summarize_qwen3_5_config,
    validate_qwen3_5_config,
    write_qwen3_5_config_file,
)

__all__ = [
    "QWEN35_08B_BASE_MODEL",
    "QWEN35_08B_CONFIG_SUFFIXES",
    "QWEN35_08B_DEFAULT_OUTPUT_DIR",
    "QWEN35_08B_POST_TRAINED_MODEL",
    "QWEN35_08B_STARTER_PROFILE_CHOICES",
    "QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS",
    "QWEN35_08B_SUPPORTED_VARIANTS",
    "QWEN35_08B_TASK_CHOICES",
    "QWEN35_08B_LORA_TARGET_MODULES",
    "Qwen35Config",
    "create_qwen3_5_agent_config",
    "create_qwen3_5_preview",
    "describe_qwen3_5_starter_profiles",
    "finetune_qwen3_5_0_8b",
    "get_qwen3_5_gspo_overrides",
    "get_qwen3_5_profile_description",
    "get_qwen3_5_profile_overrides",
    "get_qwen3_5_config",
    "get_qwen3_5_gspo_config",
    "get_qwen3_5_system_prompt",
    "load_qwen3_5_config_file",
    "run_qwen3_5_0_8b_config",
    "summarize_qwen3_5_config",
    "validate_qwen3_5_config",
    "write_qwen3_5_config_file",
]
