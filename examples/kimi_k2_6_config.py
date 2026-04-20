"""Backward-compatible re-export for the packaged Kimi-K2.6 starter helpers."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateset_agents.training.kimi_k2_6_starter import (
    KIMI_K26_BASE_MODEL,
    KIMI_K26_CONFIG_SUFFIXES,
    KIMI_K26_DEFAULT_OUTPUT_DIR,
    KIMI_K26_LORA_TARGET_MODULES,
    KIMI_K26_STARTER_PROFILE_CHOICES,
    KIMI_K26_STARTER_PROFILE_DESCRIPTIONS,
    KIMI_K26_SUPPORTED_VARIANTS,
    KIMI_K26_TASK_CHOICES,
    KimiK26Config,
    create_kimi_k2_6_agent_config,
    create_kimi_k2_6_preview,
    describe_kimi_k2_6_starter_profiles,
    finetune_kimi_k2_6,
    get_kimi_k2_6_config,
    get_kimi_k2_6_gspo_config,
    get_kimi_k2_6_gspo_overrides,
    get_kimi_k2_6_profile_description,
    get_kimi_k2_6_profile_overrides,
    get_kimi_k2_6_system_prompt,
    load_kimi_k2_6_config_file,
    run_kimi_k2_6_config,
    summarize_kimi_k2_6_config,
    validate_kimi_k2_6_config,
    write_kimi_k2_6_config_file,
)

__all__ = [
    "KIMI_K26_BASE_MODEL",
    "KIMI_K26_CONFIG_SUFFIXES",
    "KIMI_K26_DEFAULT_OUTPUT_DIR",
    "KIMI_K26_LORA_TARGET_MODULES",
    "KIMI_K26_STARTER_PROFILE_CHOICES",
    "KIMI_K26_STARTER_PROFILE_DESCRIPTIONS",
    "KIMI_K26_SUPPORTED_VARIANTS",
    "KIMI_K26_TASK_CHOICES",
    "KimiK26Config",
    "create_kimi_k2_6_agent_config",
    "create_kimi_k2_6_preview",
    "describe_kimi_k2_6_starter_profiles",
    "finetune_kimi_k2_6",
    "get_kimi_k2_6_config",
    "get_kimi_k2_6_gspo_config",
    "get_kimi_k2_6_gspo_overrides",
    "get_kimi_k2_6_profile_description",
    "get_kimi_k2_6_profile_overrides",
    "get_kimi_k2_6_system_prompt",
    "load_kimi_k2_6_config_file",
    "run_kimi_k2_6_config",
    "summarize_kimi_k2_6_config",
    "validate_kimi_k2_6_config",
    "write_kimi_k2_6_config_file",
]
