"""Backward-compatible re-export for the packaged Kimi-K3 starter helpers."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateset_agents.training.kimi_k3_starter import (
    KIMI_K3_BASE_MODEL,
    KIMI_K3_CONFIG_SUFFIXES,
    KIMI_K3_DEFAULT_OUTPUT_DIR,
    KIMI_K3_LORA_TARGET_MODULES,
    KIMI_K3_STARTER_PROFILE_CHOICES,
    KIMI_K3_STARTER_PROFILE_DESCRIPTIONS,
    KIMI_K3_SUPPORTED_VARIANTS,
    KIMI_K3_TASK_CHOICES,
    KimiK3Config,
    create_kimi_k3_agent_config,
    create_kimi_k3_preview,
    describe_kimi_k3_starter_profiles,
    finetune_kimi_k3,
    get_kimi_k3_config,
    get_kimi_k3_gspo_config,
    get_kimi_k3_gspo_overrides,
    get_kimi_k3_profile_description,
    get_kimi_k3_profile_overrides,
    get_kimi_k3_system_prompt,
    load_kimi_k3_config_file,
    run_kimi_k3_config,
    summarize_kimi_k3_config,
    validate_kimi_k3_config,
    write_kimi_k3_config_file,
)

__all__ = [
    "KIMI_K3_BASE_MODEL",
    "KIMI_K3_CONFIG_SUFFIXES",
    "KIMI_K3_DEFAULT_OUTPUT_DIR",
    "KIMI_K3_LORA_TARGET_MODULES",
    "KIMI_K3_STARTER_PROFILE_CHOICES",
    "KIMI_K3_STARTER_PROFILE_DESCRIPTIONS",
    "KIMI_K3_SUPPORTED_VARIANTS",
    "KIMI_K3_TASK_CHOICES",
    "KimiK3Config",
    "create_kimi_k3_agent_config",
    "create_kimi_k3_preview",
    "describe_kimi_k3_starter_profiles",
    "finetune_kimi_k3",
    "get_kimi_k3_config",
    "get_kimi_k3_gspo_config",
    "get_kimi_k3_gspo_overrides",
    "get_kimi_k3_profile_description",
    "get_kimi_k3_profile_overrides",
    "get_kimi_k3_system_prompt",
    "load_kimi_k3_config_file",
    "run_kimi_k3_config",
    "summarize_kimi_k3_config",
    "validate_kimi_k3_config",
    "write_kimi_k3_config_file",
]
