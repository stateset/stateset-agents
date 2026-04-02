"""Backward-compatible re-export for the packaged Gemma 4 31B starter helpers."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateset_agents.training.gemma4_starter import (
    GEMMA4_31B_BASE_MODEL,
    GEMMA4_31B_CONFIG_SUFFIXES,
    GEMMA4_31B_DEFAULT_OUTPUT_DIR,
    GEMMA4_31B_LORA_TARGET_MODULES,
    GEMMA4_31B_STARTER_PROFILE_CHOICES,
    GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS,
    GEMMA4_31B_SUPPORTED_VARIANTS,
    GEMMA4_31B_TASK_CHOICES,
    Gemma4Config,
    create_gemma4_31b_agent_config,
    create_gemma4_31b_preview,
    describe_gemma4_31b_starter_profiles,
    finetune_gemma4_31b,
    get_gemma4_31b_config,
    get_gemma4_31b_gspo_config,
    get_gemma4_31b_gspo_overrides,
    get_gemma4_31b_profile_description,
    get_gemma4_31b_profile_overrides,
    get_gemma4_31b_system_prompt,
    load_gemma4_31b_config_file,
    run_gemma4_31b_config,
    summarize_gemma4_31b_config,
    validate_gemma4_31b_config,
    write_gemma4_31b_config_file,
)

__all__ = [
    "GEMMA4_31B_BASE_MODEL",
    "GEMMA4_31B_CONFIG_SUFFIXES",
    "GEMMA4_31B_DEFAULT_OUTPUT_DIR",
    "GEMMA4_31B_LORA_TARGET_MODULES",
    "GEMMA4_31B_STARTER_PROFILE_CHOICES",
    "GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS",
    "GEMMA4_31B_SUPPORTED_VARIANTS",
    "GEMMA4_31B_TASK_CHOICES",
    "Gemma4Config",
    "create_gemma4_31b_agent_config",
    "create_gemma4_31b_preview",
    "describe_gemma4_31b_starter_profiles",
    "finetune_gemma4_31b",
    "get_gemma4_31b_config",
    "get_gemma4_31b_gspo_config",
    "get_gemma4_31b_gspo_overrides",
    "get_gemma4_31b_profile_description",
    "get_gemma4_31b_profile_overrides",
    "get_gemma4_31b_system_prompt",
    "load_gemma4_31b_config_file",
    "run_gemma4_31b_config",
    "summarize_gemma4_31b_config",
    "validate_gemma4_31b_config",
    "write_gemma4_31b_config_file",
]
