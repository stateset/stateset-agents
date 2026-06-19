"""Backward-compatible re-export for the packaged GLM 5.2 starter helpers."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateset_agents.training.glm5_2_starter import (
    GLM5_2_BASE_MODEL,
    GLM5_2_CONFIG_SUFFIXES,
    GLM5_2_DEFAULT_OUTPUT_DIR,
    GLM5_2_FP8_MODEL,
    GLM5_2_LORA_TARGET_MODULES,
    GLM5_2_STARTER_PROFILE_CHOICES,
    GLM5_2_STARTER_PROFILE_DESCRIPTIONS,
    GLM5_2_SUPPORTED_VARIANTS,
    GLM5_2_TASK_CHOICES,
    Glm52Config,
    create_glm5_2_agent_config,
    create_glm5_2_preview,
    describe_glm5_2_starter_profiles,
    finetune_glm5_2,
    get_glm5_2_config,
    get_glm5_2_gspo_config,
    get_glm5_2_gspo_overrides,
    get_glm5_2_profile_description,
    get_glm5_2_profile_overrides,
    get_glm5_2_serving_recommendations,
    get_glm5_2_system_prompt,
    load_glm5_2_config_file,
    run_glm5_2_config,
    summarize_glm5_2_config,
    validate_glm5_2_config,
    write_glm5_2_config_file,
)

__all__ = [
    "GLM5_2_BASE_MODEL",
    "GLM5_2_CONFIG_SUFFIXES",
    "GLM5_2_DEFAULT_OUTPUT_DIR",
    "GLM5_2_FP8_MODEL",
    "GLM5_2_LORA_TARGET_MODULES",
    "GLM5_2_STARTER_PROFILE_CHOICES",
    "GLM5_2_STARTER_PROFILE_DESCRIPTIONS",
    "GLM5_2_SUPPORTED_VARIANTS",
    "GLM5_2_TASK_CHOICES",
    "Glm52Config",
    "create_glm5_2_agent_config",
    "create_glm5_2_preview",
    "describe_glm5_2_starter_profiles",
    "finetune_glm5_2",
    "get_glm5_2_config",
    "get_glm5_2_gspo_config",
    "get_glm5_2_gspo_overrides",
    "get_glm5_2_profile_description",
    "get_glm5_2_profile_overrides",
    "get_glm5_2_serving_recommendations",
    "get_glm5_2_system_prompt",
    "load_glm5_2_config_file",
    "run_glm5_2_config",
    "summarize_glm5_2_config",
    "validate_glm5_2_config",
    "write_glm5_2_config_file",
]
