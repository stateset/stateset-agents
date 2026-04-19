"""Backward-compatible re-export for the packaged GLM 5.1 starter helpers."""

# ruff: noqa: E402

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from stateset_agents.training.glm5_1_starter import (
    GLM5_1_BASE_MODEL,
    GLM5_1_CONFIG_SUFFIXES,
    GLM5_1_DEFAULT_OUTPUT_DIR,
    GLM5_1_FP8_MODEL,
    GLM5_1_LORA_TARGET_MODULES,
    GLM5_1_STARTER_PROFILE_CHOICES,
    GLM5_1_STARTER_PROFILE_DESCRIPTIONS,
    GLM5_1_SUPPORTED_VARIANTS,
    GLM5_1_TASK_CHOICES,
    Glm51Config,
    create_glm5_1_agent_config,
    create_glm5_1_preview,
    describe_glm5_1_starter_profiles,
    finetune_glm5_1,
    get_glm5_1_config,
    get_glm5_1_gspo_config,
    get_glm5_1_gspo_overrides,
    get_glm5_1_profile_description,
    get_glm5_1_profile_overrides,
    get_glm5_1_serving_recommendations,
    get_glm5_1_system_prompt,
    load_glm5_1_config_file,
    run_glm5_1_config,
    summarize_glm5_1_config,
    validate_glm5_1_config,
    write_glm5_1_config_file,
)

__all__ = [
    "GLM5_1_BASE_MODEL",
    "GLM5_1_CONFIG_SUFFIXES",
    "GLM5_1_DEFAULT_OUTPUT_DIR",
    "GLM5_1_FP8_MODEL",
    "GLM5_1_LORA_TARGET_MODULES",
    "GLM5_1_STARTER_PROFILE_CHOICES",
    "GLM5_1_STARTER_PROFILE_DESCRIPTIONS",
    "GLM5_1_SUPPORTED_VARIANTS",
    "GLM5_1_TASK_CHOICES",
    "Glm51Config",
    "create_glm5_1_agent_config",
    "create_glm5_1_preview",
    "describe_glm5_1_starter_profiles",
    "finetune_glm5_1",
    "get_glm5_1_config",
    "get_glm5_1_gspo_config",
    "get_glm5_1_gspo_overrides",
    "get_glm5_1_profile_description",
    "get_glm5_1_profile_overrides",
    "get_glm5_1_serving_recommendations",
    "get_glm5_1_system_prompt",
    "load_glm5_1_config_file",
    "run_glm5_1_config",
    "summarize_glm5_1_config",
    "validate_glm5_1_config",
    "write_glm5_1_config_file",
]
