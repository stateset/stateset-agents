from stateset_agents.training import (
    QWEN38_27B_BASE_MODEL,
    QWEN38_27B_CONFIG_SUFFIXES,
    QWEN38_27B_DEFAULT_OUTPUT_DIR,
    QWEN38_27B_LORA_TARGET_MODULES,
    QWEN38_27B_STARTER_PROFILE_CHOICES,
    QWEN38_27B_STARTER_PROFILE_DESCRIPTIONS,
    QWEN38_27B_SUPPORTED_VARIANTS,
    QWEN38_27B_TASK_CHOICES,
    Qwen38Config,
    create_qwen3_8_preview,
    describe_qwen3_8_starter_profiles,
    finetune_qwen3_8,
    get_qwen3_8_config,
    get_qwen3_8_profile_description,
    get_qwen3_8_profile_overrides,
    load_qwen3_8_config_file,
    run_qwen3_8_config,
    summarize_qwen3_8_config,
    write_qwen3_8_config_file,
)
from stateset_agents.training.qwen3_8_starter import (
    QWEN38_27B_BASE_MODEL as DirectBaseModel,
)
from stateset_agents.training.qwen3_8_starter import Qwen38Config as DirectQwen38Config


def test_qwen3_8_training_exports_remain_available() -> None:
    assert QWEN38_27B_BASE_MODEL == DirectBaseModel
    assert ".json" in QWEN38_27B_CONFIG_SUFFIXES
    assert QWEN38_27B_BASE_MODEL == "Qwen/Qwen3.8-27B"
    assert "memory" in QWEN38_27B_STARTER_PROFILE_CHOICES
    assert "balanced" in QWEN38_27B_STARTER_PROFILE_DESCRIPTIONS
    assert QWEN38_27B_DEFAULT_OUTPUT_DIR == "./outputs/qwen3_8_27b_gspo"
    assert QWEN38_27B_BASE_MODEL in QWEN38_27B_SUPPORTED_VARIANTS
    assert "q_proj" in QWEN38_27B_LORA_TARGET_MODULES
    assert "in_proj_qkv" in QWEN38_27B_LORA_TARGET_MODULES
    assert "gate_proj" in QWEN38_27B_LORA_TARGET_MODULES
    assert "Qwen/Qwen3.8-27B-FP8" in QWEN38_27B_SUPPORTED_VARIANTS
    assert "customer_service" in QWEN38_27B_TASK_CHOICES
    assert Qwen38Config is DirectQwen38Config
    assert callable(get_qwen3_8_config)
    assert callable(get_qwen3_8_profile_description)
    assert callable(get_qwen3_8_profile_overrides)
    assert callable(create_qwen3_8_preview)
    assert callable(describe_qwen3_8_starter_profiles)
    assert callable(finetune_qwen3_8)
    assert callable(load_qwen3_8_config_file)
    assert callable(run_qwen3_8_config)
    assert callable(summarize_qwen3_8_config)
    assert callable(write_qwen3_8_config_file)
