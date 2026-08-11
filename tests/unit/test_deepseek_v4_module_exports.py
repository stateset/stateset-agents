from stateset_agents.training import (
    DEEPSEEK_V4_BASE_MODEL,
    DEEPSEEK_V4_CONFIG_SUFFIXES,
    DEEPSEEK_V4_DEFAULT_OUTPUT_DIR,
    DEEPSEEK_V4_LORA_TARGET_MODULES,
    DEEPSEEK_V4_STARTER_PROFILE_CHOICES,
    DEEPSEEK_V4_STARTER_PROFILE_DESCRIPTIONS,
    DEEPSEEK_V4_SUPPORTED_VARIANTS,
    DEEPSEEK_V4_TASK_CHOICES,
    DeepseekV4Config,
    create_deepseek_v4_preview,
    describe_deepseek_v4_starter_profiles,
    finetune_deepseek_v4,
    get_deepseek_v4_config,
    get_deepseek_v4_profile_description,
    get_deepseek_v4_profile_overrides,
    load_deepseek_v4_config_file,
    run_deepseek_v4_config,
    summarize_deepseek_v4_config,
    write_deepseek_v4_config_file,
)
from stateset_agents.training.deepseek_v4_starter import (
    DEEPSEEK_V4_BASE_MODEL as DirectBaseModel,
)
from stateset_agents.training.deepseek_v4_starter import (
    DeepseekV4Config as DirectDeepseekV4Config,
)


def test_deepseek_v4_training_exports_remain_available() -> None:
    assert DEEPSEEK_V4_BASE_MODEL == DirectBaseModel
    assert ".json" in DEEPSEEK_V4_CONFIG_SUFFIXES
    assert DEEPSEEK_V4_BASE_MODEL == "deepseek-ai/DeepSeek-V4-Flash"
    assert "memory" in DEEPSEEK_V4_STARTER_PROFILE_CHOICES
    assert "balanced" in DEEPSEEK_V4_STARTER_PROFILE_DESCRIPTIONS
    assert DEEPSEEK_V4_DEFAULT_OUTPUT_DIR == "./outputs/deepseek_v4_gspo"
    assert DEEPSEEK_V4_BASE_MODEL in DEEPSEEK_V4_SUPPORTED_VARIANTS
    assert "wq_a" in DEEPSEEK_V4_LORA_TARGET_MODULES
    assert "wkv" in DEEPSEEK_V4_LORA_TARGET_MODULES
    assert "q_proj" not in DEEPSEEK_V4_LORA_TARGET_MODULES
    assert "gate_proj" not in DEEPSEEK_V4_LORA_TARGET_MODULES
    assert "customer_service" in DEEPSEEK_V4_TASK_CHOICES
    assert DeepseekV4Config is DirectDeepseekV4Config
    assert callable(get_deepseek_v4_config)
    assert callable(get_deepseek_v4_profile_description)
    assert callable(get_deepseek_v4_profile_overrides)
    assert callable(create_deepseek_v4_preview)
    assert callable(describe_deepseek_v4_starter_profiles)
    assert callable(finetune_deepseek_v4)
    assert callable(load_deepseek_v4_config_file)
    assert callable(run_deepseek_v4_config)
    assert callable(summarize_deepseek_v4_config)
    assert callable(write_deepseek_v4_config_file)
