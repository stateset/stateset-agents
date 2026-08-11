from stateset_agents.training import (
    GPT_OSS_BASE_MODEL,
    GPT_OSS_CONFIG_SUFFIXES,
    GPT_OSS_DEFAULT_OUTPUT_DIR,
    GPT_OSS_LORA_TARGET_MODULES,
    GPT_OSS_STARTER_PROFILE_CHOICES,
    GPT_OSS_STARTER_PROFILE_DESCRIPTIONS,
    GPT_OSS_SUPPORTED_VARIANTS,
    GPT_OSS_TASK_CHOICES,
    GptOssConfig,
    create_gpt_oss_preview,
    describe_gpt_oss_starter_profiles,
    finetune_gpt_oss,
    get_gpt_oss_config,
    get_gpt_oss_profile_description,
    get_gpt_oss_profile_overrides,
    load_gpt_oss_config_file,
    run_gpt_oss_config,
    summarize_gpt_oss_config,
    write_gpt_oss_config_file,
)
from stateset_agents.training.gpt_oss_starter import (
    GPT_OSS_BASE_MODEL as DirectBaseModel,
)
from stateset_agents.training.gpt_oss_starter import GptOssConfig as DirectGptOssConfig


def test_gpt_oss_training_exports_remain_available() -> None:
    assert GPT_OSS_BASE_MODEL == DirectBaseModel
    assert ".json" in GPT_OSS_CONFIG_SUFFIXES
    assert GPT_OSS_BASE_MODEL == "openai/gpt-oss-20b"
    assert "memory" in GPT_OSS_STARTER_PROFILE_CHOICES
    assert "balanced" in GPT_OSS_STARTER_PROFILE_DESCRIPTIONS
    assert GPT_OSS_DEFAULT_OUTPUT_DIR == "./outputs/gpt_oss_gspo"
    assert GPT_OSS_BASE_MODEL in GPT_OSS_SUPPORTED_VARIANTS
    assert "openai/gpt-oss-120b" in GPT_OSS_SUPPORTED_VARIANTS
    assert "q_proj" in GPT_OSS_LORA_TARGET_MODULES
    assert "in_proj" not in GPT_OSS_LORA_TARGET_MODULES
    assert "gate_proj" not in GPT_OSS_LORA_TARGET_MODULES
    assert "customer_service" in GPT_OSS_TASK_CHOICES
    assert GptOssConfig is DirectGptOssConfig
    assert callable(get_gpt_oss_config)
    assert callable(get_gpt_oss_profile_description)
    assert callable(get_gpt_oss_profile_overrides)
    assert callable(create_gpt_oss_preview)
    assert callable(describe_gpt_oss_starter_profiles)
    assert callable(finetune_gpt_oss)
    assert callable(load_gpt_oss_config_file)
    assert callable(run_gpt_oss_config)
    assert callable(summarize_gpt_oss_config)
    assert callable(write_gpt_oss_config_file)
