from stateset_agents.training import (
    KIMI_K26_BASE_MODEL,
    KIMI_K26_CONFIG_SUFFIXES,
    KIMI_K26_DEFAULT_OUTPUT_DIR,
    KIMI_K26_LORA_TARGET_MODULES,
    KIMI_K26_STARTER_PROFILE_CHOICES,
    KIMI_K26_STARTER_PROFILE_DESCRIPTIONS,
    KIMI_K26_SUPPORTED_VARIANTS,
    KIMI_K26_TASK_CHOICES,
    KimiK26Config,
    create_kimi_k2_6_preview,
    describe_kimi_k2_6_starter_profiles,
    finetune_kimi_k2_6,
    get_kimi_k2_6_config,
    get_kimi_k2_6_profile_description,
    get_kimi_k2_6_profile_overrides,
    load_kimi_k2_6_config_file,
    run_kimi_k2_6_config,
    summarize_kimi_k2_6_config,
    write_kimi_k2_6_config_file,
)
from stateset_agents.training.kimi_k2_6_starter import (
    KIMI_K26_BASE_MODEL as DirectBaseModel,
    KimiK26Config as DirectKimiK26Config,
)


def test_kimi_k2_6_training_exports_remain_available() -> None:
    assert KIMI_K26_BASE_MODEL == DirectBaseModel
    assert ".json" in KIMI_K26_CONFIG_SUFFIXES
    assert KIMI_K26_BASE_MODEL == "moonshotai/Kimi-K2.6"
    assert "memory" in KIMI_K26_STARTER_PROFILE_CHOICES
    assert "balanced" in KIMI_K26_STARTER_PROFILE_DESCRIPTIONS
    assert KIMI_K26_DEFAULT_OUTPUT_DIR == "./outputs/kimi_k2_6_gspo"
    assert KIMI_K26_BASE_MODEL in KIMI_K26_SUPPORTED_VARIANTS
    assert "q_proj" in KIMI_K26_LORA_TARGET_MODULES
    assert "customer_service" in KIMI_K26_TASK_CHOICES
    assert KimiK26Config is DirectKimiK26Config
    assert callable(get_kimi_k2_6_config)
    assert callable(get_kimi_k2_6_profile_description)
    assert callable(get_kimi_k2_6_profile_overrides)
    assert callable(create_kimi_k2_6_preview)
    assert callable(describe_kimi_k2_6_starter_profiles)
    assert callable(finetune_kimi_k2_6)
    assert callable(load_kimi_k2_6_config_file)
    assert callable(run_kimi_k2_6_config)
    assert callable(summarize_kimi_k2_6_config)
    assert callable(write_kimi_k2_6_config_file)
