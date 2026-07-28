from stateset_agents.training import (
    KIMI_K3_BASE_MODEL,
    KIMI_K3_CONFIG_SUFFIXES,
    KIMI_K3_DEFAULT_OUTPUT_DIR,
    KIMI_K3_LORA_TARGET_MODULES,
    KIMI_K3_STARTER_PROFILE_CHOICES,
    KIMI_K3_STARTER_PROFILE_DESCRIPTIONS,
    KIMI_K3_SUPPORTED_VARIANTS,
    KIMI_K3_TASK_CHOICES,
    KimiK3Config,
    create_kimi_k3_preview,
    describe_kimi_k3_starter_profiles,
    finetune_kimi_k3,
    get_kimi_k3_config,
    get_kimi_k3_profile_description,
    get_kimi_k3_profile_overrides,
    load_kimi_k3_config_file,
    run_kimi_k3_config,
    summarize_kimi_k3_config,
    write_kimi_k3_config_file,
)
from stateset_agents.training.kimi_k3_starter import (
    KIMI_K3_BASE_MODEL as DirectBaseModel,
)
from stateset_agents.training.kimi_k3_starter import KimiK3Config as DirectKimiK3Config


def test_kimi_k3_training_exports_remain_available() -> None:
    assert KIMI_K3_BASE_MODEL == DirectBaseModel
    assert ".json" in KIMI_K3_CONFIG_SUFFIXES
    assert KIMI_K3_BASE_MODEL == "moonshotai/Kimi-K3"
    assert "memory" in KIMI_K3_STARTER_PROFILE_CHOICES
    assert "balanced" in KIMI_K3_STARTER_PROFILE_DESCRIPTIONS
    assert KIMI_K3_DEFAULT_OUTPUT_DIR == "./outputs/kimi_k3_gspo"
    assert KIMI_K3_BASE_MODEL in KIMI_K3_SUPPORTED_VARIANTS
    assert "q_proj" in KIMI_K3_LORA_TARGET_MODULES
    assert "customer_service" in KIMI_K3_TASK_CHOICES
    assert KimiK3Config is DirectKimiK3Config
    assert callable(get_kimi_k3_config)
    assert callable(get_kimi_k3_profile_description)
    assert callable(get_kimi_k3_profile_overrides)
    assert callable(create_kimi_k3_preview)
    assert callable(describe_kimi_k3_starter_profiles)
    assert callable(finetune_kimi_k3)
    assert callable(load_kimi_k3_config_file)
    assert callable(run_kimi_k3_config)
    assert callable(summarize_kimi_k3_config)
    assert callable(write_kimi_k3_config_file)
