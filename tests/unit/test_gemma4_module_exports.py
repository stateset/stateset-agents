from stateset_agents.training import (
    GEMMA4_31B_BASE_MODEL,
    GEMMA4_31B_CONFIG_SUFFIXES,
    GEMMA4_31B_DEFAULT_OUTPUT_DIR,
    GEMMA4_31B_LORA_TARGET_MODULES,
    GEMMA4_31B_STARTER_PROFILE_CHOICES,
    GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS,
    GEMMA4_31B_SUPPORTED_VARIANTS,
    GEMMA4_31B_TASK_CHOICES,
    Gemma4Config,
    create_gemma4_31b_preview,
    describe_gemma4_31b_starter_profiles,
    finetune_gemma4_31b,
    get_gemma4_31b_config,
    get_gemma4_31b_profile_description,
    get_gemma4_31b_profile_overrides,
    load_gemma4_31b_config_file,
    run_gemma4_31b_config,
    summarize_gemma4_31b_config,
    write_gemma4_31b_config_file,
)
from stateset_agents.training.gemma4_starter import (
    GEMMA4_31B_BASE_MODEL as DirectBaseModel,
    Gemma4Config as DirectGemma4Config,
)


def test_gemma4_training_exports_remain_available() -> None:
    assert GEMMA4_31B_BASE_MODEL == DirectBaseModel
    assert ".json" in GEMMA4_31B_CONFIG_SUFFIXES
    assert GEMMA4_31B_BASE_MODEL == "google/gemma-4-31B-it"
    assert "memory" in GEMMA4_31B_STARTER_PROFILE_CHOICES
    assert "balanced" in GEMMA4_31B_STARTER_PROFILE_DESCRIPTIONS
    assert GEMMA4_31B_DEFAULT_OUTPUT_DIR == "./outputs/gemma4_31b_gspo"
    assert GEMMA4_31B_BASE_MODEL in GEMMA4_31B_SUPPORTED_VARIANTS
    assert "q_proj" in GEMMA4_31B_LORA_TARGET_MODULES
    assert "customer_service" in GEMMA4_31B_TASK_CHOICES
    assert Gemma4Config is DirectGemma4Config
    assert callable(get_gemma4_31b_config)
    assert callable(get_gemma4_31b_profile_description)
    assert callable(get_gemma4_31b_profile_overrides)
    assert callable(create_gemma4_31b_preview)
    assert callable(describe_gemma4_31b_starter_profiles)
    assert callable(finetune_gemma4_31b)
    assert callable(load_gemma4_31b_config_file)
    assert callable(run_gemma4_31b_config)
    assert callable(summarize_gemma4_31b_config)
    assert callable(write_gemma4_31b_config_file)
