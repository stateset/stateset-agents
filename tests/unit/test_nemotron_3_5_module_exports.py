from stateset_agents.training import (
    NEMOTRON_3_5_BASE_MODEL,
    NEMOTRON_3_5_CONFIG_SUFFIXES,
    NEMOTRON_3_5_DEFAULT_OUTPUT_DIR,
    NEMOTRON_3_5_LORA_TARGET_MODULES,
    NEMOTRON_3_5_STARTER_PROFILE_CHOICES,
    NEMOTRON_3_5_STARTER_PROFILE_DESCRIPTIONS,
    NEMOTRON_3_5_SUPPORTED_VARIANTS,
    NEMOTRON_3_5_TASK_CHOICES,
    Nemotron35Config,
    create_nemotron_3_5_preview,
    describe_nemotron_3_5_starter_profiles,
    finetune_nemotron_3_5,
    get_nemotron_3_5_config,
    get_nemotron_3_5_profile_description,
    get_nemotron_3_5_profile_overrides,
    load_nemotron_3_5_config_file,
    run_nemotron_3_5_config,
    summarize_nemotron_3_5_config,
    write_nemotron_3_5_config_file,
)
from stateset_agents.training.nemotron_3_5_starter import (
    NEMOTRON_3_5_BASE_MODEL as DirectBaseModel,
)
from stateset_agents.training.nemotron_3_5_starter import (
    Nemotron35Config as DirectNemotron35Config,
)


def test_nemotron_3_5_training_exports_remain_available() -> None:
    assert NEMOTRON_3_5_BASE_MODEL == DirectBaseModel
    assert ".json" in NEMOTRON_3_5_CONFIG_SUFFIXES
    assert (
        NEMOTRON_3_5_BASE_MODEL == "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16"
    )
    assert "memory" in NEMOTRON_3_5_STARTER_PROFILE_CHOICES
    assert "balanced" in NEMOTRON_3_5_STARTER_PROFILE_DESCRIPTIONS
    assert NEMOTRON_3_5_DEFAULT_OUTPUT_DIR == "./outputs/nemotron_3_5_gspo"
    assert NEMOTRON_3_5_BASE_MODEL in NEMOTRON_3_5_SUPPORTED_VARIANTS
    assert "q_proj" in NEMOTRON_3_5_LORA_TARGET_MODULES
    assert "in_proj" in NEMOTRON_3_5_LORA_TARGET_MODULES
    assert "customer_service" in NEMOTRON_3_5_TASK_CHOICES
    assert Nemotron35Config is DirectNemotron35Config
    assert callable(get_nemotron_3_5_config)
    assert callable(get_nemotron_3_5_profile_description)
    assert callable(get_nemotron_3_5_profile_overrides)
    assert callable(create_nemotron_3_5_preview)
    assert callable(describe_nemotron_3_5_starter_profiles)
    assert callable(finetune_nemotron_3_5)
    assert callable(load_nemotron_3_5_config_file)
    assert callable(run_nemotron_3_5_config)
    assert callable(summarize_nemotron_3_5_config)
    assert callable(write_nemotron_3_5_config_file)
