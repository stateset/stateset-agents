from stateset_agents.training import (
    INKLING2_MINI_BASE_MODEL,
    INKLING2_MINI_CONFIG_SUFFIXES,
    INKLING2_MINI_DEFAULT_OUTPUT_DIR,
    INKLING2_MINI_LORA_TARGET_MODULES,
    INKLING2_MINI_STARTER_PROFILE_CHOICES,
    INKLING2_MINI_STARTER_PROFILE_DESCRIPTIONS,
    INKLING2_MINI_SUPPORTED_VARIANTS,
    INKLING2_MINI_TASK_CHOICES,
    Inkling2MiniConfig,
    create_inkling2_mini_preview,
    describe_inkling2_mini_starter_profiles,
    finetune_inkling2_mini,
    get_inkling2_mini_config,
    get_inkling2_mini_profile_description,
    get_inkling2_mini_profile_overrides,
    load_inkling2_mini_config_file,
    run_inkling2_mini_config,
    summarize_inkling2_mini_config,
    write_inkling2_mini_config_file,
)
from stateset_agents.training.inkling2_mini_starter import (
    INKLING2_MINI_BASE_MODEL as DirectBaseModel,
)
from stateset_agents.training.inkling2_mini_starter import (
    Inkling2MiniConfig as DirectInkling2MiniConfig,
)


def test_inkling2_mini_training_exports_remain_available() -> None:
    assert INKLING2_MINI_BASE_MODEL == DirectBaseModel
    assert ".json" in INKLING2_MINI_CONFIG_SUFFIXES
    assert INKLING2_MINI_BASE_MODEL == "thinkingmachines/Inkling-2-Mini"
    assert "memory" in INKLING2_MINI_STARTER_PROFILE_CHOICES
    assert "balanced" in INKLING2_MINI_STARTER_PROFILE_DESCRIPTIONS
    assert INKLING2_MINI_DEFAULT_OUTPUT_DIR == "./outputs/inkling2_mini_gspo"
    assert INKLING2_MINI_BASE_MODEL in INKLING2_MINI_SUPPORTED_VARIANTS
    assert INKLING2_MINI_LORA_TARGET_MODULES == ["wq_du", "wk_dv", "wv_dv", "wo_ud"]
    assert "q_proj" not in INKLING2_MINI_LORA_TARGET_MODULES
    assert "gate_proj" not in INKLING2_MINI_LORA_TARGET_MODULES
    assert "customer_service" in INKLING2_MINI_TASK_CHOICES
    assert Inkling2MiniConfig is DirectInkling2MiniConfig
    assert callable(get_inkling2_mini_config)
    assert callable(get_inkling2_mini_profile_description)
    assert callable(get_inkling2_mini_profile_overrides)
    assert callable(create_inkling2_mini_preview)
    assert callable(describe_inkling2_mini_starter_profiles)
    assert callable(finetune_inkling2_mini)
    assert callable(load_inkling2_mini_config_file)
    assert callable(run_inkling2_mini_config)
    assert callable(summarize_inkling2_mini_config)
    assert callable(write_inkling2_mini_config_file)
