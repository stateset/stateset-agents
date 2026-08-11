from stateset_agents.training import (
    QWEN3_CODER_BASE_MODEL,
    QWEN3_CODER_CONFIG_SUFFIXES,
    QWEN3_CODER_DEFAULT_OUTPUT_DIR,
    QWEN3_CODER_LORA_TARGET_MODULES,
    QWEN3_CODER_STARTER_PROFILE_CHOICES,
    QWEN3_CODER_STARTER_PROFILE_DESCRIPTIONS,
    QWEN3_CODER_SUPPORTED_VARIANTS,
    QWEN3_CODER_TASK_CHOICES,
    Qwen3CoderConfig,
    create_qwen3_coder_preview,
    describe_qwen3_coder_starter_profiles,
    finetune_qwen3_coder,
    get_qwen3_coder_config,
    get_qwen3_coder_profile_description,
    get_qwen3_coder_profile_overrides,
    load_qwen3_coder_config_file,
    run_qwen3_coder_config,
    summarize_qwen3_coder_config,
    write_qwen3_coder_config_file,
)
from stateset_agents.training.qwen3_coder_starter import (
    QWEN3_CODER_BASE_MODEL as DirectBaseModel,
)
from stateset_agents.training.qwen3_coder_starter import (
    Qwen3CoderConfig as DirectQwen3CoderConfig,
)


def test_qwen3_coder_training_exports_remain_available() -> None:
    assert QWEN3_CODER_BASE_MODEL == DirectBaseModel
    assert ".json" in QWEN3_CODER_CONFIG_SUFFIXES
    assert QWEN3_CODER_BASE_MODEL == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert "memory" in QWEN3_CODER_STARTER_PROFILE_CHOICES
    assert "balanced" in QWEN3_CODER_STARTER_PROFILE_DESCRIPTIONS
    assert QWEN3_CODER_DEFAULT_OUTPUT_DIR == "./outputs/qwen3_coder_gspo"
    assert QWEN3_CODER_BASE_MODEL in QWEN3_CODER_SUPPORTED_VARIANTS
    assert "q_proj" in QWEN3_CODER_LORA_TARGET_MODULES
    assert "in_proj" not in QWEN3_CODER_LORA_TARGET_MODULES
    assert "gate_proj" not in QWEN3_CODER_LORA_TARGET_MODULES
    assert "customer_service" in QWEN3_CODER_TASK_CHOICES
    assert Qwen3CoderConfig is DirectQwen3CoderConfig
    assert callable(get_qwen3_coder_config)
    assert callable(get_qwen3_coder_profile_description)
    assert callable(get_qwen3_coder_profile_overrides)
    assert callable(create_qwen3_coder_preview)
    assert callable(describe_qwen3_coder_starter_profiles)
    assert callable(finetune_qwen3_coder)
    assert callable(load_qwen3_coder_config_file)
    assert callable(run_qwen3_coder_config)
    assert callable(summarize_qwen3_coder_config)
    assert callable(write_qwen3_coder_config_file)
