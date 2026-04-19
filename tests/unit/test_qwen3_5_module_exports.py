from stateset_agents.training import (
    QWEN35_08B_BASE_MODEL,
    QWEN35_08B_CONFIG_SUFFIXES,
    QWEN35_08B_DEFAULT_OUTPUT_DIR,
    QWEN35_08B_LORA_TARGET_MODULES,
    QWEN35_08B_POST_TRAINED_MODEL,
    QWEN35_08B_STARTER_PROFILE_CHOICES,
    QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS,
    QWEN35_08B_SUPPORTED_VARIANTS,
    QWEN35_08B_TASK_CHOICES,
    Qwen35Config,
    create_qwen3_5_preview,
    describe_qwen3_5_starter_profiles,
    build_serving_manifest,
    export_merged_model_for_serving,
    finetune_qwen3_5_0_8b,
    get_qwen3_5_config,
    get_qwen3_5_profile_description,
    get_qwen3_5_profile_overrides,
    load_qwen3_5_config_file,
    run_qwen3_5_0_8b_config,
    summarize_qwen3_5_config,
    write_serving_manifest,
    write_qwen3_5_config_file,
)
from stateset_agents.training.qwen3_5_starter import (
    QWEN35_08B_BASE_MODEL as DirectBaseModel,
    Qwen35Config as DirectQwen35Config,
)


def test_qwen3_5_training_exports_remain_available() -> None:
    assert QWEN35_08B_BASE_MODEL == DirectBaseModel
    assert ".json" in QWEN35_08B_CONFIG_SUFFIXES
    assert QWEN35_08B_BASE_MODEL == "Qwen/Qwen3.5-0.8B-Base"
    assert "memory" in QWEN35_08B_STARTER_PROFILE_CHOICES
    assert "balanced" in QWEN35_08B_STARTER_PROFILE_DESCRIPTIONS
    assert QWEN35_08B_DEFAULT_OUTPUT_DIR == "./outputs/qwen3_5_0_8b_gspo"
    assert QWEN35_08B_POST_TRAINED_MODEL == "Qwen/Qwen3.5-0.8B"
    assert QWEN35_08B_BASE_MODEL in QWEN35_08B_SUPPORTED_VARIANTS
    assert "q_proj" in QWEN35_08B_LORA_TARGET_MODULES
    assert "customer_service" in QWEN35_08B_TASK_CHOICES
    assert Qwen35Config is DirectQwen35Config
    assert callable(get_qwen3_5_config)
    assert callable(get_qwen3_5_profile_description)
    assert callable(get_qwen3_5_profile_overrides)
    assert callable(create_qwen3_5_preview)
    assert callable(describe_qwen3_5_starter_profiles)
    assert callable(build_serving_manifest)
    assert callable(export_merged_model_for_serving)
    assert callable(finetune_qwen3_5_0_8b)
    assert callable(load_qwen3_5_config_file)
    assert callable(run_qwen3_5_0_8b_config)
    assert callable(summarize_qwen3_5_config)
    assert callable(write_serving_manifest)
    assert callable(write_qwen3_5_config_file)
