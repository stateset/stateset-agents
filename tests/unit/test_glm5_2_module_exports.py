from stateset_agents.training import (
    GLM5_2_BASE_MODEL,
    GLM5_2_CONFIG_SUFFIXES,
    GLM5_2_DEFAULT_OUTPUT_DIR,
    GLM5_2_FP8_MODEL,
    GLM5_2_LORA_TARGET_MODULES,
    GLM5_2_STARTER_PROFILE_CHOICES,
    GLM5_2_STARTER_PROFILE_DESCRIPTIONS,
    GLM5_2_SUPPORTED_VARIANTS,
    GLM5_2_TASK_CHOICES,
    Glm52Config,
    build_serving_manifest,
    create_glm5_2_preview,
    describe_glm5_2_starter_profiles,
    export_merged_model_for_serving,
    finetune_glm5_2,
    get_glm5_2_config,
    get_glm5_2_profile_description,
    get_glm5_2_profile_overrides,
    get_glm5_2_serving_recommendations,
    load_glm5_2_config_file,
    run_glm5_2_config,
    summarize_glm5_2_config,
    write_glm5_2_config_file,
    write_serving_manifest,
)
from stateset_agents.training.glm5_2_starter import (
    GLM5_2_BASE_MODEL as DirectBaseModel,
    Glm52Config as DirectGlm52Config,
)


def test_glm5_2_training_exports_remain_available() -> None:
    assert GLM5_2_BASE_MODEL == DirectBaseModel
    assert ".json" in GLM5_2_CONFIG_SUFFIXES
    assert GLM5_2_BASE_MODEL == "zai-org/GLM-5.2"
    assert GLM5_2_FP8_MODEL == "your-org/GLM-5.2-FP8"
    assert "memory" in GLM5_2_STARTER_PROFILE_CHOICES
    assert "balanced" in GLM5_2_STARTER_PROFILE_DESCRIPTIONS
    assert GLM5_2_DEFAULT_OUTPUT_DIR == "./outputs/glm5_2_gspo"
    assert GLM5_2_BASE_MODEL in GLM5_2_SUPPORTED_VARIANTS
    assert GLM5_2_FP8_MODEL in GLM5_2_SUPPORTED_VARIANTS
    assert "q_a_proj" in GLM5_2_LORA_TARGET_MODULES
    assert "kv_a_proj_with_mqa" in GLM5_2_LORA_TARGET_MODULES
    assert "customer_service" in GLM5_2_TASK_CHOICES
    assert Glm52Config is DirectGlm52Config
    assert callable(get_glm5_2_config)
    assert callable(get_glm5_2_profile_description)
    assert callable(get_glm5_2_profile_overrides)
    assert callable(get_glm5_2_serving_recommendations)
    assert callable(create_glm5_2_preview)
    assert callable(describe_glm5_2_starter_profiles)
    assert callable(build_serving_manifest)
    assert callable(export_merged_model_for_serving)
    assert callable(finetune_glm5_2)
    assert callable(load_glm5_2_config_file)
    assert callable(run_glm5_2_config)
    assert callable(summarize_glm5_2_config)
    assert callable(write_serving_manifest)
    assert callable(write_glm5_2_config_file)
