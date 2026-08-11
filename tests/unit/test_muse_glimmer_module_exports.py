from stateset_agents.training import (
    MUSE_GLIMMER_BASE_MODEL,
    MUSE_GLIMMER_CONFIG_SUFFIXES,
    MUSE_GLIMMER_DEFAULT_OUTPUT_DIR,
    MUSE_GLIMMER_LORA_TARGET_MODULES,
    MUSE_GLIMMER_STARTER_PROFILE_CHOICES,
    MUSE_GLIMMER_STARTER_PROFILE_DESCRIPTIONS,
    MUSE_GLIMMER_SUPPORTED_VARIANTS,
    MUSE_GLIMMER_TASK_CHOICES,
    MuseGlimmerConfig,
    create_muse_glimmer_preview,
    describe_muse_glimmer_starter_profiles,
    finetune_muse_glimmer,
    get_muse_glimmer_config,
    get_muse_glimmer_profile_description,
    get_muse_glimmer_profile_overrides,
    load_muse_glimmer_config_file,
    run_muse_glimmer_config,
    summarize_muse_glimmer_config,
    write_muse_glimmer_config_file,
)
from stateset_agents.training.muse_glimmer_starter import (
    MUSE_GLIMMER_BASE_MODEL as DirectBaseModel,
)
from stateset_agents.training.muse_glimmer_starter import (
    MuseGlimmerConfig as DirectMuseGlimmerConfig,
)


def test_muse_glimmer_training_exports_remain_available() -> None:
    assert MUSE_GLIMMER_BASE_MODEL == DirectBaseModel
    assert ".json" in MUSE_GLIMMER_CONFIG_SUFFIXES
    assert MUSE_GLIMMER_BASE_MODEL == "meta-models/Muse-Glimmer-30B"
    assert "memory" in MUSE_GLIMMER_STARTER_PROFILE_CHOICES
    assert "balanced" in MUSE_GLIMMER_STARTER_PROFILE_DESCRIPTIONS
    assert MUSE_GLIMMER_DEFAULT_OUTPUT_DIR == "./outputs/muse_glimmer_gspo"
    assert MUSE_GLIMMER_BASE_MODEL in MUSE_GLIMMER_SUPPORTED_VARIANTS
    assert "q_proj" in MUSE_GLIMMER_LORA_TARGET_MODULES
    assert "customer_service" in MUSE_GLIMMER_TASK_CHOICES
    assert MuseGlimmerConfig is DirectMuseGlimmerConfig
    assert callable(get_muse_glimmer_config)
    assert callable(get_muse_glimmer_profile_description)
    assert callable(get_muse_glimmer_profile_overrides)
    assert callable(create_muse_glimmer_preview)
    assert callable(describe_muse_glimmer_starter_profiles)
    assert callable(finetune_muse_glimmer)
    assert callable(load_muse_glimmer_config_file)
    assert callable(run_muse_glimmer_config)
    assert callable(summarize_muse_glimmer_config)
    assert callable(write_muse_glimmer_config_file)
