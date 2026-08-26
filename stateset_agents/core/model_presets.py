"""Model preset registry for the unified GSPO finetune driver.

This module consolidates the per-model hyperparameters that used to be
scattered across ~15 near-clone ``examples/finetune_*_gspo.py`` scripts and
~7 ``examples/*_config.py`` files into a single, faithfully-copied registry.

Each :class:`ModelPreset` captures the fields that varied across the clone
family. Values are copied as-is from the source scripts; nothing here is a
new recommendation. See each preset's ``notes`` field for provenance and any
discrepancies found while consolidating.

Use :func:`examples.finetune_gspo.main` (``python examples/finetune_gspo.py
--model <preset> --dry-run``) to exercise a preset without downloading a
real model.

This registry used to live at ``examples/model_presets.py``; that module is
now a backwards-compatible re-export shim over this one.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPreset:
    """Per-model hyperparameters extracted from the clone finetune scripts."""

    model_id: str
    tokenizer_id: str
    lora_target_modules: tuple[str, ...]
    max_prompt_length: int
    max_completion_length: int
    learning_rate: float
    num_generations: int
    bf16: bool = True
    use_4bit: bool = False
    use_8bit: bool = False
    chat_template_override: str | None = None
    notes: str = ""
    starter_module: str | None = None
    """Name of the packaged ``stateset_agents.training.<module>`` starter
    (e.g. ``"kimi_k3_starter"``) that owns this preset's hyperparameters, if
    any. When set, ``examples/finetune_gspo.py --starter-profile`` delegates
    profile resolution (balanced/memory/quality) to that starter's
    ``get_<name>_config``/``run_<name>_config`` functions instead of the
    driver's own ``build_gspo_config``.
    """

    cli_command: str | None = None
    """Name of the packaged ``stateset-agents <cli_command>`` Typer command
    generated for this preset (e.g. ``"qwen3-5-0-8b"``). CLI command names do
    not always match the preset key (``qwen3.5-0.8b``), so the mapping is
    explicit. ``None`` means no CLI command is generated for this preset.
    """

    cli_display_name: str = ""
    """Long label used in ``--help`` text and the command docstring
    (e.g. ``"Qwen/Qwen3.5-0.8B"``)."""

    cli_echo_label: str = ""
    """Short label used in the command's console messages
    (e.g. ``"Qwen3.5-0.8B"``)."""

    cli_write_label: str = ""
    """Label used in the ``--write-config`` help string (e.g. ``"Qwen"``)."""

    cli_config_stem: str = ""
    """File stem suggested by the dry-run ``--write-config`` hint
    (e.g. ``"qwen3_5_0_8b"`` for ``./qwen3_5_0_8b.json``)."""

    cli_symbol_prefix: str = ""
    """Prefix of the starter module's module-level constants
    (e.g. ``"QWEN35_08B"`` for ``QWEN35_08B_TASK_CHOICES``)."""

    cli_symbol_infix: str = ""
    """Infix of the starter module's helper functions
    (e.g. ``"qwen3_5"`` for ``get_qwen3_5_config``)."""

    cli_run_function: str | None = None
    """Name of the starter's run function. Defaults to
    ``run_<cli_symbol_infix>_config`` when ``None``."""

    cli_model_help_verb: str = "prefer"
    """Verb used in the ``--model`` help string ("prefer" or "use")."""

    cli_default_iterations: int = 16
    """Fallback used by ``--iterations`` coercion when the value is invalid."""

    def __post_init__(self) -> None:
        if self.cli_command is None:
            return
        required = (
            "cli_display_name",
            "cli_echo_label",
            "cli_write_label",
            "cli_config_stem",
            "cli_symbol_prefix",
            "cli_symbol_infix",
        )
        missing = [name for name in required if not getattr(self, name)]
        if missing:
            raise ValueError(
                f"Preset with cli_command={self.cli_command!r} is missing "
                f"required CLI metadata: {', '.join(missing)}."
            )
        if self.starter_module is None:
            raise ValueError(
                f"Preset with cli_command={self.cli_command!r} must set starter_module."
            )


PRESETS: dict[str, ModelPreset] = {
    "muse-glimmer": ModelPreset(
        model_id="meta-models/Muse-Glimmer-30B",
        tokenizer_id="meta-models/Muse-Glimmer-30B",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=4096,
        max_completion_length=1024,
        learning_rate=3e-6,
        num_generations=4,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.muse_glimmer_starter "
            "(MUSE_GLIMMER_*). examples/finetune_muse_glimmer_gspo.py delegates "
            "to the packaged starter rather than duplicating hyperparameters."
        ),
        starter_module="muse_glimmer_starter",
        cli_command="muse-glimmer",
        cli_display_name="meta-models/Muse-Glimmer-30B",
        cli_echo_label="Muse Glimmer",
        cli_write_label="Muse Glimmer",
        cli_config_stem="muse_glimmer",
        cli_symbol_prefix="MUSE_GLIMMER",
        cli_symbol_infix="muse_glimmer",
    ),
    "nemotron-3-5": ModelPreset(
        model_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        tokenizer_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj",
            "out_proj",
        ),
        max_prompt_length=4096,
        max_completion_length=1024,
        learning_rate=3e-6,
        num_generations=4,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.nemotron_3_5_starter "
            "(NEMOTRON_3_5_*). examples/finetune_nemotron_3_5_gspo.py delegates "
            "to the packaged starter rather than duplicating hyperparameters."
        ),
        starter_module="nemotron_3_5_starter",
        cli_command="nemotron-3-5",
        cli_display_name="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        cli_echo_label="Nemotron 3.5",
        cli_write_label="Nemotron 3.5",
        cli_config_stem="nemotron_3_5",
        cli_symbol_prefix="NEMOTRON_3_5",
        cli_symbol_infix="nemotron_3_5",
    ),
    "qwen3.8-27b": ModelPreset(
        model_id="Qwen/Qwen3.8-27B",
        tokenizer_id="Qwen/Qwen3.8-27B",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "in_proj_qkv",
            "out_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=4096,
        max_completion_length=1024,
        learning_rate=3e-6,
        num_generations=4,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.qwen3_8_starter "
            "(QWEN38_27B_*). examples/finetune_qwen3_8_27b_gspo.py delegates "
            "to the packaged starter rather than duplicating hyperparameters."
        ),
        starter_module="qwen3_8_starter",
        cli_command="qwen3-8-27b",
        cli_display_name="Qwen/Qwen3.8-27B",
        cli_echo_label="Qwen3.8 27B",
        cli_write_label="Qwen3.8 27B",
        cli_config_stem="qwen3_8_27b",
        cli_symbol_prefix="QWEN38_27B",
        cli_symbol_infix="qwen3_8",
    ),
    "qwen3.8-flash-next": ModelPreset(
        model_id="Qwen/Qwen3.8-Flash-Next",
        tokenizer_id="Qwen/Qwen3.8-Flash-Next",
        lora_target_modules=(
            # Gated DeltaNet layers.
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_a",
            "in_proj_b",
            "out_proj",
            # Qwen Sparse Attention layers and lightweight indexer.
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "index_qk_proj",
        ),
        max_prompt_length=8192,
        max_completion_length=2048,
        learning_rate=2e-6,
        num_generations=4,
        bf16=True,
        use_4bit=True,
        use_8bit=False,
        notes=(
            "Official qwen4_exp native-multimodal checkpoint (125B main MoE / "
            "6B active, plus 51B n-gram embeddings and 4B MTP; 262K native "
            "context). LoRA targets were verified against the official weight "
            "index on 2026-08-26 and cover Gated DeltaNet plus Qwen Sparse "
            "Attention while excluding the 512-expert MoE and vision tensors. "
            "The full-attention leaf names also match the single MTP layer. "
            "StateSet's RL path is text-only; use AutoProcessor or a supported "
            "serving engine for image/video inference. The FP8 repository is "
            "intended for inference rather than adapter training."
        ),
    ),
    "qwen3-coder": ModelPreset(
        model_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        tokenizer_id="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ),
        max_prompt_length=4096,
        max_completion_length=1024,
        learning_rate=3e-6,
        num_generations=4,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.qwen3_coder_starter "
            "(QWEN3_CODER_*). examples/finetune_qwen3_coder_gspo.py delegates "
            "to the packaged starter rather than duplicating hyperparameters. "
            "LoRA targets are attention-only: the 128-expert MoE MLPs are "
            "impractical LoRA targets."
        ),
        starter_module="qwen3_coder_starter",
        cli_command="qwen3-coder",
        cli_display_name="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        cli_echo_label="Qwen3 Coder",
        cli_write_label="Qwen3 Coder",
        cli_config_stem="qwen3_coder",
        cli_symbol_prefix="QWEN3_CODER",
        cli_symbol_infix="qwen3_coder",
    ),
    "gpt-oss": ModelPreset(
        model_id="openai/gpt-oss-20b",
        tokenizer_id="openai/gpt-oss-20b",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ),
        max_prompt_length=4096,
        max_completion_length=1024,
        learning_rate=3e-6,
        num_generations=4,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.gpt_oss_starter "
            "(GPT_OSS_*). examples/finetune_gpt_oss_gspo.py delegates to the "
            "packaged starter rather than duplicating hyperparameters. LoRA "
            "targets are attention-only (verified against the gpt-oss-20b "
            "weight map); the MoE expert weights are fused per-layer tensors."
        ),
        starter_module="gpt_oss_starter",
        cli_command="gpt-oss",
        cli_display_name="openai/gpt-oss-20b",
        cli_echo_label="gpt-oss",
        cli_write_label="gpt-oss",
        cli_config_stem="gpt_oss",
        cli_symbol_prefix="GPT_OSS",
        cli_symbol_infix="gpt_oss",
    ),
    "deepseek-v4": ModelPreset(
        model_id="deepseek-ai/DeepSeek-V4-Flash",
        tokenizer_id="deepseek-ai/DeepSeek-V4-Flash",
        lora_target_modules=(
            "wq_a",
            "wq_b",
            "wkv",
            "wo_a",
            "wo_b",
        ),
        max_prompt_length=8192,
        max_completion_length=1536,
        learning_rate=2e-6,
        num_generations=4,
        bf16=True,
        use_4bit=True,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.deepseek_v4_starter "
            "(DEEPSEEK_V4_*). examples/finetune_deepseek_v4_gspo.py delegates "
            "to the packaged starter rather than duplicating hyperparameters. "
            "LoRA targets use the checkpoint's MLA projection names (verified "
            "against the safetensors weight map); llama-style q_proj/k_proj/"
            "v_proj do not exist in this architecture."
        ),
        starter_module="deepseek_v4_starter",
        cli_command="deepseek-v4",
        cli_display_name="deepseek-ai/DeepSeek-V4-Flash",
        cli_echo_label="deepseek-v4",
        cli_write_label="deepseek-v4",
        cli_config_stem="deepseek_v4",
        cli_symbol_prefix="DEEPSEEK_V4",
        cli_symbol_infix="deepseek_v4",
    ),
    "kimi-k3": ModelPreset(
        model_id="moonshotai/Kimi-K3",
        tokenizer_id="moonshotai/Kimi-K3",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=4096,
        max_completion_length=1024,
        learning_rate=3e-6,
        num_generations=4,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.kimi_k3_starter "
            "(KIMI_K3_*). examples/finetune_kimi_k3_gspo.py delegates to the "
            "packaged starter rather than duplicating hyperparameters."
        ),
        starter_module="kimi_k3_starter",
        cli_command="kimi-k3",
        cli_display_name="moonshotai/Kimi-K3",
        cli_echo_label="Kimi-K3",
        cli_write_label="Kimi",
        cli_config_stem="kimi_k3",
        cli_symbol_prefix="KIMI_K3",
        cli_symbol_infix="kimi_k3",
    ),
    "kimi-k2.5": ModelPreset(
        model_id="moonshotai/Kimi-K2.5",
        tokenizer_id="moonshotai/Kimi-K2.5",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=2048,
        max_completion_length=2048,
        learning_rate=5e-6,
        num_generations=6,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Two clone scripts existed for this model: "
            "finetune_kimi_k2_5_gspo.py (last touched 2026-03-23, "
            "learning_rate=3e-6, num_generations=8, "
            "max_prompt/completion_length=1536/1024) vs "
            "finetune_kimi_k25_gspo.py (last touched 2026-04-19, "
            "learning_rate=5e-6, num_generations=6, "
            "max_prompt/completion_length=2048/2048). Per git log, "
            "finetune_kimi_k25_gspo.py is newer, so its values are used "
            "here. examples/kimi_k25_config.py (KimiK25Config defaults) "
            "instead uses learning_rate=3e-6, num_generations=8, "
            "max_prompt_length=8192, max_completion_length=4096 -- a third "
            "divergent set of defaults, recorded here for visibility."
        ),
    ),
    "kimi-k2.6": ModelPreset(
        model_id="moonshotai/Kimi-K2.6",
        tokenizer_id="moonshotai/Kimi-K2.6",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=4096,
        max_completion_length=1024,
        learning_rate=3e-6,
        num_generations=4,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.kimi_k2_6_starter "
            "(KIMI_K26_*). examples/finetune_kimi_k2_6_gspo.py delegates to "
            "the packaged starter."
        ),
        starter_module="kimi_k2_6_starter",
        cli_command="kimi-k2-6",
        cli_display_name="moonshotai/Kimi-K2.6",
        cli_echo_label="Kimi-K2.6",
        cli_write_label="Kimi",
        cli_config_stem="kimi_k2_6",
        cli_symbol_prefix="KIMI_K26",
        cli_symbol_infix="kimi_k2_6",
    ),
    "glm5.1": ModelPreset(
        model_id="zai-org/GLM-5.1",
        tokenizer_id="zai-org/GLM-5.1",
        lora_target_modules=(
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=8192,
        max_completion_length=1536,
        learning_rate=2e-6,
        num_generations=4,
        bf16=True,
        use_4bit=True,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.glm5_1_starter "
            "(GLM5_1_*). examples/glm5_1_config.py and "
            "examples/finetune_glm5_1_gspo.py are re-export shims over the "
            "same packaged starter, so there is no discrepancy to reconcile."
        ),
        starter_module="glm5_1_starter",
    ),
    "glm5.2": ModelPreset(
        model_id="zai-org/GLM-5.2",
        tokenizer_id="zai-org/GLM-5.2",
        lora_target_modules=(
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=8192,
        max_completion_length=1536,
        learning_rate=2e-6,
        num_generations=4,
        bf16=True,
        use_4bit=True,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.glm5_2_starter "
            "(GLM5_2_*); mirrors GLM 5.1's defaults. The starter also "
            "exposes GLM5_2_FP8_MODEL as an alternate checkpoint."
        ),
        starter_module="glm5_2_starter",
    ),
    "glm5.3-flash": ModelPreset(
        model_id="zai-org/GLM-5.3-Flash",
        tokenizer_id="zai-org/GLM-5.3-Flash",
        lora_target_modules=(
            # Linear-attention layers.
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "f_a_proj",
            "f_b_proj",
            "g_a_proj",
            "g_b_proj",
            "b_proj",
            # DeepSeek sparse-attention layers.
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
        ),
        max_prompt_length=8192,
        max_completion_length=2048,
        learning_rate=2e-6,
        num_generations=4,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Official native-multimodal FP8 checkpoint (320B total / 18B active, "
            "1M context, glm5_next architecture). LoRA targets were verified "
            "against the official weight index on 2026-08-26 and cover both "
            "linear- and sparse-attention text layers while excluding the vision "
            "tower and 288-expert MoE tensors. StateSet's RL path is text-only; "
            "native image/video inference uses AutoProcessor or vLLM/SGLang."
        ),
    ),
    "qwen3": ModelPreset(
        model_id="Qwen/Qwen2.5-7B",
        tokenizer_id="Qwen/Qwen2.5-7B",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=2048,
        max_completion_length=2048,
        learning_rate=5e-6,
        num_generations=6,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "examples/finetune_qwen3_gspo.py is a generic driver that "
            "branches on model size (0.5B/0.8B/1.5B-3B/7B-14B/27B/32B+). "
            "This preset captures its 7B-14B branch (representative "
            "'larger model with LoRA' example from the script's docstring, "
            "Qwen/Qwen2.5-7B). The 0.8B and 27B branches are captured "
            "separately as the qwen3.5-0.8b and qwen3.5-27b presets."
        ),
    ),
    "qwen3.5-0.8b": ModelPreset(
        model_id="Qwen/Qwen3.5-0.8B-Base",
        tokenizer_id="Qwen/Qwen3.5-0.8B-Base",
        lora_target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        max_prompt_length=1024,
        max_completion_length=768,
        learning_rate=8e-6,
        num_generations=4,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.qwen3_5_starter "
            "(QWEN35_08B_*). examples/qwen3_5_config.py and "
            "examples/finetune_qwen3_5_0_8b_gspo.py are re-export shims "
            "over the same packaged starter."
        ),
        starter_module="qwen3_5_starter",
        cli_command="qwen3-5-0-8b",
        cli_display_name="Qwen/Qwen3.5-0.8B",
        cli_echo_label="Qwen3.5-0.8B",
        cli_write_label="Qwen",
        cli_config_stem="qwen3_5_0_8b",
        cli_symbol_prefix="QWEN35_08B",
        cli_symbol_infix="qwen3_5",
        cli_run_function="run_qwen3_5_0_8b_config",
        cli_default_iterations=25,
    ),
    "qwen3.5-27b": ModelPreset(
        model_id="Qwen/Qwen3.5-27B",
        tokenizer_id="Qwen/Qwen3.5-27B",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=4096,
        max_completion_length=2048,
        learning_rate=3e-6,
        num_generations=8,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "Values copied from the 27B branch of "
            "examples/finetune_qwen3_gspo.py::get_qwen3_config, matching "
            "examples/finetune_qwen3_5_27b_gspo.py's QWEN35_27B_MODEL "
            "target. This model is tuned for LoRA + vLLM-backed serving "
            "(use_reference_model=True, top_p=0.95 upstream; not modeled "
            "as ModelPreset fields since they are training-loop knobs, not "
            "per-model identity)."
        ),
    ),
    "gemma3": ModelPreset(
        model_id="google/gemma-2-9b-it",
        tokenizer_id="google/gemma-2-9b-it",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=2048,
        max_completion_length=2048,
        learning_rate=5e-6,
        num_generations=6,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "examples/finetune_gemma3_gspo.py branches on size "
            "(2b/9b/27b). This preset captures the 9b branch "
            "(google/gemma-2-9b-it), the script's middle example."
        ),
    ),
    "gemma4-31b": ModelPreset(
        model_id="google/gemma-4-31B-it",
        tokenizer_id="google/gemma-4-31B-it",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=4096,
        max_completion_length=1024,
        learning_rate=3e-6,
        num_generations=4,
        bf16=True,
        use_4bit=True,
        use_8bit=False,
        notes=(
            "Values copied from stateset_agents.training.gemma4_starter "
            "(GEMMA4_31B_*). examples/gemma4_config.py and "
            "examples/finetune_gemma4_31b_gspo.py are re-export shims over "
            "the same packaged starter."
        ),
        starter_module="gemma4_starter",
        cli_command="gemma-4-31b",
        cli_display_name="Gemma 4 31B",
        cli_echo_label="Gemma 4 31B",
        cli_write_label="Gemma",
        cli_config_stem="gemma4_31b",
        cli_symbol_prefix="GEMMA4_31B",
        cli_symbol_infix="gemma4_31b",
        cli_model_help_verb="use",
        cli_default_iterations=20,
    ),
    "llama3": ModelPreset(
        model_id="meta-llama/Llama-3.1-8B-Instruct",
        tokenizer_id="meta-llama/Llama-3.1-8B-Instruct",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=4096,
        max_completion_length=2048,
        learning_rate=5e-6,
        num_generations=6,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "examples/finetune_llama3_gspo.py branches on size "
            "(1B/3B/8B/70B). This preset captures the 8B branch "
            "(meta-llama/Llama-3.1-8B-Instruct)."
        ),
    ),
    "mistral": ModelPreset(
        model_id="mistralai/Mistral-7B-Instruct-v0.3",
        tokenizer_id="mistralai/Mistral-7B-Instruct-v0.3",
        lora_target_modules=(
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ),
        max_prompt_length=2048,
        max_completion_length=1024,
        learning_rate=1e-5,
        num_generations=4,
        bf16=True,
        use_4bit=False,
        use_8bit=False,
        notes=(
            "examples/finetune_mistral_gspo.py branches on size/MoE "
            "(7B/nemo-12b/mixtral-8x7b/mixtral-8x22b). This preset "
            "captures the 7B, non-MoE branch "
            "(mistralai/Mistral-7B-Instruct-v0.3)."
        ),
    ),
}


def list_preset_names() -> list[str]:
    """Return preset names in a stable, deterministic order."""

    return list(PRESETS.keys())


def get_preset(name: str) -> ModelPreset:
    """Look up a preset by name, raising a clear error for typos."""

    try:
        return PRESETS[name]
    except KeyError as exc:
        available = ", ".join(sorted(PRESETS))
        raise KeyError(
            f"Unknown model preset {name!r}. Available presets: {available}"
        ) from exc


__all__ = ["ModelPreset", "PRESETS", "get_preset", "list_preset_names"]
