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
