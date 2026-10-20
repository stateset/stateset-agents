"""Pinned public checkpoints and explicit training limits for large open models.

A catalog entry records an integration, not a successful full-size GPU run.
No model downloads or optional training imports occur when reading this module.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class FrontierModel:
    """An audited checkpoint identity and text attention adapter specification."""

    key: str
    model_id: str
    revision: str
    architecture: str
    model_type: str
    attention_prefix: str
    attention_targets: tuple[str, ...]
    checkpoint_format: str
    checkpoint_bytes: int
    license: str
    blockers: tuple[str, ...] = ()
    trust_remote_code: bool = False
    chat_template_options: tuple[tuple[str, str | bool], ...] = ()
    audit_date: str = "2026-10-15"
    """Date of the checkpoint metadata audit; independent of training verification."""

    @property
    def lora_regex(self) -> str:
        """Match language attention only, excluding encoders, experts and MTP."""
        import re

        targets = "|".join(re.escape(name) for name in self.attention_targets)
        return (
            self.attention_prefix
            + r"\.\d+\.(?:self_attn|linear_attn|attn)\.(?:(?:indexer|forget_gate)\.)?(?:"
            + targets
            + ")"
        )

    def describe(self) -> dict[str, Any]:
        """Return an automation-friendly preview with honest evidence scope."""
        return {
            **asdict(self),
            "lora_regex": self.lora_regex,
            "source": f"https://huggingface.co/{self.model_id}/tree/{self.revision}",
            "training_path_available": not self.blockers,
            "full_size_training_verified": False,
            "multimodal_training_verified": False,
            "optimizer_resume_supported": False,
            "training_dtype": "bfloat16",
            "memory_note": (
                "Checkpoint bytes describe stored tensors, not training memory. "
                "Dequantized base weights, adapters, activations and sharding buffers "
                "must fit; active parameter counts do not determine capacity."
            ),
        }


QWEN_HYBRID_TARGETS = (
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_a",
    "in_proj_b",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
)
GLM_SPARSE_TARGETS = (
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
    "o_proj",
)
KIMI_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "f_a_proj",
    "f_b_proj",
    "g_proj",
    "b_proj",
    "q_a_proj",
    "q_b_proj",
    "kv_a_proj_with_mqa",
    "kv_b_proj",
)

FRONTIER_MODELS: tuple[FrontierModel, ...] = (
    FrontierModel(
        "mimo-v2.6-pro",
        "XiaomiMiMo/MiMo-V2.6-Pro-RL",
        "b61f21852680ee02a29e946e18cf8697f4844061",
        "MiMoV2ForCausalLM",
        "mimo_v2",
        r"model\.layers",
        ("qkv_proj", "o_proj"),
        "mixed-mxfp4-fp8",
        566027990784,
        "mit",
        (
            "The index declares MXFP4 packed experts but config declares FP8; the published "
            "Transformers code has no MXFP4 expert unpacker. A validated checkpoint converter "
            "or upstream training loader is required before training these released weights.",
        ),
        True,
    ),
    FrontierModel(
        "mimo-v2.6-pro-mopd",
        "XiaomiMiMo/MiMo-V2.6-Pro-MOPD",
        "6a7f202e8dbd0e57b3fe1da03917d08d75abc4b8",
        "MiMoV2ForCausalLM",
        "mimo_v2",
        r"model\.layers",
        ("qkv_proj", "o_proj"),
        "mixed-mxfp4-fp8",
        566027990784,
        "mit",
        (
            "The index declares MXFP4 packed experts but config declares FP8; the published "
            "Transformers code has no MXFP4 expert unpacker. A validated checkpoint converter "
            "or upstream training loader is required before training these released weights.",
        ),
        True,
    ),
    FrontierModel(
        "glm5.3",
        "zai-org/GLM-5.3",
        "ba32f8c19c8738e2c57c71798c601d477a7bf844",
        "GlmMoeDsaForCausalLM",
        "glm_moe_dsa",
        r"model\.layers",
        GLM_SPARSE_TARGETS,
        "fp8",
        755617140416,
        "glm-5.3",
    ),
    FrontierModel(
        "kimi-k3",
        "moonshotai/Kimi-K3",
        "f831ab66814297da540d832a5235f8e904f29d06",
        "KimiK3ForConditionalGeneration",
        "kimi_k3",
        r"language_model\.model\.layers",
        KIMI_TARGETS,
        "mxfp4-pack-quantized",
        1560860324864,
        "kimi-k3",
        (
            "Released MXFP4 experts use compressed-tensors mxfp4-pack-quantized. This exact format "
            "needs a validated training/decompression loader and the custom KDA backward kernels; "
            "generic bitsandbytes QLoRA is not a compatible replacement.",
        ),
        True,
    ),
    FrontierModel(
        "glm5.3-flash",
        "zai-org/GLM-5.3-Flash",
        "46a050bec13002ad29ee795c16e61cf4fa3c4a9c",
        "Glm5NextForConditionalGeneration",
        "glm5_next",
        r"model\.language_model\.layers",
        (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "f_a_proj",
            "f_b_proj",
            "g_a_proj",
            "g_b_proj",
            "b_proj",
            "q_a_proj",
            "q_b_proj",
            "kv_a_proj_with_mqa",
            "kv_b_proj",
        ),
        "fp8",
        328326771576,
        "mit",
    ),
    FrontierModel(
        "deepseek-v4.1-flash",
        "deepseek-ai/DeepSeek-V4.1-Flash",
        "5254448103b6254fc646dacbca1cbbf20e9216c1",
        "DeepseekV41ForCausalLM",
        "deepseek_v41",
        r"layers",
        ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"),
        "fp8-custom-layout",
        510286023000,
        "mit",
        (
            "Released tensors use the inference layout (layers.*.attn.*, .scale tensors). "
            "No deepseek_v41 implementation exists in the audited Transformers 5.20 release. "
            "A differentiable CED/CSA2 loader and validated weight mapping are required.",
        ),
    ),
    FrontierModel(
        "deepseek-v4.1-flash-1017",
        "deepseek-ai/DeepSeek-V4.1-Flash-1017",
        "1bae0e1088993b9b82de08f59de5be5a34e42170",
        "DeepseekV41ForCausalLM",
        "deepseek_v41",
        "layers",
        ("wq_a", "wq_b", "wkv", "wo_a", "wo_b"),
        "fp8-custom-layout",
        510286023000,
        "mit",
        (
            (
                "Released tensors use the custom inference layout (layers.*.attn.* and .scale "
                "tensors), and the supplied Transformer.forward runs under "
                "torch.inference_mode(). A validated differentiable loader, packed-expert "
                "handling and backward kernels are required before training these weights."
            ),
            (
                "The pinned release has no chat_template.jinja or inline tokenizer "
                "chat_template. Its encoding/encoding.py differs from the pinned base "
                "release; a reviewed model-specific training codec is required, and the base "
                "template must not be borrowed."
            ),
        ),
        audit_date="2026-10-17",
    ),
    FrontierModel(
        "qwen3.8-2.4t-a95b",
        "Qwen/Qwen3.8-2.4T-A95B",
        "207bd685a7e3696cfaff12ded7c6a7ea0f88c996",
        "Qwen3_5MoeForCausalLM",
        "qwen3_5_moe_text",
        r"model\.layers",
        QWEN_HYBRID_TARGETS,
        "bfloat16",
        4892365451008,
        "qwen3.8-max",
        chat_template_options=(("reasoning_effort", "low"),),
    ),
    FrontierModel(
        "qwen3.8-flash-next",
        "Qwen/Qwen3.8-Flash-Next",
        "de4b8e4d43b917e7706784d8bb445c9af86a3540",
        "Qwen4ExpForConditionalGeneration",
        "qwen4_exp",
        r"model\.language_model\.layers",
        QWEN_HYBRID_TARGETS + ("index_qk_proj",),
        "bfloat16",
        359999963128,
        "qwen-community-1.0",
        chat_template_options=(("enable_thinking", False),),
    ),
)


def get_frontier_model(name: str) -> FrontierModel:
    """Resolve a preset key or exact official repository ID, case-insensitively."""
    for model in FRONTIER_MODELS:
        if name.casefold() in (model.key.casefold(), model.model_id.casefold()):
            return model
    raise ValueError(
        f"Unknown frontier model {name!r}; choose "
        + ", ".join(m.key for m in FRONTIER_MODELS)
    )


def require_frontier_training(
    model: FrontierModel, *, generic_loader: bool = False
) -> None:
    """Reject known layout blockers and incompatible generic training loaders.

    The dedicated frontier SFT recipe can dequantize native FP8 checkpoints;
    generic packaged SFT cannot. Admission and loading share this decision.
    """
    if model.blockers:
        raise RuntimeError(
            f"{model.model_id} training is blocked: " + " ".join(model.blockers)
        )

    if generic_loader and model.checkpoint_format == "fp8":
        raise RuntimeError(
            "Use stateset-agents frontier-sft for this FP8 checkpoint: "
            "the distributed training path explicitly dequantizes to BF16"
        )


__all__ = [
    "FRONTIER_MODELS",
    "FrontierModel",
    "get_frontier_model",
    "require_frontier_training",
]
