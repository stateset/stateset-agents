"""Model maturity and conservative RunPod resource recommendations.

This catalog is intentionally small and evidence-oriented.  It is not a list
of every Hugging Face checkpoint the generic loader can open; it records the
models for which StateSet can offer a deliberate product posture and, where
we have enough information, a safe starting topology.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal, TypedDict

SupportTier = Literal["default", "frontier-preview", "compatibility"]
CertificationStage = Literal[
    "configured",
    "smoke-tested",
    "training-verified",
    "serving-verified",
    "production-certified",
]


class RunPodPlan(TypedDict):
    """Resolved non-billable resource plan returned to CLIs and automation."""

    provider: str
    model: str
    tier: SupportTier
    certification: CertificationStage
    gpu: str
    gpu_count: int
    container_disk_gb: int
    recommendation_evidence: Literal["measured", "estimated", "unknown"]
    manual_review_required: bool
    explicit_overrides: dict[str, bool]
    provisions_hardware: bool
    note: str


@dataclass(frozen=True)
class RunPodResources:
    """A conservative starting topology, not a capacity guarantee."""

    gpu: str
    gpu_count: int
    container_disk_gb: int
    evidence: Literal["measured", "estimated"]
    note: str

    def to_dict(self) -> dict[str, str | int]:
        """Return a stable JSON representation."""
        return asdict(self)


@dataclass(frozen=True)
class ModelSupport:
    """Product tier and highest honestly attained certification stage."""

    model: str
    tier: SupportTier
    certification: CertificationStage
    recommended_provider: str
    runpod: RunPodResources | None = None
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        """Return a stable JSON representation."""
        row = asdict(self)
        row["aliases"] = list(self.aliases)
        return row


_H100 = "NVIDIA H100 80GB HBM3"

MODEL_CATALOG: tuple[ModelSupport, ...] = (
    ModelSupport(
        model="thinkingmachines/Inkling-Small",
        tier="frontier-preview",
        certification="configured",
        recommended_provider="tinker",
    ),
    ModelSupport(
        model="Qwen/Qwen3.5-0.8B",
        aliases=("Qwen/Qwen3.5-0.8B-Base",),
        tier="default",
        certification="training-verified",
        recommended_provider="runpod",
        runpod=RunPodResources(
            gpu="NVIDIA RTX A4000",
            gpu_count=1,
            container_disk_gb=40,
            evidence="measured",
            note="Weekly bounded QLoRA and CUDA GSPO verification target.",
        ),
    ),
    ModelSupport(
        model="Qwen/Qwen3.8-27B",
        tier="default",
        certification="training-verified",
        recommended_provider="runpod",
        runpod=RunPodResources(
            gpu=_H100,
            gpu_count=1,
            container_disk_gb=160,
            evidence="measured",
            note="Live QLoRA run returned an adapter and passed held-out checks.",
        ),
    ),
    ModelSupport(
        model="meta-models/Muse-Glimmer-30B",
        tier="compatibility",
        certification="serving-verified",
        recommended_provider="runpod",
        runpod=RunPodResources(
            gpu=_H100,
            gpu_count=1,
            container_disk_gb=160,
            evidence="measured",
            note="Live training, artifact retrieval, and adapter chat verified.",
        ),
    ),
    ModelSupport(
        model="Qwen/Qwen3.8-Flash-Next",
        aliases=("Qwen/Qwen3.8-Flash-Next-FP8",),
        tier="frontier-preview",
        certification="smoke-tested",
        recommended_provider="runpod",
        runpod=RunPodResources(
            gpu=_H100,
            gpu_count=4,
            container_disk_gb=400,
            evidence="estimated",
            note=(
                "Planning estimate from checkpoint scale and official four-way "
                "serving examples; successful StateSet training is not yet proven."
            ),
        ),
    ),
    ModelSupport(
        model="zai-org/GLM-5.3-Flash",
        tier="frontier-preview",
        certification="smoke-tested",
        recommended_provider="runpod",
        runpod=RunPodResources(
            gpu=_H100,
            gpu_count=8,
            container_disk_gb=800,
            evidence="estimated",
            note=(
                "Conservative planning estimate for the 320B MoE checkpoint; "
                "capacity and a successful StateSet run are not yet proven."
            ),
        ),
    ),
    ModelSupport(
        model="Qwen/Qwen3.5-9B",
        tier="compatibility",
        certification="training-verified",
        recommended_provider="river",
    ),
    ModelSupport(
        model="Qwen/Qwen3-Coder-30B-A3B-Instruct",
        tier="compatibility",
        certification="smoke-tested",
        recommended_provider="runpod",
    ),
    ModelSupport(
        model="openai/gpt-oss-20b",
        aliases=("openai/gpt-oss-120b",),
        tier="compatibility",
        certification="smoke-tested",
        recommended_provider="runpod",
    ),
    ModelSupport(
        model="deepseek-ai/DeepSeek-V4-Flash",
        aliases=("deepseek-ai/DeepSeek-V4-Flash-Base",),
        tier="compatibility",
        certification="smoke-tested",
        recommended_provider="runpod",
    ),
    ModelSupport(
        model="google/gemma-4-31B-it",
        tier="compatibility",
        certification="smoke-tested",
        recommended_provider="runpod",
    ),
    ModelSupport(
        model="moonshotai/Kimi-K2.6",
        tier="compatibility",
        certification="smoke-tested",
        recommended_provider="runpod",
    ),
    ModelSupport(
        model="moonshotai/Kimi-K3",
        tier="compatibility",
        certification="configured",
        recommended_provider="runpod",
    ),
    ModelSupport(
        model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16",
        tier="compatibility",
        certification="smoke-tested",
        recommended_provider="runpod",
    ),
    ModelSupport(
        model="zai-org/GLM-5.1",
        tier="compatibility",
        certification="smoke-tested",
        recommended_provider="runpod",
    ),
    ModelSupport(
        model="zai-org/GLM-5.2",
        tier="compatibility",
        certification="smoke-tested",
        recommended_provider="runpod",
    ),
)


def get_model_support(model: str) -> ModelSupport | None:
    """Resolve a canonical model id or documented alias, case-insensitively."""
    wanted = model.strip().casefold()
    for entry in MODEL_CATALOG:
        if wanted == entry.model.casefold() or any(
            wanted == alias.casefold() for alias in entry.aliases
        ):
            return entry
    return None


def model_catalog() -> list[dict[str, object]]:
    """Return the catalog in deterministic display order."""
    return [entry.to_dict() for entry in MODEL_CATALOG]


def plan_runpod_resources(
    model: str,
    *,
    gpu: str | None = None,
    gpu_count: int | None = None,
    container_disk_gb: int | None = None,
) -> RunPodPlan:
    """Build a non-billable RunPod plan, preserving explicit overrides.

    Unknown Hugging Face models deliberately fall back to RunPod's small-job
    defaults and require manual review. A generic loader accepting a model is
    not evidence that its checkpoint fits on a particular GPU topology.
    """
    entry = get_model_support(model)
    recommendation = entry.runpod if entry is not None else None
    planned_gpu = gpu or (recommendation.gpu if recommendation else "NVIDIA RTX A4000")
    planned_count = gpu_count or (recommendation.gpu_count if recommendation else 1)
    planned_disk = container_disk_gb or (
        recommendation.container_disk_gb if recommendation else 40
    )
    return {
        "provider": "runpod",
        "model": entry.model if entry else model,
        "tier": entry.tier if entry else "compatibility",
        "certification": entry.certification if entry else "configured",
        "gpu": planned_gpu,
        "gpu_count": planned_count,
        "container_disk_gb": planned_disk,
        "recommendation_evidence": (
            recommendation.evidence if recommendation else "unknown"
        ),
        "manual_review_required": (
            recommendation is None or recommendation.evidence != "measured"
        ),
        "explicit_overrides": {
            "gpu": gpu is not None,
            "gpu_count": gpu_count is not None,
            "container_disk_gb": container_disk_gb is not None,
        },
        "provisions_hardware": False,
        "note": (
            recommendation.note
            if recommendation
            else "Unknown model size; defaults are not a capacity guarantee."
        ),
    }


__all__ = [
    "CertificationStage",
    "MODEL_CATALOG",
    "ModelSupport",
    "RunPodPlan",
    "RunPodResources",
    "SupportTier",
    "get_model_support",
    "model_catalog",
    "plan_runpod_resources",
]
