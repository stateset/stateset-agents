"""Auditable model/provider verification claims.

Registration, unit coverage, hardware allocation, and successful inference are
different claims. Keeping them as structured data prevents documentation and
release messaging from silently promoting one into another.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

VerificationLevel = Literal[
    "unverified",
    "unit-tested",
    "hardware-started",
    "inference-verified",
]
VerificationOutcome = Literal["pass", "fail", "blocked", "not-run"]


@dataclass(frozen=True)
class ModelProviderEvidence:
    """One scoped claim about a model on a provider or in the framework."""

    model: str
    provider: str
    level: VerificationLevel
    outcome: VerificationOutcome
    checked_at: str | None
    evidence: str

    def to_dict(self) -> dict[str, str | None]:
        """Return a stable JSON-serializable representation."""
        return asdict(self)


MODEL_PROVIDER_EVIDENCE: tuple[ModelProviderEvidence, ...] = (
    ModelProviderEvidence(
        model="thinkingmachines/Inkling-Small",
        provider="tinker",
        level="unit-tested",
        outcome="pass",
        checked_at="2026-08-31",
        evidence=(
            "Tinker Datum construction, remote-autograd loop, optimizer steps, "
            "checkpoint pointers, and lifecycle are unit-pinned; no live claim."
        ),
    ),
    ModelProviderEvidence(
        model="Qwen/Qwen3.5-0.8B",
        provider="runpod",
        level="inference-verified",
        outcome="pass",
        checked_at="2026-08-27",
        evidence=(
            "Live bounded QLoRA and CUDA GSPO jobs passed held-out checks; "
            "artifacts returned and pods terminated. See docs/RELEASE_EVIDENCE.md."
        ),
    ),
    ModelProviderEvidence(
        model="Qwen/Qwen3.8-27B",
        provider="runpod",
        level="inference-verified",
        outcome="pass",
        checked_at="2026-08-05",
        evidence=(
            "Live H100 QLoRA run passed 2/2 held-out assertions, returned a "
            "467MB adapter, recorded $0.96 spend, and terminated the pod."
        ),
    ),
    ModelProviderEvidence(
        model="meta-models/Muse-Glimmer-30B",
        provider="runpod",
        level="inference-verified",
        outcome="pass",
        checked_at="2026-08-01",
        evidence=(
            "Live H100 training returned an adapter and adapter-backed chat "
            "resolved a held-out multi-turn support case."
        ),
    ),
    ModelProviderEvidence(
        model="Qwen/Qwen3.5-9B",
        provider="river",
        level="inference-verified",
        outcome="pass",
        checked_at="2026-08-18",
        evidence=(
            "Remote-autograd training completed and the resulting checkpoint "
            "answered 3/3 held-out tickets with the required resolutions."
        ),
    ),
    ModelProviderEvidence(
        model="Qwen/Qwen3.5-0.8B",
        provider="coreweave",
        level="unit-tested",
        outcome="pass",
        checked_at="2026-08-31",
        evidence=(
            "CKS Job, object-store transport, durable lifecycle, cleanup, and "
            "Dedicated Inference payloads are unit-pinned; no live claim."
        ),
    ),
    ModelProviderEvidence(
        model="Qwen/Qwen3.5-0.8B",
        provider="nebius",
        level="unit-tested",
        outcome="pass",
        checked_at="2026-08-31",
        evidence=(
            "Serverless AI job and endpoint lifecycles are unit-pinned; "
            "no live training or inference claim."
        ),
    ),
    ModelProviderEvidence(
        model="Qwen/Qwen3.8-Flash-Next",
        provider="framework",
        level="unit-tested",
        outcome="pass",
        checked_at="2026-08-26",
        evidence="Architecture registry, loader, dependency, and LoRA-target tests.",
    ),
    ModelProviderEvidence(
        model="Qwen/Qwen3.8-Flash-Next-FP8",
        provider="runpod",
        level="hardware-started",
        outcome="fail",
        checked_at="2026-08-26",
        evidence=(
            "Bounded 4x H100 attempts never reached container networking; "
            "no inference claim. See docs/RUNPOD_GUIDE.md."
        ),
    ),
    ModelProviderEvidence(
        model="zai-org/GLM-5.3-Flash",
        provider="framework",
        level="unit-tested",
        outcome="pass",
        checked_at="2026-08-26",
        evidence="Architecture registry, loader, dependency, and LoRA-target tests.",
    ),
    ModelProviderEvidence(
        model="zai-org/GLM-5.3-Flash",
        provider="runpod",
        level="unverified",
        outcome="blocked",
        checked_at="2026-08-26",
        evidence="Required multi-GPU capacity was unavailable; no pod was rented.",
    ),
)


def model_provider_evidence() -> list[dict[str, str | None]]:
    """Return all evidence rows in deterministic display order."""
    return [record.to_dict() for record in MODEL_PROVIDER_EVIDENCE]


__all__ = [
    "MODEL_PROVIDER_EVIDENCE",
    "ModelProviderEvidence",
    "VerificationLevel",
    "VerificationOutcome",
    "model_provider_evidence",
]
