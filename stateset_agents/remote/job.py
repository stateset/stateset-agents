"""Provider-agnostic description of a remote fine-tune job.

``RemoteJobSpec`` mirrors ``scripts/sft_from_curated.py``'s argparse surface
exactly. That script is the frozen job contract: if a capability is missing,
the fix is a change to the script, not a new field here. Keeping the two in
lockstep is what lets every executor shell out to the same entrypoint instead
of reimplementing training.

Secrets are deliberately absent. ``HF_TOKEN`` and provider credentials are
read from the environment at submit time and never serialized.
"""

from __future__ import annotations

import enum
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["JobHandle", "JobStatus", "RemoteJobResult", "RemoteJobSpec"]

#: Fields that configure the *provider*, not the training script. These are
#: excluded from ``to_cli_args()``.
_RESOURCE_FIELDS = frozenset(
    {"gpu", "timeout_s", "package_version", "container_disk_gb"}
)

#: Positive-valued hyperparameters, validated uniformly in ``__post_init__``.
_POSITIVE_FIELDS = (
    "num_epochs",
    "lora_r",
    "lora_alpha",
    "learning_rate",
    "max_length",
    "per_device_batch_size",
    "gradient_accumulation_steps",
    "eval_max_new_tokens",
    "timeout_s",
)


class JobStatus(enum.Enum):
    """Lifecycle state of a remote job."""

    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        """True when no further state transition will occur."""
        return self in (JobStatus.SUCCEEDED, JobStatus.FAILED, JobStatus.CANCELLED)


@dataclass(frozen=True)
class JobHandle:
    """Opaque pointer to a submitted job.

    Serializable, but note the current limit: both shipped executors run the
    job synchronously inside ``submit()`` and keep outcomes in memory, so a
    handle is only meaningful to the process that created it. Reconnecting to
    a job from a later process needs asynchronous submission (Modal's
    ``Function.spawn`` + ``FunctionCall.from_id``), which is not implemented.
    """

    provider: str
    job_id: str

    def to_dict(self) -> dict[str, Any]:
        return {"provider": self.provider, "job_id": self.job_id}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> JobHandle:
        return cls(provider=data["provider"], job_id=data["job_id"])


@dataclass
class RemoteJobSpec:
    """A fine-tune job, described independently of where it runs."""

    # --- Job contract: one field per sft_from_curated.py argument -----------
    dataset: Path
    base_model: str
    output_dir: Path = Path("outputs/sft_v1")
    num_epochs: int = 3
    lora_r: int = 16
    lora_alpha: int = 32
    learning_rate: float = 2e-5
    max_length: int = 1024
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    dry_run: bool = False
    #: Prompts to compare base-vs-tuned after training. When set, the job
    #: generates a completion per prompt with the base model before LoRA is
    #: applied and again with the trained adapter, and writes
    #: ``eval_results.json`` into the output directory.
    eval_prompts: list[str] | None = None
    #: Token budget per eval completion. 90 suits direct-answer models;
    #: reasoning models (thinking preamble before the answer) need more.
    eval_max_new_tokens: int = 90

    # --- Provider resources: never reach the training script ---------------
    #: GPU to request. Deliberately has no default: GPU names are provider
    #: vocabulary ("A10G" on Modal, "NVIDIA RTX A4000" on RunPod), so a shared
    #: default would silently send an invalid id to whichever provider did not
    #: coin it. ``None`` means "use the executor's own default".
    gpu: str | None = None
    timeout_s: int = 3600
    package_version: str | None = None
    #: GPU-pool container disk (GiB) for the model download — RunPod only.
    #: Size it at roughly 2.5x the checkpoint: a 63GB BF16 checkpoint dies
    #: mid-download on the old fixed 40GB (verified live). ``None`` means
    #: "use the executor's own default".
    container_disk_gb: int | None = None

    def __post_init__(self) -> None:
        self.dataset = Path(self.dataset)
        self.output_dir = Path(self.output_dir)

        if not self.dataset.exists():
            raise ValueError(f"dataset does not exist: {self.dataset}")
        if not self.base_model.strip():
            raise ValueError("base_model must be a non-empty model name")

        for name in _POSITIVE_FIELDS:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")

        if self.container_disk_gb is not None and self.container_disk_gb <= 0:
            raise ValueError(
                f"container_disk_gb must be positive, got {self.container_disk_gb!r}"
            )

    def to_cli_args(self) -> list[str]:
        """Render as ``sft_from_curated.py`` command-line arguments."""
        args = [
            "--dataset",
            str(self.dataset),
            "--base-model",
            self.base_model,
            "--output-dir",
            str(self.output_dir),
            "--num-epochs",
            str(self.num_epochs),
            "--lora-r",
            str(self.lora_r),
            "--lora-alpha",
            str(self.lora_alpha),
            "--learning-rate",
            str(self.learning_rate),
            "--max-length",
            str(self.max_length),
            "--per-device-batch-size",
            str(self.per_device_batch_size),
            "--gradient-accumulation-steps",
            str(self.gradient_accumulation_steps),
        ]
        if self.dry_run:
            args.append("--dry-run")
        if self.eval_prompts:
            args += ["--eval-prompts-json", json.dumps(self.eval_prompts)]
            args += ["--eval-max-new-tokens", str(self.eval_max_new_tokens)]
        return args

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["dataset"] = str(self.dataset)
        data["output_dir"] = str(self.output_dir)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RemoteJobSpec:
        return cls(**data)


@dataclass
class RemoteJobResult:
    """Terminal outcome of a remote job, as rendered by the CLI."""

    handle: JobHandle
    status: JobStatus
    output_dir: Path | None
    logs: list[str] = field(default_factory=list)

    @property
    def succeeded(self) -> bool:
        return self.status is JobStatus.SUCCEEDED
