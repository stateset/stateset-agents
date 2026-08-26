"""Provider-agnostic description of a remote fine-tune job.

For SFT, ``RemoteJobSpec`` mirrors ``stateset_agents.training.sft``'s argparse
surface. Harvest and RL reuse the same transport/resource envelope, while
providers declare which modes they can execute. That capability check keeps a
mode from silently changing meaning on a different provider.

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

#: Execution modes understood by the remote job contract. Providers support
#: different subsets; :class:`RemoteExecutor` checks that boundary before any
#: dataset upload or compute allocation happens.
JOB_KINDS = frozenset({"sft", "harvest", "rl"})

#: Fields that configure the *provider*, not the training script. These are
#: excluded from ``to_cli_args()``.
_RESOURCE_FIELDS = frozenset(
    {
        "gpu",
        "gpu_count",
        "timeout_s",
        "package_version",
        "container_disk_gb",
        "cloud_type",
        "network_volume_id",
        "max_cost_usd",
    }
)

#: RunPod pod pools. SECURE is reserved capacity; COMMUNITY is spot-priced —
#: markedly cheaper, but the pod can be reclaimed mid-job.
_CLOUD_TYPES = frozenset({"SECURE", "COMMUNITY"})

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
    "gpu_count",
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

    Serializable, although handle durability is provider-specific. Fireworks
    job ids can be polled from a later process; the local, Modal, River, and
    RunPod executors retain some or all job metadata in memory and therefore
    require the submitting process for parts of the lifecycle (especially
    artifact fetching).
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
    #: Resume from the newest ``checkpoint-*`` directory already present in
    #: ``output_dir``, when one exists (otherwise the job logs it and trains
    #: fresh). Only useful when the worker can actually see prior checkpoints:
    #: local reruns, a manual rerun on a pod that kept its disk, or a RunPod
    #: pod with a ``network_volume_id`` mounted (the checkpoints live on the
    #: volume). A fresh RunPod pod without a volume starts with an empty
    #: output dir, so this is a no-op there.
    resume: bool = False
    #: Prompts to compare base-vs-tuned after training. When set, the job
    #: generates a completion per prompt with the base model before LoRA is
    #: applied and again with the trained adapter, and writes
    #: ``eval_results.json`` into the output directory. An entry is a plain
    #: prompt string, or a spec dict ``{"prompt": str, "expect": [substr],
    #: "forbid": [substr], "judge": str, "min_judge_score": float}`` whose
    #: assertions gate the job's exit code (checked against the finetuned
    #: completion, case-insensitively, after the adapter is saved).
    eval_prompts: list[str | dict[str, Any]] | None = None
    #: Which packaged module the worker runs. ``"sft"`` (the default) trains
    #: on ``dataset``; ``"harvest"`` runs best-of-N rejection sampling
    #: (:mod:`stateset_agents.training.harvest`) where ``dataset`` is a JSON
    #: file of ``{prompt, expect, forbid}`` specs and ``harvest`` carries the
    #: sampling knobs. One spec shape for both, because they rent the same
    #: hardware the same way.
    job_kind: str = "sft"
    #: Harvest-only knobs: ``{"adapter_dir": str | None, "best_of": int,
    #: "temperature": float, "top_p": float, "max_new_tokens": int}``.
    #: ``adapter_dir`` is a LOCAL adapter directory (the current generation)
    #: that the executor ships to the worker.
    harvest: dict[str, Any] | None = None
    #: Token budget per eval completion. 90 suits direct-answer models;
    #: reasoning models (thinking preamble before the answer) need more.
    eval_max_new_tokens: int = 90

    # --- Provider resources: never reach the training script ---------------
    #: GPU to request. Deliberately has no default: GPU names are provider
    #: vocabulary ("A10G" on Modal, "NVIDIA RTX A4000" on RunPod), so a shared
    #: default would silently send an invalid id to whichever provider did not
    #: coin it. ``None`` means "use the executor's own default".
    gpu: str | None = None
    #: How many GPUs of that type to attach to the pod — RunPod only.
    #: The training job shards the model across every visible GPU via
    #: ``device_map="auto"`` when more than one is present, which is what
    #: lets a checkpoint bigger than one card train at all.
    gpu_count: int = 1
    timeout_s: int = 3600
    package_version: str | None = None
    #: GPU-pool container disk (GiB) for the model download — RunPod only.
    #: Size it at roughly 2.5x the checkpoint: a 63GB BF16 checkpoint dies
    #: mid-download on the old fixed 40GB (verified live). ``None`` means
    #: "use the executor's own default".
    container_disk_gb: int | None = None
    #: RunPod pod pool: "SECURE" (default, reserved capacity) or "COMMUNITY"
    #: (spot-priced — noticeably cheaper, but interruptible). RunPod only.
    cloud_type: str = "SECURE"
    #: RunPod network volume to mount at the pod's workspace. Checkpoints
    #: then land on durable storage that outlives the pod, so when a pod dies
    #: mid-job the executor reruns *with* ``--resume`` instead of from
    #: scratch — an interruption costs at most one epoch. Volumes are
    #: datacenter-scoped: pod creation is pinned to the volume's datacenter.
    #: RunPod only. ``None`` (default) keeps the ephemeral-disk behaviour.
    network_volume_id: str | None = None
    #: Refuse to run if the pod's worst case (its full ``timeout_s`` at the
    #: provider's quoted hourly rate) would exceed this many dollars. None
    #: means no ceiling — the historical behavior.
    max_cost_usd: float | None = None

    #: Adapter this run descends from, recorded in the trained adapter's
    #: manifest so the improvement loop's generations stay linked.
    parent_adapter: str | None = None

    def __post_init__(self) -> None:
        self.dataset = Path(self.dataset)
        self.output_dir = Path(self.output_dir)

        self.job_kind = str(self.job_kind).strip().lower()
        if self.job_kind not in JOB_KINDS:
            raise ValueError(
                f"job_kind must be one of {sorted(JOB_KINDS)}, "
                f"got {self.job_kind!r}"
            )

        if not self.dataset.exists():
            raise ValueError(f"dataset does not exist: {self.dataset}")
        if not self.base_model.strip():
            raise ValueError("base_model must be a non-empty model name")

        for name in _POSITIVE_FIELDS:
            value = getattr(self, name)
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value!r}")

        if self.eval_prompts is not None:
            # Validate spec entries at submit time — on this machine, before
            # a GPU is rented — with the same rules the job applies. Episode
            # scripts (multi-turn: {"turns", "turn_expect", "forbid"}) have
            # their own shape and are validated by the episode consumer.
            from stateset_agents.training.sft import normalize_eval_prompts

            plain = [
                e
                for e in self.eval_prompts
                if not (isinstance(e, dict) and "turns" in e)
            ]
            if plain:
                normalize_eval_prompts(plain)

        if self.container_disk_gb is not None and self.container_disk_gb <= 0:
            raise ValueError(
                f"container_disk_gb must be positive, got {self.container_disk_gb!r}"
            )

        if self.network_volume_id is not None and not self.network_volume_id.strip():
            raise ValueError("network_volume_id must be non-empty when set")
        if self.max_cost_usd is not None and self.max_cost_usd <= 0:
            raise ValueError("max_cost_usd must be positive when set")

        self.cloud_type = str(self.cloud_type).upper()
        if self.cloud_type not in _CLOUD_TYPES:
            raise ValueError(
                f"cloud_type must be one of {sorted(_CLOUD_TYPES)}, "
                f"got {self.cloud_type!r}"
            )

    def harvest_cli_args(self, *, adapter_dir: str | None = None) -> list[str]:
        """Render as ``stateset_agents.training.harvest`` arguments.

        ``adapter_dir`` overrides the harvest dict's local path — executors
        pass the REMOTE directory they shipped the adapter to.
        """
        knobs = self.harvest or {}
        args = [
            "--base-model",
            self.base_model,
            "--prompts-file",
            str(self.dataset),
            "--output-dir",
            str(self.output_dir),
            "--best-of",
            str(knobs.get("best_of", 8)),
            "--temperature",
            str(knobs.get("temperature", 0.9)),
            "--top-p",
            str(knobs.get("top_p", 0.95)),
            "--max-new-tokens",
            str(knobs.get("max_new_tokens", 300)),
            "--eval-max-new-tokens",
            str(self.eval_max_new_tokens),
        ]
        adapter = adapter_dir or knobs.get("adapter_dir")
        if adapter:
            args += ["--adapter", str(adapter)]
        if self.eval_prompts:
            args += ["--eval-prompts-json", json.dumps(self.eval_prompts)]
        if self.dry_run:
            args.append("--dry-run")
        return args

    def to_cli_args(self) -> list[str]:
        """Render as command-line arguments for the job's module."""
        if self.job_kind == "harvest":
            return self.harvest_cli_args()
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
        if self.resume:
            args.append("--resume")
        if self.eval_prompts:
            args += ["--eval-prompts-json", json.dumps(self.eval_prompts)]
            args += ["--eval-max-new-tokens", str(self.eval_max_new_tokens)]
        if self.parent_adapter:
            args += ["--parent-adapter", str(self.parent_adapter)]
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
    #: Measured pod lifetime and spend. None means the provider quoted no
    #: price — unknown, never free.
    duration_s: float | None = None
    cost_usd: float | None = None

    @property
    def succeeded(self) -> bool:
        return self.status is JobStatus.SUCCEEDED
