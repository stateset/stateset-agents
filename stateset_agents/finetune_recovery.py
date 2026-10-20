"""Review and privately snapshot worker-published local recovery checkpoints.

Inspections hash regular files, never deserialize optimizer state or load models.
Recovery preserves the original training budget and never falls back to fresh
training. Preparation publishes a new run; its launch remains a separate action.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import shutil
import stat
import sys
import tempfile
from itertools import islice
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.finetune_workspace import (
    MAX_STUDIO_CHECKPOINTS,
    TERMINAL_STATES,
    FinetuneConfig,
    RunPlan,
    create_run,
    process_token,
    read_json,
    read_run_plan,
    read_training_data,
    run_status,
    verified_run_settings,
    write_json,
)
from stateset_agents.utils.process_probe import probe_process

MAX_CHECKPOINT_FILES = 4096
MAX_CHECKPOINT_BYTES = 1024**4
MAX_CHECKPOINTS = MAX_STUDIO_CHECKPOINTS
CHECKPOINT_NAME_PATTERN = r"(?:checkpoint-|stopped-step-)[1-9][0-9]{0,17}"
_NAME = re.compile(CHECKPOINT_NAME_PATTERN)
_RESUME_FIELDS = {
    "parent_run_id",
    "parent_plan_sha256",
    "checkpoint",
    "checkpoint_sha256",
    "kind",
    "step",
    "epoch",
    "recovery_sha256",
    "training_config_sha256",
    "dataset_sha256",
}


def _single_process() -> None:
    if not sys.platform.startswith("linux"):
        raise ValueError("Studio recovery currently verifies worker ownership on Linux")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        raise ValueError(
            "Studio checkpoint recovery requires a single training process"
        )


def _directory(path: Path) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError("Recovery requires regular checkpoint directories")


def _checkpoint(run: Path, name: str) -> Path:
    if not isinstance(name, str) or not _NAME.fullmatch(name):
        raise ValueError("Choose a published checkpoint name")
    _directory(run)
    _directory(run / "artifacts")
    path = run / "artifacts" / name
    _directory(path)
    return path


def _inventory(
    path: Path,
    *,
    copy_to: Path | None = None,
    max_bytes: int | None = None,
    max_files: int | None = None,
) -> dict[str, Any]:
    """Hash stable bounded regular files, optionally copying into a private tree."""
    _directory(path)
    files: dict[str, str] = {}
    total = 0
    byte_limit = MAX_CHECKPOINT_BYTES if max_bytes is None else max_bytes
    file_limit = MAX_CHECKPOINT_FILES if max_files is None else max_files
    if copy_to is not None:
        copy_to.mkdir(mode=0o700)
    for item in sorted(path.rglob("*")):
        if item.is_symlink():
            raise ValueError("Checkpoint links are unsupported")
        if item.is_dir():
            continue
        if len(files) >= file_limit:
            raise ValueError("Checkpoint file count exceeds the recovery limit")
        relative = item.relative_to(path)
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        with os.fdopen(os.open(item, flags), "rb") as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("Checkpoint files must be regular files")
            total += before.st_size
            if total > byte_limit:
                raise ValueError("Checkpoint bytes exceed the recovery limit")
            destination = None
            if copy_to is not None:
                target = copy_to / relative
                target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
                destination = os.fdopen(
                    os.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb"
                )
            digest = hashlib.sha256()
            try:
                remaining = before.st_size
                while remaining:
                    chunk = source.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError("Checkpoint changed during inspection")
                    remaining -= len(chunk)
                    digest.update(chunk)
                    if destination is not None:
                        destination.write(chunk)
                after = os.fstat(source.fileno())
                if source.read(1) or (before.st_size, before.st_mtime_ns) != (
                    after.st_size,
                    after.st_mtime_ns,
                ):
                    raise ValueError("Checkpoint changed during inspection")
                if destination is not None:
                    destination.flush()
                    os.fsync(destination.fileno())
            finally:
                if destination is not None:
                    destination.close()
            files[relative.as_posix()] = digest.hexdigest()
    if not files:
        raise ValueError("Checkpoint has no files")
    return {"sha256": content_hash(files), "bytes": total, "files": files}


def _shape(path: Path, kind: str, step: int, epoch: Any, files: dict[str, str]) -> None:
    if type(step) is not int or step < 1:
        raise ValueError("Checkpoint has no committed training step")
    if not any(
        name.endswith((".safetensors", ".bin")) and (path / name).stat().st_size > 0
        for name in files
    ):
        raise ValueError("Checkpoint is missing model weights")
    if kind == "native":
        marker = read_json(path / ".stateset-checkpoint.json", max_bytes=512 * 1024)
        if (
            set(marker) != {"schema_version", "files"}
            or type(marker.get("schema_version")) is not int
            or marker["schema_version"] != 1
            or marker.get("files")
            != {
                name: digest
                for name, digest in files.items()
                if name != ".stateset-checkpoint.json"
            }
            or not {"training_state.pt", "resolved_config.json"} <= files.keys()
            or epoch is not None
        ):
            raise ValueError("Native checkpoint publication is incomplete")
    elif kind == "sft":
        if (
            not {
                "trainer_state.json",
                "optimizer.pt",
                "scheduler.pt",
                "rng_state.pth",
                "adapter_config.json",
            }
            <= files.keys()
        ):
            raise ValueError(
                "SFT checkpoint lacks complete optimizer/scheduler/RNG state"
            )
        state = read_json(path / "trainer_state.json", max_bytes=512 * 1024)
        if (
            type(state.get("global_step")) is not int
            or state["global_step"] != step
            or type(epoch) not in {int, float}
            or not math.isfinite(epoch)
            or epoch < 0
            or state.get("epoch") != epoch
        ):
            raise ValueError("SFT checkpoint progress is invalid")
    else:
        raise ValueError("Unsupported recovery checkpoint kind")


def publish_checkpoint(run: Path, checkpoint: Path, *, step: int) -> dict[str, Any]:
    """Publish a plan-bound recovery receipt after an own-worker save completes."""
    _single_process()
    plan, plan_sha = read_run_plan(run)
    if plan["config"]["provider"] != "local":
        raise ValueError("Only local workers publish studio recovery receipts")
    expected = _checkpoint(run, checkpoint.name)
    if expected.resolve() != checkpoint.resolve():
        raise ValueError("Checkpoint must belong to this worker's artifact tree")
    kind = "sft" if plan["config"]["method"] == "sft" else "native"
    inventory = _inventory(checkpoint)
    epoch = (
        read_json(checkpoint / "trainer_state.json", max_bytes=512 * 1024).get("epoch")
        if kind == "sft"
        else None
    )
    _shape(checkpoint, kind, step, epoch, inventory["files"])
    receipt = {
        "schema_version": 1,
        "parent_plan_sha256": plan_sha,
        "checkpoint": checkpoint.name,
        "checkpoint_sha256": inventory["sha256"],
        "kind": kind,
        "step": step,
        "epoch": epoch,
        "training_config_sha256": content_hash(plan["config"]),
        "dataset_sha256": plan["dataset_sha256"],
    }
    directory = run / "recovery"
    if directory.is_symlink():
        raise ValueError("Recovery receipt directory must be regular")
    directory.mkdir(exist_ok=True, mode=0o700)
    write_json(directory / (checkpoint.name + ".json"), receipt)
    return receipt


def _parent(
    run: Path, *, verify_implementation: bool = True
) -> tuple[dict[str, Any], FinetuneConfig, str]:
    _single_process()
    if verify_implementation:
        plan, config = verified_run_settings(run)
    else:
        plan, _ = read_run_plan(run)
        config = FinetuneConfig(**plan["config"])
        config.validate()
    if config.provider != "local":
        raise ValueError(
            "Remote checkpoint recovery requires provider-specific controls"
        )
    if (
        not re.fullmatch(r"[0-9a-f]{40}", config.revision)
        or config.revision == "0" * 40
    ):
        raise ValueError("Checkpoint recovery requires a pinned nonzero model commit")
    state = run_status(run)
    if state.get("status") not in TERMINAL_STATES:
        raise ValueError("Wait for the parent run to finish before recovery")
    owner_present = False
    for filename in ("supervisor.json", "execution.json"):
        if (run / filename).exists() or (run / filename).is_symlink():
            owner_present = True
            owner = read_json(run / filename)
            pid, token = owner.get("pid"), owner.get("process_token")
            if type(pid) is not int or pid < 1 or not isinstance(token, str):
                raise ValueError("Cannot verify the parent worker's lifecycle")
            current = process_token(pid)
            if current is None:
                try:
                    probe_process(pid)
                except ProcessLookupError:
                    continue
                except PermissionError:
                    raise ValueError(
                        "Cannot verify the parent worker's lifecycle"
                    ) from None
                raise ValueError("Cannot verify whether the parent worker has exited")
            if current == token:
                raise ValueError("Wait for the parent worker and supervisor to exit")
    if not owner_present and (
        state.get("status") == "interrupted"
        or state.get("execution_unconfirmed") is True
    ):
        raise ValueError("Cannot verify the parent worker's lifecycle")
    return plan, config, read_run_plan(run)[1]


def _receipt(
    run: Path, name: str, plan: dict[str, Any], config: FinetuneConfig, plan_sha: str
) -> dict[str, Any]:
    """Check published metadata without hashing or deserializing training state."""
    _checkpoint(run, name)
    _directory(run / "recovery")
    receipt = read_json(run / "recovery" / (name + ".json"), max_bytes=4096)
    kind = "sft" if config.method == "sft" else "native"
    if (
        set(receipt)
        != {
            "schema_version",
            "parent_plan_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "kind",
            "step",
            "epoch",
            "training_config_sha256",
            "dataset_sha256",
        }
        or type(receipt.get("schema_version")) is not int
        or receipt["schema_version"] != 1
        or receipt["parent_plan_sha256"] != plan_sha
        or receipt["checkpoint"] != name
        or receipt["kind"] != kind
        or receipt["training_config_sha256"] != content_hash(plan["config"])
        or receipt["dataset_sha256"] != plan["dataset_sha256"]
    ):
        raise ValueError("Checkpoint receipt differs from the parent plan")
    if (
        type(receipt["step"]) is not int
        or receipt["step"] < 1
        or int(name.rsplit("-", 1)[1]) != receipt["step"]
    ):
        raise ValueError("Checkpoint progress differs from its published name")
    if not isinstance(receipt["checkpoint_sha256"], str) or not re.fullmatch(
        r"[0-9a-f]{64}", receipt["checkpoint_sha256"]
    ):
        raise ValueError("Checkpoint receipt has no valid inventory checksum")
    return receipt


def _selected(
    run: Path,
    name: str,
    *,
    require_remaining: bool = True,
    verify_implementation: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    plan, config, plan_sha = _parent(run, verify_implementation=verify_implementation)
    path = _checkpoint(run, name)
    receipt = _receipt(run, name, plan, config, plan_sha)
    kind = receipt["kind"]
    inventory = _inventory(path)
    _shape(path, kind, receipt["step"], receipt["epoch"], inventory["files"])
    if int(name.rsplit("-", 1)[1]) != receipt["step"]:
        raise ValueError("Checkpoint progress differs from its published name")
    if inventory["sha256"] != receipt["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes differ from the saved recovery receipt")
    remaining = (
        config.epochs - receipt["epoch"]
        if kind == "sft"
        else config.steps - receipt["step"]
    )
    if remaining < 0 or (require_remaining and remaining <= 0):
        raise ValueError("Checkpoint already exhausted the original training budget")
    return plan, {
        **receipt,
        "parent_run_id": run.name,
        "recovery_sha256": content_hash(receipt),
        "bytes": inventory["bytes"],
        "file_count": len(inventory["files"]),
        "remaining": remaining,
        "budget_unit": "epochs" if kind == "sft" else "updates",
    }


def inspect_evaluation_checkpoint(run: Path, name: str) -> dict[str, Any]:
    """Verify a published checkpoint for inference, including exhausted budgets.

    Optimizer state is hashed, never loaded. Inference does not restore a
    trainer, so it requires the published protocol rather than matching current
    trainer implementation. Recovery keeps its stricter original requirements.
    """
    _, row = _selected(run, name, require_remaining=False, verify_implementation=False)
    return {**row, "path": str(_checkpoint(run, name))}


def list_evaluation_checkpoints(run: Path) -> list[dict[str, Any]]:
    """List bounded receipt metadata; selection must later verify actual bytes."""
    plan, config, plan_sha = _parent(run, verify_implementation=False)
    directory = run / "recovery"
    if not directory.exists():
        if directory.is_symlink():
            raise ValueError("Recovery receipt directory must be regular")
        return []
    _directory(directory)
    paths = list(islice(directory.iterdir(), MAX_CHECKPOINTS + 1))
    if len(paths) > MAX_CHECKPOINTS:
        raise ValueError("Recovery receipt count exceeds the inspection limit")
    rows = []
    for path in paths:
        name = path.name.removesuffix(".json")
        if path.suffix != ".json" or not _NAME.fullmatch(name):
            continue
        try:
            receipt = _receipt(run, name, plan, config, plan_sha)
            location = _checkpoint(run, name)
            available = all(
                (location / part).is_file() and not (location / part).is_symlink()
                for part in ["adapter_config.json", "adapter_model.safetensors"]
            )
            rows.append(
                {
                    "checkpoint": name,
                    "step": receipt["step"],
                    "kind": receipt["kind"],
                    "available": available,
                    "bytes_verified": False,
                }
            )
        except (
            OSError,
            ValueError,
            KeyError,
            TypeError,
            OverflowError,
            RecursionError,
        ):
            rows.append(
                {"checkpoint": name, "available": False, "bytes_verified": False}
            )
    return sorted(rows, key=lambda row: row.get("step", 0), reverse=True)


def inspect_checkpoints(run: Path) -> dict[str, Any]:
    """Inspect up to 100 published checkpoints without loading optimizer state."""
    plan, _, _ = _parent(run)
    directory = run / "recovery"
    if directory.is_symlink():
        raise ValueError("Recovery receipts cannot be symlinked")
    rows = []
    if directory.exists():
        _directory(directory)
        paths = sorted(directory.iterdir())
        if len(paths) > MAX_CHECKPOINTS:
            raise ValueError("Recovery receipt count exceeds the inspection limit")
        for path in paths:
            name = path.name.removesuffix(".json")
            try:
                _, row = _selected(run, name)
                rows.append({**row, "available": True})
            except (
                OSError,
                ValueError,
                TypeError,
                KeyError,
                RecursionError,
                OverflowError,
            ):
                rows.append(
                    {
                        "checkpoint": (
                            name if _NAME.fullmatch(name) else "invalid record"
                        ),
                        "available": False,
                        "reason": "Cannot verify this checkpoint's complete state, receipt or remaining budget.",
                    }
                )
    rows.sort(key=lambda row: row.get("step", 0), reverse=True)
    return {
        "schema_version": 1,
        "parent_run_id": run.name,
        "parent_plan_sha256": read_run_plan(run)[1],
        "method": plan["config"]["method"],
        "checkpoints": rows,
        "prepared": False,
        "training_started": False,
        "notes": [
            "Only own-worker, plan-bound local checkpoints are eligible.",
            "Preparation copies a checkpoint and retained dataset into a new private run; launch needs a separate review.",
            "The original total budget and training settings are preserved. This does not certify resumed model quality.",
        ],
    }


def prepare_resume(
    run: Path, checkpoint: str, *, expected_recovery_sha256: str
) -> Path:
    """Privately copy exact reviewed recovery bytes and publish a new prepared run."""
    plan, row = _selected(run, checkpoint)
    if row["recovery_sha256"] != expected_recovery_sha256:
        raise ValueError("Recovery checksum differs from the reviewed checkpoint")
    resume = {key: row[key] for key in _RESUME_FIELDS}
    config = FinetuneConfig(**plan["config"])
    data = read_training_data(run / "dataset")
    if hashlib.sha256(data).hexdigest() != plan["dataset_sha256"]:
        raise ValueError("Parent dataset changed during recovery review")
    if shutil.disk_usage(run.parent).free < row["bytes"] + len(data) + 64 * 1024**2:
        raise ValueError("Insufficient disk headroom for a private recovery snapshot")
    prepared = RunPlan(
        config,
        Path(plan["cwd"]),
        data,
        plan["examples"],
        plan["resources"],
        tuple(plan["notes"])
        + (
            "Resumes the reviewed checkpoint with the original total training budget; no fresh-training fallback.",
        ),
        plan["package_implementation_sha256"],
        resume,
    )
    staging = Path(tempfile.mkdtemp(prefix=".resume-", dir=run.parent))
    try:
        child = create_run(staging, prepared)
        copied = _inventory(
            _checkpoint(run, checkpoint), copy_to=child / "resume-checkpoint"
        )
        if (
            copied["sha256"] != row["checkpoint_sha256"]
            or read_run_plan(run)[1] != row["parent_plan_sha256"]
        ):
            raise ValueError("Recovery inputs changed while copying")
        verify_resume_input(child, prepared.describe())
        target = run.parent / child.name
        if target.exists() or target.is_symlink():
            raise ValueError("Recovery run destination already exists")
        child.rename(target)
        return target.resolve()
    finally:
        shutil.rmtree(staging)


def verify_resume_input(run: Path, plan: dict[str, Any]) -> Path:
    """Verify the private exact checkpoint before launch and again before training."""
    _single_process()
    resume = plan.get("resume")
    if (
        plan["config"]["provider"] != "local"
        or not isinstance(resume, dict)
        or set(resume) != _RESUME_FIELDS
        or not isinstance(resume.get("parent_run_id"), str)
        or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", resume["parent_run_id"]
        )
        or not isinstance(resume.get("checkpoint"), str)
        or not _NAME.fullmatch(resume["checkpoint"])
        or any(
            not isinstance(resume.get(key), str)
            or not re.fullmatch(r"[0-9a-f]{64}", resume[key])
            for key in (
                "parent_plan_sha256",
                "checkpoint_sha256",
                "recovery_sha256",
                "training_config_sha256",
                "dataset_sha256",
            )
        )
    ):
        raise ValueError("Invalid recovery lineage")
    receipt = {
        key: resume[key]
        for key in (
            "parent_plan_sha256",
            "checkpoint",
            "checkpoint_sha256",
            "kind",
            "step",
            "epoch",
            "training_config_sha256",
            "dataset_sha256",
        )
    }
    receipt["schema_version"] = 1
    if content_hash(receipt) != resume["recovery_sha256"]:
        raise ValueError("Recovery lineage differs from its reviewed receipt")
    if (
        content_hash(plan["config"]) != resume["training_config_sha256"]
        or plan["dataset_sha256"] != resume["dataset_sha256"]
    ):
        raise ValueError(
            "Recovery training settings or dataset differ from the saved checkpoint"
        )
    kind = "sft" if plan["config"]["method"] == "sft" else "native"
    if resume["kind"] != kind:
        raise ValueError("Recovery method differs from the reviewed run")
    checkpoint = run / "resume-checkpoint"
    inventory = _inventory(checkpoint)
    _shape(checkpoint, kind, resume["step"], resume["epoch"], inventory["files"])
    if inventory["sha256"] != resume["checkpoint_sha256"]:
        raise ValueError("Recovery snapshot differs from the reviewed checkpoint")
    remaining = (
        plan["config"]["epochs"] - resume["epoch"]
        if kind == "sft"
        else plan["config"]["steps"] - resume["step"]
    )
    if remaining <= 0:
        raise ValueError("Recovery has no remaining training budget")
    return checkpoint
