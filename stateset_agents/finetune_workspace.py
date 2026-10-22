"""Durable, private fine-tuning workspaces shared by the TUI and its worker."""

from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Collection, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stateset_agents.finetune_remote_code import RemoteCodeSelection
from stateset_agents.utils.process_probe import probe_process, process_start_token

MAX_DATA_BYTES = 64 * 1024 * 1024
MAX_STUDIO_CHECKPOINTS = 100
METHODS = ("sft", "grpo", "gspo", "dapo")
PROVIDERS = ("local", "runpod", "modal", "river")
BACKENDS: dict[str, dict[str, Any]] = {
    "local": {
        "label": "Local CUDA",
        "methods": list(METHODS),
        "result_kind": "local_weights",
        "cost": "Your hardware; shared Buildr queue",
    },
    "runpod": {
        "label": "RunPod",
        "methods": ["sft"],
        "result_kind": "local_weights",
        "cost": "Enforced quoted pod-lifetime ceiling; ancillary costs separate",
    },
    "modal": {
        "label": "Modal",
        "methods": ["sft"],
        "result_kind": "local_weights",
        "cost": "Provider billed; no enforced dollar ceiling in this integration",
    },
    "river": {
        "label": "River",
        "methods": ["sft"],
        "result_kind": "hosted_pointer",
        "cost": "Provider billed; no enforced dollar ceiling in this integration",
    },
}


def redact_finetune_secrets(value: Any) -> Any:
    """Redact known credential values from a structured agent-facing response."""
    if isinstance(value, str):
        for name in (
            "RUNPOD_API_KEY",
            "RIVER_API_KEY",
            "MODAL_TOKEN_ID",
            "MODAL_TOKEN_SECRET",
            "MODAL_OAUTH_REFRESH_TOKEN",
            "MODAL_OAUTH_CLIENT_ID",
            "MODAL_OAUTH_CLIENT_SECRET",
            "MODAL_OAUTH_JWT_KEY",
            "HF_TOKEN",
            "HUGGING_FACE_HUB_TOKEN",
            "WANDB_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
        ):
            secret = os.environ.get(name)
            if secret and len(secret) >= 4:
                value = value.replace(secret, "[REDACTED]")
        return value
    if isinstance(value, list):
        return [redact_finetune_secrets(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_finetune_secrets(item) for key, item in value.items()}
    return value


def require_backend_credentials(provider: str) -> None:
    """Check required environment credentials without constructing a client.

    Modal resolves its own active profile, environment token pair or OAuth setup.
    No credential values are returned or serialized.
    """
    name = {"runpod": "RUNPOD_API_KEY", "river": "RIVER_API_KEY"}.get(provider)
    if name and not os.environ.get(name, "").strip():
        raise ValueError(f"{name} is missing from this session's environment")


#: Version of the ``stop.json`` receipt this package's workers honor. A plan
#: records it, so a later package can still stop that run's (older) worker as
#: long as it writes the same receipt version.
STOP_PROTOCOL_VERSION = 1

TERMINAL_STATES = frozenset({"succeeded", "failed", "interrupted", "cancelled"})
FAILURE_GUIDANCE = {
    "cancelled": (
        "Local stop request completed",
        "Use z to inspect published local recovery checkpoints, or c for a fresh run. Recovery requires matching code, complete state and a separate launch review.",
    ),
    "review_invalid": (
        "Reviewed inputs cannot be verified",
        "Check the model revision, dataset snapshot and package version. Reuse intact settings with c, then prepare a fresh p/r review.",
    ),
    "cuda_unavailable": (
        "CUDA training is unavailable",
        "Use CUDA hardware with the training extra installed. Check i for readiness before a fresh launch review.",
    ),
    "dependencies": (
        "A training dependency is unavailable",
        "Check the training extra and compatible package versions in the worker environment. Use i to inspect local readiness.",
    ),
    "out_of_memory": (
        "Training ran out of memory",
        "Check GPU and host RAM. Reduce context length or RL generations, or choose hardware with more memory, then review a new run.",
    ),
    "disk_full": (
        "Storage space or quota was exhausted",
        "Check workspace, model download and artifact storage, plus queue disk headroom. Restore enough space before reviewing a new run.",
    ),
    "timeout": (
        "Training exceeded its time limit",
        "Inspect the last reported progress and logs. Review the time and spending budget before changing the timeout for a new run.",
    ),
    "missing_weights": (
        "No usable training artifact was produced",
        "Inspect the trainer/export logs and artifact paths. River needs a valid hosted checkpoint receipt; other backends need weight files. A successful command alone does not establish a usable checkpoint.",
    ),
    "incomplete_updates": (
        "The requested RL optimizer updates were not completed",
        "Inspect optimizer and early-exit logs. An incomplete update budget is not a completed training run.",
    ),
    "provider_failed": (
        "Remote training did not complete",
        "Inspect the retained provider handle, provider status and artifact/download logs before considering a new paid launch.",
    ),
    "worker_exit": (
        "The training worker exited unsuccessfully",
        "Inspect run.log and the exit code. Reuse settings with c only after addressing the recorded failure.",
    ),
    "missing_receipt": (
        "The worker produced no completion receipt",
        "Inspect worker and export logs. A zero exit code without an artifact receipt is not successful training.",
    ),
    "supervisor_stopped": (
        "The run's supervisor was stopped",
        "The supervising process was interrupted (Ctrl+C, shutdown or a closed console) and stopped the worker it owned. Inspect run.log and supervisor.log, then prepare a fresh run.",
    ),
    "permissions": (
        "A file or executable permission was denied",
        "Check access to the workspace, dataset, output directory and configured queue executable before a fresh review.",
    ),
    "interrupted": (
        "Training execution is unconfirmed",
        "Supervisor or worker ownership could not be confirmed; detached work may still be queued or running. For a local run, review a cooperative stop before retrying. Reconnecting does not restart the worker.",
    ),
    "unknown": (
        "Fine-tuning failed",
        "Inspect run.log and supervisor.log for the original failure before changing settings or reviewing a new run.",
    ),
}
FAILURE_STAGES = frozenset({"review", "runtime", "training", "artifacts", "supervisor"})


class FinetuneFailure(RuntimeError):
    """A known failure with controlled, credential-free recovery guidance."""

    def __init__(self, code: str) -> None:
        if code not in FAILURE_GUIDANCE:
            raise ValueError("Unknown fine-tuning failure code")
        self.code = code
        super().__init__(FAILURE_GUIDANCE[code][0])


def failure_info(run: Path, state: dict[str, Any]) -> dict[str, Any] | None:
    """Read a bounded diagnosis without displaying untrusted exception messages."""
    if not isinstance(state.get("status"), str) or state["status"] not in {
        "failed",
        "interrupted",
        "cancelled",
    }:
        return None
    code = (
        state["status"]
        if state.get("status") in {"interrupted", "cancelled"}
        else "unknown"
    )
    stage, recorded = "unavailable", False
    try:
        record = read_json(run / "failure.json", max_bytes=16 * 1024)
        if (
            isinstance(record, dict)
            and type(record.get("schema_version")) is int
            and record["schema_version"] == 1
            and isinstance(record.get("code"), str)
            and record["code"] in FAILURE_GUIDANCE
            and isinstance(record.get("stage"), str)
            and record["stage"] in FAILURE_STAGES
        ):
            code, stage, recorded = record["code"], record["stage"], True
    except (OSError, ValueError, RecursionError):
        pass
    title, action = FAILURE_GUIDANCE[code]
    return {
        "code": code,
        "stage": stage,
        "recorded": recorded,
        "title": title,
        "action": action,
    }


@dataclass(frozen=True)
class FinetuneConfig:
    """Editable run settings; credentials are inherited only through the environment."""

    model: str = "Qwen/Qwen3.5-0.8B"
    revision: str = ""
    method: str = "sft"
    provider: str = "local"
    dataset: str = ""
    epochs: int = 3
    steps: int = 50
    learning_rate: float = 1e-5
    kl_coefficient: float = 0.1
    generations: int = 4
    lora_rank: int = 16
    context_length: int = 1024
    max_new_tokens: int = 192
    max_turns: int = 4
    max_cost_usd: float = 10.0
    timeout_seconds: int = 3600
    cpu_slots: int = 2
    memory_gb: float = 4.0
    scratch_gb: float = 5.0
    cost_policy: str = "require_cap"
    gpu: str = ""
    gpu_count: int = 0
    checkpoint_every: int = 0
    seed: int = 42

    def __post_init__(self) -> None:
        for name in (
            "model",
            "revision",
            "method",
            "provider",
            "dataset",
            "cost_policy",
            "gpu",
        ):
            if not isinstance(getattr(self, name), str):
                raise ValueError(f"{name} must be text")
        if self.method not in METHODS or self.provider not in PROVIDERS:
            raise ValueError("Choose SFT/GRPO/GSPO/DAPO and local/RunPod/Modal/River")
        if self.cost_policy not in {"require_cap", "provider_billed"}:
            raise ValueError("Choose require_cap or provider_billed cost policy")

    def validate(self) -> None:
        """Reject invalid settings using the shared human/agent configuration checks."""
        _, issues = check_config_values(asdict(self))
        if issues:
            raise ValueError(issues[0].message)


@dataclass(frozen=True)
class ConfigIssue:
    """One controlled, field-addressable configuration problem, without raw input."""

    field: str
    code: str
    message: str


def check_config_values(
    values: Mapping[str, Any], *, parse_text: bool = False
) -> tuple[dict[str, Any], list[ConfigIssue]]:
    """Return normalized values and all settings issues without reading data or APIs.

    Text parsing is opt-in for terminal edits. JSON callers retain strict numeric
    types; neither path coerces booleans into numbers. Messages do not echo input.
    Missing values use the same defaults as FinetuneConfig.
    """
    defaults = asdict(FinetuneConfig())
    parsed = {name: values.get(name, default) for name, default in defaults.items()}
    issues: list[ConfigIssue] = []
    seen: set[str] = set()

    def add(field: str, code: str, message: str) -> None:
        if field not in seen:
            issues.append(ConfigIssue(field, code, message))
            seen.add(field)

    if set(values) - defaults.keys():
        add(
            "config",
            "unknown_fields",
            "Unknown configuration fields; use finetune schema for supported fields.",
        )
    for name, default in defaults.items():
        value = parsed[name]
        if isinstance(default, str):
            if not isinstance(value, str):
                add(name, "invalid_type", f"{name} must be text")
        elif type(default) in {int, float}:
            if parse_text and isinstance(value, str):
                try:
                    value = int(value) if type(default) is int else float(value)
                    parsed[name] = value
                except (ValueError, OverflowError):
                    add(
                        name,
                        "invalid_type",
                        f"{name} must be {'an integer' if type(default) is int else 'a finite number'}",
                    )
            if (
                isinstance(value, bool)
                or (type(default) is int and type(value) is not int)
                or (type(default) is float and not isinstance(value, (int, float)))
            ):
                add(
                    name,
                    "invalid_type",
                    f"{name} must be {'an integer' if type(default) is int else 'a finite number'}",
                )

    method, provider = parsed["method"], parsed["provider"]
    method_valid = isinstance(method, str) and method in METHODS
    provider_valid = isinstance(provider, str) and provider in PROVIDERS
    if not method_valid:
        add(
            "method",
            "unsupported_method",
            "Choose SFT/GRPO/GSPO/DAPO and local/RunPod/Modal/River",
        )
    if not provider_valid:
        add(
            "provider",
            "unsupported_backend",
            "Choose SFT/GRPO/GSPO/DAPO and local/RunPod/Modal/River",
        )
    if isinstance(parsed["model"], str) and (
        not parsed["model"].strip() or parsed["model"].startswith("stub://")
    ):
        add("model", "invalid_model", "Choose a real Hugging Face model")
    if (
        provider_valid
        and provider in {"local", "runpod", "modal"}
        and isinstance(parsed["model"], str)
    ):
        from stateset_agents.core.frontier_models import (
            get_frontier_model,
            require_frontier_training,
        )

        try:
            frontier = get_frontier_model(parsed["model"].strip())
        except ValueError:
            pass
        else:
            try:
                require_frontier_training(frontier, generic_loader=True)
            except RuntimeError:
                if frontier.blockers:
                    add(
                        "model",
                        "blocked_training_layout",
                        "This checkpoint has an unsupported training layout for this backend; inspect frontier-models for blockers and choose a supported checkpoint.",
                    )
                else:
                    add(
                        "model",
                        "requires_frontier_recipe",
                        "This FP8 checkpoint requires the dedicated frontier-sft BF16 dequantization recipe; inspect frontier-sft --help or choose a supported Studio checkpoint.",
                    )
    if isinstance(parsed["dataset"], str) and not parsed["dataset"].strip():
        add("dataset", "dataset_required", "Set the dataset path first")
    revision = parsed["revision"]
    if (
        isinstance(revision, str)
        and revision
        and not re.fullmatch(r"[0-9a-f]{40}", revision)
    ):
        add(
            "revision",
            "invalid_revision",
            "Revision must be a full 40-character commit or blank",
        )
    if provider_valid and provider != "local":
        if method_valid and method != "sft":
            add(
                "method",
                "backend_method",
                f"Studio supports {BACKENDS[provider]['label']} SFT; choose local for native RL or the dedicated River RL workflow",
            )
        if isinstance(revision, str) and revision:
            add(
                "revision",
                "backend_revision",
                f"{BACKENDS[provider]['label']}'s SFT contract cannot enforce a revision; use local",
            )
    policy = parsed["cost_policy"]
    if not isinstance(policy, str) or policy not in {"require_cap", "provider_billed"}:
        add(
            "cost_policy",
            "invalid_cost_policy",
            "Choose require_cap or provider_billed cost policy",
        )
    if (
        isinstance(provider, str)
        and provider in {"modal", "river"}
        and policy != "provider_billed"
    ):
        add(
            "cost_policy",
            "uncapped_backend",
            "This backend has no enforced dollar ceiling; explicitly choose provider_billed cost policy before review",
        )
    seed = parsed["seed"]
    if type(seed) is int and not 0 <= seed <= 2**32 - 1:
        add(
            "seed",
            "invalid_range",
            "Training seed must be an integer in [0, 4294967295]",
        )
    gpu, count = parsed["gpu"], parsed["gpu_count"]
    cadence = parsed["checkpoint_every"]
    if type(cadence) is int:
        if cadence < 0:
            add(
                "checkpoint_every",
                "invalid_range",
                "Checkpoint interval must be nonnegative (0 saves only the final checkpoint)",
            )
        elif cadence and (provider != "local" or method == "sft"):
            add(
                "checkpoint_every",
                "checkpoint_backend",
                "Periodic checkpoint intervals currently require local native RL",
            )
        elif (
            cadence
            and type(parsed["steps"]) is int
            and parsed["steps"] > 0
            and (parsed["steps"] + cadence - 1) // cadence > MAX_STUDIO_CHECKPOINTS
        ):
            add(
                "checkpoint_every",
                "checkpoint_limit",
                "Increase the checkpoint interval to retain at most 100 checkpoints",
            )
    if (
        isinstance(gpu, str)
        and gpu
        and (not gpu.strip() or any(not c.isprintable() for c in gpu))
    ):
        add("gpu", "invalid_gpu", "GPU must be printable nonempty text or blank")
    if type(count) is int and count < 0:
        add(
            "gpu_count",
            "invalid_range",
            "GPU count must be a nonnegative integer (0 is automatic)",
        )
    if provider == "river":
        if isinstance(gpu, str) and gpu:
            add(
                "gpu", "managed_hardware", "River manages hardware; clear GPU overrides"
            )
        if type(count) is int and count != 0:
            add(
                "gpu_count",
                "managed_hardware",
                "River manages hardware; clear GPU overrides",
            )
    for name, default in defaults.items():
        value = parsed[name]
        if name in seen or name in {"gpu_count", "checkpoint_every", "seed"}:
            continue
        if type(default) is int and value < 1:
            add(name, "invalid_range", f"{name} must be a positive integer")
        elif type(default) is float:
            try:
                valid = math.isfinite(value) and value > 0
            except (OverflowError, TypeError):
                valid = False
            if not valid:
                add(name, "invalid_range", f"{name} must be finite and positive")
    rank = parsed["lora_rank"]
    if provider == "river" and type(rank) is int and rank > 32:
        add("lora_rank", "backend_rank", "River SFT supports LoRA ranks up to 32")
    generations = parsed["generations"]
    if (
        method_valid
        and method != "sft"
        and type(generations) is int
        and generations < 2
    ):
        add(
            "generations",
            "rl_generations",
            "RL needs at least two generations per scenario",
        )
    return parsed, issues


def write_json(path: Path, value: Any) -> None:
    """Atomically replace a private workspace record."""
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


_STATE_LOCK_TIMEOUT = 1.0
#: Outcomes that, once recorded, only ever gain detail (exit codes, timings).
FINAL_STATUSES = frozenset({"succeeded", "failed", "cancelled"})


# The lock primitive is chosen once, at import, from the real platform: a
# caller (or a test) that later fakes ``sys.platform`` for process checks must
# not switch file locking to a module this platform does not have.
if sys.platform == "win32":
    import msvcrt

    def _state_lock_operation(descriptor: int, *, release: bool = False) -> None:
        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)

else:
    import fcntl

    def _state_lock_operation(descriptor: int, *, release: bool = False) -> None:
        fcntl.flock(
            descriptor, fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB
        )


@contextmanager
def state_lock(directory: Path) -> Iterator[None]:
    """Serialize bounded state transitions through a stable private descriptor."""
    with private_file_lock(
        directory / ".state.lock",
        timeout_s=_STATE_LOCK_TIMEOUT,
        unavailable="Fine-tuning state lock is unavailable",
        busy="Fine-tuning state is busy",
    ):
        yield


@contextmanager
def private_file_lock(
    path: Path, *, timeout_s: float, unavailable: str, busy: str
) -> Iterator[None]:
    """Hold an exclusive OS lock on a private regular file, on any platform.

    The OS releases it if the holder dies. Waits up to ``timeout_s``, then
    raises ``ValueError(busy)``; a symlink, non-regular or group/world
    accessible lock file raises ``ValueError(unavailable)``.
    """
    if path.is_symlink():
        raise ValueError(unavailable)
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or (
            os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise ValueError(unavailable)
        if os.name == "nt" and info.st_size == 0:
            os.write(descriptor, b"\0")
        deadline = time.monotonic() + timeout_s
        while True:
            try:
                _state_lock_operation(descriptor)
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                    raise
                if time.monotonic() >= deadline:
                    raise ValueError(busy) from None
                time.sleep(0.01)
        try:
            yield
        finally:
            _state_lock_operation(descriptor, release=True)
    finally:
        os.close(descriptor)


def transition_state(
    directory: Path,
    *,
    allowed_from: Collection[str] | None = None,
    success_is_final: Callable[[Path], bool] | None = None,
    before_write: Callable[[dict[str, Any]], None] | None = None,
    **changes: Any,
) -> bool:
    """Atomically merge ``changes`` into ``directory/state.json`` under its lock.

    Writes nothing and returns ``False`` for an illegal transition:

    * ``allowed_from`` is given and the current status is not in it;
    * the current status is final (succeeded, failed, cancelled) and the
      change would move it to any other status. The one exception is a
      success that ``success_is_final`` does not confirm, which a failure or
      cancellation may still replace (no callback: never confirmed).

    ``before_write`` runs under the lock once the transition is known to be
    legal, for records that must only exist alongside it (failure receipts).
    An unreadable state is replaced, keeping a ``previous_state_unreadable``
    marker, unless ``allowed_from`` demands a known status.
    """
    with state_lock(directory):
        try:
            state = read_json(directory / "state.json")
        except (OSError, ValueError):
            state = {"previous_state_unreadable": True}
        current, status = state.get("status"), changes.get("status")
        if allowed_from is not None and current not in allowed_from:
            return False
        if current in FINAL_STATUSES and status is not None and status != current:
            demotable = (
                current == "succeeded"
                and status in {"failed", "cancelled"}
                and not (success_is_final is not None and success_is_final(directory))
            )
            if not demotable:
                return False
        if before_write is not None:
            before_write(state)
        write_json(
            directory / "state.json", {**state, **changes, "updated_at": time.time()}
        )
        return True


def completion_verified(run: Path) -> bool:
    """The worker's completion receipt still points at a usable artifact."""
    try:
        completion = read_json(run / "completion.json", max_bytes=128 * 1024)
    except (OSError, ValueError, TypeError, RecursionError):
        return False
    if completion.get("result_kind") == "hosted_pointer":
        checkpoint = completion.get("checkpoint")
        return (
            isinstance(checkpoint, str)
            and checkpoint.startswith("river://")
            and len(checkpoint) > len("river://")
        )
    output, weights = completion.get("output_dir"), completion.get("weights")
    if not isinstance(output, str) or not isinstance(weights, list) or not weights:
        return False
    for name in weights:
        if (
            not isinstance(name, str)
            or not name
            or Path(name).is_absolute()
            or ".." in Path(name).parts
        ):
            return False
        path = Path(output) / name
        try:
            if path.is_symlink() or not path.is_file() or path.stat().st_size <= 0:
                return False
        except OSError:
            return False
    return True


def transition_run(
    run: Path,
    *,
    allowed_from: Collection[str] | None = None,
    before_write: Callable[[dict[str, Any]], None] | None = None,
    **changes: Any,
) -> bool:
    """The one way to change a fine-tuning run's ``state.json``.

    A success whose committed artifact still verifies is final; see
    :func:`transition_state` for every other rule.
    """
    return transition_state(
        run,
        allowed_from=allowed_from,
        success_is_final=completion_verified,
        before_write=before_write,
        **changes,
    )


def read_json(path: Path, *, max_bytes: int = 128 * 1024) -> dict[str, Any]:
    """Read a bounded regular object record without following symbolic links."""
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if path.is_symlink():
        raise ValueError("Workspace records must be regular files")
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
            raise ValueError("Workspace record exceeds the regular-file limit")
        raw = stream.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Workspace record exceeds the size limit")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object in {path.name}")
    return value


def read_scenarios(raw: bytes) -> list[dict[str, Any]]:
    """Validate support scenarios for the native conversation reward/environment."""
    from stateset_agents.data.text_sft import DatasetFormatError

    try:
        rows = json.loads(raw)
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(
            "invalid_encoding",
            "Dataset must use a JSON-compatible text encoding.",
            byte=exc.start + 1,
        ) from None
    except json.JSONDecodeError as exc:
        raise DatasetFormatError(
            "invalid_json",
            "Expected a JSON list of support scenarios.",
            line=exc.lineno,
            column=exc.colno,
        ) from None
    except (ValueError, RecursionError):
        raise DatasetFormatError(
            "invalid_json", "JSON nesting or numeric values exceed parser limits."
        ) from None
    if not isinstance(rows, list) or not rows:
        raise DatasetFormatError(
            "invalid_scenarios",
            "RL data must be a nonempty JSON list of support scenarios",
        )
    ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise DatasetFormatError(
                "invalid_scenario", "Scenario must be an object", example=index + 1
            )
        for name in ("id", "context", "expected_resolution"):
            if not isinstance(row.get(name), str) or not row[name].strip():
                raise DatasetFormatError(
                    "invalid_scenario",
                    f"Scenario needs nonempty {name}",
                    example=index + 1,
                    field=name,
                )
        for name in ("policy_facts", "user_responses"):
            if (
                not isinstance(row.get(name), list)
                or not row[name]
                or any(
                    not isinstance(text, str) or not text.strip() for text in row[name]
                )
            ):
                raise DatasetFormatError(
                    "invalid_scenario",
                    f"Scenario needs nonempty text list {name}",
                    example=index + 1,
                    field=name,
                )
        if row["id"] in ids:
            raise DatasetFormatError(
                "duplicate_id",
                "Scenario IDs must be distinct",
                example=index + 1,
                field="id",
            )
        ids.add(row["id"])
        avoid = row.get("must_avoid", [])
        if not isinstance(avoid, list) or any(
            not isinstance(text, str) for text in avoid
        ):
            raise DatasetFormatError(
                "invalid_scenario",
                "must_avoid must be a text list",
                example=index + 1,
                field="must_avoid",
            )
    return list(rows)


@dataclass(frozen=True)
class RunPlan:
    """Validated snapshot shown in review and passed unchanged to a worker."""

    config: FinetuneConfig
    cwd: Path
    data: bytes
    examples: int
    resources: dict[str, Any]
    notes: tuple[str, ...]
    implementation_sha256: str
    resume: dict[str, Any] | None = None
    remote_code: RemoteCodeSelection | None = None

    def describe(self) -> dict[str, Any]:
        """Describe a run without exposing transcript contents or environment values."""
        from stateset_agents import __version__

        return {
            "schema_version": 1,
            "framework_version": __version__,
            "package_implementation_sha256": self.implementation_sha256,
            "config": asdict(self.config),
            "cwd": str(self.cwd),
            "dataset_sha256": hashlib.sha256(self.data).hexdigest(),
            "examples": self.examples,
            "resources": self.resources,
            "notes": list(self.notes),
            "quality_gain_verified": False,
            "stop_protocol": STOP_PROTOCOL_VERSION,
            **({"resume": self.resume} if self.resume is not None else {}),
            **(
                {"remote_code": self.remote_code.describe()}
                if self.remote_code is not None
                else {}
            ),
        }


@dataclass(frozen=True)
class PreparedRunReview:
    """An existing private run and the exact plan bytes reviewed for launch."""

    run: Path
    plan: RunPlan
    plan_sha256: str


def read_run_plan(run: Path) -> tuple[dict[str, Any], str]:
    """Read bounded plan bytes and their checksum without trusting a symlink."""
    path = run / "plan.json"
    if run.is_symlink() or path.is_symlink() or not run.is_dir() or not path.is_file():
        raise ValueError("Run and plan must be regular workspace entries")
    with path.open("rb") as stream:
        raw = stream.read(128 * 1024 + 1)
    if len(raw) > 128 * 1024:
        raise ValueError("Reviewed plan exceeds the record size limit")
    try:
        plan = json.loads(raw)
    except (ValueError, RecursionError):
        raise ValueError("Reviewed plan is not a valid JSON object") from None
    if (
        not isinstance(plan, dict)
        or type(plan.get("schema_version")) is not int
        or plan["schema_version"] != 1
    ):
        raise ValueError("Unsupported reviewed plan schema")
    return plan, hashlib.sha256(raw).hexdigest()


def verified_run_settings(run: Path) -> tuple[dict[str, Any], FinetuneConfig]:
    """Verify saved config, package identity, launch binding and dataset bytes."""
    plan, digest = read_run_plan(run)
    config = FinetuneConfig(**plan["config"])
    config.validate()
    from stateset_agents import __version__
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.evaluation.implementation import package_implementation

    if plan["framework_version"] != __version__:
        raise ValueError("Worker version differs from the reviewed plan")
    if plan["package_implementation_sha256"] != content_hash(package_implementation()):
        raise ValueError("Package code changed after launch review; prepare a new run")
    if config.provider == "runpod":
        from stateset_agents.finetune_remote_code import verify_remote_code

        verify_remote_code(
            run, plan.get("remote_code"), package_version=plan["framework_version"]
        )
    lock = run / "launch.lock"
    if lock.is_symlink():
        raise ValueError("Launch record must not be a symlink")
    if lock.exists() and lock.stat().st_size:
        with lock.open("rb") as stream:
            raw_receipt = stream.read(4097)
        if len(raw_receipt) > 4096:
            raise ValueError("Invalid launch record")
        try:
            receipt = json.loads(raw_receipt)
        except (ValueError, RecursionError):
            raise ValueError("Invalid launch record") from None
        if (
            not isinstance(receipt, dict)
            or type(receipt.get("schema_version")) is not int
            or receipt["schema_version"] != 1
            or receipt.get("plan_sha256") != digest
        ):
            raise ValueError("Reviewed plan bytes changed after launch submission")
    snapshot = run / "dataset"
    if snapshot.is_symlink() or not snapshot.is_file():
        raise ValueError("Reviewed dataset must be a regular file without a symlink")
    with snapshot.open("rb") as stream:
        data = stream.read(MAX_DATA_BYTES + 1)
    if (
        len(data) > MAX_DATA_BYTES
        or hashlib.sha256(data).hexdigest() != plan["dataset_sha256"]
    ):
        raise ValueError("Reviewed dataset bytes changed")
    if "resume" in plan:
        from stateset_agents.finetune_recovery import verify_resume_input

        verify_resume_input(run, plan)
    return plan, config


def review_prepared_run(run: Path) -> PreparedRunReview:
    """Restore an existing prepared plan without replanning or reading source data."""
    if (
        read_json(run / "state.json").get("status") != "prepared"
        or (run / "launch.lock").exists()
        or (run / "launch.lock").is_symlink()
    ):
        raise ValueError(
            "Only an unsubmitted prepared run can be reviewed; use c for new settings"
        )
    record, digest = read_run_plan(run)
    verified, config = verified_run_settings(run)
    if verified != record:
        raise ValueError("Plan changed while opening its review")
    resources = record.get("resources")
    if (
        not isinstance(record.get("cwd"), str)
        or not Path(record["cwd"]).is_absolute()
        or type(record.get("examples")) is not int
        or record["examples"] < 1
        or not isinstance(resources, dict)
        or resources.get("provider") != config.provider
        or resources.get("cuda_required") is not (config.provider == "local")
        or not isinstance(record.get("notes"), list)
        or any(not isinstance(note, str) for note in record["notes"])
    ):
        raise ValueError("Malformed retained review metadata")
    if config.provider in {"runpod", "modal"} and (
        type(resources.get("gpu_count")) is not int
        or resources["gpu_count"] < 1
        or not isinstance(resources.get("gpu"), str)
        or not resources["gpu"].strip()
    ):
        raise ValueError("Malformed retained resource metadata")
    if config.provider == "runpod" and (
        any(
            type(resources.get(key)) is not int or resources[key] < 1
            for key in ("gpu_count", "container_disk_gb")
        )
        or any(
            not isinstance(resources.get(key), str) or not resources[key]
            for key in ("gpu", "recommendation_evidence")
        )
    ):
        raise ValueError("Malformed retained resource metadata")
    with (run / "dataset").open("rb") as stream:
        data = stream.read(MAX_DATA_BYTES + 1)
    remote_code = None
    if config.provider == "runpod":
        from stateset_agents.finetune_remote_code import normalize_remote_code

        code = normalize_remote_code(
            record.get("remote_code"), package_version=record["framework_version"]
        )
        remote_code = RemoteCodeSelection(
            code["package_version"],
            code.get("wheel_filename"),
            code.get("wheel_sha256"),
            code.get("wheel_bytes"),
        )
    plan = RunPlan(
        config,
        Path(record["cwd"]),
        data,
        record["examples"],
        record["resources"],
        tuple(record["notes"]),
        record["package_implementation_sha256"],
        record.get("resume"),
        remote_code,
    )
    if plan.describe() != record or read_run_plan(run)[1] != digest:
        raise ValueError("Retained review does not match its recorded plan")
    return PreparedRunReview(run, plan, digest)


def read_training_data(source: Path) -> bytes:
    """Capture at most 64 MiB from a regular file, including a file symlink.

    A fixed read bound also applies if a file grows after its size check.
    Nonblocking open plus the descriptor type check prevents FIFO reads.
    """
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    with os.fdopen(os.open(source, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_DATA_BYTES:
            raise ValueError("Dataset must be a regular file no larger than 64 MiB")
        raw = stream.read(MAX_DATA_BYTES + 1)
    if len(raw) > MAX_DATA_BYTES:
        raise ValueError("Dataset exceeds 64 MiB")
    return raw


def plan_run(config: FinetuneConfig, *, cwd: Path | None = None) -> RunPlan:
    """Validate and snapshot inputs without importing a trainer or calling a provider."""
    config.validate()
    cwd = (cwd or Path.cwd()).resolve()
    source = Path(config.dataset).expanduser()
    if not source.is_absolute():
        source = cwd / source
    try:
        raw = read_training_data(source)
    except (FileNotFoundError, IsADirectoryError):
        raise ValueError("Dataset must be a file no larger than 64 MiB") from None
    except PermissionError:
        # Windows reports opening a directory as EACCES rather than EISDIR.
        if not source.is_dir():
            raise
        raise ValueError("Dataset must be a file no larger than 64 MiB") from None
    data_provenance: dict[str, Any] = {}
    if config.method == "sft":
        # The strict parser rejects every malformed row instead of skipping it.
        from stateset_agents.data.text_sft import parse_text_sft_records

        rows, data_provenance = parse_text_sft_records(raw, str(source))
        examples = len(rows)
    else:
        examples = len(read_scenarios(raw))
    notes = [
        "Completion is not evidence of a quality gain.",
        f"Training seed: {config.seed}. Controls initialization and training randomness; identical GPU results are not guaranteed.",
    ]
    if data_provenance.get("synthetic_examples"):
        notes.insert(
            0,
            f"Dataset: {data_provenance['synthetic_examples']} declared synthetic industry examples; not independently reviewed. Source bytes and provenance are retained.",
        )
    if not config.revision:
        notes.append("Model revision is mutable; set a commit for reproducibility.")
    if config.method != "sft":
        notes.append("Support reward v3; the final checkpoint is not judge-selected.")
        notes.append(
            "Use the recorded recovery study for independent checkpoint selection."
        )
        if config.checkpoint_every:
            count = (
                config.steps + config.checkpoint_every - 1
            ) // config.checkpoint_every
            notes.append(
                f"Retain up to {count} full-state checkpoints at {config.checkpoint_every}-update intervals, including the final save. Reserve workspace disk before launch; use j/c for reviewed checkpoint validation."
            )
    resources: dict[str, Any] = {
        "provider": config.provider,
        "cuda_required": config.provider == "local",
    }
    if config.method != "sft":
        resources["retained_checkpoint_count"] = (
            (config.steps + config.checkpoint_every - 1) // config.checkpoint_every
            if config.checkpoint_every
            else 1
        )
    if config.provider == "runpod":
        from stateset_agents.remote.model_catalog import plan_runpod_resources

        resources.update(
            plan_runpod_resources(
                config.model,
                gpu=config.gpu or None,
                gpu_count=config.gpu_count or None,
            )
        )
        notes.append(
            "RunPod is paid; ceiling covers the provider's quoted pod lifetime, not all ancillary costs."
        )
    elif config.provider == "modal":
        resources.update(
            gpu=config.gpu or "A10",
            gpu_count=config.gpu_count or 1,
            result_kind="local_weights",
        )
        notes.append(
            "Modal installs this framework version from PyPI on remote GPU compute; local source edits are not uploaded. Remote timeout applies to the function, not image builds or downloads."
        )
    elif config.provider == "river":
        resources.update(result_kind="hosted_pointer")
        notes.append(
            "River account model access is checked at launch. This integration retains a hosted river:// checkpoint pointer, not local weights. River request timeouts do not bound the whole run."
        )
    if config.provider in {"modal", "river"}:
        notes.append(
            "Provider billed: no enforced dollar ceiling. The RunPod ceiling setting does not apply. Review provider account limits and pricing before launch."
        )
    from stateset_agents import __version__
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.evaluation.implementation import package_implementation
    from stateset_agents.finetune_remote_code import capture_remote_code

    remote_code = capture_remote_code(config.provider, __version__, cwd=cwd)
    if remote_code is not None:
        code = remote_code.describe()
        if code["mode"] == "wheel":
            notes.append(
                f"Remote wheel: {code['wheel_filename']} / SHA256 {code['wheel_sha256']} / {code['wheel_bytes']} bytes. Preparation retains a private copy; installation verifies uploaded bytes."
            )
        else:
            notes.append(
                f"Remote package: stateset-agents {code['package_version']}; ambient wheel overrides after review are ignored. A version pin does not attest remote dependencies."
            )
    return RunPlan(
        config,
        cwd,
        raw,
        examples,
        resources,
        tuple(notes),
        content_hash(package_implementation()),
        remote_code=remote_code,
    )


def create_run(workspace: Path, plan: RunPlan) -> Path:
    """Persist an exclusive run with the exact reviewed input bytes."""
    workspace.mkdir(parents=True, exist_ok=True, mode=0o700)
    run = workspace / (
        time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:8]
    )
    run.mkdir(mode=0o700)
    try:
        descriptor = os.open(
            run / "dataset", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
        )
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(plan.data)
        if plan.config.provider == "runpod":
            from stateset_agents.finetune_remote_code import snapshot_remote_code

            if plan.remote_code is None:
                raise ValueError(
                    "RunPod preparation requires a reviewed package identity"
                )
            if (
                snapshot_remote_code(run, plan.remote_code)
                != plan.describe()["remote_code"]
            ):
                raise ValueError("Remote package differs from the reviewed selection")
        write_json(run / "plan.json", plan.describe())
        write_json(
            run / "state.json", {"status": "prepared", "created_at": time.time()}
        )
    except BaseException:
        # Only this call's new, exclusive private directory is removed.
        shutil.rmtree(run, ignore_errors=True)
        raise
    return run.resolve()


#: Detached supervisors this process started. Their handles are kept so the
#: long-lived terminal studio can reap them once they exit: an unreaped child
#: lingers as a zombie, and on macOS ``kill(pid, 0)`` succeeds on a zombie, so
#: a finished supervisor would keep looking alive to the owner checks.
_DETACHED_CHILDREN: list[Any] = []
_DETACHED_CHILDREN_LOCK = threading.Lock()


def retain_child(process: Any) -> None:
    """Keep a detached supervisor's handle so its exit can be reaped later."""
    with _DETACHED_CHILDREN_LOCK:
        _DETACHED_CHILDREN.append(process)


def reap_children() -> None:
    """Collect every retained supervisor that has exited, without waiting."""
    with _DETACHED_CHILDREN_LOCK:
        running = []
        for process in _DETACHED_CHILDREN:
            poll = getattr(process, "poll", None)
            try:
                if callable(poll) and poll() is None:
                    running.append(process)
            except OSError:
                continue
        _DETACHED_CHILDREN[:] = running


#: Windows process-creation flags: a new process group (so the whole tree can
#: be stopped together) with its own hidden console, detached from the
#: launching terminal so that closing it cannot end the run.
_WINDOWS_DETACHED = (
    0x00000200 | 0x08000000
)  # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW


def detached_process_options(*, windows: bool | None = None) -> dict[str, Any]:
    """``Popen`` options for a child that must outlive the launching terminal.

    POSIX starts a new session. Windows ignores ``start_new_session``, so a
    supervisor or worker started without these flags shares the console and
    dies (or sees Ctrl+C) with it.
    """
    if os.name == "nt" if windows is None else windows:
        return {"creationflags": _WINDOWS_DETACHED}
    return {"start_new_session": True}


def queue_executable() -> str | None:
    """Find the shared queue: ``STATESET_BUILDR`` if set, else ``buildr`` on PATH.

    There is deliberately no fixed-path fallback: the queue wrapper runs with
    the caller's environment, so only an executable the user configured or
    put on their own PATH may be chosen.
    """
    configured = os.environ.get("STATESET_BUILDR")
    if configured:
        if not Path(configured).is_file() or not os.access(configured, os.X_OK):
            raise ValueError("STATESET_BUILDR must point to an executable")
        return configured
    candidate = shutil.which("buildr")
    if candidate and os.path.isfile(candidate) and os.access(candidate, os.X_OK):
        return candidate
    return None


def launch_run(run: Path, *, expected_plan_sha256: str | None = None) -> int:
    """Start a detached supervisor; closing Herdr or the TUI does not stop training."""
    if read_json(run / "state.json").get("status") != "prepared":
        raise ValueError("Only a fresh prepared run can be launched")
    plan, digest = read_run_plan(run)
    if expected_plan_sha256 is not None and digest != expected_plan_sha256:
        raise ValueError("Plan checksum differs from the reviewed plan")
    descriptor = os.open(
        run / "launch.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as receipt:
            json.dump({"schema_version": 1, "plan_sha256": digest}, receipt)
            receipt.flush()
            os.fsync(receipt.fileno())
        verified_run_settings(run)
        if stop_requested(run):
            raise ValueError("This run has a stop request; prepare a fresh run")
        if read_run_plan(run)[1] != digest:
            raise ValueError("Plan changed while submitting the reviewed run")
        with (run / "supervisor.log").open("ab") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "stateset_agents.finetune_worker",
                    "--supervise",
                    str(run),
                ],
                cwd=plan["cwd"],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **detached_process_options(),
            )
        retain_child(process)
    except (OSError, ValueError, TypeError, KeyError, RecursionError):
        cancelled = False
        try:
            cancelled = stop_requested(run)
        except (OSError, ValueError, TypeError, RecursionError):
            pass
        transition_run(
            run,
            allowed_from={"prepared"},
            status="cancelled" if cancelled else "failed",
            error=(
                "Local stop requested before launch"
                if cancelled
                else "Worker could not start or reviewed inputs changed"
            ),
        )
        raise
    return process.pid


def stop_requested(run: Path) -> bool:
    """Read a bounded cooperative stop receipt bound to the exact run plan."""
    path = run / "stop.json"
    if not path.exists() and not path.is_symlink():
        return False
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if path.is_symlink():
        raise ValueError("Stop receipt must be a regular file")
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > 4096:
            raise ValueError("Invalid stop receipt")
        raw = stream.read(4097)
    value = json.loads(raw)
    if (
        len(raw) > 4096
        or not isinstance(value, dict)
        or type(value.get("schema_version")) is not int
        or value["schema_version"] != STOP_PROTOCOL_VERSION
        or value.get("plan_sha256") != read_run_plan(run)[1]
    ):
        raise ValueError("Stop receipt differs from the reviewed plan")
    return True


def _stop_protocol_supported(plan: Mapping[str, Any]) -> bool:
    """Whether this package writes a stop receipt the run's worker honors.

    A worker runs the package that planned it, and the plan records the
    receipt version that package checks. Source changes since then do not
    matter while the version matches. Plans from before the version was
    recorded fall back to requiring the identical package implementation.
    """
    if "stop_protocol" in plan:
        version = plan["stop_protocol"]
        return type(version) is int and version == STOP_PROTOCOL_VERSION
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.evaluation.implementation import package_implementation

    return plan.get("package_implementation_sha256") == content_hash(
        package_implementation()
    )


#: Remote providers whose worker ends its own allocation when signalled.
REMOTE_STOP_PROVIDERS = frozenset({"runpod"})


def remote_stop_supported(provider: object) -> bool:
    """Whether this host can stop a paid run of ``provider`` from the studio.

    RunPod workers turn SIGTERM into their cleanup (pod termination). Windows
    cannot deliver that signal to a detached worker without killing it
    outright, so remote stop is POSIX-only.
    """
    return provider in REMOTE_STOP_PROVIDERS and os.name != "nt"


def _signal_remote_worker(run: Path) -> bool:
    """Ask the verified execution owner to stop; return whether it was signalled.

    Only a process whose start-time token matches ``execution.json`` is
    signalled, so a reused PID is never touched. The worker converts SIGTERM
    into ``SystemExit``: the executor terminates its pod, and the stop receipt
    makes the outcome ``cancelled``.
    """
    import signal

    owner = run / "execution.json"
    try:
        if not _present(owner) or owner_evidence(owner) != "alive":
            return False
        os.kill(read_json(owner, max_bytes=4096)["pid"], signal.SIGTERM)
    except (OSError, ValueError, TypeError, KeyError, RecursionError, OverflowError):
        return False
    return True


def request_stop(run: Path, *, expected_plan_sha256: str) -> dict[str, Any]:
    """Request a plan-bound stop: cooperative for local runs, signalled for RunPod.

    Local workers stop at their next safe boundary and are never signalled.
    A RunPod worker is signalled (only after its identity is verified) so it
    terminates its pod; other remote providers are refused.
    """
    plan, digest = read_run_plan(run)
    if digest != expected_plan_sha256:
        raise ValueError("Plan checksum differs from the reviewed plan")
    provider = plan["config"].get("provider")
    remote = provider != "local"
    if remote and not remote_stop_supported(provider):
        raise ValueError(
            "Remote stop is available for RunPod runs on POSIX hosts; use "
            "provider-specific cleanup controls for this run"
        )
    state = run_status(run)
    if run_terminal_status(state) is True:
        return {
            "status": state["status"],
            "terminal": True,
            "execution_unconfirmed": False,
            "stop_requested": stop_requested(run),
        }
    if not _stop_protocol_supported(plan):
        raise ValueError(
            "Stop support requires a stop receipt version the run's worker honors"
        )
    if not stop_requested(run):
        # Publish a complete receipt atomically; concurrent callers are idempotent.
        write_json(
            run / "stop.json",
            {"schema_version": STOP_PROTOCOL_VERSION, "plan_sha256": digest},
        )
    # The stop receipt is published before this check, and every supervisor or
    # direct worker claims admission before it checks for a stop. So a launch
    # that no process has claimed yet can be cancelled outright: any process
    # that claims it later stops before doing work.
    if not (run / "launch.lock").exists() or launch_unclaimed(run):
        # Only a still-prepared run changes; anything else is left as is.
        transition_run(
            run,
            allowed_from={"prepared"},
            status="cancelled",
            finished_at=time.time(),
        )
    signalled = remote and _signal_remote_worker(run)
    current = run_status(run)
    result = {
        "status": current["status"],
        "terminal": run_terminal_status(current),
        "execution_unconfirmed": current.get("execution_unconfirmed") is True,
        "stop_requested": True,
        "scope": "Cooperative local stop at the next safe boundary; queued entries remain until admitted. Supervisor loss does not establish worker exit. Remote resources are not controlled.",
    }
    if remote:
        result["worker_signalled"] = signalled
        result["scope"] = (
            "The verified RunPod worker was asked to stop; it terminates its pod and records the cancellation. An acknowledged deletion does not prove billing ended."
            if signalled
            else "Stop receipt saved, but no verified RunPod worker is running to signal; inspect the run and `stateset-agents runpod-orphans`."
        )
    return result


def process_token(pid: int) -> str | None:
    """A start-time token that distinguishes a worker from a reused PID.

    Linux reads start ticks from ``/proc``. Other platforms use the process
    creation time (``psutil``, or ``ps`` on macOS); ``None`` when unavailable.
    """
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        if fields[0] in {"Z", "X"}:
            return None
        return fields[19]
    except (OSError, IndexError):
        if sys.platform == "linux":
            return None
    return process_start_token(pid)


def run_terminal_status(state: Mapping[str, Any]) -> bool | None:
    """Describe execution completion without equating supervisor loss with exit.

    Interrupted supervision leaves detached queued/running work unconfirmed.
    This metadata helper neither inspects processes nor changes retained state.
    """
    status = state.get("status")
    if not isinstance(status, str):
        return None
    if status in {"succeeded", "failed", "cancelled"}:
        return True
    if status in {"prepared", "starting", "queued", "running"}:
        return False
    return None


OWNER_EVIDENCE = frozenset(
    {"alive", "reused", "gone", "running", "denied", "malformed"}
)


def owner_evidence(
    path: Path,
    *,
    use_tokens: bool = True,
    probe: bool = True,
    max_bytes: int = 128 * 1024,
) -> str:
    """Classify the process named by one owner record, without signalling it.

    * ``"alive"``: its Linux start-time token matches the record;
    * ``"reused"``: the PID now belongs to a different process;
    * ``"gone"``: no process has that PID;
    * ``"running"``: a process with that PID answers, identity unproven
      (no token available: non-Linux, a zombie, or an unreadable ``/proc``);
    * ``"denied"``: a process exists but may not be inspected;
    * ``"malformed"``: the record names no valid PID.

    ``use_tokens``/``probe`` select which evidence to gather (a skipped probe
    reports ``"running"`` as unknown would). An unreadable record raises.
    """
    owner = read_json(path, max_bytes=max_bytes)
    pid = owner.get("pid")
    if type(pid) is not int or pid < 1:
        return "malformed"
    if use_tokens:
        current = process_token(pid)
        if current is not None:
            return "alive" if current == owner.get("process_token") else "reused"
    if not probe:
        return "running"
    try:
        probe_process(pid)
    except ProcessLookupError:
        return "gone"
    except PermissionError:
        return "denied"
    return "running"


def owner_alive(path: Path) -> bool | None:
    """Whether a recorded owner is provably running (True) or gone (False).

    ``None`` means the evidence cannot decide: a PID that answers without a
    matching start-time token, a process that may not be inspected, or a
    record without a valid PID. Callers choose how cautious to be about it.
    """
    evidence = owner_evidence(path)
    if evidence == "alive":
        return True
    if evidence in {"gone", "reused"}:
        return False
    return None


def _owner_confirmed(path: Path) -> bool:
    """Boundedly inspect one local owner without signaling it or proving exit.

    Linux trusts only a matching start-time token. Without tokens, a PID that
    still answers (or refuses inspection) is taken as the owner.
    """
    try:
        if sys.platform == "linux":
            return owner_evidence(path, probe=False, max_bytes=4096) == "alive"
        else:
            # A recorded creation-time token tells a reused PID apart; records
            # without one (or hosts that cannot read one now) fall back to
            # presence, where an answering PID counts as the owner.
            recorded = read_json(path, max_bytes=4096).get("process_token")
            return owner_evidence(
                path, use_tokens=isinstance(recorded, str), max_bytes=4096
            ) in {"alive", "running", "denied"}
    except (OSError, ValueError, TypeError, RecursionError, OverflowError):
        return False


#: How long a submitted launch may go without any process claiming it before
#: its owner counts as lost. A supervisor normally claims admission within a
#: second of ``launch_run``; this only bounds how long a launch whose process
#: died first (OOM, reboot, import failure) can keep looking "prepared".
LAUNCH_CLAIM_GRACE_SECONDS = 120.0


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def launch_unclaimed(directory: Path) -> bool:
    """A launch receipt exists, but no admission/supervisor/execution owner does."""
    return _present(directory / "launch.lock") and not any(
        _present(directory / name)
        for name in ("admission.json", "supervisor.json", "execution.json")
    )


def launch_claim_lost(
    directory: Path, *, grace_seconds: float = LAUNCH_CLAIM_GRACE_SECONDS
) -> bool:
    """An unclaimed launch receipt is older than the startup grace window."""
    if not launch_unclaimed(directory):
        return False
    try:
        submitted = (directory / "launch.lock").lstat().st_mtime
    except OSError:
        return False
    return time.time() - submitted > grace_seconds


def detached_owner_lost(directory: Path, status: object) -> bool:
    """Whether unfinished detached work in ``directory`` has lost its owner.

    The supervisor record owns the work once it exists, otherwise a worker's
    execution record does; a submitted launch that no process claimed within
    the startup grace window is lost too. Like :func:`run_status`, this cannot
    establish that detached queue work has exited.
    """
    if status not in {"prepared", "starting", "queued", "running"}:
        return False
    supervisor = directory / "supervisor.json"
    execution = directory / "execution.json"
    if _present(supervisor):
        return not _owner_confirmed(supervisor)
    if status == "running" or _present(execution):
        return not _owner_confirmed(execution)
    return status == "prepared" and launch_claim_lost(directory)


def run_status(run: Path) -> dict[str, Any]:
    """Read committed outcomes; missing owners leave unfinished work unconfirmed."""
    reap_children()
    state = read_json(run / "state.json")
    if state.get("status") not in TERMINAL_STATES:
        supervisor = run / "supervisor.json"
        execution = run / "execution.json"
        admission = run / "admission.json"
        owner = None
        lost = False
        if _present(supervisor):
            owner = supervisor
        elif state.get("status") == "running" or _present(execution):
            owner = execution
        elif state.get("status") == "prepared":
            # Between admission and the supervisor/execution claim, the
            # admitting process owns the run. A launch nobody ever claimed
            # is lost once its startup grace window has passed.
            if _present(admission):
                owner = admission
            else:
                lost = launch_claim_lost(run)
        if lost or (owner is not None and not _owner_confirmed(owner)):
            state = {
                **state,
                "status": "interrupted",
                "error": "Training ownership could not be confirmed; inspect retained records before retrying.",
            }
    # The supervisor launches the queue/worker in its own session. Losing the
    # supervisor cannot establish whether that detached work has exited.
    state = {**state, "execution_unconfirmed": state.get("status") == "interrupted"}
    if stop_requested(run):
        state = {**state, "stop_requested": True}
    return state


def list_runs(workspace: Path, *, limit: int | None = 100) -> list[dict[str, Any]]:
    """Read retained runs; ``None`` includes all history for search and filtering."""
    if limit is not None and limit < 0:
        raise ValueError("Run limit must be nonnegative or None")
    rows = []
    for path in sorted(workspace.glob("*/plan.json"), reverse=True)[:limit]:
        if path.is_symlink() or path.parent.is_symlink():
            continue
        try:
            plan = read_json(path)
            state = run_status(path.parent)
            if not isinstance(plan.get("config"), dict) or not isinstance(
                state.get("status"), str
            ):
                raise ValueError("Invalid run settings or state")
            rows.append(
                {
                    "id": path.parent.name,
                    "path": str(path.parent),
                    "plan": plan,
                    "state": state,
                }
            )
        except (OSError, ValueError, KeyError, TypeError) as error:
            rows.append(
                {
                    "id": path.parent.name,
                    "path": str(path.parent),
                    "plan": {},
                    "state": {
                        "status": "failed",
                        "error": f"Unreadable workspace record ({type(error).__name__})",
                    },
                }
            )
    return rows


LOG_SOURCES = ("auto", "training", "supervisor")
MAX_LOG_BYTES = 128 * 1024


@dataclass(frozen=True)
class LogSnapshot:
    """One bounded, terminal-safe log window and its actual source."""

    source: str
    filename: str
    lines: tuple[str, ...]
    bytes_read: int = 0
    window_truncated: bool = False
    available: bool = False
    error: str | None = None


def read_log_snapshot(run: Path, *, source: str = "auto") -> LogSnapshot:
    """Read a fixed file window; never follow log symlinks or block on a FIFO.

    Auto selects nonempty training output, falling back to supervisor output.
    A truncated window may begin within a line. Appends after the initial file
    size snapshot do not increase the byte budget of this read.
    """
    if source not in LOG_SOURCES:
        raise ValueError("Choose auto, training or supervisor log source")
    selected = source
    if selected == "auto":
        try:
            training = run / "run.log"
            selected = (
                "training"
                if training.is_file()
                and not training.is_symlink()
                and training.stat().st_size
                else "supervisor"
            )
            if (
                selected == "supervisor"
                and not (run / "supervisor.log").exists()
                and training.exists()
            ):
                selected = "training"
        except OSError:
            selected = "training"
    filename = "run.log" if selected == "training" else "supervisor.log"
    path = run / filename
    if run.is_symlink() or path.is_symlink():
        return LogSnapshot(selected, filename, (), error="unreadable")
    try:
        flags = (
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
        )
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                return LogSnapshot(selected, filename, (), error="unreadable")
            end = stream.seek(0, os.SEEK_END)
            start = max(0, end - MAX_LOG_BYTES)
            stream.seek(start)
            raw = stream.read(end - start)
    except FileNotFoundError:
        return LogSnapshot(selected, filename, (), error="missing")
    except OSError:
        return LogSnapshot(selected, filename, (), error="unreadable")
    text = raw.decode("utf-8", errors="replace")
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]", "", text)
    text = "".join(char for char in text if char in "\n\r\t" or char.isprintable())
    rows = tuple(
        text.replace("\r\n", "\n").replace("\r", "\n").expandtabs(4).splitlines()
    )
    return LogSnapshot(selected, filename, rows, len(raw), start > 0, True)


def log_tail(
    run: Path, *, lines: int = 80, offset: int = 0, source: str = "auto"
) -> list[str]:
    """Read a bounded source tail, stripping terminal escapes and control characters."""
    if type(lines) is not int or lines < 1 or type(offset) is not int or offset < 0:
        raise ValueError("Log lines must be positive and offset nonnegative integers")
    snapshot = read_log_snapshot(run, source=source)
    if not snapshot.available:
        return [
            (
                "Waiting for worker output..."
                if snapshot.error == "missing"
                else "Log unavailable; inspect workspace permissions and source."
            )
        ]
    end = max(0, len(snapshot.lines) - offset)
    return list(snapshot.lines[max(0, end - lines) : end])


def finetune_schema() -> dict[str, Any]:
    """Describe shared configuration rules and agent command effects without I/O."""
    from stateset_agents.finetune_evaluation import evaluation_schema
    from stateset_agents.finetune_validation import MAX_COMPARISON_RECORDS

    defaults = asdict(FinetuneConfig())
    properties: dict[str, Any] = {}
    for name, value in defaults.items():
        kind = (
            "integer"
            if type(value) is int
            else "number" if type(value) is float else "string"
        )
        properties[name] = {"type": kind, "default": value}
        if kind in {"integer", "number"}:
            properties[name]["exclusiveMinimum"] = 0
    properties["method"]["enum"] = list(METHODS)
    properties["provider"]["enum"] = list(PROVIDERS)
    properties["cost_policy"]["enum"] = ["require_cap", "provider_billed"]
    properties["gpu_count"].pop("exclusiveMinimum", None)
    properties["gpu_count"]["minimum"] = 0
    properties["checkpoint_every"].pop("exclusiveMinimum", None)
    properties["checkpoint_every"]["minimum"] = 0
    properties["seed"].pop("exclusiveMinimum", None)
    properties["seed"].update(minimum=0, maximum=2**32 - 1)
    properties["dataset"]["minLength"] = 1
    properties["dataset"]["pattern"] = r"\S"
    properties["model"]["pattern"] = r"^(?!stub://).*\S"
    properties["revision"]["pattern"] = r"^(?:[0-9a-f]{40})?$"
    return {
        "config_schema": {
            "type": "object",
            "properties": properties,
            "required": ["dataset"],
            "additionalProperties": False,
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "provider": {"enum": ["runpod", "modal", "river"]}
                        },
                        "required": ["provider"],
                    },
                    "then": {
                        "properties": {
                            "method": {"const": "sft"},
                            "revision": {"const": ""},
                        }
                    },
                },
                {
                    "if": {
                        "properties": {"provider": {"enum": ["modal", "river"]}},
                        "required": ["provider"],
                    },
                    "then": {
                        "properties": {"cost_policy": {"const": "provider_billed"}},
                        "required": ["cost_policy"],
                    },
                },
                {
                    "if": {
                        "properties": {"provider": {"const": "river"}},
                        "required": ["provider"],
                    },
                    "then": {
                        "properties": {
                            "gpu": {"const": ""},
                            "gpu_count": {"const": 0},
                            "lora_rank": {"maximum": 32},
                        }
                    },
                },
                {
                    "if": {
                        "properties": {"method": {"enum": ["grpo", "gspo", "dapo"]}},
                        "required": ["method"],
                    },
                    "then": {"properties": {"generations": {"minimum": 2}}},
                },
            ],
        },
        "dataset_max_bytes": MAX_DATA_BYTES,
        "local_methods": list(METHODS),
        "runpod_methods": ["sft"],
        "backends": BACKENDS,
        "effects": {
            "schema": "read_only",
            "models": "read_only",
            "readiness": "local_inspection",
            "validate": "configuration_only",
            "prepare": "local_snapshot_only",
            "launch": "starts_training",
            "status": "read_only",
            "provider-status": "reads_private_plan_bound_allocation_receipts; no_provider_calls",
            "logs": "reads_private_logs",
            "compare": "read_only",
            "inspect-data": "reads_dataset",
            "workflow": "local_inspection",
            "quality": "reads_artifact_and_evidence; --evidence retains private reports",
            "stop": "requests_cooperative_local_stop",
            "checkpoints": "reads_local_recovery_artifacts",
            "prepare-resume": "copies_checkpoint_and_dataset; separate_launch_required",
            "evaluation-plan": "reads_local_artifact_and_held_out_cases",
            "evaluation-prepare": "snapshots_evaluation_cases; no_inference",
            "evaluation-launch": "starts_queued_local_policy_and_judge_inference",
            "evaluations": "reads_retained_evaluation_metadata",
            "evaluation-targets": "reads_terminal_local_checkpoint_metadata; review_hashes_selected_bytes",
            "validation-compare": "retains_metadata_and_queues_checkpoint_hashing_and_recorded_judge_replay; no_inference_or_training",
            "validation-comparison-status": "reads_retained_comparison_metadata; no_resubmission_or_weight_hashing",
            "nomination-plan": "reviews_selection_metadata; no_weight_hashing_or_writes",
            "nomination-prepare": "retains_reviewed_checkpoint_selection; no_weight_copy_or_inference",
            "nominations": "reads_bounded_saved_nomination_metadata; no_weight_hashing_or_writes",
            "nomination-status": "reads_saved_nomination_and_selection_metadata; no_weight_hashing",
            "nomination-config": "restores_exact_saved_selection_config_with_blank_test_cases; no_writes_or_inference",
            "evaluation-status": "reads_metadata; --verify replays_judge_and_hashes_weights",
            "evaluation-stop": "requests_cooperative_evaluation_stop; training_unchanged",
        },
        "launch_requires": ["run_id", "expected_plan_sha256"],
        "paid_launch_requires_allow_paid": True,
        "remote_submission_receipts": {
            "providers": ["runpod", "modal"],
            "inspection": "retained_identities_and_uncertainty_only",
            "automatic_restart": False,
            "cleanup_acknowledgement_verifies_billing_stopped": False,
            "runpod_wheel": "private_reviewed_snapshot_verified_before_remote_install",
        },
        "checkpoint_resume_supported": True,
        "checkpoint_validation_comparison": {
            "supported": True,
            "queue_required": True,
            "max_history_entries": MAX_COMPARISON_RECORDS,
            "partition": "validation",
            "order": ["success_rate", "judge_score", "earliest_saved_step"],
            "automatic_artifact_promotion": False,
            "model_inference": False,
        },
        "checkpoint_nomination": {
            "supported": True,
            "requires": "complete_checkpoint_recommendation_and_exact_nomination_review",
            "test_requires": "matching_receipt_and_checkpoint_bytes, reviewed_test_provenance, no_training_or_validation_overlap",
            "automatic_promotion": False,
            "copies_weights": False,
            "max_validation_history_entries": MAX_COMPARISON_RECORDS,
        },
        "checkpoint_retention": {
            "maximum_count": MAX_STUDIO_CHECKPOINTS,
            "periodic_provider": "local",
            "periodic_methods": ["grpo", "gspo", "dapo"],
            "default_interval": 0,
        },
        "local_evaluation": evaluation_schema(),
        "checkpoint_recovery": {
            "providers": ["local"],
            "methods": list(METHODS),
            "requires": "own_worker_receipt, matching_code, terminal_parent, remaining_budget",
            "launch": "separate_exact_plan_review",
        },
        "log_sources": list(LOG_SOURCES),
        "dataset_inspection": {
            "sample_flag": "--include-samples",
            "max_samples": 3,
            "max_sample_characters": 4096,
            "length_unit": "unicode_characters",
            "requires_config": False,
        },
        "credentials": "Environment or backend-managed credential profile; never put credentials in the config file.",
    }
