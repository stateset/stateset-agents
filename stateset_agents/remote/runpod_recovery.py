"""Durable allocation identities and process ownership for RunPod recovery."""

from __future__ import annotations

import os
import re
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeGuard

from stateset_agents.finetune_workspace import read_json, write_json
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.utils.snapshot_files import exclusive_file_lock, sync_directory

RentalKind = Literal["serving", "training", "chat"]
_PREFIXES = {
    "serving": "stateset-serve-",
    "training": "stateset-sft-",
    "chat": "stateset-chat-",
}

_ID = re.compile(r"[a-zA-Z0-9_-]+")


def valid_pod_id(value: object) -> TypeGuard[str]:
    """Accept a provider ID that identifies one pod in the RunPod API."""
    return isinstance(value, str) and _ID.fullmatch(value) is not None


def session_lease_dir(kind: Literal["serving", "chat"]) -> Path:
    """Return the local state directory for persistent session allocation records."""
    root = Path(os.environ.get("XDG_STATE_HOME") or Path.home() / ".local" / "state")
    return root / "stateset-agents" / f"runpod-{kind}"


class LeaseBusy(RuntimeError):
    """A starter or another recovery process owns this attempt."""


@dataclass(frozen=True)
class PodLease:
    """One pending allocation, identified by the exact name sent to RunPod."""

    path: Path

    @contextmanager
    def starting(self, metadata: dict[str, Any] | None = None) -> Iterator[None]:
        """Publish an allocation intent and hold its lock until the caller exits."""
        try:
            with self.locked():
                self.record(None, metadata=metadata)
                yield
        except LeaseBusy as exc:
            raise RemoteExecutionError(
                f"RunPod allocation {self.path.stem} is already in use"
            ) from exc

    @contextmanager
    def locked(self) -> Iterator[None]:
        """Exclude concurrent provisioning and recovery for this rental."""
        lock = self.path.with_suffix(".lock")
        with ExitStack() as stack:
            try:
                self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
                stack.enter_context(exclusive_file_lock(lock, LeaseBusy))
            except (OSError, ValueError) as exc:
                raise RemoteExecutionError(
                    f"Cannot lock RunPod allocation record {self.path}: {exc}"
                ) from exc
            try:
                yield
            finally:
                stack.close()
                # Names are unique to a single allocation. After its record is
                # gone, a concurrent scanner rechecks existence under the lock.
                if not self.path.exists():
                    try:
                        lock.unlink(missing_ok=True)
                    except OSError:
                        pass  # A leftover lock file carries no allocation evidence.

    def record(
        self, pod_id: str | None, *, metadata: dict[str, Any] | None = None
    ) -> None:
        """Create the pre-allocation record or bind a returned pod ID."""
        if pod_id is not None:
            self.record_id(pod_id)
            return
        if self.path.exists():
            raise RemoteExecutionError(
                f"A pending allocation already has the name {self.path.stem}"
            )
        row: dict[str, Any] = {"name": self.path.stem, "pod_id": None}
        if metadata is not None:
            row["metadata"] = metadata
        self._write(row)

    def record_id(self, pod_id: str) -> None:
        """Bind the provider response to the original durable allocation intent."""
        row = self.read_record()
        if not valid_pod_id(pod_id) or row["pod_id"] not in (None, pod_id):
            raise RemoteExecutionError(
                f"Invalid or conflicting pod ID for {self.path.stem}"
            )
        row["pod_id"] = pod_id
        self._write(row)

    def _write(self, row: dict[str, Any]) -> None:
        try:
            write_json(self.path, row)
            sync_directory(self.path.parent)
        except (OSError, ValueError) as exc:
            raise RemoteExecutionError(
                f"Cannot record RunPod allocation {self.path}: {exc}"
            ) from exc

    def read_record(self) -> dict[str, Any]:
        """Read the pending ID, checking that the record belongs to this name."""
        row = read_json(self.path)
        if (
            set(row) - {"name", "pod_id", "metadata"}
            or row.get("name") != self.path.stem
            or "pod_id" not in row
            or not isinstance(row.get("metadata", {}), dict)
        ):
            raise ValueError("Invalid RunPod allocation record")
        pod_id = row["pod_id"]
        if pod_id is not None and not valid_pod_id(pod_id):
            raise ValueError("Invalid RunPod pod ID in RunPod allocation record")
        return row

    def clear(self) -> None:
        """Retire the record after endpoint handoff or acknowledged deletion."""
        try:
            self.path.unlink()
            sync_directory(self.path.parent)
        except FileNotFoundError:
            return
        except OSError as exc:
            raise RemoteExecutionError(
                f"Cannot retire RunPod allocation record {self.path}: {exc}"
            ) from exc


@dataclass(frozen=True)
class RecoveryResult:
    """Outcome of inspecting or cleaning up one interrupted rental."""

    name: str
    status: str
    detail: str
    record: dict[str, Any] | None = None


def recover_pods(
    root: Path,
    get_api: Callable[[], Any] | None,
    *,
    kind: RentalKind,
    terminate: bool = False,
) -> list[RecoveryResult]:
    """Inspect interrupted launches, optionally deleting uniquely matched pods."""
    if terminate and get_api is None:
        raise ValueError("RunPod cleanup requires a provider client")
    prefix = _PREFIXES[kind]
    pattern = re.compile(
        re.escape(prefix)
        + r"[0-9a-f]{12}"
        + (r"(?:-[1-9][0-9]*)?" if kind == "training" else "")
    )
    results: list[RecoveryResult] = []
    try:
        paths = sorted(path for path in root.iterdir() if path.match(f"{prefix}*.json"))
    except FileNotFoundError:
        return results
    except OSError as exc:
        raise RemoteExecutionError(
            f"Cannot read RunPod allocation directory {root}: {exc}"
        ) from exc
    for path in paths:
        name = path.stem
        lease = PodLease(path)
        try:
            with lease.locked():
                if not path.exists():  # Startup finished after the directory scan.
                    continue
                if not pattern.fullmatch(name):
                    raise ValueError("Unexpected RunPod allocation record name")
                record = lease.read_record()
                recorded_id = record["pod_id"]
                if get_api is None:
                    results.append(
                        RecoveryResult(
                            name, "pending", recorded_id or "pod ID unknown", record
                        )
                    )
                    continue
                api = get_api()
                pods = api.list_pods()
                matches = [p for p in pods if p.get("name") == name]
                if len(matches) != 1:
                    raise ValueError(
                        f"Found {len(matches)} RunPod pods with the recorded name"
                    )
                pod_id = matches[0].get("id")
                if not valid_pod_id(pod_id):
                    raise ValueError("RunPod did not return a usable pod ID")
                if sum(p.get("id") == pod_id for p in pods) != 1:
                    raise ValueError(
                        "RunPod returned conflicting entries for the pod ID"
                    )
                if recorded_id is not None and recorded_id != pod_id:
                    raise ValueError("RunPod pod ID differs from the recorded ID")
                if terminate:
                    # Remember the matched ID even if the DELETE response gets lost.
                    if recorded_id is None:
                        lease.record_id(pod_id)
                    api.terminate_pod(pod_id)
                    lease.clear()
                status = "termination accepted" if terminate else "recoverable"
                result = RecoveryResult(name, status, pod_id)
        except LeaseBusy:
            result = RecoveryResult(name, "busy", "owner or recovery in progress")
        except Exception as exc:
            result = RecoveryResult(name, "unresolved", str(exc))
        results.append(result)
    return results


def training_pod_name(job_id: str, attempt: int) -> str:
    """Use a distinct provider name for each attempt of a logical training job."""
    base = f"stateset-sft-{job_id}"
    return base if attempt == 1 else f"{base}-{attempt}"
