"""Private, plan-bound receipts for allocations made during remote submission."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import threading
import time
from pathlib import Path
from typing import Any

from stateset_agents.finetune_workspace import read_run_plan, write_json

_LIMIT = 128 * 1024
_EVENT_LIMIT = 128
_EVENTS = frozenset(
    {
        "allocation_intent",
        "allocated",
        "cleanup_acknowledged",
        "cleanup_confirmed",
        "cleanup_unconfirmed",
        "execution_finished",
    }
)
_KINDS = frozenset({"pod", "volume", "function_call"})
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_JOB = re.compile(r"[A-Za-z0-9_-]{1,128}")


def _sync_directory(run: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(run, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _event(value: Any, provider: str) -> dict[str, Any]:
    allowed = {
        "schema_version",
        "provider",
        "job_id",
        "event",
        "resource_kind",
        "resource_id",
        "resource_name",
        "attempt",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Invalid provider allocation event")
    if (
        type(value.get("schema_version")) is not int
        or value["schema_version"] != 1
        or value.get("provider") != provider
        or not isinstance(value.get("job_id"), str)
        or not _JOB.fullmatch(value["job_id"])
        or not isinstance(value.get("event"), str)
        or value["event"] not in _EVENTS
        or not isinstance(value.get("resource_kind"), str)
        or value["resource_kind"] not in _KINDS
    ):
        raise ValueError("Invalid provider allocation event")
    identifier = value.get("resource_id")
    if identifier is not None and (
        not isinstance(identifier, str) or not _IDENTIFIER.fullmatch(identifier)
    ):
        raise ValueError("Invalid provider allocation identity")
    if value["event"] != "allocation_intent" and identifier is None:
        raise ValueError("Provider allocation event requires a resource identity")
    name = value.get("resource_name")
    if name is not None and (
        not isinstance(name, str) or not _IDENTIFIER.fullmatch(name)
    ):
        raise ValueError("Invalid provider allocation name")
    if value["resource_kind"] in {"pod", "volume"} and name != (
        "stateset-sft-" + value["job_id"]
    ):
        raise ValueError("Allocation name differs from the owned submission")
    if "attempt" in value and (
        type(value["attempt"]) is not int or not 1 <= value["attempt"] <= 10_000
    ):
        raise ValueError("Invalid allocation attempt")
    return dict(value)


class SubmissionJournal:
    """Persist allocation identity before a synchronous provider call returns.

    A run has one exclusive worker and one journal owner. An exclusive initial
    create refuses resubmission after a crash; an in-process lock serializes SDK
    callbacks. Reading these receipts never authenticates or controls a provider.
    """

    def __init__(self, run: Path, provider: str, *, durable_supported: bool) -> None:
        self.run = run
        plan, self.plan_sha256 = read_run_plan(run)
        if provider not in {"runpod", "modal", "river"} or (
            plan.get("config", {}).get("provider") != provider
        ):
            raise ValueError("Provider submission differs from the reviewed plan")
        self._lock = threading.RLock()
        self._record: dict[str, Any] = {
            "schema_version": 1,
            "plan_sha256": self.plan_sha256,
            "provider": provider,
            "durable_receipts": bool(durable_supported),
            "job_id": None,
            "submission_state": "submitting",
            "observed_status": None,
            "created_at": time.time(),
            "updated_at": time.time(),
            "events": [],
        }
        descriptor = os.open(
            run / "provider-submission.json",
            os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(self._record, stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        _sync_directory(run)

    def _write(self) -> None:
        if read_run_plan(self.run)[1] != self.plan_sha256:
            raise ValueError("Reviewed plan changed during provider submission")
        self._record["updated_at"] = time.time()
        if len(json.dumps(self._record, allow_nan=False).encode()) > _LIMIT:
            raise ValueError("Provider allocation journal exceeds its record limit")
        write_json(self.run / "provider-submission.json", self._record)
        _sync_directory(self.run)

    def record_handle(self, provider: str, job_id: str) -> None:
        """Persist a logical executor handle independently of job completion."""
        with self._lock:
            if (
                provider != self._record["provider"]
                or not isinstance(job_id, str)
                or not _JOB.fullmatch(job_id)
            ):
                raise ValueError("Provider handle differs from its submission")
            if self._record["job_id"] not in {None, job_id}:
                raise ValueError("Provider submission changed its logical job identity")
            self._record["job_id"] = job_id
            self._write()
            write_json(
                self.run / "provider-job.json", {"provider": provider, "job_id": job_id}
            )
            _sync_directory(self.run)

    def publish(self, value: dict[str, Any]) -> None:
        """Acknowledge an SDK event only after its private receipt is flushed."""
        with self._lock:
            event = _event(value, self._record["provider"])
            if not self._record["durable_receipts"]:
                raise ValueError(
                    "This provider did not declare durable submission receipts"
                )
            if self._record["job_id"] not in {None, event["job_id"]}:
                raise ValueError("Provider submission changed its logical job identity")
            if len(self._record["events"]) >= _EVENT_LIMIT:
                raise ValueError("Provider allocation journal exceeds its event limit")
            self._record["job_id"] = event["job_id"]
            self._record["events"].append({**event, "observed_at": time.time()})
            self._write()
            self.record_handle(event["provider"], event["job_id"])

    def finish(self, status: str) -> None:
        """Record the provider result, without certifying cleanup or quality."""
        if status not in {"succeeded", "failed", "cancelled"}:
            raise ValueError("Provider result must have an observed terminal status")
        with self._lock:
            self._record["submission_state"] = "returned"
            self._record["observed_status"] = status
            self._write()


def inspect_submission(run: Path) -> dict[str, Any]:
    """Read retained allocation identities and uncertainty without provider calls."""
    path = run / "provider-submission.json"
    if not path.exists() and not path.is_symlink():
        return {"available": False, "live_status_checked": False}
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    if path.is_symlink():
        raise ValueError("Provider journal must be a regular private record")
    with os.fdopen(os.open(path, flags), "rb") as stream:
        info = os.fstat(stream.fileno())
        if (
            not stat.S_ISREG(info.st_mode)
            or info.st_size > _LIMIT
            or (os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077)
        ):
            raise ValueError(
                "Provider journal must be a bounded regular private record"
            )
        raw = stream.read(_LIMIT + 1)
    if len(raw) > _LIMIT:
        raise ValueError("Provider journal exceeds its record limit")
    record = json.loads(raw)
    plan, digest = read_run_plan(run)
    if (
        not isinstance(record, dict)
        or type(record.get("schema_version")) is not int
        or record["schema_version"] != 1
        or record.get("plan_sha256") != digest
        or not isinstance(record.get("provider"), str)
        or record["provider"] not in {"runpod", "modal", "river"}
        or record.get("provider") != plan.get("config", {}).get("provider")
        or (
            record.get("job_id") is not None
            and (
                not isinstance(record["job_id"], str)
                or not _JOB.fullmatch(record["job_id"])
            )
        )
        or (
            record.get("observed_status") is not None
            and (
                not isinstance(record["observed_status"], str)
                or record["observed_status"] not in {"succeeded", "failed", "cancelled"}
            )
        )
        or type(record.get("durable_receipts")) is not bool
        or not isinstance(record.get("submission_state"), str)
        or record["submission_state"] not in {"submitting", "returned"}
        or (record["submission_state"] == "returned")
        != (record.get("observed_status") is not None)
        or not isinstance(record.get("events"), list)
        or len(record["events"]) > _EVENT_LIMIT
    ):
        raise ValueError("Provider journal differs from the reviewed plan")
    for name in ("created_at", "updated_at"):
        value = record.get(name)
        if (
            isinstance(value, bool)
            or not isinstance(value, (float, int))
            or not math.isfinite(value)
        ):
            raise ValueError("Invalid provider observation time")
    resources: dict[tuple[str, str], dict[str, Any]] = {}
    intents: dict[tuple[str, str | None, int | None], dict[str, Any]] = {}
    for item in record["events"]:
        if not isinstance(item, dict):
            raise ValueError("Invalid provider allocation event")
        event = _event(
            {key: value for key, value in item.items() if key != "observed_at"},
            record["provider"],
        )
        observed = item.get("observed_at")
        if (
            isinstance(observed, bool)
            or not isinstance(observed, (float, int))
            or not math.isfinite(observed)
        ):
            raise ValueError("Invalid provider observation time")
        if event["job_id"] != record.get("job_id"):
            raise ValueError("Provider allocation has a different logical identity")
        intent_key = (
            event["resource_kind"],
            event.get("resource_name"),
            event.get("attempt"),
        )
        if event["event"] == "allocation_intent":
            intents[intent_key] = event
        else:
            intents.pop(intent_key, None)
            key = (event["resource_kind"], event["resource_id"])
            resources[key] = {
                "kind": event["resource_kind"],
                "id": event["resource_id"],
                "name": event.get("resource_name"),
                "state": event["event"],
            }
    entries = list(resources.values())
    unconfirmed = (
        not record["durable_receipts"] or not record["events"] or bool(intents)
    ) or any(
        entry["state"] != "cleanup_confirmed"
        and not (
            entry["kind"] == "function_call" and entry["state"] == "execution_finished"
        )
        for entry in entries
    )
    return {
        "available": True,
        "provider": record["provider"],
        "job_id": record.get("job_id"),
        "plan_sha256": digest,
        "receipt_sha256": hashlib.sha256(raw).hexdigest(),
        "durable_receipts": record["durable_receipts"],
        "submission_state": record["submission_state"],
        "observed_status": record.get("observed_status"),
        "resources": entries,
        "unresolved_intents": list(intents.values()),
        "cleanup_required_or_unconfirmed": unconfirmed,
        "live_status_checked": False,
        "automatic_restart_supported": False,
        "provider_cleanup_verified": False,
    }
