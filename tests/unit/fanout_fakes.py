"""A scripted in-memory backend, so every failure path can be driven on demand."""

from __future__ import annotations

import json
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

from stateset_agents.fanout.backend import (
    Allocation,
    AllocationLostError,
    BackendError,
    FileEntry,
    LogChunk,
    ProcessState,
    RunBackend,
)
from stateset_agents.fanout.manifest import FanoutManifest, Resources


@dataclass
class Script:
    """How one attempt of one run behaves. Defaults to a clean success."""

    #: Raised by successive allocate calls before anything is created.
    allocate_errors: list[Exception] = field(default_factory=list)
    #: Raised by allocate calls after the sandbox already exists, the way a lost
    #: connection hides a create that landed.
    ambiguous_errors: list[Exception] = field(default_factory=list)
    write_error: Exception | None = None
    #: Raised by successive write_file calls.
    write_errors: list[Exception] = field(default_factory=list)
    start_error: Exception | None = None
    #: Raised by successive start calls.
    start_errors: list[Exception] = field(default_factory=list)
    #: Raised by successive read_file calls.
    read_errors: list[Exception] = field(default_factory=list)
    #: Polls that report running before the process exits.
    polls: int = 0
    #: Exceptions raised by successive process() calls.
    poll_errors: list[Exception] = field(default_factory=list)
    exit_code: int | None = 0
    status: str = "exited"
    #: Written as result.json. "default" builds a matching success; None omits it.
    result: Any = "default"
    metrics: dict[str, Any] = field(default_factory=lambda: {"reward": 1.0})
    stdout: list[bytes] = field(default_factory=list)
    stderr: list[bytes] = field(default_factory=list)
    #: Whether process() reports log sizes, as the real exec agent does.
    report_log_bytes: bool = False
    files: dict[str, bytes] = field(default_factory=dict)
    cost_cents: float | None = 2.0
    loss_reason: str | None = None
    #: The lifetime the platform actually granted. None means it did not say.
    lifetime_s: float | None = None


@dataclass
class _Live:
    run_id: str
    script: Script
    polls_seen: int = 0
    process_id: str | None = None
    stdout_sent: int = 0
    stderr_sent: int = 0


class FakeBackend(RunBackend):
    name = "fake"
    metered = True

    def __init__(self, scripts: dict[str, list[Script]] | None = None) -> None:
        self.scripts = {run: list(items) for run, items in (scripts or {}).items()}
        self.live: dict[str, _Live] = {}
        self.released: list[str] = []
        self.started: list[dict[str, Any]] = []
        self.written: dict[str, dict[str, bytes]] = defaultdict(dict)
        self.allocations: list[tuple[str, str]] = []
        self.allocate_calls = 0
        self.create_keys: list[str] = []
        self.log_reads = 0
        self.max_live = 0
        self.release_errors: list[Exception] = []
        #: The order of stops and cost reads.
        self.calls: list[str] = []
        self._by_key: dict[str, Allocation] = {}
        #: Called on every allocate, for tests that must observe that moment.
        self.on_allocate: Any = None
        self.on_write: Any = None
        self.on_poll: Any = None
        self._lock = threading.Lock()
        self._counter = 0

    def _script_for(self, run_id: str) -> Script:
        queue = self.scripts.get(run_id)
        if not queue:
            return Script()
        return queue[0]

    def allocate(
        self, *, run_id: str, resources: Resources, idempotency_key: str
    ) -> Allocation:
        if self.on_allocate:
            self.on_allocate(run_id, idempotency_key)
        with self._lock:
            self.allocate_calls += 1
            self.create_keys.append(idempotency_key)
            existing = self._by_key.get(idempotency_key)
            if existing is not None:
                # The same key always answers with the same sandbox.
                errors = self.live[existing.id].script.ambiguous_errors
                if errors:
                    raise errors.pop(0)
                return existing
            script = self._script_for(run_id)
            if script.allocate_errors:
                raise script.allocate_errors.pop(0)
            if self.scripts.get(run_id):
                self.scripts[run_id].pop(0)
            self._counter += 1
            allocation_id = f"alloc-{self._counter}"
            self.live[allocation_id] = _Live(run_id=run_id, script=script)
            self.allocations.append((run_id, allocation_id))
            active = len(self.live) - len(self.released)
            self.max_live = max(self.max_live, active)
            allocation = Allocation(
                id=allocation_id,
                root=f"/root/{run_id}",
                lifetime_s=script.lifetime_s,
            )
            self._by_key[idempotency_key] = allocation
            if script.ambiguous_errors:
                raise script.ambiguous_errors.pop(0)
        return allocation

    def _get(self, allocation_id: str) -> _Live:
        return self.live[allocation_id]

    def write_file(self, allocation_id: str, path: str, content: bytes) -> None:
        live = self._get(allocation_id)
        if live.script.write_error:
            raise live.script.write_error
        if live.script.write_errors:
            raise live.script.write_errors.pop(0)
        self.written[allocation_id][path] = content
        if self.on_write:
            self.on_write(allocation_id, path)

    def worker_argv(
        self, manifest: FanoutManifest, *, root: str, worker_args: list[str]
    ) -> list[str]:
        return ["worker", *worker_args]

    def start(
        self,
        allocation_id: str,
        argv: list[str],
        *,
        env: dict[str, str],
        timeout_seconds: int,
        name: str,
    ) -> str:
        live = self._get(allocation_id)
        if live.script.start_error:
            raise live.script.start_error
        if live.script.start_errors:
            raise live.script.start_errors.pop(0)
        live.process_id = f"proc-{allocation_id}"
        self.started.append(
            {
                "allocation_id": allocation_id,
                "argv": argv,
                "env": env,
                "timeout_seconds": timeout_seconds,
                "name": name,
                "run_id": live.run_id,
            }
        )
        return live.process_id

    def process(self, allocation_id: str, process_id: str) -> ProcessState:
        if self.on_poll:
            self.on_poll(allocation_id)
        live = self._get(allocation_id)
        if live.script.poll_errors:
            raise live.script.poll_errors.pop(0)
        sizes: dict[str, int | None] = {"stdout_bytes": None, "stderr_bytes": None}
        if live.script.report_log_bytes:
            sizes = {
                "stdout_bytes": sum(len(chunk) for chunk in live.script.stdout),
                "stderr_bytes": sum(len(chunk) for chunk in live.script.stderr),
            }
        if live.polls_seen < live.script.polls:
            live.polls_seen += 1
            return ProcessState(status="running", running=True, **sizes)
        return ProcessState(
            status=live.script.status,
            running=False,
            exit_code=live.script.exit_code,
            **sizes,
        )

    def logs(
        self, allocation_id: str, process_id: str, stream: str, offset: int
    ) -> LogChunk:
        self.log_reads += 1
        live = self._get(allocation_id)
        chunks = live.script.stdout if stream == "stdout" else live.script.stderr
        attr = "stdout_sent" if stream == "stdout" else "stderr_sent"
        index = getattr(live, attr)
        if index >= len(chunks):
            return LogChunk(data=b"", next_offset=offset, eof=True)
        setattr(live, attr, index + 1)
        data = chunks[index]
        return LogChunk(data=data, next_offset=offset + len(data), eof=False)

    def _default_result(self, live: _Live) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "run_id": live.run_id,
            "status": "succeeded" if live.script.exit_code == 0 else "failed",
            "exit_code": live.script.exit_code,
            "metrics": live.script.metrics,
        }

    def read_file(self, allocation_id: str, path: str) -> bytes | None:
        live = self._get(allocation_id)
        if live.script.read_errors:
            raise live.script.read_errors.pop(0)
        if path.endswith("/output/result.json"):
            if live.script.result is None:
                return None
            value = (
                self._default_result(live)
                if live.script.result == "default"
                else live.script.result
            )
            return json.dumps(value).encode() if isinstance(value, dict) else value
        for relative, content in live.script.files.items():
            if path.endswith("/output/" + relative):
                return content
        return None

    def list_files(self, allocation_id: str, path: str) -> list[FileEntry]:
        live = self._get(allocation_id)
        return [
            FileEntry(path=f"{path}/{relative}", size=len(content))
            for relative, content in live.script.files.items()
        ]

    def cost_cents(self, allocation_id: str) -> float | None:
        self.calls.append("cost")
        return self._get(allocation_id).script.cost_cents

    def loss_reason(self, allocation_id: str) -> str | None:
        return self._get(allocation_id).script.loss_reason

    def release(self, allocation_id: str) -> None:
        self.calls.append("release")
        with self._lock:
            if self.release_errors:
                raise self.release_errors.pop(0)
            if allocation_id not in self.released:
                self.released.append(allocation_id)


__all__ = [
    "AllocationLostError",
    "BackendError",
    "FakeBackend",
    "Script",
]
