"""Fan out on this machine: one OS process and one scratch directory per run.

Same worker, same orchestration and same report as a sandbox fan-out, so a
manifest can be proved here for free before it spends anything. Isolation is only
a process and a directory, so this is for your own workloads, not untrusted code.

Process handles live in memory. After a driver crash the runs it started are not
reattachable: a resume counts them as lost and starts fresh ones, and stops the old
ones. Each process's id is kept on disk, and checked against its command line before
it is signalled, so a recycled process id can never be killed by mistake.
"""

from __future__ import annotations

import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import stateset_agents
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

__all__ = ["LocalBackend"]

WORKER_MODULE = "stateset_agents.fanout.worker"
PID_FILE = "pid"
_LOG_READ_BYTES = 1024 * 1024
_TERMINATE_GRACE_S = 5.0


@dataclass
class _Process:
    popen: subprocess.Popen[bytes]
    deadline: float
    stdout: Path
    stderr: Path
    timed_out: bool = False


class LocalBackend(RunBackend):
    """Runs each allocation as a subprocess in its own process group."""

    name = "local"
    metered = False

    def __init__(self, scratch_dir: str | Path) -> None:
        self._scratch = Path(scratch_dir).resolve()
        self._roots: dict[str, Path] = {}
        self._processes: dict[str, _Process] = {}
        self._by_key: dict[str, Allocation] = {}
        self._lock = threading.Lock()

    def _root(self, allocation_id: str) -> Path:
        try:
            return self._roots[allocation_id]
        except KeyError:
            raise AllocationLostError(
                f"unknown local allocation {allocation_id}"
            ) from None

    def _inside_root(self, allocation_id: str, path: str) -> Path:
        root = self._root(allocation_id)
        target = Path(path).resolve()
        if root not in target.parents and target != root:
            raise BackendError(f"path escapes the run directory: {path}")
        return target

    def allocate(
        self, *, run_id: str, resources: Resources, idempotency_key: str
    ) -> Allocation:
        with self._lock:
            existing = self._by_key.get(idempotency_key)
            if existing is not None:
                return existing
            allocation_id = f"local-{uuid.uuid4().hex[:12]}"
            root = self._scratch / allocation_id
            root.mkdir(parents=True)
            self._roots[allocation_id] = root
            allocation = Allocation(id=allocation_id, root=str(root))
            self._by_key[idempotency_key] = allocation
        return allocation

    def write_file(self, allocation_id: str, path: str, content: bytes) -> None:
        target = self._inside_root(allocation_id, path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)

    def worker_argv(
        self, manifest: FanoutManifest, *, root: str, worker_args: list[str]
    ) -> list[str]:
        # The local interpreter already has this package; nothing gets installed.
        return [sys.executable, "-m", WORKER_MODULE, *worker_args]

    def start(
        self,
        allocation_id: str,
        argv: list[str],
        *,
        env: dict[str, str],
        timeout_seconds: int,
        name: str,
    ) -> str:
        root = self._root(allocation_id)
        package_parent = str(Path(stateset_agents.__file__).resolve().parents[1])
        existing_path = os.environ.get("PYTHONPATH")
        process_env = {
            **os.environ,
            "PYTHONPATH": (
                f"{package_parent}{os.pathsep}{existing_path}"
                if existing_path
                else package_parent
            ),
            **env,
        }
        logs = root / ".logs"
        logs.mkdir()
        stdout, stderr = logs / "stdout.log", logs / "stderr.log"
        try:
            with stdout.open("wb") as out, stderr.open("wb") as err:
                popen = subprocess.Popen(
                    argv,
                    cwd=root,
                    env=process_env,
                    stdout=out,
                    stderr=err,
                    # Its own process group, so stopping reaches its children too.
                    start_new_session=True,
                )
        except OSError as exc:
            raise BackendError(f"cannot start {argv[0]}: {exc}") from exc
        (logs / PID_FILE).write_text(str(popen.pid))
        with self._lock:
            self._processes[allocation_id] = _Process(
                popen=popen,
                deadline=time.monotonic() + timeout_seconds,
                stdout=stdout,
                stderr=stderr,
            )
        return str(popen.pid)

    def _process(self, allocation_id: str) -> _Process:
        try:
            return self._processes[allocation_id]
        except KeyError:
            raise AllocationLostError(f"no local process for {allocation_id}") from None

    def process(self, allocation_id: str, process_id: str) -> ProcessState:
        entry = self._process(allocation_id)
        code = entry.popen.poll()
        if code is None:
            if time.monotonic() <= entry.deadline:
                return ProcessState(status="running", running=True)
            entry.timed_out = True
            self._terminate(entry)
            code = entry.popen.poll()
        if entry.timed_out:
            return ProcessState(status="timed_out", running=False, exit_code=code)
        return ProcessState(status="exited", running=False, exit_code=code)

    def logs(
        self, allocation_id: str, process_id: str, stream: str, offset: int
    ) -> LogChunk:
        entry = self._process(allocation_id)
        path = entry.stdout if stream == "stdout" else entry.stderr
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(_LOG_READ_BYTES)
        return LogChunk(
            data=data,
            next_offset=offset + len(data),
            eof=len(data) < _LOG_READ_BYTES,
        )

    def read_file(self, allocation_id: str, path: str) -> bytes | None:
        target = self._inside_root(allocation_id, path)
        try:
            return target.read_bytes()
        except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
            return None

    def list_files(self, allocation_id: str, path: str) -> list[FileEntry]:
        target = self._inside_root(allocation_id, path)
        if not target.is_dir():
            return []
        return [
            FileEntry(path=str(item), size=item.stat().st_size)
            for item in sorted(target.rglob("*"))
            if item.is_file()
        ]

    def cost_cents(self, allocation_id: str) -> float | None:
        # Your own machine: nothing here is metered or billed by the framework.
        return 0.0

    def loss_reason(self, allocation_id: str) -> str | None:
        return None

    @staticmethod
    def _terminate(entry: _Process) -> None:
        group = entry.popen.pid
        try:
            os.killpg(group, signal.SIGTERM)
            entry.popen.wait(timeout=_TERMINATE_GRACE_S)
        except ProcessLookupError:
            return
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        entry.popen.wait()

    def _stop_orphan(self, allocation_id: str) -> None:
        """Stop what an earlier driver left running, if it really is still ours."""
        if not re.fullmatch(r"local-[0-9a-f]+", allocation_id):
            raise BackendError(f"not a local allocation: {allocation_id}")
        root = self._scratch / allocation_id
        try:
            pid = int((root / ".logs" / PID_FILE).read_text())
        except (OSError, ValueError):
            return
        proc = Path("/proc") / str(pid)
        if not Path("/proc/self").exists():
            raise BackendError(
                f"cannot verify process {pid} on this platform; stop it by hand"
            )
        try:
            command_line = (proc / "cmdline").read_bytes()
        except OSError:
            return
        if str(root).encode() not in command_line:
            return
        for signum in (signal.SIGTERM, signal.SIGKILL):
            try:
                os.killpg(pid, signum)
            except ProcessLookupError:
                return
            deadline = time.monotonic() + _TERMINATE_GRACE_S
            while time.monotonic() < deadline:
                if not proc.exists():
                    return
                time.sleep(0.05)

    def release(self, allocation_id: str) -> None:
        with self._lock:
            entry = self._processes.pop(allocation_id, None)
            root = self._roots.pop(allocation_id, None)
        if entry is not None:
            self._terminate(entry)
        else:
            self._stop_orphan(allocation_id)
        target = root if root is not None else self._scratch / allocation_id
        shutil.rmtree(target, ignore_errors=True)
