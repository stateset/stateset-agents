"""Run the fine-tune job on this machine, via subprocess.

This is the reference implementation of :class:`RemoteExecutor`. It exists
for two reasons: it keeps the abstraction honest (an interface with a single
implementation tends to become that implementation with extra indirection),
and it makes the whole submit -> poll -> fetch path testable on CPU-only CI,
since ``sft_from_curated.py`` prints its plan and exits 0 without a GPU.

It is also genuinely useful: users who *do* own a GPU get the same command
as everyone else.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec

__all__ = ["LocalExecutor"]

#: The job, as an installed module. Deliberately not ``scripts/
#: sft_from_curated.py``: ``scripts*`` is excluded from the wheel, so a remote
#: worker cannot run it. Using the module here means the local and remote
#: providers execute byte-identical code.
_SFT_MODULE = "stateset_agents.training.sft"
_HARVEST_MODULE = "stateset_agents.training.harvest"


@dataclass
class _LocalJob:
    """Bookkeeping for one subprocess run."""

    spec: RemoteJobSpec
    status: JobStatus
    logs: list[str]


class LocalExecutor(RemoteExecutor):
    """Executes the job synchronously in a subprocess on the local machine."""

    name = "local"
    compute_model = "self-managed-local"
    verification_status = "unit-tested"
    supported_job_kinds = frozenset({"sft", "harvest"})

    def __init__(self) -> None:
        self._jobs: dict[str, _LocalJob] = {}
        self._counter = 0

    def _entrypoint_args(self, job_kind: str = "sft") -> list[str]:
        """The interpreter arguments that invoke the job's module.

        Isolated so tests can substitute a stand-in process.
        """
        return ["-m", _HARVEST_MODULE if job_kind == "harvest" else _SFT_MODULE]

    def _job(self, handle: JobHandle) -> _LocalJob:
        try:
            return self._jobs[handle.job_id]
        except KeyError:
            raise RemoteExecutionError(
                f"unknown job: {handle.job_id}", provider=self.name
            ) from None

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        self.validate_spec(spec)
        self._counter += 1
        job_id = str(self._counter)
        handle = JobHandle(provider=self.name, job_id=job_id)

        cmd = [
            sys.executable,
            *self._entrypoint_args(spec.job_kind),
            *spec.to_cli_args(),
        ]
        try:
            completed = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=spec.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired:
            self._jobs[job_id] = _LocalJob(
                spec=spec,
                status=JobStatus.FAILED,
                logs=[f"job timed out after {spec.timeout_s}s"],
            )
            return handle
        except OSError as exc:
            raise RemoteExecutionError.wrap(
                exc, "failed to start local training subprocess", provider=self.name
            ) from exc

        output = (completed.stdout or "") + (completed.stderr or "")
        self._jobs[job_id] = _LocalJob(
            spec=spec,
            status=(
                JobStatus.SUCCEEDED if completed.returncode == 0 else JobStatus.FAILED
            ),
            logs=output.splitlines(),
        )
        return handle

    def status(self, handle: JobHandle) -> JobStatus:
        return self._job(handle).status

    def logs(self, handle: JobHandle) -> Iterator[str]:
        yield from self._job(handle).logs

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        job = self._jobs.get(handle.job_id)
        if job is None or not job.status.is_terminal:
            raise RemoteExecutionError(
                f"job {handle.job_id} is not finished successfully; nothing to fetch",
                provider=self.name,
            )
        # A FAILED job's artifacts (if any) are already at output_dir; the
        # eval gate in particular fails jobs AFTER saving them.
        # The subprocess wrote straight to the requested output_dir — there is
        # no transfer step for a local run.
        return job.spec.output_dir

    def cancel(self, handle: JobHandle) -> None:
        job = self._job(handle)
        if not job.status.is_terminal:
            job.status = JobStatus.CANCELLED
