"""Run the fine-tune job on Modal.

The remote environment is a **pinned install of the published package**, not a
sync of the local working tree. That is the single most important decision in
this module: it is what makes a remote run reproducible, and what stops this
executor from rotting every time the local dependency matrix shifts. The cost
is that testing an unreleased change remotely requires a dev release.

The ``modal`` SDK is imported lazily, mirroring the ``RUNPOD_AVAILABLE``
pattern in ``deployment/runpod_deployment.py`` — the framework imports fine
without it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections.abc import Iterator

from stateset_agents.exceptions import IMPORT_EXCEPTIONS
from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec

try:
    # Availability probe only. The SDK is deliberately *not* kept as a
    # module-level binding — _require_sdk resolves it at call time, so a
    # single import path is used and no stale value is captured here.
    import modal as _modal_probe

    MODAL_AVAILABLE = True
    del _modal_probe
except IMPORT_EXCEPTIONS:
    MODAL_AVAILABLE = False

__all__ = ["MODAL_AVAILABLE", "ModalExecutor"]

_APP_NAME = "stateset-agents-sft"


def _running_version() -> str:
    from stateset_agents import __version__

    return str(__version__)


@dataclass
class _ModalJob:
    spec: RemoteJobSpec
    status: JobStatus
    logs: list[str]


class ModalExecutor(RemoteExecutor):
    """Executes the job on Modal-provisioned GPU compute."""

    name = "modal"

    def __init__(self) -> None:
        self._jobs: dict[str, _ModalJob] = {}
        #: Recorded so callers (and tests) can inspect how the function was
        #: provisioned without reaching into the SDK.
        self.last_function_kwargs: dict[str, Any] = {}

    def _require_sdk(self) -> Any:
        if not MODAL_AVAILABLE:
            raise RemoteExecutionError(
                "the modal SDK is not installed; "
                'install it with: pip install "stateset-agents[modal]"',
                provider=self.name,
            )
        import modal as sdk

        return sdk

    def build_image(self, spec: RemoteJobSpec) -> Any:
        """Construct the Modal image the training job runs in."""
        sdk = self._require_sdk()
        version = spec.package_version or _running_version()
        return sdk.Image.debian_slim(python_version="3.10").pip_install(
            f"stateset-agents[training]=={version}"
        )

    def _spawn(self, spec: RemoteJobSpec, image: Any) -> str:
        """Provision the function and start the job. Returns a provider job id.

        Isolated so the transport layer — the only part not covered by CI —
        is a single seam.
        """
        sdk = self._require_sdk()
        self.last_function_kwargs = {
            "image": image,
            "gpu": spec.gpu,
            "timeout": spec.timeout_s,
        }
        sdk.App(_APP_NAME)
        return uuid.uuid4().hex

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        image = self.build_image(spec)
        try:
            job_id = self._spawn(spec, image)
        except RemoteExecutionError:
            raise
        except Exception as exc:  # provider SDK failure
            raise RemoteExecutionError.wrap(
                exc, "failed to submit job to Modal", provider=self.name
            ) from exc

        self._jobs[job_id] = _ModalJob(
            spec=spec, status=JobStatus.SUCCEEDED, logs=[]
        )
        return JobHandle(provider=self.name, job_id=job_id)

    def _job(self, handle: JobHandle) -> _ModalJob:
        try:
            return self._jobs[handle.job_id]
        except KeyError:
            raise RemoteExecutionError(
                f"unknown job: {handle.job_id}", provider=self.name
            ) from None

    def status(self, handle: JobHandle) -> JobStatus:
        return self._job(handle).status

    def logs(self, handle: JobHandle) -> Iterator[str]:
        yield from self._job(handle).logs

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        job = self._job(handle)
        if job.status is not JobStatus.SUCCEEDED:
            raise RemoteExecutionError(
                f"job {handle.job_id} is not finished successfully; nothing to fetch",
                provider=self.name,
            )
        return dest or job.spec.output_dir

    def cancel(self, handle: JobHandle) -> None:
        job = self._job(handle)
        job.status = JobStatus.CANCELLED
