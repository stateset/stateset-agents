"""Run the fine-tune job on Modal.

Two decisions shape this module.

**Artifacts ship, not code.** The container installs a pinned, published
``stateset-agents[training]`` rather than syncing the local working tree. That
is what makes a remote run reproducible and what stops this executor rotting
every time the local dependency matrix shifts. The cost is real: testing an
unreleased change remotely needs a dev release. It also means the job must be
importable from the wheel — which is why it lives in
``stateset_agents.training.sft`` and not in ``scripts/`` (excluded from the
wheel by ``[tool.setuptools.packages.find]``).

**Status comes from the work, never from having submitted.** The job's own
return value decides SUCCEEDED or FAILED, and a run that produces no adapter
fails even if the container exited cleanly. An earlier version reported
success on submission alone; a user would then point ``serve --checkpoint`` at
a directory that was never written, and discover the problem far from its
cause.

The ``modal`` SDK is optional and resolved at call time, so installing it
later works without restarting.
"""

from __future__ import annotations

import asyncio
import inspect
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

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
_DEFAULT_MOUNT = "/outputs"


def _running_version() -> str:
    from stateset_agents import __version__

    return str(__version__)


def _remote_entrypoint(payload: dict[str, Any]) -> dict[str, Any]:
    """The body that executes inside the container.

    Imports from the installed package only — it has no checkout to fall back
    on. Returns the job outcome plus the adapter's location in the mounted
    volume so the caller can download it.
    """
    from stateset_agents.training.sft import run_sft_job

    outcome: dict[str, Any] = run_sft_job(payload)
    produced = Path(outcome["output_dir"])
    outcome["artifacts"] = (
        sorted(str(p.relative_to(produced)) for p in produced.rglob("*") if p.is_file())
        if produced.exists()
        else []
    )
    return outcome


@dataclass
class _ModalJob:
    spec: RemoteJobSpec
    status: JobStatus
    logs: list[str] = field(default_factory=list)
    fetched: Path | None = None


class ModalExecutor(RemoteExecutor):
    """Executes the job on Modal-provisioned GPU compute."""

    name = "modal"
    compute_model = "rented-serverless-gpu"
    verification_status = "transport-unverified"
    #: Modal's own GPU vocabulary.
    DEFAULT_GPU = "A10G"

    def __init__(self, remote_mount: str = _DEFAULT_MOUNT) -> None:
        self._jobs: dict[str, _ModalJob] = {}
        #: Where the adapter volume is mounted inside the container.
        self.remote_mount = remote_mount

    # -- SDK plumbing ------------------------------------------------------

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
        """Construct the container image: a pinned install, nothing local."""
        sdk = self._require_sdk()
        version = spec.package_version or _running_version()
        return sdk.Image.debian_slim(python_version="3.10").pip_install(
            f"stateset-agents[training]=={version}"
        )

    @staticmethod
    def _volume_name(job_id: str) -> str:
        return f"stateset-sft-{job_id}"

    def _delete_volume(self, name: str) -> None:
        """Delete one executor-owned persistent volume."""
        sdk = self._require_sdk()
        try:
            deletion = sdk.Volume.objects.delete(name, allow_missing=True)
            if inspect.isawaitable(deletion):

                async def wait_for_deletion() -> Any:
                    return await deletion

                asyncio.run(wait_for_deletion())
        except Exception as exc:
            raise RemoteExecutionError.wrap(
                exc,
                f"could not delete Modal volume {name}",
                provider=self.name,
                volume=name,
            ) from exc

    def _run_remote(
        self, spec: RemoteJobSpec, job_id: str
    ) -> tuple[dict[str, Any], Any, str]:
        """Provision and run the job. Returns (outcome, volume).

        The single seam not covered by CI — everything the executor *decides*
        is tested around it, but the network transport itself is verified
        manually against a live account.
        """
        sdk = self._require_sdk()
        image = self.build_image(spec)
        volume_name = self._volume_name(job_id)
        volume = sdk.Volume.from_name(volume_name, create_if_missing=True)
        app = sdk.App(_APP_NAME)

        function = app.function(
            image=image,
            gpu=spec.gpu or self.DEFAULT_GPU,
            timeout=spec.timeout_s,
            volumes={self.remote_mount: volume},
        )(_remote_entrypoint)

        payload = spec.to_dict()
        # Redirect the job's output into the mounted volume; the caller's
        # requested output_dir is a *local* path and means nothing in there.
        payload["output_dir"] = f"{self.remote_mount.rstrip('/')}/{job_id}"

        try:
            with app.run():
                outcome = function.remote(payload)
        except Exception:
            # The caller never receives the handle on this path, so cleanup
            # must happen here rather than in submit().
            self._delete_volume(volume_name)
            raise

        return outcome, volume, volume_name

    # -- Executor interface ------------------------------------------------

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        self.validate_spec(spec)
        job_id = uuid.uuid4().hex
        handle = JobHandle(provider=self.name, job_id=job_id)

        try:
            outcome, volume, volume_name = self._run_remote(spec, job_id)
        except RemoteExecutionError:
            raise
        except Exception as exc:  # provider SDK / transport failure
            raise RemoteExecutionError.wrap(
                exc, "failed to run job on Modal", provider=self.name
            ) from exc

        logs = list(outcome.get("logs", []))

        if outcome.get("returncode", 1) != 0:
            self._delete_volume(volume_name)
            self._jobs[job_id] = _ModalJob(spec, JobStatus.FAILED, logs)
            return handle

        if spec.dry_run:
            # Nothing is written on a dry run — that is the whole point.
            self._delete_volume(volume_name)
            self._jobs[job_id] = _ModalJob(spec, JobStatus.SUCCEEDED, logs)
            return handle

        try:
            downloaded = self._download(volume, job_id, spec.output_dir)
        except Exception as exc:
            logs.append(f"failed to download adapter: {exc}")
            try:
                self._delete_volume(volume_name)
            except Exception as cleanup_exc:  # noqa: BLE001 - retain both failures
                logs.append(
                    f"failed to delete Modal volume {volume_name}: {cleanup_exc}"
                )
            self._jobs[job_id] = _ModalJob(spec, JobStatus.FAILED, logs)
            return handle

        try:
            self._delete_volume(volume_name)
        except Exception as exc:
            logs.append(f"failed to delete Modal volume {volume_name}: {exc}")
            self._jobs[job_id] = _ModalJob(spec, JobStatus.FAILED, logs)
            return handle

        if not downloaded:
            logs.append(
                "job exited cleanly but produced no artifacts — "
                "nothing was written to the output volume"
            )
            self._jobs[job_id] = _ModalJob(spec, JobStatus.FAILED, logs)
            return handle

        self._jobs[job_id] = _ModalJob(
            spec, JobStatus.SUCCEEDED, logs, fetched=spec.output_dir
        )
        return handle

    def _download(self, volume: Any, job_id: str, dest: Path) -> list[Path]:
        """Copy the adapter out of the volume onto local disk."""
        volume.reload()
        written: list[Path] = []
        for entry in volume.iterdir(job_id, recursive=True):
            relative = Path(entry.path).relative_to(job_id)
            target = dest / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "wb") as handle:
                for chunk in volume.read_file(entry.path):
                    handle.write(chunk)
            written.append(target)
        return written

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
        if not job.status.is_terminal:
            job.status = JobStatus.CANCELLED
