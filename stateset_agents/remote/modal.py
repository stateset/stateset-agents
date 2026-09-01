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
import os
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
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
    #: Modal's own current GPU vocabulary.
    DEFAULT_GPU = "A10"

    def __init__(
        self,
        remote_mount: str = _DEFAULT_MOUNT,
        *,
        secret_names: Sequence[str] | None = None,
        region: str | None = None,
    ) -> None:
        self._jobs: dict[str, _ModalJob] = {}
        #: Where the adapter volume is mounted inside the container.
        mount = remote_mount.strip().rstrip("/")
        mount_path = PurePosixPath(mount)
        if not mount or not mount_path.is_absolute() or ".." in mount_path.parts:
            raise ValueError("Modal remote_mount must be an absolute container path")
        self.remote_mount = mount
        if secret_names is None:
            secret_names = os.environ.get("STATESET_MODAL_SECRET_NAMES", "").split(",")
        self.secret_names = tuple(name.strip() for name in secret_names if name.strip())
        configured_region = region or os.environ.get("STATESET_MODAL_REGION")
        self.region = configured_region.strip() if configured_region else None

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

    @staticmethod
    def _gpu_request(spec: RemoteJobSpec) -> str:
        """Render Modal's ``GPU[:count]`` resource string."""
        gpu = spec.gpu or ModalExecutor.DEFAULT_GPU
        if ":" in gpu:
            if spec.gpu_count != 1:
                raise RemoteExecutionError(
                    "set Modal GPU count either in --gpu or --gpu-count, not both",
                    provider="modal",
                )
            return gpu
        return f"{gpu}:{spec.gpu_count}" if spec.gpu_count > 1 else gpu

    def _secrets(self, sdk: Any) -> list[Any]:
        """Resolve configured Modal Secret objects without reading their values."""
        return [sdk.Secret.from_name(name) for name in self.secret_names]

    def _upload_dataset(self, volume: Any, spec: RemoteJobSpec, job_id: str) -> str:
        """Upload the local dataset and return its mounted container path."""
        relative = f"inputs/{job_id}/{spec.dataset.name}"
        with volume.batch_upload() as upload:
            upload.put_file(str(spec.dataset), f"/{relative}")
        return f"{self.remote_mount}/{relative}"

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

        try:
            remote_dataset = self._upload_dataset(volume, spec, job_id)
            payload = spec.to_dict()
            payload["dataset"] = remote_dataset
            # The caller's output_dir is a local path and means nothing in the
            # container. Inputs and outputs use separate prefixes so dataset
            # transport can never be mistaken for a produced adapter.
            payload["output_dir"] = f"{self.remote_mount}/outputs/{job_id}"

            def invoke(payload: dict[str, Any]) -> dict[str, Any]:
                outcome = _remote_entrypoint(payload)
                volume.commit()
                return outcome

            function_options: dict[str, Any] = {
                "image": image,
                "gpu": self._gpu_request(spec),
                "timeout": spec.timeout_s,
                "volumes": {self.remote_mount: volume},
                # ``invoke`` closes over the hydrated Volume so it can commit
                # outputs before the local client starts downloading them.
                "serialized": True,
            }
            secrets = self._secrets(sdk)
            if secrets:
                function_options["secrets"] = secrets
            if self.region:
                function_options["region"] = self.region
            function = app.function(**function_options)(invoke)

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
        if not spec.dataset.is_file():
            raise RemoteExecutionError(
                f"Modal dataset must be a regular file: {spec.dataset}",
                provider=self.name,
            )
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
            fetched: Path | None = None
            if outcome.get("artifacts"):
                try:
                    downloaded = self._download(
                        volume, f"outputs/{job_id}", spec.output_dir
                    )
                    if downloaded:
                        fetched = spec.output_dir
                except Exception as exc:
                    logs.append(f"failed to download failure artifacts: {exc}")
            try:
                self._delete_volume(volume_name)
            except Exception as exc:
                logs.append(f"failed to delete Modal volume {volume_name}: {exc}")
            self._jobs[job_id] = _ModalJob(
                spec, JobStatus.FAILED, logs, fetched=fetched
            )
            return handle

        if spec.dry_run:
            # Nothing is written on a dry run — that is the whole point.
            self._delete_volume(volume_name)
            self._jobs[job_id] = _ModalJob(spec, JobStatus.SUCCEEDED, logs)
            return handle

        try:
            downloaded = self._download(volume, f"outputs/{job_id}", spec.output_dir)
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
            self._jobs[job_id] = _ModalJob(
                spec, JobStatus.FAILED, logs, fetched=spec.output_dir
            )
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

    def _download(self, volume: Any, remote_dir: str, dest: Path) -> list[Path]:
        """Copy the adapter out of the volume onto local disk."""
        volume.reload()
        written: list[Path] = []
        prefix = PurePosixPath(remote_dir.strip("/"))
        destination = dest.resolve()
        for entry in volume.iterdir(remote_dir, recursive=True):
            entry_type = getattr(entry, "type", None)
            if entry_type is not None and getattr(entry_type, "name", "") != "FILE":
                continue
            entry_path = PurePosixPath(str(entry.path).lstrip("/"))
            try:
                relative = entry_path.relative_to(prefix)
            except ValueError as exc:
                raise RemoteExecutionError(
                    f"Modal returned an artifact outside {remote_dir}: {entry.path}",
                    provider=self.name,
                ) from exc
            if ".." in relative.parts or not relative.parts:
                raise RemoteExecutionError(
                    f"Modal returned an unsafe artifact path: {entry.path}",
                    provider=self.name,
                )
            target = (destination / Path(*relative.parts)).resolve()
            try:
                target.relative_to(destination)
            except ValueError as exc:
                raise RemoteExecutionError(
                    f"Modal artifact escapes output directory: {entry.path}",
                    provider=self.name,
                ) from exc
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
        if job.status is not JobStatus.SUCCEEDED and job.fetched is None:
            raise RemoteExecutionError(
                f"job {handle.job_id} is not finished successfully; nothing to fetch",
                provider=self.name,
            )
        return dest or job.fetched or job.spec.output_dir

    def cancel(self, handle: JobHandle) -> None:
        job = self._job(handle)
        if not job.status.is_terminal:
            job.status = JobStatus.CANCELLED
