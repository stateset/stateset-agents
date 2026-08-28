"""Run the fine-tune on Fireworks AI's managed fine-tuning service.

Fireworks sits between the two shapes already in this package. Like River,
it is a *managed* service: you do not rent a machine and we do not ship
``stateset_agents.training.sft`` anywhere — you upload a dataset, create a
supervised fine-tuning job, and Fireworks trains a LoRA addon on its own
fleet. Like RunPod, the job is genuinely asynchronous: ``submit()`` returns
as soon as the job resource exists, and the job id remains meaningful to
other processes and to the Fireworks dashboard.

The call sequence is::

    datasets.create -> datasets.upload
      -> supervised_fine_tuning_jobs.create
      -> supervised_fine_tuning_jobs.get (poll)
      -> models.get_download_endpoint (best effort)

and, optionally, ``deploy()``::

    deployments.create(enable_addons=True) -> lora.load

.. warning::

   **NOT LIVE-VERIFIED.** This adapter is written against the real
   ``fireworks-ai`` SDK — the resource names, keyword arguments, and
   response fields below come from the installed, generated client rather
   than from prose docs, so the *shapes* are trustworthy. What has not been
   exercised against the live service is its *behaviour*: state-transition
   timing, whether a PEFT addon's weights are actually downloadable for
   your account, and the deployment/addon-load sequence. See
   ``docs/FIREWORKS_PROVIDER.md``.

Two things worth knowing before you use it:

**The weights may or may not land locally.** ``fetch()`` always writes
``fireworks_checkpoint.json`` (a pointer to the tuned model on Fireworks)
plus the usual ``stateset_manifest.json``. It *also* attempts to download
the addon's files through ``models.get_download_endpoint``; when that
succeeds you get real adapter weights and ``stateset-agents serve
--checkpoint`` works, and when it does not you still have a usable pointer.
``weights_downloaded`` in the pointer says which happened.

**Provider-resource spec fields do not apply to training.** ``gpu``,
``gpu_count``, ``container_disk_gb``, ``cloud_type``, and
``network_volume_id`` describe rented machines. Fireworks picks the
training hardware itself, so they are ignored (and logged as ignored)
rather than treated as errors — the same spec should be submittable to any
provider. They come back into play only at ``deploy()`` time, which does
rent hardware, and which takes its accelerator explicitly.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import (
    JobHandle,
    JobStatus,
    RemoteJobResult,
    RemoteJobSpec,
)
from stateset_agents.remote.ledger import CostEntry, record_entry

__all__ = [
    "CHECKPOINT_POINTER_NAME",
    "FIREWORKS_ACCOUNT_ENV",
    "FIREWORKS_API_KEY_ENV",
    "FIREWORKS_VERBOSE_ENV",
    "FireworksExecutor",
]

#: Environment variable holding the Fireworks API key (``fw_...``).
FIREWORKS_API_KEY_ENV = "FIREWORKS_API_KEY"

#: Environment variable holding the Fireworks account id. Every control-plane
#: resource is scoped to an account, and the SDK will not guess one.
FIREWORKS_ACCOUNT_ENV = "FIREWORKS_ACCOUNT_ID"

#: When set (any non-empty value), progress lines are ALSO printed to stderr
#: as they are observed, rather than only after the job resolves. Same
#: flashlight as ``STATESET_RIVER_VERBOSE``: a fine-tune queued behind other
#: tenants looks identical to a hung client without it.
FIREWORKS_VERBOSE_ENV = "STATESET_FIREWORKS_VERBOSE"

#: Written by ``fetch()`` alongside (or in place of) adapter weights.
CHECKPOINT_POINTER_NAME = "fireworks_checkpoint.json"

#: OpenAI-compatible inference base URL for deployed models.
FIREWORKS_INFERENCE_BASE_URL = "https://api.fireworks.ai/inference/v1"

#: Durable metadata for asynchronous jobs. Unlike credentials, the job spec
#: and provider resource ids are safe to persist and are required to fetch an
#: addon after the submitting CLI process exits.
DEFAULT_FIREWORKS_STATE_DIR = (
    Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    / "stateset-agents"
    / "fireworks-jobs"
)
_STATE_SCHEMA_VERSION = 1

logger = logging.getLogger(__name__)

#: Spec fields that describe rented training hardware. Fireworks schedules
#: the fine-tune itself, so a spec carrying them is accepted and they are
#: reported as ignored.
_IGNORED_SPEC_FIELDS = (
    "gpu",
    "gpu_count",
    "container_disk_gb",
    "cloud_type",
    "network_volume_id",
)

#: Fireworks' ``JobState`` enum, mapped onto our five-state lifecycle.
#: Anything absent from this table is treated as PENDING — an unrecognised
#: state is more likely a new queueing stage than a silent success.
_JOB_STATES: dict[str, JobStatus] = {
    "JOB_STATE_UNSPECIFIED": JobStatus.PENDING,
    "JOB_STATE_CREATING": JobStatus.PENDING,
    "JOB_STATE_CREATING_INPUT_DATASET": JobStatus.PENDING,
    "JOB_STATE_PENDING": JobStatus.PENDING,
    "JOB_STATE_VALIDATING": JobStatus.PENDING,
    "JOB_STATE_RE_QUEUEING": JobStatus.PENDING,
    "JOB_STATE_IDLE": JobStatus.PENDING,
    "JOB_STATE_PAUSED": JobStatus.PENDING,
    "JOB_STATE_RUNNING": JobStatus.RUNNING,
    "JOB_STATE_WRITING_RESULTS": JobStatus.RUNNING,
    # Cancellation in flight is not yet terminal; reporting CANCELLED here
    # would let `wait()` return before the job actually stopped.
    "JOB_STATE_CANCELLING": JobStatus.RUNNING,
    "JOB_STATE_COMPLETED": JobStatus.SUCCEEDED,
    # Early stopping is Fireworks doing what it was asked to do: the addon
    # exists and is trained. A success, not a failure.
    "JOB_STATE_EARLY_STOPPED": JobStatus.SUCCEEDED,
    "JOB_STATE_FAILED": JobStatus.FAILED,
    "JOB_STATE_EXPIRED": JobStatus.FAILED,
    "JOB_STATE_CANCELLED": JobStatus.CANCELLED,
    "JOB_STATE_DELETING": JobStatus.CANCELLED,
    "JOB_STATE_DELETING_CLEANING_UP": JobStatus.CANCELLED,
}


def _verbose_log(message: str) -> None:
    """Log, and additionally echo to stderr under the verbose flag."""
    logger.info(message)
    if os.environ.get(FIREWORKS_VERBOSE_ENV):
        import sys

        print(f"[fireworks] {message}", file=sys.stderr, flush=True)


def _download(url: str, dest: Path) -> None:
    """Stream a signed URL to ``dest``.

    Module-level (rather than a method) so tests can substitute it without
    reaching into the executor, and so the network boundary is one obvious
    function.
    """
    import requests

    with requests.get(url, stream=True, timeout=300) as response:
        response.raise_for_status()
        with dest.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)


@dataclass
class _FireworksJob:
    """What this process knows about a submitted job."""

    spec: RemoteJobSpec
    dataset_id: str
    job_name: str
    status: JobStatus = JobStatus.PENDING
    logs: list[str] = field(default_factory=list)
    output_model: str | None = None
    cost_usd: float | None = None
    submitted_at: float = field(default_factory=time.time)
    duration_s: float | None = None
    #: Last progress line appended, so repeated polls at the same percent do
    #: not fill the log with identical lines.
    last_progress: str | None = None


class FireworksExecutor(RemoteExecutor):
    """Runs a :class:`RemoteJobSpec` as a Fireworks supervised fine-tune."""

    name = "fireworks"
    durable_handles = True
    managed_deployments = True
    result_kind = "hosted_pointer_or_local_artifacts"

    def __init__(
        self,
        client: Any | None = None,
        account_id: str | None = None,
        ledger_path: Path | None = None,
        state_dir: Path | None = None,
    ) -> None:
        self._client = client
        self._account_id = account_id
        self.ledger_path = ledger_path
        # A custom ledger normally means a test or isolated application run;
        # colocating state beside it avoids leaking test records into the
        # user's global cache without making persistence opt-in in production.
        self.state_dir = (
            Path(state_dir)
            if state_dir is not None
            else (
                Path(ledger_path).parent / "fireworks-jobs"
                if ledger_path is not None
                else DEFAULT_FIREWORKS_STATE_DIR
            )
        )
        self._jobs: dict[str, _FireworksJob] = {}

    # -- durable job metadata ---------------------------------------------

    def _state_path(self, job_id: str) -> Path:
        safe_id = re.sub(r"[^A-Za-z0-9._-]", "_", job_id)
        return self.state_dir / f"{safe_id}.json"

    def _persist_record(self, job_id: str, record: _FireworksJob) -> None:
        """Atomically persist the non-secret state needed by a later process."""
        payload = {
            "schema_version": _STATE_SCHEMA_VERSION,
            "provider": self.name,
            "job_id": job_id,
            "spec": record.spec.to_dict(),
            "dataset_id": record.dataset_id,
            "job_name": record.job_name,
            "status": record.status.value,
            "logs": record.logs,
            "output_model": record.output_model,
            "cost_usd": record.cost_usd,
            "submitted_at": record.submitted_at,
            "duration_s": record.duration_s,
            "last_progress": record.last_progress,
        }
        target = self._state_path(job_id)
        temporary = target.with_suffix(".tmp")
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary.write_text(
                json.dumps(payload, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary.replace(target)
        except OSError as exc:
            logger.warning("could not persist Fireworks job %s: %s", job_id, exc)
            temporary.unlink(missing_ok=True)

    def _load_record(self, job_id: str) -> _FireworksJob | None:
        path = self._state_path(job_id)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != _STATE_SCHEMA_VERSION:
                raise ValueError("unsupported state schema")
            if payload.get("provider") != self.name or payload.get("job_id") != job_id:
                raise ValueError("state identity mismatch")
            record = _FireworksJob(
                spec=RemoteJobSpec.from_dict(payload["spec"]),
                dataset_id=str(payload["dataset_id"]),
                job_name=str(payload["job_name"]),
                status=JobStatus(str(payload.get("status", "pending"))),
                logs=[str(line) for line in payload.get("logs", [])],
                output_model=payload.get("output_model"),
                cost_usd=payload.get("cost_usd"),
                submitted_at=float(payload.get("submitted_at", time.time())),
                duration_s=payload.get("duration_s"),
                last_progress=payload.get("last_progress"),
            )
        except FileNotFoundError:
            return None
        except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
            raise RemoteExecutionError(
                f"durable metadata for Fireworks job {job_id} is invalid: {exc}",
                provider=self.name,
                job_id=job_id,
            ) from exc
        self._jobs[job_id] = record
        return record

    # -- client ------------------------------------------------------------

    @property
    def account_id(self) -> str:
        account = self._account_id or os.environ.get(FIREWORKS_ACCOUNT_ENV)
        if not account:
            raise RemoteExecutionError(
                f"no Fireworks account id: set {FIREWORKS_ACCOUNT_ENV} (the "
                "account slug from https://app.fireworks.ai, e.g. 'my-org') "
                "or pass account_id=",
                provider=self.name,
            )
        return account

    def _get_client(self) -> Any:
        """Return the SDK client, constructing it from the environment once.

        An injected client (tests, or a caller with custom retry settings)
        is used as-is and skips the credential check, which is what makes
        the fakes in the test-suite possible without a key.
        """
        if self._client is not None:
            return self._client

        api_key = os.environ.get(FIREWORKS_API_KEY_ENV)
        if not api_key:
            raise RemoteExecutionError(
                f"no Fireworks API key: set {FIREWORKS_API_KEY_ENV} "
                "(create one at https://app.fireworks.ai/settings/users/api-keys)",
                provider=self.name,
            )
        try:
            from fireworks import Fireworks
        except ImportError as exc:  # pragma: no cover - exercised by hand
            raise RemoteExecutionError.wrap(
                exc,
                "the fireworks provider needs the Fireworks SDK: "
                "pip install 'stateset-agents[fireworks]'",
                provider=self.name,
            ) from exc

        self._client = Fireworks(api_key=api_key, account_id=self.account_id)
        return self._client

    # -- submit ------------------------------------------------------------

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        """Upload the dataset, create the fine-tuning job, return its handle."""
        self.validate_spec(spec)
        self._warn_ignored_fields(spec)
        # Credentials first: failing on a missing key after validating a
        # large dataset wastes the user's time.
        client = self._get_client()
        rows = _read_chat_rows(Path(spec.dataset))

        dataset_id = _dataset_id(spec)
        _verbose_log(f"creating dataset {dataset_id} ({len(rows)} examples)")
        try:
            client.datasets.create(
                dataset_id=dataset_id,
                dataset={
                    "display_name": f"stateset {Path(spec.dataset).name}",
                    "format": "CHAT",
                    "example_count": str(len(rows)),
                    "user_uploaded": {},
                },
            )
        except Exception as exc:  # noqa: BLE001 - provider SDK error
            raise RemoteExecutionError.wrap(
                exc,
                f"could not create Fireworks dataset {dataset_id}",
                provider=self.name,
                dataset_id=dataset_id,
            ) from exc

        _verbose_log(f"uploading {len(rows)} examples to {dataset_id}")
        try:
            with Path(spec.dataset).open("rb") as handle:
                client.datasets.upload(dataset_id, file=handle)
        except Exception as exc:  # noqa: BLE001 - provider SDK error
            raise RemoteExecutionError.wrap(
                exc,
                f"could not upload data to Fireworks dataset {dataset_id}",
                provider=self.name,
                dataset_id=dataset_id,
            ) from exc

        _verbose_log(f"creating fine-tuning job on {spec.base_model}")
        try:
            job = client.supervised_fine_tuning_jobs.create(
                dataset=dataset_id,
                base_model=spec.base_model,
                display_name=f"stateset {Path(spec.output_dir).name}",
                epochs=spec.num_epochs,
                learning_rate=spec.learning_rate,
                lora_rank=spec.lora_r,
                max_context_length=spec.max_length,
                batch_size=spec.per_device_batch_size,
                gradient_accumulation_steps=spec.gradient_accumulation_steps,
            )
        except Exception as exc:  # noqa: BLE001 - provider SDK error
            raise RemoteExecutionError.wrap(
                exc,
                f"could not create a Fireworks fine-tuning job on {spec.base_model}",
                provider=self.name,
                base_model=spec.base_model,
                dataset_id=dataset_id,
            ) from exc

        job_name = getattr(job, "name", "") or ""
        job_id = _resource_id(job_name)
        self._jobs[job_id] = _FireworksJob(
            spec=spec,
            dataset_id=dataset_id,
            job_name=job_name,
            output_model=getattr(job, "output_model", None),
            logs=[f"fireworks job {job_id} submitted on {spec.base_model}"],
        )
        self._persist_record(job_id, self._jobs[job_id])
        _verbose_log(f"submitted job {job_id}")
        return JobHandle(provider=self.name, job_id=job_id)

    def _warn_ignored_fields(self, spec: RemoteJobSpec) -> None:
        ignored = [name for name in _IGNORED_SPEC_FIELDS if _is_set(spec, name)]
        if ignored:
            logger.info(
                "fireworks schedules its own training hardware; ignoring "
                "machine-shaped spec fields: %s",
                ", ".join(ignored),
            )

    # -- polling -----------------------------------------------------------

    def wait(self, handle: JobHandle, poll_interval_s: float = 15.0) -> RemoteJobResult:
        """Poll to completion, on a cadence suited to a managed service.

        The base class defaults to one second, which is right for a pod whose
        log you are tailing. A Fireworks fine-tune sits in a queue and then
        runs for minutes to hours; polling its control plane 3600 times an
        hour learns nothing the 15s cadence does not.
        """
        return super().wait(handle, poll_interval_s=poll_interval_s)

    def _remote_job(self, job_id: str) -> Any:
        client = self._get_client()
        try:
            return client.supervised_fine_tuning_jobs.get(job_id)
        except Exception as exc:  # noqa: BLE001 - provider SDK error
            raise RemoteExecutionError.wrap(
                exc,
                f"could not read Fireworks job {job_id}",
                provider=self.name,
                job_id=job_id,
            ) from exc

    def status(self, handle: JobHandle) -> JobStatus:
        """Poll Fireworks for the job's state, recording progress as it goes.

        Works for any job id, not only ones this process submitted — the job
        lives on Fireworks, so a handle from a previous run is still pollable.
        """
        remote = self._remote_job(handle.job_id)
        state = getattr(remote, "state", None) or "JOB_STATE_UNSPECIFIED"
        status = _JOB_STATES.get(state, JobStatus.PENDING)

        record = self._jobs.get(handle.job_id) or self._load_record(handle.job_id)
        if record is not None:
            self._record_progress(record, remote, state, status)
        return status

    def _record_progress(
        self,
        record: _FireworksJob,
        remote: Any,
        state: str,
        status: JobStatus,
    ) -> None:
        """Turn one poll into at most one log line, and settle terminal state."""
        line = _progress_line(state, remote)
        if line != record.last_progress:
            record.last_progress = line
            record.logs.append(line)
            _verbose_log(line)

        record.status = status
        record.output_model = (
            getattr(remote, "output_model", None) or record.output_model
        )
        record.cost_usd = _money(getattr(remote, "estimated_cost", None))
        if status.is_terminal and record.duration_s is None:
            record.duration_s = max(0.0, time.time() - record.submitted_at)
            message = getattr(getattr(remote, "status", None), "message", None)
            if status is JobStatus.FAILED and message:
                record.logs.append(f"fireworks: {message}")
            self._record_cost(record)
        self._persist_record(_resource_id(record.job_name), record)

    def logs(self, handle: JobHandle) -> Iterator[str]:
        """Yield the progress lines observed while polling.

        Fireworks does not expose trainer stdout for a fine-tuning job
        through this API, so these are *progress events* — state changes and
        percent-complete — not the training log. Said plainly here rather
        than implied by an empty iterator.
        """
        record = self._jobs.get(handle.job_id) or self._load_record(handle.job_id)
        if record is None:
            return
        yield from record.logs

    def job_cost(self, handle: JobHandle) -> tuple[float | None, float | None]:
        """Wall-clock duration and Fireworks' own cost estimate.

        The dollar figure is the ``estimatedCost`` the job resource reports,
        not a price computed here. When the job reports none, this returns
        None — unknown, never zero.
        """
        record = self._jobs.get(handle.job_id) or self._load_record(handle.job_id)
        if record is None:
            return (None, None)
        return (record.duration_s, record.cost_usd)

    def _record_cost(self, record: _FireworksJob) -> None:
        record_entry(
            CostEntry(
                provider=self.name,
                job_id=_resource_id(record.job_name),
                base_model=record.spec.base_model,
                gpu="fireworks-managed",
                gpu_count=0,
                cost_per_hr=None,
                duration_s=(
                    round(record.duration_s, 1)
                    if record.duration_s is not None
                    else None
                ),
                cost_usd=record.cost_usd,
                status=record.status.value,
            ),
            path=self.ledger_path,
        )

    # -- fetch -------------------------------------------------------------

    def _local_record(self, handle: JobHandle) -> _FireworksJob:
        record = self._jobs.get(handle.job_id) or self._load_record(handle.job_id)
        if record is None:
            raise RemoteExecutionError(
                f"job {handle.job_id} has no local durable metadata, so its "
                "training spec is unknown; submit it with this version or "
                "download the addon from the Fireworks console",
                provider=self.name,
                job_id=handle.job_id,
            )
        return record

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        """Write the checkpoint pointer and manifest; download weights if offered.

        The pointer is always written — it is what identifies the tuned addon
        on Fireworks. The weights are best-effort: a PEFT addon is not always
        downloadable, and a failed download must not cost the user the record
        of a fine-tune they paid for.
        """
        record = self._local_record(handle)
        status = self.status(handle)
        if status is not JobStatus.SUCCEEDED:
            raise RemoteExecutionError(
                f"job {handle.job_id} is not finished successfully "
                f"({status.value}); nothing to fetch",
                provider=self.name,
                job_id=handle.job_id,
            )

        spec = record.spec
        output_dir = Path(dest) if dest is not None else Path(spec.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        downloaded = self._download_weights(record, output_dir)
        self._write_pointer(record, output_dir, downloaded)
        self._write_manifest(record, output_dir, downloaded)
        return output_dir

    def _download_weights(self, record: _FireworksJob, output_dir: Path) -> list[str]:
        """Best-effort download of the addon's files. Returns what landed."""
        if not record.output_model:
            logger.info("fireworks job reported no output model; pointer only")
            return []
        client = self._get_client()
        try:
            endpoint = client.models.get_download_endpoint(
                _resource_id(record.output_model)
            )
            urls = getattr(endpoint, "filename_to_signed_urls", None) or {}
        except Exception as exc:  # noqa: BLE001 - not fatal, pointer still works
            logger.info(
                "fireworks addon weights are not downloadable (%s); "
                "writing a pointer instead",
                exc,
            )
            return []

        downloaded: list[str] = []
        for filename, url in sorted(urls.items()):
            target = output_dir / filename
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                _download(url, target)
            except Exception as exc:  # noqa: BLE001 - the pointer still works
                logger.info("could not download %s from Fireworks: %s", filename, exc)
                # Weights are all-or-nothing: a half-written file, or a
                # directory holding one of two shards, would masquerade as a
                # loadable adapter and fail far from the real cause.
                target.unlink(missing_ok=True)
                for done in downloaded:
                    (output_dir / done).unlink(missing_ok=True)
                return []
            downloaded.append(filename)
        if downloaded:
            _verbose_log(f"downloaded {len(downloaded)} addon file(s) to {output_dir}")
        return downloaded

    def _write_pointer(
        self, record: _FireworksJob, output_dir: Path, downloaded: list[str]
    ) -> None:
        spec = record.spec
        note = (
            "The trained LoRA also lives on Fireworks; these local weights are a copy."
            if downloaded
            else (
                "Fireworks hosts the trained addon; this file is a pointer, not "
                "an adapter. Sample it through the Fireworks OpenAI-compatible "
                "API using the model id above — `stateset-agents serve "
                "--checkpoint` cannot load it."
            )
        )
        pointer = {
            "provider": self.name,
            "account": self.account_id,
            "job": record.job_name,
            "model": record.output_model,
            "base_model": spec.base_model,
            "dataset": record.dataset_id,
            "lora": {"rank": spec.lora_r},
            "num_epochs": spec.num_epochs,
            "learning_rate": spec.learning_rate,
            "max_context_length": spec.max_length,
            "estimated_cost_usd": record.cost_usd,
            "weights_downloaded": bool(downloaded),
            "files": downloaded,
            "inference_base_url": FIREWORKS_INFERENCE_BASE_URL,
            "note": note,
        }
        (output_dir / CHECKPOINT_POINTER_NAME).write_text(
            json.dumps(pointer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def _write_manifest(
        self, record: _FireworksJob, output_dir: Path, downloaded: list[str]
    ) -> None:
        from stateset_agents.training.lineage import (
            AdapterManifest,
            hash_dataset,
            write_manifest,
        )

        spec = record.spec
        digest, rows = hash_dataset(Path(spec.dataset))
        write_manifest(
            output_dir,
            AdapterManifest(
                base_model=spec.base_model,
                dataset_path=str(spec.dataset),
                dataset_sha256=digest,
                dataset_rows=rows,
                hyperparameters={
                    "provider": self.name,
                    "lora_r": spec.lora_r,
                    "num_epochs": spec.num_epochs,
                    "learning_rate": spec.learning_rate,
                    "max_length": spec.max_length,
                    "per_device_batch_size": spec.per_device_batch_size,
                    "fireworks_job": record.job_name,
                    "fireworks_model": record.output_model,
                    "weights_downloaded": bool(downloaded),
                },
                parent_adapter=spec.parent_adapter,
                package_version=spec.package_version,
            ),
        )

    # -- cancel ------------------------------------------------------------

    def cancel(self, handle: JobHandle) -> None:
        """Stop the job on Fireworks.

        The SDK exposes deletion rather than a cancel verb for supervised
        fine-tuning jobs (RFT jobs have ``cancel``), so that is what stops
        the billing.
        """
        client = self._get_client()
        try:
            client.supervised_fine_tuning_jobs.delete(handle.job_id)
        except Exception as exc:  # noqa: BLE001 - provider SDK error
            raise RemoteExecutionError.wrap(
                exc,
                f"could not cancel Fireworks job {handle.job_id}",
                provider=self.name,
                job_id=handle.job_id,
            ) from exc
        record = self._jobs.get(handle.job_id) or self._load_record(handle.job_id)
        if record is not None and not record.status.is_terminal:
            record.status = JobStatus.CANCELLED
            self._persist_record(handle.job_id, record)

    # -- deployment --------------------------------------------------------

    def deploy(
        self,
        handle: JobHandle,
        accelerator_type: str | None = None,
        accelerator_count: int | None = None,
        min_replica_count: int = 0,
        max_replica_count: int = 1,
    ) -> dict[str, Any]:
        """Serve the tuned addon on a Fireworks on-demand deployment.

        This is the step that actually rents hardware, and it bills for as
        long as the deployment exists — which is why it is a separate call
        and never part of ``submit()``. ``min_replica_count`` defaults to 0
        so an idle deployment scales to nothing.

        Returns the deployment name, the addon model id, and the
        OpenAI-compatible base URL to point a client at.
        """
        record = self._local_record(handle)
        status = self.status(handle)
        if status is not JobStatus.SUCCEEDED:
            raise RemoteExecutionError(
                f"job {handle.job_id} is not finished successfully "
                f"({status.value}); there is no addon to deploy",
                provider=self.name,
                job_id=handle.job_id,
            )
        if not record.output_model:
            raise RemoteExecutionError(
                f"job {handle.job_id} finished without reporting an output model",
                provider=self.name,
                job_id=handle.job_id,
            )

        client = self._get_client()
        create_kwargs: dict[str, Any] = {
            "base_model": record.spec.base_model,
            # Without this the deployment serves the base model only and the
            # LoRA load below has nowhere to land.
            "enable_addons": True,
            "display_name": f"stateset {Path(record.spec.output_dir).name}",
            "min_replica_count": min_replica_count,
            "max_replica_count": max_replica_count,
        }
        if accelerator_type:
            create_kwargs["accelerator_type"] = accelerator_type
        if accelerator_count:
            create_kwargs["accelerator_count"] = accelerator_count

        _verbose_log(f"creating deployment for {record.spec.base_model}")
        try:
            deployment = client.deployments.create(**create_kwargs)
        except Exception as exc:  # noqa: BLE001 - provider SDK error
            raise RemoteExecutionError.wrap(
                exc,
                f"could not create a Fireworks deployment for {record.spec.base_model}",
                provider=self.name,
                base_model=record.spec.base_model,
            ) from exc

        deployment_name = getattr(deployment, "name", "") or ""
        _verbose_log(f"loading addon {record.output_model} onto {deployment_name}")
        try:
            client.lora.load(model=record.output_model, deployment=deployment_name)
        except Exception as exc:  # noqa: BLE001 - provider SDK error
            raise RemoteExecutionError.wrap(
                exc,
                f"deployment {deployment_name} was created but the addon could "
                "not be loaded onto it; delete it with `stateset-agents "
                "fireworks-undeploy` so it stops billing",
                provider=self.name,
                deployment=deployment_name,
            ) from exc

        return {
            "deployment": deployment_name,
            "model": record.output_model,
            "base_url": FIREWORKS_INFERENCE_BASE_URL,
        }

    def undeploy(self, deployment: str) -> None:
        """Delete a deployment created by :meth:`deploy`, stopping its billing.

        Accepts either the bare deployment id or the full resource name.
        """
        client = self._get_client()
        deployment_id = _resource_id(deployment)
        try:
            client.deployments.delete(deployment_id)
        except Exception as exc:  # noqa: BLE001 - provider SDK error
            raise RemoteExecutionError.wrap(
                exc,
                f"could not delete Fireworks deployment {deployment_id}",
                provider=self.name,
                deployment=deployment_id,
            ) from exc


# --- helpers ------------------------------------------------------------


def _is_set(spec: RemoteJobSpec, field_name: str) -> bool:
    """True when a machine-shaped spec field carries a non-default value."""
    value = getattr(spec, field_name, None)
    if field_name == "gpu_count":
        return isinstance(value, int) and value > 1
    return value not in (None, "", 0)


def _resource_id(name: str) -> str:
    """Last path segment of a Fireworks resource name.

    The SDK's resource methods take bare ids (``sftj-1``) while responses
    carry full names (``accounts/acct/supervisedFineTuningJobs/sftj-1``).
    """
    return (name or "").rstrip("/").rsplit("/", 1)[-1]


def _dataset_id(spec: RemoteJobSpec) -> str:
    """A Fireworks-legal dataset id derived from the run.

    Fireworks ids are lowercase alphanumerics and dashes. A random suffix
    keeps repeat runs of the same output dir from colliding with an existing
    dataset, which the API rejects rather than overwrites.
    """
    stem = re.sub(r"[^a-z0-9-]+", "-", Path(spec.output_dir).name.lower()).strip("-")
    return f"stateset-{stem or 'sft'}-{uuid.uuid4().hex[:8]}"[:63]


def _read_chat_rows(dataset: Path) -> list[dict[str, Any]]:
    """Load and validate the JSONL, which is already Fireworks' CHAT format.

    Both this framework and Fireworks use ``{"messages": [{"role", "content"}]}``
    per line, so there is no conversion — only validation, done here so a
    malformed row fails before anything is uploaded or billed.
    """
    if not dataset.exists():
        raise RemoteExecutionError(
            f"dataset not found: {dataset}", provider="fireworks"
        )
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(
        dataset.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RemoteExecutionError.wrap(
                exc,
                f"{dataset}:{lineno} is not valid JSON",
                provider="fireworks",
            ) from exc
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(messages, list) or not messages:
            raise RemoteExecutionError(
                f"{dataset}:{lineno} has no 'messages' list; Fireworks' CHAT "
                "format needs one conversation per line",
                provider="fireworks",
            )
        rows.append(row)
    if not rows:
        raise RemoteExecutionError(f"dataset is empty: {dataset}", provider="fireworks")
    return rows


def _progress_line(state: str, remote: Any) -> str:
    """One human-readable line describing the job's current state."""
    label = state.removeprefix("JOB_STATE_").lower()
    progress = getattr(remote, "job_progress", None)
    parts = [f"fireworks job {label}"]
    percent = getattr(progress, "percent", None) if progress else None
    if percent is not None:
        parts.append(f"{percent}%")
    epoch = getattr(progress, "epoch", None) if progress else None
    if epoch is not None:
        parts.append(f"epoch {epoch}")
    tokens = getattr(progress, "output_tokens", None) if progress else None
    if tokens:
        parts.append(f"{tokens} tokens")
    return " - ".join(parts)


def _money(amount: Any) -> float | None:
    """Convert Fireworks' ``{units, nanos}`` money object to dollars.

    Returns None when the job reports no estimate — an unknown cost must
    read as unknown, not as free.
    """
    if amount is None:
        return None
    units = getattr(amount, "units", None)
    nanos = getattr(amount, "nanos", None)
    if units is None and nanos is None:
        return None
    try:
        return float(units or 0) + float(nanos or 0) / 1e9
    except (TypeError, ValueError):
        return None
