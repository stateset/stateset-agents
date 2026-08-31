"""Together AI managed supervised fine-tuning integration."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec

TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"
TOGETHER_POINTER = "together_checkpoint.json"

_STATES = {
    "pending": JobStatus.PENDING,
    "queued": JobStatus.PENDING,
    "running": JobStatus.RUNNING,
    "training": JobStatus.RUNNING,
    "compressing": JobStatus.RUNNING,
    "uploading": JobStatus.RUNNING,
    "cancel_requested": JobStatus.RUNNING,
    "completed": JobStatus.SUCCEEDED,
    "succeeded": JobStatus.SUCCEEDED,
    "failed": JobStatus.FAILED,
    "error": JobStatus.FAILED,
    "cancelled": JobStatus.CANCELLED,
    "canceled": JobStatus.CANCELLED,
}


def _field(value: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        candidate = (
            value.get(name) if isinstance(value, dict) else getattr(value, name, None)
        )
        if candidate is not None:
            return candidate
    return default


class TogetherExecutor(RemoteExecutor):
    """Upload JSONL and run an asynchronous Together LoRA fine-tune."""

    name = "together"
    supported_job_kinds = frozenset({"sft"})
    durable_handles = True
    result_kind = "hosted_pointer_or_local_artifacts"
    compute_model = "managed-finetuning"
    verification_status = "unit-tested-live-lifecycle-pending"

    def __init__(self, client: Any | None = None) -> None:
        self._client_instance = client

    def _client(self) -> Any:
        if self._client_instance is not None:
            return self._client_instance
        key = os.environ.get(TOGETHER_API_KEY_ENV)
        if not key:
            raise RemoteExecutionError(
                f"set {TOGETHER_API_KEY_ENV} before using Together",
                provider=self.name,
            )
        try:
            from together import Together
        except ImportError as exc:
            raise RemoteExecutionError.wrap(
                exc,
                "the Together provider needs the Together SDK; install "
                "'stateset-agents[together]'",
                provider=self.name,
            ) from exc
        self._client_instance = Together(api_key=key)
        return self._client_instance

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        self.validate_spec(spec)
        client = self._client()
        try:
            uploaded = client.files.upload(
                file=str(spec.dataset), purpose="fine-tune"
            )
            file_id = str(_field(uploaded, "id", "file_id"))
            job = client.fine_tuning.create(
                training_file=file_id,
                model=spec.base_model,
                n_epochs=spec.num_epochs,
                learning_rate=spec.learning_rate,
                lora=True,
                suffix=f"stateset-{spec.output_dir.name}"[:40],
            )
        except Exception as exc:  # noqa: BLE001
            raise RemoteExecutionError.wrap(
                exc, "could not create Together fine-tune", provider=self.name
            ) from exc
        return JobHandle(self.name, str(_field(job, "id")))

    def _job(self, handle: JobHandle) -> Any:
        try:
            return self._client().fine_tuning.retrieve(id=handle.job_id)
        except Exception as exc:  # noqa: BLE001
            raise RemoteExecutionError.wrap(
                exc, "could not retrieve Together fine-tune", job_id=handle.job_id
            ) from exc

    def status(self, handle: JobHandle) -> JobStatus:
        state = str(_field(self._job(handle), "status", default="pending")).lower()
        return _STATES.get(state, JobStatus.PENDING)

    def logs(self, handle: JobHandle) -> Iterator[str]:
        job = self._job(handle)
        events = _field(job, "events", default=[]) or []
        for event in events:
            message = _field(event, "message", "description", default=event)
            yield str(message)

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        job = self._job(handle)
        if self.status(handle) is not JobStatus.SUCCEEDED:
            raise RemoteExecutionError("Together fine-tune has not succeeded")
        target = Path(dest or f"outputs/together-{handle.job_id}")
        target.mkdir(parents=True, exist_ok=True)
        archive = target / "together_model.tar.gz"
        download_error: str | None = None
        try:
            with self._client().fine_tuning.with_streaming_response.content(
                ft_id=handle.job_id
            ) as response:
                with archive.open("wb") as stream:
                    for chunk in response.iter_bytes():
                        stream.write(chunk)
        except Exception as exc:  # noqa: BLE001 - pointer remains usable
            archive.unlink(missing_ok=True)
            download_error = type(exc).__name__
        payload = {
            "provider": self.name,
            "job_id": handle.job_id,
            "output_model": _field(job, "output_name", "model_output_name", "model"),
            "checkpoint_path": _field(
                job, "model_output_path", "checkpoint_path", "training_file"
            ),
            "weights_downloaded": archive.exists(),
            "download_error": download_error,
        }
        (target / TOGETHER_POINTER).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    def cancel(self, handle: JobHandle) -> None:
        try:
            self._client().fine_tuning.cancel(id=handle.job_id)
        except Exception as exc:  # noqa: BLE001
            raise RemoteExecutionError.wrap(
                exc, "could not cancel Together fine-tune", job_id=handle.job_id
            ) from exc


__all__ = ["TogetherExecutor"]
