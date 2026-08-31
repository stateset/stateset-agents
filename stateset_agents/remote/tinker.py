"""Thinking Machines Tinker remote-autograd training integration."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec
from stateset_agents.remote.river_batches import build_sft_batch

TINKER_API_KEY_ENV = "TINKER_API_KEY"
TINKER_POINTER = "tinker_checkpoint.json"
INKLING_SMALL = "thinkingmachines/Inkling-Small"


@dataclass
class _TinkerJob:
    spec: RemoteJobSpec
    status: JobStatus = JobStatus.PENDING
    logs: list[str] = field(default_factory=list)
    sampler_uri: str | None = None
    state_uri: str | None = None
    error: str | None = None
    cancelled: bool = False


def _result(value: Any) -> Any:
    method = getattr(value, "result", None)
    return method() if callable(method) else value


def _path(value: Any) -> str:
    if isinstance(value, dict):
        return str(value.get("path") or value.get("uri") or value)
    return str(getattr(value, "path", value))


class TinkerExecutor(RemoteExecutor):
    """Drive Tinker's LoRA training client from a local control loop."""

    name = "tinker"
    supported_job_kinds = frozenset({"sft", "rl"})
    result_kind = "hosted_pointer"
    compute_model = "managed-remote-autograd"
    verification_status = "unit-tested-live-lifecycle-pending"

    def __init__(
        self,
        service_client: Any | None = None,
        *,
        tinker_module: Any | None = None,
        tokenizer: Any | None = None,
    ) -> None:
        self._service_client = service_client
        self._tinker = tinker_module
        self._tokenizer = tokenizer
        self._jobs: dict[str, _TinkerJob] = {}

    def _sdk(self) -> tuple[Any, Any]:
        module = self._tinker
        if module is None:
            try:
                import tinker as imported_tinker
            except ImportError as exc:
                raise RemoteExecutionError.wrap(
                    exc,
                    "the Tinker provider needs the Tinker SDK; install "
                    "'stateset-agents[tinker]'",
                    provider=self.name,
                ) from exc
            module = imported_tinker
            self._tinker = module
        if self._service_client is None:
            if not os.environ.get(TINKER_API_KEY_ENV):
                raise RemoteExecutionError(
                    f"set {TINKER_API_KEY_ENV} before using Tinker",
                    provider=self.name,
                )
            assert module is not None
            self._service_client = module.ServiceClient()
        return module, self._service_client

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        self.validate_spec(spec)
        self._sdk()
        job_id = uuid.uuid4().hex
        record = _TinkerJob(spec=spec)
        self._jobs[job_id] = record
        thread = threading.Thread(
            target=self._train, args=(job_id, record), daemon=True
        )
        thread.start()
        return JobHandle(self.name, job_id)

    def _train(self, job_id: str, record: _TinkerJob) -> None:
        try:
            module, service = self._sdk()
            record.status = JobStatus.RUNNING
            record.logs.append(f"creating Tinker LoRA model {record.spec.base_model}")
            training = service.create_lora_training_client(
                base_model=record.spec.base_model, rank=record.spec.lora_r
            )
            tokenizer = self._tokenizer or training.get_tokenizer()
            rows = [
                json.loads(line)
                for line in record.spec.dataset.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if record.spec.job_kind == "sft":
                datums = build_sft_batch(
                    rows, tokenizer, max_length=record.spec.max_length
                )
                loss_fn = "cross_entropy"
            else:
                datums = self._validate_rl_rows(rows)
                loss_fn = "importance_sampling"
            if not datums:
                raise ValueError("dataset contains no trainable assistant tokens")
            batch_size = record.spec.per_device_batch_size
            step = 0
            for epoch in range(record.spec.num_epochs):
                for offset in range(0, len(datums), batch_size):
                    if record.cancelled:
                        record.status = JobStatus.CANCELLED
                        return
                    batch = [
                        self._datum(module, item)
                        for item in datums[offset : offset + batch_size]
                    ]
                    _result(training.forward_backward(batch, loss_fn=loss_fn))
                    step += 1
                    if step % record.spec.gradient_accumulation_steps == 0:
                        adam_params = getattr(module, "AdamParams", None)
                        if adam_params is None:
                            adam_params = module.types.AdamParams
                        params = adam_params(learning_rate=record.spec.learning_rate)
                        _result(training.optim_step(params))
                    record.logs.append(f"epoch={epoch + 1} step={step}")
            if step % record.spec.gradient_accumulation_steps:
                adam_params = getattr(module, "AdamParams", None)
                if adam_params is None:
                    adam_params = module.types.AdamParams
                params = adam_params(learning_rate=record.spec.learning_rate)
                _result(training.optim_step(params))
            label = f"stateset-{job_id}"
            record.sampler_uri = _path(
                _result(training.save_weights_for_sampler(f"{label}-sampler"))
            )
            record.state_uri = _path(_result(training.save_state(f"{label}-state")))
            record.status = JobStatus.SUCCEEDED
            record.logs.append("Tinker checkpoint saved")
        except Exception as exc:  # noqa: BLE001 - background boundary
            record.error = str(exc)
            record.logs.append(f"failed: {exc}")
            record.status = JobStatus.FAILED

    @staticmethod
    def _datum(module: Any, item: dict[str, Any]) -> Any:
        import torch

        tensor = module.TensorData.from_torch
        loss_inputs = {
            key: tensor(torch.tensor(item[key]))
            for key in ("target_tokens", "weights", "logprobs", "advantages")
            if key in item
        }
        return module.Datum(
            model_input=module.ModelInput.from_ints(item["input_ids"]),
            loss_fn_inputs=loss_inputs,
        )

    @staticmethod
    def _validate_rl_rows(rows: list[Any]) -> list[dict[str, Any]]:
        required = ("input_ids", "target_tokens", "logprobs", "advantages")
        clean: list[dict[str, Any]] = []
        for index, row in enumerate(rows):
            if not isinstance(row, dict) or any(key not in row for key in required):
                raise ValueError(f"RL row {index} must contain {', '.join(required)}")
            lengths = {len(row[key]) for key in required}
            if len(lengths) != 1 or next(iter(lengths)) == 0:
                raise ValueError(f"RL row {index} arrays must be non-empty and aligned")
            clean.append(row)
        return clean

    def _record(self, handle: JobHandle) -> _TinkerJob:
        try:
            return self._jobs[handle.job_id]
        except KeyError:
            raise RemoteExecutionError(
                "Tinker handles are process-local; submit and poll in one process",
                job_id=handle.job_id,
            ) from None

    def status(self, handle: JobHandle) -> JobStatus:
        return self._record(handle).status

    def logs(self, handle: JobHandle) -> Iterator[str]:
        yield from list(self._record(handle).logs)

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        record = self._record(handle)
        if record.status is not JobStatus.SUCCEEDED:
            raise RemoteExecutionError("Tinker job has not succeeded")
        target = Path(dest or record.spec.output_dir)
        target.mkdir(parents=True, exist_ok=True)
        payload = {
            "provider": self.name,
            "base_model": record.spec.base_model,
            "sampler_uri": record.sampler_uri,
            "state_uri": record.state_uri,
            "created_at": int(time.time()),
        }
        (target / TINKER_POINTER).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        return target

    def cancel(self, handle: JobHandle) -> None:
        record = self._record(handle)
        record.cancelled = True
        if record.status is JobStatus.PENDING:
            record.status = JobStatus.CANCELLED


__all__ = ["INKLING_SMALL", "TinkerExecutor"]
