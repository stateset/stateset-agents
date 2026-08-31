"""Prime Intellect Lab bridge for verifiers/OpenEnv reinforcement learning."""

from __future__ import annotations

import json
import re
import subprocess
import threading
import uuid
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from stateset_agents.remote.executor import RemoteExecutionError, RemoteExecutor
from stateset_agents.remote.job import JobHandle, JobStatus, RemoteJobSpec

PRIME_POINTER = "prime_training_run.json"


def _toml_string(value: str) -> str:
    return json.dumps(value)


def prime_lab_config(spec: RemoteJobSpec) -> str:
    """Render a Prime Lab TOML config from a StateSet RL job specification.

    RL-specific values live in ``spec.harvest`` for backwards compatibility:
    ``environment`` is a verifiers/OpenEnv environment id, while ``harness``
    and ``runtime`` select an optional custom harness.
    """
    if spec.job_kind != "rl":
        raise ValueError("Prime Lab accepts job_kind='rl'")
    knobs = spec.harvest or {}
    environment = str(knobs.get("environment") or "stateset/customer-service")
    lines = [
        f"model = {_toml_string(spec.base_model)}",
        f"max_steps = {int(knobs.get('max_steps', spec.num_epochs))}",
        f"batch_size = {spec.per_device_batch_size}",
        f"rollouts_per_example = {int(knobs.get('rollouts_per_example', 8))}",
        f"learning_rate = {spec.learning_rate}",
        f"lora_alpha = {spec.lora_alpha}",
        "",
        "[sampling]",
        f"max_tokens = {int(knobs.get('max_tokens', spec.eval_max_new_tokens))}",
        f"temperature = {float(knobs.get('temperature', 0.8))}",
        "",
        "[[env]]",
        f"id = {_toml_string(environment)}",
    ]
    if knobs.get("harness"):
        lines += [
            f"harness = {_toml_string(str(knobs['harness']))}",
            f"runtime = {_toml_string(str(knobs.get('runtime', 'uv')))}",
        ]
    return "\n".join(lines) + "\n"


@dataclass
class _PrimeJob:
    spec: RemoteJobSpec
    config_path: Path
    status: JobStatus = JobStatus.PENDING
    logs: list[str] = field(default_factory=list)
    run_id: str | None = None
    process: subprocess.Popen[str] | None = None


class PrimeLabExecutor(RemoteExecutor):
    """Launch a Prime hosted RL run using the documented ``prime`` CLI."""

    name = "prime"
    supported_job_kinds = frozenset({"rl"})
    result_kind = "hosted_pointer"
    compute_model = "managed-verifiers-rl"
    verification_status = "unit-tested-live-lifecycle-pending"

    def __init__(self, *, command: Sequence[str] = ("prime",)) -> None:
        self.command = tuple(command)
        self._jobs: dict[str, _PrimeJob] = {}

    def submit(self, spec: RemoteJobSpec) -> JobHandle:
        self.validate_spec(spec)
        job_id = uuid.uuid4().hex
        config_dir = spec.output_dir / ".prime"
        config_dir.mkdir(parents=True, exist_ok=True)
        config_path = config_dir / f"{job_id}.toml"
        config_path.write_text(prime_lab_config(spec), encoding="utf-8")
        record = _PrimeJob(spec=spec, config_path=config_path)
        self._jobs[job_id] = record
        threading.Thread(target=self._run, args=(record,), daemon=True).start()
        return JobHandle(self.name, job_id)

    def _run(self, record: _PrimeJob) -> None:
        try:
            record.status = JobStatus.RUNNING
            record.process = subprocess.Popen(
                [*self.command, "train", "run", str(record.config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            assert record.process.stdout is not None
            for raw in record.process.stdout:
                line = raw.rstrip()
                record.logs.append(line)
                match = re.search(
                    r"(?:run[-_ ]?id|runs?/)([A-Za-z0-9._-]+)", line, re.I
                )
                if match:
                    record.run_id = match.group(1)
            code = record.process.wait()
            record.status = JobStatus.SUCCEEDED if code == 0 else JobStatus.FAILED
        except FileNotFoundError:
            record.logs.append(
                "prime CLI not found; install it with 'uv tool install prime'"
            )
            record.status = JobStatus.FAILED
        except Exception as exc:  # noqa: BLE001
            record.logs.append(f"failed: {exc}")
            record.status = JobStatus.FAILED

    def _record(self, handle: JobHandle) -> _PrimeJob:
        try:
            return self._jobs[handle.job_id]
        except KeyError:
            raise RemoteExecutionError(
                "Prime CLI handles are process-local; use the recorded Prime run id "
                "in the Prime dashboard after this process exits",
                job_id=handle.job_id,
            ) from None

    def status(self, handle: JobHandle) -> JobStatus:
        return self._record(handle).status

    def logs(self, handle: JobHandle) -> Iterator[str]:
        yield from list(self._record(handle).logs)

    def fetch(self, handle: JobHandle, dest: Path | None = None) -> Path:
        record = self._record(handle)
        if record.status is not JobStatus.SUCCEEDED:
            raise RemoteExecutionError("Prime Lab run has not succeeded")
        target = Path(dest or record.spec.output_dir)
        target.mkdir(parents=True, exist_ok=True)
        (target / PRIME_POINTER).write_text(
            json.dumps(
                {
                    "provider": self.name,
                    "local_job_id": handle.job_id,
                    "prime_run_id": record.run_id,
                    "config": str(record.config_path),
                    "environment": (record.spec.harvest or {}).get("environment"),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return target

    def cancel(self, handle: JobHandle) -> None:
        record = self._record(handle)
        if record.process and record.process.poll() is None:
            record.process.terminate()
        record.status = JobStatus.CANCELLED


__all__ = ["PrimeLabExecutor", "prime_lab_config"]
