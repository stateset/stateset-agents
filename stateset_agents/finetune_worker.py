"""Detached fine-tuning supervisor and training process for the terminal studio."""

from __future__ import annotations

import argparse
import asyncio
import errno
import json
import logging
import os
import secrets
import stat
import subprocess
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from stateset_agents.finetune_workspace import (
    FAILURE_GUIDANCE,
    FinetuneConfig,
    FinetuneFailure,
    failure_info,
    process_token,
    queue_executable,
    read_json,
    read_scenarios,
    stop_requested,
    verified_run_settings,
    write_json,
)


def _settings(run: Path) -> tuple[dict[str, Any], FinetuneConfig]:
    return verified_run_settings(run)


_STATE_LOCK_TIMEOUT = 1.0


def _state_lock_operation(descriptor: int, *, release: bool = False) -> None:
    if sys.platform == "win32":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK if release else msvcrt.LK_NBLCK, 1)
    else:
        import fcntl

        fcntl.flock(
            descriptor, fcntl.LOCK_UN if release else fcntl.LOCK_EX | fcntl.LOCK_NB
        )


@contextmanager
def _state_lock(run: Path) -> Iterator[None]:
    """Serialize bounded state transitions through a stable private descriptor."""
    path = run / ".state.lock"
    if path.is_symlink():
        raise ValueError("Fine-tuning state lock is unavailable")
    descriptor = os.open(
        path,
        os.O_RDWR
        | os.O_CREAT
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0),
        0o600,
    )
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or (
            os.name != "nt" and stat.S_IMODE(info.st_mode) & 0o077
        ):
            raise ValueError("Fine-tuning state lock is unavailable")
        if os.name == "nt" and info.st_size == 0:
            os.write(descriptor, b"\0")
        deadline = time.monotonic() + _STATE_LOCK_TIMEOUT
        while True:
            try:
                _state_lock_operation(descriptor)
                break
            except OSError as error:
                if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK}:
                    raise
                if time.monotonic() >= deadline:
                    raise ValueError("Fine-tuning state is busy") from None
                time.sleep(0.01)
        try:
            yield
        finally:
            _state_lock_operation(descriptor, release=True)
    finally:
        os.close(descriptor)


def _state(run: Path, **changes: Any) -> bool:
    """Merge progress without promoting an already failed or cancelled run."""
    with _state_lock(run):
        try:
            state = read_json(run / "state.json")
        except (OSError, ValueError):
            state = {"previous_state_unreadable": True}
        status = changes.get("status")
        if (
            state.get("status") in {"failed", "cancelled"}
            and status is not None
            and status not in {"failed", "cancelled"}
        ):
            return False
        write_json(run / "state.json", {**state, **changes, "updated_at": time.time()})
        return True


def _record_failure(
    run: Path,
    error: Exception,
    *,
    stage: str,
    preserve: bool = False,
    exit_code: int | None = None,
) -> None:
    """Retain a controlled diagnosis, preserving a child's original failure."""
    previous = failure_info(run, {"status": "failed"}) if preserve else None
    if previous and previous["recorded"]:
        code, stage = previous["code"], previous["stage"]
    elif isinstance(error, FinetuneFailure):
        code = error.code
    elif isinstance(error, OSError) and error.errno in {errno.ENOSPC, errno.EDQUOT}:
        code = "disk_full"
    elif isinstance(error, PermissionError):
        code = "permissions"
    elif isinstance(error, (TimeoutError, subprocess.TimeoutExpired)):
        code = "timeout"
    elif isinstance(error, ImportError):
        code = "dependencies"
    elif isinstance(error, MemoryError) or (
        type(error).__name__ == "OutOfMemoryError"
        and (
            type(error).__module__ == "torch"
            or type(error).__module__.startswith("torch.")
        )
    ):
        code = "out_of_memory"
    else:
        code = "review_invalid" if stage == "review" else "unknown"
    try:
        if not (previous and previous["recorded"]):
            write_json(
                run / "failure.json",
                {
                    "schema_version": 1,
                    "code": code,
                    "stage": stage,
                    "recorded_at": time.time(),
                },
            )
        changes: dict[str, Any] = {
            "status": "cancelled" if code == "cancelled" else "failed",
            "finished_at": time.time(),
            "error": FAILURE_GUIDANCE[code][0] + "; open run details for next steps.",
        }
        if exit_code is not None:
            changes["exit_code"] = exit_code
        _state(run, **changes)
    except (OSError, ValueError):
        # Full or inaccessible storage must not replace the original failure
        # with a new exception or a successful exit. Logs remain the fallback.
        logging.error("Could not persist the fine-tuning failure diagnosis")


SUPERVISOR_TOKEN_ENV = "STATESET_FINETUNE_SUPERVISOR_TOKEN"


def _present(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _claim_record(run: Path, filename: str, value: dict[str, Any]) -> bool:
    """Claim private ownership without replacing existing or uncertain records."""
    if run.is_symlink():
        logging.error("Fine-tuning ownership requires a regular run directory")
        return False
    try:
        descriptor = os.open(
            run / filename, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
        )
    except FileExistsError:
        logging.error("Fine-tuning ownership is already claimed; no work started")
        return False
    except OSError:
        logging.error("Could not claim fine-tuning ownership; no work started")
        return False
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as receipt:
            json.dump(value, receipt)
            receipt.write("\n")
            receipt.flush()
            os.fsync(receipt.fileno())
    except OSError:
        # Keep even an incomplete receipt after a write failure: its owner is
        # uncertain and retrying this run must never silently duplicate work.
        logging.error("Could not retain fine-tuning ownership; no work started")
        return False
    return True


def _claim_owner(run: Path, filename: str) -> bool:
    return _claim_record(
        run,
        filename,
        {"pid": os.getpid(), "process_token": process_token(os.getpid())},
    )


def _claim_admission(run: Path, mode: str, token: str | None = None) -> bool:
    return _claim_record(
        run,
        "admission.json",
        {
            "schema_version": 1,
            "mode": mode,
            "pid": os.getpid(),
            "process_token": process_token(os.getpid()),
            "supervisor_token": token,
        },
    )


def _supervised_admission(run: Path, token: str | None) -> bool:
    """Verify bounded local child authority without querying or signaling owners."""
    if (
        not isinstance(token, str)
        or len(token) != 64
        or any(char not in "0123456789abcdef" for char in token)
    ):
        return False
    try:
        admission = read_json(run / "admission.json", max_bytes=4096)
        owner = read_json(run / "supervisor.json", max_bytes=4096)
        return (
            set(admission)
            == {
                "schema_version",
                "mode",
                "pid",
                "process_token",
                "supervisor_token",
            }
            and type(admission.get("schema_version")) is int
            and admission["schema_version"] == 1
            and admission.get("mode") == "supervised"
            and type(admission.get("pid")) is int
            and admission["pid"] > 0
            and (
                admission.get("process_token") is None
                or isinstance(admission["process_token"], str)
                and 0 < len(admission["process_token"]) <= 128
            )
            and isinstance(admission.get("supervisor_token"), str)
            and len(admission["supervisor_token"]) == 64
            and admission["supervisor_token"].isascii()
            and secrets.compare_digest(admission["supervisor_token"], token)
            and owner
            == {
                "pid": admission["pid"],
                "process_token": admission["process_token"],
            }
        )
    except (OSError, ValueError, TypeError, RecursionError):
        return False


def _supervision_ready(run: Path) -> bool:
    """Check existing phase metadata without claiming a cross-process transition."""
    try:
        return (
            not run.is_symlink()
            and read_json(run / "state.json").get("status") == "prepared"
            and not (run / "execution.json").exists()
            and not (run / "execution.json").is_symlink()
        )
    except (OSError, ValueError, TypeError, RecursionError):
        return False


def _execution_ready(run: Path, *, supervised: bool = False) -> bool:
    """Admit only first execution phases, preserving terminal or uncertain runs."""
    try:
        allowed = {"starting", "queued"} if supervised else {"prepared"}
        return (
            not run.is_symlink()
            and read_json(run / "state.json").get("status") in allowed
        )
    except (OSError, ValueError, TypeError, RecursionError):
        return False


def supervise(run: Path) -> int:
    """Hold lifecycle ownership while a shared-queue training command runs."""
    os.umask(0o077)
    if not _supervision_ready(run) or _present(run / "supervisor.json"):
        logging.error("Fine-tuning supervision is unavailable; no work started")
        return 1
    supervisor_token = secrets.token_hex(32)
    if not _claim_admission(run, "supervised", supervisor_token):
        return 1
    if not _claim_owner(run, "supervisor.json"):
        return 1
    stage = "review"
    try:
        _, config = _settings(run)
        _check_stop(run)
        stage = "supervisor"
        return _supervise(run, config, supervisor_token=supervisor_token)
    except Exception as error:
        _record_failure(run, error, stage=stage, preserve=True)
        logging.exception("Fine-tuning supervisor could not start")
        return 1


def _supervise(
    run: Path, config: FinetuneConfig, *, supervisor_token: str | None = None
) -> int:
    """Run the already claimed supervisor; direct unowned entry is unsupported."""
    if (
        supervisor_token is None
        or not _supervised_admission(run, supervisor_token)
        or read_json(run / "supervisor.json")
        != {"pid": os.getpid(), "process_token": process_token(os.getpid())}
    ):
        raise ValueError("Fine-tuning supervision requires the current owner")
    if not _supervision_ready(run):
        logging.error(
            "Fine-tuning supervision changed before admission; no work started"
        )
        return 1
    # Provider waits must not hold local CPU/RAM reservations.
    queue = queue_executable() if config.provider == "local" else None
    command = [
        sys.executable,
        "-u",
        "-m",
        "stateset_agents.finetune_worker",
        "--supervised",
        "--execute",
        str(run),
    ]
    if queue:
        command = [
            queue,
            "run",
            "--lane",
            "heavy",
            "--jobs",
            str(config.cpu_slots),
            "--memory-gb",
            str(config.memory_gb),
            "--scratch-gb",
            str(config.scratch_gb),
            "--timeout",
            str(config.timeout_seconds),
            "--resource-key",
            f"stateset-finetune:{run}",
            "--",
            *command,
        ]
    write_json(run / "command.json", {"argv": command, "queue": queue})
    if not _state(
        run, status="queued" if queue else "starting", started_at=time.time()
    ):
        logging.error("Fine-tuning outcome already recorded; no work started")
        return 1
    environment = dict(
        os.environ,
        OMP_NUM_THREADS=str(config.cpu_slots),
        MKL_NUM_THREADS=str(config.cpu_slots),
        TOKENIZERS_PARALLELISM="false",
        PYTHONUNBUFFERED="1",
    )
    environment[SUPERVISOR_TOKEN_ENV] = supervisor_token
    try:
        with (run / "run.log").open("ab") as log:
            process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                start_new_session=True,
            )
            # Buildr owns the timeout and its child cleanup. Without Buildr,
            # this supervisor owns the process group and bounds its lifetime.
            try:
                code = process.wait(
                    timeout=(
                        None
                        if queue or config.provider != "local"
                        else config.timeout_seconds
                    )
                )
            except subprocess.TimeoutExpired:
                import signal

                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                raise FinetuneFailure("timeout") from None
        completed = (run / "completion.json").is_file()
        success = code == 0 and completed
        if success:
            success = _state(
                run,
                status="succeeded",
                exit_code=code,
                finished_at=time.time(),
                error=None,
            )
        else:
            _record_failure(
                run,
                FinetuneFailure("missing_receipt" if code == 0 else "worker_exit"),
                stage="supervisor",
                preserve=True,
                exit_code=code,
            )
        return 0 if success else (code or 1)
    except Exception as error:
        _record_failure(run, error, stage="supervisor", preserve=True)
        logging.exception("Fine-tuning supervisor failed")
        return 1


def _check_stop(run: Path) -> None:
    if stop_requested(run):
        raise FinetuneFailure("cancelled")


def _metrics(run: Path, step: int, metrics: dict[str, Any]) -> None:
    # Do not serialize callback objects, tensors, prompts or credentials.
    import math

    values = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    }
    event = {"step": step, "metrics": values, "updated_at": time.time()}
    descriptor = os.open(
        run / "metrics.jsonl", os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
    )
    with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
        stream.write(json.dumps(event, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    write_json(run / "metrics.json", event)


def _train_sft(run: Path, config: FinetuneConfig) -> Path:
    from stateset_agents.training.sft import load_sft_dataset, run_sft

    plan = read_json(run / "plan.json")
    rows, dataset_identity = load_sft_dataset(
        run / "dataset", text_only=True, expected_sha256=plan["dataset_sha256"]
    )
    from transformers import TrainerCallback

    class Progress(TrainerCallback):
        def on_save(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            from stateset_agents.finetune_recovery import publish_checkpoint

            path = run / "artifacts" / f"checkpoint-{state.global_step}"
            publish_checkpoint(run, path, step=state.global_step)
            _state(run, recovery_checkpoint=str(path))
            return control

        def on_step_end(
            self, args: Any, state: Any, control: Any, **kwargs: Any
        ) -> Any:
            if stop_requested(run):
                control.should_training_stop = True
                control.should_save = True
            return control

        def on_log(
            self, args: Any, state: Any, control: Any, logs: Any = None, **kwargs: Any
        ) -> None:
            _metrics(run, state.global_step, logs or {})

    return run_sft(
        rows=rows,
        base_model=config.model,
        model_revision=config.revision or None,
        output_dir=run / "artifacts",
        num_epochs=config.epochs,
        lora_r=config.lora_rank,
        lora_alpha=2 * config.lora_rank,
        learning_rate=config.learning_rate,
        max_length=config.context_length,
        seed=config.seed,
        per_device_batch_size=1,
        gradient_accumulation_steps=4,
        dataset_path=run / "dataset",
        dataset_identity=dataset_identity,
        callbacks=[Progress()],
        resume_from_checkpoint=(
            run / "resume-checkpoint"
            if "resume" in read_json(run / "plan.json")
            else None
        ),
    )


async def _train_rl(run: Path, config: FinetuneConfig) -> tuple[Path, int]:
    from transformers import set_seed

    from stateset_agents.core.agent import MultiTurnAgent
    from stateset_agents.core.agent_config import AgentConfig
    from stateset_agents.core.conversation_environment import ConversationEnvironment
    from stateset_agents.rewards.support_conversation_reward import (
        SupportConversationRewardV3,
    )
    from stateset_agents.training.config import TrainingConfig
    from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer

    scenarios = read_scenarios((run / "dataset").read_bytes())
    # Policy facts are supplied to the agent, not just to its reward function.
    scenarios = [
        {
            **row,
            "context": row["context"]
            + "\nPolicy facts:\n"
            + "\n".join(row["policy_facts"]),
        }
        for row in scenarios
    ]
    # Adapter initialization happens before trainer.initialize(), so seed here
    # rather than relying only on the trainer's later sampling seed.
    set_seed(config.seed)
    agent = MultiTurnAgent(
        AgentConfig(
            model_name=config.model,
            model_revision=config.revision or None,
            max_new_tokens=config.max_new_tokens,
            temperature=1.0,
            top_p=1.0,
            repetition_penalty=1.0,
            use_peft=True,
            trust_remote_code=False,
            peft_config={
                "r": config.lora_rank,
                "lora_alpha": 2 * config.lora_rank,
                "lora_dropout": 0.0,
                "target_modules": "all-linear",
            },
        )
    )
    await agent.initialize()

    class Progress:
        fail_on_error = True

        def on_checkpoint_saved(
            self, path: str, step: int, is_best: bool = False
        ) -> None:
            if is_best:
                return
            from stateset_agents.finetune_recovery import publish_checkpoint

            publish_checkpoint(run, Path(path), step=step)
            _state(run, recovery_checkpoint=path)

        def on_step_end(self, step: int, metrics: dict[str, Any]) -> None:
            _metrics(run, step, metrics)

        async def on_optimizer_step(self, step: int, metrics: dict[str, Any]) -> None:
            _state(run, committed_updates=step)
            if stop_requested(run):
                name = f"stopped-step-{step}"
                await trainer.save_checkpoint(checkpoint_name=name)
                _state(run, recovery_checkpoint=str(run / "artifacts" / name))
                raise FinetuneFailure("cancelled")

    training = TrainingConfig(
        model_name=config.model,
        model_revision=config.revision or None,
        objective=config.method,
        num_episodes=config.steps,
        num_generations=config.generations,
        learning_rate=config.learning_rate,
        beta=config.kl_coefficient,
        use_reference_model=True,
        max_prompt_length=config.context_length,
        max_completion_length=config.max_new_tokens,
        max_steps_per_episode=config.max_turns,
        output_dir=str(run / "artifacts"),
        eval_steps=config.steps + 1,
        save_steps=config.checkpoint_every or config.steps + 1,
        logging_steps=1,
        gradient_accumulation_steps=1,
        num_gradient_updates=1,
        bf16=True,
        fp16=False,
        report_to="none",
        seed=config.seed,
        resume_from_checkpoint=(
            str(run / "resume-checkpoint")
            if "resume" in read_json(run / "plan.json")
            else None
        ),
    )
    trainer = MultiTurnGRPOTrainer(
        agent,
        ConversationEnvironment(scenarios, max_turns=config.max_turns),
        reward_fn=SupportConversationRewardV3(),
        config=training,
        callbacks=[Progress()],
    )
    await trainer.initialize()
    await trainer.train()
    if trainer.global_step != config.steps:
        raise FinetuneFailure("incomplete_updates")
    return run / "artifacts" / f"checkpoint-{trainer.global_step}", trainer.global_step


def _train_runpod(run: Path, config: FinetuneConfig) -> Path:
    """Compatibility entry for the original studio RunPod path."""
    return _train_remote(run, config)


def _train_remote(run: Path, config: FinetuneConfig) -> Path:
    from stateset_agents.finetune_provider_submission import SubmissionJournal
    from stateset_agents.remote.job import RemoteJobSpec
    from stateset_agents.remote.registry import get_executor

    plan, reviewed = verified_run_settings(run)
    if config != reviewed:
        raise ValueError("Remote configuration differs from the reviewed plan")
    resources = plan["resources"]
    executor = get_executor(config.provider)
    if config.provider == "runpod":
        from stateset_agents.finetune_remote_code import verify_remote_code

        wheel = verify_remote_code(
            run, plan["remote_code"], package_version=plan["framework_version"]
        )
        binder = getattr(executor, "bind_reviewed_wheel", None)
        if not callable(binder):
            raise ValueError("RunPod executor cannot bind the reviewed package")
        binder(
            wheel,
            sha256=plan["remote_code"].get("wheel_sha256"),
            size_bytes=plan["remote_code"].get("wheel_bytes"),
        )
    journal = SubmissionJournal(
        run,
        config.provider,
        durable_supported=getattr(executor, "durable_submission_receipts", False),
    )
    if getattr(executor, "durable_submission_receipts", False):
        executor.set_submission_observer(journal.publish)
    spec = RemoteJobSpec(
        dataset=run / "dataset",
        base_model=config.model,
        output_dir=run / "artifacts",
        num_epochs=config.epochs,
        lora_r=config.lora_rank,
        lora_alpha=2 * config.lora_rank,
        learning_rate=config.learning_rate,
        max_length=config.context_length,
        seed=config.seed,
        package_version=plan["framework_version"],
        gpu=resources.get("gpu"),
        gpu_count=resources.get("gpu_count", 1),
        container_disk_gb=resources.get("container_disk_gb"),
        max_cost_usd=config.max_cost_usd if config.provider == "runpod" else None,
        timeout_s=config.timeout_seconds,
    )
    handle = executor.submit(spec)
    journal.record_handle(handle.provider, handle.job_id)
    result = executor.wait(handle)
    journal.finish(result.status.value)
    for line in result.logs:
        print(line, flush=True)
    write_json(
        run / "provider-result.json",
        {"cost_usd": result.cost_usd, "duration_s": result.duration_s},
    )
    if not result.succeeded or result.output_dir is None:
        raise FinetuneFailure("provider_failed")
    return Path(result.output_dir)


def _hosted_checkpoint(output: Path, config: FinetuneConfig) -> str:
    """Verify a River result's local receipt; do not claim to download its weights."""
    path = output / "river_checkpoint.json"
    if path.is_symlink() or not path.is_file():
        raise FinetuneFailure("missing_weights")
    with path.open("rb") as stream:
        raw = stream.read(64 * 1024 + 1)
    if len(raw) > 64 * 1024:
        raise FinetuneFailure("missing_weights")
    try:
        record = json.loads(raw)
        uri = record["checkpoint"]
        valid = (
            record["provider"] == "river"
            and record["base_model"] == config.model
            and type(record["lora"]["rank"]) is int
            and record["lora"]["rank"] == config.lora_rank
            and type(record["steps"]) is int
            and record["steps"] > 0
            and type(record["num_epochs"]) is int
            and record["num_epochs"] == config.epochs
            and record["learning_rate"] == config.learning_rate
            and isinstance(uri, str)
            and uri.startswith("river://")
            and len(uri) > len("river://")
            and not any(c.isspace() or not c.isprintable() for c in uri)
        )
    except (ValueError, TypeError, KeyError, RecursionError):
        valid = False
    if not valid:
        raise FinetuneFailure("missing_weights")
    return str(uri)


def execute(run: Path, *, supervisor_token: str | None = None) -> int:
    """Run actual training, failing rather than substituting a CPU dry run."""
    os.umask(0o077)
    # Explicit callers pass child authority; ambient values never grant it or
    # reach model/provider setup through an inherited environment.
    os.environ.pop(SUPERVISOR_TOKEN_ENV, None)
    supervised = supervisor_token is not None
    if not _execution_ready(run, supervised=supervised) or _present(
        run / "execution.json"
    ):
        logging.error("Fine-tuning execution is unavailable; no work started")
        return 1
    if supervised:
        if not _supervised_admission(run, supervisor_token):
            logging.error("Fine-tuning child admission is unavailable; no work started")
            return 1
    elif _present(run / "supervisor.json") or not _claim_admission(run, "direct"):
        return 1
    if not _claim_owner(run, "execution.json"):
        return 1
    if not _execution_ready(run, supervised=supervised):
        logging.error("Fine-tuning execution changed before admission; no work started")
        return 1
    stage = "review"
    try:
        _, config = _settings(run)
        _check_stop(run)
        stage = "runtime"
        if not _state(run, status="running", training_started_at=time.time()):
            logging.error("Fine-tuning outcome already recorded; no work started")
            return 1
        updates = None
        if config.provider != "local":
            stage = "training"
            output = (
                _train_runpod(run, config)
                if config.provider == "runpod"
                else _train_remote(run, config)
            )
        else:
            from stateset_agents.training.sft import gpu_available

            if not gpu_available():
                raise FinetuneFailure("cuda_unavailable")
            stage = "training"
            if config.method == "sft":
                output = _train_sft(run, config)
            else:
                output, updates = asyncio.run(_train_rl(run, config))
        # Exit zero from a plan-only provider is not a successful training run.
        _check_stop(run)
        stage = "artifacts"
        weights = [
            p
            for pattern in ("*.safetensors", "pytorch_model*.bin", "adapter_model.bin")
            for p in output.glob(pattern)
            if p.is_file() and p.stat().st_size > 0
        ]
        hosted = (
            _hosted_checkpoint(output, config) if config.provider == "river" else None
        )
        if not weights and hosted is None:
            raise FinetuneFailure("missing_weights")
        write_json(
            run / "completion.json",
            {
                "output_dir": str(output),
                "committed_updates": updates,
                "weights": [str(p.relative_to(output)) for p in weights],
                "result_kind": (
                    "hosted_pointer" if hosted is not None else "local_weights"
                ),
                "checkpoint": hosted,
                "quality_gain_verified": False,
            },
        )
        # The owned worker knows its training/artifact pipeline completed even
        # when its supervisor has disappeared. Only a supervisor can record an
        # observed subprocess exit code; this is the worker's committed outcome.
        if not _state(run, status="succeeded", finished_at=time.time(), error=None):
            logging.error("Fine-tuning outcome already recorded; success not committed")
            return 1
        return 0
    except Exception as error:
        _record_failure(run, error, stage=stage)
        logging.exception("Fine-tuning failed")
        return 1


def main() -> int:
    """Run a supervisor or its training child; not an interactive entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--supervise", type=Path)
    group.add_argument("--execute", type=Path)
    parser.add_argument("--supervised", action="store_true")
    args = parser.parse_args()
    # Consume local coordination before validation, model setup or providers.
    child_token = os.environ.pop(SUPERVISOR_TOKEN_ENV, None)
    if args.supervised and (args.execute is None or child_token is None):
        logging.error("Fine-tuning child admission is unavailable; no work started")
        return 1
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    run = (args.supervise or args.execute).resolve()
    try:
        if args.supervise:
            return supervise(run)
        return execute(run, supervisor_token=child_token if args.supervised else None)
    except Exception:
        # Public entrypoints retain their own owned failures. An unexpected
        # admission/dispatch error has no authority to rewrite a shared run.
        logging.error("Worker could not start; no additional run state was written")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
