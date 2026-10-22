"""Reviewed local base/adapter evaluations, independent of training rewards.

Planning and inspection import no model libraries. Execution is an explicit,
queued operation; raw cases, transcripts and judge recordings stay private.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.support_judge import (
    CHECKS,
    RUBRIC_PATH,
    rubric_revision,
)
from stateset_agents.finetune_quality import _artifact
from stateset_agents.finetune_recovery import (
    CHECKPOINT_NAME_PATTERN,
    _inventory,
    inspect_evaluation_checkpoint,
    list_evaluation_checkpoints,
)
from stateset_agents.finetune_workspace import (
    detached_owner_lost,
    launch_unclaimed,
    queue_executable,
    read_json,
    read_run_plan,
    read_training_data,
    reap_children,
    retain_child,
    transition_state,
    write_json,
)

MAX_CASE_BYTES = 2 * 1024 * 1024
MAX_TRANSCRIPT_BYTES = 16 * 1024 * 1024


class EvaluationProtocolError(ValueError):
    """A controlled, actionable protocol error containing no source values."""

    def __init__(self, kind: str, *, split: str = "test") -> None:
        if split not in {"test", "validation"}:
            raise ValueError("Unsupported expected evaluation partition")
        messages = {
            "checkpoint_partition": "Checkpoint selection requires validation cases; reserve fresh test cases for final assessment.",
            "case_partition": f"Choose case JSON with schema_version 1 and split {split}.",
            "provenance": f"Declare reviewed source, reviewer and {split}_is_fresh for these held-out cases.",
        }
        if kind not in messages:
            raise ValueError("Unknown evaluation protocol error")
        self.code = "evaluation_" + kind
        self.diagnostic = {
            "field": "split" if kind == "checkpoint_partition" else "cases",
            "expected_partition": (
                "validation" if kind == "checkpoint_partition" else split
            ),
        }
        super().__init__(messages[kind])


REQUIRED_CHECKS = (
    *CHECKS["resolution"],
    *CHECKS["accuracy"],
    *CHECKS["policy_compliance"],
    "conciseness_no_repetition",
)
EVALUATION_FIELDS = (
    ("cases", "Held-out cases JSON"),
    ("judge_model", "Judge model"),
    ("judge_revision", "Judge commit"),
    ("max_new_tokens", "Reply tokens / turn"),
    ("judge_max_new_tokens", "Judge reply tokens"),
    ("context_length", "Context tokens"),
    ("seed", "Evaluation seed"),
    ("timeout_seconds", "Timeout seconds"),
    ("cpu_slots", "Queue CPU slots"),
    ("memory_gb", "Queue RAM GiB"),
    ("scratch_gb", "Queue scratch GiB"),
    ("checkpoint", "Published checkpoint (blank: final)"),
    ("split", "Case partition (test / validation)"),
    ("nomination_id", "Nomination for fresh test (optional)"),
)


_HUB_ID_PATTERN = (
    r"^(?!.*(?:\.\.|--))(?!.*[.-](?:/|$))"
    r"[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$"
)


def _hub_model(value: Any) -> bool:
    """Accept a repository identifier, never a relative/local model path."""
    return isinstance(value, str) and bool(re.fullmatch(_HUB_ID_PATTERN, value))


@dataclass(frozen=True)
class EvaluationConfig:
    """Bounded diagnostic protocol; no endpoint credentials or paid provider."""

    cases: str = ""
    judge_model: str = ""
    judge_revision: str = ""
    max_new_tokens: int = 192
    judge_max_new_tokens: int = 1024
    context_length: int = 8192
    seed: int = 42
    timeout_seconds: int = 3600
    cpu_slots: int = 2
    memory_gb: float = 4.0
    scratch_gb: float = 5.0
    checkpoint: str = ""
    split: str = "test"
    nomination_id: str = ""

    def validate(self) -> None:
        """Reject unknown runtime behavior and unbounded resource settings."""
        if not isinstance(self.cases, str) or not self.cases.strip():
            raise ValueError("Choose a held-out cases JSON file")
        if not isinstance(self.split, str) or self.split not in {"test", "validation"}:
            raise ValueError("Choose test or validation for the case partition")
        if not isinstance(self.checkpoint, str) or (
            self.checkpoint
            and not re.fullmatch(CHECKPOINT_NAME_PATTERN, self.checkpoint)
        ):
            raise ValueError(
                "Choose a published checkpoint name or blank for the final artifact"
            )
        if not isinstance(self.nomination_id, str) or (
            self.nomination_id
            and not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{8}", self.nomination_id)
        ):
            raise ValueError("Choose a retained nomination ID or leave it blank")
        if self.nomination_id and (not self.checkpoint or self.split != "test"):
            raise ValueError(
                "A nomination requires its named checkpoint and test cases"
            )
        if self.checkpoint and self.split != "validation" and not self.nomination_id:
            raise EvaluationProtocolError("checkpoint_partition")
        if not _hub_model(self.judge_model):
            raise ValueError("Judge model must be a Hugging Face repository ID")
        if (
            not isinstance(self.judge_revision, str)
            or not re.fullmatch(r"[0-9a-f]{40}", self.judge_revision)
            or self.judge_revision == "0" * 40
        ):
            raise ValueError("Pin the judge to a nonzero full 40-character commit")
        for name, low, high in (
            ("max_new_tokens", 1, 2048),
            ("judge_max_new_tokens", 128, 4096),
            ("context_length", 512, 65536),
            ("seed", 0, 2**32 - 1),
            ("timeout_seconds", 30, 86400),
            ("cpu_slots", 1, 32),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")
        for name, high in (("memory_gb", 1024), ("scratch_gb", 4096)):
            value = getattr(self, name)
            if (
                type(value) not in {int, float}
                or not 0 < value <= high
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite and positive, at most {high}")
        if max(self.max_new_tokens, self.judge_max_new_tokens) >= self.context_length:
            raise ValueError("Context tokens must exceed both generation token limits")


@dataclass(frozen=True)
class EvaluationReview:
    """Captured cases plus a content-free, hash-bound launch review."""

    run: Path
    plan: dict[str, Any]
    cases: dict[str, Any]

    @property
    def sha256(self) -> str:
        """Canonical checksum required for preparation and launch."""
        return content_hash(self.plan)


def _binding(
    run: Path, *, checkpoint: str = "", nomination_id: str = ""
) -> dict[str, Any]:
    plan, digest = read_run_plan(run)
    state = read_json(run / "state.json")
    config = plan["config"]
    training_seed = config.get("seed")
    if training_seed is not None and (
        type(training_seed) is not int or not 0 <= training_seed <= 2**32 - 1
    ):
        raise ValueError("The recorded training seed is invalid")
    revision = config.get("revision")
    if config.get("provider") != "local" or (
        not checkpoint and state.get("status") != "succeeded"
    ):
        raise ValueError(
            "Evaluation requires a successful local final artifact or a terminal published checkpoint"
        )
    if (
        not isinstance(revision, str)
        or not re.fullmatch(r"[0-9a-f]{40}", revision)
        or revision == "0" * 40
    ):
        raise ValueError("The trained model needs a nonzero full commit pin")
    if not _hub_model(config["model"]):
        raise ValueError("Evaluation requires a pinned Hub base model")
    selected = inspect_evaluation_checkpoint(run, checkpoint) if checkpoint else None
    completion_path = run / "completion.json"
    completion = (
        read_json(completion_path)
        if not checkpoint or completion_path.exists() or completion_path.is_symlink()
        else {}
    )
    if completion.get("result_kind") == "hosted_pointer":
        raise ValueError("Hosted checkpoints require provider-specific evaluation")
    output = Path(selected["path"]) if selected else _artifact(run, completion)
    if selected:
        artifact_sha, size = selected["checkpoint_sha256"], selected["bytes"]
    else:
        inventory = _inventory(output)
        artifact_sha, size = inventory["sha256"], inventory["bytes"]
    if (
        not (output / "adapter_model.safetensors").is_file()
        or not (output / "adapter_config.json").is_file()
        or (output / "adapter_model.safetensors").stat().st_size == 0
    ):
        raise ValueError("Evaluation requires a local safetensors adapter export")
    adapter = read_json(output / "adapter_config.json")
    if adapter.get("base_model_name_or_path") != config["model"]:
        raise ValueError("Adapter base model differs from the reviewed model")
    training = read_training_data(run / "dataset")
    if hashlib.sha256(training).hexdigest() != plan["dataset_sha256"]:
        raise ValueError("Retained training dataset changed")
    binding: dict[str, Any] = {
        "run_id": run.name,
        "parent_plan_sha256": digest,
        "completion_sha256": content_hash(completion),
        "parent_state_sha256": content_hash(state),
        "checkpoint": checkpoint,
        "checkpoint_receipt_sha256": selected["recovery_sha256"] if selected else None,
        "checkpoint_step": selected["step"] if selected else None,
        "artifact_path": str(output),
        "artifact_sha256": artifact_sha,
        "artifact_bytes": size,
        "training_dataset_sha256": plan["dataset_sha256"],
        "model": config["model"],
        "model_revision": revision,
        "method": config["method"],
        "training_seed": training_seed,
    }
    if nomination_id:
        from stateset_agents.finetune_nomination import verify_nomination

        binding["nomination"] = verify_nomination(run, nomination_id, binding)
    return binding


def evaluation_targets(run: Path) -> dict[str, Any]:
    """Discover terminal local checkpoint metadata without hashing model bytes."""
    run = run.resolve()
    rows = list_evaluation_checkpoints(run)
    state = read_json(run / "state.json")
    if state.get("status") == "succeeded":
        completion = read_json(run / "completion.json")
        output = _artifact(run, completion)
        rows.insert(
            0,
            {
                "checkpoint": "",
                "label": "Final adapter",
                "available": completion.get("result_kind") != "hosted_pointer"
                and all(
                    (output / part).is_file() and not (output / part).is_symlink()
                    for part in ["adapter_model.safetensors", "adapter_config.json"]
                ),
                "bytes_verified": False,
            },
        )
    return {
        "schema_version": 1,
        "run_id": run.name,
        "targets": rows,
        "bytes_verified": False,
        "notes": [
            "Receipt metadata only; a reviewed evaluation rehashes the selected checkpoint before inference.",
            "Published checkpoints use validation cases. Final quality assessment uses separate fresh test cases.",
        ],
    }


def _cases(
    document: dict[str, Any], run: Path, *, split: str = "test"
) -> list[dict[str, Any]]:
    if (
        type(document.get("schema_version")) is not int
        or document["schema_version"] != 1
        or document.get("split") != split
    ):
        raise EvaluationProtocolError("case_partition", split=split)
    provenance = document.get("provenance", {})
    if (
        not isinstance(provenance, dict)
        or provenance.get("source") != "reviewed"
        or not isinstance(provenance.get("reviewer"), str)
        or not provenance["reviewer"].strip()
        or provenance.get(split + "_is_fresh") is not True
    ):
        raise EvaluationProtocolError("provenance", split=split)
    cases = document.get("cases")
    if not isinstance(cases, list) or not 1 <= len(cases) <= 64:
        raise ValueError("Evaluation needs 1 to 64 distinct support cases")
    ids: set[str] = set()
    queries: set[str] = set()
    for case in cases:
        if not isinstance(case, dict) or set(case) != {
            "id",
            "context",
            "user_turns",
            "policy_facts",
            "expected_resolution",
            "must_avoid",
        }:
            raise ValueError(
                "Cases require id, context, user_turns, policy_facts, expected_resolution and must_avoid"
            )
        for key in ("id", "context", "expected_resolution"):
            if (
                not isinstance(case[key], str)
                or not case[key].strip()
                or len(case[key]) > 16384
            ):
                raise ValueError("Case text fields must be nonempty and bounded")
        for key in ("user_turns", "policy_facts", "must_avoid"):
            values = case[key]
            maximum = 8 if key == "user_turns" else 32
            if (
                not isinstance(values, list)
                or len(values) > maximum
                or any(
                    not isinstance(v, str) or not v.strip() or len(v) > 4096
                    for v in values
                )
            ):
                raise ValueError("Case lists must contain bounded nonempty text")
        if not case["user_turns"] or not case["policy_facts"]:
            raise ValueError("Each case needs user turns and policy facts")
        if case["id"] in ids or case["user_turns"][0].strip().casefold() in queries:
            raise ValueError("Case IDs and initial questions must be distinct")
        ids.add(case["id"])
        queries.add(case["user_turns"][0].strip().casefold())
    parent, _ = read_run_plan(run)
    raw = read_training_data(run / "dataset")
    if parent["config"]["method"] == "sft":
        from stateset_agents.data.text_sft import _parse_conversations

        rows = _parse_conversations(raw, "retained training dataset")
        training_queries = {
            turn["content"].strip().casefold()
            for row in rows
            for turn in row["messages"]
            if turn["role"] == "user"
        }
    else:
        from stateset_agents.finetune_workspace import read_scenarios

        rows = read_scenarios(raw)
        if ids & {row["id"] for row in rows}:
            raise ValueError("Test case IDs overlap retained RL training cases")
        training_queries = {row["context"].strip().casefold() for row in rows}
        training_queries.update(
            turn.strip().casefold() for row in rows for turn in row["user_responses"]
        )
    if any(
        turn.strip().casefold() in training_queries
        for case in cases
        for turn in case["user_turns"]
    ):
        raise ValueError("Test turns overlap retained training text")
    content_hash(document)
    if (
        len(json.dumps(document, indent=2, allow_nan=False).encode()) + 1
        > MAX_CASE_BYTES
    ):
        raise ValueError("Private case snapshot exceeds the retained byte limit")
    return cases


def plan_evaluation(run: Path, config: EvaluationConfig) -> EvaluationReview:
    """Capture and check cases/artifacts without models, writes or submission."""
    config.validate()
    run = run.resolve()
    document = read_json(Path(config.cases).expanduser(), max_bytes=MAX_CASE_BYTES)
    cases = _cases(document, run, split=config.split)
    binding = _binding(
        run, checkpoint=config.checkpoint, nomination_id=config.nomination_id
    )
    if config.nomination_id:
        from stateset_agents.finetune_nomination import assert_fresh_test_cases

        assert_fresh_test_cases(run, cases)
    if (config.judge_model, config.judge_revision) == (
        binding["model"],
        binding["model_revision"],
    ):
        raise ValueError("Choose an independent judge model or revision")
    plan = {
        "schema_version": 1,
        "kind": "stateset-local-paired-evaluation",
        "binding": binding,
        "config": asdict(config),
        "cases_sha256": content_hash(document),
        "cases": len(cases),
        "reply_generations": 2 * sum(len(case["user_turns"]) for case in cases),
        "max_judge_generations": 8 * len(cases),
        "rubric_sha256": hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest(),
        "rubric_revision": rubric_revision(),
        "implementation_sha256": content_hash(package_implementation()),
        "decoding": {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0},
        "thresholds": evaluation_thresholds(),
        "model_access": "local_cache_only; trust_remote_code=false",
    }
    return EvaluationReview(
        run, json.loads(json.dumps(plan)), json.loads(json.dumps(document))
    )


def _verify(review: EvaluationReview) -> None:
    if (
        type(review.plan.get("schema_version")) is not int
        or review.plan["schema_version"] != 1
        or review.plan.get("kind") != "stateset-local-paired-evaluation"
    ):
        raise ValueError("Unsupported evaluation plan")
    config = EvaluationConfig(**review.plan["config"])
    config.validate()
    cases = _cases(review.cases, review.run, split=config.split)
    if config.nomination_id:
        from stateset_agents.finetune_nomination import assert_fresh_test_cases

        assert_fresh_test_cases(review.run, cases)
    if (config.judge_model, config.judge_revision) == (
        review.plan["binding"]["model"],
        review.plan["binding"]["model_revision"],
    ):
        raise ValueError("Choose an independent judge model or revision")
    if (
        review.plan["binding"]
        != _binding(
            review.run, checkpoint=config.checkpoint, nomination_id=config.nomination_id
        )
        or review.plan["cases_sha256"] != content_hash(review.cases)
        or review.plan["implementation_sha256"]
        != content_hash(package_implementation())
        or review.plan["rubric_revision"] != rubric_revision()
        or review.plan["rubric_sha256"]
        != hashlib.sha256(RUBRIC_PATH.read_bytes()).hexdigest()
    ):
        raise ValueError("Evaluation inputs or code changed; capture a fresh review")
    # Rebuild derived workload and fixed protocol too; a modified plan cannot
    # lower the declared workload or change acceptance thresholds after review.
    cases = review.cases["cases"]
    if (
        review.plan["cases"] != len(cases)
        or review.plan["reply_generations"]
        != 2 * sum(len(case["user_turns"]) for case in cases)
        or review.plan["max_judge_generations"] != 8 * len(cases)
        or review.plan["thresholds"] != evaluation_thresholds()
        or review.plan["decoding"]
        != {"do_sample": False, "num_beams": 1, "repetition_penalty": 1.0}
        or review.plan["model_access"] != "local_cache_only; trust_remote_code=false"
    ):
        raise ValueError(
            "Evaluation protocol differs from the supported fixed protocol"
        )


def prepare_evaluation(review: EvaluationReview, *, expected_plan_sha256: str) -> Path:
    """Retain one private evaluation snapshot; do not launch or change training."""
    if review.sha256 != expected_plan_sha256:
        raise ValueError("Evaluation checksum differs from the reviewed plan")
    _verify(review)
    directory = review.run / "evaluations"
    if directory.is_symlink():
        raise ValueError("Evaluation directory cannot be a symlink")
    directory.mkdir(mode=0o700, exist_ok=True)
    suffix = uuid.uuid4().hex[:8]
    path = directory / (time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + suffix)
    staging = directory / (".prepare-" + uuid.uuid4().hex)
    staging.mkdir(mode=0o700)
    try:
        write_json(staging / "plan.json", review.plan)
        write_json(staging / "cases.json", review.cases)
        write_json(
            staging / "state.json", {"status": "prepared", "created_at": time.time()}
        )
        staging.rename(path)
    finally:
        import shutil

        if staging.exists():
            shutil.rmtree(staging)
    return path


def read_evaluation(path: Path) -> EvaluationReview:
    """Verify an existing snapshot against its parent and current implementation."""
    if (
        path.is_symlink()
        or path.parent.is_symlink()
        or path.parent.name != "evaluations"
    ):
        raise ValueError("Expected a regular retained evaluation directory")
    review = EvaluationReview(
        path.parent.parent.resolve(),
        read_json(path / "plan.json"),
        read_json(path / "cases.json", max_bytes=MAX_CASE_BYTES),
    )
    _verify(review)
    lock = path / "launch.lock"
    if lock.exists() and read_json(lock).get("plan_sha256") != review.sha256:
        raise ValueError("Evaluation plan changed after launch")
    return review


def launch_evaluation(path: Path, *, expected_plan_sha256: str) -> int:
    """Submit a reviewed snapshot once; the supervisor uses the shared queue."""
    review = read_evaluation(path)
    if review.sha256 != expected_plan_sha256:
        raise ValueError("Evaluation checksum differs from the reviewed plan")
    if not queue_executable():
        raise ValueError(
            "Install Buildr or configure the shared queue before evaluation"
        )
    if read_json(path / "state.json").get("status") != "prepared":
        raise ValueError("Only a prepared evaluation can be launched")
    fd = os.open(path / "launch.lock", os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w") as stream:
        json.dump({"plan_sha256": review.sha256}, stream)
    # A failed submission keeps its exclusive receipt. Never retry a possibly
    # launched supervisor automatically; its state/logs make uncertainty visible.
    try:
        with (path / "run.log").open("ab") as log:
            os.chmod(path / "run.log", 0o600)
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    "stateset_agents.finetune_evaluation_worker",
                    "--supervise",
                    str(path.resolve()),
                ],
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        retain_child(process)
        return int(process.pid)
    except OSError:
        transition_state(
            path,
            allowed_from={"prepared"},
            status="failed",
            error="Supervisor submission failed; inspect logs before preparing a new evaluation",
        )
        raise


def _queue_progress(path: Path) -> dict[str, Any]:
    """Read a bounded lifecycle tail; never inspect the shared queue or expose argv."""
    record = path / "queue-events.jsonl"
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)
    try:
        if record.is_symlink():
            return {}
        with os.fdopen(os.open(record, flags), "rb") as stream:
            info = os.fstat(stream.fileno())
            if not stat.S_ISREG(info.st_mode):
                return {}
            start = max(0, info.st_size - 65536)
            stream.seek(start)
            raw = stream.read(65536)
        if start:
            raw = raw.partition(b"\n")[2]
    except OSError:
        return {}
    result: dict[str, Any] = {}

    def number(value: Any) -> bool:
        return (
            type(value) in {int, float} and 0 <= value <= 1e15 and math.isfinite(value)
        )

    states = {
        "queued",
        "waiting",
        "running",
        "succeeded",
        "failed",
        "cancelled",
        "timed_out",
    }
    reasons = {
        "CPU slot budget": "CPU slots",
        "memory reservation budget": "RAM reservations",
        "available RAM headroom": "free RAM headroom",
        "heavy job budget": "heavy-job capacity",
        "machine concurrency budget": "concurrent-job capacity",
        "queue paused": "queue to resume",
        "legacy worker must drain": "older runner to finish",
        "earlier job in this repository": "earlier repository job",
        "repository already building": "repository lock",
        "disk below ": "free disk headroom",
        "resource key": "shared resource lock",
        "older ": "older queued work",
    }
    for line in raw.split(b"\n")[:-1]:  # An appended partial event is not a receipt.
        try:
            event = json.loads(line)
        except (ValueError, UnicodeError, RecursionError):
            continue
        if not isinstance(event, dict):
            continue
        job = event.get("job_id")
        timestamp = event.get("timestamp")
        if (
            not isinstance(job, str)
            or not re.fullmatch(r"[0-9a-f]{12}", job)
            or not isinstance(event.get("state"), str)
            or event["state"] not in states
            or timestamp is None
            or not number(timestamp)
            or timestamp <= 0
            or result
            and job != result["job_id"]
        ):
            continue
        result.update(job_id=job, state=event["state"], sampled_at=timestamp)
        wait = event.get("wait_seconds")
        if number(wait):
            result["wait_seconds"] = wait
        if event["state"] == "waiting":
            reason = event.get("reason")
            result["waiting_for"] = next(
                (
                    label
                    for prefix, label in reasons.items()
                    if isinstance(reason, str) and reason.startswith(prefix)
                ),
                "another admission constraint",
            )
            resources = event.get("resources")
            if isinstance(resources, dict):
                result["resources"] = {
                    key: resources[key]
                    for key in (
                        "disk_free_gb",
                        "required_free_gb",
                        "available_memory_gb",
                        "required_memory_gb",
                    )
                    if number(resources.get(key))
                }
        else:
            result.pop("waiting_for", None)
            result.pop("resources", None)
    if result:
        result["sample_age_seconds"] = max(0, time.time() - result["sampled_at"])
    return result


def evaluation_status(path: Path) -> dict[str, Any]:
    """Return content-free retained progress; no inference or model hashing."""
    reap_children()
    plan = read_json(path / "plan.json")
    state = read_json(path / "state.json")
    if detached_owner_lost(path, state.get("status")):
        # Reboot/OOM: nobody will ever finish or fail this job. Detached queue
        # work may still be running, so this is unconfirmed, not failed.
        state = {**state, "status": "interrupted", "error": "owner lost"}
    config = plan["config"]
    checkpoint = config.get("checkpoint", "")
    split = config.get("split", "test")
    nomination_id = config.get("nomination_id", "")
    if (
        not isinstance(checkpoint, str)
        or (checkpoint and not re.fullmatch(CHECKPOINT_NAME_PATTERN, checkpoint))
        or not isinstance(split, str)
        or split not in {"test", "validation"}
        or not isinstance(nomination_id, str)
        or (
            nomination_id
            and not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{8}", nomination_id)
        )
    ):
        raise ValueError("Retained evaluation target metadata is invalid")
    public: dict[str, Any] = {
        key: state[key]
        for key in (
            "status",
            "phase",
            "completed_cases",
            "created_at",
            "updated_at",
            "finished_at",
            "exit_code",
        )
        if key in state
    }
    if state.get("error"):
        public["error"] = (
            "Evaluation incomplete; inspect its private log and current inputs before preparing a new job."
        )
    public["execution_unconfirmed"] = state.get("status") == "interrupted"
    return {
        "evaluation_id": path.name,
        "plan_sha256": content_hash(plan),
        "state": public,
        "queue": _queue_progress(path),
        "cases": plan["cases"],
        "reply_generations": plan["reply_generations"],
        "max_judge_generations": plan["max_judge_generations"],
        "judge_model": plan["config"]["judge_model"],
        "checkpoint": checkpoint,
        "split": split,
        "nomination_id": nomination_id,
        "stop_requested": (path / "stop.json").exists(),
        "quality_gain_verified": False,
    }


def list_evaluations(run: Path) -> list[dict[str, Any]]:
    """List up to 30 newest own evaluation records without reading transcripts."""
    directory = run / "evaluations"
    if directory.is_symlink() or not directory.is_dir():
        return []
    rows = []
    for path in sorted(directory.iterdir(), reverse=True)[:30]:
        if (
            path.is_symlink()
            or not path.is_dir()
            or not re.fullmatch(r"[0-9]{8}-[0-9]{6}-[0-9a-f]{8}", path.name)
        ):
            continue
        try:
            rows.append(evaluation_status(path))
        except (OSError, ValueError, TypeError, KeyError):
            rows.append({"evaluation_id": path.name, "state": {"status": "unreadable"}})
    return rows


def stop_evaluation(path: Path, *, expected_plan_sha256: str) -> None:
    """Request cooperative stop of only the exact reviewed evaluation."""
    plan = read_json(path / "plan.json")
    if content_hash(plan) != expected_plan_sha256:
        raise ValueError("Evaluation checksum differs from the reviewed plan")
    write_json(path / "stop.json", {"plan_sha256": expected_plan_sha256})
    if not (path / "launch.lock").exists() or launch_unclaimed(path):
        transition_state(
            path, allowed_from={"prepared"}, status="cancelled", finished_at=time.time()
        )


def summarize_evaluation(
    review: EvaluationReview, reports: dict[str, Any]
) -> dict[str, Any]:
    """Bind one recorded pair to this evaluation, partition and selected artifact."""
    from stateset_agents.finetune_quality import _summarize

    if (
        reports.get("schema_version") != 1
        or reports.get("review") != review.cases["provenance"]
    ):
        raise ValueError("Evaluation provenance differs from its reviewed cases")
    config, binding = review.plan["config"], review.plan["binding"]
    for arm in ("baselines", "candidates"):
        rows = reports.get(arm)
        if (
            not isinstance(rows, list)
            or len(rows) != 1
            or not isinstance(rows[0], dict)
        ):
            raise ValueError(
                "One evaluation must contain exactly one baseline and candidate"
            )
        if rows[0].get("seed") != report_training_seed(review) or rows[0].get(
            "settings"
        ) != report_settings(review):
            raise ValueError(
                "Evaluation report protocol differs from its reviewed plan"
            )
    baseline, candidate = reports["baselines"][0], reports["candidates"][0]
    if baseline.get("selected_checkpoint") != {
        "path": binding["model"] + "@" + binding["model_revision"]
    } or candidate.get("selected_checkpoint") != {
        "path": binding["artifact_path"],
        "inventory_sha256": binding["artifact_sha256"],
    }:
        raise ValueError("Evaluation reports must match the reviewed base and adapter")
    return _summarize(
        review.run,
        reports,
        expected_split=config.get("split", "test"),
        artifact=Path(binding["artifact_path"]) if config.get("checkpoint") else None,
        nomination_id=config.get("nomination_id", ""),
    )


def inspect_evaluation(path: Path) -> dict[str, Any]:
    """Rehash weights and replay the retained judge without running any model."""
    from stateset_agents.evaluation.judge_evidence import replay_judge_evidence

    review = read_evaluation(path)
    if read_json(path / "state.json").get("status") != "succeeded":
        raise ValueError("Evaluation has not completed successfully")
    result = read_json(path / "result.json", max_bytes=4 * 1024 * 1024)
    reports = read_json(path / "paired-reports.json", max_bytes=4 * 1024 * 1024)
    if result.get("evaluation_plan_sha256") != review.sha256 or result.get(
        "summary"
    ) != summarize_evaluation(review, reports):
        raise ValueError("Evaluation result differs from the retained reports")
    means = {}
    for index, arm in enumerate(("baseline", "candidate")):
        recording = result["judge_recordings"][arm]
        directory = path / "judge-recordings" / f"batch-{index:06d}"
        if directory.parent.is_symlink() or Path(recording["path"]) != directory:
            raise ValueError("Judge recordings must belong to this evaluation")
        _inventory(directory, max_bytes=64 * 1024 * 1024, max_files=32)
        replay = replay_judge_evidence(
            directory, expected_manifest_sha256=recording["manifest_sha256"]
        )
        if replay["context"] != {
            "split": review.plan["config"].get("split", "test"),
            "arm": arm,
            "evaluation_plan_sha256": review.sha256,
        } or replay["judge"] != {
            "model": review.plan["config"]["judge_model"],
            "revision": review.plan["config"]["judge_revision"],
            "rubric_revision": review.plan["rubric_revision"],
        }:
            raise ValueError("Judge recording protocol differs from the evaluation")
        report = reports["baselines" if arm == "baseline" else "candidates"][0]
        source = read_json(path / (arm + ".json"), max_bytes=MAX_TRANSCRIPT_BYTES)
        if source["evaluation_plan_sha256"] != review.sha256 or len(
            source["rows"]
        ) != len(review.cases["cases"]):
            raise ValueError("Transcript binding differs")
        if (
            report["case_hashes"]
            != {case["id"]: content_hash(case) for case in review.cases["cases"]}
            or report["seed"] != report_training_seed(review)
            or report["settings"] != report_settings(review)
        ):
            raise ValueError(
                "Paired report protocol differs from the reviewed evaluation"
            )
        means[arm] = sum(score["score"] for score in replay["results"]) / len(
            replay["results"]
        )
        for case, row, item, score in zip(
            review.cases["cases"],
            source["rows"],
            replay["items"],
            replay["results"],
            strict=True,
        ):
            # Reports sort IDs, while model/judge phases preserve input order.
            outcome = next(o for o in report["outcomes"] if o["case_id"] == case["id"])
            if (
                item["id"] != case["id"]
                or row["case_id"] != case["id"]
                or item["scenario"] != case
                or item["transcript"] != row["transcript"]
                or len(row["transcript"]) != 2 * len(case["user_turns"])
                or row["transcript"][::2]
                != [["user", turn] for turn in case["user_turns"]]
                or any(
                    turn[0] != "assistant" or not turn[1].strip()
                    for turn in row["transcript"][1::2]
                )
                or score["id"] != outcome["case_id"]
                or score["score"] != outcome["reward"]
                or score["parse_failures"]
                or score["disagreement"] > 0.2
            ):
                raise ValueError("Replay differs from the paired outcome")
            violations = int(
                any(
                    not all(
                        value
                        for key, value in checks.items()
                        if key.startswith("policy_")
                    )
                    for checks in score["checks"]
                )
            )
            if (
                outcome["policy_violations"] != violations
                or outcome["success"]
                != judge_success(score["score"], score["checks"], row["truncated"])
                or any(
                    outcome[key] != row[key]
                    for key in ("truncated", "generated_tokens", "elapsed_seconds")
                )
            ):
                raise ValueError("Outcome differs from the generated/judged transcript")
    if result["mean_judge_scores"] != means:
        raise ValueError("Judge score summary differs from replay")
    return {
        "evaluation_id": path.name,
        "evaluation_plan_sha256": review.sha256,
        "summary": result["summary"],
        "mean_judge_scores": means,
        "judge_replay_verified": True,
        "checkpoint": review.plan["config"].get("checkpoint", ""),
        "split": review.plan["config"].get("split", "test"),
        "quality_gain_verified": False,
        "notes": evaluation_notes(review.plan["config"].get("split", "test")),
    }


def report_training_seed(review: EvaluationReview) -> int:
    """Label pairs by recorded training seed; old runs retain a legacy label.

    A missing training seed stays unknown in the binding and quality gate;
    varying an evaluation seed never establishes training independence.
    """
    seed = review.plan["binding"].get("training_seed")
    if seed is None:
        seed = review.plan["config"]["seed"]
    if type(seed) is not int or not 0 <= seed <= 2**32 - 1:
        raise ValueError("The paired report seed is invalid")
    return int(seed)


def report_settings(review: EvaluationReview) -> dict[str, Any]:
    """Shared exact paired-report protocol, also required during replay."""
    config, binding = review.plan["config"], review.plan["binding"]
    return {
        "model_revision": binding["model_revision"],
        "judge_model": config["judge_model"],
        "judge_revision": config["judge_revision"],
        "rubric_revision": review.plan["rubric_sha256"],
        "max_new_tokens": config["max_new_tokens"],
        "context_length": config["context_length"],
        "decoding": review.plan["decoding"],
        "evaluation_protocol": "scripted_support_v1",
        "evaluation_partition": config.get("split", "test"),
        "evaluation_seed": config["seed"],
    }


def evaluation_notes(split: str = "test") -> list[str]:
    """State what a single retained diagnostic can establish."""
    return [
        *(
            [
                "Validation supports checkpoint selection; final quality assessment requires separate fresh test cases."
            ]
            if split == "validation"
            else []
        ),
        "One retained local diagnostic; independent training seeds and reviewed held-out data are still required for a quality study.",
        "Greedy scripted user turns do not evaluate tools, adaptive customers or deployment behavior.",
    ]


def evaluation_thresholds() -> dict[str, Any]:
    """Predeclare critical checks so a high aggregate cannot hide a failed outcome."""
    return {
        "success_score": 0.8,
        "max_disagreement": 0.2,
        "required_checks": list(REQUIRED_CHECKS),
    }


def judge_success(
    score: float, checks: Sequence[Mapping[str, bool]], truncated: str | None
) -> bool:
    """Require both passes to resolve correctly, obey facts/policy and avoid repetition."""
    return (
        score >= 0.8
        and truncated is None
        and len(checks) == 2
        and all(
            all(answers.get(key) is True for key in REQUIRED_CHECKS)
            for answers in checks
        )
    )


def evaluation_schema() -> dict[str, Any]:
    """Describe the supported evaluation settings without reading files or models."""
    defaults = asdict(EvaluationConfig())
    properties = {
        name: {
            "type": (
                "integer"
                if type(value) is int
                else "number" if type(value) is float else "string"
            ),
            "default": value,
        }
        for name, value in defaults.items()
    }
    limits = {
        "max_new_tokens": (1, 2048),
        "judge_max_new_tokens": (128, 4096),
        "context_length": (512, 65536),
        "seed": (0, 2**32 - 1),
        "timeout_seconds": (30, 86400),
        "cpu_slots": (1, 32),
        "memory_gb": (0, 1024),
        "scratch_gb": (0, 4096),
    }
    for name, (low, high) in limits.items():
        properties[name].update(
            {"minimum" if low else "exclusiveMinimum": low, "maximum": high}
        )
    # Seed legitimately accepts zero.
    properties["seed"].pop("exclusiveMinimum", None)
    properties["seed"]["minimum"] = 0
    properties["judge_revision"]["pattern"] = r"^(?!0{40}$)[0-9a-f]{40}$"
    properties["judge_model"]["pattern"] = _HUB_ID_PATTERN
    properties["cases"]["pattern"] = r"\S"
    properties["split"]["enum"] = ["test", "validation"]
    properties["checkpoint"]["pattern"] = "^(?:" + CHECKPOINT_NAME_PATTERN + ")?$"
    properties["nomination_id"]["pattern"] = r"^(?:[0-9]{8}-[0-9]{6}-[0-9a-f]{8})?$"
    return {
        "methods": ["sft", "grpo", "gspo", "dapo"],
        "requires": "successful_local_final_or_terminal_published_checkpoint, pinned_base_and_independent_judge, safetensors_adapter, cached_models, reviewed_partition_cases, Buildr",
        "config_schema": {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
            "required": ["cases", "judge_model", "judge_revision"],
            "allOf": [
                {
                    "if": {
                        "properties": {"checkpoint": {"minLength": 1}},
                        "required": ["checkpoint"],
                        "not": {
                            "properties": {"nomination_id": {"minLength": 1}},
                            "required": ["nomination_id"],
                        },
                    },
                    "then": {
                        "properties": {"split": {"const": "validation"}},
                        "required": ["split"],
                    },
                },
                {
                    "if": {
                        "properties": {"nomination_id": {"minLength": 1}},
                        "required": ["nomination_id"],
                    },
                    "then": {
                        "properties": {
                            "checkpoint": {"minLength": 1},
                            "split": {"const": "test"},
                        },
                        "required": ["checkpoint", "split"],
                    },
                },
            ],
        },
        "config_defaults": defaults,
        "success_thresholds": evaluation_thresholds(),
        "queue_progress": {
            "source": "bounded_retained_lifecycle_events",
            "fields": [
                "job_id",
                "state",
                "wait_seconds",
                "waiting_for",
                "resources",
                "sampled_at",
                "sample_age_seconds",
            ],
            "stale_sample_proves_failure": False,
        },
        "max_cases": 64,
        "max_user_turns_per_case": 8,
        "max_cases_bytes": MAX_CASE_BYTES,
        "selection_or_promotion": False,
        "evaluation_target_selection": True,
        "published_checkpoint_evaluation": "validation; fresh test requires an explicit byte-bound nomination and disjoint validation history",
        "http_execution_supported": False,
    }
