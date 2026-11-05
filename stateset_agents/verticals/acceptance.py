"""Revalidate native learning and hardware contracts for the existing A+ gate.

These are local artifact-integrity checks, not independent execution attestation.
The broad publication decision remains ``benchmarks/a_plus_gate.py``. Tiny
checkpoint contracts establish neither pretrained-model quality nor scaling.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from stateset_agents.evaluation.agent_runs import (
    compare_runs,
    content_hash,
    validate_report,
)
from stateset_agents.utils.provenance import validate_implementation

NATIVE_GPU_HARNESS_FILES = (
    "benchmarks/native_gpu_checkpoint_contract.py",
    "tests/integration/test_ddp_checkpoint_publication.py",
    "tests/integration/test_native_ddp_checkpoint.py",
    "tests/integration/test_replica_consensus.py",
    "tests/integration/test_learner_preflight.py",
    "tests/unit/test_token_group_aggregation.py",
)
NATIVE_GPU_MODES = ("true", "none", "absent", "disabled", "false", "exception", "rng")


def _require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def _file(root: Path, relative: str) -> Path:
    _require(isinstance(relative, str), "Artifact path must be a string")
    name = PurePosixPath(relative)
    _require(
        bool(relative)
        and not name.is_absolute()
        and ".." not in name.parts
        and "\\" not in relative
        and name.as_posix() == relative,
        "Artifact paths must be canonical relative paths",
    )
    path = root
    for part in name.parts:
        path = path / part
        _require(not path.is_symlink(), "Artifact paths cannot contain symlinks")
    _require(path.is_file(), f"Missing regular artifact: {relative}")
    return path


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            _require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    value = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=pairs)
    if not isinstance(value, dict):
        raise ValueError("Artifact must contain a JSON object")
    content_hash(value)
    return value


def native_gpu_harness_identity(root: Path) -> dict[str, str]:
    """Bind the checkout-only producer and its actual native test fixtures."""
    return {name: _digest(_file(root, name)) for name in NATIVE_GPU_HARNESS_FILES}


def audit_native_quality(
    directory: Path, *, expected_implementation: Mapping[str, Any]
) -> dict[str, Any]:
    """Recompute the existing reviewed six-seed study from its retained outputs.

    Review provenance remains an attestation. Independent training and absence
    of semantic train/test leakage require human review. Checkpoints are hashed
    in place; no model is allocated and no pickle is loaded by this audit.
    """
    from stateset_agents.training.checkpoint_publication import (
        validate_checkpoint_publication,
    )
    from stateset_agents.training.native_refund_study import (
        load_cases,
        validate_study_config,
    )

    validate_implementation(expected_implementation)
    _require(
        not directory.is_symlink() and directory.is_dir(), "Missing study directory"
    )
    artifacts: dict[str, str] = {}

    def read(relative: str) -> dict[str, Any]:
        path = _file(directory, relative)
        artifacts[relative] = _digest(path)
        return _json(path)

    manifest = read("manifest.json")
    _require(manifest.get("synthetic") is False, "Synthetic studies cannot qualify")
    validate_study_config(Namespace(**manifest))
    _require(
        manifest.get("implementation") == expected_implementation,
        "Study implementation differs from the release",
    )
    cases_path = _file(directory, "cases.json")
    artifacts["cases.json"] = _digest(cases_path)
    _json(cases_path)
    cases = load_cases(cases_path)
    _require(
        manifest.get("cases_sha256") == content_hash(cases)
        and manifest.get("provenance") == cases["provenance"],
        "Reviewed cases differ from the study manifest",
    )
    baselines, candidates = [], []
    expected_steps = [0] + [
        step
        for step in range(1, manifest["updates"] + 1)
        if step % manifest["eval_every"] == 0 or step == manifest["updates"]
    ]
    for seed in manifest["seeds"]:
        prefix = f"seed-{seed}"
        baseline, candidate = read(f"{prefix}/baseline.json"), read(
            f"{prefix}/candidate.json"
        )
        selection = read(f"{prefix}/selection.json")
        updates_path = _file(directory, f"{prefix}/updates.json")
        artifacts[f"{prefix}/updates.json"] = _digest(updates_path)
        updates = json.loads(updates_path.read_text(encoding="utf-8"))
        content_hash(updates)
        _require(
            isinstance(updates, list) and len(updates) == manifest["updates"],
            "Training update ledger is incomplete",
        )
        committed = 0
        for row in updates:
            _require(isinstance(row, dict), "Invalid training update ledger")
            count = row.get("optimizer_updates")
            _require(
                type(count) is int and count >= 0, "Invalid committed update count"
            )
            committed += count
            _require(
                type(row.get("global_step")) is int and row["global_step"] == committed,
                "Training update counters disagree",
            )
        _require(committed > 0, "Study did not commit an optimizer update")
        _require(
            selection.get("schema_version") == 1
            and selection.get("criterion") == "validation_success_rate"
            and selection.get("tie_break") == "earliest_step"
            and selection.get("committed_optimizer_updates") == committed,
            "Invalid validation checkpoint selection protocol",
        )
        selected_step = selection.get("selected_step")
        _require(
            type(selected_step) is int
            and 0 < selected_step <= manifest["updates"]
            and sum(row["optimizer_updates"] for row in updates[:selected_step]) > 0,
            "Selected checkpoint must follow a committed training update",
        )
        references = selection.get("validation_reports")
        if (
            not isinstance(references, list)
            or [row.get("step") for row in references if isinstance(row, dict)]
            != expected_steps
        ):
            raise ValueError("Validation schedule differs from the predeclared study")
        validation_results = []
        for row in references:
            name = f"validation-{row['step']}.json"
            _require(row.get("report") == name, "Unexpected validation report path")
            report = read(f"{prefix}/{name}")
            _require(
                row.get("sha256") == content_hash(report), "Validation report changed"
            )
            summary = validate_report(report, expected_split="validation")
            _check_report(
                report, cases["validation"], manifest, seed, expected_implementation
            )
            asyncio.run(_replay_native_report(report, cases["validation"]))
            _require(
                report["selected_checkpoint"]["path"]
                == ("initial" if row["step"] == 0 else str(row["step"])),
                "Validation checkpoint identity differs from its training step",
            )
            validation_results.append((row["step"], summary["success_rate"]))
        best_step, best_score = max(validation_results, key=lambda item: item[1])
        _require(
            selection.get("selected_step") == best_step
            and selection.get("selected_success_rate") == best_score,
            "Selected checkpoint is not the earliest best validation checkpoint",
        )
        for report in (baseline, candidate):
            _check_report(
                report, cases["test"], manifest, seed, expected_implementation
            )
            asyncio.run(_replay_native_report(report, cases["test"]))
        _require(
            baseline["selected_checkpoint"]["path"]
            == f"hf://{manifest['model']}@{manifest['model_revision']}",
            "Baseline model identity differs from the pinned study model",
        )
        selected = directory / prefix / "selected"
        _require(not selected.is_symlink(), "Selected checkpoint cannot be symlinked")
        validate_checkpoint_publication(selected, required=True)
        inventory = read(f"{prefix}/selected/.stateset-checkpoint.json")
        inventory_hash = content_hash(inventory)
        _require(
            selection.get("checkpoint_inventory_sha256") == inventory_hash
            and candidate["selected_checkpoint"].get("inventory_sha256")
            == inventory_hash
            and candidate["selected_checkpoint"].get("selection_sha256")
            == content_hash(selection),
            "Candidate report is not bound to the selected checkpoint",
        )
        # The runner records its original path; the bundle may later be moved.
        recorded_root = Path(manifest["output"])
        _require(
            candidate["selected_checkpoint"]["path"]
            == str(recorded_root / prefix / "selected"),
            "Candidate checkpoint path differs from its study seed",
        )
        for name, digest in inventory["files"].items():
            artifacts[f"{prefix}/selected/{name}"] = digest
        baselines.append(baseline)
        candidates.append(candidate)
    recomputed = compare_runs(
        baselines, candidates, min_seeds=6, require_significance=True
    )
    recomputed["representative_cases"] = True
    recorded = read("comparison.json")
    _require(
        recorded == recomputed, "Stored comparison differs from recomputed evidence"
    )
    _require(recomputed["passed"] is True, "Representative native quality gate failed")
    return {
        "schema_version": 1,
        "kind": "stateset-native-quality-acceptance",
        "passed": True,
        "comparison": recomputed,
        "artifacts": artifacts,
        "scope": "Reviewed native refund learning evidence; review and execution are not independently attested.",
    }


def _check_report(
    report: Mapping[str, Any],
    cases: list[dict[str, Any]],
    manifest: Mapping[str, Any],
    seed: int,
    implementation: Mapping[str, Any],
) -> None:
    from stateset_agents.evaluation.isolation import (
        EVALUATION_RNG_PROTOCOL,
        paired_evaluation_seed,
    )

    _require(
        report.get("seed") == seed
        and report.get("environment") == "refund-policy-v2"
        and report.get("base_model") == manifest["model"]
        and report.get("implementation") == implementation
        and report.get("settings")
        == {
            "model_revision": manifest["model_revision"],
            "temperature": 1.0,
            "max_new_tokens": 192,
            "max_turns": 4,
            "evaluation_randomness": {
                "protocol": EVALUATION_RNG_PROTOCOL,
                "scope": "local-global-rngs",
            },
        }
        and report.get("case_hashes")
        == {case["order_id"]: content_hash(case) for case in cases},
        "Evaluation source, model, seed or reviewed cases differ from the study",
    )
    for outcome in report.get("outcomes", []):
        _require(
            outcome.get("evaluation_seed")
            == paired_evaluation_seed(
                seed, outcome["case_id"], report["case_hashes"][outcome["case_id"]]
            ),
            "Evaluation outcome does not use its paired per-case seed",
        )


async def _replay_native_report(
    report: Mapping[str, Any], cases: list[dict[str, Any]]
) -> None:
    """Re-execute retained native actions against the isolated refund ledger."""
    from stateset_agents.core.environments.refund_policy_environment import (
        RefundPolicyEnvironment,
    )
    from stateset_agents.core.trajectory import ConversationTurn

    by_id = {case["order_id"]: case for case in cases}
    for outcome in report["outcomes"]:
        trace = outcome.get("trace")
        _require(
            isinstance(trace, list) and bool(trace), "Missing native evaluation trace"
        )
        actions = iter(row for row in trace if row.get("role") == "assistant")

        async def policy(messages: Any, context: Any, actions: Any = actions) -> Any:
            action = next(actions, None)
            if action is None:
                raise ValueError("Native trace ended before environment completion")
            return ConversationTurn(
                role="assistant",
                content=action["content"],
                metadata={"token_ids": action.get("token_ids")},
            )

        trajectory = await RefundPolicyEnvironment().run_episode(
            policy, by_id[outcome["case_id"]]
        )
        rendered = [
            {
                "role": turn.role,
                "content": turn.content,
                "token_ids": (turn.metadata or {}).get("token_ids"),
            }
            for turn in trajectory.turns
        ]
        state = trajectory.final_state or {}
        _require(
            content_hash(rendered) == content_hash(trace)
            and next(actions, None) is None
            and outcome["success"]
            is bool(
                state.get("resolution") is not None
                and state.get("violations") == 0
                and trajectory.metadata.get("status") == "completed"
            )
            and outcome["policy_violations"] == int(state.get("violations", 0))
            and outcome["reward"] == trajectory.total_reward
            and outcome["truncated"]
            == (
                "turn_limit" if trajectory.metadata.get("status") == "timeout" else None
            ),
            "Native evaluation trace disagrees with replayed ledger outcomes",
        )


def _same_snapshot(actual: Any, expected: Any, torch: Any) -> None:
    if torch.is_tensor(actual):
        _require(torch.is_tensor(expected), "Snapshot tensor type differs")
        _require(
            actual.dtype == expected.dtype
            and actual.shape == expected.shape
            and torch.equal(actual, expected),
            "Native next-update tensor state differs",
        )
    elif isinstance(actual, dict):
        _require(
            isinstance(expected, dict) and actual.keys() == expected.keys(),
            "Snapshot keys differ",
        )
        for key in actual:
            _same_snapshot(actual[key], expected[key], torch)
    elif isinstance(actual, (list, tuple)):
        _require(
            type(actual) is type(expected) and len(actual) == len(expected),
            "Snapshot population differs",
        )
        for value, wanted in zip(actual, expected, strict=True):
            _same_snapshot(value, wanted, torch)
    else:
        _require(
            type(actual) is type(expected) and actual == expected,
            "Snapshot scalar differs",
        )


def _audit_native_gpu_recovery(
    directory: Path,
    *,
    expected_implementation: Mapping[str, Any],
    expected_harness: Mapping[str, str],
) -> dict[str, Any]:
    """Recompare actual retained native CUDA/NCCL continuation snapshots.

    The producer executes the same worker as the CPU/Gloo publication tests.
    Device metadata alone is insufficient: every retained learner/RNG snapshot
    must match its uninterrupted next update, and each injected late failure
    must contain concrete rejection records followed by successful recovery.
    This is tiny single-node DDP/AMP mechanics, not multi-node or quality proof.
    """
    from stateset_agents.core.checkpoint_io import load_checkpoint_file

    validate_implementation(expected_implementation)
    _require(
        not directory.is_symlink() and directory.is_dir(), "Missing native GPU bundle"
    )
    report = _json(_file(directory, "contract.json"))
    _require(
        type(report.get("schema_version")) is int
        and report["schema_version"] == 1
        and report.get("kind") == "stateset-native-checkpoint-hardware-contract"
        and report.get("status") == "completed"
        and report.get("device") == "cuda"
        and report.get("backend") == "nccl"
        and type(report.get("world_size")) is int
        and report["world_size"] == 2,
        "Completed two-GPU CUDA/NCCL native checkpoint evidence required",
    )
    _require(
        report.get("implementation") == expected_implementation
        and report.get("harness") == expected_harness
        and isinstance(expected_harness, Mapping)
        and set(expected_harness) == set(NATIVE_GPU_HARNESS_FILES)
        and all(
            isinstance(value, str)
            and len(value) == 64
            and all(character in "0123456789abcdef" for character in value)
            for value in expected_harness.values()
        ),
        "Native GPU implementation or test harness differs from the release",
    )
    artifacts = report.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Missing native GPU artifacts")
    for name, digest in artifacts.items():
        _require(
            isinstance(name, str)
            and isinstance(digest, str)
            and len(digest) == 64
            and all(character in "0123456789abcdef" for character in digest)
            and _digest(_file(directory, name)) == digest,
            "Native GPU retained artifact changed",
        )
    expected_cases = {
        (api, pending, precision)
        for api in ("runtime", "async")
        for pending in (False, True)
        for precision in ("fp32", "fp16")
    }
    records = report.get("cases")
    if not isinstance(records, list) or len(records) != len(expected_cases):
        raise ValueError("Incomplete native GPU case matrix")
    observed_cases = set()
    import torch

    def snapshot(relative: str) -> Any:
        _require(
            isinstance(relative, str) and relative in artifacts,
            "Snapshot is absent from retained inventory",
        )
        return load_checkpoint_file(_file(directory, relative), map_location="cpu")

    for case in records:
        _require(isinstance(case, dict), "Invalid native GPU case")
        _require(
            isinstance(case.get("api"), str)
            and isinstance(case.get("precision"), str)
            and type(case.get("pending")) is bool,
            "Invalid native GPU case roster",
        )
        key = (case.get("api"), case.get("pending"), case.get("precision"))
        _require(
            type(case.get("pending")) is bool
            and key in expected_cases
            and key not in observed_cases,
            "Invalid native GPU case roster",
        )
        observed_cases.add(key)
        reports = case.get("rank_reports")
        if not isinstance(reports, list) or len(reports) != 2:
            raise ValueError("Both rank reports required")
        rank_streams = []
        for rank, relative in enumerate(reports):
            _require(
                isinstance(relative, str) and relative in artifacts,
                "Rank report is absent from retained inventory",
            )
            peer = _json(_file(directory, relative))
            _require(
                type(peer.get("rank")) is int
                and peer["rank"] == rank
                and peer.get("device") == f"cuda:{rank}"
                and peer.get("backend") == "nccl"
                and type(peer.get("world_size")) is int
                and peer["world_size"] == 2
                and isinstance(peer.get("gpu"), str)
                and bool(peer["gpu"])
                and isinstance(peer.get("cuda"), str)
                and bool(peer["cuda"])
                and peer.get("api") == key[0]
                and peer.get("pending") is key[1]
                and peer.get("precision") == key[2],
                "Rank-observed CUDA/NCCL topology differs from the contract",
            )
            modes = peer.get("scenarios")
            if (
                not isinstance(modes, list)
                or not all(isinstance(row, dict) for row in modes)
                or [row.get("mode") for row in modes] != list(NATIVE_GPU_MODES)
            ):
                raise ValueError("Missing native GPU publication scenarios")
            for row in modes:
                evidence = snapshot(row["snapshot"])
                _require(isinstance(evidence, dict), "Invalid learner snapshot")
                for phase in ("checkpoint", "restored", "expected", "actual"):
                    _require(
                        isinstance(evidence.get(phase), dict)
                        and set(evidence[phase])
                        == {
                            "policy",
                            "reference",
                            "optimizer",
                            "scheduler",
                            "scaler",
                            "gradients",
                            "normalization",
                            "counters",
                            "baseline",
                            "rng",
                        }
                        and bool(evidence[phase]["policy"])
                        and bool(evidence[phase]["reference"]),
                        "Native continuation snapshot omits learner components",
                    )
                _same_snapshot(evidence["actual"], evidence["expected"], torch)
                _same_snapshot(evidence["restored"], evidence["checkpoint"], torch)
                _same_snapshot(
                    evidence["actual_draw"], evidence["expected_draw"], torch
                )
                _require(
                    evidence["expected"]["counters"][0] == 2
                    and evidence["checkpoint"]["counters"][0] == 1
                    and evidence["checkpoint"]["counters"][3] == (3 if key[1] else 1)
                    and evidence["expected"]["counters"][3] == (4 if key[1] else 2)
                    and bool(evidence["expected"]["optimizer"]["state"]),
                    "Checkpoint does not prove a committed stochastic continuation",
                )
                if key[1]:
                    _require(
                        isinstance(evidence["checkpoint"]["normalization"], dict)
                        and evidence["checkpoint"]["normalization"].get(
                            "policy_count", 0
                        )
                        > 0
                        and any(
                            torch.is_tensor(value)
                            for value in evidence["checkpoint"]["gradients"]
                        ),
                        "Pending checkpoint omits its live gradient/count window",
                    )
                _require(
                    any(
                        not torch.equal(value, evidence["checkpoint"]["policy"][name])
                        for name, value in evidence["expected"]["policy"].items()
                    ),
                    "Continuation did not change policy weights",
                )
                _require(
                    bool(evidence["checkpoint"]["rng"]["cuda"])
                    and bool(evidence["expected"]["rng"]["cuda"]),
                    "Native GPU snapshots omit CUDA RNG state",
                )
                if key[2] == "fp16":
                    _require(
                        isinstance(evidence["checkpoint"].get("scaler"), dict)
                        and evidence["checkpoint"]["scaler"].get("scale", 0) > 0,
                        "FP16 continuation omits active GradScaler state",
                    )
                    _require(
                        evidence["expected"]["scaler"]["_growth_tracker"]
                        == (evidence["checkpoint"]["scaler"]["_growth_tracker"] + 1)
                        % evidence["checkpoint"]["scaler"]["growth_interval"],
                        "FP16 continuation did not advance GradScaler",
                    )
                _require(
                    torch.is_tensor(evidence.get("forward_probe"))
                    and evidence["forward_probe"].dtype
                    == (torch.float16 if key[2] == "fp16" else torch.float32)
                    and bool(torch.isfinite(evidence["forward_probe"]).all()),
                    "Native CUDA forward did not exercise its declared precision",
                )
                if row["mode"] in {"false", "exception", "rng"}:
                    failure = row.get("failure")
                    _require(
                        isinstance(failure, dict)
                        and all(
                            isinstance(failure.get(name), str) and failure[name]
                            for name in ("load_error", "train_error", "save_error")
                        )
                        and ("rng-restore" if row["mode"] == "rng" else "rollout-sync")
                        in failure["load_error"]
                        and "preflight" in failure["train_error"]
                        and "capture" in failure["save_error"]
                        and failure.get("restore_failed") is True
                        and failure.get("consensus_pending") is True,
                        "Injected late failure lacks quarantine/rejection evidence",
                    )
                if row["mode"] == "true":
                    rank_streams.append(evidence["expected_draw"])
        _require(
            len(rank_streams) == 2
            and rank_streams[0][0] != rank_streams[1][0]
            and rank_streams[0][1] != rank_streams[1][1]
            and not torch.equal(rank_streams[0][2], rank_streams[1][2])
            and not torch.equal(rank_streams[0][3], rank_streams[1][3]),
            "Rank-local Python/NumPy/Torch/CUDA streams must be distinct",
        )
    return {
        "schema_version": 1,
        "kind": "stateset-native-gpu-recovery-acceptance",
        "passed": True,
        "cases": len(records),
        "scenarios_per_rank": len(records) * len(NATIVE_GPU_MODES),
        "artifacts": {
            **artifacts,
            "contract.json": _digest(directory / "contract.json"),
        },
        "scope": "Tiny single-node native CUDA/NCCL and FP16 checkpoint mechanics; not released-weight quality, scaling or independent provider attestation.",
    }


def audit_native_gpu_recovery(
    directory: Path,
    *,
    expected_implementation: Mapping[str, Any],
    expected_harness: Mapping[str, str],
) -> dict[str, Any]:
    """Revalidate raw native CUDA/NCCL snapshots, rejecting malformed bundles."""
    try:
        return _audit_native_gpu_recovery(
            directory,
            expected_implementation=expected_implementation,
            expected_harness=expected_harness,
        )
    except (TypeError, KeyError, IndexError, AttributeError, ZeroDivisionError) as exc:
        raise ValueError("Malformed native GPU checkpoint evidence") from exc
