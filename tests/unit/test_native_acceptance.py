"""Raw native study/hardware evidence is required; summary flags never suffice.

The passing study fixture below is invented test data, not reviewed evidence.
No model training, downloads, devices, or providers are used in this module.
"""

from __future__ import annotations

import asyncio
import copy
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    _resolution,
    refund_policy_benchmark,
)
from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation import acceptance
from stateset_agents.evaluation.agent_runs import (
    compare_runs,
    content_hash,
    evaluation_report,
)
from stateset_agents.evaluation.isolation import (
    EVALUATION_RNG_PROTOCOL,
    paired_evaluation_seed,
)

IDENTITY = {
    "schema_version": 1,
    "scope": "stateset_agents_python_sources",
    "files": {"__init__.py": "a" * 64},
}


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _read(path):
    return json.loads(path.read_text())


@pytest.fixture
def study(tmp_path):
    root = tmp_path / "study"
    cases = {
        "provenance": {"source": "reviewed", "reviewer": "synthetic-unit-fixture-only"},
        **{
            split: refund_policy_benchmark(split, 32, 0)
            for split in ("train", "validation", "test")
        },
    }
    manifest = {
        "synthetic": False,
        "model": "fixture/policy",
        "model_revision": "a" * 40,
        "seeds": list(range(6)),
        "updates": 2,
        "generations": 2,
        "eval_every": 1,
        "learning_rate": 1e-5,
        "output": str(root),
        "cases_sha256": content_hash(cases),
        "provenance": cases["provenance"],
        "implementation": IDENTITY,
    }
    _write(root / "manifest.json", manifest)
    _write(root / "cases.json", cases)
    baselines, candidates = [], []

    async def replay_outcomes(split, success):
        outcomes = []
        for case in cases[split]:
            calls = 0

            async def policy(messages, context, case=case):
                nonlocal calls
                calls += 1
                if not success or calls == 3:
                    tool, args = "finish", {}
                elif calls == 1:
                    tool, args = "lookup_order", {"order_id": case["order_id"]}
                else:
                    tool, value = _resolution(case)
                    args = {
                        "order_id": case["order_id"],
                        "amount_cents" if tool == "refund" else "reason": value,
                    }
                return ConversationTurn(
                    role="assistant", content=json.dumps({"tool": tool, "args": args})
                )

            trajectory = await RefundPolicyEnvironment().run_episode(policy, case)
            outcomes.append(
                {
                    "case_id": case["order_id"],
                    "case_hash": content_hash(case),
                    "success": success,
                    "policy_violations": 0,
                    "truncated": None,
                    "reward": trajectory.total_reward,
                    "generated_tokens": 0,
                    "tool_calls": calls,
                    "elapsed_seconds": 0.1,
                    "trace": [
                        {
                            "role": turn.role,
                            "content": turn.content,
                            "token_ids": (turn.metadata or {}).get("token_ids"),
                        }
                        for turn in trajectory.turns
                    ],
                }
            )
        return outcomes

    outcomes = {
        (split, success): asyncio.run(replay_outcomes(split, success))
        for split in ("validation", "test")
        for success in (False, True)
    }

    def report(seed, split, checkpoint, success):
        records = [
            {
                **row,
                "evaluation_seed": paired_evaluation_seed(
                    seed, row["case_id"], row["case_hash"]
                ),
            }
            for row in outcomes[split, success]
        ]
        return evaluation_report(
            environment="refund-policy-v2",
            base_model=manifest["model"],
            seed=seed,
            cases={case["order_id"]: case for case in cases[split]},
            settings={
                "model_revision": manifest["model_revision"],
                "temperature": 1.0,
                "max_new_tokens": 192,
                "max_turns": 4,
                "evaluation_randomness": {
                    "protocol": EVALUATION_RNG_PROTOCOL,
                    "scope": "local-global-rngs",
                },
            },
            checkpoint={"path": checkpoint},
            implementation=IDENTITY,
            split=split,
            outcomes=records,
        )

    for seed in manifest["seeds"]:
        directory = root / f"seed-{seed}"
        baseline = report(
            seed,
            "test",
            f"hf://{manifest['model']}@{manifest['model_revision']}",
            False,
        )
        _write(directory / "baseline.json", baseline)
        _write(
            directory / "updates.json",
            [
                {"global_step": 1, "optimizer_updates": 1},
                {"global_step": 2, "optimizer_updates": 1},
            ],
        )
        validations = []
        for step in (0, 1, 2):
            validation = report(
                seed, "validation", "initial" if step == 0 else str(step), step == 1
            )
            name = f"validation-{step}.json"
            _write(directory / name, validation)
            validations.append(
                {"step": step, "report": name, "sha256": content_hash(validation)}
            )
        selected = directory / "selected"
        selected.mkdir()
        # Publication hashing is audited without deserializing model weights.
        (selected / "training_state.pt").write_bytes(
            b"unit-test fixture, not learner evidence"
        )
        inventory = {
            "schema_version": 1,
            "files": {
                "training_state.pt": acceptance._digest(selected / "training_state.pt")
            },
        }
        _write(selected / ".stateset-checkpoint.json", inventory)
        selection = {
            "schema_version": 1,
            "criterion": "validation_success_rate",
            "tie_break": "earliest_step",
            "selected_step": 1,
            "selected_success_rate": 1.0,
            "committed_optimizer_updates": 2,
            "validation_reports": validations,
            "checkpoint_inventory_sha256": content_hash(inventory),
        }
        _write(directory / "selection.json", selection)
        candidate = report(seed, "test", str(selected), True)
        candidate["selected_checkpoint"].update(
            inventory_sha256=content_hash(inventory),
            selection_sha256=content_hash(selection),
        )
        _write(directory / "candidate.json", candidate)
        baselines.append(baseline)
        candidates.append(candidate)
    comparison = compare_runs(
        baselines, candidates, min_seeds=6, require_significance=True
    )
    comparison["representative_cases"] = True
    _write(root / "comparison.json", comparison)
    return root


def test_native_quality_recomputes_raw_seed_reports_and_checkpoint_inventory(study):
    report = acceptance.audit_native_quality(study, expected_implementation=IDENTITY)
    assert report["passed"] is True
    assert report["comparison"]["inference"]["positive_seeds"] == 6
    assert "seed-0/selected/training_state.pt" in report["artifacts"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("synthetic", "Synthetic"),
        ("source", "implementation"),
        ("cases", "Reviewed cases"),
        ("update", "counters disagree"),
        ("selection", "earliest best"),
        ("comparison", "recomputed"),
        ("checkpoint", "completed publication"),
    ],
)
def test_native_quality_rejects_missing_or_resigned_false_evidence(
    study, mutation, match
):
    if mutation in {"synthetic", "source", "cases"}:
        path = study / "manifest.json"
        data = _read(path)
        if mutation == "synthetic":
            data["synthetic"] = True
        elif mutation == "source":
            data["implementation"] = {**IDENTITY, "files": {"__init__.py": "b" * 64}}
        else:
            data["cases_sha256"] = "b" * 64
    elif mutation == "update":
        path = study / "seed-0/updates.json"
        data = _read(path)
        data[1]["global_step"] = 9
    elif mutation == "selection":
        path = study / "seed-0/selection.json"
        data = _read(path)
        data["selected_step"] = 2
    elif mutation == "comparison":
        path = study / "comparison.json"
        data = _read(path)
        data["mean_success_gain"] = 0.0
    else:
        (study / "seed-0/selected/training_state.pt").write_bytes(b"changed")
        with pytest.raises(ValueError, match=match):
            acceptance.audit_native_quality(study, expected_implementation=IDENTITY)
        return
    _write(path, data)
    with pytest.raises(ValueError, match=match):
        acceptance.audit_native_quality(study, expected_implementation=IDENTITY)


def test_native_quality_missing_artifacts_and_symlinks_fail_closed(study):
    path = study / "seed-0/candidate.json"
    path.rename(path.with_suffix(".backup"))
    with pytest.raises(ValueError, match="Missing regular artifact"):
        acceptance.audit_native_quality(study, expected_implementation=IDENTITY)
    path.symlink_to(path.with_suffix(".backup").name)
    with pytest.raises(ValueError, match="symlinks"):
        acceptance.audit_native_quality(study, expected_implementation=IDENTITY)


def test_native_quality_replays_ledger_instead_of_trusting_success_flag(study):
    path = study / "seed-0/candidate.json"
    candidate = _read(path)
    baseline = _read(study / "seed-0/baseline.json")
    candidate["outcomes"][0]["trace"] = baseline["outcomes"][0]["trace"]
    _write(path, candidate)
    with pytest.raises(ValueError, match="replayed ledger"):
        acceptance.audit_native_quality(study, expected_implementation=IDENTITY)


@pytest.fixture
def gpu_bundle(tmp_path):
    """Fabricated CPU tensor fixtures exercise the parser, not GPU execution."""
    torch = pytest.importorskip("torch")
    root = tmp_path / "gpu-bundle"
    root.mkdir()
    harness = dict.fromkeys(acceptance.NATIVE_GPU_HARNESS_FILES, "a" * 64)
    report = {
        "schema_version": 1,
        "kind": "stateset-native-checkpoint-hardware-contract",
        "status": "completed",
        "device": "cuda",
        "backend": "nccl",
        "world_size": 2,
        "implementation": IDENTITY,
        "harness": harness,
        "cases": [],
        "artifacts": {},
    }
    for api in ("runtime", "async"):
        for pending in (False, True):
            for precision in ("fp32", "fp16"):
                name = f"{api}-{pending}-{precision}"
                (root / name).mkdir()
                rank_reports = []
                for rank in (0, 1):
                    checkpoint = {
                        "policy": {"weight": torch.tensor([1.0])},
                        "reference": {"weight": torch.tensor([3.0])},
                        "optimizer": {"state": {0: {"step": torch.tensor(1)}}},
                        "scheduler": {"last_epoch": 1},
                        "scaler": (
                            {
                                "scale": 128.0,
                                "_growth_tracker": 1,
                                "growth_interval": 2000,
                            }
                            if precision == "fp16"
                            else None
                        ),
                        "gradients": [torch.tensor([0.1])] if pending else [None],
                        "normalization": {"policy_count": 2} if pending else None,
                        "counters": [1, 0, 0, 3 if pending else 1, float("-inf"), 0],
                        "baseline": [0.1, 2, 0.2],
                        "rng": {"cuda": [torch.tensor([rank], dtype=torch.uint8)]},
                    }
                    expected = copy.deepcopy(checkpoint)
                    expected["policy"]["weight"] += 1
                    expected["counters"][0] = 2
                    expected["counters"][3] = 4 if pending else 2
                    if precision == "fp16":
                        expected["scaler"]["_growth_tracker"] = 2
                    draw = (
                        rank + 0.1,
                        rank + 0.2,
                        torch.tensor([rank + 0.3]),
                        torch.tensor([rank + 0.4]),
                    )
                    tensor_name = f"{name}/rank-{rank}.pt"
                    torch.save(
                        {
                            "checkpoint": checkpoint,
                            "restored": copy.deepcopy(checkpoint),
                            "expected": expected,
                            "actual": copy.deepcopy(expected),
                            "expected_draw": draw,
                            "actual_draw": copy.deepcopy(draw),
                            "forward_probe": torch.tensor(
                                [0.1],
                                dtype=(
                                    torch.float16
                                    if precision == "fp16"
                                    else torch.float32
                                ),
                            ),
                        },
                        root / tensor_name,
                    )
                    peer_name = f"{name}/rank-{rank}.json"
                    _write(
                        root / peer_name,
                        {
                            "rank": rank,
                            "world_size": 2,
                            "device": f"cuda:{rank}",
                            "backend": "nccl",
                            "gpu": "synthetic parser fixture",
                            "cuda": "fixture",
                            "api": api,
                            "pending": pending,
                            "precision": precision,
                            "scenarios": [
                                {
                                    "mode": mode,
                                    "snapshot": tensor_name,
                                    "failure": (
                                        {
                                            "load_error": (
                                                "ValueError: rng-restore"
                                                if mode == "rng"
                                                else "ValueError: rollout-sync"
                                            ),
                                            "train_error": "ValueError: preflight",
                                            "save_error": "ValueError: capture",
                                            "restore_failed": True,
                                            "consensus_pending": True,
                                        }
                                        if mode in {"false", "exception", "rng"}
                                        else None
                                    ),
                                }
                                for mode in acceptance.NATIVE_GPU_MODES
                            ],
                        },
                    )
                    rank_reports.append(peer_name)
                report["cases"].append(
                    {
                        "api": api,
                        "pending": pending,
                        "precision": precision,
                        "rank_reports": rank_reports,
                    }
                )
    report["artifacts"] = {
        path.relative_to(root).as_posix(): acceptance._digest(path)
        for path in root.rglob("*")
        if path.is_file()
    }
    _write(root / "contract.json", report)
    return root, harness


def test_gpu_validator_recomputes_retained_tensor_and_rng_equality(gpu_bundle):
    root, harness = gpu_bundle
    result = acceptance.audit_native_gpu_recovery(
        root, expected_implementation=IDENTITY, expected_harness=harness
    )
    assert result["cases"] == 8 and result["scenarios_per_rank"] == 56
    assert "contract.json" in result["artifacts"]


@pytest.mark.parametrize(
    "mutation,match",
    [
        ("missing_case", "case matrix"),
        ("wrong_source", "implementation"),
        ("changed_bytes", "artifact changed"),
        ("resigned_tensor", "tensor state differs"),
        ("resigned_fault", "quarantine"),
        ("missing_amp", "GradScaler"),
        ("fake_precision", "declared precision"),
    ],
)
def test_gpu_validator_rejects_incomplete_and_resigned_raw_evidence(
    gpu_bundle, mutation, match
):
    torch = pytest.importorskip("torch")
    from stateset_agents.core.checkpoint_io import load_checkpoint_file

    root, harness = gpu_bundle
    report = _read(root / "contract.json")
    if mutation == "missing_case":
        report["cases"].pop()
    elif mutation == "wrong_source":
        report["implementation"]["files"]["__init__.py"] = "b" * 64
    elif mutation in {
        "changed_bytes",
        "resigned_tensor",
        "missing_amp",
        "fake_precision",
    }:
        case = next(case for case in report["cases"] if case["precision"] == "fp16")
        peer = _read(root / case["rank_reports"][0])
        name = peer["scenarios"][0]["snapshot"]
        path = root / name
        snapshot = load_checkpoint_file(path)
        if mutation == "missing_amp":
            snapshot["checkpoint"]["scaler"] = None
            snapshot["restored"]["scaler"] = None
        elif mutation == "fake_precision":
            snapshot["forward_probe"] = snapshot["forward_probe"].float()
        else:
            snapshot["actual"]["policy"]["weight"] += 10
        torch.save(snapshot, path)
        if mutation != "changed_bytes":
            report["artifacts"][name] = acceptance._digest(path)
    else:
        name = report["cases"][0]["rank_reports"][0]
        peer = _read(root / name)
        next(row for row in peer["scenarios"] if row["mode"] == "false")["failure"][
            "save_error"
        ] = ""
        _write(root / name, peer)
        report["artifacts"][name] = acceptance._digest(root / name)
    _write(root / "contract.json", report)
    with pytest.raises(ValueError, match=match):
        acceptance.audit_native_gpu_recovery(
            root, expected_implementation=IDENTITY, expected_harness=harness
        )


@pytest.mark.parametrize(
    "target,value",
    [
        ("api", []),
        ("precision", {}),
        ("pending", 0),
        ("world_size", True),
        ("rank_reports", [[], "unused"]),
    ],
)
def test_gpu_malformed_matrix_is_a_controlled_rejection(gpu_bundle, target, value):
    root, harness = gpu_bundle
    report = _read(root / "contract.json")
    if target == "world_size":
        report[target] = value
    else:
        report["cases"][0][target] = value
    _write(root / "contract.json", report)
    with pytest.raises(ValueError):
        acceptance.audit_native_gpu_recovery(
            root, expected_implementation=IDENTITY, expected_harness=harness
        )


@pytest.mark.parametrize(
    "status,device,backend",
    [
        ("completed", "cpu", "gloo"),
        ("failed", "cuda", "nccl"),
        ("completed", "cuda", "gloo"),
    ],
)
def test_gpu_acceptance_rejects_cpu_skipped_failed_or_wrong_backend_before_torch(
    tmp_path, monkeypatch, status, device, backend
):
    _write(
        tmp_path / "contract.json",
        {
            "schema_version": 1,
            "kind": "stateset-native-checkpoint-hardware-contract",
            "status": status,
            "device": device,
            "backend": backend,
            "world_size": 2,
            "passed": True,
        },
    )
    import builtins

    original = builtins.__import__

    def no_tensor_import(name, *args, **kwargs):
        if name == "torch":
            raise AssertionError(
                "Rejected evidence must not import optional tensor stack"
            )
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_tensor_import)
    with pytest.raises(ValueError, match="two-GPU CUDA/NCCL"):
        acceptance.audit_native_gpu_recovery(
            tmp_path,
            expected_implementation=IDENTITY,
            expected_harness=dict.fromkeys(
                acceptance.NATIVE_GPU_HARNESS_FILES, "a" * 64
            ),
        )


def _producer():
    path = (
        Path(__file__).resolve().parents[2]
        / "benchmarks/native_gpu_checkpoint_contract.py"
    )
    spec = importlib.util.spec_from_file_location(
        "native_gpu_checkpoint_contract", path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "cuda,count,nccl,match",
    [
        (False, 0, True, "two CUDA GPUs"),
        (True, 1, True, "two CUDA GPUs"),
        (True, 2, False, "NCCL"),
    ],
)
def test_gpu_producer_never_silently_falls_back_to_cpu(cuda, count, nccl, match):
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: cuda, device_count=lambda: count),
        distributed=SimpleNamespace(
            is_available=lambda: True, is_nccl_available=lambda: nccl
        ),
    )
    with pytest.raises(RuntimeError, match=match):
        _producer().validate_hardware(torch, "cuda")


def test_gpu_producer_preserves_existing_evidence_on_failure(tmp_path, monkeypatch):
    module = _producer()
    directory = tmp_path / "evidence"
    directory.mkdir()
    sentinel = directory / "unrelated.json"
    sentinel.write_text("retained")
    assert module.main(["--output", str(directory)]) == 1
    assert sentinel.read_text() == "retained"
    assert not (directory / "contract.json").exists()


@pytest.mark.parametrize("selected_step", [0, 1])
def test_native_quality_rejects_selection_before_any_committed_training(
    study, selected_step
):
    selection_path = study / "seed-0/selection.json"
    selection = _read(selection_path)
    selection["selected_step"] = selected_step
    _write(selection_path, selection)
    if selected_step == 1:
        _write(
            study / "seed-0/updates.json",
            [
                {"global_step": 0, "optimizer_updates": 0},
                {"global_step": 2, "optimizer_updates": 2},
            ],
        )
    with pytest.raises(
        ValueError, match="Selected checkpoint must follow a committed training update"
    ):
        acceptance.audit_native_quality(study, expected_implementation=IDENTITY)
