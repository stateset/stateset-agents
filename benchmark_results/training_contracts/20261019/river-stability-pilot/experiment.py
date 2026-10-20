"""Bounded, one-parent River stability pilot with frozen validation safety gates."""

import asyncio
from collections import Counter
from contextlib import closing
from dataclasses import asdict
import getpass
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path("/home/dom/stateset-agents")
ROOT = REPO / "outputs/river-stability-pilot-20261019"
PREPARED = REPO / "outputs/river-boundary-stability-20261019"
MODEL = "Qwen/Qwen3.5-9B"
SEED = 2026104001
ARMS = ("original", "control", "parameter")
TOKEN_LIMIT = 3 * (104 + 200) * 1024


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    from stateset_agents.remote.river_rl import atomic_json

    atomic_json(Path(path), value)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def seal(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())


def spec_for(arm):
    from stateset_agents.remote.job import RemoteJobSpec

    return RemoteJobSpec(
        dataset=ROOT / "data" / f"{arm}.jsonl",
        base_model=MODEL,
        output_dir=ROOT / arm,
        num_epochs=1,
        lora_r=16,
        learning_rate=2e-5,
        max_length=4096,
        per_device_batch_size=16,
        gradient_accumulation_steps=1,
        seed=SEED,
        timeout_s=1200,
        job_kind="sft",
        parent_adapter=str(ROOT / "original"),
        harvest={
            "checkpoint": read(ROOT / "original/river_checkpoint.json")["checkpoint"]
        },
    )


async def prepare_async():
    from transformers import AutoTokenizer
    from stateset_agents.core.environments.refund_policy_environment import (
        RefundPolicyEnvironment,
    )
    from stateset_agents.data.refund_curriculum import (
        PARAMETER_POLICY,
        audit_refund_curriculum,
    )
    from stateset_agents.data.refund_demonstrations import replay_refund_candidate
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.evaluation.implementation import package_implementation
    from stateset_agents.remote.river_batches import build_sft_batch
    from stateset_agents.training.sft import parse_chat_dataset

    assert not (ROOT / "plan.json").exists(), "Never overwrite a frozen plan"
    ready_path = PREPARED / "attempt-3/readiness.json"
    ready = read(ready_path)
    assert ready["implementation"] == package_implementation()
    entry = ready["parents"][0]
    assert entry["parent_seed"] == 2026101901
    canonical = REPO / entry["canonical_data"]
    parameter = REPO / entry["curriculum_path"]
    legacy = (
        REPO / "outputs/river-curriculum-review-20261019/seed-2026101901/curriculum"
    )
    assert (
        await audit_refund_curriculum(canonical, parameter, policy=PARAMETER_POLICY)
    )["passed"]
    assert (await audit_refund_curriculum(canonical, legacy))["passed"]
    control = load("prospective_control", ROOT / "control.py")
    new_cases = read(parameter / "train.json")
    control_cases, control_rows, lineage = await control.repetition_control(
        read(legacy / "train.json"), dict(Counter(r["family"] for r in new_cases)), SEED
    )
    data = ROOT / "data"
    data.mkdir()
    shutil.copytree(canonical.parent / "sft", ROOT / "original")
    write(data / "control-cases.json", control_cases)
    write(data / "control-lineage.json", lineage)
    with (data / "control.jsonl").open("x") as stream:
        for row in control_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    shutil.copyfile(parameter / "train.json", data / "parameter-cases.json")
    shutil.copyfile(parameter / "train.jsonl", data / "parameter.jsonl")
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    history = read(PREPARED / "attempt-3/historical-sources.json")
    excluded_ids, excluded_facts = set(), set()
    for name, digest in history.items():
        assert sha(REPO / name) == digest
        rows = read(REPO / name)
        if isinstance(rows, list):
            for row in rows:
                if isinstance(row, dict) and "return_window_days" in row:
                    excluded_ids.add(row["order_id"])
                    excluded_facts.add(fingerprint(row))
    for parent in ready["parents"]:
        for case in read(REPO / parent["curriculum_path"] / "train.json"):
            excluded_ids.add(case["order_id"])
            excluded_facts.add(fingerprint(case))
    for case in control_cases:
        excluded_ids.add(case["order_id"])
        excluded_facts.add(fingerprint(case))
    for parent in ready["parents"][1:]:
        for split in ("validation", "test"):
            path = (
                PREPARED
                / "attempt-3"
                / f"parent-{parent['parent_seed']}"
                / f"{split}.json"
            )
            assert sha(path) == parent["holdouts"][split]["sha256"]
            for case in read(path):
                excluded_ids.add(case["order_id"])
                excluded_facts.add(fingerprint(case))
    for split, count in (("validation", 104), ("test", 200)):
        path = PREPARED / "attempt-3/parent-2026101901" / f"{split}.json"
        assert sha(path) == entry["holdouts"][split]["sha256"]
        cases = read(path)
        assert len(cases) == count
        for case in cases:
            assert (
                case["order_id"] not in excluded_ids
                and fingerprint(case) not in excluded_facts
            )
            excluded_ids.add(case["order_id"])
            excluded_facts.add(fingerprint(case))
        shutil.copyfile(path, data / f"{split}.json")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    exposure = {}
    for arm, cases in (("control", control_cases), ("parameter", new_cases)):
        rows = parse_chat_dataset((data / f"{arm}.jsonl").read_bytes())
        by_id = {c["order_id"]: c for c in cases}
        for row in rows:
            meta = row["metadata"]
            assert meta["split"] == "train"
            await replay_refund_candidate(
                by_id[meta["case_id"]],
                {
                    "case_id": meta["case_id"],
                    "case_hash": meta["case_hash"],
                    "messages": row["messages"],
                    "truncated": None,
                },
            )
        batches = build_sft_batch(rows, tokenizer, max_length=4096)
        assert len(batches) == len(rows) == 512
        lengths = []
        for row, batch in zip(rows, batches, strict=True):
            length = len(
                tokenizer.encode(
                    tokenizer.apply_chat_template(
                        row["messages"], tokenize=False, add_generation_prompt=False
                    ),
                    add_special_tokens=False,
                )
            )
            assert length <= 4096 and len(batch["input_ids"]) == length - 1
            targets = tokenizer.decode(
                [
                    t
                    for t, w in zip(
                        batch["target_tokens"], batch["weights"], strict=True
                    )
                    if w
                ]
            )
            assert "lookup_order" in targets and "finish" in targets
            assert "Please resolve my refund request." not in targets
            lengths.append(length)
        exposure[arm] = {
            "rows": len(rows),
            "updates": 32,
            "families": dict(Counter(c["family"] for c in cases)),
            "input_tokens": sum(len(b["input_ids"]) for b in batches),
            "assistant_target_tokens": sum(sum(b["weights"]) for b in batches),
            "maximum_sequence_tokens": max(lengths),
            "dropped_or_truncated_rows": 0,
        }
    assert exposure["control"]["families"] == exposure["parameter"]["families"]
    for field in ("input_tokens", "assistant_target_tokens"):
        assert (
            abs(exposure["control"][field] / exposure["parameter"][field] - 1) <= 0.005
        )
    priors = {}
    for path in sorted(
        (REPO / "benchmark_results/training_contracts").rglob("artifacts.sha256.json")
    ):
        for name, digest in read(path).items():
            assert sha(path.parent / name) == digest
            priors[str((path.parent / name).relative_to(REPO))] = digest
    write(ROOT / "prior-evidence.json", priors)
    runtime = ROOT / "runtime"
    runtime.mkdir()
    implementation = package_implementation()
    for name, digest in implementation["files"].items():
        source = REPO / "stateset_agents" / name
        assert sha(source) == digest
        destination = runtime / "stateset_agents" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    for name in ("experiment.py", "control.py", "gates.py"):
        shutil.copyfile(ROOT / name, runtime / name)
    shutil.copyfile(
        REPO / "outputs/river-boundary-six-seed-20261019/runtime/evaluation.py",
        runtime / "evaluation.py",
    )
    plan = {
        "schema_version": 1,
        "kind": "single-parent-parameter-stability-sft-pilot",
        "base_model": MODEL,
        "training_seed": SEED,
        "evaluation_case_seed": entry["seed"],
        "parent_seed": entry["parent_seed"],
        "parent_selection": "First retained original SFT parent by ascending seed, fixed before new evaluation.",
        "initial_checkpoint_sha256": sha(ROOT / "original/river_checkpoint.json"),
        "implementation": implementation,
        "training_order": ["control", "parameter"],
        "sft_specs": {arm: spec_for(arm).to_dict() for arm in ("control", "parameter")},
        "exposure": exposure,
        "input_and_target_relative_tolerance": 0.005,
        "matching_scope": "Same family counts, rows, updates, rank, lr, batch, seed and parent; near-matched tokens, not exact provider compute or dollars.",
        "selection": "Fixed final one-epoch checkpoints. Validation controls admission to test; no checkpoint tuning or candidate selection.",
        "validation_gate": "Parameter versus both control and original: no success/policy/truncation regression in every planned marginal stratum; gain threshold zero.",
        "test_gate": "Primary parameter versus control gain >= .03 with the same nonregression strata. Parameter versus original safety is also required.",
        "strata": [
            "all",
            "family",
            "ordinary/stress",
            "stress_source_family",
            "stress_window",
        ],
        "evaluation": {
            "arms": list(ARMS),
            "validation_per_arm": 104,
            "test_per_arm": 200,
            "max_validation_trajectories": 312,
            "max_test_trajectories": 600,
            "test_rule": "If either validation comparison fails, stop without test. Otherwise seal all three test attempts before any test sampling; no retry of uncertain attempts.",
            "concurrency_per_arm": 4,
            "max_total_concurrency": 12,
        },
        "limits": {
            "additional_sft_updates": 64,
            "training_attempts_per_arm": 1,
            "generated_output_reservation": TOKEN_LIMIT,
            "trajectory_output_tokens": 1024,
            "wall_seconds": 1800,
            "sdk_request_timeout_seconds": 180,
            "transport_retries": False,
            "operational_retries": 0,
            "verified_dollar_cost_cap": None,
        },
        "input_sha256": {
            str(p.relative_to(ROOT)): sha(p) for p in sorted(data.iterdir())
        },
        "runner_sha256": {
            name: sha(runtime / name)
            for name in ("experiment.py", "control.py", "gates.py", "evaluation.py")
        },
        "source_readiness_sha256": sha(ready_path),
        "prior_evidence_sha256": sha(ROOT / "prior-evidence.json"),
        "quality_certification": False,
        "inference_unit": "One retained parent, exploratory conditional pilot; no population significance or production readiness.",
    }
    plan["plan_hash"] = content_hash(plan)
    write(ROOT / "plan.json", plan)
    print(
        json.dumps({"prepared": True, "exposure": exposure, "limits": plan["limits"]}),
        flush=True,
    )


def validate():
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.evaluation.implementation import package_implementation

    plan = read(ROOT / "plan.json")
    assert (
        content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
        == plan["plan_hash"]
    )
    assert package_implementation() == plan["implementation"]
    assert (
        sha(ROOT / "original/river_checkpoint.json")
        == plan["initial_checkpoint_sha256"]
    )
    assert sha(ROOT / "prior-evidence.json") == plan["prior_evidence_sha256"]
    for name, digest in plan["input_sha256"].items():
        assert sha(ROOT / name) == digest
    for name, digest in plan["runner_sha256"].items():
        assert sha(ROOT / "runtime" / name) == digest
    for name, digest in plan["implementation"]["files"].items():
        assert sha(ROOT / "runtime/stateset_agents" / name) == digest
    for arm in ("control", "parameter"):
        assert spec_for(arm).to_dict() == plan["sft_specs"][arm]
    return plan


def worker():
    import river_client as river
    from stateset_agents.remote.river import RiverExecutor
    from stateset_agents.remote.river_renderers import get_river_renderer
    from stateset_agents.remote.river_rl import exclusive_run
    from stateset_agents.remote.river_runtime import (
        require_model_access,
        require_native_runtime,
    )
    from stateset_agents.remote.rollout_budget import RolloutAdmissionBudget

    plan = validate()
    require_native_runtime()
    gates = load("frozen_gates", ROOT / "runtime/gates.py")
    with exclusive_run(ROOT / "worker"):
        seal(ROOT / "worker-attempt.json", {"plan_hash": plan["plan_hash"]})
        budget = RolloutAdmissionBudget(
            ROOT / "rollout-budget.json",
            limit=TOKEN_LIMIT,
            trajectory_tokens=1024,
            fingerprint=plan["plan_hash"],
            create=True,
        )
        with closing(
            river.Client(
                api_key=os.environ["RIVER_API_KEY"], timeout=180, enable_retries=False
            )
        ) as client:
            require_model_access(client, MODEL)
            renderer = get_river_renderer(MODEL, thinking=False)
            for arm in plan["training_order"]:
                seal(
                    ROOT / f"{arm}-training-attempt.json",
                    {"plan_hash": plan["plan_hash"], "spec": plan["sft_specs"][arm]},
                )
                executor = RiverExecutor(
                    client=client,
                    tokenizer=renderer.tokenizer,
                    ledger_path=ROOT / "river-ledger.jsonl",
                )
                executor.MAX_TRANSIENT_ATTEMPTS = 1
                result = executor.wait(executor.submit(spec_for(arm)))
                write(
                    ROOT / f"{arm}-training-result.json",
                    {
                        "status": result.status.value,
                        "logs": result.logs,
                        "duration_s": result.duration_s,
                        "cost_usd": result.cost_usd,
                    },
                )
                assert result.succeeded
                assert read(ROOT / arm / "river_checkpoint.json")["steps"] == 32
            helpers = load("frozen_evaluator", ROOT / "runtime/evaluation.py")
            helpers.ROOT, helpers.MODEL, helpers.SEED = ROOT, MODEL, SEED
            checkpoints = [
                (
                    arm,
                    river.Checkpoint(
                        path=read(ROOT / arm / "river_checkpoint.json")["checkpoint"],
                        step=read(ROOT / arm / "river_checkpoint.json")["steps"],
                        checkpoint_type="inference",
                    ),
                )
                for arm in ARMS
            ]
            with client.session(
                experiment="stateset-stability-pilot", role="paired-evaluation"
            ) as session:
                for split in ("validation", "test"):
                    cases = read(ROOT / "data" / f"{split}.json")
                    for arm, checkpoint in checkpoints:
                        seal(
                            ROOT / f"{arm}-{split}-attempt.json",
                            {
                                "plan_hash": plan["plan_hash"],
                                "checkpoint": asdict(checkpoint),
                                "split_sha256": plan["input_sha256"][
                                    f"data/{split}.json"
                                ],
                            },
                        )

                    async def evaluate():
                        await asyncio.gather(
                            *(
                                helpers.evaluate(
                                    session,
                                    renderer,
                                    checkpoint,
                                    cases,
                                    split,
                                    arm,
                                    budget,
                                    plan["implementation"],
                                )
                                for arm, checkpoint in checkpoints
                            )
                        )

                    asyncio.run(evaluate())
                    reports = {arm: read(ROOT / f"{arm}-{split}.json") for arm in ARMS}
                    comparisons = {
                        "parameter_vs_control": gates.assess(
                            reports["control"],
                            reports["parameter"],
                            cases,
                            split=split,
                            min_gain=0.03 if split == "test" else 0,
                        ),
                        "parameter_vs_original": gates.assess(
                            reports["original"],
                            reports["parameter"],
                            cases,
                            split=split,
                            min_gain=0,
                        ),
                    }
                    passed = all(v["passed"] for v in comparisons.values())
                    write(
                        ROOT / f"{split}-decision.json",
                        {
                            "passed": passed,
                            "comparisons": comparisons,
                            "plan_hash": plan["plan_hash"],
                            "checkpoint_selection": "Fixed final epoch; no tuning",
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "split": split,
                                "passed": passed,
                                "counts": {a: r["passed"] for a, r in reports.items()},
                            }
                        ),
                        flush=True,
                    )
                    if split == "validation" and not passed:
                        write(
                            ROOT / "pilot-result.json",
                            {
                                "status": "validation_blocked_test_not_attempted",
                                "plan_hash": plan["plan_hash"],
                                "test_attempted": False,
                                "pilot_accepted": False,
                            },
                        )
                        return
            write(
                ROOT / "pilot-result.json",
                {
                    "status": "completed",
                    "plan_hash": plan["plan_hash"],
                    "test_attempted": True,
                    "pilot_accepted": passed,
                    "quality_certification": False,
                },
            )


def launch():
    from stateset_agents.remote.river_runtime import river_preflight

    plan = validate()
    assert not (ROOT / "execution-attempt.json").exists(), "Never retry a sealed pilot"
    credential = (
        os.environ.get("RIVER_API_KEY")
        or getpass.getpass("River credential (hidden): ").strip()
    )
    assert credential
    os.environ["RIVER_API_KEY"] = credential
    seal(
        ROOT / "execution-attempt.json",
        {"plan_hash": plan["plan_hash"], "started_at_unix": time.time()},
    )
    preflight = river_preflight([MODEL], live=True, timeout_seconds=20)
    write(ROOT / "preflight-live.json", preflight)
    assert preflight["passed"]
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "runtime"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "STATESET_RIVER_VERBOSE": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    started, timed_out = time.monotonic(), False
    with (ROOT / "live-run.txt").open("x") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", str(ROOT / "runtime/experiment.py"), "worker"],
            cwd=REPO,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            code = process.wait(timeout=plan["limits"]["wall_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                code = process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                code = process.wait(timeout=10)
    result = {
        "exit_code": code,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "plan_hash": plan["plan_hash"],
        "pilot_result": (
            read(ROOT / "pilot-result.json")
            if (ROOT / "pilot-result.json").exists()
            else None
        ),
    }
    write(ROOT / "execution-summary.json", result)
    print(json.dumps(result), flush=True)
    if code:
        raise SystemExit(code)


if __name__ == "__main__":
    if sys.argv[1] == "prepare":
        asyncio.run(prepare_async())
    else:
        {"validate": validate, "launch": launch, "worker": worker}[sys.argv[1]]()
