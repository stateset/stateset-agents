"""Frozen, single-attempt boundary-curriculum River RL experiment."""

import argparse
import asyncio
from contextlib import closing
import getpass
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

REPO = Path("/home/dom/stateset-agents")
ROOT = REPO / "outputs/river-boundary-rl-20261019"
MODEL = "Qwen/Qwen3.5-9B"
SEED = 2026101907
CURRICULUM = (
    REPO / "outputs/river-curriculum-review-20261019/seed-2026101901/curriculum"
)
SOURCE = REPO / "outputs/river-six-seed-recovery-20261019/seed-2026101901"


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    from stateset_agents.remote.river_rl import atomic_json

    atomic_json(Path(path), value)


def args_for(arm):
    return argparse.Namespace(
        output=ROOT / arm,
        base_model=MODEL,
        benchmark="refund-policy-v2",
        seed=SEED,
        steps=4,
        concurrency=16,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=str(ROOT / "sft"),
        evaluate_only=arm == "baseline",
        collect_only=False,
        dry_run=False,
        zero_update_patience=3,
        rollout_token_budget=65536 if arm == "baseline" else 393216,
    )


def prepare():
    from stateset_agents.core.environments.refund_policy_environment import (
        RefundPolicyEnvironment,
        refund_policy_benchmark,
    )
    from stateset_agents.data.refund_curriculum import audit_refund_curriculum
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.evaluation.implementation import package_implementation
    from stateset_agents.remote.river_rl import exclusive_run
    from stateset_agents.training.river_refund import prepare_run

    assert not (ROOT / "plan.json").exists(), "Experiment already prepared"
    audit = asyncio.run(audit_refund_curriculum(SOURCE / "data", CURRICULUM))
    exclusions = []
    for directory in (
        REPO / "outputs/river-six-seed-20261019",
        REPO / "outputs/river-six-seed-recovery-20261019",
        REPO / "outputs/river-curriculum-review-20261019",
        REPO / "outputs/river-sft-warmup-20261018",
        REPO / "outputs/river-prompt-fix-20261018-live",
        REPO / "outputs/river-integration-review-20261018-live",
        REPO / "outputs/industry-datasets-20261019",
    ):
        for name in ("train.json", "validation.json", "test.json", "primary-test.json"):
            exclusions.extend(directory.rglob(name))
    seen_ids, seen_facts, sources = set(), set(), {}
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    for path in sorted(set(exclusions)):
        rows = read(path)
        if not isinstance(rows, list):
            continue
        cases = [row for row in rows if isinstance(row, dict) and "order_id" in row]
        if not cases:
            continue
        for row in cases:
            RefundPolicyEnvironment.validate_scenario(row)
            seen_ids.add(row["order_id"])
            seen_facts.add(fingerprint(row))
        sources[str(path.relative_to(REPO))] = sha(path)
    historical_ids, historical_facts = len(seen_ids), len(seen_facts)
    splits = {"train": read(CURRICULUM / "train.json")}
    for split, count in (("validation", 32), ("test", 64)):
        rows = refund_policy_benchmark(split, count, SEED)
        for row in rows:
            assert row["order_id"] not in seen_ids, "Historical identity collision"
            key = fingerprint(row)
            assert key not in seen_facts, "Historical fact collision"
            seen_ids.add(row["order_id"])
            seen_facts.add(key)
        splits[split] = rows
    shutil.copytree(SOURCE / "sft", ROOT / "sft")
    runtime = ROOT / "runtime"
    runtime.mkdir()
    shutil.copytree(
        REPO / "stateset_agents",
        runtime / "stateset_agents",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(__file__, runtime / "experiment.py")
    for arm in ("rl", "baseline"):
        arm_splits = {"test": splits["test"]} if arm == "baseline" else splits
        with exclusive_run(ROOT / arm):
            prepare_run(args_for(arm), arm_splits)
    write(
        ROOT / "exclusion-audit.json",
        {
            "passed": True,
            "historical_unique_ids": historical_ids,
            "historical_unique_facts": historical_facts,
            "source_sha256": sources,
            "fresh_validation_cases": 32,
            "fresh_test_cases": 64,
            "historical_and_cross_split_fact_overlap": 0,
            "curriculum_audit": audit,
        },
    )
    plan = {
        "schema_version": 1,
        "kind": "exploratory-boundary-curriculum-native-rl",
        "seed": SEED,
        "base_model": MODEL,
        "implementation": package_implementation(),
        "checkpoint_choice": "First study seed, fixed independently of held-out scores",
        "checkpoint_pointer_sha256": sha(ROOT / "sft/river_checkpoint.json"),
        "curriculum_source": str(CURRICULUM.relative_to(REPO)),
        "curriculum_manifest_sha256": sha(CURRICULUM / "data_manifest.json"),
        "counts": {name: len(rows) for name, rows in splits.items()},
        "split_hashes": {name: content_hash(rows) for name, rows in splits.items()},
        "limits": {
            "rl_batches": 4,
            "groups_per_batch": 4,
            "group_size": 8,
            "rl_reserved_output_tokens": 393216,
            "baseline_reserved_output_tokens": 65536,
            "total_reserved_output_tokens": 458752,
            "wall_seconds": 1800,
            "request_timeout_seconds": 180,
            "transport_retries": False,
            "operational_retries": 0,
            "zero_update_patience": 3,
            "verified_dollar_cost_cap": None,
        },
        "protocol": [
            "Additional RL only; no SFT training in this experiment",
            "Synthetic training-only adjacent boundary contrasts; unchanged environment rewards",
            "Validation chooses a checkpoint before any test sampling",
            "Test selected RL checkpoint once, then fixed SFT checkpoint once on paired cases",
            "Baseline cannot influence training or selection; do not retry uncertain attempts",
            "Count consumed batches separately from optimizer updates and selected-weight updates",
            "Single-seed exploratory results do not certify A+ or causal/generalized gains",
        ],
        "runner_sha256": sha(runtime / "experiment.py"),
        "exclusion_audit_sha256": sha(ROOT / "exclusion-audit.json"),
        "run_manifest_sha256": {
            arm: sha(ROOT / arm / "run_manifest.json") for arm in ("rl", "baseline")
        },
    }
    plan["plan_hash"] = content_hash(plan)
    write(ROOT / "plan.json", plan)
    print(
        json.dumps(
            {"prepared": True, "counts": plan["counts"], "limits": plan["limits"]}
        ),
        flush=True,
    )


def validate():
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.evaluation.implementation import package_implementation
    from stateset_agents.training.river_refund import _validated_run_inputs

    plan = read(ROOT / "plan.json")
    assert (
        content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
        == plan["plan_hash"]
    )
    assert package_implementation() == plan["implementation"]
    assert sha(__file__) == plan["runner_sha256"]
    assert sha(ROOT / "sft/river_checkpoint.json") == plan["checkpoint_pointer_sha256"]
    assert sha(ROOT / "exclusion-audit.json") == plan["exclusion_audit_sha256"]
    for arm in ("rl", "baseline"):
        assert sha(ROOT / arm / "run_manifest.json") == plan["run_manifest_sha256"][arm]
        names = ("test",) if arm == "baseline" else ("train", "validation", "test")
        splits = {name: read(ROOT / arm / f"{name}.json") for name in names}
        assert all(
            content_hash(rows) == plan["split_hashes"][name]
            for name, rows in splits.items()
        )
        _validated_run_inputs(args_for(arm), splits)
    return plan


def worker():
    import river_client as river
    from stateset_agents.remote.river_renderers import get_river_renderer
    from stateset_agents.remote.river_runtime import (
        require_model_access,
        require_native_runtime,
    )
    from stateset_agents.remote.river_rl import exclusive_run
    from stateset_agents.training.river_refund import campaign

    plan = validate()
    require_native_runtime()
    with closing(
        river.Client(
            api_key=os.environ["RIVER_API_KEY"], timeout=180, enable_retries=False
        )
    ) as client:
        require_model_access(client, MODEL)
        renderer = get_river_renderer(MODEL, thinking=False)
        checkpoint = read(ROOT / "sft/river_checkpoint.json")["checkpoint"]
        for arm in ("rl", "baseline"):
            names = ("test",) if arm == "baseline" else ("train", "validation", "test")
            splits = {name: read(ROOT / arm / f"{name}.json") for name in names}
            with exclusive_run(ROOT / arm):
                with client.session(
                    experiment="stateset-boundary-rl", role=f"{arm}-train"
                ) as session:
                    with client.session(
                        experiment="stateset-boundary-rl", role=f"{arm}-eval"
                    ) as evaluation:
                        model = session.create_model(
                            base_model=MODEL,
                            tokenizer=renderer.tokenizer,
                            lora=river.LoraConfig(
                                rank=16, seed=SEED, train_unembed=False
                            ),
                            checkpoint=river.Checkpoint(
                                path=checkpoint, step=0, checkpoint_type="inference"
                            ),
                        )
                        asyncio.run(
                            campaign(model, evaluation, renderer, args_for(arm), splits)
                        )
            print(
                json.dumps({"arm_completed": arm, "plan_hash": plan["plan_hash"]}),
                flush=True,
            )


def launch():
    from stateset_agents.remote.river_runtime import river_preflight

    plan = validate()
    credential = (
        os.environ.get("RIVER_API_KEY")
        or getpass.getpass("River credential (hidden): ").strip()
    )
    assert credential, "River credential is required"
    os.environ["RIVER_API_KEY"] = credential
    preflight = river_preflight([MODEL], live=True, timeout_seconds=20)
    write(ROOT / "preflight-live.json", preflight)
    if not preflight["passed"]:
        print(
            json.dumps({"preflight_passed": False, "issues": preflight["issues"]}),
            flush=True,
        )
        return
    with (ROOT / "execution-attempt.json").open("x") as stream:
        json.dump(
            {"plan_hash": plan["plan_hash"], "started_at_unix": time.time()}, stream
        )
        stream.flush()
        os.fsync(stream.fileno())
    environment = {
        **os.environ,
        "PYTHONPATH": str(ROOT / "runtime"),
        "HF_HUB_OFFLINE": "1",
        "HF_HUB_DISABLE_PROGRESS_BARS": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "WANDB_MODE": "disabled",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    started = time.monotonic()
    timed_out = False
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
            exit_code = process.wait(timeout=plan["limits"]["wall_seconds"])
        except subprocess.TimeoutExpired:
            timed_out = True
            os.killpg(process.pid, signal.SIGTERM)
            try:
                exit_code = process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                exit_code = process.wait()
    summary = {
        "exit_code": exit_code,
        "timed_out": timed_out,
        "elapsed_seconds": time.monotonic() - started,
        "plan_hash": plan["plan_hash"],
    }
    write(ROOT / "execution-summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    {"prepare": prepare, "worker": worker, "launch": launch, "validate": validate}[
        sys.argv[1]
    ]()
