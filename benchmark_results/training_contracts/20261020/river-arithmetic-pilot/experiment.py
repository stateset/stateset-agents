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
ROOT = REPO / "outputs/river-arithmetic-pilot-20261020"
PREPARED = REPO / "outputs/river-boundary-stability-20261019"
MODEL = "Qwen/Qwen3.5-9B"
SEED = 2026106001
ARMS = ("control", "arithmetic")
TOKEN_LIMIT = 2 * (158 + 308) * 1024


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
    for arm in ("control", "arithmetic"):
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
                assert (
                    read(ROOT / arm / "river_checkpoint.json")["steps"]
                    == plan["exposure"][arm]["updates"]
                )
            helper = load("frozen_evaluation", ROOT / "runtime/evaluation.py")
            helper.ROOT, helper.MODEL, helper.SEED = ROOT, MODEL, SEED
            checkpoints = [
                (
                    arm,
                    river.Checkpoint(
                        path=read(ROOT / arm / "river_checkpoint.json")["checkpoint"],
                        step=plan["exposure"][arm]["updates"],
                        checkpoint_type="inference",
                    ),
                )
                for arm in ARMS
            ]
            with client.session(
                experiment="stateset-arithmetic-pilot", role="paired-evaluation"
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
                                helper.evaluate(
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
                    decision = gates.assess(
                        reports["control"],
                        reports["arithmetic"],
                        cases,
                        split=split,
                        require_arithmetic_gain=split == "test",
                    )
                    write(
                        ROOT / f"{split}-decision.json",
                        {"plan_hash": plan["plan_hash"], **decision},
                    )
                    print(
                        json.dumps(
                            {
                                "split": split,
                                "passed": decision["passed"],
                                "arithmetic_case_gain": decision[
                                    "arithmetic_case_gain"
                                ],
                                "counts": {a: r["passed"] for a, r in reports.items()},
                            }
                        ),
                        flush=True,
                    )
                    if split == "validation" and not decision["passed"]:
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
                    "pilot_accepted": decision["passed"],
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
        asyncio.run(load("preparation", ROOT / "preparation.py").prepare())
    else:
        {"validate": validate, "launch": launch, "worker": worker}[sys.argv[1]]()
