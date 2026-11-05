"""One frozen SFT warm-start plus native RL pilot with three fixed test arms."""

import argparse
import asyncio
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
ROOT = REPO / "outputs/river-boundary-warmup-20261019"
SOURCE = REPO / "outputs/river-six-seed-recovery-20261019/seed-2026101901"
CURRICULUM = (
    REPO / "outputs/river-curriculum-review-20261019/seed-2026101901/curriculum"
)
MODEL = "Qwen/Qwen3.5-9B"
SEED = 2026101908


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    from stateset_agents.remote.river_rl import atomic_json

    atomic_json(Path(path), value)


def sft_spec():
    from stateset_agents.remote.job import RemoteJobSpec

    return RemoteJobSpec(
        dataset=ROOT / "data/train.jsonl",
        base_model=MODEL,
        output_dir=ROOT / "warmup",
        num_epochs=1,
        lora_r=16,
        learning_rate=2e-5,
        max_length=4096,
        per_device_batch_size=16,
        gradient_accumulation_steps=1,
        seed=SEED,
        timeout_s=2400,
        harvest={
            "checkpoint": read(ROOT / "original/river_checkpoint.json")["checkpoint"]
        },
        parent_adapter=str(ROOT / "original"),
        job_kind="sft",
    )


def rl_args():
    return argparse.Namespace(
        output=ROOT / "rl",
        base_model=MODEL,
        benchmark="refund-policy-v2",
        seed=SEED,
        steps=4,
        concurrency=16,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=str(ROOT / "warmup"),
        evaluate_only=False,
        collect_only=False,
        dry_run=False,
        zero_update_patience=3,
        rollout_token_budget=622592,
    )


def evaluation_module():
    path = ROOT / "runtime/evaluation.py"
    spec = importlib.util.spec_from_file_location("fixed_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT, module.MODEL, module.SEED = ROOT, MODEL, SEED
    return module


def prepare():
    from transformers import AutoTokenizer
    from stateset_agents.core.environments.refund_policy_environment import (
        RefundPolicyEnvironment,
        refund_policy_benchmark,
    )
    from stateset_agents.data.refund_curriculum import audit_refund_curriculum
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.evaluation.implementation import package_implementation
    from stateset_agents.remote.river_batches import build_sft_batch
    from stateset_agents.training.sft import parse_chat_dataset

    assert not (ROOT / "plan.json").exists(), "Experiment already prepared"
    curriculum_audit = asyncio.run(audit_refund_curriculum(SOURCE / "data", CURRICULUM))
    data = ROOT / "data"
    data.mkdir()
    for name in (
        "train.json",
        "train.jsonl",
        "lineage.json",
        "contrast_groups.json",
        "data_manifest.json",
    ):
        shutil.copyfile(CURRICULUM / name, data / name)
    seen_ids, seen_facts, sources = set(), set(), {}
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    for name in (
        "river-six-seed-20261019",
        "river-six-seed-recovery-20261019",
        "river-curriculum-review-20261019",
        "river-boundary-rl-20261019",
        "river-sft-warmup-20261018",
        "river-prompt-fix-20261018-live",
        "river-integration-review-20261018-live",
        "industry-datasets-20261019",
    ):
        directory = REPO / "outputs" / name
        for basename in (
            "train.json",
            "validation.json",
            "test.json",
            "primary-test.json",
        ):
            for path in sorted(directory.rglob(basename)):
                rows = read(path)
                cases = (
                    [row for row in rows if isinstance(row, dict) and "order_id" in row]
                    if isinstance(rows, list)
                    else []
                )
                if not cases:
                    continue
                for row in cases:
                    RefundPolicyEnvironment.validate_scenario(row)
                    seen_ids.add(row["order_id"])
                    seen_facts.add(fingerprint(row))
                sources[str(path.relative_to(REPO))] = sha(path)
    historical_ids, historical_facts = len(seen_ids), len(seen_facts)
    splits = {"train": read(data / "train.json")}
    for split, count in (("validation", 64), ("test", 128)):
        rows = refund_policy_benchmark(split, count, SEED)
        for row in rows:
            assert row["order_id"] not in seen_ids
            facts = fingerprint(row)
            assert facts not in seen_facts
            seen_ids.add(row["order_id"])
            seen_facts.add(facts)
        write(data / f"{split}.json", rows)
        splits[split] = rows
    rows = parse_chat_dataset((data / "train.jsonl").read_bytes())
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    batches = build_sft_batch(rows, tokenizer, max_length=4096)
    assert len(rows) == len(batches) == len(splits["train"]) == 255
    lengths = [
        len(
            tokenizer.encode(
                tokenizer.apply_chat_template(
                    row["messages"], tokenize=False, add_generation_prompt=False
                ),
                add_special_tokens=False,
            )
        )
        for row in rows
    ]
    assert max(lengths) <= 4096
    for batch, length in zip(batches, lengths, strict=True):
        assert len(batch["input_ids"]) == length - 1
        trained = tokenizer.decode(
            [
                token
                for token, weight in zip(
                    batch["target_tokens"], batch["weights"], strict=True
                )
                if weight
            ]
        )
        assert all(tool in trained for tool in ("lookup_order", "finish"))
        assert "Please resolve my refund request." not in trained
    write(
        ROOT / "data-check.json",
        {
            "curriculum_replay": curriculum_audit,
            "training_examples": len(rows),
            "admitted_sft_rows": len(batches),
            "dropped_or_truncated_rows": 0,
            "maximum_sequence_tokens": max(lengths),
            "sft_input_tokens_once": sum(len(batch["input_ids"]) for batch in batches),
            "sft_assistant_target_tokens_once": sum(
                sum(batch["weights"]) for batch in batches
            ),
            "tokenizer": MODEL,
            "local_files_only": True,
        },
    )
    write(
        ROOT / "exclusion-audit.json",
        {
            "scope": "Fresh validation/test versus historical identities and policy facts; existing training curriculum deliberately reused",
            "historical_ids": historical_ids,
            "historical_facts": historical_facts,
            "fresh_validation_cases": 64,
            "fresh_test_cases": 128,
            "overlap": 0,
            "sources_sha256": sources,
        },
    )
    shutil.copytree(SOURCE / "sft", ROOT / "original")
    runtime = ROOT / "runtime"
    runtime.mkdir()
    shutil.copytree(
        REPO / "stateset_agents",
        runtime / "stateset_agents",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    shutil.copyfile(__file__, runtime / "experiment.py")
    shutil.copyfile(
        REPO / "outputs/river-six-seed-recovery-20261019/runtime/warmup_runner.py",
        runtime / "evaluation.py",
    )
    plan = {
        "schema_version": 1,
        "kind": "exploratory-boundary-sft-warmstart-plus-native-rl",
        "seed": SEED,
        "base_model": MODEL,
        "implementation": package_implementation(),
        "original_choice": "First study seed fixed independently of held-out scores",
        "original_pointer_sha256": sha(ROOT / "original/river_checkpoint.json"),
        "sft_spec": sft_spec().to_dict(),
        "expected_sft_updates": 16,
        "rl_settings": vars(rl_args()) | {"output": str(rl_args().output)},
        "counts": {name: len(cases) for name, cases in splits.items()},
        "split_hashes": {name: content_hash(cases) for name, cases in splits.items()},
        "input_artifacts_sha256": {
            str(path.relative_to(ROOT)): sha(path) for path in sorted(data.iterdir())
        },
        "data_check_sha256": sha(ROOT / "data-check.json"),
        "exclusion_audit_sha256": sha(ROOT / "exclusion-audit.json"),
        "runner_sha256": sha(runtime / "experiment.py"),
        "evaluation_helper_sha256": sha(runtime / "evaluation.py"),
        "limits": {
            "sft_attempts": 1,
            "sft_epochs": 1,
            "sft_batches": 16,
            "rl_batches": 4,
            "rl_reserved_output_tokens": 622592,
            "external_test_reserved_output_tokens": 262144,
            "total_reserved_output_tokens": 884736,
            "wall_seconds": 2400,
            "request_timeout_seconds": 180,
            "transport_retries": False,
            "operational_retries": 0,
            "verified_dollar_cost_cap": None,
        },
        "protocol": [
            "Warm-start original first-seed SFT weights; one epoch of verified train-only boundary contrasts",
            "No prompt, reward, policy, teacher or hyperparameter changes after preparation",
            "Native RL starts from the saved boundary SFT checkpoint and selects using validation alone",
            "Test selected RL checkpoint once, then fixed original and boundary SFT checkpoints once on paired cases",
            "Original versus boundary SFT also changes training exposure; this is not a controlled attribution to curriculum alone",
            "No test-driven deployment/promotion or operational retry; preserve failed/uncertain attempts",
            "Single-seed synthetic pilot, not A+ or independently reviewed production-quality evidence",
        ],
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

    plan = read(ROOT / "plan.json")
    assert (
        content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
        == plan["plan_hash"]
    )
    assert package_implementation() == plan["implementation"]
    assert sha(__file__) == plan["runner_sha256"]
    assert sha(ROOT / "runtime/evaluation.py") == plan["evaluation_helper_sha256"]
    assert (
        sha(ROOT / "original/river_checkpoint.json") == plan["original_pointer_sha256"]
    )
    assert sft_spec().to_dict() == plan["sft_spec"]
    for name, digest in plan["input_artifacts_sha256"].items():
        assert sha(ROOT / name) == digest
    assert sha(ROOT / "data-check.json") == plan["data_check_sha256"]
    assert sha(ROOT / "exclusion-audit.json") == plan["exclusion_audit_sha256"]
    return plan


def worker():
    import river_client as river
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.remote.river import RiverExecutor
    from stateset_agents.remote.river_renderers import get_river_renderer
    from stateset_agents.remote.river_rl import exclusive_run
    from stateset_agents.remote.river_runtime import (
        require_model_access,
        require_native_runtime,
    )
    from stateset_agents.remote.rollout_budget import RolloutAdmissionBudget
    from stateset_agents.training.river_refund import prepare_run, campaign

    plan = validate()
    require_native_runtime()
    with exclusive_run(ROOT):
        budget = RolloutAdmissionBudget(
            ROOT / "external-budget.json",
            limit=262144,
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
            executor = RiverExecutor(
                client=client,
                tokenizer=renderer.tokenizer,
                ledger_path=ROOT / "river-ledger.jsonl",
            )
            executor.MAX_TRANSIENT_ATTEMPTS = 1
            handle = executor.submit(sft_spec())
            result = executor.wait(handle)
            write(
                ROOT / "sft-result.json",
                {
                    "status": result.status.value,
                    "logs": result.logs,
                    "duration_s": result.duration_s,
                    "cost_usd": result.cost_usd,
                },
            )
            assert result.succeeded
            warmup = read(ROOT / "warmup/river_checkpoint.json")
            original = read(ROOT / "original/river_checkpoint.json")
            assert warmup["steps"] == 16
            assert warmup["sft_options"]["checkpoint"] == original["checkpoint"]
            args = rl_args()
            splits = {
                name: read(ROOT / "data" / f"{name}.json")
                for name in ("train", "validation", "test")
            }
            with exclusive_run(args.output):
                prepare_run(args, splits)
                write(
                    ROOT / "stage-binding.json",
                    {
                        "plan_hash": plan["plan_hash"],
                        "warmup_pointer_sha256": sha(
                            ROOT / "warmup/river_checkpoint.json"
                        ),
                        "run_manifest_hash": content_hash(
                            read(args.output / "run_manifest.json")
                        ),
                    },
                )
                with client.session(
                    experiment="stateset-boundary-warmup", role="rl-train"
                ) as session:
                    with client.session(
                        experiment="stateset-boundary-warmup", role="rl-eval"
                    ) as evaluation:
                        model = session.create_model(
                            base_model=MODEL,
                            tokenizer=renderer.tokenizer,
                            lora=river.LoraConfig(
                                rank=16, seed=SEED, train_unembed=False
                            ),
                            checkpoint=river.Checkpoint(
                                path=warmup["checkpoint"],
                                step=0,
                                checkpoint_type="inference",
                            ),
                        )
                        asyncio.run(campaign(model, evaluation, renderer, args, splits))
            helpers = evaluation_module()
            checkpoints = [
                (
                    "original",
                    river.Checkpoint(
                        path=original["checkpoint"],
                        step=16,
                        checkpoint_type="inference",
                    ),
                ),
                (
                    "warmup",
                    river.Checkpoint(
                        path=warmup["checkpoint"], step=16, checkpoint_type="inference"
                    ),
                ),
            ]
            with client.session(
                experiment="stateset-boundary-warmup", role="fixed-eval"
            ) as evaluation:
                for arm, checkpoint in checkpoints:
                    with (ROOT / f"{arm}-test-attempt.json").open("x") as stream:
                        json.dump(
                            {
                                "plan_hash": plan["plan_hash"],
                                "checkpoint": asdict(checkpoint),
                                "test_hash": plan["split_hashes"]["test"],
                            },
                            stream,
                        )
                        stream.flush()
                        os.fsync(stream.fileno())

                async def evaluate_fixed():
                    await asyncio.gather(
                        *(
                            helpers.evaluate(
                                evaluation,
                                renderer,
                                checkpoint,
                                splits["test"],
                                "test",
                                arm,
                                budget,
                                plan["implementation"],
                            )
                            for arm, checkpoint in checkpoints
                        )
                    )

                asyncio.run(evaluate_fixed())
    print(json.dumps({"completed": True, "plan_hash": plan["plan_hash"]}), flush=True)


def launch():
    from stateset_agents.remote.river_runtime import river_preflight

    plan = validate()
    key = (
        os.environ.get("RIVER_API_KEY")
        or getpass.getpass("River credential (hidden): ").strip()
    )
    assert key
    os.environ["RIVER_API_KEY"] = key
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
    {"prepare": prepare, "validate": validate, "worker": worker, "launch": launch}[
        sys.argv[1]
    ]()
