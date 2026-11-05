"""Frozen six-parent paired continuation study with an exposure-matched control."""

import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
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
ROOT = REPO / "outputs/river-boundary-six-seed-20261019"
PARENTS = REPO / "outputs/river-six-seed-recovery-20261019"
CURRICULA = REPO / "outputs/river-curriculum-review-20261019"
MODEL = "Qwen/Qwen3.5-9B"
SEEDS = tuple(range(2026101931, 2026101937))
ARMS = ("original", "repeat", "boundary")
TOKEN_LIMIT_PER_SEED = 491520


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    from stateset_agents.remote.river_rl import atomic_json

    atomic_json(Path(path), value)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def spec_for(seed, arm):
    from stateset_agents.remote.job import RemoteJobSpec

    directory = ROOT / f"seed-{seed}"
    return RemoteJobSpec(
        dataset=directory / "data" / f"{arm}.jsonl",
        base_model=MODEL,
        output_dir=directory / arm,
        num_epochs=1,
        lora_r=16,
        learning_rate=2e-5,
        max_length=4096,
        per_device_batch_size=16,
        gradient_accumulation_steps=1,
        seed=seed,
        timeout_s=1800,
        job_kind="sft",
        parent_adapter=str(directory / "original"),
        harvest={
            "checkpoint": read(directory / "original/river_checkpoint.json")[
                "checkpoint"
            ]
        },
    )


def write_jsonl(path, rows):
    with path.open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


async def prepare_async():
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

    assert not (ROOT / "plan.json").exists(), "Never overwrite a prepared study"
    control = load_module("control_preparation", ROOT / "control.py")
    tokenizer = AutoTokenizer.from_pretrained(MODEL, local_files_only=True)
    seen_ids, seen_facts, historical_sources = set(), set(), {}
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    for name in (
        "river-six-seed-20261019",
        "river-six-seed-recovery-20261019",
        "river-curriculum-review-20261019",
        "river-boundary-rl-20261019",
        "river-boundary-warmup-20261019",
        "river-sft-warmup-20261018",
        "river-prompt-fix-20261018-live",
        "river-integration-review-20261018-live",
        "industry-datasets-20261019",
    ):
        for basename in (
            "train.json",
            "validation.json",
            "test.json",
            "primary-test.json",
        ):
            for path in sorted((REPO / "outputs" / name).rglob(basename)):
                value = read(path)
                cases = (
                    [
                        row
                        for row in value
                        if isinstance(row, dict) and "order_id" in row
                    ]
                    if isinstance(value, list)
                    else []
                )
                if not cases:
                    continue
                for case in cases:
                    RefundPolicyEnvironment.validate_scenario(case)
                    seen_ids.add(case["order_id"])
                    seen_facts.add(fingerprint(case))
                historical_sources[str(path.relative_to(REPO))] = sha(path)
    historical_counts = {"ids": len(seen_ids), "facts": len(seen_facts)}
    entries = []
    all_training_facts = set()
    for index, seed in enumerate(SEEDS):
        parent_seed = 2026101901 + index
        source = PARENTS / f"seed-{parent_seed}"
        curriculum = CURRICULA / f"seed-{parent_seed}/curriculum"
        audit = await audit_refund_curriculum(source / "data", curriculum)
        directory = ROOT / f"seed-{seed}"
        data = directory / "data"
        data.mkdir(parents=True)
        shutil.copytree(source / "sft", directory / "original")
        original_cases = read(source / "data/train.json")
        boundary_cases = read(curriculum / "train.json")
        repeat_cases, repeat_rows, lineage = await control.repetition_control(
            original_cases, len(boundary_cases), parent_seed
        )
        assert len({row["order_id"] for row in repeat_cases}) == len(repeat_cases)
        assert (
            {row["order_id"] for row in repeat_cases}
            & {row["order_id"] for row in boundary_cases}
        ) == {row["order_id"] for row in original_cases}
        canonical_facts = {fingerprint(row) for row in original_cases}
        assert {fingerprint(row) for row in repeat_cases} == canonical_facts
        facts = canonical_facts | {fingerprint(row) for row in boundary_cases}
        assert not facts & all_training_facts, "Cross-parent training facts overlap"
        all_training_facts.update(facts)
        seen_facts.update(facts)
        seen_ids.update(row["order_id"] for row in repeat_cases + boundary_cases)
        write(data / "repeat-cases.json", repeat_cases)
        write(data / "repeat-lineage.json", lineage)
        write_jsonl(data / "repeat.jsonl", repeat_rows)
        shutil.copyfile(curriculum / "train.jsonl", data / "boundary.jsonl")
        shutil.copyfile(curriculum / "train.json", data / "boundary-cases.json")
        counts = {}
        for arm in ("repeat", "boundary"):
            rows = parse_chat_dataset((data / f"{arm}.jsonl").read_bytes())
            batches = build_sft_batch(rows, tokenizer, max_length=4096)
            assert len(batches) == len(rows) == len(boundary_cases)
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
            counts[arm] = {
                "rows": len(rows),
                "updates": (len(rows) + 15) // 16,
                "input_tokens": sum(len(batch["input_ids"]) for batch in batches),
                "assistant_target_tokens": sum(
                    sum(batch["weights"]) for batch in batches
                ),
            }
        for name in ("input_tokens", "assistant_target_tokens"):
            assert abs(counts["repeat"][name] / counts["boundary"][name] - 1) <= 0.005
        assert counts["repeat"]["updates"] == counts["boundary"]["updates"] == 16
        entries.append(
            {
                "seed": seed,
                "parent_seed": parent_seed,
                "evaluation_case_seed": 2026101951 + index,
                "training_order": (
                    ["repeat", "boundary"] if index % 2 == 0 else ["boundary", "repeat"]
                ),
                "parent_pointer_sha256": sha(
                    directory / "original/river_checkpoint.json"
                ),
                "canonical_training_path": str(
                    (source / "data/train.json").relative_to(REPO)
                ),
                "canonical_training_sha256": sha(source / "data/train.json"),
                "curriculum_path": str(curriculum.relative_to(REPO)),
                "curriculum_audit": audit,
                "exposure": counts,
            }
        )
        print(
            json.dumps({"prepared_training_seed": seed, "exposure": counts}), flush=True
        )
    for entry in entries:
        directory = ROOT / f'seed-{entry["seed"]}'
        for split, count in (("validation", 32), ("test", 128)):
            cases = refund_policy_benchmark(split, count, entry["evaluation_case_seed"])
            for case in cases:
                assert case["order_id"] not in seen_ids
                facts = fingerprint(case)
                assert facts not in seen_facts
                seen_ids.add(case["order_id"])
                seen_facts.add(facts)
            write(directory / "data" / f"{split}.json", cases)
        entry["input_sha256"] = {
            str(path.relative_to(directory)): sha(path)
            for path in sorted((directory / "data").iterdir())
        }
        entry["sft_specs"] = {
            arm: spec_for(entry["seed"], arm).to_dict()
            for arm in ("repeat", "boundary")
        }
    write(
        ROOT / "exclusion-audit.json",
        {
            "scope": "Fresh validation/test versus all retained historical and current training policy facts and identities, and versus other new heldouts",
            "historical_counts": historical_counts,
            "historical_sources_sha256": historical_sources,
            "fresh_validation_cases": 192,
            "fresh_test_cases": 768,
            "overlap": 0,
            "cross_parent_training_fact_overlap": 0,
        },
    )
    runtime = ROOT / "runtime"
    runtime.mkdir()
    shutil.copytree(
        REPO / "stateset_agents",
        runtime / "stateset_agents",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    for name in ("experiment.py", "control.py"):
        shutil.copyfile(ROOT / name, runtime / name)
    evaluation_source = (
        REPO / "outputs/river-six-seed-recovery-20261019/runtime/warmup_runner.py"
    ).read_text()
    assert evaluation_source.count("rl.Schedule(concurrency=8)") == 1
    (runtime / "evaluation.py").write_text(
        evaluation_source.replace(
            "rl.Schedule(concurrency=8)", "rl.Schedule(concurrency=4)"
        )
    )
    plan = {
        "schema_version": 1,
        "kind": "six-parent-paired-boundary-vs-repetition-sft-continuation",
        "base_model": MODEL,
        "implementation": package_implementation(),
        "seeds": list(SEEDS),
        "arms": list(ARMS),
        "datasets": entries,
        "parent_protocol": "Reuse all six independently trained original SFT parents; includes the prior pilot parent. This is conditional continuation replication, not six new base initializations.",
        "selection": "Fixed one-epoch final checkpoint per trained arm. Validation is descriptive; no model selection or tuning.",
        "primary_comparison": ["repeat", "boundary"],
        "acceptance": {
            "pairs": [
                ["repeat", "boundary"],
                ["original", "repeat"],
                ["original", "boundary"],
            ],
            "min_seeds": 6,
            "min_gain": 0.03,
            "require_significance": True,
            "alpha": 0.05,
            "comparisons": 3,
        },
        "exposure_matching": {
            "exact": [
                "training_rows",
                "optimizer_updates",
                "learning_rate",
                "batch_size",
                "epochs",
                "rank",
                "initial_checkpoint",
                "shuffle_seed",
            ],
            "input_and_assistant_token_relative_tolerance": 0.005,
            "scope": "Near-matched local token exposure, not exact provider compute or dollar cost",
        },
        "limits": {
            "additional_sft_updates": 192,
            "sft_attempts_per_arm": 1,
            "operational_retries": 0,
            "generated_output_reservation_per_seed": TOKEN_LIMIT_PER_SEED,
            "total_reserved_output_tokens": TOKEN_LIMIT_PER_SEED * 6,
            "max_parallel_seeds": 2,
            "seconds_per_seed": 1800,
            "wall_seconds": 3600,
            "sdk_request_timeout_seconds": 180,
            "transport_retries": False,
            "verified_dollar_cost_cap": None,
        },
        "evaluation": {
            "validation_per_seed": 32,
            "test_per_seed": 128,
            "arms_per_seed": 3,
            "expected_validation_outcomes": 576,
            "expected_test_outcomes": 2304,
            "holdout_protocol": "Seal each arm test before sampling; complete validation first; no repeat of uncertain tests",
        },
        "exclusion_audit_sha256": sha(ROOT / "exclusion-audit.json"),
        "runner_sha256": {
            name: sha(runtime / name)
            for name in ("experiment.py", "control.py", "evaluation.py")
        },
        "quality_certification": False,
    }
    plan["plan_hash"] = content_hash(plan)
    write(ROOT / "plan.json", plan)
    print(
        json.dumps(
            {
                "prepared": True,
                "limits": plan["limits"],
                "evaluation": plan["evaluation"],
            }
        ),
        flush=True,
    )


def prepare():
    asyncio.run(prepare_async())


def validate():
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.evaluation.implementation import package_implementation

    plan = read(ROOT / "plan.json")
    assert (
        content_hash({k: v for k, v in plan.items() if k != "plan_hash"})
        == plan["plan_hash"]
    )
    assert package_implementation() == plan["implementation"]
    for name, digest in plan["runner_sha256"].items():
        assert sha(ROOT / "runtime" / name) == digest
    assert sha(ROOT / "exclusion-audit.json") == plan["exclusion_audit_sha256"]
    for entry in plan["datasets"]:
        directory = ROOT / f'seed-{entry["seed"]}'
        assert (
            sha(directory / "original/river_checkpoint.json")
            == entry["parent_pointer_sha256"]
        )
        assert (
            sha(REPO / entry["canonical_training_path"])
            == entry["canonical_training_sha256"]
        )
        for name, digest in entry["input_sha256"].items():
            assert sha(directory / name) == digest
        for arm in ("repeat", "boundary"):
            assert spec_for(entry["seed"], arm).to_dict() == entry["sft_specs"][arm]
    return plan


def worker(seed):
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
    entry = next(item for item in plan["datasets"] if item["seed"] == seed)
    directory = ROOT / f"seed-{seed}"
    with exclusive_run(directory):
        with (directory / "execution-attempt.json").open("x") as stream:
            json.dump({"plan_hash": plan["plan_hash"], "seed": seed}, stream)
            stream.flush()
            os.fsync(stream.fileno())
        budget = RolloutAdmissionBudget(
            directory / "rollout-budget.json",
            limit=TOKEN_LIMIT_PER_SEED,
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
            for arm in entry["training_order"]:
                executor = RiverExecutor(
                    client=client,
                    tokenizer=renderer.tokenizer,
                    ledger_path=directory / "river-ledger.jsonl",
                )
                executor.MAX_TRANSIENT_ATTEMPTS = 1
                handle = executor.submit(spec_for(seed, arm))
                result = executor.wait(handle)
                write(
                    directory / f"{arm}-training-result.json",
                    {
                        "status": result.status.value,
                        "logs": result.logs,
                        "duration_s": result.duration_s,
                        "cost_usd": result.cost_usd,
                    },
                )
                assert result.succeeded
                assert read(directory / arm / "river_checkpoint.json")["steps"] == 16
            helpers = load_module(
                f"fixed_evaluation_{seed}", ROOT / "runtime/evaluation.py"
            )
            helpers.ROOT, helpers.MODEL, helpers.SEED = directory, MODEL, seed
            checkpoints = [
                (
                    arm,
                    river.Checkpoint(
                        path=read(directory / arm / "river_checkpoint.json")[
                            "checkpoint"
                        ],
                        step=16,
                        checkpoint_type="inference",
                    ),
                )
                for arm in ARMS
            ]
            with client.session(
                experiment="stateset-boundary-six-seed", role=f"paired-eval-{seed}"
            ) as evaluation:
                for split in ("validation", "test"):
                    cases = read(directory / "data" / f"{split}.json")
                    for arm, checkpoint in checkpoints:
                        with (directory / f"{arm}-{split}-attempt.json").open(
                            "x"
                        ) as stream:
                            json.dump(
                                {
                                    "plan_hash": plan["plan_hash"],
                                    "checkpoint": asdict(checkpoint),
                                    "split_sha256": entry["input_sha256"][
                                        f"data/{split}.json"
                                    ],
                                },
                                stream,
                            )
                            stream.flush()
                            os.fsync(stream.fileno())

                    async def evaluate(cases=cases, split=split):
                        await asyncio.gather(
                            *(
                                helpers.evaluate(
                                    evaluation,
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
        write(
            directory / "seed-result.json",
            {"seed": seed, "status": "completed", "plan_hash": plan["plan_hash"]},
        )


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
    assert preflight["passed"]
    started = time.monotonic()
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

    def supervise(seed):
        directory = ROOT / f"seed-{seed}"
        elapsed_start, timed_out = time.monotonic(), False
        remaining = plan["limits"]["wall_seconds"] - (elapsed_start - started)
        if remaining <= 1:
            result = {"seed": seed, "status": "not_started_deadline", "exit_code": None}
            write(directory / "execution-summary.json", result)
            return result
        with (directory / "live-run.txt").open("x") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    str(ROOT / "runtime/experiment.py"),
                    "worker",
                    str(seed),
                ],
                cwd=REPO,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            try:
                code = process.wait(
                    timeout=min(plan["limits"]["seconds_per_seed"], remaining)
                )
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    code = process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    code = process.wait()
        result = {
            "seed": seed,
            "exit_code": code,
            "timed_out": timed_out,
            "elapsed_seconds": time.monotonic() - elapsed_start,
        }
        write(directory / "execution-summary.json", result)
        print(json.dumps(result), flush=True)
        return result

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [
            future.result()
            for future in as_completed([pool.submit(supervise, seed) for seed in SEEDS])
        ]
    write(
        ROOT / "execution-summary.json",
        {
            "seeds": sorted(results, key=lambda item: item["seed"]),
            "elapsed_seconds": time.monotonic() - started,
            "plan_hash": plan["plan_hash"],
        },
    )


if __name__ == "__main__":
    if sys.argv[1] == "worker":
        worker(int(sys.argv[2]))
    else:
        {"prepare": prepare, "validate": validate, "launch": launch}[sys.argv[1]]()
