"""Freeze a training-only arithmetic pilot before any new model outcome."""

import asyncio
from collections import Counter
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

REPO = Path("/home/dom/stateset-agents")
ROOT = Path(__file__).parent


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(name):
    spec = importlib.util.spec_from_file_location(
        "preparation_" + name, ROOT / (name + ".py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


async def prepare():
    from transformers import AutoTokenizer
    from stateset_agents.core.environments.refund_policy_environment import (
        RefundPolicyEnvironment,
        refund_policy_benchmark,
    )
    from stateset_agents.data.refund_arithmetic_stress import (
        refund_arithmetic_stress_cases,
    )
    from stateset_agents.data.refund_stress import refund_policy_stress_cases
    from stateset_agents.data.refund_curriculum import (
        ARITHMETIC_POLICY,
        audit_refund_curriculum,
        prepare_refund_curriculum,
    )
    from stateset_agents.evaluation.agent_runs import content_hash
    from stateset_agents.evaluation.implementation import package_implementation
    from stateset_agents.remote.river_batches import build_sft_batch
    from stateset_agents.training.sft import parse_chat_dataset

    runner, control = load("experiment"), load("control")
    assert not (ROOT / "plan.json").exists()
    canonical = REPO / "outputs/river-six-seed-recovery-20261019/seed-2026101901/data"
    previous = (
        REPO
        / "outputs/river-boundary-stability-20261019/attempt-2/seed-2026102901/curriculum"
    )
    curriculum = ROOT / "curriculum"
    manifest = await prepare_refund_curriculum(
        canonical, curriculum, policy=ARITHMETIC_POLICY
    )
    audit = await audit_refund_curriculum(
        canonical, curriculum, policy=ARITHMETIC_POLICY
    )
    candidate = read(curriculum / "train.json")
    control_cases, control_rows, lineage = await control.repetition_control(
        read(previous / "train.json"),
        dict(Counter(r["family"] for r in candidate)),
        runner.SEED,
    )
    data = ROOT / "data"
    data.mkdir()
    runner.write(data / "control-cases.json", control_cases)
    runner.write(data / "control-lineage.json", lineage)
    with (data / "control.jsonl").open("x") as stream:
        for row in control_rows:
            stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")
    shutil.copyfile(curriculum / "train.json", data / "arithmetic-cases.json")
    shutil.copyfile(curriculum / "train.jsonl", data / "arithmetic.jsonl")
    # All earlier prepared cases are excluded, including unused and rejected bundles.
    history = read(
        REPO
        / "outputs/river-boundary-stability-20261019/attempt-3/historical-sources.json"
    )
    for directory in (
        "river-boundary-stability-20261019",
        "river-stability-pilot-20261019",
    ):
        for basename in (
            "train.json",
            "control-cases.json",
            "parameter-cases.json",
            "validation.json",
            "test.json",
        ):
            for path in sorted((REPO / "outputs" / directory).rglob(basename)):
                if "snapshot" not in path.parts and "runtime" not in path.parts:
                    history[str(path.relative_to(REPO))] = sha(path)
    excluded = []
    for name, digest in history.items():
        assert sha(REPO / name) == digest
        rows = read(REPO / name)
        if isinstance(rows, list):
            excluded.extend(
                r for r in rows if isinstance(r, dict) and "return_window_days" in r
            )
    excluded.extend(candidate + control_cases)
    runner.write(ROOT / "historical-sources.json", history)
    fingerprint = RefundPolicyEnvironment.case_fingerprint
    rejected = []
    for attempt in range(100):
        case_seed = 2026107001 + attempt * 100
        working, proposed = list(excluded), {}
        try:
            for split, count, windows, quantum, repeat in (
                ("validation", 32, (5, 18, 40), 10, 1),
                ("test", 128, (8, 25, 75), 100, 2),
            ):
                ordinary = refund_policy_benchmark(split, count=count, seed=case_seed)
                ids, facts = {r["order_id"] for r in working}, {
                    fingerprint(r) for r in working
                }
                if (
                    ids & {r["order_id"] for r in ordinary}
                    or facts & {fingerprint(r) for r in ordinary}
                    or len({fingerprint(r) for r in ordinary}) != len(ordinary)
                ):
                    raise ValueError(
                        "Ordinary cases overlap historical or generated cases"
                    )
                working.extend(ordinary)
                window = refund_policy_stress_cases(
                    split, seed=case_seed, windows=windows, excluded=working
                )
                working.extend(window)
                arithmetic = refund_arithmetic_stress_cases(
                    split,
                    seed=case_seed,
                    windows=windows,
                    quantum=quantum,
                    repetitions=repeat,
                    excluded=working,
                )
                working.extend(arithmetic)
                proposed[split] = sorted(
                    ordinary + window + arithmetic, key=lambda r: r["order_id"]
                )
        except ValueError as exc:
            if "overlap" not in str(exc):
                raise
            rejected.append({"seed": case_seed, "reason": str(exc)})
            continue
        break
    else:
        raise ValueError("No fresh input seed found within 100 prospective candidates")
    for split, cases in proposed.items():
        runner.write(data / f"{split}.json", cases)
    shutil.copytree(
        REPO / "outputs/river-stability-pilot-20261019/parameter", ROOT / "original"
    )
    tokenizer = AutoTokenizer.from_pretrained(runner.MODEL, local_files_only=True)
    exposure = {}
    for arm, cases in (("control", control_cases), ("arithmetic", candidate)):
        rows = parse_chat_dataset((data / f"{arm}.jsonl").read_bytes())
        batches = build_sft_batch(rows, tokenizer, max_length=4096)
        assert len(batches) == len(rows) == len(candidate)
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
            lengths.append(length)
        exposure[arm] = {
            "rows": len(rows),
            "updates": (len(rows) + 15) // 16,
            "families": dict(Counter(r["family"] for r in cases)),
            "input_tokens": sum(len(b["input_ids"]) for b in batches),
            "assistant_target_tokens": sum(sum(b["weights"]) for b in batches),
            "maximum_sequence_tokens": max(lengths),
            "dropped_or_truncated_rows": 0,
        }
    assert exposure["control"]["families"] == exposure["arithmetic"]["families"]
    for field in ("input_tokens", "assistant_target_tokens"):
        assert (
            abs(exposure["control"][field] / exposure["arithmetic"][field] - 1) <= 0.005
        )
    assert sum(x["updates"] for x in exposure.values()) <= 90
    prior = {}
    for path in sorted(
        (REPO / "benchmark_results/training_contracts").rglob("artifacts.sha256.json")
    ):
        for name, digest in read(path).items():
            assert sha(path.parent / name) == digest
            prior[str((path.parent / name).relative_to(REPO))] = digest
    runner.write(ROOT / "prior-evidence.json", prior)
    runtime = ROOT / "runtime"
    runtime.mkdir()
    implementation = package_implementation()
    for name, digest in implementation["files"].items():
        source = REPO / "stateset_agents" / name
        assert sha(source) == digest
        dest = runtime / "stateset_agents" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
    for name in ("experiment.py", "preparation.py", "control.py", "gates.py"):
        shutil.copyfile(ROOT / name, runtime / name)
    evaluation = (
        REPO / "outputs/river-stability-pilot-20261019/runtime/evaluation.py"
    ).read_text()
    assert evaluation.count("rl.Schedule(concurrency=4)") == 1
    (runtime / "evaluation.py").write_text(
        evaluation.replace("rl.Schedule(concurrency=4)", "rl.Schedule(concurrency=6)")
    )
    plan = {
        "schema_version": 1,
        "kind": "one-parent-borrow-boundary-continuation-pilot",
        "base_model": runner.MODEL,
        "training_seed": runner.SEED,
        "evaluation_case_seed": case_seed,
        "initial_checkpoint_source": "Retained, unpromoted parameter arm from previous failed pilot; both new arms load the same weights with fresh optimizers.",
        "initial_checkpoint_sha256": sha(ROOT / "original/river_checkpoint.json"),
        "implementation": implementation,
        "training_order": ["control", "arithmetic"],
        "sft_specs": {a: runner.spec_for(a).to_dict() for a in runner.ARMS},
        "exposure": exposure,
        "curriculum_manifest": manifest,
        "curriculum_audit": audit,
        "selection": "Fixed final-epoch checkpoints. Validation gates admission to test; no tuning.",
        "acceptance": "No success/policy/truncation regression overall, by family, suite, suite-family, stress source family/window, arithmetic paid/refund/day offsets. Test additionally requires strictly more arithmetic successes than control.",
        "scope": "Exploratory after prior test failure; all new supervision derives from canonical training sources. Single parent; no population or A+ claim.",
        "matching_scope": "Exact family, row/update and hyperparameter budgets; local input/assistant token counts within 0.5%. Window/amount mixtures, provider compute and dollars are not exactly matched.",
        "evaluation": {
            "arms": list(runner.ARMS),
            "validation_per_arm": 158,
            "test_per_arm": 308,
            "arithmetic_validation": 54,
            "arithmetic_test": 108,
            "arithmetic_training_quantum": 1000,
            "arithmetic_validation_quantum": 10,
            "arithmetic_test_quantum": 100,
            "max_total_concurrency": 12,
            "test_rule": "Stop if validation gate fails; otherwise seal both test attempts before sampling, one attempt only.",
        },
        "limits": {
            "additional_sft_updates": sum(e["updates"] for e in exposure.values()),
            "training_attempts_per_arm": 1,
            "generated_output_reservation": runner.TOKEN_LIMIT,
            "trajectory_output_tokens": 1024,
            "wall_seconds": 2400,
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
            for name in (
                "experiment.py",
                "preparation.py",
                "control.py",
                "gates.py",
                "evaluation.py",
            )
        },
        "prior_evidence_sha256": sha(ROOT / "prior-evidence.json"),
        "historical_sources_sha256": sha(ROOT / "historical-sources.json"),
        "rejected_input_seeds": rejected,
        "id_and_fact_overlap": 0,
        "quality_certification": False,
    }
    plan["plan_hash"] = content_hash(plan)
    runner.write(ROOT / "plan.json", plan)
    print(
        json.dumps(
            {
                "prepared": True,
                "exposure": exposure,
                "limits": plan["limits"],
                "evaluation_case_seed": case_seed,
                "rejected_input_seeds": rejected,
            }
        ),
        flush=True,
    )
