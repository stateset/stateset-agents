"""Execute paired multi-seed native RL studies on a ledger-backed refund task.

Reviewed external cases are required for representative-quality claims. The
explicit synthetic option exercises the complete pipeline without such a claim.
Validation chooses checkpoints; the final test set is evaluated exactly once.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import random
import re
import time
from pathlib import Path
from typing import Any

import numpy as np

from stateset_agents.core.environments.refund_policy_environment import (
    RefundPolicyEnvironment,
    refund_policy_benchmark,
)
from stateset_agents.evaluation.agent_runs import (
    compare_runs,
    content_hash,
    evaluation_report,
)
from stateset_agents.evaluation.implementation import package_implementation
from stateset_agents.evaluation.isolation import (
    EVALUATION_RNG_PROTOCOL,
    isolated_evaluation,
    paired_evaluation_seed,
)


def load_cases(path: Path) -> dict[str, Any]:
    """Validate reviewed provenance and disjoint independent case partitions."""
    data: dict[str, Any] = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError("Study cases must be a JSON object")
    provenance = data.get("provenance", {})
    if (
        not isinstance(provenance, dict)
        or provenance.get("source") != "reviewed"
        or not isinstance(provenance.get("reviewer"), str)
        or not provenance["reviewer"].strip()
    ):
        raise ValueError("Representative studies require reviewed case provenance")
    ids: set[str] = set()
    fingerprints: set[str] = set()
    for split in ("train", "validation", "test"):
        cases = data.get(split)
        if not isinstance(cases, list) or len(cases) < 32:
            raise ValueError(
                "Each study partition requires at least 32 independent cases"
            )
        for case in cases:
            RefundPolicyEnvironment.validate_scenario(case)
            identifier = case["order_id"]
            fingerprint = RefundPolicyEnvironment.case_fingerprint(case)
            if identifier in ids or fingerprint in fingerprints:
                raise ValueError(
                    "Study cases must be unique and disjoint across partitions"
                )
            ids.add(identifier)
            fingerprints.add(fingerprint)
    content_hash(data)  # Reject nonfinite metadata before allocating a model.
    return data


def validate_study_config(args: argparse.Namespace) -> None:
    """Reject unusable study budgets and protocols before allocating a model."""
    if (
        not isinstance(args.model, str)
        or not args.model.strip()
        or not isinstance(args.model_revision, str)
        or not re.fullmatch("[0-9a-f]{40}", args.model_revision)
        or args.model_revision == "0" * 40
    ):
        raise ValueError("Pin a nonempty model to a nonzero full model revision")
    if any(
        type(getattr(args, name)) is not int or getattr(args, name) < 1
        for name in ("updates", "generations", "eval_every")
    ):
        raise ValueError("Run budgets must be positive integers")
    if args.generations < 2:
        raise ValueError("Group-relative learning requires at least two generations")
    if (
        not isinstance(args.seeds, (list, tuple))
        or not args.seeds
        or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in args.seeds)
        or len(set(args.seeds)) != len(args.seeds)
    ):
        raise ValueError("Use unique integer seeds in [0, 2**32)")
    if not args.synthetic and len(args.seeds) < 6:
        raise ValueError("Reviewed quality studies require at least six training seeds")
    try:
        valid_rate = (
            type(args.learning_rate) in (int, float)
            and math.isfinite(args.learning_rate)
            and args.learning_rate > 0
        )
    except OverflowError:
        valid_rate = False
    if not valid_rate:
        raise ValueError("Learning rate must be finite and positive")
    if args.synthetic and (
        type(args.synthetic_case_count) is not int
        or args.synthetic_case_count < 8
        or args.synthetic_case_count % 8
    ):
        raise ValueError("Synthetic case count must be a positive multiple of eight")


async def evaluate(
    agent: Any,
    cases: list[dict[str, Any]],
    *,
    seed: int,
    settings: dict[str, Any],
    checkpoint: str,
    implementation: dict[str, Any] | None = None,
    split: str = "test",
) -> dict[str, Any]:
    """Score paired cases without changing local training RNGs or model modes."""
    if implementation is not None and package_implementation() != implementation:
        raise ValueError("Implementation changed during the study")

    if not isinstance(cases, list) or not cases:
        raise ValueError("Evaluation requires nonempty cases")
    if split not in {"validation", "test"}:
        raise ValueError("Evaluation split must be validation or test")
    if not isinstance(settings, dict) or not settings:
        raise ValueError("Evaluation settings must be a nonempty object")
    content_hash(settings)
    identities = {}
    for case in cases:
        RefundPolicyEnvironment.validate_scenario(case)
        case_id = case["order_id"]
        if case_id in identities:
            raise ValueError("Evaluation case identifiers must be unique")
        case_hash = content_hash(case)
        identities[case_id] = (
            case_hash,
            paired_evaluation_seed(seed, case_id, case_hash),
        )
    rng_settings = {"protocol": EVALUATION_RNG_PROTOCOL, "scope": "local-global-rngs"}
    if (
        "evaluation_randomness" in settings
        and settings["evaluation_randomness"] != rng_settings
    ):
        raise ValueError(
            "Evaluation randomness settings differ from the native protocol"
        )
    outcomes = []
    for case in cases:
        case_hash, evaluation_seed = identities[case["order_id"]]
        started = time.monotonic()
        with isolated_evaluation(
            seed=evaluation_seed, model=getattr(agent, "model", None)
        ):
            environment = RefundPolicyEnvironment()
            trajectory = await environment.run_episode(agent.generate_turn, case)
        state = trajectory.final_state or {}
        success = bool(
            state.get("resolution") is not None
            and state.get("violations") == 0
            and trajectory.metadata.get("status") == "completed"
        )
        outcomes.append(
            {
                "case_id": case["order_id"],
                "case_hash": case_hash,
                "evaluation_seed": evaluation_seed,
                "success": success,
                "policy_violations": int(state.get("violations", 0)),
                "truncated": (
                    "turn_limit"
                    if trajectory.metadata.get("status") == "timeout"
                    else None
                ),
                "reward": trajectory.total_reward,
                "generated_tokens": sum(
                    len((turn.metadata or {}).get("token_ids", ()))
                    for turn in trajectory.turns
                    if turn.role == "assistant"
                ),
                "tool_calls": len(
                    [turn for turn in trajectory.turns if turn.role == "assistant"]
                ),
                "elapsed_seconds": time.monotonic() - started,
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
    if implementation is not None and package_implementation() != implementation:
        raise ValueError("Implementation changed during evaluation")
    return evaluation_report(
        environment="refund-policy-v2",
        base_model=agent.config.model_name,
        seed=seed,
        cases={c["order_id"]: c for c in cases},
        settings={**settings, "evaluation_randomness": rng_settings},
        checkpoint={"path": checkpoint},
        outcomes=outcomes,
        implementation=implementation,
        split=split,
    )


async def run_seed(
    args: argparse.Namespace, data: dict[str, Any], seed: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train against environment rewards and select only on validation outcomes."""
    import torch

    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    from .config import TrainingConfig
    from .multi_turn_trainer import MultiTurnGRPOTrainer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = args.output / f"seed-{seed}"
    root.mkdir(parents=True)
    implementation = getattr(args, "implementation", None) or package_implementation()
    config = TrainingConfig(
        model_name=args.model,
        model_revision=args.model_revision,
        num_episodes=args.updates,
        num_generations=args.generations,
        gradient_accumulation_steps=1,
        generation_batch_size=2,
        learning_rate=args.learning_rate,
        max_prompt_length=2048,
        max_completion_length=192,
        use_lora=True,
        report_to="none",
        seed=seed,
        output_dir=str(root),
        entropy_coef=0.0,
    )
    config.validate_for_training()
    agent = MultiTurnAgent(
        AgentConfig(
            model_name=args.model,
            model_revision=args.model_revision,
            max_new_tokens=192,
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            repetition_penalty=1.0,
            use_peft=True,
            peft_config={
                "r": 8,
                "lora_alpha": 16,
                "lora_dropout": 0.0,
                "target_modules": ["q_proj", "v_proj"],
            },
        )
    )
    await agent.initialize()
    environment = RefundPolicyEnvironment(data["train"])
    trainer = MultiTurnGRPOTrainer(agent, environment, config=config)
    await trainer.initialize()
    settings = {
        "model_revision": args.model_revision,
        "temperature": 1.0,
        "max_new_tokens": 192,
        "max_turns": 4,
    }
    baseline = await evaluate(
        agent,
        data["test"],
        seed=seed,
        settings=settings,
        checkpoint=f"hf://{args.model}@{args.model_revision}",
        implementation=implementation,
    )
    (root / "baseline.json").write_text(json.dumps(baseline, indent=2) + "\n")
    await trainer.save_checkpoint(checkpoint_name="selected")
    initial = await evaluate(
        agent,
        data["validation"],
        seed=seed,
        settings=settings,
        checkpoint="initial",
        implementation=implementation,
        split="validation",
    )
    (root / "validation-0.json").write_text(json.dumps(initial, indent=2) + "\n")
    best = initial["success_rate"]
    selected_step = 0
    validations = [
        {"step": 0, "report": "validation-0.json", "sha256": content_hash(initial)}
    ]
    updates = []
    for step in range(args.updates):
        torch.manual_seed(seed + 20000 + step)
        scenario = data["train"][step % len(data["train"])]
        groups = await trainer.generate_trajectories([scenario])
        metrics = await trainer.training_step(groups)
        updates.append(
            {
                key: value
                for key, value in metrics.items()
                if isinstance(value, (int, float, str, bool))
            }
        )
        if (step + 1) % args.eval_every == 0 or step + 1 == args.updates:
            validation = await evaluate(
                agent,
                data["validation"],
                seed=seed,
                settings=settings,
                checkpoint=str(step + 1),
                implementation=implementation,
                split="validation",
            )
            (root / f"validation-{step + 1}.json").write_text(
                json.dumps(validation, indent=2) + "\n"
            )
            validations.append(
                {
                    "step": step + 1,
                    "report": f"validation-{step + 1}.json",
                    "sha256": content_hash(validation),
                }
            )
            if validation["success_rate"] > best:
                best = validation["success_rate"]
                selected_step = step + 1
                await trainer.save_checkpoint(checkpoint_name="selected")
    # Preserve actual update counts even if validation selects the initial policy.
    (root / "updates.json").write_text(
        json.dumps(updates, indent=2, allow_nan=False) + "\n"
    )
    committed_updates = 0
    for metrics in updates:
        count = metrics.get("optimizer_updates")
        if type(count) is not int or count < 0:
            raise RuntimeError(
                "Missing committed optimizer updates; candidate test remains sealed"
            )
        committed_updates += count
        if (
            type(metrics.get("global_step")) is not int
            or metrics["global_step"] != committed_updates
        ):
            raise RuntimeError(
                "Inconsistent optimizer counters; candidate test remains sealed"
            )
    if committed_updates < 1 or committed_updates != trainer.global_step:
        raise RuntimeError(
            "No verified training progress; candidate test remains sealed"
        )
    if not trainer.load_checkpoint(root / "selected"):
        raise RuntimeError("Selected checkpoint did not reload")
    checkpoint_inventory = content_hash(
        json.loads((root / "selected" / ".stateset-checkpoint.json").read_text())
    )
    selection = {
        "schema_version": 1,
        "criterion": "validation_success_rate",
        "tie_break": "earliest_step",
        "selected_step": selected_step,
        "selected_success_rate": best,
        "committed_optimizer_updates": committed_updates,
        "validation_reports": validations,
        "checkpoint_inventory_sha256": checkpoint_inventory,
    }
    (root / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    candidate = await evaluate(
        agent,
        data["test"],
        seed=seed,
        settings=settings,
        checkpoint=str(root / "selected"),
        implementation=implementation,
    )
    candidate["selected_checkpoint"]["inventory_sha256"] = checkpoint_inventory
    candidate["selected_checkpoint"]["selection_sha256"] = content_hash(selection)
    (root / "candidate.json").write_text(json.dumps(candidate, indent=2) + "\n")
    del trainer, agent
    torch.cuda.empty_cache()
    return baseline, candidate


def main() -> None:
    """Run a prespecified study; a failed quality gate returns a nonzero status."""
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--cases", type=Path)
    source.add_argument("--synthetic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--model-revision", required=True)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=[42, 43, 44, 45, 46, 47]
    )
    parser.add_argument("--updates", type=int, default=20)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--synthetic-case-count", type=int, default=32)
    args = parser.parse_args()
    try:
        validate_study_config(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.output.exists():
        parser.error("Output directory must be new")
    data = (
        load_cases(args.cases)
        if args.cases
        else {
            "provenance": {"source": "synthetic"},
            **{
                split: refund_policy_benchmark(split, args.synthetic_case_count, 0)
                for split in ("train", "validation", "test")
            },
        }
    )
    args.output.mkdir(parents=True, mode=0o700)
    args.implementation = package_implementation()
    (args.output / "cases.json").write_text(json.dumps(data, indent=2) + "\n")
    manifest = {
        **{k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "cases_sha256": content_hash(data),
        "provenance": data["provenance"],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    pairs = [asyncio.run(run_seed(args, data, seed)) for seed in args.seeds]
    result = compare_runs(
        [p[0] for p in pairs],
        [p[1] for p in pairs],
        min_seeds=6,
        require_significance=True,
    )
    result["representative_cases"] = not args.synthetic
    result["passed"] = result["passed"] and not args.synthetic
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
