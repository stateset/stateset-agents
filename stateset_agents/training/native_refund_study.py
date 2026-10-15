"""Execute paired multi-seed native RL studies on a ledger-backed refund task.

Reviewed external cases are required for representative-quality claims. The
explicit synthetic option exercises the complete pipeline without such a claim.
Validation chooses checkpoints; the final test set is evaluated exactly once.
"""

from __future__ import annotations

import argparse
import asyncio
import json
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


def load_cases(path: Path) -> dict[str, Any]:
    """Validate reviewed provenance and disjoint independent case partitions."""
    data: dict[str, Any] = json.loads(path.read_text())
    provenance = data.get("provenance", {})
    if (
        provenance.get("source") != "reviewed"
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
            fingerprint = content_hash(
                {
                    key: value
                    for key, value in case.items()
                    if key not in {"order_id", "family", "split"}
                }
            )
            if identifier in ids or fingerprint in fingerprints:
                raise ValueError(
                    "Study cases must be unique and disjoint across partitions"
                )
            ids.add(identifier)
            fingerprints.add(fingerprint)
    return data


async def evaluate(
    agent: Any,
    cases: list[dict[str, Any]],
    *,
    seed: int,
    settings: dict[str, Any],
    checkpoint: str,
    implementation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute tools and score outcomes, retaining failures and bounded truncation."""
    import torch

    if implementation is not None and package_implementation() != implementation:
        raise ValueError("Implementation changed during the study")

    torch.manual_seed(seed + 10000)
    environment = RefundPolicyEnvironment()
    outcomes = []
    for case in cases:
        started = time.monotonic()
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
        settings=settings,
        checkpoint={"path": checkpoint},
        outcomes=outcomes,
        implementation=implementation,
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
    )
    best = initial["success_rate"]
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
            )
            (root / f"validation-{step + 1}.json").write_text(
                json.dumps(validation, indent=2) + "\n"
            )
            if validation["success_rate"] > best:
                best = validation["success_rate"]
                await trainer.save_checkpoint(checkpoint_name="selected")
    # Preserve actual update counts even if validation selects the initial policy.
    (root / "updates.json").write_text(
        json.dumps(updates, indent=2, allow_nan=False) + "\n"
    )
    if not trainer.load_checkpoint(root / "selected"):
        raise RuntimeError("Selected checkpoint did not reload")
    candidate = await evaluate(
        agent,
        data["test"],
        seed=seed,
        settings=settings,
        checkpoint=str(root / "selected"),
        implementation=implementation,
    )
    candidate["selected_checkpoint"]["inventory_sha256"] = content_hash(
        json.loads((root / "selected" / ".stateset-checkpoint.json").read_text())
    )
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
    if (
        not re.fullmatch("[0-9a-f]{40}", args.model_revision)
        or min(args.updates, args.generations, args.eval_every) < 1
    ):
        parser.error("Pin a full model revision and use positive run budgets")
    if len(set(args.seeds)) != len(args.seeds) or any(seed < 0 for seed in args.seeds):
        parser.error("Use unique nonnegative seeds")
    if args.output.exists():
        parser.error("Output directory must be new")
    if args.synthetic_case_count < 8 or args.synthetic_case_count % 8:
        parser.error("Synthetic case count must be a positive multiple of eight")
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
