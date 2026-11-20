"""Three matched arms on τ-bench retail: prompted, filtered fine-tuning and RL.

Every arm starts from the same base model and the same untrained adapter.
The RL arm and the fine-tuning arm draw the same number of training rollouts
on the same schedule of official train tasks and take the same number of
optimizer updates at the same learning rate. Fine-tuning imitates the base
policy's own successful rollouts (no reference actions are used by any arm).
All arms are scored on the official test split with identical customer seeds.

Run ``python -m stateset_agents.training.tau_retail_study --help``.
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import random
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

from stateset_agents.core.trajectory import ConversationTurn
from stateset_agents.evaluation.replication import paired_seed_case_summary

from .tau_retail import (
    EvaluationUnavailable,
    TauRetailEnvironment,
    TauRetailSettings,
    evaluate_policy,
    retail_task_ids,
    success_counts,
    tau2_provenance,
)

ARMS = ("prompted", "sft", "rl")
COMPARISONS = (("sft", "rl"), ("prompted", "rl"), ("prompted", "sft"))
REQUIRED = {
    "model": str,
    "model_revision": str,
    "seeds": list,
    "updates": int,
    "tasks_per_update": int,
    "generations": int,
    "learning_rate": float,
    "lora_r": int,
    "max_prompt_tokens": int,
    "max_new_tokens": int,
    "sft_turns_per_update": int,
    "evaluation_trials": int,
    "evaluation_temperature": float,
    "analysis_seed": int,
    "environment": dict,
}


def load_protocol(path: Path) -> dict[str, Any]:
    """Load a protocol and reject missing, mistyped or non-positive fields."""
    protocol: dict[str, Any] = json.loads(path.read_text())
    for name, kind in REQUIRED.items():
        value = protocol.get(name)
        if type(value) is not kind:
            raise ValueError(f"protocol.{name} must be {kind.__name__}")
        if isinstance(value, int) and name != "analysis_seed" and value < 1:
            raise ValueError(f"protocol.{name} must be positive")
    seeds = protocol["seeds"]
    if len(set(seeds)) != len(seeds) or any(type(s) is not int for s in seeds):
        raise ValueError("protocol.seeds must be unique integers")
    if protocol.get("measured"):
        validate_measured(protocol)
    return protocol


def validate_measured(protocol: Mapping[str, Any]) -> None:
    """A measured protocol pins the benchmark data and prices GPU time."""
    digest = protocol.get("tau2_retail_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("A measured protocol needs tau2_retail_sha256")
    rate = protocol.get("gpu_hourly_usd")
    if isinstance(rate, bool) or not isinstance(rate, (int, float)) or rate <= 0:
        raise ValueError("A measured protocol needs a positive gpu_hourly_usd")
    if not isinstance(protocol.get("cost_source"), str):
        raise ValueError("A measured protocol needs a cost_source")
    environment = protocol["environment"]
    for key in ("user_llm_args", "judge_llm_args"):
        if "mock_response" in (environment.get(key) or {}):
            raise ValueError("A measured protocol cannot mock the customer or judge")


def gpu_cost(protocol: Mapping[str, Any], seconds: float) -> float | None:
    """Price wall time at the protocol's GPU rate; ``None`` when unpriced."""
    rate = protocol.get("gpu_hourly_usd")
    return None if rate is None else seconds / 3600 * float(rate)


def peak_memory_gb() -> dict[str, Any]:
    """Peak accelerator memory when CUDA is present, else peak process memory."""
    import resource

    import torch

    if torch.cuda.is_available():
        return {
            "source": "cuda",
            "gb": max(
                torch.cuda.max_memory_allocated(i)
                for i in range(torch.cuda.device_count())
            )
            / 1e9,
        }
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"source": "process", "gb": peak / 1e6}


def estimate_full_run(
    protocol: Mapping[str, Any], report: Mapping[str, Any], *, test_tasks: int
) -> dict[str, Any]:
    """Extrapolate a preflight seed to the whole protocol.

    Scales measured seconds per training rollout and per scored conversation
    to the planned counts. Conversation length varies by task, so treat the
    result as an order-of-magnitude estimate.
    """
    arms = report["arms"]
    planned = (
        protocol["updates"] * protocol["tasks_per_update"] * protocol["generations"]
    )
    episodes = test_tasks * protocol["evaluation_trials"]
    training = sum(
        arms[arm]["training_seconds"] / arms[arm]["training_rollouts"] * planned
        for arm in ("rl", "sft")
    )
    scoring = sum(
        arms[arm]["evaluation_seconds"] / arms[arm]["evaluation_episodes"] * episodes
        for arm in ARMS
    )
    seconds = (training + scoring) * len(protocol["seeds"])
    return {
        "planned_training_rollouts_per_arm_per_seed": planned,
        "planned_scored_conversations_per_arm_per_seed": episodes,
        "seeds": len(protocol["seeds"]),
        "estimated_gpu_hours": seconds / 3600,
        "estimated_gpu_cost_usd": gpu_cost(protocol, seconds),
        "excludes": "customer-simulator and judge API usage, provisioning time",
    }


def training_schedule(
    train_ids: Sequence[str], protocol: Mapping[str, Any], seed: int
) -> list[list[dict[str, Any]]]:
    """Return the task batch for every update; shared by the RL and SFT arms."""
    order = list(train_ids)
    random.Random(seed).shuffle(order)
    size = protocol["tasks_per_update"]
    return [
        [
            {"id": order[(step * size + k) % len(order)], "seed": seed + step}
            for k in range(size)
        ]
        for step in range(protocol["updates"])
    ]


def assistant_turns(trajectory: Any) -> list[ConversationTurn]:
    """Return the sampled assistant turns of one trajectory."""
    turns = [
        ConversationTurn.from_dict(t) if isinstance(t, dict) else t
        for t in trajectory.turns
    ]
    return [t for t in turns if t.role == "assistant"]


@contextlib.contextmanager
def evaluation_sampling(agent: Any, temperature: float) -> Iterator[None]:
    """Temporarily decode at the evaluation temperature (0 means greedy)."""
    config = agent.config
    saved = (config.temperature, config.do_sample)
    config.do_sample = temperature > 0
    config.temperature = temperature if temperature > 0 else 1.0
    try:
        yield
    finally:
        config.temperature, config.do_sample = saved


def sft_update(
    model: Any,
    optimizer: Any,
    turns: Sequence[ConversationTurn],
    max_prompt_tokens: int,
) -> float:
    """One optimizer update maximising the likelihood of ``turns``' tokens."""
    import torch

    model.train()
    optimizer.zero_grad(set_to_none=True)
    total = 0.0
    for turn in turns:
        prompt = list(turn.metadata["prompt_token_ids"])[-max_prompt_tokens:]
        action = list(turn.metadata["token_ids"])
        ids = torch.tensor([prompt + action], device=model.device)
        labels = ids.clone()
        labels[0, : len(prompt)] = -100
        loss = model(input_ids=ids, labels=labels).loss / len(turns)
        loss.backward()
        total += float(loss.detach())
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    model.eval()
    return total


async def run_seed(
    protocol: Mapping[str, Any],
    seed: int,
    output: Path,
    *,
    environment: TauRetailEnvironment | None = None,
    train_ids: Sequence[str] | None = None,
    test_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Train the RL and SFT arms for one seed and score all three arms."""
    import numpy as np
    import torch

    from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

    from .config import TrainingConfig
    from .multi_turn_trainer import MultiTurnGRPOTrainer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    root = output / f"seed-{seed}"
    root.mkdir(parents=True)
    env_settings = TauRetailSettings(**protocol["environment"])
    provenance = tau2_provenance()
    if protocol.get("measured") and (
        provenance["retail_sha256"] != protocol["tau2_retail_sha256"]
    ):
        raise EvaluationUnavailable(
            "Installed tau2 retail data differs from the protocol's pinned digest"
        )
    environment = environment or TauRetailEnvironment(env_settings)
    train_ids = list(train_ids or retail_task_ids("train"))
    test_ids = list(test_ids or retail_task_ids("test"))
    if set(train_ids) & set(test_ids):
        raise ValueError("Train and test tasks overlap")
    config = TrainingConfig(
        model_name=protocol["model"],
        model_revision=protocol["model_revision"],
        num_episodes=protocol["updates"],
        num_generations=protocol["generations"],
        gradient_accumulation_steps=1,
        generation_batch_size=protocol.get("generation_batch_size", 2),
        learning_rate=protocol["learning_rate"],
        max_prompt_length=protocol["max_prompt_tokens"],
        max_completion_length=protocol["max_new_tokens"],
        use_lora=True,
        report_to="none",
        seed=seed,
        output_dir=str(root),
        entropy_coef=0.0,
        zero_signal_patience=0,
        bf16=bool(protocol.get("bf16", True)),
        gradient_checkpointing=bool(protocol.get("gradient_checkpointing", False)),
        bounded_token_backward=bool(protocol.get("bounded_token_backward", False)),
        rollout_concurrency=protocol.get("rollout_concurrency", 1),
        strict_rewards=True,
    )
    agent = MultiTurnAgent(
        AgentConfig(
            model_name=protocol["model"],
            model_revision=protocol["model_revision"],
            max_new_tokens=protocol["max_new_tokens"],
            top_p=1.0,
            top_k=0,
            temperature=1.0,
            repetition_penalty=1.0,
            use_peft=True,
            peft_config={
                "r": protocol["lora_r"],
                "lora_alpha": 2 * protocol["lora_r"],
                "lora_dropout": 0.0,
                "target_modules": protocol.get(
                    "lora_target_modules", ["q_proj", "v_proj"]
                ),
            },
        )
    )
    await agent.initialize()
    trainer = MultiTurnGRPOTrainer(agent, environment, config=config)
    await trainer.initialize()
    model = agent.model
    if model is None:
        raise RuntimeError("Agent initialised without a model")
    initial = {
        name: p.detach().clone()
        for name, p in model.named_parameters()
        if p.requires_grad
    }
    schedule = training_schedule(train_ids, protocol, seed)

    async def score(arm: str) -> list[dict[str, Any]]:
        torch.manual_seed(seed + 10_000)
        with evaluation_sampling(agent, protocol["evaluation_temperature"]):
            rows: list[dict[str, Any]] = await evaluate_policy(
                environment,
                agent.generate_turn,
                test_ids,
                trials=protocol["evaluation_trials"],
                seed=seed + 10_000,
                concurrency=protocol.get("evaluation_concurrency", 1),
            )
        (root / f"{arm}.json").write_text(json.dumps(rows, indent=2) + "\n")
        return rows

    def budget(groups: Sequence[Any]) -> tuple[int, int, int]:
        trajectories = [t for g in groups for t in g.trajectories]
        tokens = sum(
            len(turn.metadata["token_ids"])
            for t in trajectories
            for turn in assistant_turns(t)
        )
        wins = sum(
            bool((t.final_state or {}).get("task_success")) for t in trajectories
        )
        return len(trajectories), tokens, wins

    async def timed_score(arm: str) -> dict[str, Any]:
        started = time.monotonic()
        rows = await score(arm)
        seconds = time.monotonic() - started
        return {
            "rows": rows,
            "evaluation_seconds": seconds,
            "evaluation_episodes": len(rows),
            "evaluation_gpu_cost_usd": gpu_cost(protocol, seconds),
        }

    def trained(
        rollouts: int, tokens: int, wins: int, seconds: float
    ) -> dict[str, Any]:
        return {
            "training_rollouts": rollouts,
            "training_action_tokens": tokens,
            "training_successes": wins,
            "training_seconds": seconds,
            "training_rollouts_per_second": rollouts / seconds,
            "training_action_tokens_per_second": tokens / seconds,
            "training_gpu_cost_usd": gpu_cost(protocol, seconds),
        }

    arms: dict[str, dict[str, Any]] = {"prompted": await timed_score("prompted")}

    # RL arm.
    started = time.monotonic()
    rollouts = tokens = wins = 0
    for step, batch in enumerate(schedule):
        torch.manual_seed(seed + 20_000 + step)
        groups = await trainer.generate_trajectories(batch)
        n, t, w = budget(groups)
        rollouts, tokens, wins = rollouts + n, tokens + t, wins + w
        await trainer.training_step(groups)
    arms["rl"] = (
        trained(rollouts, tokens, wins, time.monotonic() - started)
        | {"optimizer_updates": len(schedule)}
        | await timed_score("rl")
    )

    # SFT arm: back to the untrained adapter, same rollout schedule, imitate wins.
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.requires_grad:
                p.copy_(initial[name])
    started = time.monotonic()
    rollouts = tokens = wins = 0
    winning_turns: list[ConversationTurn] = []
    for step, batch in enumerate(schedule):
        torch.manual_seed(seed + 20_000 + step)
        groups = await trainer.generate_trajectories(batch)
        n, t, w = budget(groups)
        rollouts, tokens, wins = rollouts + n, tokens + t, wins + w
        for group in groups:
            for trajectory in group.trajectories:
                if (getattr(trajectory, "final_state", None) or {}).get("task_success"):
                    winning_turns.extend(assistant_turns(trajectory))
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=protocol["learning_rate"],
    )
    sampler = random.Random(seed + 30_000)
    losses = []
    if winning_turns:
        for _ in schedule:
            picked = sampler.choices(winning_turns, k=protocol["sft_turns_per_update"])
            losses.append(
                sft_update(model, optimizer, picked, protocol["max_prompt_tokens"])
            )
    arms["sft"] = {
        "imitated_turns_available": len(winning_turns),
        "optimizer_updates": len(losses),
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
    }
    arms["sft"] = (
        trained(rollouts, tokens, wins, time.monotonic() - started)
        | arms["sft"]
        | await timed_score("sft")
    )
    report = {
        "seed": seed,
        "measured": bool(protocol.get("measured", False)),
        "tau2": provenance,
        "cost_source": protocol.get("cost_source", "unmeasured"),
        "peak_memory": peak_memory_gb(),
        "environment": env_settings.as_dict(),
        "train_tasks": len(train_ids),
        "test_tasks": test_ids,
        "arms": {
            arm: {k: v for k, v in arms[arm].items() if k != "rows"}
            | {"passed": success_counts(arms[arm]["rows"])}
            for arm in ARMS
        },
    }
    (root / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def compare(protocol: Mapping[str, Any], reports: Sequence[Mapping[str, Any]]) -> dict:
    """Paired seed-and-task analysis for RL vs SFT and each arm vs prompted."""
    tasks = reports[0]["test_tasks"]
    if any(r["test_tasks"] != tasks for r in reports):
        raise ValueError("Seeds were scored on different test tasks")
    result: dict[str, Any] = {
        "measured": all(r["measured"] for r in reports),
        "primary": "rl minus sft",
        "comparisons": {},
    }
    for parent, candidate in COMPARISONS:
        runs = {
            r["seed"]: {
                task: (
                    r["arms"][parent]["passed"].get(task, 0),
                    r["arms"][candidate]["passed"].get(task, 0),
                )
                for task in tasks
            }
            for r in reports
        }
        result["comparisons"][f"{candidate}-minus-{parent}"] = paired_seed_case_summary(
            runs,
            expected_seeds=protocol["seeds"],
            case_families=dict.fromkeys(tasks, "retail-test"),
            group_size=protocol["evaluation_trials"],
            analysis_seed=protocol["analysis_seed"],
            bootstrap_replicates=protocol.get("bootstrap_replicates", 10_000),
        )
    return result


def preflight(protocol: Mapping[str, Any], output: Path) -> int:
    """Measure throughput and memory on a small slice and estimate the full run.

    Uses the protocol's model, prompt length and environment with one update
    and two test tasks. The result is never a measured benchmark result.
    """
    trial = dict(protocol) | {"measured": False, "updates": 1}
    test = retail_task_ids("test")
    report = asyncio.run(
        run_seed(trial, protocol["seeds"][0], output, test_ids=test[:2])
    )
    estimate = {
        "preflight": True,
        "peak_memory": report["peak_memory"],
        "arms": report["arms"],
        "full_run": estimate_full_run(protocol, report, test_tasks=len(test)),
    }
    (output / "preflight.json").write_text(json.dumps(estimate, indent=2) + "\n")
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run every protocol seed and write the paired comparison."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit-train", type=int, help="Plumbing runs only")
    parser.add_argument("--limit-test", type=int, help="Plumbing runs only")
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="One seed, one update, two test tasks; writes a cost estimate",
    )
    args = parser.parse_args(argv)
    protocol = load_protocol(args.protocol)
    if args.preflight:
        return preflight(protocol, args.output)
    limited = args.limit_train is not None or args.limit_test is not None
    if limited and protocol.get("measured"):
        raise SystemExit("A measured protocol cannot run on a task subset")
    train = retail_task_ids("train")[: args.limit_train]
    test = retail_task_ids("test")[: args.limit_test]
    reports = [
        asyncio.run(
            run_seed(protocol, seed, args.output, train_ids=train, test_ids=test)
        )
        for seed in protocol["seeds"]
    ]
    comparison = compare(protocol, reports)
    (args.output / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
