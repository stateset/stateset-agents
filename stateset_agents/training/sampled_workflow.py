"""Offline, genuinely sampled multi-turn tool-workflow regression.

This synthetic task validates training mechanics, not real-world domain quality.
Representative quality studies use evaluation.agent_study and reviewed cases.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import numpy as np

from stateset_agents.core.environment_base import (
    Environment,
    EnvironmentState,
    EpisodeStatus,
)
from stateset_agents.core.trajectory import ConversationTurn

logger = logging.getLogger(__name__)


def workflow_tokenizer() -> Any:
    """Build a real, local fast tokenizer without network or pretrained assets."""
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    words = [
        "<pad>",
        "<eos>",
        "<unk>",
        "request",
        "eligible",
        "ineligible",
        "resolved",
        "lookup",
        "refund",
        "deny",
        "finish",
    ]
    backend = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)), strict=True)), unk_token="<unk>")
    )
    backend.pre_tokenizer = Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )


class WorkflowEnvironment(Environment):
    """Execute lookup, a verified refund decision and finish against a ledger."""

    def __init__(self, scenarios: list[dict[str, Any]]) -> None:
        super().__init__(max_turns=3)
        self.scenarios = scenarios

    def observation_context(self, state: EnvironmentState) -> dict[str, Any]:
        """Keep eligibility private until the lookup tool returns it."""
        return {"observation": state.context["observation"]}

    async def get_initial_prompt(self, scenario: dict[str, Any] | None = None) -> str:
        """Expose a request without the evaluator's private eligibility label."""
        return "request"

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        """Start an isolated ledger with no applied resolution."""
        assert scenario is not None
        return EnvironmentState(
            scenario["id"],
            0,
            EpisodeStatus.ONGOING,
            {
                "scenario": dict(scenario),
                "observation": "request",
                "resolution": None,
                "policy_violations": 0,
            },
        )

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        """Reward only actual valid state transitions; reject incorrect actions."""
        if state.is_done:
            raise ValueError("Episode already ended")
        stage, word = state.turn_count, (action.content or "").strip()
        state.turn_count += 1
        expected = (
            "lookup"
            if stage == 0
            else (
                ("refund" if state.context["scenario"]["eligible"] else "deny")
                if stage == 1
                else "finish"
            )
        )
        valid = word == expected
        if not valid:
            state.context["policy_violations"] += 1
        elif stage == 0:
            state.context["observation"] = (
                "eligible" if state.context["scenario"]["eligible"] else "ineligible"
            )
        elif stage == 1:
            state.context["resolution"] = word
            state.context["observation"] = "resolved"
        done = not valid or stage == 2
        if done:
            state.status = EpisodeStatus.COMPLETED if valid else EpisodeStatus.FAILED
        state.context["task_success"] = bool(valid and stage == 2)
        return (
            state,
            (1.0 if valid and stage == 2 else 0.0),
            done,
            {
                "env_response": ConversationTurn(
                    role="user", content=state.context["observation"]
                ),
            },
        )


class WorkflowAgent:
    """A tiny real transformer policy using unmodified stochastic HF generation."""

    def __init__(self, model: Any, tokenizer: Any) -> None:
        self.model, self.tokenizer = model, tokenizer

    async def generate_turn(
        self, messages: Any, context: dict[str, Any]
    ) -> ConversationTurn:
        """Sample exactly one token and retain the exact conditioning and likelihood."""
        import torch

        from stateset_agents.core.generation_mode import inference_mode

        tokens = self.tokenizer(context["observation"], return_tensors="pt")[
            "input_ids"
        ].to(self.model.device)
        with inference_mode(self.model), torch.no_grad():
            generated = self.model.generate(
                tokens,
                max_new_tokens=1,
                do_sample=True,
                temperature=1.0,
                top_p=1.0,
                top_k=0,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
            sampled = generated[0, tokens.shape[1] :]
            logp = self.model(tokens).logits[0, -1].float().log_softmax(-1)[sampled[0]]
        return ConversationTurn(
            role="assistant",
            content=self.tokenizer.decode(sampled, skip_special_tokens=False),
            metadata={
                "prompt_token_ids": tokens[0].tolist(),
                "token_ids": sampled.tolist(),
                "sampler_log_probs": [float(logp)],
            },
        )


async def run_seed(
    seed: int, output_dir: Path, *, updates: int = 80, device: str = "cpu"
) -> dict[str, Any]:
    """Train, reload, and evaluate paired held-out workflow cases for one seed."""
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel

    from .config import TrainingConfig
    from .multi_turn_trainer import MultiTurnGRPOTrainer

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    tokenizer = workflow_tokenizer()
    model = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=len(tokenizer),
            n_positions=8,
            bos_token_id=0,
            eos_token_id=1,
            n_embd=24,
            n_layer=1,
            n_head=2,
            resid_pdrop=0.0,
            embd_pdrop=0.0,
            attn_pdrop=0.0,
        )
    ).to(device)
    # Establish tool syntax with supervised warmup shared by both policies.
    # Decision examples are deliberately balanced for each observation, so
    # only subsequent environment rewards teach the eligibility decision.
    warmup_optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    prompts = torch.tensor(
        [
            [tokenizer.convert_tokens_to_ids(x)]
            for x in (
                "request",
                "resolved",
                "eligible",
                "eligible",
                "ineligible",
                "ineligible",
            )
        ],
        device=device,
    )
    labels = torch.tensor(
        [
            tokenizer.convert_tokens_to_ids(x)
            for x in ("lookup", "finish", "refund", "deny", "refund", "deny")
        ],
        device=device,
    )
    for _ in range(100):
        warmup_optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(
            model(prompts).logits[:, -1], labels
        ).backward()
        warmup_optimizer.step()
    model.zero_grad(set_to_none=True)
    agent = WorkflowAgent(model, tokenizer)
    train_cases = [{"id": f"train-{i}", "eligible": bool(i % 2)} for i in range(8)]
    test_cases = [{"id": f"test-{i}", "eligible": bool(i % 2)} for i in range(128)]
    config = TrainingConfig(
        model_name="local-workflow-transformer",
        num_episodes=updates,
        num_generations=24,
        generation_batch_size=24,
        gradient_accumulation_steps=1,
        learning_rate=0.003,
        bf16=False,
        report_to="none",
        entropy_coef=0.03,
        use_lora=False,
        max_prompt_length=4,
        max_completion_length=1,
        output_dir=str(output_dir),
    )
    trainer = MultiTurnGRPOTrainer(
        agent, WorkflowEnvironment(train_cases), config=config
    )
    # Use the production optimizer setup without model loading or seed resets.
    trainer._setup_optimizer()

    async def evaluate() -> list[dict[str, Any]]:
        torch.manual_seed(seed + 10000)
        env = WorkflowEnvironment(test_cases)
        outcomes = []
        for case in test_cases:
            trajectory = await env.run_episode(agent.generate_turn, case)
            outcomes.append(
                {
                    "case_id": case["id"],
                    "success": (trajectory.final_state or {})["task_success"],
                    "policy_violations": (trajectory.final_state or {})[
                        "policy_violations"
                    ],
                    "reward": trajectory.total_reward,
                    "generated_tokens": sum(
                        t.role == "assistant" for t in trajectory.turns
                    ),
                }
            )
        return outcomes

    baseline = await evaluate()
    torch.manual_seed(seed + 20000)
    started = time.monotonic()
    metrics = []
    for step in range(updates):
        groups = await trainer.generate_trajectories(
            [train_cases[step % len(train_cases)]]
        )
        result = await trainer.training_step(groups)
        # training_step owns the optimizer-update counter.
        metrics.append(
            {k: v for k, v in result.items() if isinstance(v, (int, float, str, bool))}
        )
        if step % 10 == 0:
            successes = sum(
                bool((getattr(t, "final_state", None) or {}).get("task_success"))
                for g in groups
                for t in g.trajectories
                if getattr(t, "final_state", None)
            )
            logger.info(
                "seed=%s update=%s successful_rollouts=%s", seed, step, successes
            )
    checkpoint = output_dir / "checkpoint"
    await trainer.save_checkpoint(checkpoint_name="checkpoint")
    candidate = await evaluate()
    original = {name: p.detach().clone() for name, p in model.named_parameters()}
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    if not trainer.load_checkpoint(checkpoint):
        raise RuntimeError("Saved checkpoint failed to reload")
    if any(not torch.equal(original[name], p) for name, p in model.named_parameters()):
        raise RuntimeError("Reloaded policy differs from saved policy")
    reloaded = await evaluate()
    if candidate != reloaded:
        raise RuntimeError("Reloaded policy outcomes differ")
    delta = sum(
        int(b["success"]) - int(a["success"])
        for a, b in zip(baseline, candidate, strict=True)
    ) / len(test_cases)
    report = {
        "baseline_preparation": "100 supervised syntax updates; balanced decision labels",
        "evidence_class": "synthetic",
        "train_case_sha256": hashlib.sha256(
            json.dumps(train_cases, sort_keys=True).encode()
        ).hexdigest(),
        "test_case_sha256": hashlib.sha256(
            json.dumps(test_cases, sort_keys=True).encode()
        ).hexdigest(),
        "training_cost_usd": None,
        "cost_source": "unmeasured",
        "seed": seed,
        "device": device,
        "optimizer_updates": trainer.global_step,
        "baseline": baseline,
        "candidate": candidate,
        "improvement": delta,
        "duration_seconds": time.monotonic() - started,
        "reload_verified": True,
        "metrics": metrics,
        "resolved_config": config.to_dict(),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "result.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    """Run bounded multi-seed sampled regression and publish honest evidence."""
    import torch

    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--updates", type=int, default=80)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    if args.updates < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Use positive updates and unique seeds")
    torch.set_num_threads(1)
    if args.output.exists():
        parser.error("Output directory must be new")
    reports = [
        asyncio.run(
            run_seed(
                seed,
                args.output / f"seed-{seed}",
                updates=args.updates,
                device=args.device,
            )
        )
        for seed in args.seeds
    ]
    deltas = np.asarray([r["improvement"] for r in reports])
    rng = np.random.default_rng(0)
    interval_values = np.quantile(
        rng.choice(deltas, (10000, len(deltas)), replace=True).mean(axis=1),
        [0.025, 0.975],
    )
    interval = [float(x) for x in np.asarray(interval_values).reshape(-1)]
    summary = {
        "schema_version": 1,
        "task": "synthetic-tool-workflow-v1",
        "claim_scope": "sampled_training_regression_only",
        "seeds": args.seeds,
        "mean_improvement": float(deltas.mean()),
        "seed_bootstrap_95": interval,
        "passed": len(reports) >= 3
        and interval[0] > 0
        and all(r["reload_verified"] for r in reports),
        "implementation_sha256": hashlib.sha256(
            Path(__file__).read_bytes()
        ).hexdigest(),
        "results": [str(Path(f"seed-{seed}") / "result.json") for seed in args.seeds],
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary))
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
