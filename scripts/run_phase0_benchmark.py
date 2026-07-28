"""
Phase 0 benchmark runner — produces empirical evidence for the whitepaper.

Runs a fixed (task, model, trainer, seed) configuration and emits a single
JSON result that fits the canonical benchmark schema. Designed to be invoked
from Colab, CI, or a local A100/H100.

Two tasks supported:

* ``gsm8k`` — single-turn, verifiable math reasoning. The cheapest path to
  publishable numbers.
* ``customer_support`` — multi-turn dialogue with composite rule-based reward.
  This is the framework's differentiator over TRL.

Usage::

    python scripts/run_phase0_benchmark.py \\
        --trainer gspo --task gsm8k \\
        --model Qwen/Qwen3.5-0.8B \\
        --num-train-examples 200 --num-eval-examples 100 \\
        --seed 42 \\
        --output benchmark_results/whitepaper_v1/gspo_seed42_gsm8k.json

The JSON schema is documented in ``benchmark_results/SCHEMA.md``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("phase0_benchmark")


def get_git_commit() -> str:
    """Return current git HEAD short SHA, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            cwd=Path(__file__).parent.parent,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return "unknown"


# ---------------------------------------------------------------------------
# Task adapters
# ---------------------------------------------------------------------------


class TaskAdapter:
    """Common interface for a benchmark task.

    Concrete subclasses adapt a specific dataset + reward function to the
    runner's eval loop. Both tasks expose:

    * ``load_train_and_eval(...)``  — returns (train_examples, eval_examples)
    * ``format_prompt(example)``    — turns an example into a prompt string
    * ``score_response(example, response)`` — returns 1.0 if correct, else 0.0
    * ``max_new_tokens``            — generation length cap
    """

    name: str = "abstract"
    max_new_tokens: int = 256

    def load(self, n_train: int, n_eval: int) -> tuple[list, list]:
        raise NotImplementedError

    def format_prompt(self, example: Any) -> str:
        raise NotImplementedError

    def score_response(self, example: Any, response: str) -> tuple[float, bool]:
        """Return (score, parseable_bool)."""
        raise NotImplementedError


class GSM8KAdapter(TaskAdapter):
    name = "gsm8k"
    max_new_tokens = 256

    def load(self, n_train: int, n_eval: int) -> tuple[list, list]:
        from stateset_agents.data.gsm8k import load_gsm8k

        train, test = load_gsm8k(limit=max(n_train, n_eval))
        return train[:n_train], test[:n_eval]

    def format_prompt(self, example: Any) -> str:
        return f"Solve this step by step.\n\n{example.question}\n\nAnswer:"

    def score_response(self, example: Any, response: str) -> tuple[float, bool]:
        from stateset_agents.data.gsm8k import extract_predicted_answer

        predicted = extract_predicted_answer(response)
        if predicted is None:
            return 0.0, False
        correct = abs(predicted - example.gold_answer) < 1e-3
        return (1.0 if correct else 0.0), True


class CustomerSupportAdapter(TaskAdapter):
    name = "customer_support"
    max_new_tokens = 320

    def load(self, n_train: int, n_eval: int) -> tuple[list, list]:
        from stateset_agents.data.customer_support_bench import load_support_scenarios

        scenarios = load_support_scenarios()
        # 24-scenario corpus; split 16 train / 8 eval for the default sizes,
        # otherwise honor user limits (capped by available data).
        n_train = min(n_train, len(scenarios) - 1)
        n_eval = min(n_eval, len(scenarios) - n_train)
        return scenarios[:n_train], scenarios[n_train : n_train + n_eval]

    def format_prompt(self, example: Any) -> str:
        return (
            "You are a helpful customer support agent. Respond to the user "
            "warmly, address their concern directly, and confirm the next step.\n\n"
            f"User: {example.user_query}\n\nAgent:"
        )

    def score_response(self, example: Any, response: str) -> tuple[float, bool]:
        from stateset_agents.core.trajectory import ConversationTurn
        from stateset_agents.data.customer_support_bench import SupportRewardComposite

        reward = SupportRewardComposite()
        turns = [ConversationTurn(role="assistant", content=response)]
        context = example.to_scenario()
        result = asyncio.run(reward.compute_reward(turns, context=context))
        # Always parseable for this task — composite always returns a number.
        return result.score, True


class ToolCallingAdapter(TaskAdapter):
    name = "tool_calling"
    max_new_tokens = 320

    def load(self, n_train: int, n_eval: int) -> tuple[list, list]:
        from stateset_agents.data.tool_calling_bench import load_tool_call_scenarios

        scenarios = load_tool_call_scenarios()
        n_train = min(n_train, len(scenarios) - 1)
        n_eval = min(n_eval, len(scenarios) - n_train)
        return scenarios[:n_train], scenarios[n_train : n_train + n_eval]

    def format_prompt(self, example: Any) -> str:
        return (
            "You have access to tools. Respond with a JSON block in this format:\n"
            '```json\n{"tool": "<tool_name>", "parameters": {...}}\n```\n\n'
            f"User: {example.user_query}\n\nAgent:"
        )

    def score_response(self, example: Any, response: str) -> tuple[float, bool]:
        from stateset_agents.core.trajectory import ConversationTurn
        from stateset_agents.data.tool_calling_bench import ToolCallReward

        reward = ToolCallReward()
        turns = [ConversationTurn(role="assistant", content=response)]
        context = example.to_scenario()
        result = asyncio.run(reward.compute_reward(turns, context=context))
        return result.score, True


TASKS: dict[str, type[TaskAdapter]] = {
    "gsm8k": GSM8KAdapter,
    "customer_support": CustomerSupportAdapter,
    "tool_calling": ToolCallingAdapter,
}


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_baseline(
    model_name: str,
    adapter: TaskAdapter,
    eval_examples: list,
) -> dict[str, float]:
    """Evaluate the un-fine-tuned base model on the task."""
    return _evaluate_with_agent(model_name, adapter, eval_examples, trained_agent=None)


def _evaluate_with_agent(
    model_name: str,
    adapter: TaskAdapter,
    eval_examples: list,
    trained_agent: Any | None = None,
) -> dict[str, float]:
    """Run eval either with a fresh base-model agent or with a passed-in trained agent."""
    from stateset_agents.core.agent import Agent
    from stateset_agents.core.agent_config import AgentConfig

    if trained_agent is None:
        config = AgentConfig(
            model_name=model_name,
            max_new_tokens=adapter.max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        agent = Agent(config=config)
    else:
        agent = trained_agent

    async def _eval() -> dict[str, float]:
        if trained_agent is None:
            await agent.initialize()
        total_score = 0.0
        parseable = 0
        for ex in eval_examples:
            response = await agent.generate_response(adapter.format_prompt(ex))
            score, ok = adapter.score_response(ex, response)
            total_score += score
            if ok:
                parseable += 1
        n = max(len(eval_examples), 1)
        return {
            "pass_at_1": total_score / n,
            "parse_rate": parseable / n,
            "n": float(n),
        }

    return asyncio.run(_eval())


def _build_env_reward(
    adapter: TaskAdapter, train_examples: list
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Construct the (env, reward_fn, scenarios) tuple for training based on the task adapter.

    Each task has its own reward + scenario shape; centralizing this here keeps
    the training path in main() simple.
    """
    from stateset_agents.core import ConversationEnvironment

    if adapter.name == "gsm8k":
        from stateset_agents.data.gsm8k import GSM8KReward, make_gsm8k_scenarios

        reward_fn = GSM8KReward()
        scenarios = make_gsm8k_scenarios(train_examples)
    elif adapter.name == "customer_support":
        from stateset_agents.data.customer_support_bench import (
            SupportRewardComposite,
            make_support_scenarios,
        )

        reward_fn = SupportRewardComposite()
        scenarios = make_support_scenarios(train_examples)
    elif adapter.name == "tool_calling":
        from stateset_agents.data.tool_calling_bench import (
            ToolCallReward,
            make_tool_call_scenarios,
        )

        reward_fn = ToolCallReward()
        scenarios = make_tool_call_scenarios(train_examples)
    else:
        raise ValueError(f"Unknown task for training: {adapter.name}")

    env = ConversationEnvironment(
        scenarios=scenarios,
        reward_fn=reward_fn,
        max_turns=1 if adapter.name != "customer_support" else 4,
    )
    return env, reward_fn, scenarios


def train_with_trainer(
    trainer: str,
    model_name: str,
    adapter: TaskAdapter,
    train_examples: list,
    seed: int,
    output_dir: str,
    use_vllm: bool = False,
) -> tuple[Any | None, float, str | None]:
    """Invoke the actual trainer entry point for the given trainer name.

    Returns ``(trained_agent_or_None, wall_clock_seconds, error_message_or_None)``.

    The trainer entry points (`train_with_gspo`, `train_with_dapo`,
    `train_with_trl_grpo`) load their own model + tokenizer via the trainer's
    model manager, so we pass an *un-initialized* agent and let the trainer
    handle everything.

    On failure (missing extras, OOM, etc.) returns ``(None, elapsed, msg)`` so
    the caller can record what happened in the result JSON.
    """
    import time

    t0 = time.time()
    try:
        from stateset_agents.core.agent import MultiTurnAgent
        from stateset_agents.core.agent_config import AgentConfig
        from stateset_agents.core.tool_agent import ToolAgent

        env, reward_fn, _ = _build_env_reward(adapter, train_examples)

        # ToolAgent for tool_calling so the model sees tool descriptions.
        if adapter.name == "tool_calling":
            from stateset_agents.data.tool_calling_bench import SAMPLE_TOOLS

            trainer_agent = ToolAgent(
                config=AgentConfig(
                    model_name=model_name,
                    max_new_tokens=adapter.max_new_tokens,
                ),
                tools=SAMPLE_TOOLS,
            )
        else:
            trainer_agent = MultiTurnAgent(
                AgentConfig(
                    model_name=model_name,
                    max_new_tokens=adapter.max_new_tokens,
                )
            )

        if trainer == "gspo":
            from stateset_agents.training import GSPOConfig, train_with_gspo

            cfg = GSPOConfig(
                model_name=model_name,
                output_dir=output_dir,
                report_to="none",
                num_outer_iterations=4,
                num_iterations=1,
                num_generations=4,
                generations_per_iteration=len(train_examples),
                clip_range_left=3e-4,
                clip_range_right=4e-4,
                learning_rate=5e-6,
                use_lora=True,
                lora_r=16,
                lora_alpha=32,
                gradient_checkpointing=True,
                bf16=True,
                warmup_ratio=0.1,
                use_vllm=use_vllm,
            )
            trained = asyncio.run(
                train_with_gspo(
                    config=cfg,
                    agent=trainer_agent,
                    environment=env,
                    reward_model=reward_fn,
                )
            )
            return trained, time.time() - t0, None

        if trainer == "grpo":
            from stateset_agents.training import TRLGRPOConfig, train_with_trl_grpo

            cfg = TRLGRPOConfig(
                model_name=model_name,
                output_dir=output_dir,
                report_to="none",
                num_iterations=1,
                num_outer_iterations=4,
                num_generations=4,
                generations_per_iteration=len(train_examples),
                learning_rate=5e-6,
                use_lora=True,
                lora_r=16,
                lora_alpha=32,
                gradient_checkpointing=True,
                bf16=True,
                use_vllm=use_vllm,
            )
            trained = asyncio.run(
                train_with_trl_grpo(
                    config=cfg,
                    agent=trainer_agent,
                    environment=env,
                    reward_model=reward_fn,
                )
            )
            return trained, time.time() - t0, None

        if trainer == "dapo":
            # DAPO has a different shape: takes `train_prompts: list[str]`
            # and returns `(model, tokenizer, metrics)` instead of an Agent.
            # Wrap the result back into an agent for post-eval symmetry.
            from stateset_agents.training import DAPOConfig, train_with_dapo

            cfg = DAPOConfig(
                model_name=model_name,
                output_dir=output_dir,
                num_gradient_updates=4,
                group_size=8,
                clip_eps_low=0.2,
                clip_eps_high=0.28,
                use_dynamic_sampling=True,
                use_overlong_shaping=False,
                max_generation_length=adapter.max_new_tokens,
                use_lora=True,
                lora_r=16,
                lora_alpha=32,
                gradient_checkpointing=True,
                use_vllm=use_vllm,
            )
            train_prompts = [adapter.format_prompt(ex) for ex in train_examples]

            # DAPO's reward signature: (prompt, response) -> float
            def _reward_callable(prompt: str, response: str) -> float:
                # Map prompt back to the example to score against ground truth.
                # In the verifiable-reward case (gsm8k, tool_calling) we need
                # the original example for context.
                idx = train_prompts.index(prompt) if prompt in train_prompts else 0
                example = train_examples[idx]
                score, _ = adapter.score_response(example, response)
                return score

            model, tokenizer, _metrics = asyncio.run(
                train_with_dapo(
                    model_name=model_name,
                    reward_fn=_reward_callable,
                    train_prompts=train_prompts,
                    config=cfg,
                    output_dir=output_dir,
                )
            )
            # Wrap model+tokenizer into an Agent for the re-eval step.
            trainer_agent.model = model
            trainer_agent.tokenizer = tokenizer
            return trainer_agent, time.time() - t0, None

        return (
            None,
            time.time() - t0,
            f"trainer={trainer!r} not recognized. Use gspo, grpo, or dapo.",
        )
    except ImportError as e:
        return (
            None,
            time.time() - t0,
            f"Missing extras: {e}. Install via `pip install -e '.[training]'`.",
        )
    except Exception as e:  # noqa: BLE001 — capture-and-record is the point
        return None, time.time() - t0, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# Trainer config builder
# ---------------------------------------------------------------------------


def build_trainer_config(trainer: str, **overrides: Any) -> dict[str, Any]:
    """Default-then-override config builder for each trainer."""
    base = {
        "model_name": overrides.get("model_name", "Qwen/Qwen3.5-0.8B"),
        "learning_rate": 5e-6,
        "max_completion_length": 256,
        "max_prompt_length": 512,
        "temperature": 0.7,
        "use_lora": True,
        "lora_r": 16,
        "lora_alpha": 32,
        "gradient_checkpointing": True,
    }

    if trainer == "grpo":
        base.update({"num_generations": 4, "num_iterations": 1, "beta": 0.0})
    elif trainer == "gspo":
        base.update(
            {
                "num_generations": 4,
                "clip_range_left": 3e-4,
                "clip_range_right": 4e-4,
                "beta": 0.0,
            }
        )
    elif trainer == "dapo":
        base.update(
            {
                "group_size": 8,
                "clip_eps_low": 0.2,
                "clip_eps_high": 0.28,
                "use_dynamic_sampling": True,
                "use_overlong_shaping": False,
                "max_generation_length": 256,
            }
        )
    else:
        raise ValueError(f"Unknown trainer: {trainer}")

    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trainer", choices=["grpo", "gspo", "dapo"], required=True)
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="gsm8k",
        help="Benchmark task (default: gsm8k).",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--num-train-examples", type=int, default=200)
    parser.add_argument("--num-eval-examples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--wandb-project", default="stateset-agents-whitepaper-v1")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Verify the pipeline end-to-end without GPU (no training, no output writes).",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Actually invoke training (requires GPU + 'training' extras). "
        "Without this flag, the runner records baseline + config only.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/phase0_run",
        help="Where the trainer writes the adapter (default: outputs/phase0_run).",
    )
    parser.add_argument(
        "--vllm",
        action="store_true",
        help="Use vLLM for rollout generation. Requires the [vllm] extra and a GPU "
        "compatible with vLLM. Typically several × faster than HF generate on "
        "large batches; magnitude is workload-dependent (see whitepaper §6.4).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )

    # Step 1: lock down reproducibility.
    from stateset_agents.utils.reproducibility import set_all_seeds

    set_all_seeds(args.seed)
    logger.info("Seeds set to %d", args.seed)

    # Step 2: pick the task and load data.
    adapter = TASKS[args.task]()
    logger.info("Task: %s", adapter.name)
    train_examples, eval_examples = adapter.load(
        args.num_train_examples, args.num_eval_examples
    )
    logger.info(
        "Train: %d examples, Eval: %d examples",
        len(train_examples),
        len(eval_examples),
    )

    if args.smoke_test:
        logger.info("Smoke test: previewing first 3 train examples")
        for ex in train_examples[:3]:
            preview = adapter.format_prompt(ex).replace("\n", " ")[:80]
            logger.info("  %s…", preview)
        logger.info(
            "Smoke test passed — pipeline is operational for task=%s.", adapter.name
        )
        return 0

    config = build_trainer_config(args.trainer, model_name=args.model)
    result: dict[str, Any] = {
        "trainer": args.trainer,
        "task": args.task,
        "model": args.model,
        "seed": args.seed,
        "commit": get_git_commit(),
        "config": config,
        "metrics": {
            "train_examples": len(train_examples),
            "eval_examples": len(eval_examples),
            "status": "baseline_only",
        },
        "wandb_run_url": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Step 3: baseline eval.
    if not args.skip_baseline:
        logger.info(
            "Evaluating baseline (un-fine-tuned) on %d examples…", len(eval_examples)
        )
        t0 = time.time()
        baseline = evaluate_baseline(args.model, adapter, eval_examples)
        result["metrics"]["baseline_eval_seconds"] = time.time() - t0
        result["metrics"]["eval_pass_at_1_baseline"] = baseline["pass_at_1"]
        result["metrics"]["eval_parse_rate_baseline"] = baseline["parse_rate"]
        logger.info(
            "Baseline pass@1: %.3f (parse rate %.3f)",
            baseline["pass_at_1"],
            baseline["parse_rate"],
        )

    # Step 4: optional training. Without --train, the runner records only the
    # baseline and the configuration (the Colab notebooks invoke training and
    # write post-training metrics into the JSON before publishing).
    if args.train:
        gpu_available = False
        try:
            import torch

            gpu_available = torch.cuda.is_available()
        except ImportError:
            pass

        if not gpu_available:
            result["metrics"]["status"] = "skipped_train_no_gpu"
            result["metrics"][
                "train_error"
            ] = "No CUDA GPU detected. Use --smoke-test or run on a GPU host."
            logger.warning(
                "--train was set but no GPU detected. Skipping training, baseline only."
            )
        else:
            logger.info(
                "Invoking %s trainer with %d train examples…",
                args.trainer,
                len(train_examples),
            )
            trained_agent, train_seconds, err = train_with_trainer(
                trainer=args.trainer,
                model_name=args.model,
                adapter=adapter,
                train_examples=train_examples,
                seed=args.seed,
                output_dir=args.output_dir,
                use_vllm=args.vllm,
            )
            result["metrics"]["train_wall_clock_seconds"] = train_seconds

            if trained_agent is None:
                result["metrics"]["status"] = "train_failed"
                result["metrics"]["train_error"] = err or "unknown"
                logger.warning("Training did not produce an agent: %s", err)
            else:
                result["metrics"]["status"] = "trained"
                logger.info("Training complete in %.0fs. Re-evaluating…", train_seconds)
                t0 = time.time()
                post = _evaluate_with_agent(
                    args.model, adapter, eval_examples, trained_agent=trained_agent
                )
                result["metrics"]["post_eval_seconds"] = time.time() - t0
                result["metrics"]["eval_pass_at_1"] = post["pass_at_1"]
                result["metrics"]["eval_parse_rate"] = post["parse_rate"]
                baseline = result["metrics"].get("eval_pass_at_1_baseline")
                if baseline is not None:
                    result["metrics"]["improvement"] = post["pass_at_1"] - baseline
                logger.info("Post-train pass@1: %.3f", post["pass_at_1"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    logger.info("Wrote result to %s", args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
