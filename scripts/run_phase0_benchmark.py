"""
Phase 0 benchmark runner — produces empirical evidence for the whitepaper.

Runs a fixed (task, model, trainer, seed) configuration and emits a single
JSON result that fits the canonical benchmark schema. Designed to be invoked
from Colab, CI, or a local A100/H100.

Three tasks supported:

* ``gsm8k`` — single-turn, verifiable math reasoning. The cheapest path to
  publishable numbers.
* ``customer_support`` — multi-turn dialogue with composite rule-based reward.
  This is the framework's differentiator over TRL.
* ``tool_calling`` — structured tool-selection and argument correctness.

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
import hashlib
import json
import logging
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("phase0_benchmark")


def canonical_config_digest(config: dict[str, Any]) -> str:
    """Return the shootout protocol's canonical configuration digest."""
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def get_git_commit() -> str:
    """Return current full git HEAD SHA, or 'unknown'."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
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

    def load(
        self, n_train: int, n_eval: int, dataset_revision: str | None = None
    ) -> tuple[list, list]:
        raise NotImplementedError

    def load_smoke(self) -> tuple[list, list]:
        """Return a tiny sample suitable for an offline pipeline smoke test."""
        return self.load(3, 1)

    def format_prompt(self, example: Any) -> str:
        raise NotImplementedError

    def score_response(self, example: Any, response: str) -> tuple[float, bool]:
        """Return (score, parseable_bool)."""
        raise NotImplementedError


class GSM8KAdapter(TaskAdapter):
    name = "gsm8k"
    max_new_tokens = 256

    def load(
        self, n_train: int, n_eval: int, dataset_revision: str | None = None
    ) -> tuple[list, list]:
        from stateset_agents.data.gsm8k import load_gsm8k

        train, test = load_gsm8k(limit=max(n_train, n_eval), revision=dataset_revision)
        return train[:n_train], test[:n_eval]

    def load_smoke(self) -> tuple[list, list]:
        """Return deterministic examples without contacting Hugging Face."""
        from stateset_agents.data.gsm8k import GSM8KExample

        examples = [
            GSM8KExample(
                question="A box has 3 rows of 4 apples. How many apples are there?",
                answer_text="Three groups of four make twelve. #### 12",
                gold_answer=12.0,
            ),
            GSM8KExample(
                question="Mia has 10 marbles and gives away 3. How many remain?",
                answer_text="Ten minus three is seven. #### 7",
                gold_answer=7.0,
            ),
            GSM8KExample(
                question="A book costs $6. What do two books cost?",
                answer_text="Two times six is twelve. #### 12",
                gold_answer=12.0,
            ),
        ]
        return examples, examples[:1]

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

    def load(
        self, n_train: int, n_eval: int, dataset_revision: str | None = None
    ) -> tuple[list, list]:
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

    def load(
        self, n_train: int, n_eval: int, dataset_revision: str | None = None
    ) -> tuple[list, list]:
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
    model_revision: str | None = None,
) -> dict[str, float]:
    """Evaluate the un-fine-tuned base model on the task."""
    return _evaluate_with_agent(
        model_name,
        adapter,
        eval_examples,
        trained_agent=None,
        model_revision=model_revision,
    )


def evaluate_shootout_baseline(
    model_name: str,
    model_revision: str,
    adapter: TaskAdapter,
    eval_examples: list[Any],
) -> dict[str, float]:
    """Evaluate the base model with the framework-neutral shootout protocol."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from stateset_agents.evaluation.framework_protocol import evaluate_causal_lm

    tokenizer = AutoTokenizer.from_pretrained(
        model_name, revision=model_revision, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, revision=model_revision, torch_dtype=torch.bfloat16
    )
    model.to("cuda")
    return evaluate_causal_lm(
        model,
        tokenizer,
        eval_examples,
        format_prompt=adapter.format_prompt,
        score_response=adapter.score_response,
        max_tokens=adapter.max_new_tokens,
    )


def _evaluate_with_agent(
    model_name: str,
    adapter: TaskAdapter,
    eval_examples: list,
    trained_agent: Any | None = None,
    model_revision: str | None = None,
) -> dict[str, float]:
    """Run eval either with a fresh base-model agent or with a passed-in trained agent."""
    from stateset_agents.core.agent import MultiTurnAgent
    from stateset_agents.core.agent_config import AgentConfig

    if trained_agent is None:
        config = AgentConfig(
            model_name=model_name,
            model_revision=model_revision,
            max_new_tokens=adapter.max_new_tokens,
            temperature=0.0,
            do_sample=False,
        )
        agent = MultiTurnAgent(config=config)
    else:
        agent = trained_agent

    async def _eval() -> dict[str, float]:
        if trained_agent is None:
            await agent.initialize()
        original_temperature = agent.config.temperature
        original_do_sample = agent.config.do_sample
        agent.config.temperature = 0.0
        agent.config.do_sample = False
        total_score = 0.0
        parseable = 0
        try:
            for ex in eval_examples:
                response = await agent.generate_response(adapter.format_prompt(ex))
                score, ok = adapter.score_response(ex, response)
                total_score += score
                if ok:
                    parseable += 1
        finally:
            agent.config.temperature = original_temperature
            agent.config.do_sample = original_do_sample
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


def build_algorithm_config(
    trainer: str, protocol_config: dict[str, Any]
) -> dict[str, Any]:
    """Resolve the algorithm-specific knobs applied to a shared protocol.

    The shared shootout protocol fixes the training budget, optimizer, model,
    generation, and LoRA settings.  This attestation records the objective
    details that intentionally differ between algorithms, so an algorithm
    comparison never claims that an opaque set of defaults was used.
    """
    shared = {
        "max_steps": int(protocol_config.get("max_steps", 4)),
        "num_generations": int(protocol_config.get("num_generations", 4)),
        "num_iterations": int(protocol_config.get("num_iterations", 1)),
        "per_device_train_batch_size": int(
            protocol_config.get("per_device_train_batch_size", 4)
        ),
        "gradient_accumulation_steps": int(
            protocol_config.get("gradient_accumulation_steps", 1)
        ),
        "learning_rate": float(protocol_config.get("learning_rate", 5e-6)),
        "max_prompt_length": int(protocol_config.get("max_prompt_length", 512)),
        "max_completion_length": int(protocol_config.get("max_completion_length", 256)),
        "temperature": float(protocol_config.get("temperature", 0.7)),
        "top_p": float(protocol_config.get("top_p", 0.9)),
        "beta": float(protocol_config.get("beta", 0.0)),
    }
    if trainer == "grpo":
        return {**shared, "objective": "trl-grpo", "clip_ratio": 0.2}
    if trainer == "gspo":
        return {
            **shared,
            "objective": "gspo-sequence",
            "clip_range_left": 3e-4,
            "clip_range_right": 4e-4,
        }
    if trainer == "dapo":
        return {
            **shared,
            "objective": "dapo-token",
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.28,
            "use_dynamic_sampling": True,
            "use_overlong_shaping": False,
            "use_token_level_loss": True,
        }
    if trainer == "vapo":
        return {
            **shared,
            "objective": "vapo-value-augmented",
            "clip_eps_low": 0.2,
            "clip_eps_high": 0.28,
            "value_warmup_steps": 1,
            "critic_learning_rate": 2.0 * shared["learning_rate"],
            "use_token_level_loss": True,
            "use_positive_lm_loss": True,
        }
    if trainer == "gepo":
        return {
            **shared,
            "objective": "gepo-group-expectation",
            "clip_eps": 0.2,
            "use_group_baseline": True,
            "use_reference_model": False,
        }
    raise ValueError(f"Unknown trainer: {trainer}")


def _prompt_reward(
    adapter: TaskAdapter, train_examples: list, train_prompts: list[str]
) -> Any:
    """Build the prompt/response reward callback used by native trainers."""
    if not train_examples:
        raise ValueError("training requires at least one example")
    prompt_examples = dict(zip(train_prompts, train_examples, strict=True))

    def reward(prompt: str, response: str) -> float:
        example = prompt_examples.get(prompt, train_examples[0])
        score, _ = adapter.score_response(example, response)
        return score

    return reward


def _attach_phase0_metadata(
    agent: Any, metrics: dict[str, list[float]], algorithm_config: dict[str, Any]
) -> Any:
    """Attach measured trainer metadata without changing the public Agent API."""
    samples = metrics.get("rollout_samples_total", [])
    if samples:
        agent._phase0_samples_processed = int(samples[-1])
    agent._phase0_algorithm_config = algorithm_config
    if getattr(agent, "generation_config", None) is None and callable(
        getattr(agent, "_build_generation_config", None)
    ):
        agent.generation_config = agent._build_generation_config()
    return agent


def save_normalized_policy_artifact(agent: Any, artifact_path: Path) -> None:
    """Save the trained policy and tokenizer under one benchmark contract.

    Native trainers retain their richer checkpoints (for example VAPO's value
    head) in their own ``final`` directory.  Comparison evidence hashes this
    normalized policy artifact so every algorithm exposes the same portable
    policy/tokenizer surface.
    """
    model = getattr(agent, "model", None)
    tokenizer = getattr(agent, "tokenizer", None)
    if model is None or not callable(getattr(model, "save_pretrained", None)):
        raise RuntimeError("trained agent has no save_pretrained-capable model")
    if tokenizer is None or not callable(getattr(tokenizer, "save_pretrained", None)):
        raise RuntimeError("trained agent has no save_pretrained-capable tokenizer")
    artifact_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(artifact_path)
    tokenizer.save_pretrained(artifact_path)
    if not any(path.is_file() for path in artifact_path.rglob("*")):
        raise RuntimeError(f"normalized policy artifact is empty: {artifact_path}")


def train_with_trainer(
    trainer: str,
    model_name: str,
    adapter: TaskAdapter,
    train_examples: list,
    seed: int,
    output_dir: str,
    use_vllm: bool = False,
    model_revision: str | None = None,
    protocol_config: dict[str, Any] | None = None,
) -> tuple[Any | None, float, str | None]:
    """Invoke the actual trainer entry point for the given trainer name.

    Returns ``(trained_agent_or_None, wall_clock_seconds, error_message_or_None)``.

    The trainer entry points load their own model + tokenizer via the trainer's
    model manager, so we pass an *un-initialized* agent and let the trainer
    handle everything.

    On failure (missing extras, OOM, etc.) returns ``(None, elapsed, msg)`` so
    the caller can record what happened in the result JSON.
    """
    import time

    t0 = time.time()
    protocol = protocol_config or {}
    if trainer not in {"grpo", "gspo", "dapo", "vapo", "gepo"}:
        return (
            None,
            time.time() - t0,
            f"trainer={trainer!r} not recognized. Use grpo, gspo, dapo, vapo, or gepo.",
        )
    try:
        algorithm_config = build_algorithm_config(trainer, protocol)
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
                    model_revision=model_revision,
                    max_new_tokens=adapter.max_new_tokens,
                ),
                tools=SAMPLE_TOOLS,
            )
        else:
            trainer_agent = MultiTurnAgent(
                AgentConfig(
                    model_name=model_name,
                    model_revision=model_revision,
                    max_new_tokens=adapter.max_new_tokens,
                )
            )

        if trainer == "gspo":
            from stateset_agents.training import GSPOConfig, train_with_gspo

            cfg = GSPOConfig(
                model_name=model_name,
                model_revision=model_revision,
                output_dir=output_dir,
                report_to="none",
                num_outer_iterations=algorithm_config["max_steps"],
                num_iterations=algorithm_config["num_iterations"],
                num_generations=algorithm_config["num_generations"],
                generations_per_iteration=(
                    algorithm_config["per_device_train_batch_size"]
                    * algorithm_config["gradient_accumulation_steps"]
                ),
                mini_batch_size=algorithm_config["per_device_train_batch_size"],
                per_device_train_batch_size=algorithm_config[
                    "per_device_train_batch_size"
                ],
                gradient_accumulation_steps=algorithm_config[
                    "gradient_accumulation_steps"
                ],
                clip_range_left=algorithm_config["clip_range_left"],
                clip_range_right=algorithm_config["clip_range_right"],
                learning_rate=algorithm_config["learning_rate"],
                adam_beta1=float(protocol.get("adam_beta1", 0.9)),
                adam_beta2=float(protocol.get("adam_beta2", 0.99)),
                weight_decay=float(protocol.get("weight_decay", 0.01)),
                max_grad_norm=float(protocol.get("max_grad_norm", 1.0)),
                warmup_ratio=float(protocol.get("warmup_ratio", 0.1)),
                lr_scheduler_type=str(protocol.get("lr_scheduler_type", "cosine")),
                num_epochs=int(protocol.get("num_train_epochs", 1)),
                max_prompt_length=algorithm_config["max_prompt_length"],
                max_completion_length=algorithm_config["max_completion_length"],
                temperature=algorithm_config["temperature"],
                top_p=algorithm_config["top_p"],
                beta=algorithm_config["beta"],
                use_lora=bool(protocol.get("use_lora", True)),
                lora_r=int(protocol.get("lora_r", 16)),
                lora_alpha=int(protocol.get("lora_alpha", 32)),
                lora_dropout=float(protocol.get("lora_dropout", 0.05)),
                lora_target_modules=protocol.get("lora_target_modules"),
                gradient_checkpointing=bool(
                    protocol.get("gradient_checkpointing", True)
                ),
                bf16=bool(protocol.get("bf16", True)),
                seed=seed,
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
            trained._phase0_algorithm_config = algorithm_config
            trained._phase0_samples_processed = (
                algorithm_config["max_steps"]
                * min(
                    len(train_examples),
                    algorithm_config["per_device_train_batch_size"]
                    * algorithm_config["gradient_accumulation_steps"],
                )
                * algorithm_config["num_generations"]
            )
            return trained, time.time() - t0, None

        if trainer == "grpo":
            from stateset_agents.training import TRLGRPOConfig, train_with_trl_grpo

            cfg = TRLGRPOConfig(
                model_name=model_name,
                model_revision=model_revision,
                output_dir=output_dir,
                report_to="none",
                num_iterations=int(protocol.get("num_iterations", 1)),
                num_outer_iterations=int(protocol.get("max_steps", 4)),
                max_steps=int(protocol.get("max_steps", 4)),
                num_generations=int(protocol.get("num_generations", 4)),
                generations_per_iteration=len(train_examples),
                learning_rate=float(protocol.get("learning_rate", 5e-6)),
                adam_beta1=float(protocol.get("adam_beta1", 0.9)),
                adam_beta2=float(protocol.get("adam_beta2", 0.99)),
                weight_decay=float(protocol.get("weight_decay", 0.01)),
                max_grad_norm=float(protocol.get("max_grad_norm", 1.0)),
                warmup_ratio=float(protocol.get("warmup_ratio", 0.1)),
                lr_scheduler_type=str(protocol.get("lr_scheduler_type", "cosine")),
                num_epochs=int(protocol.get("num_train_epochs", 1)),
                use_lora=True,
                lora_r=int(protocol.get("lora_r", 16)),
                lora_alpha=int(protocol.get("lora_alpha", 32)),
                lora_dropout=float(protocol.get("lora_dropout", 0.05)),
                lora_target_modules=protocol.get("lora_target_modules"),
                per_device_train_batch_size=int(
                    protocol.get("per_device_train_batch_size", 4)
                ),
                gradient_accumulation_steps=int(
                    protocol.get("gradient_accumulation_steps", 1)
                ),
                max_prompt_length=int(protocol.get("max_prompt_length", 512)),
                max_completion_length=int(protocol.get("max_completion_length", 256)),
                temperature=float(protocol.get("temperature", 0.7)),
                top_p=float(protocol.get("top_p", 0.9)),
                beta=float(protocol.get("beta", 0.0)),
                seed=seed,
                gradient_checkpointing=bool(
                    protocol.get("gradient_checkpointing", True)
                ),
                bf16=bool(protocol.get("bf16", True)),
                use_vllm=use_vllm,
            )
            trained = asyncio.run(
                train_with_trl_grpo(
                    config=cfg,
                    agent=trainer_agent,
                    environment=env,
                    reward_model=reward_fn,
                    train_data=[
                        {
                            "prompt": adapter.format_prompt(example),
                            "scenario_index": index,
                        }
                        for index, example in enumerate(train_examples)
                    ],
                )
            )
            trained._phase0_algorithm_config = algorithm_config
            trained._phase0_samples_processed = (
                algorithm_config["max_steps"]
                * algorithm_config["per_device_train_batch_size"]
                * algorithm_config["gradient_accumulation_steps"]
                * algorithm_config["num_generations"]
            )
            return trained, time.time() - t0, None

        if trainer == "dapo":
            # DAPO has a different shape: takes `train_prompts: list[str]`
            # and returns `(model, tokenizer, metrics)` instead of an Agent.
            # Wrap the result back into an agent for post-eval symmetry.
            from stateset_agents.training import DAPOConfig, train_with_dapo

            cfg = DAPOConfig(
                model_name=model_name,
                model_revision=model_revision,
                output_dir=output_dir,
                num_episodes=algorithm_config["max_steps"],
                num_epochs=int(protocol.get("num_train_epochs", 1)),
                num_gradient_updates=algorithm_config["num_iterations"],
                group_size=algorithm_config["num_generations"],
                num_generations=algorithm_config["num_generations"],
                per_device_train_batch_size=algorithm_config[
                    "per_device_train_batch_size"
                ],
                gradient_accumulation_steps=algorithm_config[
                    "gradient_accumulation_steps"
                ],
                mini_batch_size=algorithm_config["per_device_train_batch_size"],
                learning_rate=algorithm_config["learning_rate"],
                adam_beta1=float(protocol.get("adam_beta1", 0.9)),
                adam_beta2=float(protocol.get("adam_beta2", 0.99)),
                weight_decay=float(protocol.get("weight_decay", 0.01)),
                max_grad_norm=float(protocol.get("max_grad_norm", 1.0)),
                max_prompt_length=algorithm_config["max_prompt_length"],
                max_completion_length=algorithm_config["max_completion_length"],
                max_generation_length=algorithm_config["max_completion_length"],
                temperature=algorithm_config["temperature"],
                top_p=algorithm_config["top_p"],
                beta=algorithm_config["beta"],
                clip_eps_low=algorithm_config["clip_eps_low"],
                clip_eps_high=algorithm_config["clip_eps_high"],
                use_dynamic_sampling=algorithm_config["use_dynamic_sampling"],
                use_overlong_shaping=algorithm_config["use_overlong_shaping"],
                use_token_level_loss=algorithm_config["use_token_level_loss"],
                use_lora=bool(protocol.get("use_lora", True)),
                lora_r=int(protocol.get("lora_r", 16)),
                lora_alpha=int(protocol.get("lora_alpha", 32)),
                lora_dropout=float(protocol.get("lora_dropout", 0.05)),
                lora_target_modules=protocol.get("lora_target_modules"),
                gradient_checkpointing=bool(
                    protocol.get("gradient_checkpointing", True)
                ),
                bf16=bool(protocol.get("bf16", True)),
                seed=seed,
                use_vllm=use_vllm,
            )
            train_prompts = [adapter.format_prompt(ex) for ex in train_examples]
            reward_callable = _prompt_reward(adapter, train_examples, train_prompts)
            model, tokenizer, metrics = asyncio.run(
                train_with_dapo(
                    model_name=model_name,
                    reward_fn=reward_callable,
                    train_prompts=train_prompts,
                    config=cfg,
                    output_dir=output_dir,
                    verifier_fn=lambda prompt, response: reward_callable(
                        prompt, response
                    )
                    > 0.5,
                )
            )
            # Wrap model+tokenizer into an Agent for the re-eval step.
            trainer_agent.model = model
            trainer_agent.tokenizer = tokenizer
            return (
                _attach_phase0_metadata(trainer_agent, metrics, algorithm_config),
                time.time() - t0,
                None,
            )

        if trainer == "vapo":
            if use_vllm:
                raise ValueError("VAPO does not currently implement vLLM rollouts")
            from stateset_agents.training import VAPOConfig, train_with_vapo

            cfg = VAPOConfig(
                model_name=model_name,
                model_revision=model_revision,
                output_dir=output_dir,
                # The first VAPO call performs value warmup only. Add one
                # episode so max_steps still denotes policy-update steps.
                num_episodes=algorithm_config["max_steps"] + 1,
                num_epochs=int(protocol.get("num_train_epochs", 1)),
                num_gradient_updates=algorithm_config["num_iterations"],
                group_size=algorithm_config["num_generations"],
                num_generations=algorithm_config["num_generations"],
                per_device_train_batch_size=algorithm_config[
                    "per_device_train_batch_size"
                ],
                gradient_accumulation_steps=algorithm_config[
                    "gradient_accumulation_steps"
                ],
                actor_learning_rate=algorithm_config["learning_rate"],
                critic_learning_rate=algorithm_config["critic_learning_rate"],
                learning_rate=algorithm_config["learning_rate"],
                adam_beta1=float(protocol.get("adam_beta1", 0.9)),
                adam_beta2=float(protocol.get("adam_beta2", 0.99)),
                weight_decay=float(protocol.get("weight_decay", 0.01)),
                max_grad_norm=float(protocol.get("max_grad_norm", 1.0)),
                warmup_ratio=float(protocol.get("warmup_ratio", 0.1)),
                max_prompt_length=algorithm_config["max_prompt_length"],
                max_completion_length=algorithm_config["max_completion_length"],
                temperature=algorithm_config["temperature"],
                top_p=algorithm_config["top_p"],
                value_warmup_steps=algorithm_config["value_warmup_steps"],
                clip_eps_low=algorithm_config["clip_eps_low"],
                clip_eps_high=algorithm_config["clip_eps_high"],
                use_token_level_loss=algorithm_config["use_token_level_loss"],
                use_positive_lm_loss=algorithm_config["use_positive_lm_loss"],
                use_lora=bool(protocol.get("use_lora", True)),
                lora_r=int(protocol.get("lora_r", 16)),
                lora_alpha=int(protocol.get("lora_alpha", 32)),
                lora_dropout=float(protocol.get("lora_dropout", 0.05)),
                lora_target_modules=protocol.get("lora_target_modules"),
                gradient_checkpointing=bool(
                    protocol.get("gradient_checkpointing", True)
                ),
                bf16=bool(protocol.get("bf16", True)),
                seed=seed,
            )
            train_prompts = [adapter.format_prompt(ex) for ex in train_examples]
            reward_callable = _prompt_reward(adapter, train_examples, train_prompts)
            model, tokenizer, metrics = asyncio.run(
                train_with_vapo(
                    model_name=model_name,
                    reward_fn=reward_callable,
                    train_prompts=train_prompts,
                    config=cfg,
                    output_dir=output_dir,
                    verifier_fn=lambda prompt, response: reward_callable(
                        prompt, response
                    )
                    > 0.5,
                )
            )
            trainer_agent.model = model
            trainer_agent.tokenizer = tokenizer
            return (
                _attach_phase0_metadata(trainer_agent, metrics, algorithm_config),
                time.time() - t0,
                None,
            )

        if trainer == "gepo":
            if use_vllm:
                raise ValueError("GEPO does not currently implement vLLM rollouts")
            from stateset_agents.training import GEPOConfig, train_with_gepo

            cfg = GEPOConfig(
                model_name=model_name,
                model_revision=model_revision,
                output_dir=output_dir,
                num_episodes=algorithm_config["max_steps"],
                num_epochs=int(protocol.get("num_train_epochs", 1)),
                num_gradient_updates=algorithm_config["num_iterations"],
                group_size=algorithm_config["num_generations"],
                num_generations=algorithm_config["num_generations"],
                per_device_train_batch_size=algorithm_config[
                    "per_device_train_batch_size"
                ],
                gradient_accumulation_steps=algorithm_config[
                    "gradient_accumulation_steps"
                ],
                learning_rate=algorithm_config["learning_rate"],
                adam_beta1=float(protocol.get("adam_beta1", 0.9)),
                adam_beta2=float(protocol.get("adam_beta2", 0.99)),
                weight_decay=float(protocol.get("weight_decay", 0.01)),
                max_grad_norm=float(protocol.get("max_grad_norm", 1.0)),
                warmup_ratio=float(protocol.get("warmup_ratio", 0.1)),
                max_prompt_length=algorithm_config["max_prompt_length"],
                max_completion_length=algorithm_config["max_completion_length"],
                temperature=algorithm_config["temperature"],
                top_p=algorithm_config["top_p"],
                beta=algorithm_config["beta"],
                clip_eps=algorithm_config["clip_eps"],
                use_group_baseline=algorithm_config["use_group_baseline"],
                use_reference_model=algorithm_config["use_reference_model"],
                use_lora=bool(protocol.get("use_lora", True)),
                lora_r=int(protocol.get("lora_r", 16)),
                lora_alpha=int(protocol.get("lora_alpha", 32)),
                lora_dropout=float(protocol.get("lora_dropout", 0.05)),
                lora_target_modules=protocol.get("lora_target_modules"),
                gradient_checkpointing=bool(
                    protocol.get("gradient_checkpointing", True)
                ),
                bf16=bool(protocol.get("bf16", True)),
                seed=seed,
            )
            train_prompts = [adapter.format_prompt(ex) for ex in train_examples]
            model, tokenizer, metrics = asyncio.run(
                train_with_gepo(
                    model_name=model_name,
                    reward_fn=_prompt_reward(adapter, train_examples, train_prompts),
                    train_prompts=train_prompts,
                    config=cfg,
                    output_dir=output_dir,
                )
            )
            trainer_agent.model = model
            trainer_agent.tokenizer = tokenizer
            return (
                _attach_phase0_metadata(trainer_agent, metrics, algorithm_config),
                time.time() - t0,
                None,
            )

        return (
            None,
            time.time() - t0,
            f"trainer={trainer!r} not recognized. Use grpo, gspo, dapo, vapo, or gepo.",
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
    elif trainer == "vapo":
        base.update(
            {
                "group_size": 4,
                "value_warmup_steps": 1,
                "clip_eps_low": 0.2,
                "clip_eps_high": 0.28,
                "actor_learning_rate": 5e-6,
                "critic_learning_rate": 1e-5,
            }
        )
    elif trainer == "gepo":
        base.update(
            {
                "group_size": 4,
                "clip_eps": 0.2,
                "use_group_baseline": True,
                "use_reference_model": False,
            }
        )
    else:
        raise ValueError(f"Unknown trainer: {trainer}")

    base.update(overrides)
    return base


def estimate_rollout_samples(
    trainer: str, train_examples: int, config: dict[str, Any]
) -> int:
    """Return generated completions for the fixed Phase-0 training protocol."""
    prompts = min(train_examples, 10)
    if trainer in {"gspo", "grpo"}:
        return 4 * prompts * int(config.get("num_generations", 4))
    if trainer in {"dapo", "gepo"}:
        return prompts * int(config.get("group_size", 8))
    if trainer == "vapo":
        policy = prompts * int(config.get("group_size", 4))
        warmup = int(config.get("value_warmup_steps", 1)) * int(
            config.get("group_size", 4)
        )
        return policy + warmup
    raise ValueError(f"Unknown trainer: {trainer}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trainer",
        choices=["grpo", "gspo", "dapo", "vapo", "gepo"],
        required=True,
    )
    parser.add_argument(
        "--task",
        choices=sorted(TASKS),
        default="gsm8k",
        help="Benchmark task (default: gsm8k).",
    )
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument(
        "--model-revision",
        default=None,
        help="Pinned Hugging Face tag/commit. A full commit SHA is required when "
        "--adapter-output is used.",
    )
    parser.add_argument(
        "--dataset-revision",
        default=None,
        help="Pinned Hugging Face dataset revision (required for shootout evidence).",
    )
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
    parser.add_argument(
        "--adapter-output",
        type=Path,
        default=None,
        help="Write the neutral measured result consumed by benchmarks/shootout.py. "
        "Requires --train, baseline evaluation, and a pinned model revision.",
    )
    parser.add_argument(
        "--shootout-config-json",
        default=None,
        help="Canonical manifest config applied and attested by shootout adapters.",
    )
    args = parser.parse_args()

    if args.adapter_output is not None:
        if not args.train or args.skip_baseline:
            parser.error(
                "--adapter-output requires --train and forbids --skip-baseline"
            )
        if args.model_revision is None or len(args.model_revision) != 40:
            parser.error(
                "--adapter-output requires a full 40-character --model-revision"
            )
        if args.dataset_revision is None or len(args.dataset_revision) != 40:
            parser.error(
                "--adapter-output requires a full 40-character --dataset-revision"
            )
        if args.shootout_config_json is None:
            parser.error("--adapter-output requires --shootout-config-json")

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
    if args.smoke_test:
        train_examples, eval_examples = adapter.load_smoke()
    else:
        train_examples, eval_examples = adapter.load(
            args.num_train_examples,
            args.num_eval_examples,
            dataset_revision=args.dataset_revision,
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

    config = build_trainer_config(
        args.trainer,
        model_name=args.model,
        model_revision=args.model_revision,
    )
    if args.shootout_config_json is not None:
        try:
            shootout_config = json.loads(args.shootout_config_json)
        except json.JSONDecodeError as exc:
            parser.error(f"--shootout-config-json is invalid JSON: {exc}")
        required = {
            "num_train_examples",
            "num_eval_examples",
            "max_steps",
            "per_device_train_batch_size",
            "gradient_accumulation_steps",
            "learning_rate",
            "adam_beta1",
            "adam_beta2",
            "weight_decay",
            "max_grad_norm",
            "warmup_ratio",
            "lr_scheduler_type",
            "num_train_epochs",
            "num_generations",
            "num_iterations",
            "max_prompt_length",
            "max_completion_length",
            "temperature",
            "top_p",
            "beta",
            "lora_r",
            "lora_alpha",
            "lora_dropout",
            "lora_target_modules",
            "gradient_checkpointing",
            "bf16",
        }
        if not isinstance(shootout_config, dict) or set(shootout_config) != required:
            parser.error("--shootout-config-json has an unsupported schema")
        if (
            args.num_train_examples != shootout_config["num_train_examples"]
            or args.num_eval_examples != shootout_config["num_eval_examples"]
        ):
            parser.error("shootout example counts do not match command arguments")
        config = shootout_config
    algorithm_config = build_algorithm_config(args.trainer, config)
    result: dict[str, Any] = {
        "trainer": args.trainer,
        "task": args.task,
        "model": args.model,
        "model_revision": args.model_revision,
        "dataset_revision": args.dataset_revision,
        "seed": args.seed,
        "commit": get_git_commit(),
        "evidence_class": "measured",
        "config": config,
        "algorithm_config": algorithm_config,
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
        if args.adapter_output is not None:
            baseline = evaluate_shootout_baseline(
                args.model, args.model_revision, adapter, eval_examples
            )
        else:
            baseline = evaluate_baseline(
                args.model,
                adapter,
                eval_examples,
                model_revision=args.model_revision,
            )
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
            try:
                import torch

                torch.cuda.reset_peak_memory_stats()
            except (ImportError, RuntimeError):
                pass
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
                model_revision=args.model_revision,
                protocol_config=(
                    shootout_config if args.shootout_config_json is not None else None
                ),
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
                if args.adapter_output is not None:
                    from stateset_agents.evaluation.framework_protocol import (
                        evaluate_causal_lm,
                    )

                    post = evaluate_causal_lm(
                        trained_agent.model,
                        trained_agent.tokenizer,
                        eval_examples,
                        format_prompt=adapter.format_prompt,
                        score_response=adapter.score_response,
                        max_tokens=adapter.max_new_tokens,
                    )
                else:
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
                result["metrics"]["wall_clock_seconds"] = train_seconds
                result["metrics"]["peak_vram_mb"] = torch.cuda.max_memory_allocated(
                    0
                ) / (1024 * 1024)
                result["hardware"] = {
                    "gpu": torch.cuda.get_device_name(0),
                    "gpu_count": torch.cuda.device_count(),
                    "cuda": str(torch.version.cuda),
                }

                if args.adapter_output is not None:
                    import torch

                    artifact_path = Path(args.output_dir) / "final_model"
                    try:
                        save_normalized_policy_artifact(trained_agent, artifact_path)
                    except Exception as exc:  # noqa: BLE001 - evidence must fail closed
                        logger.error(
                            "Could not normalize final policy artifact at %s: %s",
                            artifact_path,
                            exc,
                        )
                    else:
                        from stateset_agents import __version__

                        measured_samples = getattr(
                            trained_agent, "_phase0_samples_processed", None
                        )
                        if measured_samples is None:
                            measured_samples = (
                                int(config["max_steps"])
                                * int(config["per_device_train_batch_size"])
                                * int(config["gradient_accumulation_steps"])
                                * int(config["num_generations"])
                                if args.shootout_config_json is not None
                                else estimate_rollout_samples(
                                    args.trainer, len(train_examples), config
                                )
                            )
                        adapter_result = {
                            "status": "completed",
                            "measured": True,
                            "config_sha256": canonical_config_digest(config),
                            "algorithm_config": algorithm_config,
                            "algorithm_config_sha256": canonical_config_digest(
                                algorithm_config
                            ),
                            "framework_version": __version__,
                            "artifact_path": str(artifact_path.resolve()),
                            "hardware": {
                                "gpu": torch.cuda.get_device_name(0),
                                "gpu_count": torch.cuda.device_count(),
                                "cuda": str(torch.version.cuda),
                            },
                            "metrics": {
                                "samples_processed": int(measured_samples),
                                "peak_vram_mb": torch.cuda.max_memory_allocated(0)
                                / (1024 * 1024),
                                "eval_score_baseline": result["metrics"][
                                    "eval_pass_at_1_baseline"
                                ],
                                "eval_score_final": post["pass_at_1"],
                            },
                        }
                        args.adapter_output.parent.mkdir(parents=True, exist_ok=True)
                        args.adapter_output.write_text(
                            json.dumps(adapter_result, indent=2) + "\n",
                            encoding="utf-8",
                        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    logger.info("Wrote result to %s", args.output)
    if args.adapter_output is not None and not args.adapter_output.exists():
        logger.error("Training did not produce shootout adapter evidence")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
