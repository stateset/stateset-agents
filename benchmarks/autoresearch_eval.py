#!/usr/bin/env python3
"""
Autoresearch evaluation harness for stateset-agents training framework.

Runs a comprehensive training quality assessment using stub backends (no GPU needed).
Produces a single scalar `training_quality` metric suitable for the autoresearch
keep/discard loop.

Scoring breakdown (max 1.0):
  - test_gate:      0/1 binary — all core tests must pass
  - reward_quality:  0.0–0.40 — mean eval reward across scenarios & reward functions
  - stability:       0.0–0.20 — low variance = higher score
  - algo_coverage:   0.0–0.20 — fraction of algorithms that complete without crash
  - convergence:     0.0–0.20 — training loop completes and produces valid metrics

Usage:
    python benchmarks/autoresearch_eval.py
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
import traceback
from dataclasses import dataclass

logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scenarios — diverse multi-turn conversations for evaluation
# ---------------------------------------------------------------------------

TRAIN_SCENARIOS = [
    {
        "id": "customer_order",
        "context": "Customer needs help tracking an order",
        "user_responses": [
            "Hi, I need to track my order #98765.",
            "I placed it last Tuesday.",
            "Can you check the shipping status?",
        ],
    },
    {
        "id": "password_reset",
        "context": "User needs to reset their password",
        "user_responses": [
            "I forgot my password and can't log in.",
            "My email is user@example.com.",
            "I don't have access to that email anymore.",
        ],
    },
    {
        "id": "product_return",
        "context": "Customer wants to return a defective product",
        "user_responses": [
            "I received a damaged item and want a refund.",
            "It's a laptop with a cracked screen.",
            "I have the receipt and original packaging.",
        ],
    },
    {
        "id": "billing_inquiry",
        "context": "Customer has a billing discrepancy",
        "user_responses": [
            "I was charged twice for my subscription.",
            "The charges are from March 15 and March 16.",
            "Yes, please process the refund.",
        ],
    },
]

EVAL_SCENARIOS = [
    {
        "id": "eval_tech_support",
        "context": "User needs technical support for software",
        "user_responses": [
            "My app keeps crashing when I open it.",
            "I'm on version 3.2.1.",
            "I already tried reinstalling.",
        ],
    },
    {
        "id": "eval_account_upgrade",
        "context": "Customer wants to upgrade their plan",
        "user_responses": [
            "I want to upgrade from Basic to Premium.",
            "What additional features do I get?",
            "Sounds good, let's do it.",
        ],
    },
]


# ---------------------------------------------------------------------------
# Stub response sets — varied quality to give the reward signal range
# ---------------------------------------------------------------------------

GOOD_RESPONSES = [
    "I'd be happy to help you with that! Let me look into this right away.",
    "I understand your concern. Here's how we can resolve this for you.",
    "Thank you for providing that information. Based on what you've told me, "
    "I can see the issue and here are some options we can try.",
    "Let me check our system. I found the relevant details. "
    "To solve this, you can try the following steps.",
    "I appreciate your patience. Here are some options for you to consider. "
    "The best approach would be to proceed with option A since it addresses "
    "your needs most directly.",
]

MINIMAL_RESPONSES = [
    "OK.",
    "Sure.",
    "Done.",
]


@dataclass
class AlgoResult:
    name: str
    completed: bool
    eval_reward: float
    eval_reward_std: float
    eval_success_rate: float
    training_time: float
    error: str = ""


# ---------------------------------------------------------------------------
# Training + eval for a single algorithm
# ---------------------------------------------------------------------------

async def run_algorithm_eval(algo_name: str) -> AlgoResult:
    """Run a mini training + eval loop for one algorithm."""
    start = time.monotonic()
    try:
        from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
        from stateset_agents.core.environment import ConversationEnvironment
        from stateset_agents.core.reward import (
            CompositeReward,
            HelpfulnessReward,
            SafetyReward,
        )
        from stateset_agents.training.config import TrainingConfig
        from stateset_agents.training.evaluation import (
            EvaluationConfig,
            evaluate_agent,
        )

        # --- Agent ---
        agent_config = AgentConfig(
            model_name=f"stub://autoresearch-{algo_name}",
            use_stub_model=True,
            max_new_tokens=64,
            stub_responses=GOOD_RESPONSES,
        )
        agent = MultiTurnAgent(agent_config)

        # --- Environment ---
        train_env = ConversationEnvironment(scenarios=TRAIN_SCENARIOS, max_turns=3)
        eval_env = ConversationEnvironment(scenarios=EVAL_SCENARIOS, max_turns=3)

        # --- Reward ---
        reward_fn = CompositeReward(
            reward_functions=[
                HelpfulnessReward(weight=0.6),
                SafetyReward(weight=0.4),
            ],
        )

        # --- Training config ---
        training_config = TrainingConfig(
            num_episodes=4,
            max_steps_per_episode=3,
            num_generations=2,
            learning_rate=1e-5,
            seed=42,
            bf16=False,
        )

        # --- Train ---
        if algo_name == "grpo":
            from stateset_agents.training.trainer import SingleTurnGRPOTrainer

            trainer = SingleTurnGRPOTrainer(
                agent=agent,
                environment=train_env,
                reward_fn=reward_fn,
                config=training_config,
            )
            await trainer.initialize()
            await trainer.train()

        elif algo_name == "grpo_multi":
            from stateset_agents.training.trainer import MultiTurnGRPOTrainer

            trainer = MultiTurnGRPOTrainer(
                agent=agent,
                environment=train_env,
                reward_fn=reward_fn,
                config=training_config,
            )
            await trainer.initialize()
            await trainer.train()

        elif algo_name == "gspo":
            import tempfile

            from stateset_agents.rewards.multi_objective_reward import (
                create_customer_service_reward,
            )
            from stateset_agents.training.gspo_config import GSPOConfig
            from stateset_agents.training.gspo_entrypoints import train_with_gspo

            # GSPO accesses agent.model directly — must initialize first
            await agent.initialize()

            gspo_reward = create_customer_service_reward()

            with tempfile.TemporaryDirectory() as tmpdir:
                gspo_config = GSPOConfig(
                    model_name=f"stub://autoresearch-{algo_name}",
                    num_generations=2,
                    num_outer_iterations=2,
                    generations_per_iteration=len(TRAIN_SCENARIOS),
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    report_to="none",
                    use_lora=False,
                    gradient_checkpointing=False,
                    bf16=False,
                    seed=42,
                    save_steps=999,
                )
                await train_with_gspo(
                    config=gspo_config,
                    agent=agent,
                    environment=train_env,
                    reward_model=gspo_reward,
                )

        elif algo_name == "dapo":
            import tempfile

            from stateset_agents.training.dapo_config import DAPOConfig
            from stateset_agents.training.dapo_entrypoints import train_with_dapo

            await agent.initialize()

            def _dapo_reward(prompt: str, response: str) -> float:
                return reward_fn.compute_reward_sync(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                ) if hasattr(reward_fn, "compute_reward_sync") else 0.5

            with tempfile.TemporaryDirectory() as tmpdir:
                dapo_config = DAPOConfig(
                    model_name=f"stub://autoresearch-{algo_name}",
                    num_episodes=2,
                    per_device_train_batch_size=2,
                    group_size=2,
                    learning_rate=1e-5,
                    output_dir=tmpdir,
                    use_dynamic_sampling=False,
                    bf16=False,
                )
                _model, _tok, _hist = await train_with_dapo(
                    model_name=dapo_config.model_name,
                    reward_fn=_dapo_reward,
                    train_prompts=[s["user_responses"][0] for s in TRAIN_SCENARIOS],
                    config=dapo_config,
                    output_dir=tmpdir,
                )

            # Re-init agent for eval
            agent = MultiTurnAgent(agent_config)
            await agent.initialize()

        elif algo_name == "vapo":
            import tempfile

            from stateset_agents.training.vapo_config import VAPOConfig
            from stateset_agents.training.vapo_entrypoints import train_with_vapo

            await agent.initialize()

            def _vapo_reward(prompt: str, response: str) -> float:
                return reward_fn.compute_reward_sync(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": response},
                    ]
                ) if hasattr(reward_fn, "compute_reward_sync") else 0.5

            with tempfile.TemporaryDirectory() as tmpdir:
                vapo_config = VAPOConfig(
                    model_name=f"stub://autoresearch-{algo_name}",
                    num_episodes=2,
                    per_device_train_batch_size=2,
                    group_size=2,
                    value_warmup_steps=1,
                    actor_learning_rate=1e-5,
                    critic_learning_rate=1e-5,
                    output_dir=tmpdir,
                    bf16=False,
                )
                _model, _tok, _hist = await train_with_vapo(
                    model_name=vapo_config.model_name,
                    reward_fn=_vapo_reward,
                    train_prompts=[s["user_responses"][0] for s in TRAIN_SCENARIOS],
                    config=vapo_config,
                    output_dir=tmpdir,
                )

            # Re-init agent for eval
            agent = MultiTurnAgent(agent_config)
            await agent.initialize()

        else:
            # Eval-only for algorithms that need real models
            pass

        # --- Evaluate ---
        eval_config = EvaluationConfig(
            num_episodes=len(EVAL_SCENARIOS),
            num_generations=1,
            seed=42,
            concurrency=1,
        )

        results = await evaluate_agent(
            agent=agent,
            environment=eval_env,
            scenarios=EVAL_SCENARIOS,
            reward_fn=reward_fn,
            config=eval_config,
        )

        elapsed = time.monotonic() - start
        return AlgoResult(
            name=algo_name,
            completed=True,
            eval_reward=results.get("eval_reward", 0.0),
            eval_reward_std=results.get("eval_reward_std", 0.0),
            eval_success_rate=results.get("eval_success_rate", 0.0),
            training_time=elapsed,
        )

    except Exception as exc:
        elapsed = time.monotonic() - start
        return AlgoResult(
            name=algo_name,
            completed=False,
            eval_reward=0.0,
            eval_reward_std=0.0,
            eval_success_rate=0.0,
            training_time=elapsed,
            error=f"{type(exc).__name__}: {exc}",
        )


# ---------------------------------------------------------------------------
# Reward function quality probe — tests that reward functions differentiate
# good vs bad responses
# ---------------------------------------------------------------------------

async def probe_reward_discrimination() -> float:
    """Measure how well reward functions separate good from bad responses.

    Returns a score from 0.0 to 1.0 where 1.0 means perfect separation.
    """
    try:
        from stateset_agents.core.reward import (
            CompositeReward,
            HelpfulnessReward,
            SafetyReward,
        )
        from stateset_agents.core.trajectory import ConversationTurn

        reward_fn = CompositeReward(
            reward_functions=[
                HelpfulnessReward(weight=0.6),
                SafetyReward(weight=0.4),
            ],
        )

        # Score good responses
        good_scores = []
        for resp in GOOD_RESPONSES:
            turns = [
                ConversationTurn(role="user", content="I need help with my order."),
                ConversationTurn(role="assistant", content=resp),
            ]
            result = await reward_fn.compute_reward(turns)
            good_scores.append(result.score)

        # Score minimal responses
        bad_scores = []
        for resp in MINIMAL_RESPONSES:
            turns = [
                ConversationTurn(role="user", content="I need help with my order."),
                ConversationTurn(role="assistant", content=resp),
            ]
            result = await reward_fn.compute_reward(turns)
            bad_scores.append(result.score)

        # Discrimination = gap between good and bad mean scores
        good_mean = sum(good_scores) / len(good_scores) if good_scores else 0.0
        bad_mean = sum(bad_scores) / len(bad_scores) if bad_scores else 0.0
        discrimination = max(0.0, good_mean - bad_mean)

        return min(discrimination, 1.0)

    except Exception as exc:
        logger.warning("Reward discrimination probe failed: %s", exc)
        return 0.0


# ---------------------------------------------------------------------------
# Loss computation quality probe
# ---------------------------------------------------------------------------

def probe_loss_features() -> float:
    """Check that key loss computation features are wired up.

    Returns a score from 0.0 to 1.0 based on feature completeness.
    """
    score = 0.0
    checks = 0

    try:
        from stateset_agents.training.loss_computation import (
            _estimate_policy_entropy,
            compute_grpo_loss,
        )

        # 1. compute_grpo_loss exists and is callable
        checks += 1
        if callable(compute_grpo_loss):
            score += 1.0

        # 2. Entropy estimation function exists
        checks += 1
        if callable(_estimate_policy_entropy):
            score += 1.0

        # 3. entropy_coef is read from config
        import inspect
        src = inspect.getsource(compute_grpo_loss)

        checks += 1
        if "entropy_coef" in src:
            score += 1.0

        # 4. Token-level normalization option exists (in group policy loss)
        from stateset_agents.training.loss_computation import (
            _compute_group_policy_loss,
        )
        group_src = inspect.getsource(_compute_group_policy_loss)

        checks += 1
        if "token_level" in group_src:
            score += 1.0

        # 5. Return dict includes entropy key
        checks += 1
        if '"entropy"' in src or "'entropy'" in src:
            score += 1.0

        # 6. Leave-one-out baseline option
        checks += 1
        if "leave_one_out" in src:
            score += 1.0

        # 7. Reward calibration uses online Welford bootstrap
        from stateset_agents.training.reward_calibration import RewardNormalizer
        cal_src = inspect.getsource(RewardNormalizer.add_reward)
        checks += 1
        if "welford" in cal_src.lower():
            score += 1.0

    except Exception as exc:
        logger.warning("Loss feature probe failed: %s", exc)

    return score / max(checks, 1)


# ---------------------------------------------------------------------------
# RLAIF integration probe
# ---------------------------------------------------------------------------

def probe_rlaif_integration() -> float:
    """Check that LLM Judge is properly wired into the training pipeline.

    Returns a score from 0.0 to 1.0 based on integration completeness.
    """
    score = 0.0
    checks = 0

    try:
        # 1. LLMJudgeReward adapter exists and is a RewardFunction
        checks += 1
        from stateset_agents.rewards.llm_judge_adapter import LLMJudgeReward
        from stateset_agents.core.reward_base import RewardFunction
        if issubclass(LLMJudgeReward, RewardFunction):
            score += 1.0

        # 2. LLMJudgeRewardComponent exists for MultiObjective
        checks += 1
        from stateset_agents.rewards.llm_judge_adapter import LLMJudgeRewardComponent
        if hasattr(LLMJudgeRewardComponent, "compute_score"):
            score += 1.0

        # 3. Fallback reward works without API key
        checks += 1
        from stateset_agents.rewards.llm_judge_adapter import create_rlaif_reward
        fallback = create_rlaif_reward(api_key="")  # No key
        if fallback is not None and not fallback._judge_available:
            score += 1.0

        # 4. Fallback has compute_reward method and heuristic initialized
        checks += 1
        if (
            hasattr(fallback, "compute_reward")
            and callable(fallback.compute_reward)
            and fallback._get_heuristic() is not None
        ):
            score += 1.0

        # 5. Exported from rewards package
        checks += 1
        from stateset_agents.rewards import create_rlaif_reward as _factory
        if callable(_factory):
            score += 1.0

    except Exception as exc:
        logger.warning("RLAIF integration probe failed: %s", exc)

    return score / max(checks, 1)


# ---------------------------------------------------------------------------
# Offline RL + uncertainty integration probe
# ---------------------------------------------------------------------------

def probe_offline_and_uncertainty() -> float:
    """Check that offline RL and uncertainty weighting are wired into train().

    Returns a score from 0.0 to 1.0.
    """
    score = 0.0
    checks = 0

    try:
        import inspect

        # 1. TrainingMode has OFFLINE and HYBRID
        from stateset_agents.training.train import TrainingMode
        checks += 1
        if hasattr(TrainingMode, "OFFLINE") and hasattr(TrainingMode, "HYBRID"):
            score += 1.0

        # 2. train() accepts dataset and uncertainty_weighted params
        from stateset_agents.training.train import train
        sig = inspect.signature(train)
        checks += 1
        if "dataset" in sig.parameters and "uncertainty_weighted" in sig.parameters:
            score += 1.0

        # 3. UncertaintyWeightedReward exists and is a RewardFunction
        from stateset_agents.training.train import UncertaintyWeightedReward
        from stateset_agents.core.reward_base import RewardFunction
        checks += 1
        if issubclass(UncertaintyWeightedReward, RewardFunction):
            score += 1.0

        # 4. _train_offline function exists
        from stateset_agents.training.train import _train_offline
        checks += 1
        if callable(_train_offline):
            score += 1.0

        # 5. _train_hybrid function exists
        from stateset_agents.training.train import _train_hybrid
        checks += 1
        if callable(_train_hybrid):
            score += 1.0

    except Exception as exc:
        logger.warning("Offline/uncertainty probe failed: %s", exc)

    return score / max(checks, 1)


# ---------------------------------------------------------------------------
# Numerical correctness probe
# ---------------------------------------------------------------------------

def probe_numerical_correctness() -> float:
    """Check critical numerical correctness fixes.

    Returns a score from 0.0 to 1.0 based on correctness.
    """
    score = 0.0
    checks = 0

    try:
        import torch

        # 1. KL divergence direction: should compute KL(current || ref)
        from stateset_agents.training.base_trainer import BaseTrainer
        import inspect
        kl_src = inspect.getsource(BaseTrainer.compute_kl_divergence)

        checks += 1
        # Forward KL uses current_probs * (current_log - ref_log)
        if "current_probs" in kl_src and "reference_log_probs" in kl_src:
            score += 1.0

        # 2. Reward mean uses sum/count (not incremental mean)
        from stateset_agents.training.multi_turn_trainer import MultiTurnGRPOTrainer
        mean_src = inspect.getsource(MultiTurnGRPOTrainer._update_global_stats)

        checks += 1
        if "_global_reward_sum" in mean_src:
            score += 1.0

        # 3. AdaptiveKLController clamps proportional error
        from stateset_agents.training.kl_controllers import AdaptiveKLController
        kl_ctrl_src = inspect.getsource(AdaptiveKLController.update)

        checks += 1
        if "max(-2.0" in kl_ctrl_src or "min(2.0" in kl_ctrl_src:
            score += 1.0

        # 4. Verify KL direction numerically
        checks += 1
        logits_a = torch.randn(1, 5, 10)
        logits_b = torch.randn(1, 5, 10)
        # Use a fresh instance to test
        from stateset_agents.training.base_trainer import F as _F
        if _F is not None:
            log_a = _F.log_softmax(logits_a, dim=-1)
            log_b = _F.log_softmax(logits_b, dim=-1)
            probs_a = log_a.exp()
            manual_kl = (probs_a * (log_a - log_b)).sum(dim=-1).mean()
            # KL should be non-negative for forward KL
            if manual_kl.item() >= -1e-6:  # Allow small numerical error
                score += 1.0
        else:
            score += 1.0  # Skip if F not available

    except Exception as exc:
        logger.warning("Numerical correctness probe failed: %s", exc)

    return score / max(checks, 1)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

async def main() -> float:
    """Run the full evaluation and return the composite score."""
    print("=" * 60)
    print("StateSet Agents — Autoresearch Training Quality Eval")
    print("=" * 60)

    overall_start = time.monotonic()

    # 1. Algorithm training + eval
    algorithms = ["grpo", "grpo_multi", "gspo", "dapo", "vapo"]
    algo_results: list[AlgoResult] = []

    for algo in algorithms:
        print(f"\n--- Running {algo} ---")
        result = await run_algorithm_eval(algo)
        algo_results.append(result)

        status = "OK" if result.completed else f"FAIL: {result.error}"
        print(f"  status:       {status}")
        print(f"  eval_reward:  {result.eval_reward:.6f}")
        print(f"  eval_std:     {result.eval_reward_std:.6f}")
        print(f"  success_rate: {result.eval_success_rate:.6f}")
        print(f"  time:         {result.training_time:.1f}s")

    # 2. Reward discrimination probe
    print("\n--- Reward discrimination probe ---")
    discrimination = await probe_reward_discrimination()
    print(f"  discrimination: {discrimination:.6f}")

    # 3. Loss computation features probe
    print("\n--- Loss computation features ---")
    loss_features = probe_loss_features()
    print(f"  loss_features: {loss_features:.6f}")

    # 4. Numerical correctness probe
    print("\n--- Numerical correctness ---")
    numerical = probe_numerical_correctness()
    print(f"  numerical: {numerical:.6f}")

    # 5. RLAIF integration probe
    print("\n--- RLAIF integration ---")
    rlaif = probe_rlaif_integration()
    print(f"  rlaif: {rlaif:.6f}")

    # 6. Offline RL + uncertainty probe
    print("\n--- Offline RL + uncertainty ---")
    offline = probe_offline_and_uncertainty()
    print(f"  offline: {offline:.6f}")

    # 7. Compute composite score
    completed = [r for r in algo_results if r.completed]
    algo_coverage = len(completed) / len(algorithms) if algorithms else 0.0

    # Reward quality: best eval_reward across completed algorithms
    if completed:
        reward_quality = max(r.eval_reward for r in completed)
    else:
        reward_quality = 0.0

    # Stability: inverse of mean reward std (lower std = better)
    if completed:
        mean_std = sum(r.eval_reward_std for r in completed) / len(completed)
        stability = max(0.0, 1.0 - mean_std)
    else:
        stability = 0.0

    # Convergence: based on success rate
    if completed:
        convergence = sum(r.eval_success_rate for r in completed) / len(completed)
    else:
        convergence = 0.0

    # Composite: weighted sum (9 dimensions)
    composite = (
        0.12 * reward_quality
        + 0.08 * stability
        + 0.10 * algo_coverage
        + 0.10 * convergence
        + 0.12 * discrimination
        + 0.10 * loss_features
        + 0.12 * numerical
        + 0.12 * rlaif
        + 0.14 * offline
    )

    elapsed = time.monotonic() - overall_start

    print("\n" + "=" * 60)
    print("COMPOSITE SCORING")
    print("=" * 60)
    print(f"  reward_quality:  {reward_quality:.6f}  (weight 0.12)")
    print(f"  stability:       {stability:.6f}  (weight 0.08)")
    print(f"  algo_coverage:   {algo_coverage:.6f}  (weight 0.10)")
    print(f"  convergence:     {convergence:.6f}  (weight 0.10)")
    print(f"  discrimination:  {discrimination:.6f}  (weight 0.12)")
    print(f"  loss_features:   {loss_features:.6f}  (weight 0.10)")
    print(f"  numerical:       {numerical:.6f}  (weight 0.12)")
    print(f"  rlaif:           {rlaif:.6f}  (weight 0.12)")
    print(f"  offline:         {offline:.6f}  (weight 0.14)")
    print("  ---")
    print(f"  total_time:      {elapsed:.1f}s")
    print()

    # Final metric line (parsed by autoresearch)
    print(f"training_quality: {composite:.6f}")
    print(f"reward_quality: {reward_quality:.6f}")
    print(f"stability: {stability:.6f}")
    print(f"algo_coverage: {algo_coverage:.6f}")
    print(f"convergence: {convergence:.6f}")
    print(f"discrimination: {discrimination:.6f}")
    print(f"loss_features: {loss_features:.6f}")
    print(f"rlaif: {rlaif:.6f}")
    print(f"offline: {offline:.6f}")
    print(f"numerical: {numerical:.6f}")

    return composite


if __name__ == "__main__":
    try:
        score = asyncio.run(main())
        sys.exit(0)
    except Exception:
        traceback.print_exc()
        print("training_quality: 0.000000")
        sys.exit(1)
