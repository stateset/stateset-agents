"""
Training entrypoints for GSPO workflows.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping, Sequence
from typing import Any, cast

from stateset_agents.core.agent import Agent, AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import (
    MultiObjectiveRewardFunction as MultiObjectiveReward,
)

from .callbacks import has_no_policy_signal
from .config import get_config_for_task
from .dedicated_reporting import WandbRun, reports_to_wandb
from .gspo_config import GSPOConfig
from .runtime_state import _reject_unsupported_resume
from .spine.loop import run_dedicated_loop, seed_dedicated_run

logger = logging.getLogger(__name__)


#: Scenario keys that may hold the user-facing prompt text, in priority order.
PROMPT_KEYS: tuple[str, ...] = ("prompt", "user_query", "query", "question", "context")
ZERO_SIGNAL_WARN_AFTER = 5


def queries_from_scenarios(
    scenarios: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Turn environment scenarios into GSPO training queries.

    The prompt is the first of :data:`PROMPT_KEYS` present; every other
    scenario field (plus the contents of ``task``/``metadata`` mappings) is
    passed to the reward as ``context`` so task rewards that need ground truth
    (``gold_answer``, ``expected_tool``, ...) receive it. A scenario without
    any prompt key is an error rather than a placeholder prompt.
    """
    queries: list[dict[str, Any]] = []
    for index, scenario in enumerate(scenarios):
        prompt_key = next(
            (
                k
                for k in PROMPT_KEYS
                if isinstance(scenario.get(k), str) and scenario[k]
            ),
            None,
        )
        if prompt_key is None:
            raise ValueError(
                f"scenario {index} has no prompt text under any of {PROMPT_KEYS}; "
                "pass train_queries explicitly"
            )
        context: dict[str, Any] = {}
        for nested in ("task", "metadata"):
            value = scenario.get(nested)
            if isinstance(value, Mapping):
                context.update(value)
        context.update(
            {
                k: v
                for k, v in scenario.items()
                if k != prompt_key and k not in ("task", "metadata")
            }
        )
        context.setdefault("scenario_index", index)
        queries.append({"prompt": str(scenario[prompt_key]), "context": context})
    return queries


def query_window(
    queries: Sequence[Any], iteration: int, per_iteration: int
) -> list[Any]:
    """The rotating slice of ``queries`` trained on at ``iteration`` (wraps
    around, so every query is visited before any repeats)."""
    n = len(queries)
    if n == 0:
        return []
    size = max(1, min(int(per_iteration), n))
    start = (int(iteration) * size) % n
    return [queries[(start + i) % n] for i in range(size)]


def _reward_is_identically_zero(metrics: Mapping[str, Any]) -> bool:
    """Legacy reward-only diagnostic; absent metrics are not evidence."""
    return has_no_policy_signal(
        {
            key: metrics[key]
            for key in ("average_reward", "reward_std")
            if key in metrics
        }
    )


#: Trainer-config fields that drive rollout sampling, in AgentConfig terms.
ROLLOUT_SAMPLING_FIELDS: tuple[str, ...] = (
    "temperature",
    "top_p",
    "top_k",
    "repetition_penalty",
)


def apply_rollout_sampling(agent: Any, config: Any) -> None:
    """Make the trainer config's sampling settings drive the agent's rollouts.

    Precedence: for GSPO training rollouts, ``config.temperature``,
    ``top_p``, ``top_k`` and ``repetition_penalty`` override the agent's own
    ``AgentConfig`` values (previously the agent's defaults silently won and
    the configured temperature never reached generation). The generation
    config is rebuilt afterwards. Agents without a tokenizer (stubs) keep
    their generation config.
    """
    agent_config = getattr(agent, "config", None)
    if agent_config is None:
        return
    for name in ROLLOUT_SAMPLING_FIELDS:
        value = getattr(config, name, None)
        if value is not None and hasattr(agent_config, name):
            setattr(agent_config, name, value)
    if getattr(agent, "tokenizer", None) is not None and callable(
        getattr(agent, "_build_generation_config", None)
    ):
        agent.generation_config = agent._build_generation_config()


async def train_with_gspo(
    config: GSPOConfig,
    agent: Agent,
    environment: ConversationEnvironment,
    reward_model: MultiObjectiveReward,
    train_queries: list[str | dict[str, Any]] | None = None,
    callbacks: list[Any] | None = None,
) -> Agent:
    """Train fresh with dedicated GSPO; explicit resume is unsupported.

    ``config.use_gspo_token`` selects the GSPO-token variant
    (``GSPOTokenTrainer.train_step_token_level``). The outer loop is the
    spine's :func:`run_dedicated_loop`.
    """
    _reject_unsupported_resume(config, "train_with_gspo")
    from .gspo_trainer import GSPOModelManager, GSPOTrainer

    logger.info("Initializing GSPO training")
    logger.info("Configuration: %s", json.dumps(config.to_dict(), indent=2))

    # Foot-gun guard: training without a KL anchor on a small corpus is the
    # canonical "policy goes off the rails" setup. See whitepaper §10.5
    # ('Reference-model drift') and benchmark_results/whitepaper_v1/
    # customer_support_qwen3_5_0_8b_gspo.json for a documented case.
    _has_kl_anchor = config.use_reference_model and config.beta > 0.0
    _corpus_size = (
        len(train_queries)
        if train_queries
        else (len(environment.scenarios) if hasattr(environment, "scenarios") else None)
    )
    if not _has_kl_anchor and _corpus_size is not None and _corpus_size < 100:
        logger.warning(
            "GSPO is running without a KL anchor (use_reference_model=%s, beta=%s) "
            "on a small corpus (%s queries). This combination has been observed to "
            "destabilize the policy on keyword-based rewards — the trained model "
            "starts emitting incoherent token soup while still scoring nonzero on "
            "the rubric. Recommended fix: set use_reference_model=True and beta>=0.05. "
            "See whitepaper §10.5 'Reference-model drift'.",
            config.use_reference_model,
            config.beta,
            _corpus_size,
        )

    os.makedirs(config.output_dir, exist_ok=True)

    with WandbRun(
        "gspo",
        enabled=reports_to_wandb(config.report_to),
        project=config.wandb_project,
        config=config,
        tags=["gspo"] + list(config.wandb_tags or []),
        name=config.run_name,
    ) as reporter:
        return await _run_gspo(
            config,
            agent,
            environment,
            reward_model,
            train_queries,
            callbacks or [],
            reporter,
            model_manager_cls=GSPOModelManager,
            trainer_cls=GSPOTrainer,
        )


async def _run_gspo(
    config: GSPOConfig,
    agent: Agent,
    environment: ConversationEnvironment,
    reward_model: MultiObjectiveReward,
    train_queries: list[str] | None,
    callbacks: list[Any],
    reporter: WandbRun,
    *,
    model_manager_cls: Any,
    trainer_cls: Any,
) -> Agent:
    """:func:`train_with_gspo` after its W&B run is set up."""
    seed_dedicated_run(config.seed)
    _is_stub = config.model_name.startswith("stub://") or getattr(
        config, "use_stub_model", False
    )

    if _is_stub:
        logger.info(
            "Stub backend detected — skipping model manager, using agent's model"
        )
        model = agent.model
        tokenizer = agent.tokenizer
        model_manager = None
    else:
        model_manager = model_manager_cls(config)
        model, tokenizer = model_manager.load_model_and_tokenizer()
        agent.model = model
        agent.tokenizer = tokenizer
    apply_rollout_sampling(agent, config)

    queries: list[Any] = list(train_queries) if train_queries else []
    if not queries:
        logger.info("Generating training queries from environment scenarios...")
        queries = list(queries_from_scenarios(environment.scenarios))

    queries_per_iteration = max(1, int(config.generations_per_iteration))
    logger.info(
        "Training with %s queries (%s per iteration, rotating)",
        len(queries),
        min(queries_per_iteration, len(queries)),
    )

    if getattr(config, "use_gspo_token", False):
        from .gspo_token_trainer import GSPOTokenTrainer

        trainer_cls = GSPOTokenTrainer
    trainer = trainer_cls(
        config=config,
        model=model,
        tokenizer=tokenizer,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
        ref_model=model_manager.ref_model if model_manager else None,
    )
    train_step = (
        trainer.train_step_token_level
        if getattr(config, "use_gspo_token", False)
        else trainer.train_step
    )

    if config.use_vllm:
        logger.info("Initializing vLLM for fast generation...")
        vllm_success = await trainer.generator.initialize_vllm()
        if vllm_success:
            logger.info("vLLM initialized - generation will be 5-20x faster!")
        else:
            logger.warning("vLLM initialization failed - using HuggingFace generation")

    _callbacks = callbacks or []
    zero_signal_iterations = 0

    def window(iteration: int, _rng: Any) -> list[Any]:
        logger.info(
            "=== Iteration %s/%s ===", iteration + 1, config.num_outer_iterations
        )
        return query_window(queries, iteration, queries_per_iteration)

    async def step(batch: list[Any]) -> dict[str, Any]:
        metrics: dict[str, Any] = await train_step(queries=batch, num_groups=len(batch))
        return metrics

    def report(iteration: int, metrics: dict[str, Any]) -> None:
        nonlocal zero_signal_iterations
        zero_signal_iterations = (
            zero_signal_iterations + 1 if has_no_policy_signal(metrics) else 0
        )
        if zero_signal_iterations == ZERO_SIGNAL_WARN_AFTER:
            logger.warning(
                "GSPO policy advantages have been identically zero for %s consecutive "
                "iterations: no learning signal from the policy advantage term. "
                "Check rollout diversity, task difficulty and that the reward "
                "receives the scenario context it "
                "needs (for example gold_answer) and that prompts are the real task "
                "prompts.",
                ZERO_SIGNAL_WARN_AFTER,
            )

        logger.info("Metrics: %s", json.dumps(metrics, indent=2))
        reporter.log(metrics, step=iteration)

    run = await run_dedicated_loop(
        iterations=config.num_outer_iterations,
        batch=window,
        step=step,
        save=trainer.save_model,
        output_dir=config.output_dir,
        save_steps=config.save_steps,
        final_name="final_model",
        report=report,
        callbacks=_callbacks,
        seed=config.seed,
    )

    if run.aborted:
        logger.info("GSPO training aborted early.")
        cast(Any, agent)._training_aborted = "; ".join(run.abort_reasons) or "aborted"
    else:
        logger.info("GSPO training completed successfully.")
    setattr(agent, "_training_metrics", dict(trainer.training_metrics))  # noqa: B010
    return agent


async def train_customer_service_with_gspo(
    model_name: str = "gpt2",
    num_episodes: int = 100,
    output_dir: str = "./outputs/gspo",
    **kwargs,
) -> Agent:
    """Train a customer service agent using GSPO.

    GSPO runs ``num_outer_iterations`` rounds; ``num_episodes`` sets it unless
    the caller passes ``num_outer_iterations`` explicitly.
    """
    base_config = get_config_for_task("customer_service", model_name=model_name)
    kwargs.setdefault("num_outer_iterations", num_episodes)

    config = GSPOConfig.from_training_config(
        base_config, num_episodes=num_episodes, output_dir=output_dir, **kwargs
    )
    _reject_unsupported_resume(config, "train_customer_service_with_gspo")

    agent_config = AgentConfig(
        model_name=model_name,
        system_prompt="You are a helpful and empathetic customer service representative.",
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    from stateset_agents.core.environment import CONVERSATION_CONFIGS

    env_config = CONVERSATION_CONFIGS["customer_service"].copy()
    environment = ConversationEnvironment(**env_config)

    from stateset_agents.core.grounding import log_heuristic_reward_notice
    from stateset_agents.rewards.multi_objective_reward import (
        create_customer_service_reward,
    )

    reward_model = create_customer_service_reward()
    log_heuristic_reward_notice(
        logger, "GSPO customer service training", "create_customer_service_reward()"
    )

    return await train_with_gspo(
        config=config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )


__all__ = ["train_with_gspo", "train_customer_service_with_gspo"]
