"""
Core autonomous research loop.

This is the heart of the auto-research system: an outer loop that proposes
experiments, trains with a time budget, evaluates on held-out scenarios,
and keeps only improvements — running autonomously until stopped.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import time
from collections.abc import Sequence
from typing import Any

from .checkpoint_manager import CheckpointManager
from .config import AutoResearchConfig
from .experiment_tracker import ExperimentRecord, ExperimentTracker
from .proposer import BayesianProposer, ExperimentProposer, create_proposer

logger = logging.getLogger(__name__)


class AutoResearchLoop:
    """Autonomous experiment loop that iterates on training configurations.

    Generalizes the autoresearch pattern:
    1. Propose a new configuration (via a pluggable proposer strategy)
    2. Train the agent with a time budget
    3. Evaluate on held-out scenarios
    4. Keep improvements, revert failures
    5. Repeat

    Features:
    - Pluggable proposers (perturbation, random, grid, bayesian, LLM-driven)
    - Filesystem-based checkpointing (no git dependency)
    - Resume from a previous run
    - Optional W&B experiment logging
    - Graceful shutdown via SIGINT/SIGTERM
    - Time-budgeted training with asyncio timeout
    """

    def __init__(
        self,
        *,
        agent: Any,
        environment: Any,
        eval_scenarios: Sequence[dict[str, Any]],
        reward_fn: Any,
        config: AutoResearchConfig,
        proposer: ExperimentProposer | None = None,
        baseline_params: dict[str, Any] | None = None,
    ):
        self.agent = agent
        self.environment = environment
        self.eval_scenarios = list(eval_scenarios)
        self.reward_fn = reward_fn
        self.config = config

        self.tracker = ExperimentTracker(
            output_dir=config.output_path,
            objective_metric=config.objective_metric,
            direction=config.direction,
        )
        self.checkpoint_mgr = CheckpointManager(config.output_path)

        # Resolve search space (cached for reuse in validation)
        self.search_space = self._resolve_search_space()

        # Create proposer
        self.proposer = proposer or create_proposer(
            strategy=config.proposer,
            search_space=self.search_space,
            direction=config.direction,
        )

        # Baseline params — the starting configuration
        self.baseline_params = baseline_params or {}

        self._stop_requested = False
        self._loop_start_time: float = 0.0
        self._wandb: Any = None

        # Optional reward calibration to prevent scale drift
        self._calibrated_reward_fn: Any = None
        if config.calibrate_rewards:
            self._init_reward_calibration()

    def _resolve_search_space(self) -> Any:
        """Resolve the search space from config."""
        # Try auto-research specific spaces first
        from .search_spaces import AUTO_RESEARCH_SPACES

        if self.config.search_space_name in AUTO_RESEARCH_SPACES:
            return AUTO_RESEARCH_SPACES[self.config.search_space_name]()

        # Fall back to HPO module search spaces
        from stateset_agents.training.hpo.search_spaces import (
            create_grpo_search_space,
            get_search_space,
        )

        if self.config.search_space_name:
            try:
                space = get_search_space(self.config.search_space_name)
            except (KeyError, ValueError):
                logger.warning(
                    "Search space %r not found, falling back to 'grpo'",
                    self.config.search_space_name,
                )
                space = create_grpo_search_space()
        else:
            space = create_grpo_search_space()

        return space

    # ------------------------------------------------------------------
    # Reward calibration
    # ------------------------------------------------------------------

    def _init_reward_calibration(self) -> None:
        """Wrap the reward function with calibration to prevent scale drift."""
        try:
            from stateset_agents.training.reward_calibration import (
                CalibratedRewardFunction,
                RewardNormalizer,
            )

            normalizer = RewardNormalizer(
                method=self.config.calibration_method,
                clip_range=(-3.0, 3.0),
            )
            self._calibrated_reward_fn = CalibratedRewardFunction(
                base_reward_fn=self.reward_fn,
                normalizer=normalizer,
                auto_calibrate=True,
            )
            logger.info(
                "Reward calibration enabled (method=%s)",
                self.config.calibration_method,
            )
        except (ImportError, AttributeError) as exc:
            logger.warning(
                "Reward calibration requested but unavailable: %s", exc
            )
            self._calibrated_reward_fn = None

    @property
    def _eval_reward_fn(self) -> Any:
        """Return the reward function to use for evaluation."""
        return self._calibrated_reward_fn or self.reward_fn

    # ------------------------------------------------------------------
    # W&B integration
    # ------------------------------------------------------------------

    def _init_wandb(self) -> None:
        """Initialize W&B logging if configured."""
        if not self.config.log_to_wandb:
            return
        try:
            import wandb

            self._wandb = wandb
            wandb.init(
                project=self.config.wandb_project,
                name=f"auto-research-{int(time.time())}",
                config={
                    "time_budget": self.config.time_budget,
                    "max_experiments": self.config.max_experiments,
                    "proposer": self.config.proposer,
                    "trainer_algorithm": self.config.trainer_algorithm,
                    "objective_metric": self.config.objective_metric,
                    "direction": self.config.direction,
                    "baseline_params": self.baseline_params,
                },
                tags=["auto-research"],
            )
        except ImportError:
            logger.warning("wandb not installed; disabling W&B logging")
            self._wandb = None
        except Exception as exc:
            logger.warning("Failed to initialize wandb: %s", exc)
            self._wandb = None

    def _log_wandb(self, record: ExperimentRecord) -> None:
        """Log an experiment record to W&B."""
        if self._wandb is None:
            return
        try:
            log_data: dict[str, Any] = {
                "experiment_id": record.experiment_id,
                "objective": record.objective_value,
                "training_time": record.training_time,
                "status": record.status,
                "best_so_far": self.tracker.best_value or 0.0,
                "num_experiments": self.tracker.num_experiments,
            }
            log_data.update(
                {f"metric/{k}": v for k, v in record.metrics.items()}
            )
            log_data.update(
                {f"param/{k}": v for k, v in record.params.items()
                 if isinstance(v, (int, float, str, bool))}
            )
            self._wandb.log(log_data)
        except Exception as exc:
            logger.debug("wandb log failed: %s", exc)

    def _finish_wandb(self) -> None:
        """Finalize W&B run."""
        if self._wandb is None:
            return
        try:
            summary: dict[str, float | int | str | bool | None] = {
                "best_objective": self.tracker.best_value,
                "total_experiments": self.tracker.num_experiments,
                "kept": self.tracker.num_kept,
                "discarded": self.tracker.num_discarded,
                "crashed": self.tracker.num_crashed,
            }
            if self.tracker.best_record:
                for k, v in self.tracker.best_record.params.items():
                    if isinstance(v, (int, float, str, bool)):
                        summary[f"best_param/{k}"] = v
            self._wandb.summary.update(summary)
            self._wandb.finish()
        except Exception as exc:
            logger.debug("wandb finish failed: %s", exc)

    # ------------------------------------------------------------------
    # Resume support
    # ------------------------------------------------------------------

    def _try_resume(self) -> int:
        """Attempt to resume from a previous run in the same output directory.

        Returns the next experiment number to use (1 if no resume, or N+1
        where N is the last experiment found).
        """
        jsonl_path = self.config.output_path / "experiments.jsonl"
        if not jsonl_path.exists():
            return 1

        logger.info("Found previous run at %s — attempting resume", jsonl_path)

        loaded = 0
        last_num = 0
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                record = ExperimentRecord(
                    experiment_id=data["experiment_id"],
                    params=data.get("params", {}),
                    metrics=data.get("metrics", {}),
                    objective_value=data.get("objective_value", 0.0),
                    training_time=data.get("training_time", 0.0),
                    status=data.get("status", "unknown"),
                    description=data.get("description", ""),
                    timestamp=data.get("timestamp", 0.0),
                )

                # Load without re-persisting to disk
                self.tracker.load_record(record)

                # Track experiment numbering
                if record.experiment_id.startswith("exp_"):
                    try:
                        num = int(record.experiment_id.split("_")[1])
                        last_num = max(last_num, num)
                    except (IndexError, ValueError):
                        pass

                loaded += 1

        if loaded > 0:
            logger.info(
                "Resumed %d experiments (best %s=%.6f)",
                loaded,
                self.config.objective_metric,
                self.tracker.best_value or 0.0,
            )

            # Restore best checkpoint if available
            if self.checkpoint_mgr.has_best():
                self.checkpoint_mgr.restore_best(self.agent)
                logger.info("Restored best model checkpoint")

        return last_num + 1

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    async def run(self) -> ExperimentTracker:
        """Run the autonomous research loop.

        Returns the ExperimentTracker with all results.
        """
        self._loop_start_time = time.time()
        self._stop_requested = False

        # Install signal handlers for graceful shutdown
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                loop.add_signal_handler(sig, self._request_stop)
            except (NotImplementedError, RuntimeError):
                pass  # Windows or nested loop

        # Initialize W&B
        self._init_wandb()

        logger.info("Starting autonomous research loop")
        logger.info("  Objective: %s (%s)", self.config.objective_metric, self.config.direction)
        logger.info("  Time budget per experiment: %ds", self.config.time_budget)
        logger.info("  Max experiments: %s", self.config.max_experiments or "unlimited")
        logger.info("  Proposer: %s", self.config.proposer)
        logger.info("  Output: %s", self.config.output_dir)

        # Validate baseline params against search space
        if self.baseline_params:
            try:
                from .search_spaces import validate_params_against_space

                bound_warnings = validate_params_against_space(
                    self.baseline_params, self.search_space
                )
                for w in bound_warnings:
                    logger.warning("Baseline param out of search space: %s", w)
            except Exception:
                pass

        # Try to resume from previous run
        next_experiment_num = self._try_resume()

        if next_experiment_num == 1:
            # Fresh run — establish baseline
            await self._run_baseline()
        else:
            logger.info(
                "Resuming from experiment %d (skipping baseline)",
                next_experiment_num,
            )

        # Main loop
        experiment_num = next_experiment_num
        while not self._should_stop():
            experiment_id = f"exp_{experiment_num:04d}"

            try:
                await self._run_experiment(experiment_id)
            except Exception as exc:
                logger.error("Experiment %s crashed: %s", experiment_id, exc)
                self._cleanup_gpu()
                self._record_crash(experiment_id, {}, str(exc))
                self.checkpoint_mgr.restore_best(self.agent)

            experiment_num += 1

        self._finish_wandb()
        self.tracker.print_summary()
        self._print_analysis()
        return self.tracker

    async def _run_baseline(self) -> None:
        """Run the baseline experiment (eval only, no training)."""
        logger.info("Running baseline experiment...")

        budget = self._current_experiment_timeout()
        started = time.time()
        try:
            eval_result = await asyncio.wait_for(
                self._evaluate(),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            elapsed = time.time() - started
            logger.warning(
                "Baseline evaluation timed out after %.0fs (budget=%.0fs)",
                elapsed,
                budget,
            )
            self._cleanup_gpu()
            record = ExperimentRecord(
                experiment_id="baseline",
                params=dict(self.baseline_params),
                metrics={},
                objective_value=0.0,
                training_time=elapsed,
                status="crash",
                description=f"baseline timeout after {elapsed:.0f}s",
            )
            self.tracker.record(record)
            self._log_wandb(record)
            return
        except Exception as exc:
            elapsed = time.time() - started
            logger.warning("Baseline evaluation failed: %s", exc)
            self._cleanup_gpu()
            record = ExperimentRecord(
                experiment_id="baseline",
                params=dict(self.baseline_params),
                metrics={},
                objective_value=0.0,
                training_time=elapsed,
                status="crash",
                description=f"baseline crash: {exc}",
            )
            self.tracker.record(record)
            self._log_wandb(record)
            return

        objective = eval_result.get(self.config.objective_metric, 0.0)

        if objective == 0.0:
            logger.warning(
                "Baseline %s is 0.0 — evaluation may have failed or "
                "the agent may not produce scorable responses.",
                self.config.objective_metric,
            )

        record = ExperimentRecord(
            experiment_id="baseline",
            params=dict(self.baseline_params),
            metrics=eval_result,
            objective_value=objective,
            training_time=time.time() - started,
            status="keep",
            description="baseline (no training)",
        )
        self.tracker.record(record)
        self._log_wandb(record)

        # Save baseline as first checkpoint
        if self.config.save_checkpoints:
            self.checkpoint_mgr.save_best(
                self.agent, "baseline", self.baseline_params
            )

    async def _run_experiment(self, experiment_id: str) -> None:
        """Run a single experiment: propose → train → eval → keep/revert."""
        current_best_params = (
            self.tracker.best_record.params
            if self.tracker.best_record
            else self.baseline_params
        )

        # 1. Propose
        proposed_params, description = self.proposer.propose(
            current_best=current_best_params,
            history=self.tracker.get_history_for_proposer(),
        )

        logger.info("Experiment %s: %s", experiment_id, description)

        # 2. Train + eval within a shared experiment budget
        train_start = time.time()
        budget = self._current_experiment_timeout()
        try:
            eval_result = await asyncio.wait_for(
                self._execute_experiment(proposed_params),
                timeout=budget,
            )
        except asyncio.TimeoutError:
            training_time = time.time() - train_start
            logger.warning(
                "Experiment %s timed out after %.0fs (budget=%.0fs)",
                experiment_id,
                training_time,
                budget,
            )
            self._cleanup_gpu()
            self._record_crash(
                experiment_id, proposed_params,
                f"timeout after {training_time:.0f}s",
            )
            self._report_to_proposer(0.0, crashed=True)
            self.checkpoint_mgr.restore_best(self.agent)
            return
        except Exception as exc:
            training_time = time.time() - train_start
            logger.warning("Experiment %s failed: %s", experiment_id, exc)
            self._cleanup_gpu()
            self._record_crash(
                experiment_id, proposed_params, f"train/eval crash: {exc}"
            )
            self._report_to_proposer(0.0, crashed=True)
            self.checkpoint_mgr.restore_best(self.agent)
            return

        training_time = time.time() - train_start
        objective = eval_result.get(self.config.objective_metric, 0.0)

        # 4. Keep or revert
        is_better = self.tracker.is_improvement(objective)

        if is_better and self._passes_threshold(objective):
            status = "keep"
            if self.config.save_checkpoints:
                self.checkpoint_mgr.save_best(
                    self.agent, experiment_id, proposed_params
                )
            logger.info(
                "KEEP %s: %s=%.6f (improved from %.6f)",
                experiment_id,
                self.config.objective_metric,
                objective,
                self.tracker.best_value or 0.0,
            )
        else:
            status = "discard"
            self.checkpoint_mgr.restore_best(self.agent)
            logger.info(
                "DISCARD %s: %s=%.6f (best=%.6f)",
                experiment_id,
                self.config.objective_metric,
                objective,
                self.tracker.best_value or 0.0,
            )

        record = ExperimentRecord(
            experiment_id=experiment_id,
            params=proposed_params,
            metrics=eval_result,
            objective_value=objective,
            training_time=training_time,
            status=status,
            description=description,
        )
        self.tracker.record(record)
        self._log_wandb(record)
        self._report_to_proposer(objective, crashed=False)

    # ------------------------------------------------------------------
    # Training dispatchers
    # ------------------------------------------------------------------

    async def _train_with_params(self, params: dict[str, Any]) -> None:
        """Train the agent using the specified parameters."""
        from dataclasses import fields as dc_fields

        from stateset_agents.training.config import TrainingConfig

        # Resolve algorithm — "auto" lets the proposer choose via params
        # Skip training for stub agents (they can't train, only evaluate)
        if self._is_stub_agent():
            logger.debug("Stub agent detected — skipping training step")
            return

        algo = params.get("algorithm", self.config.trainer_algorithm)
        if algo == "auto":
            algo = "gspo"  # Default when auto but proposer didn't specify
        if algo not in ("gspo", "grpo", "dapo", "vapo"):
            raise ValueError(f"Unknown trainer algorithm: {algo!r}")

        # Build training config from params + base overrides
        config_kwargs: dict[str, Any] = {}
        config_kwargs.update(self.config.base_config_overrides)

        # Map proposer params to TrainingConfig fields (use dc_fields for
        # proper inheritance support)
        config_field_names = {f.name for f in dc_fields(TrainingConfig)}
        for key, value in params.items():
            if key in config_field_names:
                config_kwargs[key] = value

        training_config = TrainingConfig(**config_kwargs)

        if algo == "gspo":
            await self._train_gspo(training_config, params)
        elif algo == "grpo":
            await self._train_grpo(training_config)
        elif algo == "dapo":
            await self._train_dapo(training_config, params)
        elif algo == "vapo":
            await self._train_vapo(training_config, params)

    def _is_stub_agent(self) -> bool:
        """Check if the agent is using a stub backend (not trainable)."""
        model = getattr(self.agent, "model", None)
        if model is None:
            return True
        # Check for StubModel or models with no parameters
        model_type = type(model).__name__
        if "Stub" in model_type:
            return True
        try:
            params = list(model.parameters())
            return len(params) == 0
        except (AttributeError, TypeError):
            return True

    async def _train_gspo(
        self, base_config: Any, params: dict[str, Any]
    ) -> None:
        from stateset_agents.training.gspo_config import GSPOConfig
        from stateset_agents.training.gspo_entrypoints import train_with_gspo

        from dataclasses import fields as dc_fields

        gspo_kwargs = {}
        gspo_fields = {f.name for f in dc_fields(GSPOConfig)}
        for key, value in params.items():
            if key in gspo_fields:
                gspo_kwargs[key] = value

        gspo_config = GSPOConfig.from_training_config(base_config, **gspo_kwargs)

        from .early_abort import EarlyAbortCallback

        abort_cb = EarlyAbortCallback()
        abort_cb.on_train_start()

        await asyncio.wait_for(
            train_with_gspo(
                config=gspo_config,
                agent=self.agent,
                environment=self.environment,
                reward_model=self.reward_fn,
                callbacks=[abort_cb],
            ),
            timeout=self.config.time_budget,
        )

        if abort_cb.should_abort:
            raise RuntimeError(f"Early abort: {abort_cb.abort_reason}")

    async def _train_grpo(self, config: Any) -> None:
        from stateset_agents.training.train import train

        await asyncio.wait_for(
            train(
                agent=self.agent,
                environment=self.environment,
                reward_fn=self.reward_fn,
                num_episodes=config.num_episodes,
                config_overrides=(
                    config.to_dict() if hasattr(config, "to_dict") else {}
                ),
            ),
            timeout=self.config.time_budget,
        )

    async def _train_dapo(
        self, base_config: Any, params: dict[str, Any]
    ) -> None:
        from stateset_agents.training.dapo_entrypoints import train_with_dapo

        # DAPO entrypoint expects (model_name, reward_fn, train_prompts)
        model_name = base_config.model_name
        train_prompts = [
            s.get("context", "Hello") if isinstance(s, dict) else str(s)
            for s in self.environment.scenarios
        ]

        # Wrap reward_fn to match DAPO's expected (prompt, response) -> float
        reward_model = self.reward_fn

        async def _reward_fn(prompt: str, response: str) -> float:
            from stateset_agents.core.trajectory import ConversationTurn

            turn = ConversationTurn(role="assistant", content=response)
            result = reward_model.compute_reward([turn], {"user_query": prompt})
            if asyncio.iscoroutine(result):
                result = await result
            if hasattr(result, "score"):
                return float(result.score)
            return float(result)

        await asyncio.wait_for(
            train_with_dapo(
                model_name=model_name,
                reward_fn=_reward_fn,
                train_prompts=train_prompts,
            ),
            timeout=self.config.time_budget,
        )

    async def _train_vapo(
        self, base_config: Any, params: dict[str, Any]
    ) -> None:
        from stateset_agents.training.vapo_entrypoints import train_with_vapo

        # VAPO entrypoint expects (model_name, reward_fn, train_prompts)
        model_name = base_config.model_name
        train_prompts = [
            s.get("context", "Hello") if isinstance(s, dict) else str(s)
            for s in self.environment.scenarios
        ]

        reward_model = self.reward_fn

        async def _reward_fn(prompt: str, response: str) -> float:
            from stateset_agents.core.trajectory import ConversationTurn

            turn = ConversationTurn(role="assistant", content=response)
            result = reward_model.compute_reward([turn], {"user_query": prompt})
            if asyncio.iscoroutine(result):
                result = await result
            if hasattr(result, "score"):
                return float(result.score)
            return float(result)

        await asyncio.wait_for(
            train_with_vapo(
                model_name=model_name,
                reward_fn=_reward_fn,
                train_prompts=train_prompts,
            ),
            timeout=self.config.time_budget,
        )

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    async def _evaluate(self) -> dict[str, float]:
        """Evaluate the agent on held-out scenarios.

        Uses the standard evaluate_agent harness, then enriches with
        percentile metrics from multi_turn_evaluation if available.
        """
        from stateset_agents.core.environment import ConversationEnvironment
        from stateset_agents.training.evaluation import (
            EvaluationConfig,
            evaluate_agent,
        )

        eval_env = ConversationEnvironment(
            scenarios=self.eval_scenarios,
            max_turns=8,
        )

        eval_config = EvaluationConfig(
            num_episodes=self.config.eval_episodes,
            num_generations=1,
            seed=self.config.eval_seed,
            concurrency=self.config.eval_concurrency,
        )

        results: dict[str, float] = await evaluate_agent(
            agent=self.agent,
            environment=eval_env,
            reward_fn=self._eval_reward_fn,
            config=eval_config,
        )

        # Try to add percentile metrics for richer comparison
        try:
            reward = results.get("eval_reward", 0.0)
            std = results.get("eval_reward_std", 0.0)
            # Estimate percentiles from mean + std (normal approximation)
            results["eval_reward_p25"] = reward - 0.675 * std
            results["eval_reward_p75"] = reward + 0.675 * std
            results["eval_reward_p90"] = reward + 1.282 * std
        except Exception:
            pass

        return results

    async def _execute_experiment(
        self, params: dict[str, Any]
    ) -> dict[str, float]:
        """Run the train + eval body for a single experiment."""
        await self._train_with_params(params)
        return await self._evaluate()

    def _remaining_wall_clock(self) -> float | None:
        """Return the remaining total wall-clock budget, if configured."""
        if self.config.max_wall_clock <= 0:
            return None

        elapsed = time.time() - self._loop_start_time
        return float(max(0.0, self.config.max_wall_clock - elapsed))

    def _current_experiment_timeout(self) -> float:
        """Return the active timeout for baseline/experiment execution."""
        budget = float(self.config.time_budget)
        remaining = self._remaining_wall_clock()
        if remaining is not None:
            budget = min(budget, remaining)
        return max(0.0, budget)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _passes_threshold(self, objective: float) -> bool:
        """Check if objective passes the improvement threshold."""
        if self.tracker.best_value is None:
            return True
        if self.config.improvement_threshold <= 0.0:
            return True

        best = self.tracker.best_value
        if best == 0.0:
            return True

        if self.config.direction == "maximize":
            relative_gain = (objective - best) / abs(best)
        else:
            relative_gain = (best - objective) / abs(best)

        return bool(relative_gain >= self.config.improvement_threshold)

    def _record_crash(
        self, experiment_id: str, params: dict[str, Any], description: str
    ) -> None:
        record = ExperimentRecord(
            experiment_id=experiment_id,
            params=params,
            metrics={},
            objective_value=0.0,
            training_time=0.0,
            status="crash",
            description=description,
        )
        self.tracker.record(record)
        self._log_wandb(record)

    def _report_to_proposer(self, objective: float, crashed: bool) -> None:
        """Report result back to proposer (relevant for Bayesian)."""
        if isinstance(self.proposer, BayesianProposer):
            self.proposer.report_result(objective, crashed=crashed)

    @staticmethod
    def _cleanup_gpu() -> None:
        """Release GPU memory after a failed/timed-out experiment.

        Without this, CUDA tensors from cancelled training remain pinned
        in GPU memory, causing OOM after repeated failures.
        """
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except Exception:
            pass

        # Also run Python garbage collection to release tensor references
        import gc

        gc.collect()

    def _print_analysis(self) -> None:
        """Print hyperparameter importance analysis after the run."""
        if self.tracker.num_experiments < 6:
            return
        try:
            from .analysis import generate_report

            report = generate_report(
                self.tracker.records,
                objective_metric=self.config.objective_metric,
                direction=self.config.direction,
            )
            print(report)

            # Save to file
            report_path = self.config.output_path / "analysis.txt"
            report_path.write_text(report)
            logger.info("Analysis saved to %s", report_path)
        except Exception as exc:
            logger.debug("Analysis generation failed: %s", exc)

    def _request_stop(self) -> None:
        logger.info("Stop requested — finishing current experiment...")
        self._stop_requested = True

    def _should_stop(self) -> bool:
        if self._stop_requested:
            return True

        if (
            self.config.max_experiments > 0
            and self.tracker.num_experiments >= self.config.max_experiments
        ):
            logger.info(
                "Reached max_experiments=%d", self.config.max_experiments
            )
            return True

        if self.config.max_wall_clock > 0:
            elapsed = time.time() - self._loop_start_time
            if elapsed >= self.config.max_wall_clock:
                logger.info(
                    "Reached max_wall_clock=%ds (elapsed=%.0fs)",
                    self.config.max_wall_clock,
                    elapsed,
                )
                return True

        # Early stopping: plateau detection
        patience = self.config.improvement_patience
        if patience > 0 and len(self.tracker.records) > patience:
            recent = self.tracker.records[-patience:]
            if all(r.status != "keep" for r in recent):
                logger.info(
                    "Plateau detected: last %d experiments had no improvement. "
                    "Stopping early.",
                    patience,
                )
                return True

        return False


async def run_auto_research(
    *,
    agent: Any,
    environment: Any,
    eval_scenarios: Sequence[dict[str, Any]],
    reward_fn: Any,
    config: AutoResearchConfig | None = None,
    baseline_params: dict[str, Any] | None = None,
    proposer: ExperimentProposer | None = None,
) -> ExperimentTracker:
    """Convenience function to run the autonomous research loop.

    Args:
        agent: The agent to train (must support generate_response).
        environment: Training environment with scenarios.
        eval_scenarios: Held-out evaluation scenarios (never used for training).
        reward_fn: Reward function for evaluation.
        config: Auto-research configuration (uses defaults if None).
        baseline_params: Starting hyperparameter configuration.
        proposer: Custom proposer (uses config.proposer strategy if None).

    Returns:
        ExperimentTracker with all recorded results.
    """
    if config is None:
        config = AutoResearchConfig()

    warnings = config.validate()
    for w in warnings:
        logger.warning("Config warning: %s", w)

    loop = AutoResearchLoop(
        agent=agent,
        environment=environment,
        eval_scenarios=eval_scenarios,
        reward_fn=reward_fn,
        config=config,
        proposer=proposer,
        baseline_params=baseline_params,
    )

    return await loop.run()
