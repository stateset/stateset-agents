"""
Distributed Training Support for GRPO Agent Framework

This module provides distributed training capabilities including:
- Multi-GPU training with DDP
- Multi-node training
- Gradient accumulation across devices
- Efficient communication strategies
"""

import inspect
import logging
import os
from collections.abc import Callable, Iterator
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, cast

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_checkpoint
from accelerate import Accelerator
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    BackwardPrefetch,
    CPUOffload,
    MixedPrecision,
    ShardingStrategy,
)
from torch.nn.parallel import DistributedDataParallel as DDP

from stateset_agents.core.agent import MultiTurnAgent
from stateset_agents.core.environment import Environment
from stateset_agents.core.reward import RewardFunction

from .callbacks import notify_checkpoint_saved
from .checkpoint_state import save_training_checkpoint
from .config import TrainingConfig
from .trainer import MultiTurnGRPOTrainer

logger = logging.getLogger(__name__)


@dataclass
class DistributedConfig:
    """Configuration for distributed training.

    An unset accumulation count inherits ``TrainingConfig``. An explicit value
    overrides it for both the base trainer and Accelerate.
    """

    # Basic distributed settings
    backend: str = "nccl"  # nccl, gloo, mpi
    world_size: int = 1
    rank: int = 0
    local_rank: int = 0
    master_addr: str = "localhost"
    master_port: str = "29500"

    # Training strategy
    strategy: str = "ddp"  # ddp, fsdp, deepspeed
    gradient_accumulation_steps: int | None = None  # Inherit TrainingConfig

    # FSDP specific
    fsdp_sharding_strategy: str = "full_shard"  # full_shard, shard_grad_op, no_shard
    fsdp_cpu_offload: bool = False
    fsdp_backward_prefetch: bool = True
    fsdp_mixed_precision: bool = True
    fsdp_activation_checkpointing: bool = False
    fsdp_auto_wrap_min_params: int = 1_000_000  # Auto-wrap layers with >= 1M params
    fsdp_sync_module_states: bool = True  # Sync params on init for correct sharding
    fsdp_limit_all_gathers: bool = True  # Reduce all-gather communication overhead

    # DeepSpeed specific
    deepspeed_config: dict[str, Any] | None = None

    # Communication optimization
    gradient_as_bucket_view: bool = True
    find_unused_parameters: bool = False
    broadcast_buffers: bool = True

    # Checkpointing
    checkpoint_parallel: bool = True
    checkpoint_dir: str = "./distributed_checkpoints"


def _resolve_fsdp_sharding_strategy(
    value: str | ShardingStrategy,
) -> ShardingStrategy:
    """Map a config value such as ``"full_shard"`` to a ``ShardingStrategy``.

    FSDP requires the enum: a raw string is stored as-is and later fails when
    FSDP reads ``sharding_strategy.name`` on multi-rank runs.
    """
    if isinstance(value, ShardingStrategy):
        return value
    key = value.strip().upper()
    try:
        return ShardingStrategy[key]
    except KeyError:
        choices = ", ".join(s.name.lower() for s in ShardingStrategy)
        raise ValueError(
            f"Unknown fsdp_sharding_strategy {value!r}; expected one of: {choices}"
        ) from None


class DistributedTrainer(MultiTurnGRPOTrainer):
    """
    Distributed GRPO trainer supporting multi-GPU and multi-node training
    """

    def __init__(
        self,
        agent: MultiTurnAgent,
        environment: Environment,
        reward_fn: RewardFunction | None = None,
        config: TrainingConfig | None = None,
        distributed_config: DistributedConfig | None = None,
        **kwargs,
    ):
        self.distributed_config = distributed_config or DistributedConfig()
        self.accelerator: Accelerator | None = None
        self.is_main_process = True
        self.config = config or TrainingConfig()
        accumulation_steps = self.distributed_config.gradient_accumulation_steps
        if accumulation_steps is None:
            accumulation_steps = self.config.gradient_accumulation_steps
        if (
            isinstance(accumulation_steps, bool)
            or not isinstance(accumulation_steps, int)
            or accumulation_steps < 1
        ):
            raise ValueError("gradient_accumulation_steps must be a positive integer")
        self.config = replace(
            self.config, gradient_accumulation_steps=accumulation_steps
        )
        self.distributed_config.gradient_accumulation_steps = accumulation_steps
        if (
            self.distributed_config.strategy == "accelerate"
            and accumulation_steps > 1
            and self.config.num_gradient_updates > 1
        ):
            raise ValueError(
                "Accelerate cannot combine gradient accumulation with "
                "multiple inner optimizer updates"
            )

        # Initialize distributed environment
        self._init_distributed()

        # Initialize base trainer
        super().__init__(agent, environment, reward_fn, self.config, **kwargs)

    def _init_distributed(self):
        """Initialize distributed training environment"""

        if self.distributed_config.strategy == "accelerate":
            # Use Accelerate for automatic distributed setup
            self.accelerator = Accelerator(
                gradient_accumulation_steps=self.distributed_config.gradient_accumulation_steps,
                mixed_precision=(
                    "bf16" if self.config.bf16 else "fp16" if self.config.fp16 else "no"
                ),
                log_with=["wandb"] if self.config.report_to == "wandb" else None,
            )
            self.is_main_process = self.accelerator.is_main_process
            self.distributed_config.world_size = self.accelerator.num_processes
            self.distributed_config.rank = self.accelerator.process_index
            self.distributed_config.local_rank = self.accelerator.local_process_index

        else:
            # Manual distributed setup
            if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
                self.distributed_config.rank = int(os.environ["RANK"])
                self.distributed_config.world_size = int(os.environ["WORLD_SIZE"])
                self.distributed_config.local_rank = int(os.environ["LOCAL_RANK"])

            if self.distributed_config.world_size > 1:
                if torch.cuda.is_available():
                    torch.cuda.set_device(self.distributed_config.local_rank)
                if not dist.is_initialized():
                    dist.init_process_group(
                        backend=self.distributed_config.backend,
                        world_size=self.distributed_config.world_size,
                        rank=self.distributed_config.rank,
                    )
                elif (
                    dist.get_world_size() != self.distributed_config.world_size
                    or dist.get_rank() != self.distributed_config.rank
                    or str(dist.get_backend()) != self.distributed_config.backend
                ):
                    raise ValueError(
                        "Distributed topology differs from the process group"
                    )
                self.is_main_process = self.distributed_config.rank == 0

                logger.info(
                    f"Initialized distributed training: "
                    f"rank {self.distributed_config.rank}/{self.distributed_config.world_size}"
                )

    def _data_parallel_rank_world(self) -> tuple[int, int]:
        return (
            int(self.distributed_config.rank),
            max(1, int(self.distributed_config.world_size)),
        )

    async def initialize(self):
        """Initialize distributed trainer"""
        await super().initialize()

        # Wrap model for distributed training
        if self.distributed_config.strategy == "accelerate":
            # Accelerate also places and wraps single-process models.
            self._wrap_model_accelerate()
        elif self.distributed_config.world_size > 1:
            if self.distributed_config.strategy == "fsdp":
                self._wrap_model_fsdp()
            elif self.distributed_config.strategy == "ddp":
                self._wrap_model_ddp()
            elif self.distributed_config.strategy == "deepspeed":
                self._wrap_model_deepspeed()

    def _wrap_model_ddp(self):
        """Wrap model with DistributedDataParallel"""
        device = next(self.agent.model.parameters()).device
        device_ids = (
            [self.distributed_config.local_rank] if device.type == "cuda" else None
        )
        self.agent.model = DDP(
            self.agent.model,
            device_ids=device_ids,
            output_device=self.distributed_config.local_rank if device_ids else None,
            gradient_as_bucket_view=self.distributed_config.gradient_as_bucket_view,
            find_unused_parameters=self.distributed_config.find_unused_parameters,
            broadcast_buffers=self.distributed_config.broadcast_buffers,
        )
        self._ddp_replica_consensus_pending = True
        logger.info("Model wrapped with DistributedDataParallel")

    def _wrap_model_fsdp(self):
        """Wrap model with FullyShardedDataParallel.

        Features:
        - Auto-wrap policy for transformer layers with configurable param threshold
        - Activation checkpointing to reduce memory usage (2-3x memory savings)
        - Mixed precision with configurable dtype
        - CPU offloading for training models larger than GPU memory
        - Synchronized module states for correct initialization
        """
        from functools import partial

        # Configure FSDP components
        cpu_offload = CPUOffload(
            offload_params=self.distributed_config.fsdp_cpu_offload
        )

        backward_prefetch = None
        if self.distributed_config.fsdp_backward_prefetch:
            backward_prefetch = BackwardPrefetch.BACKWARD_PRE

        mixed_precision = None
        if self.distributed_config.fsdp_mixed_precision:
            mixed_precision = MixedPrecision(
                param_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                reduce_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
                buffer_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            )

        # Auto-wrap policy: wrap layers with >= min_params parameters.
        # This gives FSDP proper per-layer sharding granularity instead of
        # wrapping the entire model as one flat unit.
        auto_wrap_policy: Callable[..., bool] | None = None
        min_params = self.distributed_config.fsdp_auto_wrap_min_params
        if min_params > 0:
            try:
                from torch.distributed.fsdp.wrap import size_based_auto_wrap_policy

                auto_wrap_policy = partial(
                    size_based_auto_wrap_policy, min_num_params=min_params
                )
                logger.info("FSDP auto-wrap: layers >= %d params", min_params)
            except ImportError:
                logger.debug("size_based_auto_wrap_policy not available")

        # Wrap model (``auto_wrap_policy=None`` is FSDP's own default).
        self.agent.model = FSDP(
            self.agent.model,
            cpu_offload=cpu_offload,
            backward_prefetch=backward_prefetch,
            mixed_precision=mixed_precision,
            sharding_strategy=_resolve_fsdp_sharding_strategy(
                self.distributed_config.fsdp_sharding_strategy
            ),
            device_id=torch.cuda.current_device(),
            sync_module_states=self.distributed_config.fsdp_sync_module_states,
            limit_all_gathers=self.distributed_config.fsdp_limit_all_gathers,
            auto_wrap_policy=auto_wrap_policy,
        )

        # Activation checkpointing: trade compute for memory by recomputing
        # activations during backward pass instead of storing them.
        if self.distributed_config.fsdp_activation_checkpointing:
            try:
                from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
                    CheckpointImpl,
                    apply_activation_checkpointing,
                    checkpoint_wrapper,
                )

                # Apply to all FSDP-wrapped sub-modules
                non_reentrant_wrapper = partial(
                    checkpoint_wrapper,
                    checkpoint_impl=CheckpointImpl.NO_REENTRANT,
                )
                apply_activation_checkpointing(
                    self.agent.model,
                    checkpoint_wrapper_fn=non_reentrant_wrapper,
                )
                logger.info("FSDP activation checkpointing enabled")
            except ImportError:
                logger.warning(
                    "Activation checkpointing requested but PyTorch version "
                    "does not support it — skipping"
                )

        logger.info("Model wrapped with FullyShardedDataParallel")

    def _wrap_model_deepspeed(self):
        """Wrap model with DeepSpeed"""
        try:
            import deepspeed

            # Initialize DeepSpeed
            (
                self.agent.model,
                self.optimizer,
                _,
                self.lr_scheduler,
            ) = deepspeed.initialize(
                model=self.agent.model,
                optimizer=self.optimizer,
                config=self.distributed_config.deepspeed_config,
                model_parameters=self.agent.model.parameters(),
            )
            logger.info("Model wrapped with DeepSpeed")

        except ImportError:
            logger.error("DeepSpeed not installed. Install with: pip install deepspeed")
            raise

    def _wrap_model_accelerate(self):
        """Wrap model with Accelerate"""
        if self.accelerator is None or self.optimizer is None:
            raise RuntimeError("Accelerate requires an initialized optimizer")
        if self.lr_scheduler is None:
            self.agent.model, self.optimizer = self.accelerator.prepare(
                self.agent.model, self.optimizer
            )
        else:
            self.agent.model, self.optimizer, self.lr_scheduler = (
                self.accelerator.prepare(
                    self.agent.model, self.optimizer, self.lr_scheduler
                )
            )
        logger.info("Model wrapped with Accelerate")
        self._ddp_replica_consensus_pending = True

    async def training_step(self, trajectory_groups: list[Any]) -> dict[str, Any]:
        """Execute distributed training step"""
        from .learner_preflight import verify_ddp_learner_batch

        admission = verify_ddp_learner_batch(self, trajectory_groups)
        from .ddp_population import empty_batch_metrics, token_forward_context

        if (
            isinstance(self.agent.model, DDP)
            and dist.get_world_size(self.agent.model.process_group) == 1
            and all(
                hasattr(g, "trajectories") and not g.trajectories
                for g in trajectory_groups
            )
        ):
            return empty_batch_metrics(self)
        if admission is not None and admission["global_empty"]:
            return empty_batch_metrics(self)
        token_context = (
            token_forward_context(self.agent.model, admission["rollout_source"])
            if admission is not None and admission["rollout_path"] == "token"
            else nullcontext()
        )
        accelerator = self.accelerator
        try:
            with self._ddp_step_buffers(), token_context:
                if (
                    self.distributed_config.strategy == "accelerate"
                    and accelerator is not None
                ):
                    # Use Accelerate's gradient accumulation
                    with accelerator.accumulate(self.agent.model):
                        metrics: dict[str, Any] = await super().training_step(
                            trajectory_groups
                        )
                else:
                    metrics = await super().training_step(trajectory_groups)

        except BaseException:
            self._ddp_restore_failed = True
            raise

        # Synchronize metrics across processes
        if self.distributed_config.world_size > 1:
            metrics = self._sync_metrics(metrics)

        return metrics

    def _record_training_metrics(self, result: dict[str, Any], seconds: float) -> None:
        """Finish local telemetry together before any peer starts another step."""
        if not isinstance(self.agent.model, DDP):
            return super()._record_training_metrics(result, seconds)
        from .ddp_control import DDPControlGroup
        from .diagnostics import diagnostic_control

        group = DDPControlGroup(self, operation="diagnostics")
        record = super()._record_training_metrics
        try:
            group.run("observe", lambda: record(result, seconds))
            group.run("guard", lambda: diagnostic_control(self), agree=True)
        except BaseException:
            # The learner may already have applied an update. A complete reload
            # repairs both optimizer state and the failed observation history.
            self._ddp_restore_failed = True
            raise

    @contextmanager
    def _ddp_step_buffers(self) -> Iterator[None]:
        """Broadcast buffers once before uneven local forward chunks.

        DDP's per-forward broadcasts would otherwise collide with a peer's
        loss/count collective when ranks need different numbers of forwards.
        Keep the user's buffer-sync preference and restore it on all exits.
        """
        model = self.agent.model
        if not isinstance(model, DDP) or not model.broadcast_buffers:
            yield
            return
        with torch.no_grad():
            source = dist.get_global_rank(model.process_group, 0)
            for buffer in model.module.buffers():
                dist.broadcast(buffer, src=source, group=model.process_group)
        previous = model.broadcast_buffers
        try:
            model.broadcast_buffers = False
            yield
        finally:
            model.broadcast_buffers = previous

    def _sync_metrics(self, metrics: dict[str, Any]) -> dict[str, Any]:
        """Synchronize metrics across all processes"""
        if isinstance(self.agent.model, DDP):
            from .metric_reduction import reduce_ddp_metrics

            return reduce_ddp_metrics(metrics, self.agent.model)

        synced_metrics: dict[str, Any] = {}
        accelerator = self.accelerator
        if accelerator is not None:
            device = accelerator.device
        else:
            device = next(self.agent.model.parameters()).device
        for key, value in metrics.items():
            if isinstance(value, (int, float, bool)):
                tensor = torch.tensor(float(value), device=device)
                if accelerator is not None:
                    tensor = accelerator.reduce(tensor, reduction="mean")
                else:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
                    tensor /= self.distributed_config.world_size
                synced_metrics[key] = (
                    tensor.item() == 1.0 if isinstance(value, bool) else tensor.item()
                )
            else:
                synced_metrics[key] = value

        return synced_metrics

    def _maybe_handle_task_switch(self, task_id: str | None) -> None:
        if isinstance(self.agent.model, DDP):
            from .ddp_continual import handle_ddp_task

            handle_ddp_task(self, task_id)
        else:
            super()._maybe_handle_task_switch(task_id)

    def _apply_optimizer_step(self, torch: Any, *, gradient_scale: float = 1.0) -> bool:
        if not isinstance(self.agent.model, DDP):
            return super()._apply_optimizer_step(torch, gradient_scale=gradient_scale)
        from .checkpoint_state import _prepare_normalization, _validate_gradients
        from .ddp_control import DDPControlGroup

        group = DDPControlGroup(self, operation="optimizer-step")

        def validate() -> None:
            parameters = dict(self.agent.model.named_parameters())
            state = self._grpo_accumulation
            # GradScaler must observe reduced nonfinite banks itself, skip the
            # optimizer on every peer and lower the shared scale. Checkpoint
            # finiteness admission would incorrectly turn overflow into failure.
            if self.scaler is not None and self.scaler.is_enabled():
                return
            _validate_gradients(
                {name: p.grad for name, p in parameters.items()},
                parameters,
                torch,
                allow_empty=True,
            )
            if state is not None:
                _prepare_normalization(
                    state, parameters, torch, 1, 1, copy_tensors=False, allow_empty=True
                )

        group.run("gradients", validate)
        previous = getattr(self, "_ddp_restore_failed", False)
        self._ddp_restore_failed = True
        applied = group.run(
            "apply",
            lambda: super(DistributedTrainer, self)._apply_optimizer_step(
                torch, gradient_scale=gradient_scale
            ),
            agree=True,
        )
        self._ddp_restore_failed = previous
        return bool(applied)

    async def train(self) -> Any:
        """Require full restoration after any failed native training loop."""
        native = isinstance(self.agent.model, DDP)
        try:
            return await super().train()
        except BaseException:
            if native:
                self._ddp_restore_failed = True
            raise

    def _prepare_training_loop(self) -> tuple[int, list[dict[str, Any]], int]:
        if not isinstance(self.agent.model, DDP):
            return super()._prepare_training_loop()
        from .ddp_control import DDPControlGroup

        group = DDPControlGroup(self, operation="train-loop")
        plan = group.run("prepare", super()._prepare_training_loop)

        def controls() -> dict[str, Any]:
            windows = {
                name: getattr(self.config, name, default)
                for name, default in (
                    ("logging_steps", 10),
                    ("eval_steps", 50),
                    ("save_steps", 100),
                )
            }
            if any(type(value) is not int or value < 1 for value in windows.values()):
                raise ValueError(
                    "Training log/evaluation/save windows must be positive integers"
                )
            return {
                "protocol": "stateset-native-loop-v3",
                "episodes": plan[0],
                "steps": plan[2],
                "windows": windows,
                "early_stopping": bool(getattr(self.config, "early_stopping", False)),
                "patience": getattr(self.config, "patience", 50),
                "resume": str(getattr(self.config, "resume_from_checkpoint", None)),
            }

        group.run("controls", controls, agree=True)
        return cast(tuple[int, list[dict[str, Any]], int], plan)

    async def _training_phase(
        self, name: str, function: Any, *args: Any, **kwargs: Any
    ) -> Any:
        if not isinstance(self.agent.model, DDP):
            return await super()._training_phase(name, function, *args, **kwargs)
        from .ddp_control import DDPControlGroup

        group = DDPControlGroup(self, operation="train-loop")
        phase = (
            name
            if name
            in {
                "scheduler",
                "training-start",
                "evaluation-scenarios",
                "training-end",
                "tracking-finish",
            }
            else f"episode-{self.current_epoch}:{name}"
        )
        return await group.arun(
            phase,
            lambda: super(DistributedTrainer, self)._training_phase(
                name, function, *args, **kwargs
            ),
        )

    def _resume_start(self, resumed: bool, num_episodes: int) -> int:
        if not isinstance(self.agent.model, DDP):
            return super()._resume_start(resumed, num_episodes)
        from .ddp_control import DDPControlGroup

        return int(
            DDPControlGroup(self, operation="train-loop").run(
                "resume-position",
                lambda: super(DistributedTrainer, self)._resume_start(
                    resumed, num_episodes
                ),
                agree=True,
            )
        )

    async def _generate_training_rollouts(
        self, scenarios: list[dict[str, Any]]
    ) -> list[Any]:
        if not isinstance(self.agent.model, DDP):
            return await super()._generate_training_rollouts(scenarios)
        from .ddp_control import DDPControlGroup
        from .learner_preflight import verify_ddp_learner_batch

        verify_ddp_learner_batch(self, [])
        group = DDPControlGroup(self, operation="train-loop")
        wrapper = self.agent.model

        async def local() -> Any:
            from stateset_agents.core.generation_mode import inference_mode

            self.agent.model = wrapper.module
            self._stateset_native_rollout = True
            try:
                with torch.no_grad(), inference_mode(wrapper.module):
                    return await super(
                        DistributedTrainer, self
                    )._generate_training_rollouts(scenarios)
            finally:
                self.agent.model = wrapper
                self._stateset_native_rollout = False

        with self._ddp_step_buffers():
            return cast(
                list[Any],
                await group.arun(f"episode-{self.current_epoch}:rollouts", local),
            )

    def _rollouts_available(self, groups: list[Any]) -> bool:
        if not isinstance(self.agent.model, DDP):
            return super()._rollouts_available(groups)
        from .ddp_control import DDPControlGroup

        group = DDPControlGroup(self, operation="train-loop")
        local = group.run(
            "rollout-population",
            lambda: bool(groups and any(g.trajectories for g in groups)),
        )
        flag = torch.tensor(int(local), dtype=torch.int64, device=group.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX, group=group.group)
        return bool(flag.item())

    def _should_stop_training(self) -> bool:
        if not isinstance(self.agent.model, DDP):
            return super()._should_stop_training()
        from .ddp_control import DDPControlGroup

        group = DDPControlGroup(self, operation="train-loop")
        group.run("stopping-state", lambda: self.steps_without_improvement, agree=True)
        return bool(group.run("stopping", super()._should_stop_training, agree=True))

    def _optimizer_update_receipt(
        self,
        previous_step: int,
        source: str,
        metrics: dict[str, Any] | None = None,
        task_id: str | None = None,
    ) -> dict[str, Any] | None:
        if not isinstance(self.agent.model, DDP):
            return super()._optimizer_update_receipt(
                previous_step, source, metrics, task_id
            )
        from .ddp_control import DDPControlGroup

        group = DDPControlGroup(self, operation="optimizer-report")
        prepare = super()._optimizer_update_receipt
        receipt = group.run(
            "prepare", lambda: prepare(previous_step, source, metrics, task_id)
        )
        group.run(
            "controls",
            lambda: (
                None
                if receipt is None
                else {
                    name: receipt[name]
                    for name in (
                        "optimizer_step_start",
                        "optimizer_step_end",
                        "optimizer_updates",
                        "update_source",
                        "update_task_id",
                    )
                }
            ),
            agree=True,
        )
        return cast(dict[str, Any] | None, receipt)

    def _episode_metrics(
        self, metrics: dict[str, Any], new_groups: list[Any], training_groups: list[Any]
    ) -> dict[str, Any]:
        if not isinstance(self.agent.model, DDP):
            return super()._episode_metrics(metrics, new_groups, training_groups)
        from .rollout_metrics import summarize_ddp_rollouts

        episode = summarize_ddp_rollouts(self, new_groups, phase="episode")
        result = {
            **metrics,
            "total_reward": episode["reward_mean"],
            "episode_reward": episode["reward_mean"],
            "episode_reward_std": episode["reward_std"],
            "episode_num_trajectories": episode["total_trajectories"],
            "episode_reward_scope": "ddp_global_new_rollouts",
        }
        if self._step_event_due(getattr(self.config, "logging_steps", 10), metrics):
            result["trajectory_metrics"] = summarize_ddp_rollouts(
                self, training_groups, phase="training", details=True
            )
            result["trajectory_metric_scope"] = "ddp_global_training_batch"
        return result

    async def _log_training_metrics(
        self, metrics: dict[str, Any], trajectory_groups: list[Any]
    ) -> None:
        if not isinstance(self.agent.model, DDP):
            await super()._log_training_metrics(metrics, trajectory_groups)
            return
        if not self.wandb_logger:
            return
        kwargs: dict[str, Any] = {
            "losses": self._training_log_scalars(metrics),
            "learning_rate": metrics["learning_rate"],
            "step": self.global_step,
            "trajectory_groups": None,
            "trajectory_metrics": dict(metrics["trajectory_metrics"]),
        }
        method = self.wandb_logger.log_training_step
        try:
            signature = inspect.signature(method)
        except (TypeError, ValueError):
            signature = None
        if signature is not None:
            try:
                signature.bind(**kwargs)
            except TypeError:
                # Older loggers already prefix every numeric loss entry. Keep
                # that API, but provide global scalars instead of local groups.
                kwargs["losses"] = {
                    **kwargs["losses"],
                    **kwargs.pop("trajectory_metrics"),
                }
                signature.bind(**kwargs)
        method(**kwargs)

    def _finish_current_task(self) -> None:
        if isinstance(self.agent.model, DDP):
            from .ddp_continual import handle_ddp_task

            handle_ddp_task(self, self._current_task_id, final=True)
        else:
            super()._finish_current_task()

    async def evaluate(
        self, eval_scenarios: list[dict[str, Any]], num_eval_episodes: int = 10
    ) -> dict[str, float]:
        """Collectively evaluate native DDP partitions and agree on the best save."""
        if isinstance(self.agent.model, DDP):
            from .ddp_evaluation import evaluate_ddp

            return await evaluate_ddp(self, eval_scenarios, num_eval_episodes)
        return await super().evaluate(eval_scenarios, num_eval_episodes)

    def load_checkpoint(self, checkpoint_path: Any, *, trusted: bool = False) -> bool:
        """Invalidate replica-value agreement before any restore attempt."""
        self._ddp_replica_consensus_pending = True
        if isinstance(self.agent.model, DDP):
            from .ddp_checkpoint import load_ddp_checkpoint

            return load_ddp_checkpoint(self, checkpoint_path, trusted=trusted)
        return super().load_checkpoint(checkpoint_path, trusted=trusted)

    def load_runtime_checkpoint(
        self, checkpoint_dir: str, *, trusted: bool = False
    ) -> bool:
        """Require replica-value agreement after a runtime restore attempt."""
        self._ddp_replica_consensus_pending = True
        if isinstance(self.agent.model, DDP):
            from .ddp_checkpoint import load_ddp_checkpoint

            return load_ddp_checkpoint(self, checkpoint_dir, trusted=trusted)
        return super().load_runtime_checkpoint(checkpoint_dir, trusted=trusted)

    def save_runtime_checkpoint(self, output_dir: str) -> None:
        """Collectively save native DDP state, or retain the local runtime path."""
        if isinstance(self.agent.model, DDP):
            from .ddp_checkpoint import save_ddp_checkpoint

            save_ddp_checkpoint(self, output_dir)
        else:
            super().save_runtime_checkpoint(output_dir)

    async def save_checkpoint(
        self, is_best: bool = False, checkpoint_name: str | None = None
    ):
        """Save distributed checkpoint"""

        if isinstance(self.agent.model, DDP):
            from .ddp_checkpoint import save_ddp_checkpoint

            # Native DDP requires every rank's RNG/banks even when the legacy
            # checkpoint_parallel flag is false. This path is always collective.
            name = checkpoint_name or (
                "best-checkpoint" if is_best else f"checkpoint-{self.global_step}"
            )
            save_ddp_checkpoint(
                self, Path(self.distributed_config.checkpoint_dir) / name
            )
            return
        if self.distributed_config.world_size <= 1:
            await super().save_checkpoint(is_best, checkpoint_name)
            return

        if not self.is_main_process and not self.distributed_config.checkpoint_parallel:
            return

        if self.distributed_config.checkpoint_parallel:
            # Parallel checkpoint saving
            await self._save_distributed_checkpoint(is_best, checkpoint_name)
        else:
            # Only main process saves
            await super().save_checkpoint(is_best, checkpoint_name)

    async def _save_distributed_checkpoint(
        self, is_best: bool = False, checkpoint_name: str | None = None
    ):
        """Save a resumable checkpoint under ``checkpoint_dir``.

        DDP and Accelerate replicas hold identical weights, so the main
        process writes the same complete training state as the single-process
        trainer (bare-model weights, optimizer, scheduler, scaler, gradient
        accumulation, running statistics). FSDP shards are written by every
        rank through ``torch.distributed.checkpoint``.
        """

        if checkpoint_name is None:
            checkpoint_name = (
                "best-checkpoint" if is_best else f"checkpoint-{self.global_step}"
            )
        checkpoint_path = Path(self.distributed_config.checkpoint_dir) / checkpoint_name
        distributed = self.distributed_config.world_size > 1

        if self.distributed_config.strategy == "fsdp":
            if self.is_main_process:
                checkpoint_path.mkdir(parents=True, exist_ok=True)
            if distributed:
                dist.barrier()
            dist_checkpoint.save_state_dict(
                state_dict=self.agent.model.state_dict(),
                storage_writer=dist_checkpoint.FileSystemWriter(str(checkpoint_path)),
            )
            if self.is_main_process:
                torch.save(
                    {
                        "global_step": self.global_step,
                        "current_epoch": self.current_epoch,
                        "best_eval_metric": self.best_eval_metric,
                        "distributed_config": self.distributed_config.__dict__,
                    },
                    checkpoint_path / "training_state.pt",
                )
        elif self.is_main_process:
            save_training_checkpoint(
                self, checkpoint_path, require_torch_fn=lambda: torch
            )
        if distributed:
            dist.barrier()

        logger.info(f"Distributed checkpoint saved: {checkpoint_path}")
        if self.is_main_process:
            wandb_logger = getattr(self, "wandb_logger", None)
            if wandb_logger:
                wandb_logger.log_model_checkpoint(
                    str(checkpoint_path), self.global_step, is_best=is_best
                )
            await notify_checkpoint_saved(
                getattr(self, "callbacks", []),
                path=str(checkpoint_path),
                step=int(self.global_step),
                is_best=bool(is_best),
            )

    def cleanup(self):
        """Cleanup distributed resources"""
        if self.distributed_config.world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()

        if self.accelerator:
            self.accelerator.end_training()


# Utility functions for distributed training


def get_distributed_config(
    num_gpus: int | None = None, strategy: str = "ddp", **kwargs
) -> DistributedConfig:
    """Create distributed configuration based on available resources"""

    config = DistributedConfig(strategy=strategy, **kwargs)

    # Auto-detect GPUs if not specified
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    config.world_size = num_gpus

    # Set strategy-specific defaults
    if strategy == "fsdp" and num_gpus >= 4:
        config.fsdp_sharding_strategy = "full_shard"
        config.fsdp_cpu_offload = True
    elif strategy == "deepspeed":
        config.deepspeed_config = get_default_deepspeed_config(num_gpus)

    return config


def get_default_deepspeed_config(num_gpus: int) -> dict[str, Any]:
    """Get default DeepSpeed configuration"""

    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "fp16": {
            "enabled": "auto",
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1,
        },
        "zero_optimization": {
            "stage": 2 if num_gpus >= 4 else 1,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
    }


async def train_distributed(
    agent: MultiTurnAgent,
    environment: Environment,
    config: TrainingConfig,
    num_gpus: int | None = None,
    strategy: str = "ddp",
    **kwargs,
) -> MultiTurnAgent:
    """High-level distributed training function"""

    # Create distributed config
    dist_config = get_distributed_config(num_gpus, strategy)

    # Create distributed trainer
    trainer = DistributedTrainer(
        agent=agent,
        environment=environment,
        config=config,
        distributed_config=dist_config,
        **kwargs,
    )

    try:
        # Initialize and train
        await trainer.initialize()
        trained_agent = cast(MultiTurnAgent, await trainer.train())
        return trained_agent

    finally:
        # Cleanup
        trainer.cleanup()
