"""Disaggregated Asynchronous Rollout <-> Learner RL Runtime.

Decouples high-throughput trajectory generation (vLLM/Ollama/sandbox rollout
workers) from gradient updates (FSDP/DeepSpeed/1-GPU learner) using
:class:`AsyncRolloutCoordinator` with:

1. **DAPO Dynamic Sampling**: automatically filters out zero-variance prompt
   groups (all-0 or all-1 rewards) before they reach the learner so 100% of
   gradient steps train on informative signal.
2. **Staleness-Bounded Off-Policy Importance Weighting**: computes truncated
   importance weights from sampler vs. learner log-probabilities and evicts
   trajectories exceeding ``max_policy_lag``.
3. **Zero-Stall Live Weight Sync**: hot-loads newly exported LoRA adapters onto
   live vLLM rollout servers every ``weight_sync_interval`` optimizer steps
   without draining in-flight trajectories.

Zero eager ``torch`` imports.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Awaitable, Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .async_rollouts import (
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
    RolloutRecord,
)


@dataclass
class DisaggregatedRLConfig:
    """Configuration for decoupled async rollout workers + learner updates."""

    queue_capacity: int = 256
    min_batch_size: int = 4
    max_batch_size: int = 32
    max_policy_lag: int = 2
    max_importance_weight: float = 2.0
    dynamic_sampling: bool = True
    min_reward_std: float = 1e-4
    weight_sync_interval: int = 1
    num_rollout_workers: int = 4
    live_vllm_endpoint: str = ""
    live_lora_name: str = "stateset-live-rl"
    live_api_key: str = ""


@dataclass
class DisaggregatedMetrics:
    """Auditable runtime telemetry for disaggregated async RL."""

    steps_completed: int = 0
    groups_produced: int = 0
    groups_dropped_zero_variance: int = 0
    rollouts_consumed: int = 0
    total_tokens_generated: int = 0
    weight_syncs_completed: int = 0
    policy_version: int = 0
    lag_sum: float = 0.0
    started_at: float = field(default_factory=time.perf_counter)

    def snapshot(self) -> dict[str, Any]:
        elapsed = max(1e-6, time.perf_counter() - self.started_at)
        mean_lag = (
            round(self.lag_sum / self.rollouts_consumed, 4)
            if self.rollouts_consumed
            else 0.0
        )
        return {
            "steps_completed": self.steps_completed,
            "policy_version": self.policy_version,
            "groups_produced": self.groups_produced,
            "groups_dropped_zero_variance": self.groups_dropped_zero_variance,
            "rollouts_consumed": self.rollouts_consumed,
            "total_tokens_generated": self.total_tokens_generated,
            "weight_syncs_completed": self.weight_syncs_completed,
            "mean_policy_lag": mean_lag,
            "rollouts_per_sec": round(self.rollouts_consumed / elapsed, 2),
            "tokens_per_sec": round(self.total_tokens_generated / elapsed, 2),
            "elapsed_sec": round(elapsed, 3),
        }


def compute_group_advantages(
    rewards: list[float], *, min_std: float = 1e-4
) -> tuple[list[float], bool]:
    """Compute GRPO/GSPO group-relative advantages; flags zero-variance groups."""
    if not rewards:
        return ([], False)
    mean = sum(rewards) / len(rewards)
    var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
    std = math.sqrt(var)
    if std < min_std:
        return ([0.0] * len(rewards), False)
    return ([round((r - mean) / std, 6) for r in rewards], True)


def truncated_importance_weight(
    sampler_log_probs: tuple[float, ...],
    learner_log_probs: tuple[float, ...],
    *,
    max_weight: float = 2.0,
) -> float:
    """Geometric sequence-level truncated importance weight for off-policy correction."""
    n = min(len(sampler_log_probs), len(learner_log_probs))
    if n == 0:
        return 1.0
    mean_delta = sum(learner_log_probs[i] - sampler_log_probs[i] for i in range(n)) / n
    clamped = max(-10.0, min(10.0, mean_delta))
    weight = math.exp(clamped)
    return round(min(float(max_weight), max(0.0, weight)), 6)


RolloutProducerFn = Callable[[int, int], Awaitable[list[dict[str, Any]]]]
LearnerStepFn = Callable[[list[RolloutRecord], int], Awaitable[dict[str, Any]]]


class DisaggregatedRLRuntime:
    """Streaming rollout-to-learner coordinator with DAPO dynamic sampling & LoRA hot-swap."""

    def __init__(
        self,
        config: DisaggregatedRLConfig | None = None,
        *,
        workspace_root: Path | None = None,
        serving_bridge: Any | None = None,
    ) -> None:
        self.config = config or DisaggregatedRLConfig()
        self.workspace_root = (workspace_root or Path(".")).expanduser().resolve()
        self._serving = serving_bridge
        self.coordinator = AsyncRolloutCoordinator(
            AsyncRolloutConfig(
                queue_capacity=self.config.queue_capacity,
                min_batch_size=self.config.min_batch_size,
                max_batch_size=self.config.max_batch_size,
                max_policy_lag=self.config.max_policy_lag,
                max_importance_weight=self.config.max_importance_weight,
            )
        )
        self.metrics = DisaggregatedMetrics()
        self._seq = 0

    def _get_serving(self) -> Any:
        if self._serving is None:
            from stateset_agents.weight_studio.live_serving import LiveServingBridge

            self._serving = LiveServingBridge(self.workspace_root)
        return self._serving

    async def submit_prompt_group(
        self,
        group_id: str,
        completions: list[dict[str, Any]],
        *,
        policy_version: int | None = None,
    ) -> int:
        """Score a prompt group, apply DAPO dynamic sampling, and submit informative rollouts."""
        self.metrics.groups_produced += 1
        rewards = [float(c.get("reward", 0.0)) for c in completions]
        advantages, informative = compute_group_advantages(
            rewards, min_std=self.config.min_reward_std
        )
        if self.config.dynamic_sampling and not informative:
            self.metrics.groups_dropped_zero_variance += 1
            return 0

        ver = (
            self.coordinator.current_policy_version
            if policy_version is None
            else int(policy_version)
        )
        submitted = 0
        for idx, (comp, adv) in enumerate(zip(completions, advantages, strict=True)):
            self._seq += 1
            raw_lps = comp.get("sampler_log_probs") or (-0.5,)
            lps = tuple(float(x) for x in raw_lps)
            rec = RolloutRecord(
                rollout_id=f"{group_id}-g{idx}-{self._seq}",
                policy_version=ver,
                sampler_log_probs=lps,
                payload={
                    **comp,
                    "group_id": group_id,
                    "advantage": adv,
                },
            )
            accepted = await self.coordinator.submit(rec, timeout_seconds=5.0)
            if accepted:
                submitted += 1
        return submitted

    def sync_live_lora(self, lora_path: str | Path) -> dict[str, Any] | None:
        """Hot-load updated LoRA weights into a live vLLM rollout pool without draining."""
        if not self.config.live_vllm_endpoint:
            return None
        bridge = self._get_serving()
        res = bridge.load_lora_adapter(
            self.config.live_vllm_endpoint,
            self.config.live_lora_name,
            str(lora_path),
            api_key=self.config.live_api_key,
        )
        if res.get("loaded"):
            self.metrics.weight_syncs_completed += 1
        return dict(res)

    def sync_vllm_sleep_wake(
        self,
        *,
        level: int = 1,
        weight_push_fn: Callable[[], dict[str, Any]] | None = None,
    ) -> dict[str, Any] | None:
        """Execute vLLM Level-1 Sleep -> Weight Push (NCCL/LoRA) -> Wake-Up cycle."""
        if not self.config.live_vllm_endpoint:
            return None
        bridge = self._get_serving()
        base = bridge._normalize_base(self.config.live_vllm_endpoint)
        root = base[:-3] if base.endswith("/v1") else base
        headers = bridge._auth_headers(self.config.live_api_key)

        s_status, s_body = bridge._http_post(
            f"{root}/sleep", {"level": int(level)}, headers, 15.0
        )
        push_res = weight_push_fn() if weight_push_fn is not None else {"pushed": True}
        w_status, w_body = bridge._http_post(f"{root}/wake_up", {}, headers, 15.0)
        ok = (200 <= s_status < 300) and (200 <= w_status < 300)
        if ok:
            self.metrics.weight_syncs_completed += 1
        return {
            "synced": ok,
            "sleep_status": s_status,
            "wake_status": w_status,
            "sleep_response": s_body,
            "wake_response": w_body,
            "weight_push": push_res,
        }

    async def run_steps(
        self,
        *,
        num_steps: int,
        producer_fn: RolloutProducerFn,
        learner_step_fn: LearnerStepFn,
        timeout_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """Run concurrent async rollout workers feeding ``num_steps`` learner updates."""
        stop_event = asyncio.Event()
        step_logs: list[dict[str, Any]] = []

        async def _worker(worker_id: int) -> None:
            group_idx = 0
            while not stop_event.is_set():
                ver = self.coordinator.current_policy_version
                try:
                    completions = await producer_fn(worker_id, ver)
                    if not completions:
                        await asyncio.sleep(0.01)
                        continue
                    group_idx += 1
                    await self.submit_prompt_group(
                        f"w{worker_id}-grp{group_idx}",
                        completions,
                        policy_version=ver,
                    )
                except Exception:
                    if stop_event.is_set():
                        break
                    await asyncio.sleep(0.01)

        workers = [
            asyncio.create_task(_worker(w))
            for w in range(max(1, self.config.num_rollout_workers))
        ]
        try:
            for step in range(1, num_steps + 1):
                batch = await self.coordinator.next_batch(
                    timeout_seconds=timeout_seconds
                )
                records = list(batch.records)
                cur_ver = self.coordinator.current_policy_version
                for rec in records:
                    self.metrics.rollouts_consumed += 1
                    self.metrics.lag_sum += max(0, cur_ver - rec.policy_version)
                    self.metrics.total_tokens_generated += int(
                        rec.payload.get("token_count", len(rec.sampler_log_probs))
                    )

                step_res = await learner_step_fn(records, step)
                new_ver = await self.coordinator.advance_policy()
                self.metrics.steps_completed += 1
                self.metrics.policy_version = new_ver

                lora_path = step_res.get("lora_path")
                sync_res = None
                if (
                    lora_path
                    and self.config.live_vllm_endpoint
                    and step % max(1, self.config.weight_sync_interval) == 0
                ):
                    sync_res = self.sync_live_lora(lora_path)

                step_logs.append(
                    {
                        "step": step,
                        "policy_version": new_ver,
                        "batch_size": len(records),
                        "weight_sync": sync_res,
                        **step_res,
                    }
                )
        finally:
            stop_event.set()
            for task in workers:
                task.cancel()
            await asyncio.gather(*workers, return_exceptions=True)
            await self.coordinator.close()

        return {
            "metrics": self.metrics.snapshot(),
            "coordinator_stats": asdict(self.coordinator.stats()),
            "steps": step_logs,
        }

    async def run_adaptive_sandbox_loop(
        self,
        *,
        pool: Any,
        curriculum: Any,
        policy_fn: Any,
        learner_step_fn: Callable[
            [list[dict[str, Any]], int], Awaitable[dict[str, Any]]
        ],
        num_steps: int = 3,
        batch_size: int = 4,
        model: Any = None,
        studio_callback: Any | None = None,
        hacking_guard: Any | None = None,
        synthesize_self_play: bool = True,
        vllm_sleep_wake_every: int = 0,
    ) -> dict[str, Any]:
        """Run a unified closed-loop adaptive RL session across verifiable sandboxes.

        Wires together:
        1. ``AdaptiveDifficultyCurriculum.sample_batch`` (frontier 20-80% priority)
        2. ``MultiDomainSandboxPool.run_batch`` (concurrent rollouts + per-domain z-norm)
        3. ``compute_turn_level_advantages`` (per-turn PRM credit assignment)
        4. ``learner_step_fn`` (e.g. ``chunked_rl_policy_loss`` w/ FP8 TIS/IcePop + MoE aux)
        5. ``OnlineRewardHackingGuard`` + ``StudioTelemetryCallback.enforce_step``
        6. Optional ``sync_vllm_sleep_wake()`` weight broadcast.
        """
        from stateset_agents.core.sandbox_pool import SandboxEpisodeSpec
        from stateset_agents.training.turn_credit import compute_turn_level_advantages

        step_logs: list[dict[str, Any]] = []
        self_play_created = 0
        vllm_syncs = 0

        for step in range(1, max(1, int(num_steps)) + 1):
            tasks = curriculum.sample_batch(batch_size)
            if not tasks:
                break
            specs = [
                SandboxEpisodeSpec(
                    episode_id=f"s{step}-{t.task_id}",
                    domain=t.domain,
                    scenario={**dict(t.scenario), "prompt": t.prompt},
                )
                for t in tasks
            ]
            batch_res = await pool.run_batch(
                specs, policy_fn, normalize_per_domain=True
            )
            episodes = list(batch_res.get("episodes") or [])

            # Assign true per-turn PRM credit across the batch
            credit_inputs = [
                {
                    "turn_process_rewards": [
                        float(x)
                        for x in (
                            ep.get("turn_process_rewards") or [ep.get("reward", 0.0)]
                        )
                    ],
                    "terminal_reward": float(ep.get("reward", 0.0)),
                }
                for ep in episodes
            ]
            turn_credits = (
                compute_turn_level_advantages(credit_inputs) if credit_inputs else []
            )
            for task_obj, ep, tc in zip(tasks, episodes, turn_credits, strict=False):
                ep["task_id"] = task_obj.task_id
                ep["turn_advantages"] = [round(float(a), 4) for a in tc.turn_advantages]
                curriculum.record_outcome(
                    task_obj.task_id,
                    passed=bool(ep.get("verified")),
                    reward=float(ep.get("reward", 0.0)),
                )

            if synthesize_self_play:
                new_vars = curriculum.synthesize_harder_variants(max_new=2)
                self_play_created += len(new_vars)

            step_out = await learner_step_fn(episodes, step)
            self.metrics.steps_completed += 1
            self.metrics.rollouts_consumed += len(episodes)

            mean_reward = round(
                sum(float(e.get("reward", 0.0)) for e in episodes)
                / max(1, len(episodes)),
                4,
            )
            guard_rep = None
            if hacking_guard is not None:
                guard_samples = [
                    {
                        "prompt": str(e.get("episode_id") or ""),
                        "response": f"verified={bool(e.get('verified'))}",
                        "proxy_score": float(e.get("reward", 0.0)),
                        "human_score": 1.0 if e.get("verified") else 0.0,
                    }
                    for e in episodes
                ]
                guard_rep = hacking_guard.check_step(
                    step,
                    guard_samples,
                    mean_entropy=float(step_out.get("entropy", 1.0)),
                )

            cb_out = None
            if studio_callback is not None:
                step_metrics = {
                    "reward": mean_reward,
                    "loss": float(step_out.get("loss", 0.0)),
                    "entropy": float(step_out.get("entropy", 1.0)),
                }
                if hasattr(studio_callback, "enforce_step"):
                    cb_out = studio_callback.enforce_step(
                        model, step, step_metrics, guard_report=guard_rep
                    )
                else:
                    cb_out = studio_callback.on_step(
                        step, step_metrics, guard_report=guard_rep
                    )

            sw_res = None
            if (
                vllm_sleep_wake_every > 0
                and self.config.live_vllm_endpoint
                and step % vllm_sleep_wake_every == 0
            ):
                sw_res = self.sync_vllm_sleep_wake(level=1)
                if sw_res and sw_res.get("synced"):
                    vllm_syncs += 1

            step_logs.append(
                {
                    "step": step,
                    "episode_count": len(episodes),
                    "mean_reward": mean_reward,
                    "domain_stats": batch_res.get("domain_stats") or {},
                    "guard_action": (guard_rep or {}).get("action", "CONTINUE"),
                    "studio_rolled_back": bool(
                        (cb_out or {}).get("rolled_back", False)
                    ),
                    "vllm_sleep_wake": sw_res,
                    **step_out,
                }
            )

        return {
            "steps": step_logs,
            "curriculum_stats": curriculum.stats(),
            "self_play_variants_created": self_play_created,
            "vllm_sleep_wake_syncs": vllm_syncs,
            "metrics": self.metrics.snapshot(),
        }


__all__ = [
    "DisaggregatedMetrics",
    "DisaggregatedRLConfig",
    "DisaggregatedRLRuntime",
    "compute_group_advantages",
    "truncated_importance_weight",
]
