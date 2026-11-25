"""Concurrent Multi-Domain Verifiable RL Sandbox Pool with Per-Domain Normalization.

Mixes heterogeneous verifiable agentic environments inside a single concurrent
RL rollout batch:

- ``commerce``: :class:`CommerceLedgerSandbox` (transactional SQLite orders & refunds)
- ``coding``: :class:`CodeVerificationSandbox` (isolated Python unit-test execution)
- ``sql``: :class:`SQLAnalyticsSandbox` (read-only BI / Text-to-SQL analytics)

Applies **per-domain reward normalization** (z-score within each domain bucket)
so easier domains cannot drown out harder multi-step tool-use domains during
multi-task GRPO / GSPO / DAPO training.

Zero eager ``torch`` imports.
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from .mcp_rl_env import (
    BrowserDOMSandbox,
    CodeVerificationSandbox,
    CommerceLedgerSandbox,
    DockerCodeSandbox,
    MCPRLEnvironment,
    RESTAPISandbox,
    SQLAnalyticsSandbox,
)
from .trajectory import ConversationTurn

AgentPolicyFn = Callable[[str, dict[str, Any], int], Awaitable[ConversationTurn]]


@dataclass(frozen=True)
class SandboxEpisodeSpec:
    """One episode task targeted at a specific verifiable sandbox domain."""

    episode_id: str
    domain: str
    scenario: dict[str, Any]
    max_turns: int = 5


class MultiDomainSandboxPool:
    """Concurrent multi-domain MCP sandbox pool with per-domain reward normalization."""

    SUPPORTED_DOMAINS = (
        "commerce",
        "coding",
        "sql",
        "rest_api",
        "docker_code",
        "web_agent",
        "client_matter",
    )

    def __init__(self, *, max_concurrency: int = 16) -> None:
        self.max_concurrency = max(1, int(max_concurrency))

    def _build_env(
        self,
        domain: str,
        max_turns: int,
        *,
        scenario: dict[str, Any] | None = None,
    ) -> MCPRLEnvironment:
        key = domain.strip().lower()
        if key == "commerce":
            sb = CommerceLedgerSandbox()
            return MCPRLEnvironment(
                tool_executor=sb.call_tool,
                state_verifier=lambda st, _h: sb.verify_order_refunded(
                    str(
                        (st.context.get("scenario") or {}).get(
                            "target_order_id", "ORD-77701"
                        )
                    ),
                    should_be_refunded=bool(
                        (st.context.get("scenario") or {}).get("expect_refunded", True)
                    ),
                ),
                max_turns=max_turns,
            )
        if key == "coding":
            cb = CodeVerificationSandbox()
            return MCPRLEnvironment(
                tool_executor=cb.call_tool,
                state_verifier=cb.verify,
                max_turns=max_turns,
            )
        if key == "sql":
            sq = SQLAnalyticsSandbox()
            return MCPRLEnvironment(
                tool_executor=sq.call_tool,
                state_verifier=sq.verify,
                max_turns=max_turns,
            )
        if key == "rest_api":
            ra = RESTAPISandbox()
            return MCPRLEnvironment(
                tool_executor=ra.call_tool,
                state_verifier=ra.verify,
                max_turns=max_turns,
            )
        if key == "docker_code":
            dc = DockerCodeSandbox(use_docker=False)
            return MCPRLEnvironment(
                tool_executor=dc.call_tool,
                state_verifier=dc.verify,
                max_turns=max_turns,
            )
        if key == "web_agent":
            wb = BrowserDOMSandbox()
            return MCPRLEnvironment(
                tool_executor=wb.call_tool,
                state_verifier=wb.verify,
                max_turns=max_turns,
            )
        if key == "client_matter":
            from .client_matter_sandbox import (
                ClientMatterSandbox,
                default_client_matter_tasks,
            )

            tasks = default_client_matter_tasks()
            wanted = str((scenario or {}).get("matter_id") or "").strip()
            chosen = next((t for t in tasks if t.matter_id == wanted), tasks[0])
            cm = ClientMatterSandbox(task=chosen)
            return MCPRLEnvironment(
                tool_executor=cm.call_tool,
                state_verifier=cm.verify,
                max_turns=max_turns,
                process_reward_weight=0.0,
            )
        raise ValueError(
            f"Unsupported sandbox domain {domain!r}; expected one of {self.SUPPORTED_DOMAINS}."
        )

    async def run_episode(
        self, spec: SandboxEpisodeSpec, policy_fn: AgentPolicyFn
    ) -> dict[str, Any]:
        """Run one isolated verifiable multi-turn episode to completion."""
        env = self._build_env(spec.domain, spec.max_turns, scenario=spec.scenario)
        state = await env.reset(spec.scenario)
        obs = str(state.context.get("prompt") or "")
        turn_rewards: list[float] = []
        final_reward = 0.0
        verification: dict[str, Any] = {}

        for turn_idx in range(spec.max_turns):
            action = await policy_fn(
                obs, {"domain": spec.domain, "scenario": spec.scenario}, turn_idx
            )
            state, reward, done, info = await env.step(state, action)
            turn_rewards.append(float(info.get("step_process_reward", 0.0)))
            obs = str(info.get("observation") or "")
            if done:
                final_reward = float(reward)
                verification = dict(info.get("verification") or {})
                break

        return {
            "episode_id": spec.episode_id,
            "domain": spec.domain,
            "turns": state.turn_count,
            "turn_process_rewards": turn_rewards,
            "reward": round(final_reward, 4),
            "verified": bool(verification.get("verified", False)),
            "verification": verification,
        }

    async def run_batch(
        self,
        specs: list[SandboxEpisodeSpec],
        policy_fn: AgentPolicyFn,
        *,
        normalize_per_domain: bool = True,
        advantage_mode: str = "zscore",
    ) -> dict[str, Any]:
        """Execute a mixed-domain batch concurrently and compute per-domain advantages.

        ``advantage_mode`` supports:
        - ``"zscore"`` (default): per-domain z-score when ``normalize_per_domain=True``.
        - ``"group_mean"`` (Dr. GRPO): unscaled ``reward - domain_mean`` with no std division.
        """
        mode = advantage_mode.strip().lower()
        if mode not in {"zscore", "group_mean"}:
            raise ValueError(
                f"Unsupported advantage_mode {advantage_mode!r}; expected 'zscore' or 'group_mean'."
            )
        sem = asyncio.Semaphore(self.max_concurrency)

        async def _bounded(spec: SandboxEpisodeSpec) -> dict[str, Any]:
            async with sem:
                return await self.run_episode(spec, policy_fn)

        results = list(await asyncio.gather(*(_bounded(s) for s in specs)))

        # Per-domain advantage computation (z-score or Dr. GRPO group-mean centering)
        by_domain: dict[str, list[dict[str, Any]]] = {}
        for row in results:
            by_domain.setdefault(str(row["domain"]), []).append(row)

        domain_stats: dict[str, dict[str, Any]] = {}
        for dom, rows in by_domain.items():
            rews = [float(r["reward"]) for r in rows]
            mean = sum(rews) / len(rews)
            std = math.sqrt(sum((x - mean) ** 2 for x in rews) / len(rews))
            pass_rate = sum(1 for r in rows if r["verified"]) / len(rows)
            domain_stats[dom] = {
                "count": len(rows),
                "mean_reward": round(mean, 4),
                "reward_std": round(std, 4),
                "verified_pass_rate": round(pass_rate, 4),
            }
            for r in rows:
                if mode == "zscore" and normalize_per_domain and std >= 1e-4:
                    r["normalized_advantage"] = round(
                        (float(r["reward"]) - mean) / std, 6
                    )
                else:
                    r["normalized_advantage"] = round(float(r["reward"]) - mean, 6)

        return {
            "episode_count": len(results),
            "domain_stats": domain_stats,
            "episodes": results,
        }


__all__ = [
    "MultiDomainSandboxPool",
    "SandboxEpisodeSpec",
]
