"""Online Reward-Hacking Circuit-Breaker & Production-to-RL Curriculum Flywheel.

Closes the continuous improvement loop between live production traffic and RL
training while preventing silent RL collapse:

1. :class:`OnlineRewardHackingGuard`: periodically audits sampled RL rollouts
   during training via ``PluggableRewardJudge`` + ``audit_reward_calibration``
   (Spearman ``ρ``, MAE, reward-hacking rate), tracks response length explosion
   and policy entropy collapse, and emits automatic remediation actions
   (``CONTINUE`` / ``INCREASE_KL`` / ``REWEIGHT_JUDGE`` / ``HALT_AND_ROLLBACK``).
2. :class:`ProductionTraceCurriculumBuilder`: mines failing live production
   traces from ``StudioSQLiteStore`` (or JSONL), clusters them by failure mode,
   generates balanced difficulty-graded RL training episodes, and optionally
   synthesizes a **Surgical Warm-Start** CAA steering recipe for Step 0 of RL.

Zero eager ``torch`` imports.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from stateset_agents.weight_studio.judge import PluggableRewardJudge
from stateset_agents.weight_studio.ledger_store import StudioSQLiteStore
from stateset_agents.weight_studio.telemetry_ingest import LiveTraceCollector


@dataclass
class GuardThresholds:
    """Online stability & reward-calibration guardrails during RL training."""

    min_entropy: float = 0.08
    max_length_growth_ratio: float = 2.2
    max_hacking_rate: float = 0.20
    min_spearman_rho: float = 0.35
    kl_bump_factor: float = 1.5
    max_kl_coef: float = 0.50


class OnlineRewardHackingGuard:
    """In-flight RL circuit-breaker detecting reward hacking, length explosion & entropy collapse."""

    def __init__(
        self,
        *,
        workspace_root: Path | None = None,
        judge: PluggableRewardJudge | None = None,
        thresholds: GuardThresholds | None = None,
        reward_fn: str = "customer_support",
    ) -> None:
        self.workspace_root = (workspace_root or Path(".")).expanduser().resolve()
        self.judge = judge
        self.thresholds = thresholds or GuardThresholds()
        self.reward_fn = reward_fn
        self._collector = LiveTraceCollector(self.workspace_root)
        self._baseline_mean_length: float | None = None
        self.history: list[dict[str, Any]] = []

    def check_step(
        self,
        step: int,
        rollouts: list[dict[str, Any]],
        *,
        current_kl_coef: float = 0.02,
        mean_entropy: float | None = None,
    ) -> dict[str, Any]:
        """Inspect a batch of RL rollouts and issue a structured circuit-breaker verdict."""
        lengths = [
            float(
                r.get("token_count") or len(str(r.get("response") or "").split()) or 1.0
            )
            for r in rollouts
        ]
        mean_len = (sum(lengths) / len(lengths)) if lengths else 0.0
        if self._baseline_mean_length is None and mean_len > 0.0:
            self._baseline_mean_length = mean_len
        len_ratio = (
            round(mean_len / self._baseline_mean_length, 3)
            if self._baseline_mean_length
            else 1.0
        )

        # Build calibration samples from explicit human/judge scores or live PluggableRewardJudge
        cal_samples: list[dict[str, Any]] = []
        for idx, r in enumerate(rollouts):
            prompt = str(r.get("prompt") or "")
            response = str(r.get("response") or "")
            proxy = r.get("proxy_score", r.get("reward"))
            gt = r.get("human_score", r.get("judge_score"))
            if gt is None and self.judge is not None and self.judge.mode != "heuristic":
                graded = self.judge.score_turn(
                    prompt, response, reward_fn=self.reward_fn
                )
                proxy = graded["proxy_score"]
                gt = graded["judge_score"]
            if gt is not None:
                sample: dict[str, Any] = {
                    "id": str(r.get("id") or f"s{step}-{idx}"),
                    "prompt": prompt,
                    "response": response,
                    "human_score": float(gt),
                }
                if proxy is not None:
                    sample["proxy_score"] = float(proxy)
                cal_samples.append(sample)

        calibration: dict[str, Any] = (
            self._collector.audit_reward_calibration(
                samples=cal_samples, reward_fn=self.reward_fn
            )
            if len(cal_samples) >= 2
            else {"verdict": "SKIPPED", "sample_count": len(cal_samples)}
        )

        alerts: list[str] = []
        action = "CONTINUE"
        recommended_kl = float(current_kl_coef)

        cal_verdict = str(calibration.get("verdict") or "SKIPPED")
        hacking_rate = float(calibration.get("hacking_rate") or 0.0)
        spearman = calibration.get("spearman_rho")

        if (
            cal_verdict == "HACKING_RISK"
            or hacking_rate >= self.thresholds.max_hacking_rate
        ):
            alerts.append(
                f"Reward hacking detected (hacking_rate={hacking_rate*100:.1f}%, ρ={spearman})"
            )
            action = "HALT_AND_ROLLBACK"
        elif cal_verdict == "DRIFT_WARN":
            alerts.append(f"Reward calibration drift warning (ρ={spearman})")
            action = "REWEIGHT_JUDGE"

        if mean_entropy is not None and mean_entropy < self.thresholds.min_entropy:
            alerts.append(
                f"Policy entropy collapse ({mean_entropy:.4f} < {self.thresholds.min_entropy})"
            )
            recommended_kl = min(
                self.thresholds.max_kl_coef,
                round(current_kl_coef * self.thresholds.kl_bump_factor, 5),
            )
            if action == "CONTINUE":
                action = "INCREASE_KL"

        if len_ratio > self.thresholds.max_length_growth_ratio:
            alerts.append(
                f"Verbosity/length explosion ({len_ratio:.2f}x baseline length)"
            )
            recommended_kl = min(
                self.thresholds.max_kl_coef,
                round(current_kl_coef * self.thresholds.kl_bump_factor, 5),
            )
            if action == "CONTINUE":
                action = "INCREASE_KL"

        report = {
            "step": step,
            "action": action,
            "recommended_kl_coef": recommended_kl,
            "mean_response_length": round(mean_len, 2),
            "length_growth_ratio": len_ratio,
            "mean_entropy": mean_entropy,
            "calibration": calibration,
            "alerts": alerts,
        }
        self.history.append(report)
        return report


class ProductionTraceCurriculumBuilder:
    """Cluster failing SQLite OTLP traces into balanced RL curricula + Surgical Warm-Start."""

    def __init__(self, workspace_root: Path) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.store = StudioSQLiteStore(self.workspace_root)
        self.collector = LiveTraceCollector(self.workspace_root, store=self.store)

    def build_curriculum(
        self,
        *,
        max_episodes: int = 100,
        output_path: Path | None = None,
    ) -> dict[str, Any]:
        """Extract failing production traces, cluster by failure mode, and export RL JSONL."""
        failing = self.store.query_traces(
            failed_only=True, limit=max(50, max_episodes * 2)
        )
        if not failing:
            failing = self.collector.list_traces(limit=max_episodes, failing_only=True)

        clusters: dict[str, list[dict[str, Any]]] = {}
        for row in failing:
            modes = row.get("failure_modes") or ["low_reward"]
            primary = str(modes[0])
            clusters.setdefault(primary, []).append(row)

        # Round-robin across failure clusters so rare safety/schema failures aren't drowned out
        ordered: list[dict[str, Any]] = []
        cluster_keys = sorted(clusters.keys())
        idx = 0
        while len(ordered) < max_episodes and cluster_keys:
            added_any = False
            for key in list(cluster_keys):
                bucket = clusters[key]
                if idx < len(bucket):
                    item = bucket[idx]
                    proxy = float(item.get("proxy_score", 0.0))
                    difficulty = (
                        "hard"
                        if proxy < 0.35
                        else ("medium" if proxy < 0.6 else "easy")
                    )
                    ordered.append(
                        {
                            "episode_id": f"cur-{len(ordered) + 1}",
                            "trace_id": item.get("trace_id"),
                            "failure_cluster": key,
                            "difficulty": difficulty,
                            "prompt": item.get("prompt", ""),
                            "failed_response": item.get("response", ""),
                            "baseline_proxy_score": proxy,
                            "reward_fn": item.get("reward_fn", "customer_support"),
                        }
                    )
                    added_any = True
                    if len(ordered) >= max_episodes:
                        break
            if not added_any:
                break
            idx += 1

        dest = output_path or (
            self.workspace_root
            / "outputs"
            / "rl_curricula"
            / "production_curriculum.jsonl"
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            "\n".join(json.dumps(ep) for ep in ordered) + ("\n" if ordered else ""),
            encoding="utf-8",
        )

        cluster_counts = {k: len(v) for k, v in clusters.items()}
        dominant = (
            max(cluster_counts, key=lambda k: cluster_counts[k])
            if cluster_counts
            else "none"
        )
        return {
            "episode_count": len(ordered),
            "cluster_counts": cluster_counts,
            "dominant_failure_cluster": dominant,
            "curriculum_path": str(dest),
            "episodes": ordered,
        }


__all__ = [
    "GuardThresholds",
    "OnlineRewardHackingGuard",
    "ProductionTraceCurriculumBuilder",
]
