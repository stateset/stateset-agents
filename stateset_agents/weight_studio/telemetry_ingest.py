"""Live Production Trace Ingestion (OTLP/JSON + Webhook) & Reward Calibration Guard.

Closes the real-time production observability -> self-healing loop:
1. Ingests live agent turns from native JSON webhooks, OpenAI chat payloads, OR
   OpenTelemetry OTLP/JSON ``resourceSpans`` (``gen_ai.prompt`` / ``gen_ai.completion``
   span attributes).
2. Persists traces durably to ``outputs/weight_studio_traces/live_traces.jsonl``
   and scores every ingested turn automatically so failing traces accumulate
   ready for one-click ``AutonomousSurgeryLoop`` healing.
3. Audits **Reward Calibration & Reward-Hacking Risk** by comparing StateSet's
   fast heuristic proxy reward against human-reviewed or LLM-judge ground-truth
   scores (computing Pearson ``r``, Spearman rank ``ρ``, MAE, high-proxy/low-truth
   hacking cases, and a ``CALIBRATED`` / ``DRIFT_WARN`` / ``HACKING_RISK`` verdict).

All heavy dependencies remain strictly lazy (zero eager ``torch`` import).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._stats import pearson_corr, spearman_corr
from .judge import PluggableRewardJudge
from .ledger_store import StudioSQLiteStore
from .surgery_loop import _score_transcript_turn


def _attr_value(val_obj: Any) -> str:
    if not isinstance(val_obj, dict):
        return str(val_obj or "")
    for key in ("stringValue", "intValue", "doubleValue", "boolValue"):
        if key in val_obj:
            return str(val_obj[key])
    return ""


def extract_traces_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract normalized ``{prompt, response, human_score, trace_id, source}`` rows.

    Accepts:
    - OpenTelemetry OTLP/JSON ``{"resourceSpans": [...]}``
    - Batch ``{"traces": [...]}``
    - OpenAI chat ``{"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}``
    - Single turn ``{"prompt": "...", "response": "..."}``
    """
    extracted: list[dict[str, Any]] = []

    # 1. OpenTelemetry OTLP/JSON resourceSpans
    if isinstance(payload.get("resourceSpans"), list):
        for rs in payload["resourceSpans"]:
            if not isinstance(rs, dict):
                continue
            for ss in rs.get("scopeSpans") or []:
                if not isinstance(ss, dict):
                    continue
                for span in ss.get("spans") or []:
                    if not isinstance(span, dict):
                        continue
                    attrs: dict[str, str] = {}
                    for attr in span.get("attributes") or []:
                        if isinstance(attr, dict) and attr.get("key"):
                            attrs[str(attr["key"])] = _attr_value(attr.get("value"))
                    prompt = (
                        attrs.get("gen_ai.prompt")
                        or attrs.get("gen_ai.input.messages")
                        or attrs.get("input.value")
                        or attrs.get("prompt")
                        or ""
                    )
                    response = (
                        attrs.get("gen_ai.completion")
                        or attrs.get("gen_ai.output.messages")
                        or attrs.get("output.value")
                        or attrs.get("response")
                        or ""
                    )
                    if prompt and response:
                        human_raw = attrs.get("stateset.human_score") or attrs.get(
                            "eval.human_score"
                        )
                        extracted.append(
                            {
                                "trace_id": str(
                                    span.get("traceId")
                                    or span.get("spanId")
                                    or f"otlp-{len(extracted)+1}"
                                ),
                                "source": "otlp",
                                "prompt": prompt,
                                "response": response,
                                "human_score": (
                                    float(human_raw)
                                    if human_raw is not None and human_raw != ""
                                    else None
                                ),
                            }
                        )
        return extracted

    # 2. Batch traces list
    if isinstance(payload.get("traces"), list):
        for idx, item in enumerate(payload["traces"]):
            if isinstance(item, dict):
                sub = extract_traces_from_payload(item)
                for s in sub:
                    if not s.get("trace_id"):
                        s["trace_id"] = f"batch-{idx + 1}"
                extracted.extend(sub)
        return extracted

    # 3. OpenAI messages format
    if isinstance(payload.get("messages"), list):
        last_user = ""
        for msg in payload["messages"]:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role") or "").lower()
            content = str(msg.get("content") or "")
            if role == "user":
                last_user = content
            elif role == "assistant" and last_user:
                extracted.append(
                    {
                        "trace_id": str(
                            payload.get("trace_id")
                            or payload.get("id")
                            or f"chat-{len(extracted)+1}"
                        ),
                        "source": "openai_chat",
                        "prompt": last_user,
                        "response": content,
                        "human_score": (
                            float(payload["human_score"])
                            if payload.get("human_score") is not None
                            else None
                        ),
                    }
                )
        return extracted

    # 4. Single prompt/response turn
    prompt = str(payload.get("prompt") or payload.get("user") or "").strip()
    response = str(payload.get("response") or payload.get("assistant") or "").strip()
    if prompt and response:
        extracted.append(
            {
                "trace_id": str(
                    payload.get("trace_id")
                    or payload.get("id")
                    or f"turn-{int(time.time()*1000)}"
                ),
                "source": str(payload.get("source") or "webhook"),
                "prompt": prompt,
                "response": response,
                "human_score": (
                    float(payload["human_score"])
                    if payload.get("human_score") is not None
                    else None
                ),
            }
        )
    return extracted


class LiveTraceCollector:
    """Durable production trace collector (SQLite + JSONL) and reward-calibration auditor."""

    def __init__(
        self, workspace_root: Path, *, store: StudioSQLiteStore | None = None
    ) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.store = store or StudioSQLiteStore(self.workspace_root)

    @property
    def trace_file(self) -> Path:
        return (
            self.workspace_root
            / "outputs"
            / "weight_studio_traces"
            / "live_traces.jsonl"
        )

    def ingest_payload(
        self,
        payload: dict[str, Any],
        *,
        reward_fn: str = "customer_support",
        fail_threshold: float = 0.72,
        judge: PluggableRewardJudge | None = None,
    ) -> dict[str, Any]:
        """Ingest one webhook/OTLP payload, score every turn, and persist to SQLite + JSONL."""
        rows = extract_traces_from_payload(payload)
        if not rows:
            raise ValueError(
                "No valid traces found. Supply {prompt, response}, {messages: [...]}, "
                "{traces: [...]}, or OpenTelemetry {resourceSpans: [...]}."
            )

        path = self.trace_file
        path.parent.mkdir(parents=True, exist_ok=True)
        now = datetime.now(timezone.utc).isoformat()
        ingested: list[dict[str, Any]] = []
        failing_count = 0

        with path.open("a", encoding="utf-8") as fh:
            for row in rows:
                judge_score: float | None = None
                judge_rationale = ""
                if judge is not None and judge.mode != "heuristic":
                    graded = judge.score_turn(
                        row["prompt"], row["response"], reward_fn=reward_fn
                    )
                    proxy_score = float(graded["proxy_score"])
                    judge_score = (
                        float(graded["judge_score"])
                        if graded["judge_score"] is not None
                        else None
                    )
                    modes = list(graded["failure_modes"])
                    effective_score = float(graded["score"])
                    judge_rationale = str(graded.get("rationale") or "")
                else:
                    proxy_score, modes = _score_transcript_turn(
                        row["prompt"], row["response"], reward_fn=reward_fn
                    )
                    effective_score = proxy_score
                failed = effective_score < fail_threshold or bool(modes)
                if failed:
                    failing_count += 1
                record = {
                    "timestamp": now,
                    "trace_id": row["trace_id"],
                    "source": row["source"],
                    "reward_fn": reward_fn,
                    "prompt": row["prompt"],
                    "response": row["response"],
                    "proxy_score": proxy_score,
                    "human_score": row.get("human_score"),
                    "judge_score": judge_score,
                    "judge_rationale": judge_rationale,
                    "failure_modes": modes,
                    "failed": failed,
                }
                fh.write(json.dumps(record) + "\n")
                ingested.append(record)

        self.store.record_traces(ingested)

        return {
            "ingested_count": len(ingested),
            "failing_count": failing_count,
            "trace_file": str(path),
            "sqlite_db": str(self.store.db_path),
            "records": ingested,
        }

    def query_traces(
        self,
        *,
        failed_only: bool = False,
        source: str | None = None,
        search: str | None = None,
        labeled_only: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Run an indexed SQLite query over stored production traces."""
        return self.store.query_traces(
            failed_only=failed_only,
            source=source,
            search=search,
            labeled_only=labeled_only,
            limit=limit,
        )

    def prune_traces(self, *, keep_latest: int = 1000) -> int:
        """Prune older traces from SQLite storage, retaining ``keep_latest`` rows."""
        return self.store.prune_traces(keep_latest=keep_latest)

    def list_traces(
        self, limit: int = 50, *, failing_only: bool = False
    ) -> list[dict[str, Any]]:
        path = self.trace_file
        if not path.is_file():
            return []
        rows: list[dict[str, Any]] = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if not isinstance(obj, dict):
                    continue
                if failing_only and not obj.get("failed"):
                    continue
                rows.append(obj)
            except json.JSONDecodeError:
                continue
        return list(reversed(rows[-max(1, limit) :]))

    def summary(self) -> dict[str, Any]:
        rows = self.list_traces(limit=500)
        total = len(rows)
        failing = sum(1 for r in rows if r.get("failed"))
        labeled = sum(1 for r in rows if r.get("human_score") is not None)
        mean_proxy = (
            round(sum(float(r.get("proxy_score", 0.0)) for r in rows) / total, 4)
            if total
            else 0.0
        )
        mode_counts: dict[str, int] = {}
        for r in rows:
            for m in r.get("failure_modes") or []:
                mode_counts[str(m)] = mode_counts.get(str(m), 0) + 1
        return {
            "total_traces": total,
            "failing_traces": failing,
            "human_labeled_traces": labeled,
            "mean_proxy_score": mean_proxy,
            "failure_modes": mode_counts,
            "trace_file": str(self.trace_file),
            "recent": rows[:10],
        }

    def audit_reward_calibration(
        self,
        *,
        samples: list[dict[str, Any]] | None = None,
        reward_fn: str = "customer_support",
    ) -> dict[str, Any]:
        """Audit heuristic proxy reward alignment against human/judge ground truth.

        Detects reward hacking (high proxy score >= 0.75 while human score <= 0.45),
        computes Pearson ``r``, Spearman ``ρ``, and MAE, and issues
        ``CALIBRATED`` / ``DRIFT_WARN`` / ``HACKING_RISK``.
        """
        eval_rows: list[tuple[float, float, str]] = []

        if samples:
            for idx, item in enumerate(samples):
                if not isinstance(item, dict) or item.get("human_score") is None:
                    continue
                prompt = str(item.get("prompt") or "")
                response = str(item.get("response") or "")
                if item.get("proxy_score") is not None:
                    proxy = float(item["proxy_score"])
                else:
                    proxy, _ = _score_transcript_turn(
                        prompt, response, reward_fn=reward_fn
                    )
                human = float(item["human_score"])
                eval_rows.append(
                    (proxy, human, str(item.get("id") or f"sample-{idx + 1}"))
                )
        else:
            for r in self.list_traces(limit=500):
                gt = (
                    r.get("human_score")
                    if r.get("human_score") is not None
                    else r.get("judge_score")
                )
                if gt is not None:
                    eval_rows.append(
                        (
                            float(r.get("proxy_score", 0.0)),
                            float(gt),
                            str(r.get("trace_id") or "trace"),
                        )
                    )

        if len(eval_rows) < 2:
            return {
                "schema_version": 1,
                "verdict": "INSUFFICIENT_LABELS",
                "sample_count": len(eval_rows),
                "message": "Provide at least 2 traces with 'human_score' labels to audit reward calibration.",
            }

        proxies = [r[0] for r in eval_rows]
        humans = [r[1] for r in eval_rows]
        mae = round(
            sum(abs(p - h) for p, h in zip(proxies, humans, strict=True))
            / len(eval_rows),
            4,
        )
        r_pearson = pearson_corr(proxies, humans)
        rho_spearman = spearman_corr(proxies, humans)
        hacking_cases = [
            {"id": tid, "proxy_score": round(p, 4), "human_score": round(h, 4)}
            for p, h, tid in eval_rows
            if p >= 0.75 and h <= 0.45
        ]
        hacking_rate = round(len(hacking_cases) / len(eval_rows), 4)

        if hacking_rate >= 0.20 or rho_spearman < 0.35 or mae > 0.30:
            verdict = "HACKING_RISK"
        elif rho_spearman < 0.65 or mae > 0.18:
            verdict = "DRIFT_WARN"
        else:
            verdict = "CALIBRATED"

        return {
            "schema_version": 1,
            "reward_fn": reward_fn,
            "verdict": verdict,
            "sample_count": len(eval_rows),
            "pearson_r": r_pearson,
            "spearman_rho": rho_spearman,
            "mae": mae,
            "hacking_case_count": len(hacking_cases),
            "hacking_rate": hacking_rate,
            "hacking_cases": hacking_cases[:10],
        }
