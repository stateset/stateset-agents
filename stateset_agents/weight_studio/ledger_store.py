"""Queryable SQLite WAL-mode Ledger & Trace Store for 3D Weight Studio.

Provides atomic, concurrency-safe persistence and indexed querying for:
- Live production traces (OTLP/JSON + webhooks + LLM-judge ground truth),
- Production Canary Shadow-Gate rollout decisions and Autopilot releases,
- Retention / compaction policies (`prune_traces`).

Uses only Python's standard library ``sqlite3`` module (zero extra deps,
zero eager ``torch`` imports) and coexists transparently with the append-only
``live_traces.jsonl`` and ``rollout_ledger.jsonl`` logs.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class StudioSQLiteStore:
    """Thread-safe WAL-mode SQLite ledger for traces and canary rollouts."""

    def __init__(self, workspace_root: Path, *, db_path: Path | None = None) -> None:
        self.workspace_root = workspace_root.expanduser().resolve()
        self.db_path = (
            db_path.expanduser().resolve()
            if db_path is not None
            else self.workspace_root / "outputs" / "weight_studio_store.sqlite3"
        )
        self._lock = threading.RLock()
        self._initialized = False

    def _connect(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA synchronous=NORMAL;")
        return conn

    def _ensure_schema(self) -> None:
        if self._initialized and self.db_path.is_file():
            return
        with self._lock:
            if self._initialized and self.db_path.is_file():
                return
            with self._connect() as conn:
                conn.executescript("""
                    CREATE TABLE IF NOT EXISTS traces (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        trace_id TEXT NOT NULL,
                        source TEXT NOT NULL,
                        reward_fn TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        response TEXT NOT NULL,
                        proxy_score REAL NOT NULL,
                        human_score REAL,
                        judge_score REAL,
                        failed INTEGER NOT NULL,
                        failure_modes TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_traces_failed_ts
                        ON traces(failed, id DESC);
                    CREATE INDEX IF NOT EXISTS idx_traces_source_ts
                        ON traces(source, id DESC);

                    CREATE TABLE IF NOT EXISTS rollouts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        rollout_id TEXT NOT NULL,
                        model_id TEXT NOT NULL,
                        stage TEXT NOT NULL,
                        decision TEXT NOT NULL,
                        recommended_traffic_pct INTEGER NOT NULL,
                        mean_delta REAL NOT NULL,
                        safety_verdict TEXT NOT NULL,
                        mode TEXT NOT NULL,
                        payload_json TEXT NOT NULL
                    );
                    CREATE INDEX IF NOT EXISTS idx_rollouts_decision_ts
                        ON rollouts(decision, id DESC);
                    CREATE INDEX IF NOT EXISTS idx_rollouts_model_ts
                        ON rollouts(model_id, id DESC);
                    """)
            self._initialized = True

    def record_traces(self, records: list[dict[str, Any]]) -> int:
        """Atomically insert one or more trace records into SQLite."""
        if not records:
            return 0
        self._ensure_schema()
        now = datetime.now(timezone.utc).isoformat()
        rows = [
            (
                str(r.get("timestamp") or now),
                str(r.get("trace_id") or ""),
                str(r.get("source") or "webhook"),
                str(r.get("reward_fn") or "customer_support"),
                str(r.get("prompt") or ""),
                str(r.get("response") or ""),
                float(r.get("proxy_score", 0.0)),
                float(r["human_score"]) if r.get("human_score") is not None else None,
                float(r["judge_score"]) if r.get("judge_score") is not None else None,
                1 if r.get("failed") else 0,
                json.dumps(r.get("failure_modes") or []),
                json.dumps(r, default=str),
            )
            for r in records
        ]
        with self._lock, self._connect() as conn:
            conn.executemany(
                """
                INSERT INTO traces (
                    timestamp, trace_id, source, reward_fn, prompt, response,
                    proxy_score, human_score, judge_score, failed, failure_modes, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
        return len(rows)

    def query_traces(
        self,
        *,
        failed_only: bool = False,
        source: str | None = None,
        search: str | None = None,
        labeled_only: bool = False,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query stored traces with indexed filters, newest first."""
        if not self.db_path.is_file():
            return []
        self._ensure_schema()
        clauses: list[str] = []
        params: list[Any] = []
        if failed_only:
            clauses.append("failed = 1")
        if source:
            clauses.append("source = ?")
            params.append(source.strip())
        if labeled_only:
            clauses.append("(human_score IS NOT NULL OR judge_score IS NOT NULL)")
        if search:
            like = f"%{search.strip()}%"
            clauses.append("(prompt LIKE ? OR response LIKE ? OR trace_id LIKE ?)")
            params.extend([like, like, like])
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 1000)))
        sql = f"SELECT payload_json FROM traces {where} ORDER BY id DESC LIMIT ?"
        with self._lock, self._connect() as conn:
            cur = conn.execute(sql, params)
            out: list[dict[str, Any]] = []
            for row in cur.fetchall():
                try:
                    parsed = json.loads(row["payload_json"])
                    if isinstance(parsed, dict):
                        out.append(parsed)
                except json.JSONDecodeError:
                    continue
            return out

    def prune_traces(self, *, keep_latest: int = 1000) -> int:
        """Retain only the most recent ``keep_latest`` traces; returns deleted row count."""
        if not self.db_path.is_file():
            return 0
        self._ensure_schema()
        keep = max(0, int(keep_latest))
        with self._lock, self._connect() as conn:
            cur = conn.execute(
                """
                DELETE FROM traces
                WHERE id NOT IN (
                    SELECT id FROM traces ORDER BY id DESC LIMIT ?
                )
                """,
                (keep,),
            )
            return int(cur.rowcount or 0)

    def record_rollout(self, record: dict[str, Any]) -> None:
        """Atomically insert a Canary Shadow Gate or Autopilot rollout ledger record."""
        self._ensure_schema()
        now = datetime.now(timezone.utc).isoformat()
        pr = record.get("paired_record") or {}
        safety = record.get("safety") or {}
        with self._lock, self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rollouts (
                    timestamp, rollout_id, model_id, stage, decision,
                    recommended_traffic_pct, mean_delta, safety_verdict, mode, payload_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    str(record.get("timestamp") or now),
                    str(record.get("rollout_id") or ""),
                    str(record.get("model_id") or ""),
                    str(record.get("stage") or "canary_10"),
                    str(record.get("decision") or ""),
                    int(record.get("recommended_traffic_pct", 0)),
                    float(pr.get("mean_delta", record.get("mean_delta", 0.0))),
                    str(safety.get("verdict") or record.get("safety_verdict") or ""),
                    str(record.get("mode") or "weight_probe"),
                    json.dumps(record, default=str),
                ),
            )

    def query_rollouts(
        self,
        *,
        decision: str | None = None,
        model_id: str | None = None,
        stage: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        """Query stored rollout decisions with indexed filters, newest first."""
        if not self.db_path.is_file():
            return []
        self._ensure_schema()
        clauses: list[str] = []
        params: list[Any] = []
        if decision:
            clauses.append("decision = ?")
            params.append(decision.strip().upper())
        if model_id:
            clauses.append("model_id = ?")
            params.append(model_id.strip())
        if stage:
            clauses.append("stage = ?")
            params.append(stage.strip())
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.append(max(1, min(int(limit), 500)))
        sql = f"SELECT payload_json FROM rollouts {where} ORDER BY id DESC LIMIT ?"
        with self._lock, self._connect() as conn:
            cur = conn.execute(sql, params)
            out: list[dict[str, Any]] = []
            for row in cur.fetchall():
                try:
                    parsed = json.loads(row["payload_json"])
                    if isinstance(parsed, dict):
                        out.append(parsed)
                except json.JSONDecodeError:
                    continue
            return out

    def stats(self) -> dict[str, Any]:
        """Return aggregate row counts and store metadata."""
        if not self.db_path.is_file():
            return {
                "db_path": str(self.db_path),
                "exists": False,
                "trace_count": 0,
                "failing_trace_count": 0,
                "rollout_count": 0,
            }
        self._ensure_schema()
        with self._lock, self._connect() as conn:
            t_total = conn.execute("SELECT COUNT(*) FROM traces").fetchone()[0]
            t_fail = conn.execute(
                "SELECT COUNT(*) FROM traces WHERE failed = 1"
            ).fetchone()[0]
            r_total = conn.execute("SELECT COUNT(*) FROM rollouts").fetchone()[0]
        return {
            "db_path": str(self.db_path),
            "exists": True,
            "trace_count": int(t_total),
            "failing_trace_count": int(t_fail),
            "rollout_count": int(r_total),
        }


__all__ = ["StudioSQLiteStore"]
