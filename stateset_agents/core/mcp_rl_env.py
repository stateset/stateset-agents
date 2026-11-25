"""MCP-Native Verifiable Multi-Turn RL Environment & Deterministic Commerce Sandbox.

Turns **any** MCP server (in-process JSON-RPC callable, local tool registry, or
live Streamable HTTP MCP endpoint) into a resettable multi-turn Reinforcement
Learning environment with:

1. **Structured Tool-Call Execution**: parses ``<tool_call>{...}</tool_call>``
   tags or OpenAI-style ``action.tool_calls`` metadata, invokes the MCP tool,
   and feeds structured tool results back as the next observation.
2. **Step-Level Process Rewards (PRM)** + **Terminal State Verification**:
   assigns per-turn shaping rewards for valid schema & non-error tool execution
   plus an exact terminal state verifier checking final database/sandbox state.
3. **Built-in :class:`CommerceLedgerSandbox`**: zero-dependency in-memory
   SQLite commerce ledger (orders, refunds, inventory, policy windows, double-
   refund idempotency) supporting thousands of concurrent deterministic RL
   episodes per node.

Zero eager ``torch`` imports.
"""

from __future__ import annotations

import json
import re
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import urllib.parse
import urllib.request
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .environment_base import Environment, EnvironmentState, EpisodeStatus
from .trajectory import ConversationTurn

_TOOL_CALL_TAG_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL | re.IGNORECASE
)

ToolExecutor = Callable[[str, dict[str, Any]], dict[str, Any]]
StateVerifier = Callable[
    [EnvironmentState, list[dict[str, Any]]], tuple[float, dict[str, Any]]
]


def parse_tool_calls(action: ConversationTurn) -> list[dict[str, Any]]:
    """Extract normalized ``[{"name": str, "arguments": dict}]`` from an agent turn."""
    calls: list[dict[str, Any]] = []

    meta_calls = (action.metadata or {}).get("tool_calls")
    if isinstance(meta_calls, list):
        for item in meta_calls:
            if not isinstance(item, dict):
                continue
            raw_fn = item.get("function")
            fn: dict[str, Any] = raw_fn if isinstance(raw_fn, dict) else item
            name = str(fn.get("name") or "").strip()
            raw_args = fn.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            elif isinstance(raw_args, dict):
                args = dict(raw_args)
            else:
                args = {}
            if name:
                calls.append({"name": name, "arguments": args})

    if calls:
        return calls

    for match in _TOOL_CALL_TAG_RE.finditer(action.content or ""):
        try:
            obj = json.loads(match.group(1))
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and obj.get("name"):
            raw_args = obj.get("arguments") or {}
            calls.append(
                {
                    "name": str(obj["name"]).strip(),
                    "arguments": dict(raw_args) if isinstance(raw_args, dict) else {},
                }
            )
    return calls


class CommerceLedgerSandbox:
    """Deterministic, resettable in-memory SQLite commerce sandbox for verifiable RL."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self.reset()

    def reset(self, scenario: dict[str, Any] | None = None) -> None:
        """Reset the in-memory SQLite commerce ledger to a deterministic initial state."""
        self._conn.executescript("""
            DROP TABLE IF EXISTS orders;
            DROP TABLE IF EXISTS refunds;
            CREATE TABLE orders (
                order_id TEXT PRIMARY KEY,
                customer_email TEXT NOT NULL,
                amount_cents INTEGER NOT NULL,
                days_since_delivery INTEGER NOT NULL,
                status TEXT NOT NULL
            );
            CREATE TABLE refunds (
                refund_id TEXT PRIMARY KEY,
                order_id TEXT UNIQUE NOT NULL,
                amount_cents INTEGER NOT NULL,
                reason TEXT NOT NULL
            );
            """)
        orders = (scenario or {}).get("orders") or [
            ("ORD-77701", "alex@example.com", 8900, 9, "delivered_damaged"),
            ("ORD-99102", "sam@example.com", 14500, 45, "delivered"),
        ]
        self._conn.executemany(
            "INSERT INTO orders (order_id, customer_email, amount_cents, days_since_delivery, status) VALUES (?, ?, ?, ?, ?)",
            orders,
        )
        self._conn.commit()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Execute an MCP-style commerce tool against the live SQLite ledger."""
        if name == "get_order":
            oid = str(arguments.get("order_id") or "").strip()
            row = self._conn.execute(
                "SELECT * FROM orders WHERE order_id = ?", (oid,)
            ).fetchone()
            if row is None:
                return {"isError": True, "error": f"Order {oid!r} not found."}
            return {"isError": False, "order": dict(row)}

        if name == "issue_refund":
            oid = str(arguments.get("order_id") or "").strip()
            reason = str(arguments.get("reason") or "damaged").strip()
            row = self._conn.execute(
                "SELECT * FROM orders WHERE order_id = ?", (oid,)
            ).fetchone()
            if row is None:
                return {"isError": True, "error": f"Order {oid!r} not found."}
            if int(row["days_since_delivery"]) > 30:
                return {
                    "isError": True,
                    "error": f"Order {oid} is outside the 30-day return window ({row['days_since_delivery']} days).",
                }
            existing = self._conn.execute(
                "SELECT refund_id FROM refunds WHERE order_id = ?", (oid,)
            ).fetchone()
            if existing is not None:
                return {
                    "isError": True,
                    "error": f"Refund already issued for {oid} ({existing['refund_id']}).",
                }
            rid = f"REF-{oid}"
            self._conn.execute(
                "INSERT INTO refunds (refund_id, order_id, amount_cents, reason) VALUES (?, ?, ?, ?)",
                (rid, oid, int(row["amount_cents"]), reason),
            )
            self._conn.execute(
                "UPDATE orders SET status = 'refunded' WHERE order_id = ?", (oid,)
            )
            self._conn.commit()
            return {
                "isError": False,
                "refund_id": rid,
                "order_id": oid,
                "amount_cents": int(row["amount_cents"]),
            }

        return {"isError": True, "error": f"Unknown tool: {name!r}"}

    def verify_order_refunded(
        self, order_id: str, *, should_be_refunded: bool = True
    ) -> tuple[float, dict[str, Any]]:
        """Verifiable ground-truth reward computed strictly from SQLite ledger state."""
        row = self._conn.execute(
            "SELECT * FROM refunds WHERE order_id = ?", (order_id,)
        ).fetchone()
        refunded = row is not None
        ok = refunded == should_be_refunded
        return (
            1.0 if ok else 0.0,
            {
                "order_id": order_id,
                "expected_refunded": should_be_refunded,
                "actual_refunded": refunded,
                "verified": ok,
            },
        )


class CodeVerificationSandbox:
    """Timeout-guarded Python coding & unit-test verification sandbox for verifiable SWE RL."""

    def __init__(self, *, timeout_seconds: float = 5.0) -> None:
        self.timeout_seconds = float(timeout_seconds)
        self._solution_code: str = ""
        self._test_suite: str = ""
        self._last_passed: bool = False
        self._last_output: str = ""
        self.reset()

    def reset(self, scenario: dict[str, Any] | None = None) -> None:
        self._solution_code = ""
        self._test_suite = str(
            (scenario or {}).get("test_code")
            or "assert compute_net_revenue(18, 65, 3) == 975\nassert compute_net_revenue(10, 50, 0) == 500\n"
        )
        self._last_passed = False
        self._last_output = ""

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "write_solution":
            self._solution_code = str(arguments.get("code") or "")
            return {
                "isError": False,
                "bytes_written": len(self._solution_code.encode("utf-8")),
            }

        if name == "run_tests":
            if not self._solution_code.strip():
                return {"isError": True, "error": "No solution written yet."}
            with tempfile.TemporaryDirectory() as tmpdir:
                script = Path(tmpdir) / "verify.py"
                script.write_text(
                    f"{self._solution_code}\n\n{self._test_suite}\n",
                    encoding="utf-8",
                )
                try:
                    proc = subprocess.run(
                        [sys.executable, "-I", str(script)],
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                        check=False,
                    )
                except subprocess.TimeoutExpired:
                    self._last_passed = False
                    self._last_output = f"Timed out after {self.timeout_seconds}s"
                    return {
                        "isError": True,
                        "passed": False,
                        "error": self._last_output,
                    }
            self._last_passed = proc.returncode == 0
            self._last_output = (proc.stdout + proc.stderr).strip()[:400]
            return {
                "isError": not self._last_passed,
                "passed": self._last_passed,
                "returncode": proc.returncode,
                "output": self._last_output,
            }

        return {"isError": True, "error": f"Unknown coding sandbox tool: {name!r}"}

    def verify(
        self, _state: EnvironmentState, _history: list[dict[str, Any]]
    ) -> tuple[float, dict[str, Any]]:
        if self._solution_code.strip() and not self._last_passed:
            self.call_tool("run_tests", {})
        return (
            1.0 if self._last_passed else 0.0,
            {"verified": self._last_passed, "output": self._last_output},
        )


class SQLAnalyticsSandbox:
    """Deterministic read-only SQLite BI/Text-to-SQL verification sandbox for agentic RL."""

    def __init__(self) -> None:
        self._conn = sqlite3.connect(":memory:")
        self._conn.row_factory = sqlite3.Row
        self._last_rows: list[dict[str, Any]] = []
        self._expected_value: Any = None
        self.reset()

    def reset(self, scenario: dict[str, Any] | None = None) -> None:
        self._last_rows = []
        self._expected_value = (scenario or {}).get("expected_value", 25000)
        self._conn.executescript("""
            DROP TABLE IF EXISTS sales;
            CREATE TABLE sales (
                id INTEGER PRIMARY KEY,
                region TEXT NOT NULL,
                product TEXT NOT NULL,
                revenue_cents INTEGER NOT NULL
            );
            INSERT INTO sales (region, product, revenue_cents) VALUES
                ('west', 'widget', 12000),
                ('west', 'gadget', 13000),
                ('east', 'widget', 9000);
            """)
        self._conn.commit()

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "list_tables":
            rows = self._conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
            ).fetchall()
            return {"isError": False, "tables": [r["name"] for r in rows]}

        if name == "describe_table":
            tbl = str(arguments.get("table") or "sales").strip()
            if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", tbl):
                return {"isError": True, "error": f"Invalid table name: {tbl!r}"}
            rows = self._conn.execute(
                "SELECT name, type FROM pragma_table_info(?)", (tbl,)
            ).fetchall()
            return {
                "isError": False,
                "columns": [{"name": r["name"], "type": r["type"]} for r in rows],
            }

        if name == "run_sql":
            sql = str(arguments.get("sql") or "").strip()
            if not sql.lower().startswith("select"):
                return {
                    "isError": True,
                    "error": "Only read-only SELECT queries are permitted in SQLAnalyticsSandbox.",
                }
            try:
                cur = self._conn.execute(sql)
                rows = [dict(r) for r in cur.fetchall()[:50]]
                self._last_rows = rows
                return {"isError": False, "row_count": len(rows), "rows": rows}
            except sqlite3.Error as exc:
                return {"isError": True, "error": str(exc)}

        return {"isError": True, "error": f"Unknown SQL sandbox tool: {name!r}"}

    def verify(
        self, _state: EnvironmentState, _history: list[dict[str, Any]]
    ) -> tuple[float, dict[str, Any]]:
        matched = any(self._expected_value in row.values() for row in self._last_rows)
        return (
            1.0 if matched else 0.0,
            {
                "verified": matched,
                "expected_value": self._expected_value,
                "last_rows": self._last_rows[:5],
            },
        )


class RESTAPISandbox:
    """Stateful OpenAPI / SaaS REST backend verifier with schema & idempotency checks.

    Tools:
    - ``openapi_spec()`` -> returns endpoint paths, required fields, and status codes
    - ``http_request(method, path, body=None, idempotency_key=None)`` -> mutates or reads
      in-memory resources (``/v1/tickets``, ``/v1/subscriptions``) with strict
      idempotency deduplication and schema validation.
    """

    def __init__(
        self,
        *,
        initial_subscriptions: dict[str, dict[str, Any]] | None = None,
    ) -> None:
        self._subscriptions: dict[str, dict[str, Any]] = dict(
            initial_subscriptions
            or {
                "sub_101": {
                    "id": "sub_101",
                    "customer_id": "cust_42",
                    "plan": "pro",
                    "status": "active",
                }
            }
        )
        self._tickets: dict[str, dict[str, Any]] = {}
        self._idempotency_cache: dict[str, dict[str, Any]] = {}
        self._mutation_count = 0

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "openapi_spec":
            return {
                "isError": False,
                "endpoints": [
                    "GET /v1/subscriptions/{id}",
                    "POST /v1/subscriptions/{id}/cancel (requires idempotency_key, body.reason)",
                    "POST /v1/tickets (requires body.subject, body.priority)",
                ],
            }

        if name != "http_request":
            return {
                "isError": True,
                "error": f"Unknown REST API sandbox tool: {name!r}",
            }

        method = str(arguments.get("method") or "GET").strip().upper()
        path = str(arguments.get("path") or "").strip()
        body = arguments.get("body") or {}
        idem_key = str(arguments.get("idempotency_key") or "").strip()

        if method == "GET" and path.startswith("/v1/subscriptions/"):
            sub_id = path.rsplit("/", 1)[-1]
            sub = self._subscriptions.get(sub_id)
            if sub is None:
                return {
                    "isError": True,
                    "status": 404,
                    "error": f"Subscription {sub_id} not found",
                }
            return {"isError": False, "status": 200, "data": dict(sub)}

        if (
            method == "POST"
            and path.startswith("/v1/subscriptions/")
            and path.endswith("/cancel")
        ):
            parts = path.strip("/").split("/")
            sub_id = parts[2] if len(parts) >= 4 else ""
            if not idem_key:
                return {
                    "isError": True,
                    "status": 400,
                    "error": "Missing required idempotency_key header for mutation.",
                }
            if idem_key in self._idempotency_cache:
                cached = dict(self._idempotency_cache[idem_key])
                cached["idempotent_replay"] = True
                return cached
            sub = self._subscriptions.get(sub_id)
            if sub is None:
                return {
                    "isError": True,
                    "status": 404,
                    "error": f"Subscription {sub_id} not found",
                }
            reason = str(body.get("reason") or "").strip()
            if not reason:
                return {
                    "isError": True,
                    "status": 400,
                    "error": "body.reason is required.",
                }
            sub["status"] = "canceled"
            sub["cancel_reason"] = reason
            self._mutation_count += 1
            resp = {"isError": False, "status": 200, "data": dict(sub)}
            self._idempotency_cache[idem_key] = resp
            return resp

        if method == "POST" and path == "/v1/tickets":
            subject = str(body.get("subject") or "").strip()
            priority = str(body.get("priority") or "").strip().lower()
            if not subject or priority not in ("low", "medium", "high"):
                return {
                    "isError": True,
                    "status": 400,
                    "error": "body.subject and body.priority (low|medium|high) required.",
                }
            tid = f"tkt_{len(self._tickets) + 1}"
            rec = {
                "id": tid,
                "subject": subject,
                "priority": priority,
                "status": "open",
            }
            self._tickets[tid] = rec
            self._mutation_count += 1
            return {"isError": False, "status": 201, "data": dict(rec)}

        return {
            "isError": True,
            "status": 404,
            "error": f"Unhandled route: {method} {path}",
        }

    def verify(
        self, state: EnvironmentState, _history: list[dict[str, Any]]
    ) -> tuple[float, dict[str, Any]]:
        scenario = state.context.get("scenario") or {}
        sub_id = str(scenario.get("target_subscription_id") or "sub_101")
        expected_status = str(scenario.get("expected_status") or "canceled")
        sub = self._subscriptions.get(sub_id) or {}
        ok = sub.get("status") == expected_status
        return (
            1.0 if ok else 0.0,
            {
                "verified": ok,
                "subscription_id": sub_id,
                "actual_status": sub.get("status"),
                "expected_status": expected_status,
                "mutations": self._mutation_count,
            },
        )


class DockerCodeSandbox:
    """Container-isolated SWE-agent patch & pytest verifier with clean subprocess fallback.

    Runs ``docker run --rm --network=none --cpus=1 --memory=256m`` when
    ``use_docker=True`` and Docker daemon is reachable; otherwise executes in an
    isolated temporary directory subprocess with strict timeout and reports
    ``execution_backend`` (``"docker"`` vs ``"subprocess_fallback"``).
    """

    DEFAULT_FILES: dict[str, str] = {
        "billing.py": (
            "def apply_discount(subtotal_cents: int, pct: int) -> int:\n"
            "    # BUG: subtracts pct instead of percentage\n"
            "    return subtotal_cents - pct\n"
        ),
        "test_billing.py": (
            "from billing import apply_discount\n"
            "assert apply_discount(10000, 15) == 8500\n"
            "assert apply_discount(2000, 0) == 2000\n"
            "assert apply_discount(5000, 100) == 0\n"
        ),
    }

    def __init__(
        self,
        *,
        files: dict[str, str] | None = None,
        image: str = "python:3.12-slim",
        use_docker: bool = False,
        timeout_seconds: float = 5.0,
    ) -> None:
        self._files: dict[str, str] = dict(files or self.DEFAULT_FILES)
        self.image = image
        self.use_docker = bool(use_docker)
        self.timeout_seconds = float(timeout_seconds)
        self._last_passed = False
        self._last_output = ""
        self._execution_backend = "subprocess_fallback"

    def _safe_rel_path(self, raw_path: str) -> str:
        clean = Path(raw_path).name
        if (
            not clean
            or clean in (".", "..")
            or not re.fullmatch(r"[A-Za-z0-9._-]+", clean)
        ):
            raise ValueError(f"Unsafe file path: {raw_path!r}")
        return clean

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "read_file":
            try:
                fname = self._safe_rel_path(str(arguments.get("path") or "billing.py"))
            except ValueError as exc:
                return {"isError": True, "error": str(exc)}
            if fname not in self._files:
                return {"isError": True, "error": f"File not found: {fname}"}
            return {"isError": False, "path": fname, "content": self._files[fname]}

        if name == "apply_patch":
            try:
                fname = self._safe_rel_path(str(arguments.get("path") or "billing.py"))
            except ValueError as exc:
                return {"isError": True, "error": str(exc)}
            content = str(arguments.get("content") or "")
            if not content.strip():
                return {"isError": True, "error": "Patch content must be non-empty."}
            self._files[fname] = content
            self._last_passed = False
            return {"isError": False, "path": fname, "bytes_written": len(content)}

        if name == "run_pytest":
            with tempfile.TemporaryDirectory(prefix="stateset_docker_code_") as td:
                root = Path(td)
                for fname, body in self._files.items():
                    (root / fname).write_text(body, encoding="utf-8")

                docker_bin = shutil.which("docker") if self.use_docker else None
                cmd: list[str]
                if docker_bin:
                    self._execution_backend = "docker"
                    cmd = [
                        docker_bin,
                        "run",
                        "--rm",
                        "--network=none",
                        "--cpus=1",
                        "--memory=256m",
                        "-v",
                        f"{root}:/work:ro",
                        "-w",
                        "/work",
                        self.image,
                        "python",
                        "test_billing.py",
                    ]
                else:
                    self._execution_backend = "subprocess_fallback"
                    cmd = [sys.executable, "-s", str(root / "test_billing.py")]

                try:
                    proc = subprocess.run(
                        cmd,
                        cwd=str(root),
                        capture_output=True,
                        text=True,
                        timeout=self.timeout_seconds,
                        check=False,
                    )
                    # If docker daemon is unavailable, fall back cleanly to isolated subprocess
                    if docker_bin and proc.returncode == 125:
                        self._execution_backend = "subprocess_fallback"
                        proc = subprocess.run(
                            [sys.executable, "-s", str(root / "test_billing.py")],
                            cwd=str(root),
                            capture_output=True,
                            text=True,
                            timeout=self.timeout_seconds,
                            check=False,
                        )
                    self._last_passed = proc.returncode == 0
                    self._last_output = (proc.stdout + proc.stderr).strip()[:500]
                    return {
                        "isError": not self._last_passed,
                        "passed": self._last_passed,
                        "execution_backend": self._execution_backend,
                        "output": self._last_output
                        or ("OK" if self._last_passed else "FAILED"),
                    }
                except subprocess.TimeoutExpired:
                    self._last_passed = False
                    self._last_output = "Execution timed out."
                    return {
                        "isError": True,
                        "passed": False,
                        "execution_backend": self._execution_backend,
                        "error": self._last_output,
                    }

        return {"isError": True, "error": f"Unknown docker code tool: {name!r}"}

    def verify(
        self, _state: EnvironmentState, _history: list[dict[str, Any]]
    ) -> tuple[float, dict[str, Any]]:
        return (
            1.0 if self._last_passed else 0.0,
            {
                "verified": self._last_passed,
                "execution_backend": self._execution_backend,
                "output": self._last_output,
            },
        )


class BrowserDOMSandbox:
    """Deterministic stateful Browser / Web-Agent DOM & Form Verifier.

    Simulates an interactive multi-page SaaS / Commerce support portal with
    structured DOM snapshots, form field inputs, button clicks, and terminal
    DOM/URL receipt verification (zero extra runtime dependencies; mirrors
    Playwright action semantics cleanly).

    Tools:
    - ``get_dom()`` -> returns current URL, visible elements, and form values
    - ``navigate(url)`` -> navigates to ``/portal/orders`` or ``/portal/returns``
    - ``fill(selector, value)`` -> populates a form input (e.g. ``#order-id``, ``#reason``)
    - ``click(selector)`` -> clicks an interactive button (e.g. ``#btn-submit-return``)
    """

    def __init__(self) -> None:
        self._url = "/portal/home"
        self._inputs: dict[str, str] = {}
        self._submitted_returns: list[dict[str, Any]] = []
        self._confirmation_id = ""

    def call_tool(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "get_dom":
            if self._url == "/portal/returns":
                elements = [
                    {
                        "selector": "#order-id",
                        "tag": "input",
                        "value": self._inputs.get("#order-id", ""),
                    },
                    {
                        "selector": "#reason",
                        "tag": "input",
                        "value": self._inputs.get("#reason", ""),
                    },
                    {
                        "selector": "#btn-submit-return",
                        "tag": "button",
                        "text": "Submit Return",
                    },
                ]
            elif self._url == "/portal/confirmation":
                elements = [
                    {
                        "selector": "#confirmation-id",
                        "tag": "span",
                        "text": self._confirmation_id,
                    },
                ]
            else:
                elements = [
                    {
                        "selector": "#link-returns",
                        "tag": "a",
                        "href": "/portal/returns",
                        "text": "Start Return",
                    },
                ]
            return {"isError": False, "url": self._url, "elements": elements}

        if name == "navigate":
            target = str(arguments.get("url") or "").strip()
            if target not in ("/portal/home", "/portal/returns"):
                return {
                    "isError": True,
                    "error": f"404 Unknown portal route: {target!r}",
                }
            self._url = target
            return {"isError": False, "url": self._url}

        if name == "fill":
            if self._url != "/portal/returns":
                return {
                    "isError": True,
                    "error": "Form inputs only exist on /portal/returns.",
                }
            sel = str(arguments.get("selector") or "").strip()
            val = str(arguments.get("value") or "").strip()
            if sel not in ("#order-id", "#reason"):
                return {"isError": True, "error": f"Unknown input selector: {sel!r}"}
            if not val:
                return {"isError": True, "error": f"Value for {sel} must be non-empty."}
            self._inputs[sel] = val
            return {"isError": False, "selector": sel, "value": val}

        if name == "click":
            sel = str(arguments.get("selector") or "").strip()
            if sel == "#link-returns" and self._url == "/portal/home":
                self._url = "/portal/returns"
                return {"isError": False, "url": self._url}
            if sel == "#btn-submit-return" and self._url == "/portal/returns":
                oid = self._inputs.get("#order-id", "").strip()
                reason = self._inputs.get("#reason", "").strip()
                if not oid or not reason:
                    return {
                        "isError": True,
                        "error": "Both #order-id and #reason must be filled before submitting.",
                    }
                self._confirmation_id = f"RMA-{oid}"
                rec = {
                    "confirmation_id": self._confirmation_id,
                    "order_id": oid,
                    "reason": reason,
                }
                self._submitted_returns.append(rec)
                self._url = "/portal/confirmation"
                return {"isError": False, "url": self._url, "receipt": rec}
            return {
                "isError": True,
                "error": f"Clickable selector {sel!r} not found on {self._url}.",
            }

        return {"isError": True, "error": f"Unknown browser sandbox tool: {name!r}"}

    def verify(
        self, state: EnvironmentState, _history: list[dict[str, Any]]
    ) -> tuple[float, dict[str, Any]]:
        scenario = state.context.get("scenario") or {}
        target_oid = str(scenario.get("target_order_id") or "ORD-77701")
        matched = any(r["order_id"] == target_oid for r in self._submitted_returns)
        ok = matched and self._url == "/portal/confirmation"
        return (
            1.0 if ok else 0.0,
            {
                "verified": ok,
                "url": self._url,
                "confirmation_id": self._confirmation_id,
                "expected_order_id": target_oid,
            },
        )


def make_http_mcp_executor(
    mcp_url: str, *, bearer_token: str = "", timeout: float = 15.0
) -> ToolExecutor:
    """Create a ``ToolExecutor`` calling a live Streamable HTTP MCP JSON-RPC endpoint."""
    url = mcp_url.strip()
    if urllib.parse.urlparse(url).scheme not in ("http", "https"):
        raise ValueError(f"MCP URL must use http or https scheme, got: {url!r}")
    headers = {"Content-Type": "application/json"}
    if bearer_token.strip():
        headers["Authorization"] = f"Bearer {bearer_token.strip()}"

    def _call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        body = json.dumps(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {"name": name, "arguments": arguments},
            }
        ).encode("utf-8")
        req = urllib.request.Request(url, data=body, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # nosec B310
            parsed = json.loads(resp.read().decode("utf-8", errors="replace"))
        if "error" in parsed:
            return {"isError": True, "error": parsed["error"]}
        result = parsed.get("result") or {}
        return {
            "isError": bool(result.get("isError", False)),
            "result": result.get("structuredContent", result),
        }

    return _call


class MCPRLEnvironment(Environment):
    """Multi-turn RL Environment wrapping any MCP server or deterministic sandbox."""

    def __init__(
        self,
        *,
        tool_executor: ToolExecutor | None = None,
        state_verifier: StateVerifier | None = None,
        mcp_http_url: str = "",
        mcp_bearer_token: str = "",
        max_turns: int = 6,
        process_reward_weight: float = 0.25,
    ) -> None:
        super().__init__(max_turns=max_turns)
        self._sandbox: CommerceLedgerSandbox | None = None
        if tool_executor is not None:
            self._executor = tool_executor
        elif mcp_http_url.strip():
            self._executor = make_http_mcp_executor(
                mcp_http_url, bearer_token=mcp_bearer_token
            )
        else:
            self._sandbox = CommerceLedgerSandbox()
            self._executor = self._sandbox.call_tool
        self._verifier = state_verifier
        self.process_reward_weight = max(0.0, min(1.0, float(process_reward_weight)))

    def clone(self) -> MCPRLEnvironment:
        return MCPRLEnvironment(
            tool_executor=None if self._sandbox is not None else self._executor,
            state_verifier=self._verifier,
            max_turns=self.max_turns,
            process_reward_weight=self.process_reward_weight,
        )

    async def get_initial_prompt(self, scenario: dict[str, Any] | None = None) -> str:
        if scenario and scenario.get("prompt"):
            return str(scenario["prompt"])
        return (
            "Customer: My order ORD-77701 arrived damaged 9 days ago. "
            "Please verify eligibility using get_order and issue a refund if eligible."
        )

    async def reset(self, scenario: dict[str, Any] | None = None) -> EnvironmentState:
        if self._sandbox is not None:
            self._sandbox.reset(scenario)
        ep_id = f"mcp-ep-{uuid.uuid4().hex[:10]}"
        prompt = await self.get_initial_prompt(scenario)
        state = EnvironmentState(
            episode_id=ep_id,
            turn_count=0,
            status=EpisodeStatus.ONGOING,
            context={
                "scenario": dict(scenario or {}),
                "prompt": prompt,
                "tool_history": [],
            },
            metadata={"process_rewards": []},
        )
        self.active_episodes[ep_id] = state
        return state

    async def step(
        self, state: EnvironmentState, action: ConversationTurn
    ) -> tuple[EnvironmentState, float, bool, dict[str, Any]]:
        """Execute agent tool calls (if any), compute step PRM reward & terminal state reward."""
        next_state = state.copy()
        next_state.turn_count += 1
        tool_history = list(next_state.context.get("tool_history") or [])
        calls = parse_tool_calls(action)

        turn_tool_results: list[dict[str, Any]] = []
        step_prm = 0.0
        if calls:
            ok_calls = 0
            for call in calls:
                res = self._executor(call["name"], call["arguments"])
                is_err = bool(res.get("isError", False))
                if not is_err:
                    ok_calls += 1
                entry = {
                    "turn": next_state.turn_count,
                    "name": call["name"],
                    "arguments": call["arguments"],
                    "result": res,
                    "is_error": is_err,
                }
                turn_tool_results.append(entry)
                tool_history.append(entry)
            step_prm = round(ok_calls / len(calls), 4)
        else:
            # Final natural-language response terminates the episode
            step_prm = 0.5 if len((action.content or "").strip()) >= 15 else 0.0

        next_state.context["tool_history"] = tool_history
        proc_rewards = list(next_state.metadata.get("process_rewards") or [])
        proc_rewards.append(step_prm)
        next_state.metadata["process_rewards"] = proc_rewards

        tool_signaled_done = any(
            isinstance(e.get("result"), dict) and bool(e["result"].get("done"))
            for e in turn_tool_results
        )
        done = (
            (not calls)
            or tool_signaled_done
            or (next_state.turn_count >= self.max_turns)
        )
        verification: dict[str, Any] = {}
        reward = round(step_prm * self.process_reward_weight, 4)

        if done:
            next_state.status = EpisodeStatus.COMPLETED
            if self._verifier is not None:
                term_score, verification = self._verifier(next_state, tool_history)
            elif self._sandbox is not None:
                target_oid = str(
                    (next_state.context.get("scenario") or {}).get(
                        "target_order_id", "ORD-77701"
                    )
                )
                expect_ref = bool(
                    (next_state.context.get("scenario") or {}).get(
                        "expect_refunded", True
                    )
                )
                term_score, verification = self._sandbox.verify_order_refunded(
                    target_oid, should_be_refunded=expect_ref
                )
            else:
                any_ok = any(not e["is_error"] for e in tool_history)
                term_score = 1.0 if any_ok else 0.0
                verification = {"tool_calls_succeeded": any_ok}

            mean_prm = sum(proc_rewards) / max(1, len(proc_rewards))
            reward = round(
                (1.0 - self.process_reward_weight) * float(term_score)
                + self.process_reward_weight * mean_prm,
                4,
            )

        info = {
            "tool_results": turn_tool_results,
            "step_process_reward": step_prm,
            "verification": verification,
            "observation": (
                json.dumps(turn_tool_results)
                if turn_tool_results
                else "Episode complete."
            ),
        }
        self.active_episodes[next_state.episode_id] = next_state
        return next_state, reward, done, info


__all__ = [
    "BrowserDOMSandbox",
    "CodeVerificationSandbox",
    "CommerceLedgerSandbox",
    "DockerCodeSandbox",
    "MCPRLEnvironment",
    "RESTAPISandbox",
    "SQLAnalyticsSandbox",
    "make_http_mcp_executor",
    "parse_tool_calls",
]
