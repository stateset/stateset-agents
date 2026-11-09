"""A real HTTP server speaking the sandbox controller's API, for the real SDK.

Each "sandbox" is a directory and each process is a real subprocess, so the real
``stateset_sandbox`` client, real worker and real RL code all run end to end. Only
Kubernetes is missing. Routes and response shapes follow the controller's
``openapi.yaml``; ``test_fanout_sandbox_contract.py`` checks them against it when
the sandbox repo is available.

Faults are injected the way they happen in production: a sandbox disappears, a
create is refused, a budget stops a sandbox.
"""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE = "/workspace"


@dataclass
class FakeProcess:
    process_id: str
    popen: subprocess.Popen[bytes]
    stdout: Path
    stderr: Path
    name: str | None
    command: list[str]
    killed: bool = False
    timed_out: bool = False
    started: float = field(default_factory=time.monotonic)


@dataclass
class FakeSandbox:
    sandbox_id: str
    root: Path
    body: dict[str, Any]
    status: str = "running"
    processes: dict[str, FakeProcess] = field(default_factory=dict)
    cost_cents: float = 1.25
    budget_exhausted: bool = False


class ApiFailure(Exception):
    def __init__(
        self,
        status: int,
        code: str,
        message: str,
        *,
        headers: dict[str, str] | None = None,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message
        self.headers = headers or {}
        self.details = details


class PlainFailure(Exception):
    """An error with a non-JSON body, as a generic rate limiter sends."""

    def __init__(self, status: int, text: str) -> None:
        super().__init__(text)
        self.status = status
        self.text = text


class FakeController:
    """Start with a ``with`` block; ``base_url`` is what the SDK should use."""

    def __init__(self, root: Path, *, api_key: str = "sk-sandbox-test") -> None:
        self.root = root
        self.api_key = api_key
        self.sandboxes: dict[str, FakeSandbox] = {}
        self.create_bodies: list[dict[str, Any]] = []
        self.requests: list[tuple[str, str]] = []
        self.process_starts: list[dict[str, Any]] = []
        self.concurrency_limit: int | None = None
        self.concurrency_refusals = 0
        self.max_live = 0
        #: Answer every create with this failure.
        self.refuse_creates: ApiFailure | None = None
        self.create_keys: list[str] = []
        self._created_by_key: dict[str, dict[str, Any]] = {}
        #: Answer this many creates with 503 idempotency_in_progress, having really
        #: created the sandbox, the way a lost response looks to a client.
        self.creates_in_progress = 0
        #: Answer this many creates like the per-IP limiter: 429 with a plain body.
        self.plain_429s_on_create = 0
        #: Grant at most this lifetime, as a plan's runtime limit does: silently.
        self.clamp_lifetime_s: int | None = None
        #: Enforce a process time limit this short, since real limits are minutes.
        self.process_time_limit_s: float | None = None
        self.lock = threading.RLock()
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def __enter__(self) -> FakeController:
        controller = self

        class Handler(_Handler):
            owner = controller

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self._server.daemon_threads = True
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_: Any) -> None:
        for sandbox in list(self.sandboxes.values()):
            self._kill_all(sandbox)
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()

    @property
    def base_url(self) -> str:
        assert self._server is not None
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    # ------------------------------------------------------------------
    # Faults
    # ------------------------------------------------------------------

    def _kill_all(self, sandbox: FakeSandbox) -> None:
        for process in sandbox.processes.values():
            if process.popen.poll() is None:
                process.killed = True
                try:
                    os.killpg(process.popen.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.popen.wait()

    def preempt(self, sandbox_id: str) -> None:
        """A Spot reclaim: the pod and everything in it are simply gone."""
        with self.lock:
            sandbox = self.sandboxes[sandbox_id]
            self._kill_all(sandbox)
            sandbox.status = "terminated"

    def mark_not_running(self, sandbox_id: str) -> None:
        """Past its expiry: still known to the controller, no longer usable."""
        with self.lock:
            sandbox = self.sandboxes[sandbox_id]
            self._kill_all(sandbox)
            sandbox.status = "error"

    def exhaust_budget(self, sandbox_id: str) -> None:
        """The budget enforcer stopped this sandbox (reason budget_exhausted)."""
        with self.lock:
            sandbox = self.sandboxes[sandbox_id]
            self._kill_all(sandbox)
            sandbox.status = "terminated"
            sandbox.budget_exhausted = True

    def live(self) -> list[FakeSandbox]:
        with self.lock:
            return [s for s in self.sandboxes.values() if s.status == "running"]

    # ------------------------------------------------------------------
    # API
    # ------------------------------------------------------------------

    def _sandbox(self, sandbox_id: str) -> FakeSandbox:
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None or sandbox.status == "terminated":
            raise ApiFailure(
                404, "SANDBOX_NOT_FOUND", f"Sandbox {sandbox_id} not found"
            )
        if sandbox.status != "running":
            raise ApiFailure(409, "SANDBOX_NOT_RUNNING", "Sandbox is not running")
        return sandbox

    def _real_path(self, sandbox: FakeSandbox, path: str) -> Path:
        if not path.startswith(WORKSPACE + "/") and path != WORKSPACE:
            raise ApiFailure(403, "FORBIDDEN", f"path outside the workspace: {path}")
        relative = path[len(WORKSPACE) :].lstrip("/")
        return sandbox.root / "workspace" / relative

    def _rewrite(self, sandbox: FakeSandbox, value: str) -> str:
        real = str(sandbox.root / "workspace")
        return value.replace(WORKSPACE, real)

    def create(self, body: dict[str, Any], key: str) -> tuple[int, dict[str, Any]]:
        with self.lock:
            self.create_bodies.append(body)
            self.create_keys.append(key)
            if self.refuse_creates is not None:
                raise self.refuse_creates
            if self.plain_429s_on_create:
                self.plain_429s_on_create -= 1
                raise PlainFailure(429, "Too many requests, please try again later.")
            stored = self._created_by_key.get(key)
            if stored is not None:
                if self.creates_in_progress:
                    self.creates_in_progress -= 1
                    raise self._in_progress()
                return 201, stored
            if (
                self.concurrency_limit is not None
                and len(self.live()) >= self.concurrency_limit
            ):
                self.concurrency_refusals += 1
                raise ApiFailure(
                    429,
                    "CONCURRENT_LIMIT_EXCEEDED",
                    "Concurrent sandbox limit reached",
                    headers={"Retry-After": "0"},
                    details={
                        "current": len(self.live()),
                        "limit": self.concurrency_limit,
                    },
                )
            sandbox_id = f"sbx-{uuid.uuid4().hex[:12]}"
            root = self.root / sandbox_id
            (root / "workspace").mkdir(parents=True)
            self.sandboxes[sandbox_id] = FakeSandbox(
                sandbox_id=sandbox_id, root=root, body=body
            )
            self.max_live = max(self.max_live, len(self.live()))
            response = self._sandbox_json(self.sandboxes[sandbox_id])
            self._created_by_key[key] = response
            if self.creates_in_progress:
                self.creates_in_progress -= 1
                raise self._in_progress()
        return 201, response

    @staticmethod
    def _in_progress() -> ApiFailure:
        return ApiFailure(
            503,
            "SERVICE_UNAVAILABLE",
            "An operation with this idempotency key is still in progress",
            details={"reason": "idempotency_in_progress"},
        )

    def _sandbox_json(self, sandbox: FakeSandbox) -> dict[str, Any]:
        now = datetime.now(timezone.utc)
        lifetime = sandbox.body.get("timeout_seconds", 3600)
        if self.clamp_lifetime_s is not None:
            lifetime = min(lifetime, self.clamp_lifetime_s)
        return {
            "sandbox_id": sandbox.sandbox_id,
            "org_id": "org-test",
            "session_id": str(uuid.uuid4()),
            "status": sandbox.status,
            "created_at": now.isoformat(),
            "expires_at": (now + timedelta(seconds=lifetime)).isoformat(),
            "cpus": sandbox.body.get("cpus"),
            "memory": sandbox.body.get("memory"),
            "resource_class": sandbox.body.get("resource_class", "dedicated"),
        }

    def stop(self, sandbox_id: str) -> tuple[int, dict[str, Any]]:
        with self.lock:
            sandbox = self.sandboxes.get(sandbox_id)
            if sandbox is None or sandbox.status != "running":
                raise ApiFailure(
                    404, "SANDBOX_NOT_FOUND", f"Sandbox {sandbox_id} not found"
                )
            self._kill_all(sandbox)
            sandbox.status = "terminated"
            shutil.rmtree(sandbox.root, ignore_errors=True)
        return 200, {"success": True}

    def write_files(self, sandbox_id: str, body: dict[str, Any]) -> tuple[int, Any]:
        sandbox = self._sandbox(sandbox_id)
        for item in body["files"]:
            target = self._real_path(sandbox, item["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(base64.b64decode(item["content"]))
        return 200, {"success": True, "written": len(body["files"])}

    def read_file(self, sandbox_id: str, path: str) -> tuple[int, Any]:
        sandbox = self._sandbox(sandbox_id)
        target = self._real_path(sandbox, path)
        if not target.is_file():
            raise ApiFailure(404, "FILE_NOT_FOUND", f"No such file: {path}")
        content = target.read_bytes()
        return 200, {
            "path": path,
            "content": base64.b64encode(content).decode(),
            "size": len(content),
        }

    def list_files(self, sandbox_id: str, path: str) -> tuple[int, Any]:
        sandbox = self._sandbox(sandbox_id)
        target = self._real_path(sandbox, path)
        if not target.exists():
            raise ApiFailure(404, "FILE_NOT_FOUND", f"No such directory: {path}")
        real_root = sandbox.root / "workspace"
        entries = []
        for item in sorted(target.rglob("*")):
            virtual = f"{WORKSPACE}/{item.relative_to(real_root).as_posix()}"
            entries.append(
                {
                    "name": item.name,
                    "path": virtual,
                    "type": "directory" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 4096,
                    "modified_at": datetime.now(timezone.utc).isoformat(),
                }
            )
        return 200, {"files": entries}

    def start_process(self, sandbox_id: str, body: dict[str, Any]) -> tuple[int, Any]:
        sandbox = self._sandbox(sandbox_id)
        command = body["command"]
        argv = command if isinstance(command, list) else ["sh", "-c", command]
        # CommandArgSchema in the controller's schemas/index.ts.
        for argument in argv:
            if not 1 <= len(argument) <= 10000 or re.search(r"[\0\r\t\n]", argument):
                raise ApiFailure(
                    400,
                    "VALIDATION_ERROR",
                    "Command argument must not contain control characters",
                )
        self.process_starts.append({"sandbox_id": sandbox_id, **body})
        real_argv = [self._rewrite(sandbox, arg) for arg in argv]
        # The fake image's interpreter: this repo's Python with this repo on its path.
        real_argv = [sys.executable if a == "python3" else a for a in real_argv]
        process_id = uuid.uuid4().hex
        logs = sandbox.root / "proc" / process_id
        logs.mkdir(parents=True)
        stdout, stderr = logs / "stdout.log", logs / "stderr.log"
        env = {
            **os.environ,
            "PYTHONPATH": str(REPO_ROOT),
            **(body.get("env") or {}),
        }
        env.pop("STATESET_API_KEY", None)
        workspace = sandbox.root / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        with stdout.open("wb") as out, stderr.open("wb") as err:
            popen = subprocess.Popen(
                real_argv,
                cwd=workspace,
                env=env,
                stdout=out,
                stderr=err,
                start_new_session=True,
            )
        process = FakeProcess(
            process_id=process_id,
            popen=popen,
            stdout=stdout,
            stderr=stderr,
            name=body.get("name"),
            command=list(argv),
        )
        sandbox.processes[process_id] = process
        return 201, self._process_json(process)

    def _process_json(self, process: FakeProcess) -> dict[str, Any]:
        limit = self.process_time_limit_s
        if (
            limit is not None
            and process.popen.poll() is None
            and time.monotonic() - process.started > limit
        ):
            process.timed_out = True
            os.killpg(process.popen.pid, signal.SIGKILL)
            process.popen.wait()
        code = process.popen.poll()
        if process.timed_out:
            status = "timed_out"
        elif process.killed:
            status = "killed"
        elif code is None:
            status = "running"
        else:
            status = "exited"
        return {
            "process_id": process.process_id,
            "command": process.command,
            "pid": process.popen.pid,
            "status": status,
            "restart": "never",
            "exit_code": code if status == "exited" else None,
            "name": process.name,
            "log": {
                "stdout_bytes": process.stdout.stat().st_size,
                "stderr_bytes": process.stderr.stat().st_size,
            },
        }

    def get_process(self, sandbox_id: str, process_id: str) -> tuple[int, Any]:
        process = self._process(sandbox_id, process_id)
        return 200, self._process_json(process)

    def _process(self, sandbox_id: str, process_id: str) -> FakeProcess:
        sandbox = self._sandbox(sandbox_id)
        process = sandbox.processes.get(process_id)
        if process is None:
            raise ApiFailure(404, "PROCESS_NOT_FOUND", f"No process {process_id}")
        return process

    def logs(
        self, sandbox_id: str, process_id: str, query: dict[str, list[str]]
    ) -> tuple[int, Any]:
        process = self._process(sandbox_id, process_id)
        stream = query.get("stream", ["stdout"])[0]
        offset = int(query.get("offset", ["0"])[0])
        limit = int(query.get("limit", [str(4 * 1024 * 1024)])[0])
        path = process.stdout if stream == "stdout" else process.stderr
        with path.open("rb") as handle:
            handle.seek(offset)
            data = handle.read(limit)
        end = path.stat().st_size
        return 200, {
            "process_id": process_id,
            "stream": stream,
            "offset": offset,
            "next_offset": offset + len(data),
            "end_offset": end,
            "data": base64.b64encode(data).decode(),
            "eof": offset + len(data) >= end,
            "truncated": False,
            "status": self._process_json(process)["status"],
        }

    def cost(self, sandbox_id: str) -> tuple[int, Any]:
        sandbox = self.sandboxes.get(sandbox_id)
        if sandbox is None:
            raise ApiFailure(404, "SANDBOX_NOT_FOUND", "no such sandbox")
        budget = sandbox.body.get("budget") or {}
        return 200, {
            "sandbox_id": sandbox_id,
            "total_cents": sandbox.cost_cents,
            "compute": {"cents": sandbox.cost_cents},
            "llm": {"cents": 0},
            "storage": {"cents": 0, "metered": False},
            "budget": {
                "max_cost_cents": budget.get("max_cost_cents"),
                "on_exhausted": budget.get("on_exhausted", "stop"),
                "exhausted_at": (
                    datetime.now(timezone.utc).isoformat()
                    if sandbox.budget_exhausted
                    else None
                ),
                "exhausted_reason": (
                    "compute budget exhausted" if sandbox.budget_exhausted else None
                ),
            },
        }


class _Handler(BaseHTTPRequestHandler):
    owner: FakeController
    protocol_version = "HTTP/1.1"

    ROUTES: list[tuple[str, re.Pattern[str], str]] = [
        ("POST", re.compile(r"^/api/v1/sandbox/create$"), "create"),
        ("POST", re.compile(r"^/api/v1/sandbox/([^/]+)/stop$"), "stop"),
        ("POST", re.compile(r"^/api/v1/sandbox/([^/]+)/files$"), "write_files"),
        ("GET", re.compile(r"^/api/v1/sandbox/([^/]+)/files$"), "read_file"),
        ("GET", re.compile(r"^/api/v1/sandbox/([^/]+)/files/list$"), "list_files"),
        ("POST", re.compile(r"^/api/v1/sandbox/([^/]+)/processes$"), "start_process"),
        (
            "GET",
            re.compile(r"^/api/v1/sandbox/([^/]+)/processes/([^/]+)/logs$"),
            "logs",
        ),
        (
            "GET",
            re.compile(r"^/api/v1/sandbox/([^/]+)/processes/([^/]+)$"),
            "get_process",
        ),
        ("GET", re.compile(r"^/api/v1/sandbox/([^/]+)/cost$"), "cost"),
    ]

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        return

    def _send(
        self, status: int, payload: Any, headers: dict[str, str] | None = None
    ) -> None:
        data = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Request-Id", uuid.uuid4().hex)
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(data)

    def _handle(self, method: str) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        self.owner.requests.append((method, parsed.path))
        try:
            authorization = self.headers.get("Authorization", "")
            if authorization != f"ApiKey {self.owner.api_key}":
                raise ApiFailure(401, "AUTH_INVALID_API_KEY", "Invalid API key")
            body = json.loads(raw) if raw else {}
            for route_method, pattern, name in self.ROUTES:
                match = pattern.match(parsed.path)
                if route_method != method or match is None:
                    continue
                status, payload = self._dispatch(name, match.groups(), body, query)
                self._send(status, payload)
                return
            raise ApiFailure(404, "ROUTE_NOT_FOUND", f"{method} {parsed.path}")
        except PlainFailure as failure:
            data = failure.text.encode()
            self.send_response(failure.status)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Retry-After", "0")
            self.end_headers()
            self.wfile.write(data)
        except ApiFailure as failure:
            error: dict[str, Any] = {"code": failure.code, "message": failure.message}
            if failure.details:
                error["details"] = failure.details
            self._send(failure.status, {"error": error}, failure.headers)

    def _dispatch(
        self,
        name: str,
        groups: tuple[str, ...],
        body: dict[str, Any],
        query: dict[str, list[str]],
    ) -> tuple[int, Any]:
        owner = self.owner
        if name == "create":
            return owner.create(body, self.headers.get("Idempotency-Key", ""))
        if name == "read_file":
            return owner.read_file(groups[0], query["path"][0])
        if name == "list_files":
            return owner.list_files(groups[0], query["path"][0])
        if name in ("write_files", "start_process"):
            return getattr(owner, name)(groups[0], body)
        if name == "logs":
            return owner.logs(groups[0], groups[1], query)
        return getattr(owner, name)(*groups)

    def do_GET(self) -> None:  # noqa: N802
        self._handle("GET")

    def do_POST(self) -> None:  # noqa: N802
        self._handle("POST")
