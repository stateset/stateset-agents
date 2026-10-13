"""Offline stand-in for ``vllm serve``: an authenticated OpenAI-compatible stub.

Parses the subset of vLLM's CLI the agent-quality endpoint wrapper emits and
answers ``/v1/models``, ``/v1/chat/completions`` and ``/v1/completions``. The
completion text names the policy that answered ("baseline" or "trained") so
fake suites can derive outcomes from what the endpoint actually served.
"""

from __future__ import annotations

import json
import os
import signal
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def _values(argv: list[str], flag: str) -> list[str]:
    if flag not in argv:
        return []
    values = []
    for item in argv[argv.index(flag) + 1 :]:
        if item.startswith("--"):
            break
        values.append(item)
    return values


def main() -> None:
    argv = sys.argv[1:]
    assert argv[0] == "serve", argv
    model = argv[1]
    host = _values(argv, "--host")[0]
    port = int(_values(argv, "--port")[0])
    names = _values(argv, "--served-model-name") or [model]
    token = os.environ["VLLM_API_KEY"]
    is_trained_checkpoint = (Path(model) / ".stateset-adapter-sha256").is_file()
    models = {
        name: {
            "id": name,
            "object": "model",
            "root": model,
            "parent": None,
            "policy": "trained" if is_trained_checkpoint else "baseline",
        }
        for name in names
    }
    for module in _values(argv, "--lora-modules"):
        name, path = module.split("=", 1)
        models[name] = {
            "id": name,
            "object": "model",
            "root": path,
            "parent": names[0],
            "policy": "trained",
        }

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: dict) -> None:
            payload = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def _authorized(self) -> bool:
            if self.headers.get("Authorization") != f"Bearer {token}":
                self._send(401, {"error": "unauthorized"})
                return False
            return True

        def do_GET(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            if self.path == "/v1/models":
                public = [
                    {k: v for k, v in item.items() if k != "policy"}
                    for item in models.values()
                ]
                self._send(200, {"object": "list", "data": public})
            else:
                self._send(404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            served = models.get(body.get("model"))
            if served is None:
                self._send(404, {"error": f"model {body.get('model')} not served"})
                return
            text = served["policy"]
            if self.path == "/v1/chat/completions":
                choice = {"message": {"role": "assistant", "content": text}}
            elif self.path == "/v1/completions":
                choice = {"text": text}
            else:
                self._send(404, {"error": "not found"})
                return
            self._send(200, {"model": body["model"], "choices": [choice]})

        def log_message(self, *_: object) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
