"""05 — Serve an agent via the FastAPI service.

The framework ships an OpenAI-compatible FastAPI service. This example wires
an agent into the service and shows two ways to talk to it:

  1. As a plain HTTP POST to /v1/chat/completions
  2. Via the OpenAI Python SDK (because /v1/chat/completions is compatible)

Install:
    pip install "stateset-agents[api]"

Run (terminal 1 — starts the server on :8001, stub-backed):
    python 05_serve_agent.py

Run (terminal 2 — hit the endpoint):
    curl -X POST http://localhost:8001/v1/chat/completions \\
      -H "Content-Type: application/json" \\
      -d '{"model": "stub", "messages": [{"role": "user", "content": "hello"}]}'

Or with the OpenAI SDK (after `pip install openai`):

    from openai import OpenAI
    client = OpenAI(base_url="http://localhost:8001/v1", api_key="not-needed")
    resp = client.chat.completions.create(
        model="stub", messages=[{"role": "user", "content": "hello"}]
    )
    print(resp.choices[0].message.content)

Production: replace the stub agent with a real one (load a trained checkpoint
via AgentConfig(model_name="./outputs/your_run")), set CORS / rate-limit env
vars, and run behind uvicorn workers. See whitepaper §7 for the full ops
surface (Helm chart, Prometheus, dashboard).
"""

from __future__ import annotations

import sys

try:
    import uvicorn
except ImportError:
    print("Missing uvicorn. Install with: pip install 'stateset-agents[api]'")
    sys.exit(1)

from stateset_agents.api.main import app


def main() -> int:
    print("Starting StateSet Agents service on http://0.0.0.0:8001")
    print("Try:")
    print('  curl -X POST http://localhost:8001/v1/chat/completions \\')
    print('       -H "Content-Type: application/json" \\')
    print('       -d \'{"model": "stub", "messages": [{"role": "user", "content": "hello"}]}\'')
    print()
    uvicorn.run(app, host="0.0.0.0", port=8001, log_level="info")
    return 0


if __name__ == "__main__":
    sys.exit(main())
