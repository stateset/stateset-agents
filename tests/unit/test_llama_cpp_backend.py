"""Contract tests for the llama.cpp local inference backend.

A fake HTTP opener stands in for ``llama-server``; payloads mirror the
shapes observed from a real server (`/tokenize`, `/completion` with
``n_probs``/``return_tokens``, `/v1/chat/completions``). No binary,
model file, or socket is needed.
"""

from __future__ import annotations

import json

import pytest

from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.training.llama_cpp_backend import (
    LlamaCppClient,
    LlamaCppRolloutBackend,
    LlamaCppServer,
)


class _FakeResponse:
    def __init__(self, payload: dict, status: int = 200) -> None:
        self._body = json.dumps(payload).encode()
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def _fake_opener(calls: list[tuple[str, dict]]):
    def open_(request, timeout=None):  # noqa: ANN001, ANN202
        url = request.full_url
        payload = json.loads(request.data) if request.data else {}
        calls.append((url, payload))
        if url.endswith("/health"):
            return _FakeResponse({"status": "ok"})
        if url.endswith("/tokenize"):
            return _FakeResponse({"tokens": [11, 12]})
        if url.endswith("/completion"):
            return _FakeResponse(
                {
                    "content": " Paris",
                    "tokens": [21, 22],
                    "stop_type": "eos",
                    "completion_probabilities": [
                        {"id": 21, "logprob": -0.5},
                        {"id": 22, "logprob": -1.5},
                    ],
                }
            )
        if url.endswith("/v1/chat/completions"):
            return _FakeResponse({"choices": [{"message": {"content": "126"}}]})
        raise AssertionError(f"unexpected URL {url}")

    return open_


def _client() -> tuple[LlamaCppClient, list[tuple[str, dict]]]:
    calls: list[tuple[str, dict]] = []
    client = LlamaCppClient("http://127.0.0.1:9999", opener=_fake_opener(calls))
    return client, calls


def test_client_health_tokenize_and_chat() -> None:
    client, calls = _client()

    assert client.health() is True
    assert client.tokenize("hi") == [11, 12]
    assert client.chat([{"role": "user", "content": "2+2?"}]) == "126"
    chat_payload = calls[-1][1]
    assert chat_payload["messages"][0]["content"] == "2+2?"


def test_completion_parses_tokens_logprobs_and_finish_reason() -> None:
    client, calls = _client()

    result = client.completion(
        "The capital of France is",
        max_tokens=8,
        temperature=0.0,
        top_p=0.9,
        top_k=40,
    )

    assert result.prompt == "The capital of France is"
    assert result.response == " Paris"
    assert result.prompt_token_ids == [11, 12]
    assert result.response_token_ids == [21, 22]
    assert result.token_logprobs == [-0.5, -1.5]
    assert result.finish_reason == "stop"
    completion_payload = calls[-1][1]
    assert completion_payload["n_predict"] == 8
    assert completion_payload["n_probs"] == 1
    assert completion_payload["return_tokens"] is True
    assert completion_payload["top_k"] == 40


def test_completion_rejects_mismatched_logprobs() -> None:
    def bad_open(request, timeout=None):  # noqa: ANN001, ANN202
        if request.full_url.endswith("/tokenize"):
            return _FakeResponse({"tokens": [1]})
        return _FakeResponse(
            {
                "content": "x",
                "tokens": [1, 2],
                "stop_type": "limit",
                "completion_probabilities": [{"id": 1, "logprob": -0.1}],
            }
        )

    client = LlamaCppClient("http://127.0.0.1:9999", opener=bad_open)
    with pytest.raises(ValueError, match="log-probs"):
        client.completion("prompt")


def test_server_command_and_missing_binary() -> None:
    server = LlamaCppServer(
        "/models/model.gguf", binary="/opt/llama-server", port=18099
    )
    command = server.build_command()
    assert command[0] == "/opt/llama-server"
    assert command[1:3] == ["-m", "/models/model.gguf"]
    assert "127.0.0.1" in command
    assert server.base_url == "http://127.0.0.1:18099"

    missing = LlamaCppServer("/models/model.gguf", binary="/nonexistent/llama-server")
    with pytest.raises(FileNotFoundError, match="llama-server"):
        missing.start()


@pytest.mark.asyncio
async def test_rollout_backend_drives_stub_agent() -> None:
    client, _calls = _client()
    agent = MultiTurnAgent(
        AgentConfig(model_name="stub://x", use_stub_model=True, stub_responses=["ok"])
    )
    await agent.initialize()
    agent.set_rollout_backend(LlamaCppRolloutBackend(client=client))

    assert await agent.generate_response("capital?") == " Paris"
