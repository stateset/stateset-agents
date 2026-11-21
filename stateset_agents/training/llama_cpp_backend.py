"""llama.cpp (GGUF) local inference backend.

GGUF files served by ``llama-server`` are the local-inference half of the
dual-engine design in ``docs/NATIVE_LOCAL_ENGINE.md``: safetensors stay
canonical for training and weight edits, while llama.cpp runs quantized
checkpoints on hardware (for example a no-CUDA laptop) where the installed
Transformers runtime cannot load the original architecture at all.

This module is standard-library only so importing it stays torch-free. It
provides a small HTTP client for llama-server, a loopback subprocess
manager, and a rollout backend implementing the
``MultiTurnAgent.set_rollout_backend`` contract (exact prompt/response
token ids plus per-token log-probs, shaped like
``training.vllm_backend.GenerationResult``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import shutil
import socket
import subprocess
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "LlamaCppClient",
    "LlamaCppCompletion",
    "LlamaCppRolloutBackend",
    "LlamaCppServer",
]

_FINISH_REASONS = {
    "eos": "stop",
    "word": "stop",
    "stop": "stop",
    "limit": "length",
    "length": "length",
}


def _map_finish_reason(value: Any) -> str:
    """Map a llama.cpp ``stop_type`` onto the rollout finish vocabulary."""
    return _FINISH_REASONS.get(str(value or "").lower(), "abort")


@dataclass(frozen=True)
class LlamaCppCompletion:
    """One llama.cpp completion with exact token ids and log-probs."""

    prompt: str
    response: str
    prompt_token_ids: list[int]
    response_token_ids: list[int]
    token_logprobs: list[float]
    finish_reason: str

    @property
    def full_text(self) -> str:
        """Prompt plus response, mirroring ``GenerationResult``."""
        return self.prompt + self.response

    @property
    def sequence_length(self) -> int:
        """Number of generated tokens."""
        return len(self.response_token_ids)

    @property
    def cumulative_logprob(self) -> float:
        """Sum of per-token log-probs."""
        return float(sum(self.token_logprobs))


class LlamaCppClient:
    """Minimal llama-server HTTP client (tokenize, completion, chat)."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 120.0,
        opener: Callable[..., Any] | None = None,
    ) -> None:
        normalized = str(base_url).strip().rstrip("/")
        if not normalized.startswith(("http://", "https://")):
            raise ValueError(f"base_url must be an HTTP URL: {base_url!r}")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.base_url = normalized
        self.timeout_seconds = float(timeout_seconds)
        self._opener = opener or urllib.request.urlopen

    def _request_json(
        self, path: str, payload: Mapping[str, Any] | None = None
    ) -> dict[str, Any]:
        data = None
        headers: dict[str, str] = {}
        if payload is not None:
            data = json.dumps(dict(payload)).encode("utf-8")
            headers["Content-Type"] = "application/json"
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=data,
            headers=headers,
            method="POST" if data is not None else "GET",
        )
        try:
            with self._opener(request, timeout=self.timeout_seconds) as response:
                body = response.read()
        except (OSError, TimeoutError) as exc:
            raise RuntimeError(f"llama-server request to {path} failed: {exc}") from exc
        try:
            parsed = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"llama-server returned non-JSON for {path}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"llama-server returned a non-object for {path}")
        return parsed

    def health(self) -> bool:
        """Return True when llama-server reports itself ready."""
        try:
            self._request_json("/health")
        except RuntimeError:
            return False
        return True

    def tokenize(self, text: str) -> list[int]:
        """Tokenize ``text`` with the served model's tokenizer."""
        data = self._request_json("/tokenize", {"content": text})
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) for token in tokens
        ):
            raise ValueError("llama-server /tokenize returned no token ids")
        return [int(token) for token in tokens]

    def completion(
        self,
        prompt: str,
        *,
        max_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float | None = None,
        top_k: int | None = None,
        stop: Sequence[str] | None = None,
    ) -> LlamaCppCompletion:
        """Complete ``prompt`` and return exact token ids and log-probs."""
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        payload: dict[str, Any] = {
            "prompt": prompt,
            "n_predict": int(max_tokens),
            "temperature": float(temperature),
            "n_probs": 1,
            "return_tokens": True,
        }
        if top_p is not None:
            payload["top_p"] = float(top_p)
        if top_k is not None:
            payload["top_k"] = int(top_k)
        if stop:
            payload["stop"] = list(stop)

        prompt_token_ids = self.tokenize(prompt)
        data = self._request_json("/completion", payload)

        tokens = data.get("tokens")
        if not isinstance(tokens, list) or not all(
            isinstance(token, int) for token in tokens
        ):
            raise ValueError("llama-server /completion returned no token ids")
        response_token_ids = [int(token) for token in tokens]

        probabilities = data.get("completion_probabilities")
        if not isinstance(probabilities, list) or len(probabilities) != len(
            response_token_ids
        ):
            raise ValueError("llama-server token ids and log-probs differ in length")
        token_logprobs: list[float] = []
        for entry, token_id in zip(probabilities, response_token_ids, strict=True):
            if not isinstance(entry, Mapping) or "logprob" not in entry:
                raise ValueError("llama-server omitted a token log-prob")
            if "id" in entry and int(entry["id"]) != token_id:
                raise ValueError("llama-server log-probs do not match its token ids")
            value = float(entry["logprob"])
            if not math.isfinite(value) or value > 1e-6:
                raise ValueError("llama-server returned invalid log-probs")
            token_logprobs.append(value)

        content = data.get("content")
        if not isinstance(content, str):
            raise ValueError("llama-server /completion returned no content")
        return LlamaCppCompletion(
            prompt=prompt,
            response=content,
            prompt_token_ids=prompt_token_ids,
            response_token_ids=response_token_ids,
            token_logprobs=token_logprobs,
            finish_reason=_map_finish_reason(data.get("stop_type")),
        )

    def chat(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        **extra: Any,
    ) -> str:
        """Run one OpenAI-style chat completion and return its content."""
        payload: dict[str, Any] = {"messages": [dict(message) for message in messages]}
        if model is not None:
            payload["model"] = model
        if max_tokens is not None:
            payload["max_tokens"] = int(max_tokens)
        if temperature is not None:
            payload["temperature"] = float(temperature)
        payload.update(extra)
        data = self._request_json("/v1/chat/completions", payload)
        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("llama-server chat returned no choices")
        message = choices[0].get("message") if isinstance(choices[0], dict) else None
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str):
            raise ValueError("llama-server chat returned no message content")
        return content


def _free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, 0))
        return int(sock.getsockname()[1])


class LlamaCppServer:
    """Own one loopback ``llama-server`` child process for a GGUF file."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        binary: str | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        alias: str | None = None,
        context_size: int = 4096,
        threads: int | None = None,
        startup_timeout_seconds: float = 180.0,
        timeout_seconds: float = 120.0,
        extra_args: Sequence[str] = (),
    ) -> None:
        if not 0 <= int(port) <= 65535:
            raise ValueError("port must be between 0 and 65535")
        if context_size < 1:
            raise ValueError("context_size must be at least 1")
        if threads is not None and threads < 1:
            raise ValueError("threads must be at least 1")
        if startup_timeout_seconds <= 0:
            raise ValueError("startup_timeout_seconds must be positive")
        self.model_path = Path(model_path)
        self._binary = binary
        self.host = host
        self.port = int(port) or _free_port(host)
        self.alias = alias
        self.context_size = int(context_size)
        self.threads = threads
        self.startup_timeout_seconds = float(startup_timeout_seconds)
        self.timeout_seconds = float(timeout_seconds)
        self.extra_args = tuple(str(arg) for arg in extra_args)
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def base_url(self) -> str:
        """Base URL the child serves (or will serve after ``start``)."""
        return f"http://{self.host}:{self.port}"

    @property
    def running(self) -> bool:
        """Whether the owned child process is still alive."""
        return self._process is not None and self._process.poll() is None

    @property
    def client(self) -> LlamaCppClient:
        """Client bound to this server's base URL."""
        return LlamaCppClient(self.base_url, timeout_seconds=self.timeout_seconds)

    def _binary_candidate(self) -> str:
        return self._binary or os.getenv("LLAMA_SERVER_BIN") or "llama-server"

    def build_command(self) -> list[str]:
        """Argv for the child process, without resolving the binary."""
        command = [
            self._binary_candidate(),
            "-m",
            str(self.model_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "-c",
            str(self.context_size),
        ]
        if self.alias is not None:
            command += ["--alias", self.alias]
        if self.threads is not None:
            command += ["-t", str(int(self.threads))]
        command += list(self.extra_args)
        return command

    def _resolve_binary(self) -> str:
        candidate = self._binary_candidate()
        resolved = shutil.which(candidate)
        if resolved is None:
            raise FileNotFoundError(
                f"llama-server binary not found ({candidate!r}); build "
                "llama.cpp or set LLAMA_SERVER_BIN to its llama-server path"
            )
        return resolved

    def start(self) -> LlamaCppServer:
        """Start the child and wait until it reports healthy."""
        if self.running:
            return self
        command = self.build_command()
        command[0] = self._resolve_binary()
        if not self.model_path.is_file():
            raise FileNotFoundError(f"GGUF model not found: {self.model_path}")
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._process = process
        client = self.client
        deadline = time.monotonic() + self.startup_timeout_seconds
        while time.monotonic() < deadline:
            if process.poll() is not None:
                raise RuntimeError(
                    "llama-server exited during startup "
                    f"(returncode={process.returncode})"
                )
            if client.health():
                logger.info("llama-server ready at %s", self.base_url)
                return self
            time.sleep(0.5)
        self.stop()
        raise TimeoutError(
            f"llama-server did not become healthy at {self.base_url} "
            f"within {self.startup_timeout_seconds:.0f}s"
        )

    def stop(self) -> None:
        """Terminate the owned child process, if one is running."""
        process = self._process
        self._process = None
        if process is None or process.poll() is not None:
            return
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=15)

    def __enter__(self) -> LlamaCppServer:
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def _as_generation_result(completion: LlamaCppCompletion) -> Any:
    """Return a ``GenerationResult`` when torch is present, else duck-type."""
    try:
        from stateset_agents.training.vllm_backend import GenerationResult
    except Exception:  # torch-free environments keep the local dataclass
        return completion
    return GenerationResult(
        prompt=completion.prompt,
        response=completion.response,
        full_text=completion.full_text,
        prompt_token_ids=list(completion.prompt_token_ids),
        response_token_ids=list(completion.response_token_ids),
        token_logprobs=list(completion.token_logprobs),
        cumulative_logprob=completion.cumulative_logprob,
        sequence_length=completion.sequence_length,
        finish_reason=completion.finish_reason,
    )


class LlamaCppRolloutBackend:
    """Rollout backend serving exact llama.cpp tokens and log-probs.

    Attach to an agent with ``agent.set_rollout_backend(backend)``. The
    GGUF weights are static, so this backend intentionally exposes no
    ``sync_weights``; trainers will mark it stale after an optimizer step
    instead of silently sampling an old policy.
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        client: LlamaCppClient | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        if (base_url is None) == (client is None):
            raise ValueError("pass exactly one of base_url or client")
        self.client = client or LlamaCppClient(
            str(base_url), timeout_seconds=timeout_seconds
        )

    @classmethod
    def from_server(cls, server: LlamaCppServer) -> LlamaCppRolloutBackend:
        """Build a backend bound to a (started) :class:`LlamaCppServer`."""
        return cls(client=server.client)

    async def generate_with_logprobs(
        self, prompts: Sequence[str], **kwargs: Any
    ) -> list[Any]:
        """Generate one result per prompt, sequentially per server slot."""
        sampling = dict(kwargs.pop("sampling_params", None) or {})
        sampling.update(kwargs)
        temperature = float(sampling.pop("temperature", 1.0))
        top_p_raw = sampling.pop("top_p", None)
        top_k_raw = sampling.pop("top_k", None)
        max_tokens = int(sampling.pop("max_tokens", 256))
        stop = sampling.pop("stop", None)
        if sampling:
            raise TypeError(f"unsupported sampling arguments: {sorted(sampling)}")
        results: list[Any] = []
        for prompt in prompts:
            completion = await asyncio.to_thread(
                self.client.completion,
                str(prompt),
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=None if top_p_raw is None else float(top_p_raw),
                top_k=None if top_k_raw is None else int(top_k_raw),
                stop=stop,
            )
            results.append(_as_generation_result(completion))
        return results
