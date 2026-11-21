"""Live llama.cpp integration, opt-in through environment variables.

Set ``STATESET_LLAMA_CPP_BIN`` to a directory or binary providing
``llama-server`` semantics and ``STATESET_LLAMA_CPP_MODEL`` to a local
GGUF file. Without both, the test skips; the default suite never
downloads models or requires llama.cpp.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from stateset_agents.training.llama_cpp_backend import (
    LlamaCppRolloutBackend,
    LlamaCppServer,
)

pytestmark = [pytest.mark.integration, pytest.mark.slow]


@pytest.mark.asyncio
async def test_live_gguf_rollout_generates_tokens_with_logprobs() -> None:
    binary = os.getenv("STATESET_LLAMA_CPP_BIN")
    model = os.getenv("STATESET_LLAMA_CPP_MODEL")
    if not binary or not model:
        pytest.skip(
            "set STATESET_LLAMA_CPP_BIN and STATESET_LLAMA_CPP_MODEL "
            "to run the live llama.cpp test"
        )
    binary_path = Path(binary)
    resolved_binary = (
        str(binary_path / "llama-server") if binary_path.is_dir() else str(binary_path)
    )

    with LlamaCppServer(
        model, binary=resolved_binary, context_size=1024, threads=4
    ) as server:
        backend = LlamaCppRolloutBackend.from_server(server)
        results = await backend.generate_with_logprobs(
            ["The capital of France is"],
            temperature=0.0,
            max_tokens=8,
        )

    assert len(results) == 1
    result = results[0]
    assert result.response.strip()
    assert result.prompt_token_ids
    assert len(result.response_token_ids) == len(result.token_logprobs) > 0
