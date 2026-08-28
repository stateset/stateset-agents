"""Tests for framework-neutral shootout evaluation."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from stateset_agents.evaluation.framework_protocol import evaluate_causal_lm


class _Tokenizer:
    pad_token_id = 0

    def __call__(self, prompt: str, return_tensors: str) -> dict[str, torch.Tensor]:
        assert prompt.startswith("Question:")
        assert return_tensors == "pt"
        return {"input_ids": torch.tensor([[1, 2]])}

    def decode(self, tokens: torch.Tensor, skip_special_tokens: bool) -> str:
        assert skip_special_tokens is True
        return str(int(tokens[-1]))


class _Model:
    device = torch.device("cpu")

    def eval(self) -> None:
        return None

    def generate(self, input_ids: torch.Tensor, **_: Any) -> torch.Tensor:
        answer = input_ids[0, -1]
        return torch.cat((input_ids, answer.reshape(1, 1)), dim=1)


def test_evaluate_causal_lm_uses_identical_raw_prompt_protocol() -> None:
    examples = [SimpleNamespace(question="one", answer=2)]

    result = evaluate_causal_lm(
        _Model(),
        _Tokenizer(),
        examples,
        format_prompt=lambda example: f"Question: {example.question}",
        score_response=lambda example, response: (
            float(int(response) == example.answer),
            response.isdigit(),
        ),
        max_tokens=8,
    )

    assert result == {"pass_at_1": 1.0, "parse_rate": 1.0, "n": 1.0}
