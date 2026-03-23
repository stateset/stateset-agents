from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from stateset_agents.rewards.llm_judge import JudgeConfig, JudgeProvider, LLMJudge
from stateset_agents.rewards.multi_objective_components import ModelBasedRewardComponent
from stateset_agents.training.auto_research.llm_proposer import LLMProposer


class TestLLMProposerProviderResolution:
    def test_auto_backend_prefers_openai_env_when_available(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefghijklmnopqrstuvwxyz")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")

        proposer = LLMProposer(search_space=object())

        assert proposer.backend == "openai"
        assert proposer.model == "gpt-5.4-mini"

    def test_auto_backend_uses_model_hint_for_anthropic(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("LLM_PROPOSER_MODEL", "claude-sonnet-4-20250514")

        proposer = LLMProposer(search_space=object())

        assert proposer.backend == "anthropic"
        assert proposer.model == "claude-sonnet-4-20250514"

    def test_openai_compatible_alias_normalizes_to_openai(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")

        proposer = LLMProposer(search_space=object(), backend="openai-compatible")

        assert proposer.backend == "openai"
        assert proposer.model == "gpt-4.1"


class TestModelBasedRewardComponentProviderResolution:
    @pytest.mark.asyncio
    async def test_openai_like_client_uses_openai_model_env(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=AsyncMock(
                        return_value=SimpleNamespace(
                            choices=[SimpleNamespace(message=SimpleNamespace(content="0.8"))]
                        )
                    )
                )
            )
        )

        component = ModelBasedRewardComponent(api_client=client)
        score = await component.compute_score(
            [
                {"role": "user", "content": "How do I reset my password?"},
                {"role": "assistant", "content": "Use the reset link in settings."},
            ]
        )

        assert score == 0.8
        client.chat.completions.create.assert_awaited_once()
        assert client.chat.completions.create.await_args.kwargs["model"] == "gpt-5.4-mini"

    @pytest.mark.asyncio
    async def test_anthropic_like_client_uses_anthropic_model_env(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514")
        client = SimpleNamespace(
            messages=SimpleNamespace(
                create=AsyncMock(
                    return_value=SimpleNamespace(
                        content=[SimpleNamespace(text="0.6")]
                    )
                )
            )
        )

        component = ModelBasedRewardComponent(api_client=client)
        score = await component.compute_score(
            [
                {"role": "user", "content": "Is my order delayed?"},
                {"role": "assistant", "content": "I checked it and it ships tomorrow."},
            ]
        )

        assert score == 0.6
        client.messages.create.assert_awaited_once()
        assert client.messages.create.await_args.kwargs["model"] == "claude-sonnet-4-20250514"


class TestLLMJudgeProviderResolution:
    def test_judge_config_prefers_openai_env_when_available(self, monkeypatch):
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefghijklmnopqrstuvwxyz")
        monkeypatch.setenv("OPENAI_MODEL", "gpt-5.4-mini")

        config = JudgeConfig()

        assert config.provider == JudgeProvider.OPENAI
        assert config.model_name == "gpt-5.4-mini"
        assert config.api_key == "sk-proj-abcdefghijklmnopqrstuvwxyz"

    def test_judge_config_uses_model_hint_for_anthropic(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("STATESET_JUDGE_MODEL", "claude-sonnet-4-20250514")

        config = JudgeConfig()

        assert config.provider == JudgeProvider.ANTHROPIC
        assert config.model_name == "claude-sonnet-4-20250514"

    def test_llm_judge_normalizes_openai_compatible_alias(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "gpt-4.1")

        judge = LLMJudge(provider="openai-compatible")

        assert judge.config.provider == JudgeProvider.OPENAI
        assert judge.config.model_name == "gpt-4.1"
        assert type(judge.backend).__name__ == "OpenAIJudge"
