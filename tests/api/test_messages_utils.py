import json

import pytest

from stateset_agents.api.messages_models import MessageInput, MessagesRequest
from stateset_agents.api.messages_utils import (
    anthropic_messages_to_openai,
    openai_response_to_anthropic,
    validate_tool_choice,
    validate_tools,
)
from stateset_agents.api.services.inference_service import (
    InferenceConfig,
    InferenceService,
)


def test_anthropic_to_openai_conversion_with_tools():
    messages = [
        {
            "role": "user",
            "content": [{"type": "text", "text": "Hello"}],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Sure"},
                {
                    "type": "tool_use",
                    "id": "tool_1",
                    "name": "search",
                    "input": {"q": "x"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "tool_1", "content": "result"}
            ],
        },
    ]

    openai_messages = anthropic_messages_to_openai(messages, system="You are helpful.")

    assert openai_messages[0]["role"] == "system"
    assert openai_messages[1]["role"] == "user"
    assert openai_messages[2]["role"] == "assistant"
    assert "tool_calls" in openai_messages[2]
    assert openai_messages[3]["role"] == "tool"
    assert openai_messages[3]["tool_call_id"] == "tool_1"


def test_anthropic_to_openai_preserves_tool_calls_when_content_missing():
    messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "search", "arguments": '{"q":"x"}'},
                }
            ],
        }
    ]

    openai_messages = anthropic_messages_to_openai(messages)

    assert openai_messages[0]["role"] == "assistant"
    assert openai_messages[0]["content"] == ""
    assert openai_messages[0]["tool_calls"][0]["id"] == "call_1"


def test_openai_response_to_anthropic():
    response = {
        "id": "chatcmpl_123",
        "model": "moonshotai/Kimi-K2.5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "Hello!",
                    "tool_calls": [
                        {
                            "id": "tool_1",
                            "type": "function",
                            "function": {"name": "search", "arguments": '{"q": "x"}'},
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    anthropic = openai_response_to_anthropic(response)
    assert anthropic["id"] == "chatcmpl_123"
    assert anthropic["stop_reason"] == "tool_use"
    assert any(block["type"] == "tool_use" for block in anthropic["content"])


def test_openai_response_to_anthropic_handles_malformed_tool_calls():
    response = {
        "id": "chatcmpl_123",
        "model": "moonshotai/Kimi-K2.5",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        "not-a-call",
                        {"function": {"name": 42, "arguments": "{bad-json"}},
                        {"function": {"name": "search", "arguments": {"q": "x"}}},
                        {"function": {"name": "echo", "arguments": None}, "id": ""},
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    anthropic = openai_response_to_anthropic(response)
    tool_calls = [block for block in anthropic["content"] if block["type"] == "tool_use"]

    assert len(tool_calls) == 3
    assert [block["name"] for block in tool_calls] == ["", "search", "echo"]


def test_openai_response_to_anthropic_invalid_usage_values_default_to_zero():
    response = {
        "id": "chatcmpl_123",
        "model": "moonshotai/Kimi-K2.5",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": "10", "completion_tokens": -1},
    }

    anthropic = openai_response_to_anthropic(response)

    assert anthropic["usage"]["input_tokens"] == 0
    assert anthropic["usage"]["output_tokens"] == 0


def test_validate_tools_rejects_missing_name():
    with pytest.raises(ValueError):
        validate_tools([{"type": "function", "function": {"parameters": {}}}])


def test_validate_tool_choice_rejects_invalid():
    with pytest.raises(ValueError):
        validate_tool_choice({"type": "tool"})


@pytest.mark.asyncio
async def test_inference_service_stub_response():
    config = InferenceConfig(
        backend="stub",
        base_url="http://localhost:8000",
        default_model="moonshotai/Kimi-K2.5",
    )
    service = InferenceService(config)
    request = MessagesRequest(
        model="moonshotai/Kimi-K2.5",
        max_tokens=64,
        messages=[MessageInput(role="user", content="Hello")],
    )

    response = await service.create_anthropic_response(request)
    assert response["role"] == "assistant"
    assert response["usage"]["output_tokens"] > 0


@pytest.mark.asyncio
async def test_anthropic_stream_uses_backend_usage_when_present():
    config = InferenceConfig(
        backend="stub",
        base_url="http://localhost:8000",
        default_model="moonshotai/Kimi-K2.5",
    )
    service = InferenceService(config)

    async def fake_stream_openai(_request):
        yield (
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {"content": "Hello world"},
                            "finish_reason": None,
                        }
                    ]
                }
            )
            + "\n\n"
        )
        yield (
            "data: "
            + json.dumps(
                {
                    "choices": [
                        {
                            "index": 0,
                            "delta": {},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5},
                }
            )
            + "\n\n"
        )
        yield "data: [DONE]\n\n"

    service.stream_openai = fake_stream_openai  # type: ignore[assignment]

    request = MessagesRequest(
        model="moonshotai/Kimi-K2.5",
        max_tokens=16,
        stream=True,
        messages=[MessageInput(role="user", content="Hi")],
    )
    body = "".join([chunk async for chunk in service.stream_anthropic(request)])

    assert '"output_tokens":5' in body.replace(" ", "")
