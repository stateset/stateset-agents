# Qwen4-Max hosted inference

StateSet's inference gateway supports **`qwen4-max`** through QwenCloud's
OpenAI-compatible Chat Completions API. The Singapore model page provides this
model ID and endpoint:

```text
https://ap-southeast-1.maas.qwencloudapi.com/compatible-mode/v1
```

Use a QwenCloud pay-as-you-go API key created in **Singapore (`ap-southeast-1`)**.
Region keys and endpoints must match. `qwen4-max-preview` is a separate model
with separate Token Plan access; StateSet does not substitute it for
`qwen4-max`.

## Start the gateway

Install the API dependencies from the repository:

```bash
pip install -e '.[api]'
export INFERENCE_BACKEND=qwencloud
read -rs -p 'Singapore QwenCloud API key: ' DASHSCOPE_API_KEY
export DASHSCOPE_API_KEY

# Local development, listening only on loopback.
export API_ENVIRONMENT=development
export API_REQUIRE_AUTH=false
python -m uvicorn stateset_agents.api.main:app --host 127.0.0.1 --port 8000
```

The preset selects `qwen4-max` and the Singapore endpoint. If your shell already
defines `INFERENCE_BACKEND_URL`, `INFERENCE_DEFAULT_MODEL`, or
`INFERENCE_API_KEY`, those explicit overrides take precedence. The upstream key
stays on the gateway; client applications use the gateway's own authentication
when enabled.

To retain `/api/v1/conversations` history and ownership across gateway restarts,
also set `API_V1_CONVERSATION_DB` to an application-owned SQLite file. The
[recovery workflow](API_REFERENCE.md#durable-v1-conversation-recovery) preserves
interrupted-request uncertainty without replaying Qwen calls. The native
`/v1/chat/completions` endpoint remains stateless; clients supply its history.

In another terminal:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "qwen4-max",
    "messages": [{"role": "user", "content": "Write a friendly order confirmation."}],
    "enable_thinking": false,
    "max_completion_tokens": 512
  }'
```

Add `"stream": true` to receive SSE chunks. The OpenAI endpoint preserves
`reasoning_content`, tool calls, image/video content blocks, structured-output
`response_format` objects, and provider options such as `enable_thinking` and
`reasoning_effort`. Requests and responses retain their OpenAI field names.
Tool execution remains the client's responsibility.

For multi-turn reasoning, return the provider's assistant messages with their
`reasoning_content` and `tool_calls` intact. Keep reasoning separate from
`content`, and set `preserve_thinking: true` to reuse historical reasoning.
Qwen4's completion budget covers reasoning and the final answer, so
allow room for both; do not assume a small answer budget is sufficient with
thinking enabled. The preset defaults to 8,192 tokens when no budget is supplied.
An explicit `max_completion_tokens` does not receive an additional `max_tokens`.

The existing `/v1/messages` endpoint still converts Anthropic-style text and tool
messages. Use `/v1/chat/completions` for Qwen reasoning history and native
multimodal fields. This integration does not implement the Responses API or
provider-managed built-in tools.

## Python without running a gateway

```python
import asyncio
import os

from stateset_agents.api.openai_models import OpenAIChatCompletionRequest
from stateset_agents.api.services.inference_service import (
    InferenceConfig,
    InferenceService,
)


async def main():
    service = InferenceService(
        InferenceConfig.qwencloud(api_key=os.environ["DASHSCOPE_API_KEY"])
    )
    try:
        response = await service.create_openai_response(
            OpenAIChatCompletionRequest(
                model="qwen4-max",
                messages=[{"role": "user", "content": "Hello!"}],
                enable_thinking=False,
                max_completion_tokens=512,
            )
        )
        print(response["choices"][0]["message"]["content"])
    finally:
        await service.aclose()


asyncio.run(main())
```

## Configuration and limits

| Setting | QwenCloud default |
| --- | --- |
| `INFERENCE_BACKEND` | Set to `qwencloud` to enable this provider |
| `INFERENCE_BACKEND_URL` | Singapore URL above, including `/compatible-mode/v1` |
| `INFERENCE_DEFAULT_MODEL` | `qwen4-max` |
| `INFERENCE_API_KEY` | Optional explicit upstream key; otherwise `DASHSCOPE_API_KEY` |
| `INFERENCE_DEFAULT_MAX_TOKENS` | `8192` |
| `INFERENCE_TIMEOUT_SECONDS` | `120`; increase for longer reasoning requests |
| `INFERENCE_MAX_RETRIES` | `0`; avoids automatic replay of billed requests |
| `INFERENCE_HEALTH_PATH` | `/models`, authenticated against the API base |

Model listing and health use the authenticated upstream API. Missing credentials
fail before creating an HTTP client. Discovery errors are surfaced rather than
replaced with an invented model list. Explicit `INFERENCE_MODEL_MAP` aliases
continue to expose the configured public names.

This is hosted inference support. It does not provide local weights or a
GRPO/LoRA training preset. Gateway message-size and rate limits still apply.
Offline tests cover configuration, authentication, URL paths, chat, streaming,
reasoning/tool history, multimodal payloads, structured outputs, and failures.
Live QwenCloud access requires your account and has not been certified by these
tests.

Sources: [Qwen4-Max Singapore model page](https://www.qwencloud.com/models/qwen4-max?region=ap-southeast-1),
[regional endpoints](https://docs.qwencloud.com/developer-guides/getting-started/regions),
[Qwen4 token-budget notes](https://docs.qwencloud.com/developer-guides/getting-started/latest-model),
[thinking controls](https://docs.qwencloud.com/developer-guides/text-generation/thinking).
