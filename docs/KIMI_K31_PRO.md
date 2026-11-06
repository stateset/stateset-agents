# Kimi K3.1 Pro

StateSet supports Kimi K3.1 Pro inference through its OpenAI-compatible gateway,
with streamed responses, tool calls, and preserved assistant reasoning history.
The hosted API ID is `kimi-k3.1-pro`; the released checkpoint is
`moonshotai/Kimi-K3.1-Pro`. They are separate from Kimi K3.

## Hosted inference

Install the `api` extra, configure the gateway's existing authentication, and
set the backend:

```bash
pip install -e '.[api]'
export API_ENVIRONMENT=development
export API_REQUIRE_AUTH=true
# Set STATESET_API_KEY to your gateway key (at least 32 characters).
export API_KEYS="${STATESET_API_KEY}:user"
export INFERENCE_BACKEND=openai
export INFERENCE_BACKEND_URL=https://api.moonshot.ai
export INFERENCE_BACKEND_API_KEY="${MOONSHOT_API_KEY}"
export INFERENCE_DEFAULT_MODEL=kimi-k3.1-pro
export INFERENCE_MODEL_MAP='{"moonshotai/Kimi-K3.1-Pro":"kimi-k3.1-pro"}'
python -m stateset_agents.api.main
```

The backend URL is the origin, without `/v1`: the gateway appends the API paths.
`INFERENCE_BACKEND_API_KEY` authenticates outbound requests and is independent
of the gateway key used by clients. Hosted health checks use `/v1/models` by
default; `INFERENCE_HEALTH_PATH` can override it.

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "X-API-Key: ${STATESET_API_KEY}" \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "moonshotai/Kimi-K3.1-Pro",
    "messages": [{"role":"user","content":"Explain sparse attention briefly."}],
    "reasoning_effort": "high",
    "max_tokens": 1024,
    "stream": false
  }'
```

Kimi supports `reasoning_effort` values `low`, `high`, and `max`. If omitted,
the backend chooses its default. The gateway does not add a temperature when
the caller omits it. Set `stream: true` for OpenAI-style SSE responses.

Model-specific reasoning limits also apply to public aliases configured in
`INFERENCE_MODEL_MAP`. For non-streaming requests, upstream throttling remains
HTTP 429 with the provider's `Retry-After` header; backend credential failures
return HTTP 502. The gateway does not retry permanent request errors,
throttling, or errors carrying provider backoff instructions. Other transient
failures use the bounded `INFERENCE_MAX_RETRIES` policy.

For tool loops and multi-turn use, append the complete assistant message from
`choices[0].message` to the next request, including `reasoning_content` and
`tool_calls`. Use `/v1/chat/completions` to retain this OpenAI message format.
The gateway preserves these fields and counts reasoning history toward the
configured message-length limit. Its usual token, request-size, and message
count limits also apply; the model's context capacity does not override them.

## Checkpoint and training readiness

The checkpoint is pinned to `d9e99b0792a5e93b78ea9bea62a68167e36ddd9e`.
It declares `KimiK31ForConditionalGeneration` / `kimi_k31`, 93 language layers
with KDA and Gated DSA attention, MXFP4 routed experts, and a custom tokenizer
codec. Its index records 1,561,217,377,280 stored tensor bytes. This is not a
training-memory estimate. The model has its own Kimi K3.1 Pro license.

```bash
stateset-agents frontier-models --model kimi-k3.1-pro
stateset-agents frontier-sft --model kimi-k3.1-pro --dry-run
```

These commands expose the pinned configuration and audited attention targets.
Local SFT/GSPO is blocked pending a verified packed-weight training loader,
KDA/DSA backward kernels, and a reviewed training codec. Generic QLoRA and the
older K3 loader do not satisfy these requirements. Packaged remote workers
reject training before uploads or hardware allocation. Managed training services
retain their own model-access checks; no service is certified for K3.1 training.
No local GPU topology or full-size inference/training run is certified here.

For self-hosted inference, connect the same gateway to a compatible server
using the checkpoint ID. Follow the upstream vLLM or SGLang recipe for the
exact runtime and hardware requirements.

## Verification and sources

`tests/api/test_kimi_k31.py` exercises the authenticated HTTP gateway with a
mock backend, including tool history, reasoning, streaming, and limits. The
pinned config, source hashes, and index summary are retained under
`benchmark_results/training_contracts/20261106/kimi-k3.1-pro/`. Live provider
inference and local full-size training remain unverified.

- [Official model card](https://huggingface.co/moonshotai/Kimi-K3.1-Pro)
- [API quickstart](https://platform.kimi.ai/docs/guide/kimi-k3-1-pro-quickstart)
- [vLLM serving recipe](https://recipes.vllm.ai/moonshotai/Kimi-K3.1-Pro)
- [SGLang serving recipe](https://docs.sglang.io/cookbook/autoregressive/Moonshotai/Kimi-K3.1-Pro)
