# Qwen3.5-27B Minimal Hosting Plan

This document defines the minimum viable path to host `Qwen/Qwen3.5-27B` on the
current StateSet Agents stack and expose an endpoint that accepts API calls.

It is intentionally narrow:

- One model.
- One `vLLM` replica.
- One internal gateway endpoint first.
- Minimum practical GPU shape for the current repo and serving stack.

## 1. Decision

Use direct `vLLM`, not Triton, for the first deployment.

Reasons:

- The repo already proxies the gateway to a `vLLM` OpenAI-compatible backend in
  [inference_service.py](../stateset_agents/api/services/inference_service.py).
- The Helm chart already starts `vllm.entrypoints.openai.api_server` in
  [vllm-deployment.yaml](../deployment/helm/stateset-agents/templates/vllm-deployment.yaml).
- Triton would add another serving layer without solving a current repo blocker.

For the minimum viable deployment, use the official quantized base model:

- Public model name at the gateway: `Qwen/Qwen3.5-27B`
- Internal serving model in `vLLM`: `Qwen/Qwen3.5-27B-GPTQ-Int4`

## 2. What "Minimal" Means Here

The current repo default for Qwen is still the large profile in
[values-qwen3-5-27b.yaml](../deployment/helm/stateset-agents/values-qwen3-5-27b.yaml):

- `8` GPUs
- `--tensor-parallel-size=8`
- `262144` max context

That is not the minimum path.

The minimum practical path for the current stack is:

- `GKE Standard`
- `g2-standard-24`
- `2x L4`
- `vLLM`
- `Qwen/Qwen3.5-27B-GPTQ-Int4`
- `--tensor-parallel-size=2`
- `--language-model-only`
- `--max-model-len=8192`

Why not `1x L4`:

- An L4 has `24 GB` of VRAM.
- The official `Qwen/Qwen3.5-27B-GPTQ-Int4` artifact is still larger than that
  in aggregate, and the current repo does not have a lower-memory `llama.cpp`
  or GGUF path.
- The current gateway and deployment flow are built around `vLLM`, not GGUF.

## 3. Architecture

The minimum architecture is:

1. `vLLM` serves the model behind an in-cluster `ClusterIP` Service.
2. The StateSet gateway exposes `POST /v1/messages`, `POST /v1/chat/completions`,
   and `GET /v1/models`.
3. The gateway rewrites the public model name
   `Qwen/Qwen3.5-27B` to the internal quantized model id
   `Qwen/Qwen3.5-27B-GPTQ-Int4` with `INFERENCE_MODEL_MAP`.
4. Initial client access is through `kubectl port-forward`.
5. Public ingress is a second step after the internal path is healthy.

Relevant repo files:

- Gateway proxy behavior:
  [inference_service.py](../stateset_agents/api/services/inference_service.py)
- OpenAI-compatible router:
  [openai.py](../stateset_agents/api/routers/openai.py)
- Anthropic-style router:
  [v1.py](../stateset_agents/api/routers/v1.py)
- Helm chart:
  [deployment/helm/stateset-agents/README.md](../deployment/helm/stateset-agents/README.md)
- Minimal override added for this plan:
  [values-qwen3-5-27b-minimal.yaml](../deployment/helm/stateset-agents/values-qwen3-5-27b-minimal.yaml)

## 4. Assumptions

This plan assumes:

- You are on `GKE Standard`, not Autopilot.
- You already have, or will keep, a small CPU node pool for the gateway pods.
  The current API Helm template still hardcodes `2` replicas in
  [api-deployment.yaml](../deployment/helm/stateset-agents/templates/api-deployment.yaml).
- You have a Hugging Face token with access to the Qwen model.
- You are deploying the base quantized model first.

This plan does not assume:

- Fine-tuned weights on day 1.
- Public internet exposure on day 1.
- Tool use on day 1.
- The full `262k` context window.

## 5. Phase 1: Internal Endpoint First

### 5.1 Create the namespace

```bash
kubectl create namespace stateset-agents --dry-run=client -o yaml | kubectl apply -f -
```

### 5.2 Create the GPU node pool

Create one `2x L4` node pool:

```bash
gcloud container node-pools create qwen35-l4-2g \
  --cluster="$CLUSTER_NAME" \
  --region="$REGION" \
  --machine-type="g2-standard-24" \
  --accelerator="type=nvidia-l4,count=2,gpu-driver-version=latest" \
  --num-nodes=1 \
  --enable-autoscaling --min-nodes=0 --max-nodes=1 \
  --enable-autorepair --enable-autoupgrade
```

Verify the node labels before deploying:

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,ACCEL:.metadata.labels.cloud\\.google\\.com/gke-accelerator
```

Expected result:

- One node with `cloud.google.com/gke-accelerator=nvidia-l4`

### 5.3 Create the Hugging Face secret

```bash
kubectl create secret generic hf-secret \
  --from-literal=hf_token="$HUGGING_FACE_HUB_TOKEN" \
  --namespace stateset-agents
```

### 5.4 Deploy with the minimal Helm profile

Use the new minimal values file:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  -f deployment/helm/stateset-agents/values-qwen3-5-27b-minimal.yaml
```

What this profile does:

- Loads `Qwen/Qwen3.5-27B-GPTQ-Int4` in `vLLM`
- Exposes `Qwen/Qwen3.5-27B` at the gateway
- Uses `2x L4`
- Forces text-only mode with `--language-model-only`
- Shrinks context to `8192`
- Disables gateway auth for the first internal smoke test

### 5.5 Wait for readiness

```bash
kubectl -n stateset-agents get pods
```

```bash
kubectl -n stateset-agents logs deploy/stateset-agents-vllm --tail=200
```

```bash
kubectl -n stateset-agents logs deploy/stateset-agents-api --tail=200
```

Readiness checks:

- `stateset-agents-vllm` becomes `Ready`
- `stateset-agents-api` becomes `Ready`
- `GET /health` on the `vLLM` pod succeeds
- `GET /ready` on the gateway succeeds

### 5.6 Expose the internal endpoint with port-forward

```bash
kubectl -n stateset-agents port-forward svc/stateset-agents-api 8000:8000
```

### 5.7 Smoke test the API

List models:

```bash
curl http://127.0.0.1:8000/v1/models
```

OpenAI-compatible request:

```bash
curl http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-27B",
    "messages": [
      {"role": "user", "content": "Reply with exactly: healthy"}
    ],
    "max_tokens": 32,
    "temperature": 0.1
  }'
```

Anthropic-style request:

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-27B",
    "messages": [
      {"role": "user", "content": "Reply with exactly: healthy"}
    ],
    "max_tokens": 32
  }'
```

Phase 1 is complete when:

- `GET /v1/models` returns `Qwen/Qwen3.5-27B`
- `POST /v1/chat/completions` returns a non-error response
- `POST /v1/messages` returns a non-error response
- The gateway never exposes the internal model id to clients

## 6. Phase 2: Stable Public Endpoint

Only do this after the internal path is stable.

### 6.1 Turn auth back on

For a public endpoint, do not keep `API_REQUIRE_AUTH=false`.

Create an API secret:

```bash
kubectl create secret generic stateset-agents-secrets \
  --namespace stateset-agents \
  --from-literal=API_JWT_SECRET="$API_JWT_SECRET" \
  --from-literal=API_KEYS="$STATESET_API_KEY"
```

Then upgrade the release with auth enabled and ingress on:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  -f deployment/helm/stateset-agents/values-qwen3-5-27b-minimal.yaml \
  --set api.env.API_REQUIRE_AUTH=true \
  --set ingress.enabled=true \
  --set ingress.host=api.stateset.example.com \
  --set ingress.tls.enabled=true \
  --set ingress.tls.secretName=stateset-qwen-tls
```

### 6.2 Public API call examples

OpenAI-compatible:

```bash
curl https://api.stateset.example.com/v1/chat/completions \
  -H "Authorization: Bearer $STATESET_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-27B",
    "messages": [
      {"role": "user", "content": "Say hello from Qwen."}
    ],
    "max_tokens": 64
  }'
```

Anthropic-style:

```bash
curl https://api.stateset.example.com/v1/messages \
  -H "Authorization: Bearer $STATESET_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3.5-27B",
    "messages": [
      {"role": "user", "content": "Say hello from Qwen."}
    ],
    "max_tokens": 64
  }'
```

Public endpoint acceptance criteria:

- TLS terminates at the ingress
- `Authorization: Bearer <key>` works
- `X-API-Key: <key>` also works
- `GET /v1/models` only returns the public model id

## 7. Phase 3: Promote a Fine-Tuned Model

This is where the minimum base-model path and the fine-tuned path diverge.

The current repo can already:

- fine-tune Qwen3.5-27B with
  [finetune_qwen3_5_27b_gspo.py](../examples/finetune_qwen3_5_27b_gspo.py)
- export a merged serving artifact with
  [serving_artifacts.py](../stateset_agents/training/serving_artifacts.py)
- map a public model name to a local merged path with
  [values-qwen3-5-27b-finetuned.yaml](../deployment/helm/stateset-agents/values-qwen3-5-27b-finetuned.yaml)

But the minimum `2x L4` profile should be treated as a base-model profile, not a
guaranteed fine-tuned serving profile.

Reason:

- The current export path writes merged model weights.
- There is not yet a repo-native post-merge quantization step for the fine-tuned
  Qwen3.5-27B path.
- A merged fine-tuned artifact will usually require a larger GPU shape than the
  quantized base-model profile defined here.

So Phase 3 needs one explicit decision:

1. Stay on the `2x L4` shape and add a quantized fine-tuned export path.
2. Keep the current merged fine-tuned export path and move the serving profile
   back up to a larger GPU shape.
3. Introduce controlled LoRA adapter serving and accept the extra operational
   complexity.

Recommended Phase 3 decision:

- Keep Phase 1 as the base-model endpoint.
- Add a quantized fine-tuned export path before trying to host finetuned Qwen
  on the minimum hardware shape.

## 8. Operational Notes

- Keep the first deployment text-only. The minimal profile uses
  `--language-model-only` on purpose.
- Keep the first deployment at `8192` max context. Increase only after the
  endpoint is healthy and memory headroom is measured.
- Do not optimize for tool calling first. Add
  `--enable-auto-tool-choice` and `--tool-call-parser=qwen3_coder` later.
- Keep one public model id. The gateway model mapping already gives you an alias
  boundary between client-visible ids and backend ids.

## 9. Definition of Done

This plan is complete when all of the following are true:

1. `stateset-agents-vllm` is serving `Qwen/Qwen3.5-27B-GPTQ-Int4` on `2x L4`.
2. `stateset-agents-api` exposes `Qwen/Qwen3.5-27B` through
   `GET /v1/models`.
3. Internal smoke tests pass through both `POST /v1/chat/completions` and
   `POST /v1/messages`.
4. A public ingress can be enabled without changing the backend topology.
5. The team has explicitly chosen how finetuned artifacts will be made to fit
   the minimum hardware profile.

## 10. External References

- Official quantized model:
  <https://huggingface.co/Qwen/Qwen3.5-27B-GPTQ-Int4>
- vLLM OpenAI-compatible server:
  <https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html>
- vLLM quantization hardware support:
  <https://docs.vllm.ai/en/v0.6.0/quantization/supported_hardware.html>
