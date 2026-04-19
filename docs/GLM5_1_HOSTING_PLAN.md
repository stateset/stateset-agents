# GLM 5.1 Hosting Plan

This document defines the minimum viable path to host
[`zai-org/GLM-5.1`](https://huggingface.co/zai-org/GLM-5.1) on the current
StateSet Agents stack and expose an endpoint that accepts API calls.

GLM 5.1 is a 754B-parameter Mixture-of-Experts reasoning model from Zhipu AI
(architecture `glm_moe_dsa`, DeepSeek V3-style MLA attention with 256 routed
experts and 8 active per token). Two checkpoints are published:

| Model | Repo | Notes |
| --- | --- | --- |
| GLM-5.1 | `zai-org/GLM-5.1` | BF16 weights, ~1.5 TB on disk |
| GLM-5.1-FP8 | `zai-org/GLM-5.1-FP8` | FP8 weights, ~754 GB on disk |

## 1. Decision

Use direct `vLLM`, not Triton, for the first deployment.

Reasons:

- The repo already proxies the gateway to a `vLLM` OpenAI-compatible backend in
  [inference_service.py](../stateset_agents/api/services/inference_service.py).
- The Helm chart already starts `vllm.entrypoints.openai.api_server` in
  [vllm-deployment.yaml](../deployment/helm/stateset-agents/templates/vllm-deployment.yaml).
- vLLM 0.19.0+ ships first-class support for the `glm_moe_dsa` architecture
  including FP8 weights.

For the minimum viable deployment, prefer the FP8 variant:

- Public model name at the gateway: `zai-org/GLM-5.1`
- Internal serving model in `vLLM`: `zai-org/GLM-5.1-FP8`

## 2. What "Minimal" Means Here

The two practical hosting shapes are:

| Shape | Variant | Topology | vLLM args |
| --- | --- | --- | --- |
| Single-host (FP8) | `zai-org/GLM-5.1-FP8` | 1x 8xH200 (1128 GB VRAM) | `--tensor-parallel-size=8 --quantization=fp8` |
| Multi-node (BF16) | `zai-org/GLM-5.1` | 2x 8xH100 / 8xH200 | `--tensor-parallel-size=8 --pipeline-parallel-size=2` |

The single-host FP8 path is the minimum practical setup with the current Helm
chart and is what `values-glm5-1-fp8.yaml` targets.

Why not 4x H100 80GB:

- A 4x H100 node has 320 GB of VRAM.
- The FP8 weights alone are ~754 GB and need additional headroom for KV cache.
- The current gateway and deployment flow are built around vLLM, not GGUF or
  llama.cpp, so further-quantized formats are not supported here yet.

## 3. Architecture

The minimum architecture is:

1. `vLLM` serves the model behind an in-cluster `ClusterIP` Service.
2. The StateSet gateway exposes `POST /v1/messages`, `POST /v1/chat/completions`,
   and `GET /v1/models`.
3. The gateway rewrites the public model name `zai-org/GLM-5.1` to the
   internal FP8 model id `zai-org/GLM-5.1-FP8` via `INFERENCE_MODEL_MAP`.
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
  [Chart.yaml](../deployment/helm/stateset-agents/Chart.yaml)
- vLLM deployment template:
  [vllm-deployment.yaml](../deployment/helm/stateset-agents/templates/vllm-deployment.yaml)

## 4. Deployment Recipes

### 4a. FP8 single-host (recommended start)

```bash
helm upgrade --install stateset-agents \
  ./deployment/helm/stateset-agents \
  -n stateset-agents --create-namespace \
  -f deployment/helm/stateset-agents/values-glm5-1-fp8.yaml
```

Or apply the raw manifest:

```bash
kubectl apply -f deployment/kubernetes/glm5-1-vllm-fp8.yaml
```

### 4b. BF16 multi-node base model

```bash
helm upgrade --install stateset-agents \
  ./deployment/helm/stateset-agents \
  -n stateset-agents --create-namespace \
  -f deployment/helm/stateset-agents/values-glm5-1.yaml
```

Or:

```bash
kubectl apply -f deployment/kubernetes/glm5-1-vllm.yaml
```

### 4c. Fine-tuned merged checkpoint

```bash
helm upgrade --install stateset-agents \
  ./deployment/helm/stateset-agents \
  -n stateset-agents --create-namespace \
  -f deployment/helm/stateset-agents/values-glm5-1-finetuned.yaml
```

The matching K8s manifests are in
`deployment/kubernetes/glm5-1-vllm-finetuned.yaml` and
`deployment/kubernetes/glm5-1-vllm-finetuned-gcs.yaml` (the GCS variant
syncs merged weights from a Cloud Storage bucket at pod startup).

## 5. Fine-tuning

GLM 5.1 fine-tuning is QLoRA-only on this stack — full fine-tuning of a 754B
MoE is not feasible. The dedicated starter is
[`examples/finetune_glm5_1_gspo.py`](../examples/finetune_glm5_1_gspo.py),
which wraps `stateset_agents.training.glm5_1_starter` and emits a
`serving_manifest.json` next to the merged adapter.

```bash
# Preview the resolved config
python examples/finetune_glm5_1_gspo.py --dry-run

# Memory-efficient starter profile
python examples/finetune_glm5_1_gspo.py --starter-profile memory --dry-run

# Save a reusable config
python examples/finetune_glm5_1_gspo.py --write-config ./glm5_1.json

# Real run on a multi-node cluster
python examples/finetune_glm5_1_gspo.py --no-dry-run --task customer_service
```

The Kubernetes Job manifest at
[`deployment/kubernetes/glm5-1-training-job.yaml`](../deployment/kubernetes/glm5-1-training-job.yaml)
runs this script under the StateSet trainer image with a PVC-backed
`/models/glm5-1` output directory.

## 6. Rendering Helm Values from a Run

After a fine-tuning run, you can render a Helm values override directly from
the emitted serving manifest:

```bash
python scripts/render_glm5_1_helm_values.py \
  --manifest /models/glm5-1/serving_manifest.json \
  --gcs-uri gs://stateset-models/glm5-1/runs/run-123/merged
```

Pipe the output into a values file:

```bash
python scripts/render_glm5_1_helm_values.py \
  --manifest /models/glm5-1/serving_manifest.json \
  > deployment/helm/stateset-agents/values-glm5-1-run-123.yaml
```

## 7. vLLM Serving Defaults

| Setting | Value | Notes |
| --- | --- | --- |
| `--tensor-parallel-size` | `8` | One node TP |
| `--pipeline-parallel-size` | `2` (BF16) / `1` (FP8) | BF16 needs cross-node PP |
| `--max-model-len` | `131072` | GLM 5.1 supports up to ~200K, this leaves KV headroom |
| `--reasoning-parser` | `glm45` | Inherits the GLM 4.5/4.6 reasoning parser |
| `--tool-call-parser` | `glm45` | Same for tool calls |
| `--gpu-memory-utilization` | `0.92` | Conservative for very-long-context KV cache |
| `--quantization` | `fp8` (FP8 variant only) | Use the prepacked FP8 weights |
| `--trust-remote-code` | `true` | `glm_moe_dsa` requires the model code |

These defaults are produced by
`stateset_agents.training.glm5_1_starter.get_glm5_1_serving_recommendations`
and consumed by both the example dry-run output and the Helm values renderer.

## 8. Verification Checklist

After deploying, confirm:

1. The vLLM pod reaches `Ready` (initial weight load can take 10–15 minutes).
2. `kubectl port-forward svc/glm5-1-vllm-fp8 8000:8000` and `curl
   localhost:8000/v1/models` returns `zai-org/GLM-5.1-FP8`.
3. The gateway proxies the public name through:
   ```bash
   curl -sX POST localhost:8001/v1/chat/completions \
     -H 'content-type: application/json' \
     -d '{"model":"zai-org/GLM-5.1","messages":[{"role":"user","content":"hi"}]}'
   ```
4. `INFERENCE_MODEL_MAP` is set in the api Deployment env so the gateway
   forwards the request to the FP8 backend.

## 9. Known Gotchas

- `transformers>=5.4.0` is required for the `glm_moe_dsa` architecture. The
  starter validator emits a warning if a lower version is installed.
- `peft` LoRA target modules use the MLA names (`q_a_proj`, `q_b_proj`,
  `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`) plus the standard SwiGLU FFN
  projections. Verify against the live model state dict before launching a
  long run.
- The FP8 single-host shape requires H200 (141 GB) or B200-class accelerators.
  H100 80GB is not enough for FP8 weights + KV cache.
- vLLM 0.19.0+ is required for the FP8 + `glm_moe_dsa` combination.

## 10. Related Files

- Training module: [glm5_1_starter.py](../stateset_agents/training/glm5_1_starter.py)
- Example script: [finetune_glm5_1_gspo.py](../examples/finetune_glm5_1_gspo.py)
- Helm values: [values-glm5-1.yaml](../deployment/helm/stateset-agents/values-glm5-1.yaml),
  [values-glm5-1-fp8.yaml](../deployment/helm/stateset-agents/values-glm5-1-fp8.yaml),
  [values-glm5-1-finetuned.yaml](../deployment/helm/stateset-agents/values-glm5-1-finetuned.yaml)
- K8s manifests: [glm5-1-vllm.yaml](../deployment/kubernetes/glm5-1-vllm.yaml),
  [glm5-1-vllm-fp8.yaml](../deployment/kubernetes/glm5-1-vllm-fp8.yaml),
  [glm5-1-vllm-finetuned.yaml](../deployment/kubernetes/glm5-1-vllm-finetuned.yaml),
  [glm5-1-vllm-finetuned-gcs.yaml](../deployment/kubernetes/glm5-1-vllm-finetuned-gcs.yaml),
  [glm5-1-training-job.yaml](../deployment/kubernetes/glm5-1-training-job.yaml)
- Helm renderer: [render_glm5_1_helm_values.py](../scripts/render_glm5_1_helm_values.py)
