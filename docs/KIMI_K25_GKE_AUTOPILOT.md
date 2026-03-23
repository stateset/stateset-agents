# Kimi-K2.5 on GKE Autopilot (Training + Serving)

This guide wires the **StateSet Agents** training pipeline to a vLLM-backed
inference service on **GKE Autopilot** using Kimi‑K2.5.

## 0. Preflight Checks

This repo includes a preflight helper that checks for GPU nodes, secrets, PVCs,
and expected workloads:

```bash
bash scripts/gke/kimi_k25_preflight.sh stateset-agents
```

## 1. Training (GSPO + vLLM)

Use the repo training script to fine‑tune Kimi‑K2.5 with GSPO:

```bash
python examples/finetune_kimi_k25_gspo.py \
  --model moonshotai/Kimi-K2.5 \
  --task customer_service \
  --use-lora \
  --use-vllm \
  --export-merged \
  --output-dir ./outputs/kimi_k25_gspo
```

The script writes `serving_manifest.json` into the output directory to simplify
deployment.

### Optional: Render Helm Values From Training Output

If you trained to a shared path (e.g. a PVC mounted at `/models/kimi-k25`), you can
render a Helm values override from the manifest:

```bash
python scripts/render_kimi_k25_helm_values.py \
  --manifest /models/kimi-k25/serving_manifest.json \
  > values.kimi-k25.generated.yaml
```

### Model Registry (GCS, Recommended)

For production, treat a versioned GCS bucket as the **source of truth** for trained
model artifacts. A typical layout is:

```
gs://<bucket>/kimi-k25/runs/<run_id>/
  serving_manifest.json
  merged/
```

Create a bucket (and enable versioning):

```bash
bash scripts/gke/create_model_bucket.sh \
  --bucket YOUR_BUCKET \
  --project YOUR_PROJECT \
  --location us-central1
```

Permissions:
- vLLM sync needs `roles/storage.objectViewer` on the bucket/prefix.
- Upload job needs `roles/storage.objectAdmin` (or objectCreator + viewer).
- Prefer Workload Identity so pods don't use node-level service account permissions.

Upload a trained run (from the PVC) using the included Job:

```bash
kubectl apply -f deployment/kubernetes/kimi-k25-upload-to-gcs-job.yaml
```

### GKE Job (optional)

See `deployment/kubernetes/kimi-k25-training-job.yaml` for a ready‑to‑run Job
spec that mounts a model PVC and requests 8x A100 GPUs on Autopilot.

#### Build Trainer Image

The Job references `stateset/stateset-agents-trainer:0.7.1`. Build it from this
repo using `deployment/docker/Dockerfile.trainer`:

```bash
docker build -f deployment/docker/Dockerfile.trainer \
  -t stateset/stateset-agents-trainer:0.7.1 \
  .
```

For GPU builds, pick a CUDA-enabled base image that already contains a
compatible CUDA build of PyTorch:

```bash
docker build -f deployment/docker/Dockerfile.trainer \
  --build-arg BASE_IMAGE=pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime \
  -t stateset/stateset-agents-trainer:0.7.1 \
  .
```

Create the PVC first:

```bash
kubectl apply -f deployment/kubernetes/stateset-models-pvc.yaml
```

## 2. Serving (vLLM nightly)

Use the vLLM Autopilot deployment:

```bash
kubectl apply -f deployment/kubernetes/kimi-k25-vllm.yaml
```

This deploys a vLLM OpenAI‑compatible server with:
- Tensor parallel size 8
- 256k max context
- Chunked prefill + prefix caching
- `--trust-remote-code` enabled
- `--mm-encoder-tp-mode=data` for multimodal tensor-parallel vision encoding
- `--tool-call-parser=kimi_k2` + `--reasoning-parser=kimi_k2` for tool calling and thinking parsing

For strict pinning, use a digest in the image field:

```
image: vllm/vllm-openai@sha256:YOUR_DIGEST
```

The deployment expects a `hf-secret` with key `hf_token`:

```bash
kubectl create secret generic hf-secret \
  --from-literal=hf_token=$HUGGING_FACE_HUB_TOKEN \
  --namespace stateset-agents
```

Optional: W&B secret for training telemetry:

```bash
kubectl create secret generic wandb-secret \
  --from-literal=api_key=$WANDB_API_KEY \
  --namespace stateset-agents
```

### Serving Fine‑Tuned Checkpoints

If you trained with the PVC output path (`/models/kimi-k25`), deploy the
fine‑tuned vLLM service:

```bash
kubectl apply -f deployment/kubernetes/kimi-k25-vllm-finetuned.yaml
```

This assumes you exported merged weights to `/models/kimi-k25/merged`.

If you want vLLM to sync the fine-tuned weights from GCS at startup, use:

```bash
kubectl apply -f deployment/kubernetes/kimi-k25-vllm-finetuned-gcs.yaml
```

## 3. API Gateway (/v1/messages)

The StateSet Agents API now exposes:

```
POST /v1/messages
POST /v1/chat/completions
```

`/v1/messages` accepts **Anthropic‑style** requests and can optionally return
**OpenAI‑style** responses by setting `response_format: "openai"`.

`/v1/chat/completions` is a minimal **OpenAI‑compatible** endpoint for clients
that expect the OpenAI schema.

Set the inference backend environment variables:

```
INFERENCE_BACKEND=vllm
INFERENCE_BACKEND_URL=http://kimi-k25-vllm.stateset-agents.svc.cluster.local:8000
INFERENCE_DEFAULT_MODEL=moonshotai/Kimi-K2.5
INFERENCE_HEALTH_PATH=/health
INFERENCE_STREAM_INCLUDE_USAGE=true
```

For fine‑tuned serving, point to:

```
INFERENCE_BACKEND_URL=http://kimi-k25-vllm-finetuned.stateset-agents.svc.cluster.local:8000
```

If your vLLM model is loaded from a local path (fine-tuned PVC or GCS sync),
you must map the public model name to the internal identifier:

```
INFERENCE_MODEL_MAP={"moonshotai/Kimi-K2.5":"/models/kimi-k25/merged"}
```

These are included in `deployment/kubernetes/production-deployment.yaml`.

## 4. Example Request

```bash
curl https://api.stateset.com/v1/messages \
  -H "Authorization: Bearer $STATESET_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "moonshotai/Kimi-K2.5",
    "max_tokens": 512,
    "messages": [
      {"role": "user", "content": "Explain the ReAct pattern in agentic AI."}
    ]
  }'
```

Authentication accepts API keys (`Authorization: Bearer ...` or `X-API-Key`) or
JWTs (if configured).

### Smoke Test (Port-Forward)

If you want a quick in-cluster smoke test without setting up ingress, use:

```bash
bash scripts/gke/smoke_test_gateway.sh
```

## 5. Notes

- Autopilot **does not allow custom GPU node pools**. GPUs are provisioned by
  requesting them in Pod specs.
- Standard GKE **does allow** GPU node pools. If your nodes have
  `cloud.google.com/gke-provisioning=standard` and no `cloud.google.com/gke-autopilot`
  label, you are on Standard mode and can add GPU capacity via node pools or Node
  Auto Provisioning.
- Update `nodeSelector` in the Kubernetes manifests if you are using a
  different GPU class than A100.
- For production, use a capacity reservation and pin the vLLM image tag.

## 6. Helm (optional)

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents
```

To enable Ingress:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  --set ingress.enabled=true \
  --set ingress.host=api.stateset.com
```

GPU profile examples:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  -f deployment/helm/stateset-agents/values-a100.yaml
```

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  -f deployment/helm/stateset-agents/values-h100.yaml
```
