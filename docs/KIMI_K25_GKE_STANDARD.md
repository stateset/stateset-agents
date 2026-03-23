# Kimi-K2.5 on GKE Standard (GPU Node Pools + vLLM + StateSet Gateway)

This guide is for **GKE Standard** clusters, where you manage **GPU node pools**.

If you are on **Autopilot**, use `docs/KIMI_K25_GKE_AUTOPILOT.md` instead.

## 0. Confirm Cluster Mode

Autopilot nodes typically have a `cloud.google.com/gke-autopilot` label.
Standard nodes typically have `cloud.google.com/gke-provisioning=standard`.

```bash
kubectl get nodes -o custom-columns=NAME:.metadata.name,PROVISIONING:.metadata.labels.cloud\\.google\\.com/gke-provisioning,AUTOPILOT:.metadata.labels.cloud\\.google\\.com/gke-autopilot,ACCEL:.metadata.labels.cloud\\.google\\.com/gke-accelerator
```

### Optional: Preflight Checks

This repo includes a preflight helper that checks for GPU nodes, secrets, PVCs,
and expected workloads:

```bash
bash scripts/gke/kimi_k25_preflight.sh stateset-agents
```

## 1. Create A GPU Node Pool (Standard Only)

You need GPU quota and (for scarce GPUs) often a **capacity reservation**.

Set variables:

```bash
export CLUSTER_NAME=YOUR_CLUSTER
export REGION=us-central1
```

### Helper Script

This repo includes a thin helper that prints and runs a `gcloud container node-pools create`
for common 8x GPU profiles:

```bash
bash scripts/gke/create_gpu_nodepool.sh --cluster "$CLUSTER_NAME" --region "$REGION" --profile h100
```

### Example: 8x A100 (A2)

```bash
gcloud container node-pools create a100-8g \
  --cluster="$CLUSTER_NAME" \
  --region="$REGION" \
  --machine-type="a2-highgpu-8g" \
  --accelerator="type=nvidia-tesla-a100,count=8,gpu-driver-version=latest" \
  --num-nodes=1 \
  --enable-autoscaling --min-nodes=0 --max-nodes=3 \
  --enable-autorepair --enable-autoupgrade
```

### Example: 8x H100 (A3)

```bash
gcloud container node-pools create h100-8g \
  --cluster="$CLUSTER_NAME" \
  --region="$REGION" \
  --machine-type="a3-highgpu-8g" \
  --accelerator="type=nvidia-h100-80gb,count=8,gpu-driver-version=latest" \
  --num-nodes=1 \
  --enable-autoscaling --min-nodes=0 --max-nodes=3 \
  --enable-autorepair --enable-autoupgrade
```

Notes:
- If the accelerator type or machine type differs in your project/region, list
  what is available with `gcloud compute accelerator-types list` and choose the
  closest match.
- If you use a reservation, you typically add a reservation selector (labels)
  to your workloads. See your reservation docs, then update node selectors in
  the vLLM manifests or Helm values accordingly.

Verify GPU nodes exist:

```bash
kubectl get nodes -l cloud.google.com/gke-accelerator
```

## 2. Secrets (Hugging Face + Optional W&B)

Create secrets in the `stateset-agents` namespace:

```bash
kubectl create namespace stateset-agents --dry-run=client -o yaml | kubectl apply -f -
```

```bash
kubectl create secret generic hf-secret \
  --from-literal=hf_token=$HUGGING_FACE_HUB_TOKEN \
  --namespace stateset-agents
```

Optional:

```bash
kubectl create secret generic wandb-secret \
  --from-literal=api_key=$WANDB_API_KEY \
  --namespace stateset-agents
```

## 3. Storage For Fine-Tuned Weights (PVC)

```bash
kubectl apply -f deployment/kubernetes/stateset-models-pvc.yaml
```

## 4. (Optional) Run The Training Job

Build and push the trainer image (`deployment/docker/Dockerfile.trainer`) to your
registry, then update `deployment/kubernetes/kimi-k25-training-job.yaml` to use it.

Run:

```bash
kubectl apply -f deployment/kubernetes/kimi-k25-training-job.yaml
```

The training script writes a `serving_manifest.json` to the output directory. You can
turn that into a Helm values override (to deploy the fine-tuned checkpoint) with:

```bash
python scripts/render_kimi_k25_helm_values.py \
  --manifest /models/kimi-k25/serving_manifest.json \
  > values.kimi-k25.generated.yaml
```

## 4.1 Model Registry (GCS, Recommended)

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

Or from a machine with `gcloud`:

```bash
bash scripts/gke/publish_model_to_gcs.sh \
  --local-dir /models/kimi-k25 \
  --gcs-uri gs://YOUR_BUCKET/kimi-k25/runs/YOUR_RUN_ID \
  --project YOUR_PROJECT
```

## 5. Deploy vLLM (Base Or Fine-Tuned)

Base model:

```bash
kubectl apply -f deployment/kubernetes/kimi-k25-vllm.yaml
```

Fine-tuned checkpoint (PVC path `/models/kimi-k25/merged`):

```bash
kubectl apply -f deployment/kubernetes/kimi-k25-vllm-finetuned.yaml
```

Fine-tuned checkpoint synced from GCS at startup:

```bash
kubectl apply -f deployment/kubernetes/kimi-k25-vllm-finetuned-gcs.yaml
```

These manifests include vLLM's recommended Kimi-K2.5 flags:
`--mm-encoder-tp-mode=data`, `--tool-call-parser=kimi_k2`, and
`--reasoning-parser=kimi_k2`.

## 6. Deploy The StateSet Gateway (api.stateset.com Compatible)

Deploy the gateway and point it at the vLLM service:

```bash
kubectl apply -f deployment/kubernetes/production-deployment.yaml
```

The gateway exposes:
- `POST /v1/messages` (Anthropic-style)
- `POST /v1/chat/completions` and `GET /v1/models` (OpenAI-style)

If your vLLM model is loaded from a local path (fine-tuned PVC or GCS sync),
you must map the public model name to the internal identifier:

```bash
export INFERENCE_MODEL_MAP='{"moonshotai/Kimi-K2.5":"/models/kimi-k25/merged"}'
```

If you want accurate token usage in streaming responses (when the backend supports it),
set:

```bash
export INFERENCE_STREAM_INCLUDE_USAGE=true
```

## 7. Smoke Test

```bash
kubectl -n stateset-agents port-forward svc/stateset-agents-api 8000:80
```

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-K2.5","max_tokens":64,"messages":[{"role":"user","content":"Hello"}]}'
```

Or use the helper script:

```bash
bash scripts/gke/smoke_test_gateway.sh
```
