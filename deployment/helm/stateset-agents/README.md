# Helm Chart: StateSet Agents

## Install

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents
```

## Build Images

Gateway image (FastAPI, `/v1/messages`):

```bash
docker build -f deployment/docker/Dockerfile \
  -t stateset/stateset-agents-api:0.11.2 \
  .
```

Trainer image (Kubernetes Jobs):

```bash
docker build -f deployment/docker/Dockerfile.trainer \
  -t stateset/stateset-agents-trainer:0.11.2 \
  .
```

## API Secrets (JWT/API Keys)

If deploying the gateway in production mode, you must provide `API_JWT_SECRET`.

Create the secret out-of-band:

```bash
kubectl create secret generic stateset-agents-secrets \
  --namespace stateset-agents \
  --from-literal=API_JWT_SECRET="$API_JWT_SECRET"
```

Or have Helm create it:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  --set secrets.api.create=true \
  --set secrets.api.jwtSecret="$API_JWT_SECRET"
```

## Hugging Face + W&B Secrets

vLLM (and training jobs) typically need a Hugging Face token:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  --set secrets.hf.create=true \
  --set secrets.hf.token="$HUGGING_FACE_HUB_TOKEN"
```

Optional W&B token (used by the training Job):

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  --set secrets.wandb.create=true \
  --set secrets.wandb.token="$WANDB_API_KEY"
```

## Enable Training Job

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  --set trainingJob.enabled=true
```

## Render Values From Training Output

If you ran `examples/finetune_kimi_k25_gspo.py` and have a `serving_manifest.json`,
you can render a Helm values override:

```bash
python scripts/render_kimi_k25_helm_values.py \
  --manifest /models/kimi-k25/serving_manifest.json \
  > values.kimi-k25.generated.yaml
```

Then:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  -f values.kimi-k25.generated.yaml
```

If you ran `examples/finetune_qwen3_5_27b_gspo.py`, use the Qwen renderer instead:

```bash
python scripts/render_qwen3_5_27b_helm_values.py \
  --manifest /models/qwen3-5-27b/serving_manifest.json \
  > values.qwen3-5-27b.generated.yaml
```

## Fine-Tuned Serving (Local Path)

If your merged weights are on the PVC at `/models/kimi-k25/merged`, enable
fine-tuned serving and the model name mapping with:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  -f deployment/helm/stateset-agents/values-kimi-k25-finetuned.yaml
```

For Qwen3.5-27B, use the dedicated Qwen profile first and then the finetuned
override once your merged weights are on the PVC:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  -f deployment/helm/stateset-agents/values-qwen3-5-27b.yaml \
  -f deployment/helm/stateset-agents/values-qwen3-5-27b-finetuned.yaml
```

If you want the minimum internal serving footprint for the base model first,
use the `2x L4` profile:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  -f deployment/helm/stateset-agents/values-qwen3-5-27b-minimal.yaml
```

The full runbook for that path is in:

```text
docs/QWEN3_5_27B_MINIMAL_HOSTING_PLAN.md
```

If you prefer raw Kubernetes manifests instead of Helm, the repo now includes:

```bash
kubectl apply -f deployment/kubernetes/qwen3-5-27b-vllm.yaml
kubectl apply -f deployment/kubernetes/qwen3-5-27b-vllm-finetuned.yaml
kubectl apply -f deployment/kubernetes/qwen3-5-27b-vllm-finetuned-gcs.yaml
kubectl apply -f deployment/kubernetes/qwen3-5-27b-training-job.yaml
```

## Fine-Tuned Serving (Sync From GCS)

If you store model artifacts in GCS, you can have vLLM sync the checkpoint at pod
startup via an initContainer.

1) Upload a run to GCS (example):

```bash
bash scripts/gke/publish_model_to_gcs.sh \
  --local-dir /models/kimi-k25 \
  --gcs-uri gs://YOUR_BUCKET/kimi-k25/runs/YOUR_RUN_ID
```

2) Render values from the training manifest and include a `--gcs-uri` pointing to
the `merged/` folder:

```bash
python scripts/render_kimi_k25_helm_values.py \
  --manifest /models/kimi-k25/serving_manifest.json \
  --gcs-uri gs://YOUR_BUCKET/kimi-k25/runs/YOUR_RUN_ID/merged \
  > values.kimi-k25.gcs.yaml
```

For Qwen3.5-27B, use:

```bash
python scripts/render_qwen3_5_27b_helm_values.py \
  --manifest /models/qwen3-5-27b/serving_manifest.json \
  --gcs-uri gs://YOUR_BUCKET/qwen3-5-27b/runs/YOUR_RUN_ID/merged \
  > values.qwen3-5-27b.gcs.yaml
```

3) Deploy:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  -f values.kimi-k25.gcs.yaml
```

Notes:
- For Workload Identity, configure a Kubernetes service account with a bound GCP
  service account that has `roles/storage.objectViewer` permissions on your bucket.
- For uploads, use a workload with `roles/storage.objectAdmin` (or objectCreator + viewer).
- When `vllm.modelSync.enabled=true`, the chart mounts the service account token
  so the Cloud SDK can authenticate.

## Smoke Test

After deploying, you can port-forward and test the gateway endpoints with:

```bash
bash scripts/gke/smoke_test_gateway.sh
```

## Use Digests

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  --set vllm.image.digest=sha256:YOUR_DIGEST
```

## Enable Ingress

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  --set ingress.enabled=true \
  --set ingress.host=api.stateset.com
```

## GPU Profiles

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

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  -f deployment/helm/stateset-agents/values-b200.yaml
```

## Reservations / Extra Node Selectors

If you use capacity reservations (or need extra node labels like GPU driver
version), set `vllm.nodeSelectorExtra` (and optionally `trainingJob.*`):

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents \
  --set-string vllm.nodeSelectorExtra.cloud\\.google\\.com/gke-gpu-driver-version=latest \
  --set-string vllm.nodeSelectorExtra.cloud\\.google\\.com/reservation-name="$RESERVATION_URL" \
  --set-string vllm.nodeSelectorExtra.cloud\\.google\\.com/reservation-affinity=specific
```
