# StateSet RL Agents Overview

This repo (`stateset-agents`) is a training + serving stack for multi-turn
conversational agents. The canonical Python package is `stateset_agents/`.

## What You Can Do

- Train agents with group-based policy optimization (GRPO/GSPO family) and other
  trainers in `stateset_agents/training/`.
- Compose reward functions (multi-objective, LLM-judge, domain templates) in
  `stateset_agents/rewards/`.
- Serve models through a lightweight FastAPI gateway (`stateset_agents/api/`)
  that proxies to a GPU model server (vLLM).
- Deploy training jobs and inference workloads on GKE using the Helm chart and
  Kubernetes manifests in `deployment/`.

## Quick Start (Local)

```bash
pip install -e ".[dev,api]"
stateset-agents serve --host 0.0.0.0 --port 8000
```

In another terminal:

```bash
curl http://127.0.0.1:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{"model":"moonshotai/Kimi-K2.5","max_tokens":64,"messages":[{"role":"user","content":"Hello"}]}'
```

## Kimi-K2.5 Fine-Tuning

Use the repo script:

```bash
python examples/finetune_kimi_k25_gspo.py \
  --model moonshotai/Kimi-K2.5 \
  --task customer_service \
  --use-lora \
  --use-vllm \
  --export-merged \
  --output-dir ./outputs/kimi_k25_gspo
```

## GKE Deployment (api.stateset.com Compatible)

Choose the correct guide:

- Standard (GPU node pools): `docs/KIMI_K25_GKE_STANDARD.md`
- Autopilot (GPUs via pod requests): `docs/KIMI_K25_GKE_AUTOPILOT.md`

Recommended architecture:

1. vLLM (GPU): serves `moonshotai/Kimi-K2.5` or a merged fine-tuned checkpoint.
2. StateSet Agents gateway (CPU): exposes `POST /v1/messages` and proxies to vLLM.
3. Ingress: routes `https://api.stateset.com/*` to the gateway service.

Helm is the easiest path:

```bash
helm upgrade --install stateset-agents deployment/helm/stateset-agents \
  --namespace stateset-agents
```

## Testing

```bash
python -m pytest -q
```

