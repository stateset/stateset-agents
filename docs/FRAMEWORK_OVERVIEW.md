# StateSet Agents Framework Overview

StateSet Agents is a production-oriented RL framework for training and serving
multi-turn conversational agents. It focuses on group-based policy optimization
(GRPO/GSPO family) and ships a FastAPI gateway that can proxy inference to a
GPU model server such as vLLM.

## Repository Layout (Canonical)

The canonical Python package is `stateset_agents/`. Some top-level folders
(`api/`, `rewards/`, etc.) exist as deprecated shims for source-tree
compatibility.

```
stateset-agents/
├── stateset_agents/                 # Canonical Python package
│   ├── core/                        # Agents, environments, trajectories
│   ├── training/                    # GRPO/GSPO trainers and training helpers
│   ├── rewards/                     # Reward functions (incl. Kimi-K2.5 helpers)
│   ├── utils/                       # Monitoring, logging, observability
│   └── api/                         # FastAPI gateway (OpenAI/Anthropic compatible)
├── deployment/                      # Docker, Helm, Kubernetes manifests
├── examples/                        # Runnable scripts / demos
├── scripts/                         # Utility scripts (publish, benchmarks, cluster tools)
├── tests/                           # Unit + integration tests
├── rust_core/                       # Optional Rust acceleration crate
└── grpo_agent_framework/            # Legacy runtime helpers (used via PYTHONPATH)
```

## Key Entrypoints

### Training

- `examples/finetune_kimi_k25_gspo.py`: GSPO fine-tuning for Kimi-K2.5 (LoRA),
  optional merged export for serving.
- `stateset_agents/training/`: trainer implementations (GRPO family, PPO, offline RL)
  and configuration helpers.

### Serving (Gateway)

The gateway is `stateset_agents/api/main.py` and exposes:

- `POST /v1/messages` (Anthropic-style)
- `POST /v1/chat/completions` + `GET /v1/models` (OpenAI-style)

The gateway is designed to proxy to a model backend:

```bash
export INFERENCE_BACKEND=vllm
export INFERENCE_BACKEND_URL=http://localhost:8001
export INFERENCE_DEFAULT_MODEL=moonshotai/Kimi-K2.5

stateset-agents serve --host 0.0.0.0 --port 8000
```

### GKE (Kimi-K2.5)

- Standard clusters (GPU node pools): `docs/KIMI_K25_GKE_STANDARD.md`
- Autopilot clusters (GPUs via pod requests): `docs/KIMI_K25_GKE_AUTOPILOT.md`
- Helm chart: `deployment/helm/stateset-agents`
  - GPU profiles: `deployment/helm/stateset-agents/values-a100.yaml`,
    `deployment/helm/stateset-agents/values-h100.yaml`,
    `deployment/helm/stateset-agents/values-b200.yaml`

## Development Commands

```bash
# Tests
python -m pytest -q

# Formatting / linting
ruff check . && black . && isort .

# Type checking
python scripts/check_types.py --all
```

## Notes

- GPU inference is expected to run in a dedicated model server (vLLM, etc). The
  gateway remains lightweight and focuses on API compatibility, auth, and
  observability.
- For container builds, see `deployment/docker/Dockerfile` (gateway) and
  `deployment/docker/Dockerfile.trainer` (trainer jobs).
