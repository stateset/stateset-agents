### StateSet RL Agents — Overview

StateSet RL Agents is a production‑grade framework for building, training, evaluating, and deploying reinforcement‑learning (RL) powered AI agents. It provides modular building blocks for multi‑turn agents, reward modeling (including LLM‑based and multi‑objective rewards), advanced RL algorithms (GRPO), orchestration, observability, and cloud‑native deployment.

## What the repository provides

- **Agent runtime and orchestration**
  - Multi‑turn agent abstractions and orchestration in `core/multiturn_agent.py`, `core/intelligent_orchestrator.py`, and `core/enhanced_state_management.py`
  - Enhanced agent capabilities in `core/enhanced/` (e.g., `enhanced_agent.py`, `advanced_rl_algorithms.py`)
  - Environments, trajectories, rewards, and types in `core/environment.py`, `core/trajectory.py`, `core/reward.py`, `core/types.py`, `core/type_system.py`

- **Training stack (RLHF, GRPO, TRL integration)**
  - GRPO training pipeline and TRL integration in `training/trl_grpo_trainer.py`, `training/train.py`, `training/neural_reward_trainer.py`, `training/distributed_trainer.py`
  - Practical guides and enhancements in `TRL_GRPO_TRAINING_GUIDE.md`, `GRPO_FRAMEWORK_ENHANCEMENTS_v4.md`

- **Reward modeling**
  - LLM‑based conversational reward shaping in `rewards/llm_reward.py`
  - Business‑aware multi‑objective rewards in `rewards/multi_objective_reward.py` (e.g., balance CSAT, AOV, margin, latency)

- **APIs and services**
  - Ready‑to‑serve agent endpoints in `api/enhanced_api_service.py`, `api/enhanced_grpo_gateway.py`, and `api/ultimate_grpo_service.py`

- **Observability, monitoring, and performance**
  - Advanced monitoring and profiling in `core/advanced_monitoring.py`, `utils/performance_monitor.py`, `utils/profiler.py`, `utils/wandb_integration.py`
  - Error handling, security, and reliability in `core/error_handling.py`, `utils/security.py`

- **Deployment and scaling**
  - Local dev and production deploy via `deployment/docker/`, `deployment/kubernetes/`
  - Cloud IaC in `deployment/cloud/aws/terraform/`, `deployment/cloud/gcp/terraform/`
  - Autoscaling and resilience via `deployment/kubernetes/hpa.yaml`, `deployment/kubernetes/deployment.yaml`

- **Examples, tests, and docs**
  - End‑to‑end examples in `examples/` (e.g., `customer_service_agent.py`, `enhanced_framework_demo.py`, `quick_start.py`)
  - Comprehensive tests in `tests/` including E2E, integration, and performance suites
  - Usage guides in `README.md`, `COMPREHENSIVE_USAGE_GUIDE.md`

## How this enables Intelligent Commerce agents

- **Customer service copilots that learn**
  - Use `core/multiturn_agent.py` plus `rewards/llm_reward.py` to train dialog agents that optimize first‑contact resolution, CSAT, and handling time.
  - Balance business KPIs with `rewards/multi_objective_reward.py` (e.g., CSAT vs. AOV/margin vs. SLA).

- **Sales and personalization assistants**
  - Train agents to make contextual cross‑sell/upsell suggestions using GRPO, optimizing long‑term conversion/LTV instead of single‑click metrics.

- **Dynamic pricing and promotion policies**
  - Encode pricing/promo actions in `core/environment.py` and optimize policies with `training/trl_grpo_trainer.py`, targeting revenue, margin, inventory age, fairness constraints.

- **Returns and post‑purchase automation**
  - Multi‑turn routing workflows that minimize cost‑to‑serve while maintaining satisfaction; add safety via `core/error_handling.py` and `utils/security.py`.

- **Inventory and fulfillment decisions**
  - RL policies for backorder handling, substitutions, and split shipments; orchestrate across tools via `core/intelligent_orchestrator.py`.

- **Trust, risk, and compliance‑aware flows**
  - Combine rules with learned policies; use multi‑objective rewards to manage fraud exposure while keeping false positives low.

- **Multimodal product understanding**
  - Leverage `core/multimodal_processing.py` to incorporate images and rich content for better recommendations and assistance.

- **Production‑readiness from day one**
  - Serve with `api/enhanced_api_service.py`, observe with `utils/observability.py` and `core/advanced_monitoring.py`, deploy with `deployment/kubernetes/` or Docker.

---

## How to use this repo (step‑by‑step)

### 1) Install

- **Basic install**
```bash
pip install stateset-agents
```

- **With extras**
```bash
# API serving (FastAPI + uvicorn)
pip install "stateset-agents[api]"

# Full dev setup from source
git clone https://github.com/stateset/stateset-agents
cd stateset-agents
pip install -e ".[dev,api,examples,trl]"
```

- **GPU PyTorch (example CUDA 12.1)**
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 2) Run the quick start

```bash
python examples/quick_start.py
```
What it does: creates a `MultiTurnAgent`, a `ConversationEnvironment`, composes `HelpfulnessReward` + `SafetyReward`, trains briefly via `training.train.train`, and then runs a test conversation.

### 3) Programmatic training (minimal)

```python
import asyncio
from stateset_agents import (
  MultiTurnAgent, ConversationEnvironment,
  HelpfulnessReward, SafetyReward, CompositeReward
)
from stateset_agents.core.agent import AgentConfig
# If running from source and you want direct imports:
from training.train import train

async def run():
  agent = MultiTurnAgent(AgentConfig(model_name="gpt2", temperature=0.7))
  await agent.initialize()

  env = ConversationEnvironment(
    scenarios=[{"id": "cs_demo", "user_responses": ["Hi", "Need refund", "Thanks!"]}],
    max_turns=6
  )

  reward = CompositeReward([HelpfulnessReward(0.7), SafetyReward(0.3)])

  trained = await train(agent=agent, environment=env, reward_fn=reward, num_episodes=50, profile="balanced")
  return trained

asyncio.run(run())
```

- Auto‑opt: use `AutoTrainer` to pick a profile and early‑stop
```python
from training.train import AutoTrainer
trained = await AutoTrainer(auto_adjust=True, early_stopping=True, patience=50).train(agent, env)
```

### 4) CLI usage

```bash
# Show version and environment
stateset-agents version

# Validate training environment (guidance only)
stateset-agents train --dry-run

# Evaluate scaffold (guidance only)
stateset-agents evaluate --dry-run

# Start the Ultimate GRPO Service API (FastAPI)
stateset-agents serve --host 0.0.0.0 --port 8001
```

### 5) Serve an API

- **Ultimate service (batteries‑included demo)**
```bash
stateset-agents serve --host 0.0.0.0 --port 8001
# Endpoints (selection):
#  - POST /api/train
#  - POST /api/chat
#  - GET  /api/metrics
#  - GET  /
```

- **Enhanced service (fine‑grained control)**
```bash
# From repo root
uvicorn api.enhanced_api_service:app --host 0.0.0.0 --port 8000 --reload
# Endpoints:
#  - GET  /
#  - GET  /health
#  - POST /agents
#  - POST /conversations
#  - POST /training
#  - GET  /training/{training_id}
#  - GET  /metrics
```

- **cURL example (Enhanced conversations)**
```bash
curl -X POST "http://localhost:8000/conversations" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <your-token>" \
  -d '{
        "messages": [
          {"role": "system", "content": "You are a helpful assistant."},
          {"role": "user", "content": "Hello! Can you help me?"}
        ],
        "max_tokens": 128,
        "temperature": 0.7
      }'
```

Note: The enhanced service includes an authentication hook using HTTP Bearer. Integrate your auth, or adapt for local dev.

### 6) Train with TRL GRPO script (production template)

```bash
# From repo root
bash scripts/train_trl_grpo.sh
```
Key environment variables you can override:

- **Model and data**
  - `MODEL_NAME` (default: `openai/gpt-oss-120b`)
  - `DATA_PATH` (default: `training_data.jsonl`)
- **Training scale**
  - `NUM_EPISODES`, `NUM_GENERATIONS`, `BATCH_SIZE`, `GRADIENT_ACCUMULATION_STEPS`
- **LoRA**
  - `USE_LORA`, `LORA_R`, `LORA_ALPHA`, `LORA_DROPOUT`
- **Precision/memory**
  - `USE_BF16`, `USE_FP16`, `GRADIENT_CHECKPOINTING`
- **Logging/output**
  - `WANDB_PROJECT`, `OUTPUT_DIR`, `SAVE_STEPS`, `EVAL_STEPS`
- **Smoke test**
  - `QUICK_MODE=true` to run a tiny configuration fast

### 7) Docker and Kubernetes

- **Docker Compose (local stack)**
```bash
docker compose -f deployment/docker/docker-compose.yml up -d
```

- **Kubernetes (example manifests)**
```bash
kubectl apply -f deployment/kubernetes/
```
Review `deployment/kubernetes/secret.example.yaml` and set your own secrets before production.

### 8) Run tests and benchmarks (from source)

```bash
# Dev install from step 1 required
pytest -q
python scripts/benchmark.py
```

---

## Commerce recipe: from prototype to production

1. **Define scenarios**
   - Create JSON of typical commerce dialogues (refunds, shipping status, returns, cross‑sell) and load into `ConversationEnvironment`.
2. **Compose rewards**
   - Start with `HelpfulnessReward` + `SafetyReward` and incrementally add domain metrics via `rewards/multi_objective_reward.py` (e.g., CSAT proxy, margin guardrails, SLA adherence).
3. **Train**
   - Use `training/train.py::train` for quick iteration; when scaling up, switch to `scripts/train_trl_grpo.sh`.
4. **Evaluate**
   - Add synthetic test cases; compute reward distributions and success criteria; iterate on weights.
5. **Serve**
   - Choose `stateset-agents serve` for the demo API or `uvicorn api.enhanced_api_service:app` for fine‑grained control and observability.
6. **Observe and harden**
   - Enable `utils/wandb_integration.py`, `core/advanced_monitoring.py`, and tighten `utils/security.py`.
7. **Scale**
   - Containerize via Docker, deploy on Kubernetes, enable HPA, and wire to your commerce backend.

---

## Key modules map

- **Core agent mechanics**: `core/agent.py`, `core/multiturn_agent.py`, `core/enhanced/`
- **Rewards**: `rewards/llm_reward.py`, `rewards/multi_objective_reward.py`
- **Training**: `training/trl_grpo_trainer.py`, `training/train.py`, `training/distributed_trainer.py`
- **APIs**: `api/enhanced_api_service.py`, `api/ultimate_grpo_service.py`
- **Observability**: `core/advanced_monitoring.py`, `utils/observability.py`, `utils/wandb_integration.py`
- **Deployment**: `deployment/docker/`, `deployment/kubernetes/`, `deployment/cloud/`

## Why StateSet RL Agents for commerce

- **Outcome‑centric**: Optimize toward CSAT, AOV, margin, LTV, and SLA—not just clicks.
- **Safe and observable**: Built‑in safety, monitoring, and performance tooling to operate confidently.
- **Modular and extensible**: Swap rewards, policies, and environments without re‑architecting.
- **Production‑ready**: APIs, tests, and deployment assets from prototype to scale.

Built‑in modules like `core/adaptive_learning_controller.py`, `core/performance_optimizer.py`, and `core/neural_architecture_search.py` help you continuously improve policies as your catalog, traffic, and economics evolve. 