# StateSet Agents - Technical Architecture

**Version:** 0.5.0
**Last Updated:** December 2024

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Core Components](#core-components)
- [GRPO Algorithm Implementation](#grpo-algorithm-implementation)
- [Training Pipeline](#training-pipeline)
- [Data Flow](#data-flow)
- [API Architecture](#api-architecture)
- [Deployment Architecture](#deployment-architecture)
- [Design Patterns](#design-patterns)
- [Performance Optimization](#performance-optimization)
- [Extension Points](#extension-points)

---

## Overview

StateSet Agents is a production-ready reinforcement learning framework for training multi-turn conversational AI agents using **Group Relative Policy Optimization (GRPO)**. The framework is built with a modular, async-first architecture designed for scale and extensibility.

### Key Statistics

- **~50,000 lines** of production Python code
- **98% test coverage** on core components
- **5+ RL algorithms** (GRPO, PPO, DPO, A2C, TRPO)
- **10+ pre-built reward functions**
- **Async-first** design for high concurrency
- **Production deployments** on Kubernetes with auto-scaling

### Design Philosophy

1. **Computation-First**: Leverage massive computation over hand-crafted rules (inspired by the "Bitter Lesson")
2. **Multi-Turn Native**: Built specifically for conversational AI, not retrofitted
3. **Production-Ready**: Enterprise features from day one
4. **Modular**: Easy to extend and customize
5. **Type-Safe**: Runtime validation and safe serialization

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      StateSet Agents Framework                   │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│   Core Layer     │    │  Training Layer  │    │   API Layer      │
│                  │    │                  │    │                  │
│  - Agent         │───▶│  - Trainer       │───▶│  - FastAPI       │
│  - Environment   │    │  - Config        │    │  - WebSocket     │
│  - Trajectory    │    │  - Diagnostics   │    │  - Monitoring    │
│  - Reward        │    │  - Distributed   │    │  - Auth          │
│  - Value Fn      │    │  - TRL Bridge    │    │  - Rate Limit    │
│  - Comp Engine   │    │                  │    │                  │
└──────────────────┘    └──────────────────┘    └──────────────────┘
         │                       │                       │
         └───────────────────────┴───────────────────────┘
                                 │
                   ┌─────────────────────────┐
                   │   Utilities Layer       │
                   │                         │
                   │  - Monitoring           │
                   │  - Logging              │
                   │  - Caching              │
                   │  - Performance          │
                   │  - Security             │
                   └─────────────────────────┘
```

### Directory Structure

```
stateset-agents/
├── core/                          # Core RL abstractions (~13K lines)
│   ├── agent.py                   # Agent orchestration (900+ lines)
│   ├── environment.py             # Training environments
│   ├── trajectory.py              # Conversation data structures
│   ├── reward.py                  # Reward modeling (1000+ lines)
│   ├── value_function.py          # GAE and value estimation
│   ├── computational_engine.py    # Parallel trajectory generation
│   ├── multiturn_agent.py         # Advanced dialogue management
│   ├── performance_optimizer.py   # Hardware optimization
│   ├── agent_backends.py          # Stub backends for fast demos
│   └── enhanced/                  # Advanced algorithms
│       ├── advanced_rl_algorithms.py
│       └── advanced_evaluation.py
│
├── training/                      # Training infrastructure
│   ├── trainer.py                 # Main GRPO trainer (1500+ lines)
│   ├── trl_grpo_trainer.py        # TRL integration (800+ lines)
│   ├── config.py                  # Training configurations
│   ├── train.py                   # High-level training interface
│   ├── distributed_trainer.py     # Multi-GPU training
│   └── neural_reward_trainer.py   # Learned reward models
│
├── api/                           # REST API services
│   ├── ultimate_grpo_service.py   # Complete FastAPI service (1000+ lines)
│   ├── enhanced_api_service.py
│   └── enhanced_grpo_gateway.py
│
├── rewards/                       # Multi-objective reward system
│   ├── llm_reward.py              # LLM-based rewards
│   ├── ruler_reward.py            # Rule-based rewards
│   └── multi_objective_reward.py  # Compositional rewards
│
├── utils/                         # Production utilities
│   ├── monitoring.py              # Real-time metrics
│   ├── wandb_integration.py       # W&B integration
│   ├── logging.py                 # Structured logging
│   ├── cache.py                   # Caching service
│   ├── alerts.py                  # Alert system
│   ├── performance_monitor.py     # Performance tracking
│   └── security.py                # Security utilities
│
├── deployment/                    # Production deployment
│   ├── kubernetes/                # K8s manifests
│   ├── docker/                    # Docker configurations
│   ├── monitoring/                # Grafana dashboards
│   └── cloud/                     # Cloud scripts
│
├── examples/                      # 13+ complete examples
│   ├── quick_start.py
│   ├── complete_grpo_training.py
│   ├── train_with_trl_grpo.py
│   └── customer_service_agent.py
│
└── tests/                         # Comprehensive test suite
    ├── unit/
    ├── integration/
    ├── e2e/
    └── performance/
```

---

## Core Components

### 1. Agent System (`core/agent.py`)

**Purpose**: Orchestrate LLM-based agents with conversation capabilities.

#### Agent Hierarchy

```python
Agent (Abstract Base)
├── MultiTurnAgent      # Multi-turn conversations with state
└── ToolAgent          # Agents with external tool capabilities
```

#### Key Classes

```python
@dataclass
class AgentConfig:
    """Configuration for agents"""
    model_name: str                    # HuggingFace model or "stub://name"
    max_length: int = 512              # Max generation length
    temperature: float = 0.7           # Sampling temperature
    use_stub_model: bool = False       # Enable stub mode
    stub_responses: List[str] = None   # Custom stub responses
    enable_lora: bool = False          # LoRA fine-tuning
    lora_config: Optional[dict] = None # LoRA parameters
    enable_reasoning: bool = False     # DeepSeek-R1 style reasoning

class Agent(ABC):
    """Abstract base class for all agents"""

    @abstractmethod
    async def initialize(self) -> None:
        """Initialize model and resources"""

    @abstractmethod
    async def generate_response(
        self,
        messages: List[Dict[str, str]],
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Generate response to messages"""

class MultiTurnAgent(Agent):
    """Agent specialized for multi-turn conversations"""

    async def process_turn(
        self,
        history: List[ConversationTurn],
        user_input: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """Process a single conversation turn"""
```

#### Agent Backends

- **Transformers**: Standard HuggingFace models
- **Stub**: Fast mock responses for development/CI
- **LoRA**: Parameter-efficient fine-tuning
- **Custom**: Extensible backend system

#### Usage Example

```python
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

# Production agent
agent = MultiTurnAgent(AgentConfig(
    model_name="meta-llama/Llama-2-7b-chat-hf",
    enable_lora=True,
    lora_config={"r": 16, "lora_alpha": 32}
))

# Stub agent for testing
stub_agent = MultiTurnAgent(AgentConfig(
    model_name="stub://test",
    use_stub_model=True,
    stub_responses=["Hello!", "How can I help?"]
))
```

---

### 2. Environment System (`core/environment.py`)

**Purpose**: Manage training scenarios and episode execution.

#### Environment Hierarchy

```python
Environment (Abstract Base)
├── ConversationEnvironment  # Open-ended dialogues
└── TaskEnvironment         # Goal-oriented tasks
```

#### Key Classes

```python
@dataclass
class EnvironmentState:
    """Current state of an environment episode"""
    episode_id: str
    turn_count: int
    is_done: bool
    context: Dict[str, Any]

class Environment(ABC):
    """Abstract base for training environments"""

    @abstractmethod
    async def reset(self) -> EnvironmentState:
        """Start new episode"""

    @abstractmethod
    async def step(
        self,
        state: EnvironmentState,
        agent_response: str
    ) -> Tuple[str, bool]:
        """Execute one step, return (user_response, done)"""

class ConversationEnvironment(Environment):
    """Environment for conversational training"""

    def __init__(
        self,
        scenarios: List[Dict[str, Any]],
        max_turns: int = 10,
        reward_fn: Optional[RewardFunction] = None
    ):
        self.scenarios = scenarios
        self.max_turns = max_turns
        self.reward_fn = reward_fn
```

#### Environment Features

- **Scenario Management**: Load from JSON, Python dicts, or generators
- **Turn Limiting**: Automatic episode termination
- **Context Passing**: Rich context for agents and rewards
- **Async Execution**: Non-blocking episode management

---

### 3. Trajectory System (`core/trajectory.py`)

**Purpose**: Represent conversation data structures.

#### Data Structures

```python
@dataclass
class ConversationTurn:
    """Single turn in a conversation"""
    speaker: str                        # "user" or "agent"
    content: str                        # Message content
    timestamp: float                    # Unix timestamp
    context: Optional[Dict[str, Any]]   # Additional context

    # Legacy compatibility
    role: str = field(init=False)       # Maps to speaker
    message: str = field(init=False)    # Maps to content

@dataclass
class MultiTurnTrajectory:
    """Complete conversation trajectory"""
    trajectory_id: str
    turns: List[ConversationTurn]
    rewards: List[float]                # Per-turn rewards
    total_reward: float
    metadata: Dict[str, Any]

    def to_messages(self) -> List[Dict[str, str]]:
        """Convert to HuggingFace chat format"""

@dataclass
class TrajectoryGroup:
    """Batch of trajectories for training"""
    trajectories: List[MultiTurnTrajectory]
    group_metadata: Dict[str, Any]

    def compute_group_statistics(self) -> Dict[str, float]:
        """Compute mean, std, etc."""
```

#### Serialization

- **Type-safe**: Runtime validation with TypeSafeSerializer
- **Backward compatible**: Supports legacy formats
- **Efficient**: Optimized for large trajectory batches

---

### 4. Reward System (`core/reward.py`)

**Purpose**: Evaluate agent responses and conversations.

#### Reward Architecture

```python
class RewardFunction(ABC):
    """Abstract base for reward computation"""

    @abstractmethod
    async def compute_reward(
        self,
        turns: List[ConversationTurn],
        context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward asynchronously"""

@dataclass
class RewardResult:
    """Result of reward computation"""
    score: float                        # Main reward score
    breakdown: Dict[str, float]         # Component breakdown
    metadata: Dict[str, Any]            # Additional info

class CompositeReward(RewardFunction):
    """Weighted combination of multiple rewards"""

    def __init__(self, rewards: List[Tuple[RewardFunction, float]]):
        self.rewards = rewards  # [(reward_fn, weight), ...]
```

#### Pre-built Reward Functions

| Reward | Purpose | Weight Range |
|--------|---------|--------------|
| `HelpfulnessReward` | Agent provides useful information | 0.0 - 1.0 |
| `SafetyReward` | Avoids harmful/biased content | 0.0 - 1.0 |
| `CorrectnessReward` | Factually accurate responses | 0.0 - 1.0 |
| `ConcisenessReward` | Appropriate response length | 0.0 - 1.0 |
| `EngagementReward` | Maintains conversation flow | 0.0 - 1.0 |
| `TaskCompletionReward` | Achieves conversation goals | 0.0 - 1.0 |
| `CustomerServiceReward` | Domain-specific: support quality | 0.0 - 1.0 |
| `TechnicalSupportReward` | Domain-specific: technical accuracy | 0.0 - 1.0 |
| `SalesAssistantReward` | Domain-specific: sales effectiveness | 0.0 - 1.0 |

#### Factory Functions

```python
def create_customer_service_reward() -> CompositeReward:
    """Pre-configured reward for customer service"""
    return CompositeReward([
        (HelpfulnessReward(), 0.35),
        (SafetyReward(), 0.25),
        (EngagementReward(), 0.20),
        (ConcisenessReward(), 0.20)
    ])

def create_domain_reward(domain: str) -> CompositeReward:
    """Create domain-specific reward"""
```

---

### 5. Value Function (`core/value_function.py`)

**Purpose**: Compute value estimates for advantage calculation.

#### Architecture

```python
class ValueHead(nn.Module):
    """Neural network for value estimation"""

    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.dense2 = nn.Linear(hidden_size, 1)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Predict state values"""

class ValueFunction:
    """Manages value predictions and GAE"""

    def __init__(
        self,
        model: nn.Module,
        gamma: float = 0.99,
        gae_lambda: float = 0.95
    ):
        self.value_head = ValueHead(model.config.hidden_size)
        self.gamma = gamma
        self.gae_lambda = gae_lambda

    def compute_gae(
        self,
        rewards: List[float],
        values: torch.Tensor,
        dones: List[bool]
    ) -> torch.Tensor:
        """Generalized Advantage Estimation"""

    def compute_grpo_advantages(
        self,
        group_rewards: List[float],
        baseline_type: str = "group_mean"
    ) -> List[float]:
        """Group-relative advantage computation"""
```

#### GAE Formula

```
δ_t = r_t + γ * V(s_{t+1}) - V(s_t)    [TD error]
A_t = δ_t + γλ * A_{t+1}                [GAE]
```

**Parameters:**
- `γ` (gamma): Discount factor (typically 0.99)
- `λ` (lambda): GAE parameter (typically 0.95)

---

### 6. Computational Engine (`core/computational_engine.py`)

**Purpose**: Parallel trajectory generation and policy updates.

#### Key Features

- **Parallel Processing**: Multi-worker trajectory generation
- **Trajectory Buffering**: Efficient batch processing
- **Metrics Tracking**: Real-time training statistics
- **Reward Integration**: Supports raw and learned rewards

#### Architecture

```python
class ComputationalGRPOEngine:
    """Core GRPO engine with parallel processing"""

    def __init__(
        self,
        agent: Agent,
        environment: Environment,
        reward_fn: Optional[RewardFunction] = None,
        num_workers: int = None,  # Defaults to CPU count
        buffer_size: int = 100
    ):
        self.agent = agent
        self.environment = environment
        self.reward_fn = reward_fn
        self.executor = ThreadPoolExecutor(max_workers=num_workers)

    async def generate_trajectory_batch(
        self,
        batch_size: int
    ) -> List[MultiTurnTrajectory]:
        """Generate trajectories in parallel"""

    async def compute_policy_update(
        self,
        trajectories: List[MultiTurnTrajectory]
    ) -> Dict[str, float]:
        """Compute policy gradient update"""
```

---

## GRPO Algorithm Implementation

### Algorithm Overview

**GRPO (Group Relative Policy Optimization)** is a policy gradient method that computes advantages relative to a group of sampled trajectories, providing more stable training than single-trajectory methods.

### Mathematical Formulation

#### 1. Policy Loss

```
L_policy = -E[A(s,a) * log π(a|s)]
```

Where:
- `A(s,a)`: Advantage function
- `π(a|s)`: Policy (agent's action probability)

#### 2. Group-Relative Advantage

```
A_i = R_i - baseline

baseline = mean(R_1, R_2, ..., R_n)  or  median(...)
```

Where:
- `R_i`: Total reward for trajectory i
- `baseline`: Group statistic (mean or median)

#### 3. Enhanced GRPO with KL Regularization

```
L_total = L_policy + β * KL[π || π_ref]
```

Where:
- `β`: KL penalty coefficient (typically 0.01 - 0.1)
- `π_ref`: Reference model (frozen)
- `KL`: KL divergence

#### 4. PPO-Style Clipping (Optional)

```
L_clip = max(A * loss, clip(A, -ε, +ε) * loss)
```

Where:
- `ε`: Clip ratio (typically 0.2)

### Implementation in `training/trainer.py`

```python
class MultiTurnGRPOTrainer:
    """Complete GRPO training implementation"""

    async def _compute_group_policy_loss(
        self,
        group: TrajectoryGroup,
        advantages: List[float]
    ) -> torch.Tensor:
        """
        Compute policy loss for trajectory group

        Steps:
        1. Tokenize all conversations
        2. Forward pass through model
        3. Compute NLL (negative log likelihood)
        4. Weight by advantages
        5. Apply PPO clipping if enabled
        6. Return mean loss
        """
        losses = []

        for traj, advantage in zip(group.trajectories, advantages):
            # Convert to model inputs
            messages = traj.to_messages()
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            ).to(self.device)

            # Forward pass
            with autocast(enabled=self.config.bf16):
                outputs = self.model(
                    **inputs,
                    labels=inputs["input_ids"]
                )

            # Policy loss = advantage * NLL
            loss = advantage * outputs.loss

            # Optional: PPO clipping
            if self.config.clip_ratio:
                clipped_advantage = torch.clamp(
                    advantage,
                    1.0 - self.config.clip_ratio,
                    1.0 + self.config.clip_ratio
                )
                clipped_loss = clipped_advantage * outputs.loss
                loss = torch.max(loss, clipped_loss)

            losses.append(loss)

        return torch.stack(losses).mean()

    async def _compute_kl_penalty(
        self,
        trajectories: List[MultiTurnTrajectory]
    ) -> torch.Tensor:
        """Compute KL divergence from reference model"""
        if not self.config.use_reference_model:
            return torch.tensor(0.0)

        kl_divs = []
        for traj in trajectories:
            messages = traj.to_messages()
            inputs = self.tokenizer.apply_chat_template(
                messages,
                return_tensors="pt"
            ).to(self.device)

            # Current policy logits
            with torch.no_grad():
                current_logits = self.model(**inputs).logits

            # Reference policy logits
            with torch.no_grad():
                ref_logits = self.reference_model(**inputs).logits

            # KL divergence
            kl = F.kl_div(
                F.log_softmax(current_logits, dim=-1),
                F.softmax(ref_logits, dim=-1),
                reduction="batchmean"
            )
            kl_divs.append(kl)

        return torch.stack(kl_divs).mean()
```

---

## Training Pipeline

### Training Flow

```
┌─────────────────────────────────────────────────────────────┐
│                     GRPO Training Loop                       │
└─────────────────────────────────────────────────────────────┘

1. Initialize
   ├── Load agent and environment
   ├── Create reward function
   ├── Initialize value function
   └── Setup optimizer and scheduler

2. Episode Loop
   ├── Generate Trajectories
   │   ├── Sample N scenarios
   │   ├── Generate M trajectories per scenario (parallel)
   │   ├── Compute rewards (async)
   │   └── Group into trajectory groups
   │
   ├── Compute Advantages
   │   ├── Get value estimates from value function
   │   ├── Compute GAE or group-relative advantages
   │   └── Normalize advantages (optional)
   │
   ├── Policy Update
   │   ├── Compute policy loss
   │   ├── Compute KL penalty (if enabled)
   │   ├── Total loss = policy_loss + β * kl_penalty
   │   ├── Backward pass
   │   ├── Gradient clipping
   │   └── Optimizer step
   │
   ├── Value Function Update
   │   ├── Compute value loss (MSE)
   │   ├── Backward pass
   │   └── Value optimizer step
   │
   ├── Evaluation
   │   ├── Run evaluation scenarios
   │   ├── Compute metrics
   │   └── Log to W&B
   │
   └── Checkpointing
       ├── Save model weights
       ├── Save optimizer state
       └── Save training config

3. Finalize
   └── Return trained agent
```

### Configuration System

```python
from stateset_agents.training.config import TrainingConfig, TrainingProfile

# Option 1: Pre-defined profile
config = TrainingConfig.from_profile(TrainingProfile.BALANCED)

# Option 2: Custom configuration
config = TrainingConfig(
    # Basic training
    num_episodes=1000,
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,

    # GRPO-specific
    num_generations=16,          # Trajectories per scenario
    beta=0.1,                    # KL penalty
    use_reference_model=True,
    clip_ratio=0.2,              # PPO clipping
    advantage_normalization=True,
    baseline_type="group_mean",

    # Optimization
    bf16=True,
    max_grad_norm=1.0,
    warmup_ratio=0.1,
    lr_scheduler_type="cosine",

    # Evaluation
    eval_steps=100,
    save_steps=500,
    logging_steps=10,

    # Monitoring
    report_to="wandb",
    wandb_project="stateset-agents"
)
```

---

## Data Flow

### Training Data Flow

```
Scenarios → Environment → Agent → Trajectories → Rewards → Training
     ↓                                ↑
     └────────────────────────────────┘
              Episode Loop
```

### Detailed Flow

1. **Scenario Selection**: Environment samples from scenario pool
2. **Episode Initialization**: Environment.reset() returns initial state
3. **Agent Generation**: Agent generates responses to user messages
4. **Turn Execution**: Environment.step() simulates user responses
5. **Trajectory Collection**: Turns are collected into MultiTurnTrajectory
6. **Reward Computation**: RewardFunction.compute_reward() evaluates trajectory
7. **Advantage Calculation**: ValueFunction computes advantages
8. **Policy Update**: Trainer applies gradients to agent model
9. **Iteration**: Repeat for num_episodes

---

## API Architecture

### FastAPI Service Structure

```python
# api/ultimate_grpo_service.py

from fastapi import FastAPI, WebSocket
from stateset_agents import MultiTurnAgent

app = FastAPI(title="StateSet Agents API")

# Health check
@app.get("/health")
async def health_check():
    return {"status": "healthy"}

# Generate response
@app.post("/v1/chat/completions")
async def chat_completion(request: ChatRequest):
    response = await agent.generate_response(request.messages)
    return {"response": response}

# WebSocket streaming
@app.websocket("/v1/chat/stream")
async def chat_stream(websocket: WebSocket):
    await websocket.accept()
    async for chunk in agent.stream_response(messages):
        await websocket.send_json({"chunk": chunk})

# Training endpoint
@app.post("/v1/train")
async def train_agent(config: TrainingConfig):
    trained_agent = await trainer.train()
    return {"checkpoint": checkpoint_path}
```

### API Features

- **REST Endpoints**: Standard HTTP API
- **WebSocket**: Real-time streaming
- **Rate Limiting**: Protect against abuse
- **Authentication**: API key or JWT
- **Monitoring**: Prometheus metrics
- **Health Checks**: Liveness and readiness probes

---

## Deployment Architecture

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: stateset-agents
spec:
  replicas: 3
  selector:
    matchLabels:
      app: stateset-agents
  template:
    spec:
      containers:
      - name: api
        image: stateset/agents:latest
        resources:
          requests:
            memory: "4Gi"
            cpu: "2"
          limits:
            memory: "8Gi"
            cpu: "4"
            nvidia.com/gpu: "1"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
```

### Horizontal Pod Autoscaler

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: stateset-agents-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: stateset-agents
  minReplicas: 3
  maxReplicas: 20
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

---

## Design Patterns

### 1. Factory Pattern

```python
# Create agents
def create_agent(config: AgentConfig) -> Agent:
    if config.use_stub_model:
        return create_stub_agent(config)
    return MultiTurnAgent(config)

# Create rewards
def create_customer_service_reward() -> CompositeReward:
    return CompositeReward([...])
```

### 2. Strategy Pattern

Multiple trainer implementations:
- `SingleTurnGRPOTrainer`
- `MultiTurnGRPOTrainer`
- `TRLGRPOTrainer`

### 3. Composition Pattern

```python
# Compose rewards
reward = CompositeReward([
    (HelpfulnessReward(), 0.4),
    (SafetyReward(), 0.3),
    (CustomReward(), 0.3)
])
```

### 4. Observer Pattern

```python
# Training callbacks
trainer.register_callback("on_episode_end", log_metrics)
```

### 5. Circuit Breaker Pattern

```python
# In utils/monitoring.py
@circuit_breaker(failure_threshold=5, timeout=60)
async def call_external_api():
    ...
```

---

## Performance Optimization

### Memory Optimization

1. **Gradient Checkpointing**: Trade compute for memory
2. **Mixed Precision (BF16)**: Reduce memory footprint
3. **Gradient Accumulation**: Simulate larger batches
4. **LoRA**: Reduce trainable parameters

### Computational Efficiency

1. **Parallel Trajectory Generation**: Multi-worker processing
2. **Async I/O**: Non-blocking operations
3. **Batch Processing**: Efficient tensor operations
4. **TRL Integration**: Optimized GRPO implementation

### Configuration for 8GB VRAM

```python
config = TrainingConfig(
    bf16=True,                      # Half precision
    gradient_checkpointing=True,    # Memory trade-off
    per_device_train_batch_size=1,  # Minimum batch size
    gradient_accumulation_steps=8,  # Simulate batch=8
    enable_lora=True,               # Reduce parameters
    lora_config={"r": 16}
)
```

---

## Extension Points

### Custom Agents

```python
class DomainSpecificAgent(MultiTurnAgent):
    def __init__(self, config: AgentConfig, domain_knowledge: Dict):
        super().__init__(config)
        self.knowledge = domain_knowledge

    async def process_turn(self, history, user_input, context):
        # Inject domain knowledge
        enhanced_context = self.enhance_with_knowledge(context)
        return await super().process_turn(history, user_input, enhanced_context)
```

### Custom Environments

```python
class SimulatedEnvironment(Environment):
    def __init__(self, simulator):
        self.simulator = simulator

    async def step(self, state, action):
        result = self.simulator.process(state, action)
        return self.convert_to_trajectory(result)
```

### Custom Rewards

```python
class BusinessMetricReward(RewardFunction):
    async def compute_reward(self, turns, context=None):
        # Evaluate against business KPIs
        score = calculate_business_value(turns, context)
        return RewardResult(
            score=score,
            breakdown={"conversion": 0.8, "satisfaction": 0.9}
        )
```

---

## Technical Specifications

### Dependencies

**Core (always required):**
- Python 3.8+
- numpy >= 1.21.0
- pydantic >= 2.0.0
- rich >= 13.0.0
- typing-extensions >= 4.0.0

**Training (optional):**
- torch >= 2.0.0
- transformers >= 4.30.0
- accelerate >= 0.20.0
- trl >= 0.7.0
- peft >= 0.4.0

**API (optional):**
- fastapi >= 0.110.0
- uvicorn >= 0.23.0

### System Requirements

**Minimum:**
- 16 GB RAM
- 4 CPU cores
- 8 GB GPU VRAM (for training)

**Recommended:**
- 32+ GB RAM
- 8+ CPU cores
- 24+ GB GPU VRAM (A100/H100)
- NVMe SSD storage

---

## References

- **GRPO**: Group Relative Policy Optimization for Reinforcement Learning
- **PPO**: Schulman et al. "Proximal Policy Optimization Algorithms"
- **GAE**: Schulman et al. "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
- **TRL**: https://github.com/huggingface/trl

---

**Version**: 0.5.0
**Last Updated**: December 2024
**Status**: Production Ready
