# Offline RL and Sim-to-Real Transfer Guide

This guide covers StateSet Agents' offline reinforcement learning and sim-to-real transfer capabilities for training conversational AI agents from logged data and simulation.

## Table of Contents

1. [Overview](#overview)
2. [Offline RL Algorithms](#offline-rl-algorithms)
3. [Data Utilities](#data-utilities)
4. [Sim-to-Real Transfer](#sim-to-real-transfer)
5. [Domain Randomization](#domain-randomization)
6. [Conversation Simulation](#conversation-simulation)
7. [Evaluation Metrics](#evaluation-metrics)
8. [Complete Examples](#complete-examples)
9. [Best Practices](#best-practices)

---

## Overview

### When to Use Offline RL

Offline RL enables learning from historical conversation logs without live interaction:

- **Existing datasets**: Train from customer service transcripts, chat logs, or support tickets
- **Safety-critical domains**: Healthcare, finance, or legal where online exploration is risky
- **Bootstrapping**: Pre-train before online fine-tuning to accelerate learning
- **Regulatory compliance**: Train without collecting new user data

### When to Use Sim-to-Real

Sim-to-real transfer bridges the gap between simulated and real conversations:

- **Cost reduction**: Reduce expensive human annotation by training in simulation
- **Safe exploration**: Let agents make mistakes in simulation, not with real users
- **Scalability**: Generate unlimited training data via simulation
- **Iterative development**: Rapid prototyping before real-world deployment

### Architecture Overview

```
                    OFFLINE RL PIPELINE
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   Historical Logs    ┌─────────────┐    ┌────────────────┐  │
│   ─────────────────▶ │ Conversation │───▶│ Offline RL     │  │
│   (JSONL/HF)         │ Dataset      │    │ (BCQ/BEAR/DT)  │  │
│                      └─────────────┘    └────────┬───────┘  │
│                                                  │          │
│                                                  ▼          │
│                                         ┌────────────────┐  │
│                                         │ Pre-trained    │  │
│                                         │ Policy/Value   │  │
│                                         └────────┬───────┘  │
│                                                  │          │
└──────────────────────────────────────────────────┼──────────┘
                                                   │
                    SIM-TO-REAL PIPELINE           ▼
┌──────────────────────────────────────────────────────────────┐
│                                                              │
│   ┌─────────────┐    ┌─────────────┐    ┌────────────────┐  │
│   │ Domain      │───▶│ Conversation │───▶│ Progressive    │  │
│   │ Randomizer  │    │ Simulator    │    │ Transfer       │  │
│   └─────────────┘    └─────────────┘    └────────┬───────┘  │
│                                                  │          │
│   ┌─────────────────────────────────────────────┐│          │
│   │ Gap Metrics: KL / JS / MMD divergence       ││          │
│   └─────────────────────────────────────────────┘│          │
│                                                  ▼          │
│                                         ┌────────────────┐  │
│                                         │ Production     │  │
│                                         │ Agent          │  │
│                                         └────────────────┘  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```

---

## Offline RL Algorithms

StateSet Agents provides five offline RL algorithms optimized for conversational agents:

### Algorithm Comparison

| Algorithm | Type | Best For | Conservatism | Complexity |
|-----------|------|----------|--------------|------------|
| **BCQ** | Actor-Critic | VAE-constrained actions | High | Medium |
| **BEAR** | Actor-Critic | Distribution matching | High | High |
| **CQL** | Q-Learning | Pessimistic values | Medium | Low |
| **IQL** | Q-Learning | Expectile regression | Low | Low |
| **Decision Transformer** | Sequence Model | Return-conditioned | N/A | Medium |

### BCQ (Batch-Constrained Q-Learning)

BCQ constrains the policy to stay close to the behavior policy using a VAE:

```python
from stateset_agents.training import BCQConfig, BCQTrainer, ConversationalVAE

config = BCQConfig(
    # Network architecture
    hidden_dim=256,
    latent_dim=64,
    num_layers=3,

    # VAE parameters
    vae_lr=1e-4,
    vae_epochs=50,
    kl_weight=0.1,

    # Q-network parameters
    q_lr=3e-4,
    discount=0.99,
    tau=0.005,  # Target network update rate

    # BCQ-specific
    threshold=0.3,  # Action filtering threshold
    num_samples=10,  # Actions to sample from VAE
)

trainer = BCQTrainer(config)

# Train VAE first to learn action distribution
await trainer.train_vae(dataset, epochs=50)

# Train Q-networks with VAE constraint
metrics = await trainer.train(dataset, epochs=100)
```

**Key insight**: BCQ only considers actions that the VAE assigns high likelihood, preventing out-of-distribution actions.

### BEAR (Bootstrapping Error Accumulation Reduction)

BEAR uses Maximum Mean Discrepancy (MMD) to match the learned policy to the data distribution:

```python
from stateset_agents.training import BEARConfig, BEARTrainer, MMDKernel

config = BEARConfig(
    # Network architecture
    hidden_dim=256,
    num_layers=3,

    # MMD parameters
    mmd_kernel="gaussian",  # or "laplacian"
    mmd_sigma=20.0,
    num_mmd_samples=5,

    # Lagrange multiplier
    lagrange_threshold=0.05,
    lagrange_lr=1e-3,

    # Training
    actor_lr=1e-4,
    critic_lr=3e-4,
    discount=0.99,
)

trainer = BEARTrainer(config)
metrics = await trainer.train(dataset, epochs=100)
```

**Key insight**: BEAR adaptively adjusts the MMD constraint via dual gradient descent.

### CQL (Conservative Q-Learning)

CQL adds a penalty to prevent overestimation of out-of-distribution actions:

```python
from stateset_agents.training import CQLConfig, ConservativeQLearning

config = CQLConfig(
    hidden_dim=256,
    num_layers=3,

    # CQL penalty
    cql_alpha=1.0,  # Conservative penalty weight
    cql_temperature=1.0,
    num_random_actions=10,

    # Training
    learning_rate=3e-4,
    discount=0.99,
)

cql = ConservativeQLearning(config)
metrics = await cql.train(dataset)
```

### IQL (Implicit Q-Learning)

IQL avoids querying out-of-distribution actions entirely via expectile regression:

```python
from stateset_agents.training import IQLConfig, ImplicitQLearning

config = IQLConfig(
    hidden_dim=256,
    num_layers=3,

    # IQL-specific
    expectile=0.7,  # Expectile for value function
    temperature=3.0,  # Advantage weighting temperature

    # Training
    learning_rate=3e-4,
    discount=0.99,
)

iql = ImplicitQLearning(config)
metrics = await iql.train(dataset)
```

**Key insight**: IQL never evaluates Q-values on unseen actions during training.

### Decision Transformer

A sequence modeling approach that conditions on desired returns:

```python
from stateset_agents.training import (
    DecisionTransformerConfig,
    DecisionTransformer,
    DecisionTransformerTrainer,
)

config = DecisionTransformerConfig(
    # Transformer architecture
    hidden_dim=256,
    num_layers=4,
    num_heads=4,
    context_length=20,  # Number of turns to consider

    # Embedding
    state_dim=768,  # Conversation embedding dimension
    action_dim=768,

    # Training
    learning_rate=1e-4,
    dropout=0.1,
)

trainer = DecisionTransformerTrainer(config)
metrics = await trainer.train(dataset, epochs=100)

# Generate with target return
model = DecisionTransformer(config)
action = model.generate(
    states=conversation_embeddings,
    actions=previous_responses,
    returns_to_go=[0.9, 0.85, 0.8],  # Desired quality trajectory
    timesteps=[0, 1, 2],
)
```

**Key insight**: Decision Transformer treats RL as sequence modeling, conditioning on desired outcomes.

---

## Data Utilities

### ConversationDataset

Load and manage offline conversation data:

```python
from stateset_agents.data import (
    ConversationDataset,
    ConversationDatasetConfig,
)

# Configuration
config = ConversationDatasetConfig(
    # Quality filtering
    quality_threshold=0.7,      # Minimum conversation quality
    min_turns=3,                # Minimum conversation length
    max_turns=50,               # Maximum conversation length

    # Processing
    normalize_rewards=True,
    compute_returns=True,
    discount=0.99,
)

# Load from JSONL
dataset = ConversationDataset.from_jsonl(
    "conversations.jsonl",
    config=config,
)

# Load from HuggingFace
dataset = ConversationDataset.from_huggingface(
    "customer_service_conversations",
    split="train",
    config=config,
)

# Filter by quality
high_quality = dataset.filter_by_quality(threshold=0.8)

# Get statistics
stats = dataset.get_statistics()
print(f"Total conversations: {stats['total_conversations']}")
print(f"Mean reward: {stats['mean_reward']:.3f}")
print(f"Mean turns: {stats['mean_turns']:.1f}")
```

### JSONL Format

Expected format for conversation data:

```json
{"conversation_id": "conv_001", "turns": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi! How can I help?"}], "reward": 0.85, "metadata": {"topic": "greeting"}}
{"conversation_id": "conv_002", "turns": [{"role": "user", "content": "I need a refund"}, {"role": "assistant", "content": "I'd be happy to help with your refund request."}], "reward": 0.92, "metadata": {"topic": "refunds"}}
```

### ConversationReplayBuffer

Priority-based sampling for training:

```python
from stateset_agents.data import ConversationReplayBuffer

buffer = ConversationReplayBuffer(
    capacity=100000,
    priority_alpha=0.6,    # Priority exponent
    priority_beta=0.4,     # Importance sampling
)

# Add conversations
for conv in conversations:
    buffer.add(conv, priority=conv.reward)

# Sample batch
batch = buffer.sample(batch_size=32, mode="trajectory")

# Or sample individual turns
turn_batch = buffer.sample(batch_size=64, mode="turn")
```

### EmbeddingCache

Cache conversation embeddings for efficiency:

```python
from stateset_agents.data import EmbeddingCache

cache = EmbeddingCache(
    cache_dir="./embedding_cache",
    embedding_dim=768,
)

# Get or compute embedding
embedding = await cache.get_or_compute(
    text="Hello, how can I help you?",
    compute_fn=lambda t: encoder.encode(t),
)

# Batch processing
embeddings = await cache.batch_get_or_compute(texts, compute_fn)
```

---

## Sim-to-Real Transfer

### SimToRealTransfer

Orchestrate the transition from simulation to real interactions:

```python
from stateset_agents.training import SimToRealTransfer, SimToRealConfig

config = SimToRealConfig(
    # Transfer schedule
    transfer_schedule="cosine",  # linear, exponential, step, cosine
    warmup_steps=100,
    total_steps=1000,
    initial_sim_ratio=0.9,
    final_sim_ratio=0.1,

    # Domain adaptation
    adaptation_method="dann",  # dann, mmd, coral
    adaptation_weight=0.1,

    # Monitoring
    gap_threshold=0.2,  # Alert if sim-real gap exceeds this
)

transfer = SimToRealTransfer(config)

# During training loop
for step in range(1000):
    sim_ratio = transfer.get_sim_ratio(step)

    # Mix simulated and real data
    if random.random() < sim_ratio:
        batch = sim_data.sample()
    else:
        batch = real_data.sample()

    # Train with domain adaptation
    loss = transfer.compute_adapted_loss(model, batch)
```

### Transfer Schedules

Four built-in schedules for sim-to-real ratio:

```python
# Linear: steady decrease
# sim_ratio = initial - (initial - final) * (step / total)

# Exponential: rapid initial decrease
# sim_ratio = initial * (final / initial) ** (step / total)

# Cosine: smooth S-curve
# sim_ratio = final + (initial - final) * (1 + cos(pi * step / total)) / 2

# Step: discrete jumps
# sim_ratio = initial if step < threshold else final
```

### Domain Adaptation Methods

#### DANN (Domain-Adversarial Neural Network)

```python
from stateset_agents.training import DomainAdaptationModule

adapter = DomainAdaptationModule(
    method="dann",
    feature_dim=768,
    hidden_dim=256,
)

# During training
features = encoder(batch)
domain_loss = adapter.compute_domain_loss(
    features,
    domain_labels,  # 0=sim, 1=real
)
total_loss = task_loss + 0.1 * domain_loss
```

#### MMD (Maximum Mean Discrepancy)

```python
adapter = DomainAdaptationModule(
    method="mmd",
    feature_dim=768,
    kernel="gaussian",
    kernel_bandwidth=1.0,
)

mmd_loss = adapter.compute_domain_loss(sim_features, real_features)
```

#### CORAL (Correlation Alignment)

```python
adapter = DomainAdaptationModule(
    method="coral",
    feature_dim=768,
)

coral_loss = adapter.compute_domain_loss(sim_features, real_features)
```

---

## Domain Randomization

Generate diverse training scenarios to improve generalization:

### DomainRandomizer

```python
from stateset_agents.training import (
    DomainRandomizer,
    DomainRandomizationConfig,
    UserPersona,
)

config = DomainRandomizationConfig(
    # Variation levels (0-1)
    persona_variation=0.3,
    topic_variation=0.2,
    style_variation=0.2,

    # Curriculum learning
    use_curriculum=True,
    initial_difficulty=0.3,
    final_difficulty=0.9,
    curriculum_steps=500,
)

randomizer = DomainRandomizer(config)
```

### User Personas

Pre-defined persona templates:

```python
from stateset_agents.training import PersonaGenerator

generator = PersonaGenerator()

# Built-in personas
personas = [
    "patient_expert",      # Knowledgeable, patient, clear communicator
    "frustrated_novice",   # Confused, impatient, needs hand-holding
    "busy_professional",   # Direct, time-constrained, expects efficiency
    "curious_learner",     # Asks follow-ups, wants to understand
    "skeptical_critic",    # Questions everything, hard to satisfy
]

# Sample random persona
persona = generator.sample()

# Sample with constraints
persona = generator.sample(
    patience_range=(0.3, 0.7),
    expertise_range=(0.5, 1.0),
)

# Create custom persona
custom = UserPersona(
    name="tech_savvy_senior",
    patience=0.8,
    expertise=0.6,
    verbosity=0.4,
    formality=0.7,
    traits=["methodical", "appreciative"],
)
```

### Scenario Generation

```python
from stateset_agents.training import ScenarioGenerator

generator = ScenarioGenerator()

# Generate scenario for topic
scenario = generator.generate(
    topic="returns",
    difficulty=0.7,
    persona=persona,
)

# Output:
# {
#     "id": "returns_scenario_001",
#     "topic": "returns",
#     "context": "Customer wants to return a damaged item...",
#     "user_messages": ["I received a broken product...", ...],
#     "expected_resolution": "Process return and offer replacement",
#     "difficulty": 0.7,
# }
```

### Curriculum Learning

Gradually increase difficulty during training:

```python
for step in range(1000):
    difficulty = randomizer.get_current_difficulty(step)
    scenario = generator.generate(
        topic=random.choice(topics),
        difficulty=difficulty,
    )
    # Train on progressively harder scenarios
```

---

## Conversation Simulation

### ConversationSimulator

Simulate user behavior for training:

```python
from stateset_agents.environments import (
    ConversationSimulator,
    ConversationSimulatorConfig,
    UserSimulator,
)

config = ConversationSimulatorConfig(
    # Base model for user simulation
    base_model="gpt2",

    # Realism settings
    realism_level=0.8,  # 0=fully scripted, 1=fully generative
    response_diversity=0.5,

    # Conversation parameters
    max_turns=20,
    turn_taking_delay=0.0,  # Simulated typing delay
)

simulator = ConversationSimulator(config)
await simulator.initialize()
```

### Calibration

Calibrate the simulator to match real conversation patterns:

```python
# Calibrate to real data
calibration_metrics = await simulator.calibrate(
    real_conversations=real_data,
    calibration_epochs=10,
)

print(f"KL divergence after calibration: {calibration_metrics['kl_div']:.4f}")
print(f"Length distribution match: {calibration_metrics['length_match']:.4f}")
```

### Sim-Real Gap Measurement

```python
# Generate simulated conversations
sim_conversations = []
for _ in range(100):
    conv = await simulator.generate_conversation(
        topic="customer_service",
        agent=your_agent,
    )
    sim_conversations.append(conv)

# Measure gap
gap_metrics = simulator.compute_sim_real_gap(
    real_data=real_conversations,
    sim_data=sim_conversations,
)

print(f"Response length gap: {gap_metrics['length_gap']:.3f}")
print(f"Vocabulary overlap: {gap_metrics['vocab_overlap']:.3f}")
print(f"Turn distribution KL: {gap_metrics['turn_kl']:.4f}")
```

### User Simulator

Low-level user simulation:

```python
user_sim = UserSimulator(
    persona=persona,
    topic="technical_support",
    style="conversational",
)

# Generate user response
user_response = await user_sim.respond(
    conversation_history=history,
    agent_message=agent_reply,
)
```

---

## Evaluation Metrics

### SimToRealEvaluator

Comprehensive evaluation of sim-to-real transfer:

```python
from stateset_agents.evaluation import SimToRealEvaluator, SimToRealMetrics

evaluator = SimToRealEvaluator()

# Compute comprehensive metrics
metrics: SimToRealMetrics = evaluator.evaluate(
    sim_conversations=sim_data,
    real_conversations=real_data,
    agent=trained_agent,
)

print(f"KL Divergence: {metrics.kl_divergence:.4f}")
print(f"JS Divergence: {metrics.js_divergence:.4f}")
print(f"MMD: {metrics.mmd:.4f}")
print(f"Transfer Gap: {metrics.transfer_gap:.4f}")
print(f"Policy Performance (sim): {metrics.sim_performance:.3f}")
print(f"Policy Performance (real): {metrics.real_performance:.3f}")
```

### Individual Metrics

```python
from stateset_agents.evaluation import (
    compute_kl_divergence,
    compute_js_divergence,
    compute_mmd,
    compute_distribution_divergence,
)

# Response length distributions
kl = compute_kl_divergence(sim_lengths, real_lengths)
js = compute_js_divergence(sim_lengths, real_lengths)

# Embedding-space divergence
mmd = compute_mmd(
    sim_embeddings,
    real_embeddings,
    kernel="gaussian",
    bandwidth=1.0,
)

# General distribution comparison
divergence = compute_distribution_divergence(
    sim_distribution,
    real_distribution,
    method="wasserstein",  # or "kl", "js", "mmd"
)
```

---

## Complete Examples

### Example 1: Train from Customer Service Logs

```python
import asyncio
from stateset_agents.data import ConversationDataset, ConversationDatasetConfig
from stateset_agents.training import (
    OfflineGRPOTrainer,
    OfflineGRPOConfig,
    OfflineRLAlgorithm,
)
from stateset_agents import MultiTurnAgent, ConversationEnvironment, HelpfulnessReward
from stateset_agents.core.agent import AgentConfig

async def train_from_logs():
    # 1. Load historical data
    dataset_config = ConversationDatasetConfig(
        quality_threshold=0.6,
        min_turns=2,
        normalize_rewards=True,
    )
    dataset = ConversationDataset.from_jsonl(
        "customer_service_logs.jsonl",
        config=dataset_config,
    )

    print(f"Loaded {len(dataset)} conversations")
    print(f"Statistics: {dataset.get_statistics()}")

    # 2. Set up agent and environment
    agent = MultiTurnAgent(AgentConfig(model_name="gpt2"))
    await agent.initialize()

    env = ConversationEnvironment(
        scenarios=[{"id": "cs", "topic": "support", "user_responses": ["Hi"]}],
        max_turns=10,
    )

    # 3. Configure hybrid offline + online training
    config = OfflineGRPOConfig(
        # Offline phase
        offline_algorithm=OfflineRLAlgorithm.CQL,
        offline_pretrain_steps=500,
        cql_alpha=1.0,

        # Online phase
        online_ratio=0.3,
        num_episodes=1000,

        # GRPO settings
        group_size=4,
        learning_rate=5e-6,
    )

    # 4. Train
    trainer = OfflineGRPOTrainer(config)
    trained_agent = await trainer.train(
        agent=agent,
        environment=env,
        reward_fn=HelpfulnessReward(),
        offline_dataset=dataset,
    )

    return trained_agent

asyncio.run(train_from_logs())
```

### Example 2: Sim-to-Real Pipeline

```python
import asyncio
from stateset_agents.environments import ConversationSimulator, ConversationSimulatorConfig
from stateset_agents.training import (
    SimToRealTransfer,
    SimToRealConfig,
    DomainRandomizer,
    DomainRandomizationConfig,
)
from stateset_agents.evaluation import SimToRealEvaluator
from stateset_agents import MultiTurnAgent, HelpfulnessReward
from stateset_agents.core.agent import AgentConfig

async def sim_to_real_training():
    # 1. Set up domain randomization
    rand_config = DomainRandomizationConfig(
        persona_variation=0.4,
        topic_variation=0.3,
        use_curriculum=True,
        curriculum_steps=500,
    )
    randomizer = DomainRandomizer(rand_config)

    # 2. Set up conversation simulator
    sim_config = ConversationSimulatorConfig(
        base_model="gpt2",
        realism_level=0.7,
    )
    simulator = ConversationSimulator(sim_config)
    await simulator.initialize()

    # Optional: calibrate to real data
    # await simulator.calibrate(real_conversations)

    # 3. Set up transfer orchestration
    transfer_config = SimToRealConfig(
        transfer_schedule="cosine",
        warmup_steps=100,
        total_steps=1000,
        initial_sim_ratio=0.9,
        final_sim_ratio=0.2,
        adaptation_method="mmd",
    )
    transfer = SimToRealTransfer(transfer_config)

    # 4. Set up agent
    agent = MultiTurnAgent(AgentConfig(model_name="gpt2"))
    await agent.initialize()

    # 5. Training loop with progressive transfer
    for step in range(1000):
        # Get current difficulty and sim ratio
        difficulty = randomizer.get_current_difficulty(step)
        sim_ratio = transfer.get_sim_ratio(step)

        # Sample persona and scenario
        persona = randomizer.sample_persona()
        scenario = randomizer.sample_scenario(
            topic="customer_service",
            difficulty=difficulty,
        )

        # Generate training conversation
        if random.random() < sim_ratio:
            # Simulated conversation
            conversation = await simulator.generate_conversation(
                scenario=scenario,
                persona=persona,
                agent=agent,
            )
            domain_label = 0  # sim
        else:
            # Real conversation (from buffer)
            conversation = real_buffer.sample()
            domain_label = 1  # real

        # Train with domain adaptation
        loss = train_step(agent, conversation, transfer, domain_label)

        if step % 100 == 0:
            print(f"Step {step}: sim_ratio={sim_ratio:.2f}, difficulty={difficulty:.2f}")

    # 6. Final evaluation
    evaluator = SimToRealEvaluator()

    sim_convs = [
        await simulator.generate_conversation(agent=agent)
        for _ in range(50)
    ]

    metrics = evaluator.evaluate(
        sim_conversations=sim_convs,
        real_conversations=real_test_data,
        agent=agent,
    )

    print(f"Final transfer gap: {metrics.transfer_gap:.4f}")
    print(f"Real performance: {metrics.real_performance:.3f}")

    return agent

asyncio.run(sim_to_real_training())
```

### Example 3: Decision Transformer for Quality-Conditioned Generation

```python
import asyncio
from stateset_agents.data import ConversationDataset
from stateset_agents.training import (
    DecisionTransformerConfig,
    DecisionTransformerTrainer,
    DecisionTransformer,
)

async def train_decision_transformer():
    # 1. Load dataset with rewards
    dataset = ConversationDataset.from_jsonl("rated_conversations.jsonl")

    # 2. Configure Decision Transformer
    config = DecisionTransformerConfig(
        hidden_dim=256,
        num_layers=4,
        num_heads=4,
        context_length=10,
        state_dim=768,
        action_dim=768,
        max_return=1.0,
        learning_rate=1e-4,
    )

    # 3. Train
    trainer = DecisionTransformerTrainer(config)
    metrics = await trainer.train(dataset, epochs=50)

    print(f"Final loss: {metrics['loss'][-1]:.4f}")

    # 4. Generate high-quality responses
    model = trainer.model

    # Condition on high return (quality)
    response = model.generate(
        states=conversation_embeddings,
        actions=previous_responses,
        returns_to_go=[0.95],  # Request high quality
        timesteps=[current_turn],
    )

    return model

asyncio.run(train_decision_transformer())
```

---

## Best Practices

### Data Quality

1. **Filter low-quality conversations**: Set `quality_threshold >= 0.6`
2. **Balance your dataset**: Ensure diverse topics and outcomes
3. **Validate reward labels**: Spot-check that rewards reflect true quality
4. **Handle outliers**: Remove or cap extreme reward values

### Algorithm Selection

| Scenario | Recommended Algorithm |
|----------|----------------------|
| Small dataset (< 10k) | IQL or CQL |
| Large dataset (> 100k) | BCQ or BEAR |
| Variable quality data | Decision Transformer |
| Need conservative policy | BCQ with low threshold |
| Hybrid offline+online | OfflineGRPOTrainer |

### Sim-to-Real Transfer

1. **Calibrate early**: Calibrate simulator to real data before training
2. **Monitor gap metrics**: Alert if KL divergence exceeds 0.3
3. **Use curriculum**: Start with easy scenarios, increase difficulty
4. **Domain randomization**: Higher variation improves generalization
5. **Progressive transfer**: Cosine schedule works well in practice

### Hyperparameters

```python
# Conservative starting point for offline RL
offline_defaults = {
    "learning_rate": 3e-4,
    "discount": 0.99,
    "batch_size": 64,
    "hidden_dim": 256,
    "num_layers": 3,
}

# Sim-to-real defaults
sim_to_real_defaults = {
    "initial_sim_ratio": 0.9,
    "final_sim_ratio": 0.2,
    "transfer_schedule": "cosine",
    "persona_variation": 0.3,
    "adaptation_weight": 0.1,
}
```

### Evaluation

1. **Hold out real data**: Never train on your real test set
2. **Measure multiple metrics**: KL, JS, MMD, and actual performance
3. **A/B test carefully**: Compare sim-trained vs baseline on real users
4. **Monitor drift**: Sim-real gap may change as user behavior evolves

---

## API Reference

### Data Module

```python
from stateset_agents.data import (
    ConversationDataset,
    ConversationDatasetConfig,
    ConversationReplayBuffer,
    EmbeddingCache,
)
```

### Training Module

```python
from stateset_agents.training import (
    # Offline RL
    BCQConfig, BCQTrainer, ConversationalVAE,
    BEARConfig, BEARTrainer, MMDKernel,
    CQLConfig, ConservativeQLearning,
    IQLConfig, ImplicitQLearning,
    DecisionTransformerConfig, DecisionTransformer, DecisionTransformerTrainer,
    OfflineGRPOConfig, OfflineGRPOTrainer,

    # Sim-to-Real
    SimToRealConfig, SimToRealTransfer,
    DomainRandomizationConfig, DomainRandomizer,
    PersonaGenerator, ScenarioGenerator, UserPersona,
)
```

### Environments Module

```python
from stateset_agents.environments import (
    ConversationSimulator,
    ConversationSimulatorConfig,
    UserSimulator,
)
```

### Evaluation Module

```python
from stateset_agents.evaluation import (
    SimToRealMetrics,
    SimToRealEvaluator,
    compute_kl_divergence,
    compute_js_divergence,
    compute_mmd,
)
```

---

## References

1. **BCQ**: Fujimoto et al., "Off-Policy Deep Reinforcement Learning without Exploration" (2019)
2. **BEAR**: Kumar et al., "Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction" (2019)
3. **CQL**: Kumar et al., "Conservative Q-Learning for Offline Reinforcement Learning" (2020)
4. **IQL**: Kostrikov et al., "Offline Reinforcement Learning with Implicit Q-Learning" (2021)
5. **Decision Transformer**: Chen et al., "Decision Transformer: Reinforcement Learning via Sequence Modeling" (2021)
6. **Domain Randomization**: Tobin et al., "Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World" (2017)
7. **DANN**: Ganin et al., "Domain-Adversarial Training of Neural Networks" (2016)
