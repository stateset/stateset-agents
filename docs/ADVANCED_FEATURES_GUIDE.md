# Advanced Features Guide: Achieving 10/10

This guide covers the advanced features that elevate StateSet Agents to a perfect 10/10 rating:

1. **Curriculum Learning** - Progressive task difficulty
2. **Multi-Agent Coordination** - Team-based scenarios
3. **Offline RL** - Learning from fixed datasets (CQL & IQL)
4. **Bayesian Uncertainty Quantification** - Confidence-aware rewards
5. **Few-Shot Adaptation** - Rapid domain transfer

---

## 1. Curriculum Learning

Curriculum learning progressively increases task difficulty during training, leading to better convergence and performance.

### Features

- **Automatic stage progression** based on performance
- **Adaptive difficulty adjustment** within stages
- **Multiple scheduling strategies**: performance-based, adaptive, mixed
- **Dynamic task complexity** scaling

### Quick Start

```python
from stateset_agents.core.curriculum_learning import (
    CurriculumLearning,
    CurriculumStage,
    PerformanceBasedScheduler,
)

# Define curriculum stages
stages = [
    CurriculumStage(
        stage_id="beginner",
        difficulty_level=0.3,
        task_config={
            "max_turns": 5,
            "context_complexity": 0.2,
        },
        success_threshold=0.7,
        min_episodes=100,
    ),
    CurriculumStage(
        stage_id="intermediate",
        difficulty_level=0.6,
        task_config={
            "max_turns": 10,
            "context_complexity": 0.5,
        },
        success_threshold=0.75,
        min_episodes=150,
    ),
    CurriculumStage(
        stage_id="advanced",
        difficulty_level=0.9,
        task_config={
            "max_turns": 20,
            "context_complexity": 0.9,
        },
        success_threshold=0.8,
        min_episodes=200,
    ),
]

# Create curriculum
curriculum = CurriculumLearning(
    stages=stages,
    scheduler=PerformanceBasedScheduler(
        window_size=50,
        success_threshold=0.7,
    ),
)

# Training loop
for episode in range(1000):
    # Get current config
    config = curriculum.get_current_config()

    # Run episode with current difficulty
    trajectory = await run_episode(agent, config)

    # Record episode (automatically advances stages)
    curriculum.record_episode(trajectory)

    # Check progress
    if curriculum.is_curriculum_complete():
        print("Curriculum complete!")
        break

    summary = curriculum.get_progress_summary()
    print(f"Stage: {summary['current_stage']}/{summary['total_stages']}, "
          f"Success: {summary['stage_success_rate']:.2%}")
```

### Auto-Generated Curriculum

```python
# Automatically generate curriculum
curriculum = CurriculumLearning(
    stages=[],
    auto_generate_stages=True,
    num_stages=5,
)
```

### Adaptive Scheduling

```python
from stateset_agents.core.curriculum_learning import AdaptiveScheduler

# Advance based on learning curves
scheduler = AdaptiveScheduler(
    learning_rate_threshold=0.01,  # Plateau detection
    performance_threshold=0.65,
    lookback_window=100,
)

curriculum = CurriculumLearning(stages=stages, scheduler=scheduler)
```

---

## 2. Multi-Agent Coordination

Enable multiple agents to collaborate on complex tasks.

### Features

- **Multiple coordination strategies**: sequential, parallel, consensus, competitive
- **Communication protocols**: blackboard, broadcast, peer-to-peer, hierarchical
- **Task allocation**: capability-based, performance-based
- **Cooperative reward shaping**
- **Team performance tracking**

### Quick Start

```python
from stateset_agents.core.multi_agent_coordination import (
    MultiAgentCoordinator,
    AgentRole,
    CoordinationStrategy,
    CommunicationProtocol,
)

# Create agents
agents = {
    "coordinator": coordinator_agent,
    "researcher": researcher_agent,
    "executor": executor_agent,
}

# Assign roles
roles = {
    "coordinator": AgentRole.COORDINATOR,
    "researcher": AgentRole.RESEARCHER,
    "executor": AgentRole.EXECUTOR,
}

# Create coordinator
coordinator = MultiAgentCoordinator(
    agents=agents,
    roles=roles,
    coordination_strategy=CoordinationStrategy.SEQUENTIAL,
    communication_protocol=CommunicationProtocol.BLACKBOARD,
)

# Execute collaborative task
task = {
    "task_id": "complex_task",
    "description": "Solve a multi-step problem",
    "required_capabilities": ["research", "execution"],
}

trajectory, result = await coordinator.execute_collaborative_task(
    task,
    max_iterations=10,
)

# Get team statistics
stats = coordinator.get_team_statistics()
print(f"Team completed {stats['completed_tasks']} tasks")
print(f"Average performance: {stats['agent_performance']}")
```

### Parallel Execution

```python
coordinator = MultiAgentCoordinator(
    agents=agents,
    coordination_strategy=CoordinationStrategy.PARALLEL,
)

# All agents work simultaneously
trajectory, result = await coordinator.execute_collaborative_task(task)
```

### Consensus-Based Decisions

```python
coordinator = MultiAgentCoordinator(
    agents=agents,
    coordination_strategy=CoordinationStrategy.CONSENSUS,
)

# Agents must agree on solution
trajectory, result = await coordinator.execute_collaborative_task(
    task,
    max_iterations=5,
)
```

### Cooperative Reward Shaping

```python
from stateset_agents.core.multi_agent_coordination import CooperativeRewardShaping

reward_shaper = CooperativeRewardShaping(
    team_reward_weight=0.5,
    individual_reward_weight=0.3,
    cooperation_bonus_weight=0.2,
)

# Compute rewards that encourage cooperation
agent_rewards = reward_shaper.compute_agent_rewards(
    team_reward=1.0,
    individual_contributions={"agent1": 0.8, "agent2": 0.7},
    cooperation_metrics={"agent1": 0.9, "agent2": 0.85},
)
```

---

## 3. Offline RL (CQL & IQL)

Learn from fixed datasets without online interaction.

### Features

- **Conservative Q-Learning (CQL)** - Prevents overestimation
- **Implicit Q-Learning (IQL)** - Avoids distribution shift
- **Unified trainer interface**
- **Support for custom datasets**

### Conservative Q-Learning (CQL)

```python
from stateset_agents.training.offline_rl_algorithms import (
    OfflineRLTrainer,
    CQLConfig,
)

# Configure CQL
config = CQLConfig(
    hidden_size=256,
    cql_alpha=1.0,
    min_q_weight=5.0,
    learning_rate=3e-4,
    batch_size=256,
)

# Create trainer
trainer = OfflineRLTrainer(
    algorithm="cql",
    state_dim=768,  # LLM embedding size
    action_dim=512,
    config=config,
)

# Prepare dataset
dataset = {
    "states": np.array([...]),  # [N, state_dim]
    "actions": np.array([...]),  # [N, action_dim]
    "rewards": np.array([...]),  # [N]
    "next_states": np.array([...]),  # [N, state_dim]
    "dones": np.array([...]),  # [N]
}

# Train
metrics = trainer.train(
    dataset=dataset,
    num_epochs=100,
    batch_size=256,
)

# Save model
trainer.save("cql_model.pt")
```

### Implicit Q-Learning (IQL)

```python
from stateset_agents.training.offline_rl_algorithms import IQLConfig

# Configure IQL
config = IQLConfig(
    hidden_size=256,
    expectile=0.7,  # Focus on upper tail
    temperature=3.0,
    learning_rate=3e-4,
)

trainer = OfflineRLTrainer(
    algorithm="iql",
    state_dim=768,
    action_dim=512,
    config=config,
)

# Train on offline data
metrics = trainer.train(dataset, num_epochs=100)
```

### Building Offline Datasets

```python
# Collect trajectories from existing agent
offline_data = {
    "states": [],
    "actions": [],
    "rewards": [],
    "next_states": [],
    "dones": [],
}

for trajectory in collected_trajectories:
    for turn in trajectory.turns:
        state = encode_state(turn)
        action = encode_action(turn)
        reward = turn.reward or 0.0

        offline_data["states"].append(state)
        offline_data["actions"].append(action)
        offline_data["rewards"].append(reward)
        # ... etc
```

---

## 4. Bayesian Uncertainty Quantification

Get confidence intervals and uncertainty estimates with reward predictions.

### Features

- **Epistemic uncertainty** - Model uncertainty
- **Aleatoric uncertainty** - Data uncertainty
- **Ensemble methods** for robust estimates
- **Monte Carlo Dropout**
- **Active learning** integration
- **Calibration metrics**

### Quick Start

```python
from stateset_agents.rewards.bayesian_reward_model import (
    BayesianRewardFunction,
    BayesianRewardConfig,
)

# Configure Bayesian reward model
config = BayesianRewardConfig(
    hidden_size=256,
    num_samples=20,  # MC samples
    num_ensemble=5,  # Ensemble size
    use_ensemble=True,
    high_uncertainty_threshold=0.3,
)

reward_fn = BayesianRewardFunction(
    input_dim=768,
    config=config,
)

# Get reward with uncertainty
turns = [
    ConversationTurn(role="user", content="Question"),
    ConversationTurn(role="assistant", content="Answer"),
]

result = await reward_fn.compute_reward(turns)

print(f"Reward: {result.score:.3f}")
print(f"Epistemic uncertainty: {result.breakdown['epistemic_uncertainty']:.3f}")
print(f"Aleatoric uncertainty: {result.breakdown['aleatoric_uncertainty']:.3f}")
print(f"95% CI: [{result.breakdown['confidence_interval_lower']:.3f}, "
      f"{result.breakdown['confidence_interval_upper']:.3f}]")
print(f"Confidence: {result.metadata['confidence']:.2%}")

if result.metadata['high_uncertainty']:
    print("⚠️  High uncertainty - consider getting human feedback")
```

### Active Learning

```python
from stateset_agents.rewards.bayesian_reward_model import ActiveLearningSelector

selector = ActiveLearningSelector(
    uncertainty_threshold=0.3,
    diversity_weight=0.5,
)

# Check if sample should be labeled
if selector.should_query_label(result):
    human_label = get_human_feedback(turns)
    # Use for model retraining

# Select batch for labeling
candidates = [(features, result) for features, result in candidate_pool]
indices = selector.select_batch_for_labeling(candidates, batch_size=10)
```

### Calibration

```python
# Collect predictions and actual outcomes
predicted = [0.7, 0.8, 0.6, 0.9]
actual = [0.75, 0.82, 0.55, 0.88]

# Calibrate uncertainty estimates
metrics = reward_fn.calibrate(predicted, actual)

print(f"MAE: {metrics['mae']:.3f}")
print(f"RMSE: {metrics['rmse']:.3f}")
```

---

## 5. Few-Shot Adaptation

Rapidly adapt agents to new domains with minimal examples.

### Features

- **Prompt-based adaptation** - No parameter updates
- **LoRA fine-tuning** - Efficient parameter updates
- **MAML** - Meta-learning for fast adaptation
- **Domain detection** - Automatic domain selection
- **Cross-domain transfer** - Leverage source domain knowledge

### Quick Start

```python
from stateset_agents.core.few_shot_adaptation import (
    FewShotAdaptationManager,
    FewShotExample,
    DomainProfile,
    PromptBasedAdaptation,
)

# Create adaptation manager
manager = FewShotAdaptationManager(
    base_agent=base_agent,
    default_strategy=PromptBasedAdaptation(max_examples=5),
)

# Register domains
customer_service = DomainProfile(
    domain_id="customer_service",
    name="Customer Service",
    description="Handle customer inquiries and issues",
    keywords=["help", "support", "issue", "problem"],
)

examples = [
    FewShotExample(
        input="My order hasn't arrived",
        output="I apologize for the delay. Let me check your order status...",
        reward=0.9,
    ),
    FewShotExample(
        input="How do I return an item?",
        output="Our return policy allows returns within 30 days...",
        reward=0.85,
    ),
]

manager.register_domain(customer_service, examples)

# Get adapted agent
adapted_agent = await manager.get_adapted_agent("customer_service")

# Use adapted agent
response = await adapted_agent.generate_response("I need help with my order")
```

### Multiple Domains

```python
# Register multiple domains
domains = {
    "technical": DomainProfile(
        domain_id="technical",
        name="Technical Support",
        description="Solve technical problems",
        keywords=["error", "bug", "crash", "technical"],
    ),
    "sales": DomainProfile(
        domain_id="sales",
        name="Sales",
        description="Product information and sales",
        keywords=["price", "buy", "purchase", "features"],
    ),
}

for domain_id, domain in domains.items():
    manager.register_domain(domain, domain_examples[domain_id])
```

### Automatic Domain Detection

```python
from stateset_agents.core.few_shot_adaptation import DomainDetector

detector = DomainDetector(manager.domain_profiles)

# Detect domain from input
user_input = "My software keeps crashing"
domain_id, confidence = detector.detect_domain(user_input)

print(f"Detected domain: {domain_id} (confidence: {confidence:.2%})")

# Get appropriate agent
adapted_agent = await manager.get_adapted_agent(domain_id)
response = await adapted_agent.generate_response(user_input)
```

### Cross-Domain Transfer

```python
# Transfer from source domain (many examples) to target (few examples)
transferred_agent = await manager.cross_domain_transfer(
    source_domain_id="customer_service",  # 1000 examples
    target_domain_id="technical_support",  # 10 examples
    num_target_examples=10,
)
```

### LoRA Adaptation

```python
from stateset_agents.core.few_shot_adaptation import LoRAAdaptation

# Efficient fine-tuning with LoRA
lora_adapter = LoRAAdaptation(
    rank=8,
    alpha=16,
    learning_rate=1e-4,
    num_epochs=5,
)

adapted = await lora_adapter.adapt(base_agent, examples, domain)
```

### Evaluation

```python
# Evaluate adaptation quality
test_examples = [
    FewShotExample(input="test1", output="expected1"),
    FewShotExample(input="test2", output="expected2"),
]

metrics = await manager.evaluate_adaptation(
    domain_id="customer_service",
    test_examples=test_examples,
    reward_function=reward_fn,
)

print(f"Average reward: {metrics['average_reward']:.3f}")
print(f"Test accuracy: {metrics['accuracy']:.2%}")
```

---

## Combining Features

### Example: Complete Training Pipeline

```python
from stateset_agents.core.curriculum_learning import CurriculumLearning
from stateset_agents.core.multi_agent_coordination import MultiAgentCoordinator
from stateset_agents.rewards.bayesian_reward_model import BayesianRewardFunction
from stateset_agents.core.few_shot_adaptation import FewShotAdaptationManager

# 1. Curriculum learning for progressive difficulty
curriculum = CurriculumLearning(
    stages=curriculum_stages,
    auto_generate_stages=True,
    num_stages=5,
)

# 2. Bayesian rewards with uncertainty
reward_fn = BayesianRewardFunction(config=bayesian_config)

# 3. Few-shot adaptation for rapid domain switching
adaptation_manager = FewShotAdaptationManager(base_agent)

# 4. Multi-agent for complex tasks
coordinator = MultiAgentCoordinator(agents=agents)

# Training loop
for episode in range(num_episodes):
    # Get current curriculum config
    config = curriculum.get_current_config()

    # Adapt to domain if needed
    domain_id = detect_domain(task)
    adapted_agent = await adaptation_manager.get_adapted_agent(domain_id)

    # Execute with multi-agent if complex
    if is_complex_task(task):
        trajectory, result = await coordinator.execute_collaborative_task(task)
    else:
        trajectory = await run_single_agent_episode(adapted_agent, config)

    # Get reward with uncertainty
    reward_result = await reward_fn.compute_reward(trajectory.turns)

    # Active learning: query high uncertainty cases
    if reward_result.metadata['high_uncertainty']:
        human_label = await get_human_feedback(trajectory)
        reward_fn.calibrate([reward_result.score], [human_label])

    # Record for curriculum advancement
    trajectory.total_reward = reward_result.score
    curriculum.record_episode(trajectory)
```

---

## Best Practices

### Curriculum Learning
- Start with easier tasks to build foundational skills
- Use adaptive scheduling for better efficiency
- Monitor success rates to tune thresholds
- Validate on held-out tasks at each stage

### Multi-Agent Coordination
- Assign roles based on agent capabilities
- Use sequential coordination for dependent tasks
- Use parallel for independent subtasks
- Track individual and team metrics

### Offline RL
- Ensure dataset diversity
- Use CQL for conservative estimates
- Use IQL for stable learning
- Validate on separate test set

### Bayesian Rewards
- Calibrate on labeled data regularly
- Use active learning for efficient labeling
- Monitor confidence trends over time
- Set appropriate uncertainty thresholds

### Few-Shot Adaptation
- Start with prompt-based (fastest)
- Use LoRA for better performance
- Collect diverse examples per domain
- Leverage cross-domain transfer

---

## Performance Tips

1. **Curriculum Learning**: Cache environment configurations per stage
2. **Multi-Agent**: Use async/await for parallel execution
3. **Offline RL**: Batch processing for GPU efficiency
4. **Bayesian Rewards**: Use ensemble for better uncertainty estimates
5. **Few-Shot**: Cache adapted agents per domain

---

## Troubleshooting

### Curriculum not advancing
- Check success thresholds (may be too high)
- Verify min_episodes is reasonable
- Review recent reward trends

### Multi-agent coordination issues
- Ensure agents have required capabilities
- Check communication channel is working
- Monitor individual agent performance

### Offline RL not learning
- Check dataset quality and diversity
- Tune hyperparameters (learning rate, batch size)
- Verify state/action representations

### High uncertainty persists
- Need more training data
- Check model capacity
- Calibrate with labeled samples

### Poor few-shot adaptation
- Add more diverse examples
- Try different adaptation strategies
- Use cross-domain transfer

---

## Next Steps

- See `examples/` for complete working examples
- Check individual feature documentation for advanced usage
- Join our community for tips and best practices
- Contribute your own adaptations and improvements!

---

**Congratulations!** You now have access to all the features that make StateSet Agents a 10/10 framework. 🎉
