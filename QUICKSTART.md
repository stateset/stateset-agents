# StateSet Agents - Quick Start Guide

Get up and running with StateSet Agents in 15 minutes.

---

## Table of Contents

1. [Installation](#installation)
2. [First Agent](#first-agent)
3. [Training Your First Agent](#training-your-first-agent)
4. [Optional: Continual Learning + Planning](#optional-continual-learning--planning)
5. [Using Pre-built Rewards](#using-pre-built-rewards)
6. [Stub Mode for Development](#stub-mode-for-development)
7. [Next Steps](#next-steps)

---

## Installation

### Basic Installation

```bash
# Install core framework
pip install stateset-agents
```

### Full Installation (for training)

```bash
# Clone the repository
git clone https://github.com/stateset/stateset-agents
cd stateset-agents

# Install with all dependencies
pip install -e ".[dev,api,trl]"
```

### GPU Support

```bash
# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation

```bash
python -c "import stateset_agents; print(stateset_agents.__version__)"
# Should print: 0.5.0 (or later)
```

---

## First Agent

### Hello World

Create a file `hello_agent.py`:

```python
import asyncio
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

async def main():
    # Create a simple agent
    agent = MultiTurnAgent(AgentConfig(
        model_name="gpt2",  # Small model for quick testing
        max_length=100,
        temperature=0.7
    ))

    # Initialize the agent
    await agent.initialize()

    # Have a conversation
    messages = [
        {"role": "user", "content": "Hello! How are you?"}
    ]

    response = await agent.generate_response(messages)
    print(f"Agent: {response}")

    # Continue the conversation
    messages.append({"role": "assistant", "content": response})
    messages.append({"role": "user", "content": "Tell me about StateSet Agents"})

    response = await agent.generate_response(messages)
    print(f"Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python hello_agent.py
```

---

## Training Your First Agent

### Simple Training Example

Create a file `train_simple.py`:

```python
import asyncio
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import create_customer_service_reward
from training.train import train

async def main():
    # 1. Create the agent
    print("Creating agent...")
    agent = MultiTurnAgent(AgentConfig(
        model_name="gpt2",
        enable_lora=True,  # Use LoRA for efficient training
        lora_config={"r": 8, "lora_alpha": 16}
    ))
    await agent.initialize()

    # 2. Define training scenarios
    scenarios = [
        {
            "topic": "refund",
            "user_goal": "Get a refund for delayed order",
            "context": {"order_status": "delayed", "days_late": 5}
        },
        {
            "topic": "shipping",
            "user_goal": "Track my package",
            "context": {"tracking_number": "ABC123"}
        },
        {
            "topic": "product_info",
            "user_goal": "Learn about product features",
            "context": {"product": "Pro Plan"}
        }
    ]

    # 3. Create environment
    environment = ConversationEnvironment(
        scenarios=scenarios,
        max_turns=6,  # Maximum 6 turns per conversation
    )

    # 4. Use pre-built reward function
    reward_fn = create_customer_service_reward()

    # 5. Train the agent
    print("Starting training...")
    trained_agent = await train(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        num_episodes=50,  # Start small
        save_path="./checkpoints/my_first_agent"
    )

    print("Training complete! Checkpoint saved to ./checkpoints/my_first_agent")

    # 6. Test the trained agent
    print("\nTesting trained agent...")
    messages = [
        {"role": "user", "content": "My order is delayed. I want a refund!"}
    ]
    response = await trained_agent.generate_response(messages)
    print(f"Trained Agent: {response}")

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python train_simple.py
```

**Expected output:**
- Training progress with episode numbers
- Reward metrics per episode
- Final checkpoint saved
- Test response from trained agent

---

## Optional: Continual Learning + Planning

Enable planning context and continual learning in training:

```python
agent = MultiTurnAgent(AgentConfig(
    model_name="gpt2",
    enable_planning=True,
    planning_config={"max_steps": 4},
))

trained_agent = await train(
    agent=agent,
    environment=environment,
    reward_fn=reward_fn,
    num_episodes=50,
    # resume_from_checkpoint="./outputs/checkpoint-100",
    config_overrides={
        "continual_strategy": "replay_lwf",
        "replay_ratio": 0.3,
        "replay_sampling": "balanced",
        "task_id_key": "task_id",
    },
)

context = {"conversation_id": "demo-trip", "goal": "Plan a weekend in Austin"}
first = await trained_agent.generate_response(
    [{"role": "user", "content": "Can you outline a plan?"}],
    context=context,
)

second = await trained_agent.generate_response(
    [{"role": "user", "content": "Nice. What's next?"}],
    context={"conversation_id": "demo-trip", "plan_update": {"action": "advance"}},
)

# To update the plan goal explicitly:
# context={"conversation_id": "demo-trip", "plan_goal": "Plan a weekend in Dallas"}
```

---

## Using Pre-built Rewards

StateSet Agents comes with 10+ pre-built reward functions:

### Customer Service Agent

```python
from stateset_agents.core.reward import create_customer_service_reward

reward_fn = create_customer_service_reward()
# Combines: Helpfulness (35%), Safety (25%), Engagement (20%), Conciseness (20%)
```

### Technical Support Agent

```python
from stateset_agents.core.reward import create_domain_reward

reward_fn = create_domain_reward("technical_support")
# Emphasizes: Technical accuracy, step-by-step guidance, safety
```

### Sales Assistant

```python
from stateset_agents.core.reward import create_domain_reward

reward_fn = create_domain_reward("sales")
# Emphasizes: Engagement, product knowledge, conversion potential
```

### Custom Composite Reward

```python
from stateset_agents.core.reward import (
    CompositeReward,
    HelpfulnessReward,
    SafetyReward,
    CorrectnessReward
)

# Create your own blend
reward_fn = CompositeReward([
    (HelpfulnessReward(), 0.5),   # 50% weight
    (SafetyReward(), 0.3),        # 30% weight
    (CorrectnessReward(), 0.2)    # 20% weight
])
```

### Custom Reward Function

```python
from stateset_agents.core.reward import RewardFunction, RewardResult
from stateset_agents.core.trajectory import ConversationTurn
from typing import List, Dict, Any, Optional

class MyCustomReward(RewardFunction):
    """Reward function for my specific use case"""

    async def compute_reward(
        self,
        turns: List[ConversationTurn],
        context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        # Get the last agent response
        agent_turns = [t for t in turns if t.speaker == "agent"]
        if not agent_turns:
            return RewardResult(score=0.0, breakdown={})

        last_response = agent_turns[-1].content

        # Custom scoring logic
        score = 0.0

        # Example: reward polite language
        if any(word in last_response.lower() for word in ["please", "thank you"]):
            score += 0.3

        # Example: reward specific keywords
        if context and "product" in context:
            if context["product"] in last_response:
                score += 0.4

        # Example: penalize very long responses
        if len(last_response) > 500:
            score -= 0.2

        return RewardResult(
            score=max(0.0, min(1.0, score)),  # Clamp to [0, 1]
            breakdown={
                "politeness": 0.3 if "please" in last_response.lower() else 0.0,
                "relevance": 0.4 if context and context.get("product") in last_response else 0.0,
                "length_penalty": -0.2 if len(last_response) > 500 else 0.0
            }
        )

# Use it
reward_fn = MyCustomReward()
```

---

## Stub Mode for Development

Use stub mode for fast development without downloading models:

### Offline Agent

```python
import asyncio
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent

async def main():
    # Create a stub agent (no model download!)
    agent = MultiTurnAgent(AgentConfig(
        model_name="stub://demo",
        use_stub_model=True,
        stub_responses=[
            "Hello! How can I help you today?",
            "I understand. Let me look into that.",
            "Is there anything else I can help with?"
        ]
    ))
    await agent.initialize()

    # Use just like a real agent
    messages = [{"role": "user", "content": "Hello"}]
    response = await agent.generate_response(messages)
    print(f"Stub Agent: {response}")

asyncio.run(main())
```

### Stub Training

```python
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.computational_engine import create_computational_engine

async def quick_test():
    # Stub agent for testing
    agent = MultiTurnAgent(AgentConfig(
        model_name="stub://test",
        use_stub_model=True
    ))
    await agent.initialize()

    scenarios = [{"topic": "test"}]
    env = ConversationEnvironment(scenarios=scenarios, max_turns=3)

    # Create engine (works with stub!)
    engine = create_computational_engine(agent, env)

    # Generate test trajectories
    trajectories = await engine.generate_trajectory_batch(batch_size=4)

    print(f"Generated {len(trajectories)} trajectories in stub mode!")
    for i, traj in enumerate(trajectories):
        print(f"  Trajectory {i+1}: {len(traj.turns)} turns")

asyncio.run(quick_test())
```

**Benefits of Stub Mode:**
- ✅ No model downloads (works offline)
- ✅ Fast execution (instant responses)
- ✅ Perfect for CI/CD pipelines
- ✅ Great for testing framework logic
- ✅ Deterministic responses for testing

---

## Command Line Interface

### Check Environment

```bash
stateset-agents doctor
```

### Initialize Config

```bash
stateset-agents init --path ./config.yaml
```

### Train from CLI

```bash
stateset-agents train --config ./config.yaml --save ./checkpoints
```

### Evaluate Agent

```bash
stateset-agents evaluate \
  --checkpoint ./checkpoints/my_agent \
  --message "Hello, I need help"
```

### Serve API

```bash
stateset-agents serve --host 0.0.0.0 --port 8000
```

### Stub Training (Fast Testing)

```bash
stateset-agents train --stub
```

---

## Complete Example: Customer Service Agent

Here's a complete example you can run:

```python
"""
Complete customer service agent training example
Save as: train_customer_service.py
"""

import asyncio
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.core.reward import create_customer_service_reward
from training.config import TrainingConfig, TrainingProfile
from training.trainer import MultiTurnGRPOTrainer

async def main():
    # 1. Agent configuration
    print("🤖 Creating customer service agent...")
    agent_config = AgentConfig(
        model_name="gpt2",  # Use larger model for production
        max_length=200,
        temperature=0.7,
        enable_lora=True,
        lora_config={"r": 16, "lora_alpha": 32, "target_modules": ["c_attn"]}
    )
    agent = MultiTurnAgent(agent_config)
    await agent.initialize()

    # 2. Training scenarios
    print("📋 Setting up training scenarios...")
    scenarios = [
        {
            "topic": "refund_request",
            "user_goal": "Get refund for late delivery",
            "context": {"order_id": "ORD-001", "days_late": 7}
        },
        {
            "topic": "product_question",
            "user_goal": "Understand product features",
            "context": {"product": "Premium Plan"}
        },
        {
            "topic": "billing_issue",
            "user_goal": "Resolve duplicate charge",
            "context": {"amount": "$49.99"}
        },
        {
            "topic": "shipping_tracking",
            "user_goal": "Track package location",
            "context": {"tracking": "1Z999AA10123456784"}
        },
        {
            "topic": "account_problem",
            "user_goal": "Reset forgotten password",
            "context": {"email": "user@example.com"}
        }
    ]

    environment = ConversationEnvironment(
        scenarios=scenarios,
        max_turns=8,  # Allow longer conversations
    )

    # 3. Reward function
    print("🎯 Configuring reward function...")
    reward_fn = create_customer_service_reward()

    # 4. Training configuration
    print("⚙️  Setting up training config...")
    training_config = TrainingConfig.from_profile(TrainingProfile.BALANCED)
    training_config.num_episodes = 100
    training_config.num_generations = 8  # 8 trajectories per scenario
    training_config.eval_steps = 20
    training_config.save_steps = 50

    # 5. Create trainer
    print("🏋️  Initializing trainer...")
    trainer = MultiTurnGRPOTrainer(
        agent=agent,
        environment=environment,
        reward_fn=reward_fn,
        config=training_config
    )
    await trainer.initialize()

    # 6. Train!
    print("🚀 Starting training...\n")
    trained_agent = await trainer.train()

    print("\n✅ Training complete!")

    # 7. Test the trained agent
    print("\n🧪 Testing trained agent...\n")

    test_conversations = [
        "Hi, my order still hasn't arrived and it's been a week!",
        "Can you explain what features are included in the Premium Plan?",
        "I was charged twice for my last order. Can you help?"
    ]

    for i, user_message in enumerate(test_conversations, 1):
        print(f"Test {i}:")
        print(f"User: {user_message}")

        messages = [{"role": "user", "content": user_message}]
        response = await trained_agent.generate_response(messages)

        print(f"Agent: {response}\n")

    # 8. Save checkpoint
    checkpoint_path = "./checkpoints/customer_service_agent"
    print(f"💾 Saving checkpoint to {checkpoint_path}")
    # Checkpoint is automatically saved during training

if __name__ == "__main__":
    asyncio.run(main())
```

Run it:

```bash
python train_customer_service.py
```

---

## Common Issues & Solutions

### Issue: Out of Memory

**Solution**: Reduce batch size and enable gradient accumulation

```python
config = TrainingConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    bf16=True
)
```

### Issue: Training is Slow

**Solution**: Use TRL integration for optimized training

```python
from training.trl_grpo_trainer import train_with_trl_grpo

trained_agent = await train_with_trl_grpo(
    config=config,
    agent=agent,
    environment=environment,
    reward_model=reward_fn
)
```

### Issue: Model Not Found

**Solution**: Check HuggingFace model name or use stub mode

```python
# Option 1: Correct model name
agent_config = AgentConfig(model_name="meta-llama/Llama-2-7b-chat-hf")

# Option 2: Use stub for testing
agent_config = AgentConfig(model_name="stub://test", use_stub_model=True)
```

### Issue: Import Errors

**Solution**: Install with all dependencies

```bash
pip install -e ".[dev,api,trl]"
```

---

## Next Steps

### 1. Explore Examples

```bash
# See all examples
ls examples/

# Run complete training example
python examples/complete_grpo_training.py

# Try TRL integration
python examples/train_with_trl_grpo.py

# Customer service example
python examples/customer_service_agent.py
```

### 2. Read Documentation

- **[README.md](README.md)** - Project overview
- **[ARCHITECTURE.md](ARCHITECTURE.md)** - Technical architecture
- **[GRPO_IMPLEMENTATION.md](GRPO_IMPLEMENTATION.md)** - Algorithm details
- **[TRL_GRPO_TRAINING_GUIDE.md](docs/TRL_GRPO_TRAINING_GUIDE.md)** - Advanced training
- **[USAGE_GUIDE.md](docs/USAGE_GUIDE.md)** - Comprehensive usage

### 3. Customize for Your Domain

1. **Define your scenarios** - What conversations should the agent handle?
2. **Create custom rewards** - What behaviors do you want to encourage?
3. **Configure training** - Tune hyperparameters for your use case
4. **Evaluate and iterate** - Test on real conversations and improve

### 4. Deploy to Production

```bash
# Build Docker image
docker build -t my-agent:latest -f deployment/docker/Dockerfile .

# Deploy to Kubernetes
kubectl apply -f deployment/kubernetes/

# Monitor with Grafana
# See deployment/monitoring/dashboards/
```

### 5. Join the Community

- **GitHub Issues**: Report bugs and request features
- **Contributing**: See [CONTRIBUTING.md](CONTRIBUTING.md)
- **Documentation**: Check the docs/ folder for more guides

---

## Quick Reference

### Key Classes

```python
# Agent
from stateset_agents.core.agent import AgentConfig, MultiTurnAgent, ToolAgent

# Environment
from stateset_agents.core.environment import ConversationEnvironment, TaskEnvironment

# Rewards
from stateset_agents.core.reward import (
    RewardFunction,
    CompositeReward,
    create_customer_service_reward,
    create_domain_reward
)

# Training
from training.config import TrainingConfig, TrainingProfile
from training.trainer import MultiTurnGRPOTrainer, SingleTurnGRPOTrainer
from training.train import train

# TRL Integration
from training.trl_grpo_trainer import train_with_trl_grpo
```

### Key Functions

```python
# Create agent
agent = MultiTurnAgent(AgentConfig(...))
await agent.initialize()

# Generate response
response = await agent.generate_response(messages)

# Create environment
env = ConversationEnvironment(scenarios=[...], max_turns=10)

# Create reward
reward_fn = create_customer_service_reward()

# Train
trained_agent = await train(agent, env, reward_fn, num_episodes=100)
```

---

## Support

Need help? Here are your resources:

- **Documentation**: Read the full docs in the `docs/` folder
- **Examples**: Check `examples/` for working code
- **Issues**: https://github.com/stateset/stateset-agents/issues
- **License**: Business Source License 1.1 (see LICENSE file)

---

**Happy Training! 🚀**

Built with ❤️ by the StateSet Team
