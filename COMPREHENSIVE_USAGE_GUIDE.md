# GRPO Agent Framework - Comprehensive Usage Guide

A complete guide for using the GRPO (Group Relative Policy Optimization) Agent Framework to build, train, and deploy production-ready conversational AI agents.

## Table of Contents

1. [Overview](#overview)
2. [Quick Start](#quick-start)
3. [Core Components](#core-components)
4. [Basic Usage Patterns](#basic-usage-patterns)
5. [Advanced Features](#advanced-features)
6. [Training Strategies](#training-strategies)
7. [Production Deployment](#production-deployment)
8. [API Reference](#api-reference)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)

## Overview

The GRPO Agent Framework is a comprehensive solution for building multi-turn conversational AI agents. It combines cutting-edge reinforcement learning techniques with practical production features.

### Key Innovations

- **🧠 Computational Engine**: Massive parallel trajectory generation following the "Bitter Lesson"
- **💬 Multi-Turn Conversations**: Advanced dialogue management with context preservation
- **🎯 Multi-Objective Rewards**: Sophisticated reward composition with multiple criteria
- **⚖️ Neural Reward Models**: Self-improving reward functions that learn from data
- **🔄 Distributed Training**: Multi-GPU scaling with fault tolerance
- **🛠️ Tool Integration**: Extensible agent capabilities with external APIs
- **📊 Real-Time Monitoring**: Comprehensive metrics and observability

### Framework Philosophy

The framework follows the principle that **computation > hand-crafted knowledge**. Rather than relying on manually designed rules, it leverages massive computational resources to learn optimal behaviors from data.

## Quick Start

### Installation

```bash
pip install grpo-agent-framework
```

### Basic Example

```python
import asyncio
from grpo_agent_framework import (
    MultiTurnAgent, 
    create_computational_engine,
    create_customer_service_reward
)

async def quick_start():
    # 1. Create a multi-turn agent
    agent = MultiTurnAgent(
        model_config={
            "model_type": "gpt2",
            "temperature": 0.7,
            "max_tokens": 200
        },
        max_conversation_turns=15
    )
    
    # 2. Start a conversation
    context = await agent.start_conversation(
        user_id="demo_user",
        initial_context={"topic": "customer_service"}
    )
    
    # 3. Handle user messages
    response = await agent.generate_multiturn_response(
        context.conversation_id,
        "Hi, I need help with my order",
        strategy="customer_service"
    )
    
    print(f"Agent: {response}")

# Run the example
asyncio.run(quick_start())
```

## Core Components

### 1. Multi-Turn Agents

Multi-turn agents maintain conversation context across multiple exchanges:

```python
from grpo_agent_framework.core.multiturn_agent import MultiTurnAgent, DialogueDatabase

# Create dialogue database for examples
sample_dialogues = [
    {
        "id": "cs_001",
        "content": "I need help with my order",
        "category": "order_support",
        "expected_response": "I'll help you check your order status"
    }
]

dialogue_db = DialogueDatabase(sample_dialogues)

# Create agent with dialogue database
agent = MultiTurnAgent(
    model_config={
        "model_type": "advanced_customer_service",
        "temperature": 0.7,
        "max_tokens": 200
    },
    max_context_length=2048,
    max_conversation_turns=15,
    dialogue_database=dialogue_db
)
```

### 2. Computational Engine

The computational engine enables massive parallel trajectory generation:

```python
from grpo_agent_framework.core.computational_engine import create_computational_engine
from grpo_agent_framework.core.agent import Agent
from grpo_agent_framework.core.environment import Environment

# Create simple agent
class MyAgent(Agent):
    async def generate_response(self, prompt: str) -> str:
        return f"Response to: {prompt}"

# Create environment
class MyEnvironment(Environment):
    async def reset(self):
        return {"state": "initial"}
    
    async def step(self, action: str):
        return {"reward": 0.5, "done": False}

# Create computational engine
engine = create_computational_engine(
    agent=MyAgent({"model_type": "demo"}),
    environment=MyEnvironment(),
    reward_function=my_reward_function,
    num_workers=4  # Parallel processing
)

# Run training iteration
results = await engine.train_iteration([
    "Hello, how can I help you?",
    "I need support with my account",
    "Can you help me with billing?"
])
```

### 3. Reward Functions

#### Multi-Objective Rewards

```python
from grpo_agent_framework.rewards.multi_objective_reward import (
    create_customer_service_reward,
    MultiObjectiveRewardFunction
)

# Create domain-specific reward
reward_func = create_customer_service_reward(
    expected_responses=["I'll help you with that", "Let me check for you"],
    weight=0.4
)

# Custom multi-objective reward
from grpo_agent_framework.rewards.multi_objective_reward import (
    EmpathyRewardComponent,
    ProfessionalismRewardComponent,
    ActionOrientedRewardComponent
)

multi_reward = MultiObjectiveRewardFunction(
    components=[
        EmpathyRewardComponent(weight=0.3),
        ProfessionalismRewardComponent(weight=0.3),
        ActionOrientedRewardComponent(weight=0.4)
    ]
)
```

#### Neural Reward Models

```python
from grpo_agent_framework.training.neural_reward_trainer import create_neural_reward_function

# Create neural reward that learns from trajectory data
neural_reward = create_neural_reward_function(
    embedding_dim=128,
    hidden_dim=256,
    weight=0.3,
    update_frequency=50
)

# Provide feedback to improve the model
neural_reward.add_trajectory_feedback(
    turns=[
        {"role": "user", "content": "I need help"},
        {"role": "assistant", "content": "I'd be happy to help you!"}
    ],
    reward_score=0.9,
    context={"topic": "customer_service"}
)
```

### 4. Tool Integration

```python
# Register tools for the agent
async def search_knowledge_base(query: str) -> str:
    # Simulate knowledge base search
    return f"Found relevant information about: {query}"

async def create_support_ticket(issue: str) -> str:
    # Simulate ticket creation
    return f"Created support ticket for: {issue}"

# Register tools
agent.register_tool("search_knowledge_base", search_knowledge_base)
agent.register_tool("create_support_ticket", create_support_ticket)

# Tools are automatically used during conversation
response = await agent.generate_multiturn_response(
    conversation_id,
    "I need help with a technical issue",
    use_tools=True
)
```

## Basic Usage Patterns

### 1. Customer Service Agent

```python
async def create_customer_service_agent():
    # Sample customer service dialogues
    dialogues = [
        {
            "id": "cs_001",
            "content": "I'm having trouble with my order",
            "category": "order_issues",
            "expected_response": "I understand your concern. Let me help you check your order status."
        },
        {
            "id": "cs_002",
            "content": "I want to return an item",
            "category": "returns",
            "expected_response": "I'll be happy to help you with the return process."
        }
    ]
    
    # Create agent with customer service strategy
    agent = MultiTurnAgent(
        model_config={
            "model_type": "customer_service",
            "temperature": 0.7,
            "max_tokens": 200
        },
        dialogue_database=DialogueDatabase(dialogues)
    )
    
    # Register customer service tools
    await setup_customer_service_tools(agent)
    
    return agent

async def setup_customer_service_tools(agent):
    async def check_order_status(order_id: str) -> str:
        return f"Order {order_id} is currently being processed"
    
    async def process_return(order_id: str) -> str:
        return f"Return initiated for order {order_id}"
    
    agent.register_tool("check_order_status", check_order_status)
    agent.register_tool("process_return", process_return)
```

### 2. Technical Support Agent

```python
async def create_technical_support_agent():
    agent = MultiTurnAgent(
        model_config={
            "model_type": "technical_support",
            "temperature": 0.6,
            "max_tokens": 300
        }
    )
    
    # Register technical tools
    async def run_diagnostics(system: str) -> str:
        return f"Diagnostics completed for {system}. Status: OK"
    
    async def check_system_logs(service: str) -> str:
        return f"Recent logs for {service}: No errors found"
    
    agent.register_tool("run_diagnostics", run_diagnostics)
    agent.register_tool("check_system_logs", check_system_logs)
    
    return agent

# Usage
agent = await create_technical_support_agent()
context = await agent.start_conversation()

response = await agent.generate_multiturn_response(
    context.conversation_id,
    "My API is returning 500 errors",
    strategy="technical_support"
)
```

## Advanced Features

### 1. Distributed Training

```python
from grpo_agent_framework.training.distributed_trainer import (
    DistributedGRPOTrainer,
    TrainingConfig,
    DistributedConfig
)

# Configure distributed training
training_config = TrainingConfig(
    num_episodes=1000,
    num_generations=16,
    use_lora=True,
    bf16=True,
    run_post_eval=True
)

distributed_config = DistributedConfig(
    world_size=4,
    mixed_precision=True,
    gradient_accumulation_steps=2
)

# Create distributed trainer
trainer = DistributedGRPOTrainer(
    agent=agent,
    environment=environment,
    reward_function=reward_func,
    training_config=training_config,
    distributed_config=distributed_config
)

# Train across multiple GPUs
trained_agent = await trainer.train()
```

### 2. Real-Time Monitoring

```python
from grpo_agent_framework.utils.monitoring import MonitoringService

# Initialize monitoring
monitoring = MonitoringService()

# Log metrics during training
await monitoring.log_metric("response_time", 0.45)
await monitoring.log_metric("quality_score", 0.87)

# Log events
await monitoring.log_event("user_interaction", {
    "user_id": "user123",
    "satisfaction": "high",
    "resolved": True
})

# Get comprehensive metrics
metrics = await monitoring.get_metrics()
```

### 3. Caching and Performance

```python
from grpo_agent_framework.utils.cache import CacheService

# Initialize cache
cache = CacheService()

# Use cache with agent
agent = MultiTurnAgent(
    model_config=model_config,
    cache_service=cache
)

# Cache automatically speeds up repeated operations
```

## Training Strategies

### 1. Computational Training

```python
# Example computational training workflow
async def run_computational_training():
    # Create components
    agent = MultiTurnAgent(model_config)
    environment = MyEnvironment()
    reward_func = create_customer_service_reward()
    
    # Create computational engine
    engine = create_computational_engine(
        agent=agent,
        environment=environment, 
        reward_function=reward_func,
        num_workers=8  # Scale computation
    )
    
    # Training prompts
    prompts = [
        "Hello, I need help with my account",
        "I want to cancel my subscription",
        "My order hasn't arrived yet",
        "I need to update my billing information"
    ]
    
    # Run multiple training iterations
    results = []
    for i in range(10):
        result = await engine.train_iteration(prompts)
        results.append(result)
        print(f"Iteration {i+1}: Avg reward = {result['average_reward']:.3f}")
    
    return results
```

### 2. Mixed Training Approach

```python
# Combine multiple reward functions
async def create_mixed_reward_system():
    # Multi-objective reward
    multi_reward = create_customer_service_reward(weight=0.4)
    
    # Neural reward
    neural_reward = create_neural_reward_function(weight=0.3)
    
    # RULER reward (if API available)
    try:
        ruler_reward = create_customer_service_ruler(weight=0.3)
        reward_functions = [multi_reward, neural_reward, ruler_reward]
    except Exception:
        reward_functions = [multi_reward, neural_reward]
    
    return reward_functions
```

## Production Deployment

### 1. API Service

```python
from grpo_agent_framework.api.ultimate_grpo_service import app
import uvicorn

# Run the production API
if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8001,
        workers=4
    )
```

### 2. Docker Deployment

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY grpo_agent_framework/ grpo_agent_framework/

EXPOSE 8001

CMD ["python", "-m", "grpo_agent_framework.api.ultimate_grpo_service"]
```

### 3. Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: grpo-service
spec:
  replicas: 3
  selector:
    matchLabels:
      app: grpo-service
  template:
    metadata:
      labels:
        app: grpo-service
    spec:
      containers:
      - name: grpo-service
        image: grpo-agent-framework:latest
        ports:
        - containerPort: 8001
        env:
        - name: WORKERS
          value: "4"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

## API Reference

### REST API Endpoints

#### Training

```bash
# Start training job
curl -X POST http://localhost:8001/api/train \
  -H "Content-Type: application/json" \
  -d '{
    "prompts": ["Hello", "I need help"],
    "strategy": "computational",
    "num_iterations": 5,
    "use_neural_rewards": true
  }'

# Check training status
curl http://localhost:8001/api/status/JOB_ID
```

#### Conversations

```bash
# Start conversation
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Hello, I need help",
    "strategy": "customer_service",
    "context": {"priority": "high"}
  }'

# Continue conversation
curl -X POST http://localhost:8001/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "My order is delayed",
    "conversation_id": "CONVERSATION_ID",
    "strategy": "customer_service"
  }'
```

#### Metrics

```bash
# Get system metrics
curl http://localhost:8001/api/metrics

# Health check
curl http://localhost:8001/health
```

### WebSocket API

```javascript
const ws = new WebSocket('ws://localhost:8001/ws');

// Send chat message
ws.send(JSON.stringify({
  type: 'chat',
  data: {
    message: 'Hello, I need help',
    strategy: 'customer_service'
  }
}));

// Handle responses
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  if (data.type === 'chat_response') {
    console.log('Agent:', data.data.response);
  }
};
```

## Best Practices

### 1. Agent Design

- **Start Simple**: Begin with basic conversation scenarios
- **Iterate Gradually**: Add complexity incrementally
- **Use Domain Knowledge**: Leverage dialogue databases for your domain
- **Monitor Performance**: Track metrics continuously

### 2. Reward Function Design

- **Multi-Objective**: Use composite rewards for balanced behavior
- **Domain-Specific**: Create rewards tailored to your use case
- **Continuous Learning**: Leverage neural rewards for self-improvement
- **Safety First**: Always include safety and appropriateness checks

### 3. Training Strategy

- **Computational Focus**: Use massive computation over hand-crafted rules
- **Parallel Processing**: Leverage multiple workers for faster training
- **Distributed Training**: Scale to multiple GPUs for large models
- **Regular Evaluation**: Monitor training progress and adjust as needed

### 4. Production Deployment

- **Monitoring**: Set up comprehensive metrics and alerting
- **Caching**: Use caching for improved performance
- **Scaling**: Design for horizontal scaling
- **Fault Tolerance**: Handle failures gracefully

## Troubleshooting

### Common Issues

#### Low Training Performance
```python
# Check computational resource usage
metrics = engine.get_metrics()
print(f"Workers active: {metrics['active_workers']}")
print(f"Trajectories/sec: {metrics['engine_metrics']['trajectories_per_second']}")

# Scale up if needed
engine.scale_computation(2.0)  # Double computational resources
```

#### Memory Issues
```python
# Reduce context length
agent = MultiTurnAgent(
    model_config=model_config,
    max_context_length=1024,  # Reduce from 2048
    max_conversation_turns=10  # Reduce from 20
)
```

#### Conversation Context Lost
```python
# Check conversation status
active_conversations = agent.get_active_conversations()
print(f"Active conversations: {len(active_conversations)}")

# Get conversation summary
summary = agent.get_conversation_summary(conversation_id)
print(f"Conversation summary: {summary}")
```

#### Training Instability
```python
# Use more conservative training settings
training_config = TrainingConfig(
    learning_rate=1e-5,  # Lower learning rate
    gradient_clipping=0.5,  # Add gradient clipping
    early_stopping=True,
    patience=50
)
```

### Performance Optimization

#### Scaling Computation
```python
# Check current performance
metrics = engine.get_metrics()
current_tps = metrics['engine_metrics']['trajectories_per_second']

# Scale up if throughput is low
if current_tps < 10:
    engine.scale_computation(2.0)
    print("Scaled up computational resources")
```

#### Memory Usage
```python
# Monitor memory usage
import psutil
memory_usage = psutil.virtual_memory().percent
print(f"Memory usage: {memory_usage}%")

# Enable context compression if memory is high
if memory_usage > 80:
    agent.context_compression_threshold = 0.6  # More aggressive compression
```

### Debug Mode

```python
import logging

# Enable debug logging
logging.basicConfig(level=logging.DEBUG)

# Add detailed logging to agent
agent = MultiTurnAgent(
    model_config=model_config,
    debug_mode=True  # Enable detailed logging
)
```

## Examples Repository

The framework includes comprehensive examples in the `examples/` directory:

- `grpo_showcase.py`: Complete demonstration of all features
- `ultimate_customer_service_demo.py`: Production-ready customer service agent
- `quick_start.py`: Simple getting started example
- `enhanced_customer_service.py`: Advanced customer service with all innovations

Run examples:
```bash
python -m grpo_agent_framework.examples.grpo_showcase
python -m grpo_agent_framework.examples.ultimate_customer_service_demo
```

## Support and Community

- **Documentation**: [Framework Documentation](https://grpo-framework.readthedocs.io)
- **Examples**: Check the `examples/` directory for complete working examples
- **Issues**: Report bugs and request features on GitHub
- **Discussions**: Join the community discussions for help and best practices

## Philosophy

The GRPO Agent Framework embodies the principle that **computation is the key to long-term improvement**. Rather than spending time on hand-crafted rules and heuristics, the framework leverages massive computational resources to discover optimal behaviors through data-driven learning.

This approach, inspired by the "Bitter Lesson" in AI research, focuses on:
- **Scalable Algorithms**: Methods that improve with more computation
- **Data-Driven Learning**: Learning from large amounts of interaction data
- **Minimal Human Bias**: Reducing hand-crafted constraints and rules
- **Continuous Improvement**: Self-improving systems that get better over time

Happy building! 🚀