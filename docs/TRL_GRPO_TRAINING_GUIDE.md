# TRL GRPO Training Guide

This guide explains how to use TRL's GRPO (Group Relative Policy Optimization) trainer to fine-tune the `openai/gpt-oss-120b` model within the GRPO Agent Framework.

## Overview

The TRL GRPO trainer integrates Hugging Face's TRL library with our framework, providing:
- Efficient fine-tuning using LoRA adapters
- Multi-objective reward optimization
- Memory-efficient training for large models
- Seamless integration with existing framework components

## Prerequisites

Install required dependencies:

```bash
pip install torch transformers peft datasets trl wandb accelerate
```

## Quick Start

### 1. Using the Convenience Function

The easiest way to train a customer service agent:

```python
import asyncio
from stateset_agents.training import train_customer_service_with_trl

async def main():
    agent = await train_customer_service_with_trl(
        model_name="openai/gpt-oss-120b",
        num_episodes=1000,
        output_dir="./outputs/my_agent",
        use_lora=True,
        lora_r=16
    )
    
    # Test the trained agent
    response = await agent.generate_response(
        messages=[{"role": "user", "content": "Where is my order?"}]
    )
    print(response)

asyncio.run(main())
```

### 2. Using the Training Script

Run the provided example script:

```bash
# Quick demo mode (minimal resources)
QUICK_MODE=true python examples/train_with_trl_grpo.py

# Full training
python examples/train_with_trl_grpo.py
```

### 3. Using the Production Launch Script

For production training with optimized settings:

```bash
# Run with default settings
./scripts/train_trl_grpo.sh

# Run in quick mode for testing
QUICK_MODE=true ./scripts/train_trl_grpo.sh

# Custom configuration
MODEL_NAME="openai/gpt-oss-120b" \
NUM_EPISODES=2000 \
BATCH_SIZE=2 \
LEARNING_RATE=3e-6 \
./scripts/train_trl_grpo.sh
```

## Advanced Usage

### Custom Configuration

Create a custom TRL GRPO configuration:

```python
from stateset_agents.training import TRLGRPOConfig, train_with_trl_grpo
from stateset_agents.core.agent import MultiTurnAgent, AgentConfig
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import MultiObjectiveReward

# Create configuration
config = TRLGRPOConfig(
    model_name="openai/gpt-oss-120b",
    output_dir="./outputs/custom_training",
    num_episodes=1500,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=5e-6,
    
    # TRL GRPO specific
    beta=0.01,  # Small KL penalty
    num_generations=4,
    num_iterations=1,
    
    # LoRA settings
    use_lora=True,
    lora_r=32,
    lora_alpha=64,
    lora_dropout=0.1,
    
    # Memory optimization
    gradient_checkpointing=True,
    bf16=True,
    
    # Generation parameters
    temperature=0.8,
    top_p=0.95,
    max_prompt_length=256,
    max_completion_length=256
)

# Create agent
agent = MultiTurnAgent(AgentConfig(
    model_name=config.model_name,
    system_prompt="You are a helpful assistant."
))
await agent.initialize()

# Create environment and reward model
environment = ConversationEnvironment(...)
reward_model = MultiObjectiveReward(...)

# Run training
trained_agent = await train_with_trl_grpo(
    config=config,
    agent=agent,
    environment=environment,
    reward_model=reward_model,
    train_data=my_training_data  # Optional
)
```

### Custom Training Data

Prepare training data in the required format:

```python
training_data = [
    {
        "messages": [
            {"role": "user", "content": "User query here"},
            {"role": "assistant", "content": "Assistant response here"}
        ]
    },
    # More conversations...
]
```

Load from JSONL file:

```jsonl
{"messages": [{"role": "user", "content": "Hello"}, {"role": "assistant", "content": "Hi there!"}]}
{"messages": [{"role": "user", "content": "Help me"}, {"role": "assistant", "content": "I'd be happy to help!"}]}
```

## Configuration Options

### Key Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `model_name` | Model to fine-tune | `openai/gpt-oss-120b` | - |
| `learning_rate` | Learning rate | `5e-6` | `1e-6` to `1e-5` |
| `batch_size` | Per-device batch size | `1` | `1-4` (depends on GPU) |
| `gradient_accumulation_steps` | Gradient accumulation | `16` | `8-32` |
| `num_generations` | Responses per prompt | `4` | `2-8` |
| `beta` | KL penalty coefficient | `0.0` | `0.0-0.1` |
| `lora_r` | LoRA rank | `16` | `8-64` |
| `lora_alpha` | LoRA alpha | `32` | `16-128` |

### Memory Optimization

For large models like `openai/gpt-oss-120b`:

```python
config = TRLGRPOConfig(
    # Enable all memory optimizations
    gradient_checkpointing=True,
    bf16=True,  # or fp16=True
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    
    # Use smaller LoRA rank
    lora_r=8,
    lora_alpha=16,
    
    # Limit sequence lengths
    max_prompt_length=128,
    max_completion_length=128,
    
    # Use 8-bit quantization (optional)
    use_8bit=True  # Requires bitsandbytes
)
```

## Environment Variables

The training scripts support these environment variables:

```bash
# Model configuration
export MODEL_NAME="openai/gpt-oss-120b"

# Training parameters
export NUM_EPISODES=1000
export BATCH_SIZE=1
export LEARNING_RATE=5e-6
export GRADIENT_ACCUMULATION_STEPS=16

# LoRA configuration
export USE_LORA=true
export LORA_R=16
export LORA_ALPHA=32

# GRPO parameters
export BETA=0.0
export NUM_GENERATIONS=4

# Memory optimization
export GRADIENT_CHECKPOINTING=true
export USE_BF16=true

# Output and logging
export OUTPUT_DIR="./outputs/my_training"
export WANDB_PROJECT="my-grpo-project"
```

## Monitoring Training

### Weights & Biases Integration

Training automatically logs to W&B if configured:

```bash
# Set your W&B API key
export WANDB_API_KEY="your-api-key"

# Configure project
export WANDB_PROJECT="grpo-training"
export WANDB_ENTITY="your-entity"
```

View metrics:
- Loss curves
- Reward trends
- Generation samples
- Learning rate schedule

### Local Monitoring

Check training progress:

```bash
# View real-time logs
tail -f outputs/trl_grpo_training/*/training_*.log

# Check GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
htop
```

## Troubleshooting

### Out of Memory (OOM)

If you encounter OOM errors:

1. Reduce batch size:
   ```bash
   export BATCH_SIZE=1
   export GRADIENT_ACCUMULATION_STEPS=32
   ```

2. Enable more aggressive memory optimization:
   ```python
   config = TRLGRPOConfig(
       gradient_checkpointing=True,
       use_8bit=True,  # or use_4bit=True
       lora_r=4,  # Smaller LoRA rank
       max_prompt_length=64,
       max_completion_length=64
   )
   ```

3. Use CPU offloading:
   ```python
   config = TRLGRPOConfig(
       device_map="auto",
       offload_folder="./offload"
   )
   ```

### Slow Training

Speed up training:

1. Use mixed precision:
   ```python
   config = TRLGRPOConfig(bf16=True)  # or fp16=True
   ```

2. Reduce number of generations:
   ```python
   config = TRLGRPOConfig(num_generations=2)
   ```

3. Use larger batch size if memory allows:
   ```python
   config = TRLGRPOConfig(
       per_device_train_batch_size=2,
       gradient_accumulation_steps=8
   )
   ```

## Best Practices

1. **Start Small**: Test with `QUICK_MODE=true` first
2. **Monitor Memory**: Keep GPU memory usage below 90%
3. **Save Checkpoints**: Set appropriate `save_steps`
4. **Validate Config**: Check warnings from `config.validate()`
5. **Use LoRA**: Essential for large models like `openai/gpt-oss-120b`
6. **Gradual Scaling**: Start with small datasets and scale up

## Example Results

After training, evaluate your model:

```python
# Load trained model
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./outputs/final_model")
tokenizer = AutoTokenizer.from_pretrained("./outputs/final_model")

# Test generation
inputs = tokenizer("User: Where is my order?\nAssistant:", return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=100)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)
```

## Next Steps

- Experiment with different reward functions
- Try multi-task training with diverse datasets
- Fine-tune hyperparameters for your use case
- Deploy trained models using the framework's serving capabilities

For more details, see the [main documentation](README.md) and [examples](examples/). 
