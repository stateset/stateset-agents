# Kimi-K2.5 Integration Guide

This guide explains how to use Moonshot AI's Kimi-K2.5 model with the StateSet GRPO Agent Framework for RL training.

## Overview

**Model**: `moonshotai/Kimi-K2.5`
- **Architecture**: Mixture-of-Experts (MoE)
- **Total Parameters**: 1T
- **Activated Parameters**: 32B
- **Type**: Native Multimodal (Vision + Text)
- **Context Length**: 256K
- **Special Features**: Thinking mode, agent swarming, multi-step tool calling

## Requirements

```bash
pip install -e ".[training]"
```

Ensure you have:
- Python 3.8+
- PyTorch 2.0+
- Transformers >= 4.57.1
- PEFT (for LoRA)
- CUDA for training

## Quick Start

### Basic Training

Train Kimi-K2.5 as a conversational agent:

```bash
python examples/finetune_kimi25_gspo.py \
  --model moonshotai/Kimi-K2.5 \
  --task customer_service \
  --use-lora
```

### With Weights & Biases

```bash
python examples/finetune_kimi25_gspo.py \
  --model moonshotai/Kimi-K2.5 \
  --task technical_support \
  --use-lora \
  --wandb \
  --wandb-project kimi25-rl-training
```

### Custom Configuration

```python
from stateset_agents.training.kimi25_config import get_kimi25_config

config = get_kimi25_config(
    model_name="moonshotai/Kimi-K2.5",
    task="customer_service",
    use_lora=True,
    output_dir="./outputs/kimi25_gspo"
)
```

## Configuration

### Automatic Configuration

The framework provides optimized configurations for Kimi-K2.5 based on its MoE architecture:

- **Group Size**: 8 (larger for MoE stability)
- **Learning Rate**: 3e-6 (conservative for large models)
- **LoRA Rank**: 64 (high rank for 32B activated params)
- **Clipping Range**: [1.5e-4, 2.5e-4] (tighter for stability)

### Manual Configuration

```python
from stateset_agents.training.config import TrainingConfig

config = TrainingConfig(
    model_name="moonshotai/Kimi-K2.5",
    use_lora=True,
    lora_r=64,
    lora_alpha=128,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_generations=8,
    learning_rate=3e-6,
    max_grad_norm=1.0
)
```

## Kimi-K2.5 Specific Features

### Thinking Mode

Kimi-K2.5 supports a "thinking" mode for enhanced reasoning:

```python
from stateset_agents.core.agent import AgentConfig

agent_config = AgentConfig(
    model_name="moonshotai/Kimi-K2.5",
    system_prompt="You are Kimi, an AI assistant created by Moonshot AI.",
    # Enable thinking mode via chat template kwargs
    model_kwargs={
        "chat_template_kwargs": {"thinking": True}
    },
    max_new_tokens=4096
)
```

### Multimodal Support

Kimi-K2.5 natively supports visual inputs:

```python
messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "Describe this image."},
            {"type": "image_url", "image_url": {"url": "..."}}
        ]
    }
]
```

Note: Multimodal training requires custom reward functions that can handle visual inputs.

## Training Scenarios

### Customer Service Agent

```bash
python examples/finetune_kimi25_gspo.py \
  --model moonshotai/Kimi-K2.5 \
  --task customer_service \
  --use-lora \
  --output-dir ./outputs/kimi25_customer_service
```

### Technical Support Agent

```bash
python examples/finetune_kimi25_gspo.py \
  --model moonshotai/Kimi-K2.5 \
  --task technical_support \
  --use-lora \
  --output-dir ./outputs/kimi25_tech_support
```

### Sales Assistant

```bash
python examples/finetune_kimi25_gspo.py \
  --model moonshotai/Kimi-K2.5 \
  --task sales \
  --use-lora \
  --output-dir ./outputs/kimi25_sales
```

## Advanced Features

### Continual Learning

```python
from stateset_agents.training.config import TrainingConfig

config = TrainingConfig(
    model_name="moonshotai/Kimi-K2.5",
    continual_strategy="replay_lwf",
    replay_buffer_size=2000,
    replay_ratio=0.5,
    kl_penalty_beta=0.1
)
```

### Custom Reward Functions

Create custom rewards for Kimi-K2.5:

```python
from stateset_agents.rewards.multi_objective_reward import MultiObjectiveReward

class KimiReward(MultiObjectiveReward):
    async def compute_reward(self, trajectory):
        # Custom reward logic
        return reward
```

## Performance Tips

1. **Memory Optimization**: Use LoRA with rank 64 for balanced performance
2. **Batch Training**: Use `per_device_train_batch_size=1` with `gradient_accumulation_steps=16`
3. **Stability**: Larger group sizes (8) help with MoE model stability
4. **Learning Rate**: Conservative rates (3e-6) work best for 1T parameter models

## Troubleshooting

### Out of Memory

```bash
python examples/finetune_kimi25_gspo.py \
  --model moonshotai/Kimi-K2.5 \
  --use-lora \
  --use-4bit  # Not yet supported, will use gradient checkpointing
```

### Slow Training

- Increase `rollout_concurrency` for parallel generation
- Use vLLM for faster inference (set `use_vllm=True`)
- Reduce `num_generations` for faster iterations

### Poor Performance

- Increase `num_outer_iterations`
- Adjust `learning_rate` (try 1e-6 to 5e-6)
- Increase `per_device_train_batch_size` if memory allows

## API Deployment

Fine-tuned models can be deployed using Moonshot AI's API or open-source inference engines:

- **vLLM**: Fast production deployment
- **SGLang**: Efficient serving
- **KTransformers**: Advanced optimization

See [Kimi-K2.5 Documentation](https://huggingface.co/moonshotai/Kimi-K2.5) for deployment guides.

## References

- [Kimi-K2.5 Model Card](https://huggingface.co/moonshotai/Kimi-K2.5)
- [Moonshot AI Platform](https://platform.moonshot.ai)
- [GSPO Paper](https://arxiv.org/abs/2507.18071v2)

## License

Kimi-K2.5 is released under the Modified MIT License. See the [model license](https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/LICENSE) for details.