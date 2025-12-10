# Advanced Training Examples for StateSet Agents

This directory contains advanced training examples demonstrating state-of-the-art techniques for training conversational AI agents with GRPO (Group Relative Policy Optimization).

## 📚 Table of Contents

- [Overview](#overview)
- [Examples](#examples)
  - [1. Distributed Multi-GPU Training](#1-distributed-multi-gpu-training)
  - [2. Custom Reward Functions](#2-custom-reward-functions)
  - [3. Advanced Optimization Techniques](#3-advanced-optimization-techniques)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Performance Tips](#performance-tips)

---

## Overview

These examples demonstrate production-ready training techniques including:

- **Distributed Training**: Scale training across multiple GPUs and machines
- **Custom Rewards**: Create domain-specific reward functions for your use case
- **Optimization**: Advanced techniques for efficient and stable training

Each example is self-contained and includes comprehensive documentation.

---

## Examples

### 1. Distributed Multi-GPU Training

**File**: `distributed_multi_gpu_training.py`

Demonstrates how to train agents across multiple GPUs using PyTorch Distributed Data Parallel (DDP).

#### Features

- ✅ Multi-GPU and multi-node training support
- ✅ Automatic data distribution across processes
- ✅ Gradient synchronization and accumulation
- ✅ Mixed precision training (FP16/BF16)
- ✅ Distributed checkpointing
- ✅ Monitoring and logging (W&B integration)
- ✅ Memory optimization (gradient checkpointing, CPU offloading)

#### Usage

```bash
# Single machine, 4 GPUs
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    examples/distributed_multi_gpu_training.py \
    --model gpt2 \
    --task customer_service \
    --num-episodes 200

# Memory-efficient training for large models
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    examples/distributed_multi_gpu_training.py \
    --model gpt2-large \
    --use-lora \
    --checkpoint-activations \
    --grad-accumulation 4

# Multi-node training (run on each node)
# Node 0:
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    examples/distributed_multi_gpu_training.py

# Node 1:
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    --nnodes=2 \
    --node_rank=1 \
    --master_addr="192.168.1.1" \
    --master_port=12355 \
    examples/distributed_multi_gpu_training.py
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--model` | Model name or path | `gpt2` |
| `--task` | Training task (customer_service, technical_support, sales) | `customer_service` |
| `--num-episodes` | Total training episodes (divided across GPUs) | `100` |
| `--batch-size` | Batch size per GPU | `8` |
| `--use-lora` | Enable LoRA for parameter-efficient training | `False` |
| `--no-mixed-precision` | Disable mixed precision | `False` |
| `--checkpoint-activations` | Enable gradient checkpointing | `False` |
| `--cpu-offload` | Offload optimizer to CPU | `False` |
| `--grad-accumulation` | Gradient accumulation steps | `1` |

#### Expected Performance

| Setup | Training Speed | Memory Usage |
|-------|----------------|--------------|
| 1x GPU (baseline) | 1.0x | 10GB |
| 4x GPU | 3.8x | 10GB per GPU |
| 8x GPU | 7.5x | 10GB per GPU |
| 4x GPU + LoRA | 3.8x | 5GB per GPU |
| 4x GPU + Gradient Checkpointing | 3.5x | 6GB per GPU |

---

### 2. Custom Reward Functions

**File**: `custom_reward_functions.py`

Comprehensive guide to creating custom reward functions for training agents with domain-specific objectives.

#### Features

- ✅ Simple custom rewards (length, structure)
- ✅ Domain-specific rewards (empathy, action-oriented, Q&A)
- ✅ Neural reward models (sentiment analysis, toxicity detection)
- ✅ Composite rewards with weighted combinations
- ✅ Curriculum learning with adaptive rewards
- ✅ Testing and evaluation utilities

#### Included Reward Functions

##### Basic Rewards

1. **ResponseLengthReward**: Rewards responses in optimal length range
2. **QuestionAnsweringReward**: Rewards direct answers to questions
3. **EmpathyReward**: Rewards empathetic language in customer service
4. **ActionOrientedReward**: Rewards clear next steps and action items
5. **SentimentReward**: Uses sentiment analysis for tone scoring

##### Composite Rewards

1. **Balanced Customer Service**: Empathy (30%) + Action (25%) + Q&A (25%) + Length (10%) + Sentiment (10%)
2. **Technical Support**: Action (40%) + Q&A (35%) + Length (15%) + Sentiment (10%)
3. **Empathetic Agent**: Empathy (50%) + Sentiment (25%) + Q&A (15%) + Length (10%)

##### Advanced Rewards

1. **CurriculumReward**: Adaptive difficulty progression over training

#### Usage

```bash
# Test all reward functions on sample conversations
python examples/custom_reward_functions.py

# Test specific reward type
python examples/custom_reward_functions.py --reward-type empathy

# List available rewards
python examples/custom_reward_functions.py --list-rewards
```

#### Creating Your Own Reward Function

```python
from stateset_agents.core.reward import RewardFunction, RewardResult, RewardType
from stateset_agents.core.trajectory import ConversationTurn
from typing import Any, Dict, List, Optional

class MyCustomReward(RewardFunction):
    """
    Custom reward function for your specific use case
    """

    def __init__(self, weight: float = 1.0, **kwargs):
        super().__init__(weight, RewardType.IMMEDIATE, "MyCustomReward")
        # Initialize your parameters

    async def compute_reward(
        self,
        turns: List[ConversationTurn],
        context: Optional[Dict[str, Any]] = None
    ) -> RewardResult:
        """Compute reward for conversation turns"""

        # Extract assistant responses
        assistant_turns = [t for t in turns if t.role == "assistant"]

        # Compute your custom reward logic
        score = 0.0
        breakdown = {}

        for i, turn in enumerate(assistant_turns):
            # Your scoring logic here
            turn_score = self._evaluate_turn(turn.content)
            score += turn_score
            breakdown[f"turn_{i}_score"] = turn_score

        avg_score = score / len(assistant_turns) if assistant_turns else 0.0

        return RewardResult(
            score=avg_score,
            breakdown=breakdown,
            metadata={"num_turns": len(assistant_turns)}
        )

    def _evaluate_turn(self, content: str) -> float:
        """Your evaluation logic"""
        # Implement your scoring criteria
        return 0.5  # Return score between 0 and 1
```

#### Using in Training

```python
from stateset_agents.core.environment import ConversationEnvironment

# Create your custom reward
reward_fn = MyCustomReward(weight=1.0)

# Or create a composite reward
from stateset_agents.core.reward import CompositeReward

reward_fn = CompositeReward([
    MyCustomReward(weight=0.5),
    EmpathyReward(weight=0.3),
    SentimentReward(weight=0.2),
])

# Use in environment
environment = ConversationEnvironment(
    scenarios=scenarios,
    max_turns=6,
    reward_fn=reward_fn,
)
```

---

### 3. Advanced Optimization Techniques

**File**: `advanced_optimization_techniques.py`

Demonstrates cutting-edge optimization techniques for efficient and stable training.

#### Features

- ✅ Mixed precision training (FP16/BF16)
- ✅ Advanced optimizers (AdamW, Adafactor, Lion)
- ✅ Learning rate scheduling (Cosine, Warmup, Plateau)
- ✅ Gradient accumulation for larger effective batch sizes
- ✅ Gradient clipping and normalization
- ✅ Model compilation with `torch.compile` (PyTorch 2.0+)
- ✅ Memory optimization techniques
- ✅ Performance monitoring and profiling

#### Supported Optimizers

1. **AdamW**: Adam with weight decay fix (default, recommended)
2. **Adafactor**: Memory-efficient adaptive learning rate
3. **Lion**: Recent optimizer from Google (fast convergence)
4. **SGD**: Classic with momentum

#### Supported LR Schedulers

1. **Cosine Annealing**: Smooth decay with warmup
2. **ReduceLROnPlateau**: Adaptive based on metrics
3. **Constant with Warmup**: Linear warmup then constant

#### Usage

```bash
# Basic training with all optimizations
python examples/advanced_optimization_techniques.py \
    --model gpt2 \
    --mixed-precision bf16 \
    --optimizer adamw

# Memory-efficient training for large models
python examples/advanced_optimization_techniques.py \
    --model gpt2-large \
    --gradient-checkpointing \
    --grad-accumulation 8 \
    --mixed-precision fp16 \
    --optimizer adafactor

# Maximum performance with compilation
python examples/advanced_optimization_techniques.py \
    --model gpt2 \
    --compile \
    --compile-mode max-autotune \
    --mixed-precision bf16 \
    --optimizer adamw

# Training with Lion optimizer and cosine schedule
python examples/advanced_optimization_techniques.py \
    --model gpt2 \
    --optimizer lion \
    --scheduler cosine \
    --warmup-ratio 0.1 \
    --learning-rate 1e-4
```

#### Key Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--optimizer` | Optimizer type (adamw, adafactor, lion, sgd) | `adamw` |
| `--learning-rate` | Learning rate | `5e-5` |
| `--scheduler` | LR scheduler (cosine, plateau, constant) | `cosine` |
| `--warmup-ratio` | Warmup ratio (0.0-1.0) | `0.1` |
| `--mixed-precision` | Mixed precision (none, fp16, bf16) | `none` |
| `--grad-accumulation` | Gradient accumulation steps | `1` |
| `--max-grad-norm` | Maximum gradient norm for clipping | `1.0` |
| `--gradient-checkpointing` | Enable gradient checkpointing | `False` |
| `--compile` | Enable torch.compile | `False` |
| `--compile-mode` | Compilation mode | `default` |

#### Performance Comparison

| Configuration | Speed | Memory | Stability |
|---------------|-------|--------|-----------|
| FP32 (baseline) | 1.0x | 100% | ⭐⭐⭐⭐⭐ |
| FP16 + GradScaler | 2.2x | 50% | ⭐⭐⭐⭐ |
| BF16 | 2.0x | 50% | ⭐⭐⭐⭐⭐ |
| + torch.compile | 2.8x | 50% | ⭐⭐⭐⭐⭐ |
| + Gradient Checkpointing | 1.8x | 30% | ⭐⭐⭐⭐⭐ |

#### Recommendations by Model Size

| Model Size | Recommended Configuration |
|------------|---------------------------|
| **Small** (<1B params) | `--mixed-precision bf16 --compile` |
| **Medium** (1B-7B params) | `--mixed-precision bf16 --grad-accumulation 4 --gradient-checkpointing` |
| **Large** (7B+ params) | `--mixed-precision fp16 --grad-accumulation 8 --gradient-checkpointing --optimizer adafactor` |

---

## Prerequisites

### Required

```bash
pip install stateset-agents[dev]
```

### Optional

For neural reward models:
```bash
pip install transformers textblob
```

For Lion optimizer:
```bash
pip install lion-pytorch
```

For distributed training:
```bash
# PyTorch with NCCL support (included in most PyTorch installations)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## Quick Start

### 1. Test Custom Rewards

```bash
# See all available custom rewards
python examples/custom_reward_functions.py --list-rewards

# Test empathy reward on sample conversations
python examples/custom_reward_functions.py --reward-type empathy
```

### 2. Train with Optimizations

```bash
# Single GPU with all optimizations
python examples/advanced_optimization_techniques.py \
    --model gpt2 \
    --mixed-precision bf16 \
    --compile
```

### 3. Scale to Multiple GPUs

```bash
# 4 GPUs with automatic distribution
python -m torch.distributed.launch \
    --nproc_per_node=4 \
    examples/distributed_multi_gpu_training.py \
    --model gpt2 \
    --task customer_service
```

---

## Performance Tips

### Memory Optimization

1. **Enable Mixed Precision**: Use BF16 for best stability, FP16 for maximum memory savings
   ```bash
   --mixed-precision bf16
   ```

2. **Enable Gradient Checkpointing**: Trades compute for memory (30-40% memory reduction)
   ```bash
   --gradient-checkpointing
   ```

3. **Use LoRA**: Dramatically reduces memory for fine-tuning
   ```bash
   --use-lora
   ```

4. **Increase Gradient Accumulation**: Simulate larger batches without memory increase
   ```bash
   --grad-accumulation 4
   ```

### Speed Optimization

1. **Use torch.compile**: 20-40% speedup on PyTorch 2.0+
   ```bash
   --compile --compile-mode max-autotune
   ```

2. **Choose Right Optimizer**:
   - AdamW: Best default choice
   - Lion: Faster convergence, use 3x smaller LR
   - Adafactor: Memory efficient but slower

3. **Optimize Learning Rate Schedule**:
   - Cosine: Smooth convergence
   - Warmup: Prevents early instability

### Stability Tips

1. **Use Gradient Clipping**: Prevents exploding gradients
   ```bash
   --max-grad-norm 1.0
   ```

2. **Start with Conservative Settings**:
   ```bash
   --learning-rate 5e-5 --warmup-ratio 0.1
   ```

3. **Monitor Training**: Check gradient norms and loss curves

---

## Troubleshooting

### Out of Memory (OOM)

1. Enable gradient checkpointing
2. Reduce batch size
3. Increase gradient accumulation
4. Use LoRA
5. Enable CPU offloading (slower but works)

### Training Instability

1. Reduce learning rate
2. Increase warmup steps
3. Check gradient clipping is enabled
4. Use BF16 instead of FP16
5. Verify your reward function returns values in [0, 1]

### Slow Training

1. Enable torch.compile
2. Use mixed precision (BF16)
3. Verify GPU utilization (`nvidia-smi`)
4. Check if gradient checkpointing is necessary
5. Profile with PyTorch Profiler

---

## Additional Resources

- [Main README](../README.md) - Project overview
- [GSPO Guide](../docs/GSPO_GUIDE.md) - Advanced GRPO variant
- [API Examples](./API_EXAMPLES_README.md) - API usage examples
- [Documentation](https://stateset-agents.readthedocs.io/) - Complete docs

---

## Support

- **Issues**: https://github.com/stateset/stateset-agents/issues
- **Discord**: https://discord.gg/stateset
- **Docs**: https://stateset-agents.readthedocs.io/

---

## License

See [LICENSE](../LICENSE) for details.
