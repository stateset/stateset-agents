# Performance Tuning Guide

This guide covers performance optimization strategies for StateSet Agents, including GPU memory management, batch sizing, distributed training, and inference optimization.

## Table of Contents

1. [GPU Memory Optimization](#gpu-memory-optimization)
2. [Batch Size Selection](#batch-size-selection)
3. [Training Speed Optimization](#training-speed-optimization)
4. [Inference Optimization](#inference-optimization)
5. [Distributed Training](#distributed-training)
6. [Profiling and Monitoring](#profiling-and-monitoring)
7. [Algorithm-Specific Tuning](#algorithm-specific-tuning)

---

## GPU Memory Optimization

### Memory-Efficient Techniques

#### 1. Gradient Checkpointing

Trades compute for memory by recomputing activations during backward pass:

```python
from stateset_agents.training.config import TrainingConfig

config = TrainingConfig(
    gradient_checkpointing=True,  # Reduces memory by ~60%
    gradient_accumulation_steps=4,  # Simulate larger batch sizes
)
```

**Impact:** ~60% memory reduction, ~20% slower training

#### 2. Mixed Precision Training

Use BF16 or FP16 for reduced memory and faster computation:

```python
config = TrainingConfig(
    bf16=True,  # Preferred for newer GPUs (A100, H100)
    # OR
    fp16=True,  # For older GPUs (V100, T4)
)
```

**Impact:** ~50% memory reduction, ~30% faster training

#### 3. Quantization

For inference and memory-constrained training:

```python
from stateset_agents.core.agent import AgentConfig

agent_config = AgentConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    use_8bit=True,   # 8-bit quantization
    # OR
    use_4bit=True,   # 4-bit quantization (more aggressive)
)
```

**Memory comparison for 7B model:**
| Precision | VRAM Required |
|-----------|---------------|
| FP32 | ~28 GB |
| FP16/BF16 | ~14 GB |
| 8-bit | ~7 GB |
| 4-bit | ~4 GB |

#### 4. LoRA (Low-Rank Adaptation)

Train only a small subset of parameters:

```python
config = TrainingConfig(
    use_lora=True,
    lora_r=16,        # Rank of adaptation matrices
    lora_alpha=32,    # Scaling factor
    lora_dropout=0.1,
    lora_target_modules=["q_proj", "v_proj"],  # Modules to adapt
)
```

**Impact:** ~90% parameter reduction, significantly lower memory

### Memory Estimation Formula

```
VRAM ≈ (model_params × bytes_per_param) +
        (batch_size × seq_len × hidden_dim × 4) +  # Activations
        (optimizer_states × model_params × bytes)   # Adam uses 2x
```

Quick estimation for common models:

| Model Size | FP16 Training | LoRA Training | Inference Only |
|------------|---------------|---------------|----------------|
| 1B | ~8 GB | ~4 GB | ~2 GB |
| 7B | ~56 GB | ~16 GB | ~14 GB |
| 13B | ~104 GB | ~28 GB | ~26 GB |
| 70B | ~560 GB | ~80 GB | ~140 GB |

---

## Batch Size Selection

### Effective Batch Size

```
effective_batch_size = per_device_batch_size × num_devices × gradient_accumulation_steps
```

### Recommendations by Model Size

```python
# Small models (< 1B params)
config = TrainingConfig(
    per_device_train_batch_size=16,
    gradient_accumulation_steps=1,
)

# Medium models (1B - 7B params)
config = TrainingConfig(
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
)

# Large models (> 7B params)
config = TrainingConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)
```

### Dynamic Batch Sizing

StateSet Agents supports automatic batch size finding:

```python
from stateset_agents.training.trainer import GRPOTrainer

trainer = GRPOTrainer(
    config=config,
    auto_find_batch_size=True,  # Automatically find optimal batch size
)
```

---

## Training Speed Optimization

### 1. Flash Attention

Enable for 2-4x faster attention computation:

```python
agent_config = AgentConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    attn_implementation="flash_attention_2",  # Requires flash-attn package
)
```

**Requirements:**
- GPU with compute capability >= 8.0 (A100, H100, RTX 3090+)
- Install: `pip install flash-attn --no-build-isolation`

### 2. Torch Compile

JIT compilation for faster execution:

```python
import torch

# Compile the model
model = torch.compile(model, mode="reduce-overhead")
```

**Modes:**
- `default`: Good balance of compilation time and speedup
- `reduce-overhead`: Faster inference, longer compilation
- `max-autotune`: Maximum optimization, longest compilation

### 3. Data Loading Optimization

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    dataset,
    batch_size=batch_size,
    num_workers=4,           # Parallel data loading
    pin_memory=True,         # Faster GPU transfer
    prefetch_factor=2,       # Prefetch batches
    persistent_workers=True, # Keep workers alive
)
```

### 4. Sequence Length Management

```python
config = TrainingConfig(
    max_prompt_length=512,      # Limit input length
    max_completion_length=256,  # Limit generation length
    truncation_side="left",     # Truncate beginning for context
)
```

**Rule of thumb:** Total sequence length (prompt + completion) should fit comfortably in memory with your batch size.

---

## Inference Optimization

### 1. KV Cache Optimization

```python
agent_config = AgentConfig(
    model_name="meta-llama/Llama-2-7b-hf",
    use_cache=True,  # Enable KV cache for faster generation
)
```

### 2. Batched Generation

```python
# Instead of generating one at a time:
responses = []
for prompt in prompts:
    response = await agent.generate(prompt)  # Slow
    responses.append(response)

# Batch multiple prompts:
responses = await agent.generate_batch(prompts, batch_size=8)  # Fast
```

### 3. Speculative Decoding

For models supporting it:

```python
agent_config = AgentConfig(
    model_name="meta-llama/Llama-2-70b-hf",
    assistant_model="meta-llama/Llama-2-7b-hf",  # Draft model
)
```

### 4. vLLM Integration

For high-throughput inference:

```python
from stateset_agents.core.agent_backends import create_vllm_backend

backend = create_vllm_backend(
    model_name="meta-llama/Llama-2-7b-hf",
    tensor_parallel_size=2,  # Split across GPUs
    max_num_seqs=256,        # Maximum concurrent sequences
)
```

---

## Distributed Training

### Multi-GPU Setup

StateSet Agents uses Hugging Face Accelerate for distribution:

```bash
# Configure accelerate
accelerate config

# Launch training
accelerate launch --num_processes 4 train.py
```

### DeepSpeed Integration

For large-scale training:

```python
# deepspeed_config.json
{
    "zero_optimization": {
        "stage": 2,  # ZeRO Stage 2
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": true
        }
    },
    "bf16": {
        "enabled": true
    },
    "gradient_clipping": 1.0
}
```

```python
config = TrainingConfig(
    deepspeed="deepspeed_config.json",
)
```

### ZeRO Stages

| Stage | Memory Savings | Communication | Use Case |
|-------|----------------|---------------|----------|
| ZeRO-1 | Optimizer states | Low | Multi-GPU, same node |
| ZeRO-2 | + Gradients | Medium | Multi-node |
| ZeRO-3 | + Parameters | High | Very large models |

---

## Profiling and Monitoring

### Built-in Monitoring

```python
from stateset_agents.utils.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()

# Training code here...

metrics = monitor.get_metrics()
print(f"GPU Memory Used: {metrics['gpu_memory_mb']} MB")
print(f"Throughput: {metrics['samples_per_second']} samples/sec")
```

### PyTorch Profiler

```python
import torch.profiler as profiler

with profiler.profile(
    activities=[
        profiler.ProfilerActivity.CPU,
        profiler.ProfilerActivity.CUDA,
    ],
    schedule=profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
    on_trace_ready=profiler.tensorboard_trace_handler('./logs'),
) as prof:
    # Training loop
    for batch in dataloader:
        train_step(batch)
        prof.step()
```

### Weights & Biases Integration

```python
config = TrainingConfig(
    report_to="wandb",
    wandb_project="my-project",
    logging_steps=10,
)
```

Key metrics to monitor:
- `train/loss`: Training loss
- `train/grad_norm`: Gradient magnitude
- `system/gpu_memory_allocated`: GPU memory usage
- `train/samples_per_second`: Training throughput

---

## Algorithm-Specific Tuning

### GRPO Tuning

```python
from stateset_agents.training.config import TrainingConfig

grpo_config = TrainingConfig(
    # Group size affects variance of advantage estimates
    num_generations=4,  # Larger = more stable, slower

    # Clipping for policy updates
    cliprange=0.2,  # Standard PPO clipping

    # KL penalty
    kl_coef=0.1,  # Regularization strength

    # Advantage normalization
    normalize_advantages=True,
)
```

### GSPO Tuning

```python
from stateset_agents.training.gspo_trainer import GSPOConfig

gspo_config = GSPOConfig(
    # Sequence-level clipping (tighter than GRPO)
    clip_range_left=3e-4,
    clip_range_right=4e-4,

    # Group size
    num_generations=4,  # 4-8 recommended

    # No KL penalty needed typically
    beta=0.0,
)
```

### PPO vs GRPO vs GSPO Performance

| Aspect | PPO | GRPO | GSPO |
|--------|-----|------|------|
| Memory | Higher (value head) | Medium | Medium |
| Stability | Good | Good | Best |
| Sample Efficiency | Baseline | +10-15% | +15-20% |
| Best For | General RL | Conversations | Long sequences |

---

## Quick Reference: Optimization Checklist

### Memory Constrained

- [ ] Enable `gradient_checkpointing=True`
- [ ] Use `bf16=True` or `fp16=True`
- [ ] Enable LoRA with `use_lora=True`
- [ ] Reduce `per_device_train_batch_size`
- [ ] Increase `gradient_accumulation_steps`
- [ ] Consider `use_8bit=True` or `use_4bit=True`

### Speed Constrained

- [ ] Enable Flash Attention 2
- [ ] Use `torch.compile()` for inference
- [ ] Increase `num_workers` in DataLoader
- [ ] Enable `pin_memory=True`
- [ ] Use multiple GPUs with Accelerate
- [ ] Consider vLLM for inference

### Stability Issues

- [ ] Reduce learning rate
- [ ] Increase group size (GRPO/GSPO)
- [ ] Enable gradient clipping
- [ ] Use learning rate warmup
- [ ] Check for NaN in rewards/losses

---

## Troubleshooting

### Out of Memory (OOM)

1. Enable gradient checkpointing
2. Reduce batch size
3. Use mixed precision
4. Enable LoRA
5. Reduce sequence length

### Slow Training

1. Check GPU utilization (`nvidia-smi`)
2. Enable Flash Attention
3. Increase batch size (if memory allows)
4. Profile to find bottlenecks
5. Check data loading speed

### Unstable Training

1. Lower learning rate
2. Increase warmup steps
3. Check reward scaling
4. Increase group size for GRPO/GSPO
5. Enable gradient clipping

---

## Hardware Recommendations

### Minimum Requirements

- **Development/Testing:** 1x RTX 3090/4090 (24GB)
- **Small Models (< 3B):** 1x A10/A100 (24-40GB)
- **Medium Models (3-13B):** 2-4x A100 (80GB each)
- **Large Models (> 13B):** 8x A100/H100 or cloud instances

### Cloud Instance Recommendations

| Provider | Instance | GPUs | Use Case |
|----------|----------|------|----------|
| AWS | p4d.24xlarge | 8x A100 | Large-scale training |
| AWS | g5.xlarge | 1x A10G | Development |
| GCP | a2-highgpu-1g | 1x A100 | Medium models |
| Azure | NC24ads_A100_v4 | 1x A100 | Medium models |

---

For more information, see:
- [Memory Requirements Guide](MEMORY_REQUIREMENTS_GUIDE.md)
- [GSPO Guide](GSPO_GUIDE.md)
- [Training Guide](TRL_GRPO_TRAINING_GUIDE.md)
