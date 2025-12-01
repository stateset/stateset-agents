# Memory Requirements Guide

This guide provides detailed memory requirements for training and inference with StateSet Agents, helping you choose the right hardware and configuration for your use case.

## Table of Contents

1. [Quick Reference Table](#quick-reference-table)
2. [Memory Calculation](#memory-calculation)
3. [Training Memory Requirements](#training-memory-requirements)
4. [Inference Memory Requirements](#inference-memory-requirements)
5. [Memory Optimization Strategies](#memory-optimization-strategies)
6. [Hardware Recommendations](#hardware-recommendations)

---

## Quick Reference Table

### Training Memory (GPU VRAM)

| Model Size | Full Fine-tune | LoRA | QLoRA (4-bit) |
|------------|----------------|------|---------------|
| 125M (GPT-2 small) | 2 GB | 1 GB | 0.5 GB |
| 355M (GPT-2 medium) | 4 GB | 2 GB | 1 GB |
| 774M (GPT-2 large) | 8 GB | 3 GB | 2 GB |
| 1.3B | 16 GB | 6 GB | 3 GB |
| 3B | 36 GB | 10 GB | 6 GB |
| 7B (Llama-2-7B) | 56 GB | 16 GB | 8 GB |
| 13B (Llama-2-13B) | 104 GB | 26 GB | 14 GB |
| 70B (Llama-2-70B) | 560 GB | 80 GB | 48 GB |

*Values assume batch_size=1, seq_len=2048, FP16/BF16 precision, AdamW optimizer*

### Inference Memory (GPU VRAM)

| Model Size | FP16 | 8-bit | 4-bit |
|------------|------|-------|-------|
| 125M | 0.25 GB | 0.15 GB | 0.1 GB |
| 355M | 0.7 GB | 0.4 GB | 0.25 GB |
| 774M | 1.5 GB | 0.9 GB | 0.5 GB |
| 1.3B | 2.6 GB | 1.5 GB | 0.8 GB |
| 3B | 6 GB | 3.5 GB | 2 GB |
| 7B | 14 GB | 8 GB | 4.5 GB |
| 13B | 26 GB | 15 GB | 8 GB |
| 70B | 140 GB | 75 GB | 40 GB |

---

## Memory Calculation

### Training Memory Formula

```
Total VRAM = Model Memory + Optimizer Memory + Gradient Memory + Activation Memory + KV Cache

Where:
- Model Memory = params × bytes_per_param
- Optimizer Memory = params × optimizer_factor × bytes
  - AdamW: 2× (momentum + variance)
  - SGD: 1×
- Gradient Memory = params × bytes_per_param
- Activation Memory ≈ batch_size × seq_len × hidden_dim × num_layers × bytes
- KV Cache = 2 × batch_size × seq_len × num_layers × hidden_dim × bytes
```

### Practical Example: Llama-2-7B

```python
# Model parameters
params = 7_000_000_000  # 7B
hidden_dim = 4096
num_layers = 32
seq_len = 2048
batch_size = 1

# FP16 precision (2 bytes)
bytes_per_param = 2

# Calculate components
model_mem = params * bytes_per_param / 1e9  # ~14 GB
optimizer_mem = params * 2 * 4 / 1e9  # ~56 GB (AdamW, FP32 states)
gradient_mem = params * bytes_per_param / 1e9  # ~14 GB
activation_mem = batch_size * seq_len * hidden_dim * num_layers * bytes_per_param / 1e9  # ~0.5 GB

total = model_mem + optimizer_mem + gradient_mem + activation_mem
# Total: ~84.5 GB (before optimization)
```

---

## Training Memory Requirements

### GRPO Training

GRPO requires generating multiple responses per prompt (group-relative advantages):

```python
# Additional memory for GRPO
grpo_overhead = num_generations × generation_memory

# Example with num_generations=4:
# Generates 4 responses of max_length=256 each
# Additional ~2-4 GB for generation buffers
```

**Recommended Configuration:**

```python
from stateset_agents.training.config import TrainingConfig

# For 7B model on single 80GB A100
config = TrainingConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    num_generations=4,
    max_completion_length=256,
    use_lora=True,
    lora_r=16,
    gradient_checkpointing=True,
    bf16=True,
)
# Expected memory: ~45 GB
```

### GSPO Training

GSPO has similar requirements to GRPO:

```python
from training.gspo_trainer import GSPOConfig

# For 7B model on single 80GB A100
config = GSPOConfig(
    per_device_train_batch_size=1,
    num_generations=4,
    max_completion_length=256,
    use_lora=True,
    gradient_checkpointing=True,
)
# Expected memory: ~45 GB
```

### Memory by Configuration

| Configuration | 7B Model Memory | 13B Model Memory |
|---------------|-----------------|------------------|
| Full FT, FP32 | 112 GB | 208 GB |
| Full FT, FP16 | 56 GB | 104 GB |
| LoRA, FP16 | 20 GB | 35 GB |
| LoRA, FP16, Gradient Ckpt | 16 GB | 28 GB |
| QLoRA, 4-bit | 8 GB | 14 GB |
| QLoRA, 4-bit, Gradient Ckpt | 6 GB | 12 GB |

---

## Inference Memory Requirements

### Base Inference

```
Inference Memory = Model Weights + KV Cache + Activation Buffer

KV Cache per token ≈ 2 × num_layers × hidden_dim × bytes_per_param
```

### Batch Inference Scaling

Memory scales with batch size and sequence length:

```python
# KV Cache grows linearly with batch size and context
kv_cache_size = 2 * num_layers * hidden_dim * batch_size * seq_len * bytes

# Example: 7B model, batch=8, seq_len=2048, FP16
kv_cache = 2 * 32 * 4096 * 8 * 2048 * 2 / 1e9  # ~8.6 GB
```

### Inference Memory Table

| Model | Batch=1 | Batch=8 | Batch=32 |
|-------|---------|---------|----------|
| 7B FP16 | 14 GB | 18 GB | 30 GB |
| 7B 8-bit | 8 GB | 12 GB | 24 GB |
| 7B 4-bit | 4.5 GB | 8.5 GB | 20 GB |
| 13B FP16 | 26 GB | 34 GB | 58 GB |
| 70B FP16 | 140 GB | 160 GB | 220 GB |

*Assumes seq_len=2048*

---

## Memory Optimization Strategies

### Strategy 1: LoRA Fine-Tuning

Reduces trainable parameters by 90%+:

```python
config = TrainingConfig(
    use_lora=True,
    lora_r=16,          # Rank (8-64 typical)
    lora_alpha=32,      # Scaling factor
    lora_dropout=0.05,
)

# Memory savings for 7B model:
# Full FT: 56 GB → LoRA: 16 GB (71% reduction)
```

### Strategy 2: Gradient Checkpointing

Trades compute for memory:

```python
config = TrainingConfig(
    gradient_checkpointing=True,
)

# Memory savings: ~60%
# Speed penalty: ~20%
```

### Strategy 3: Mixed Precision

```python
config = TrainingConfig(
    bf16=True,  # Recommended for Ampere+ GPUs
    # OR
    fp16=True,  # For older GPUs
)

# Memory savings: ~50%
# May improve training speed
```

### Strategy 4: Quantization

For extreme memory constraints:

```python
from stateset_agents.core.agent import AgentConfig

# 8-bit (good quality, ~50% memory reduction)
config = AgentConfig(use_8bit=True)

# 4-bit (aggressive, ~75% memory reduction)
config = AgentConfig(use_4bit=True)
```

### Strategy 5: Gradient Accumulation

Simulate larger batches without more memory:

```python
config = TrainingConfig(
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
)
# Effective batch size = 16, memory of batch size 1
```

### Combined Optimization Example

```python
# Maximum memory efficiency for 7B model on 24GB GPU
config = TrainingConfig(
    use_lora=True,
    lora_r=8,
    gradient_checkpointing=True,
    bf16=True,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,
    max_completion_length=128,
)
# Expected memory: ~12 GB (fits on RTX 3090/4090)
```

---

## Hardware Recommendations

### By Model Size

| Model Size | Minimum GPU | Recommended GPU | Multi-GPU Option |
|------------|-------------|-----------------|------------------|
| < 1B | RTX 3060 (12GB) | RTX 3090 (24GB) | - |
| 1-3B | RTX 3090 (24GB) | RTX 4090 (24GB) | 2x RTX 3090 |
| 7B | A10G (24GB) + LoRA | A100 (40GB) | 2x A10G |
| 13B | A100 (40GB) + LoRA | A100 (80GB) | 2x A100 40GB |
| 70B | 4x A100 (80GB) | 8x A100 (80GB) | 8x H100 |

### GPU Comparison

| GPU | VRAM | FP16 TFLOPs | Best For |
|-----|------|-------------|----------|
| RTX 3090 | 24 GB | 35.6 | Development, < 3B models |
| RTX 4090 | 24 GB | 82.6 | Development, < 3B models |
| A10G | 24 GB | 31.2 | Cloud inference |
| A100 40GB | 40 GB | 77.9 | Training 7-13B |
| A100 80GB | 80 GB | 77.9 | Training 13-70B |
| H100 | 80 GB | 267.6 | Large-scale training |

### Cloud Instance Recommendations

| Use Case | AWS | GCP | Azure |
|----------|-----|-----|-------|
| Development | g4dn.xlarge | n1-standard-4 + T4 | NC4as_T4_v3 |
| 7B Training | g5.2xlarge | a2-highgpu-1g | NC24ads_A100_v4 |
| 13B Training | p4d.24xlarge | a2-highgpu-2g | NC48ads_A100_v4 |
| 70B Training | p4de.24xlarge | a2-megagpu-16g | ND96amsr_A100_v4 |

### Cost-Effective Setup

For budget-conscious training:

1. **Development:** Colab Pro+ with A100 (~$50/month)
2. **Small scale:** Lambda Labs cloud ($1.10/hr for A100)
3. **Medium scale:** AWS Spot Instances (60-90% discount)
4. **Large scale:** Reserved instances or on-prem

---

## Monitoring Memory Usage

### Real-time Monitoring

```python
import torch

# Check current memory
allocated = torch.cuda.memory_allocated() / 1e9
reserved = torch.cuda.memory_reserved() / 1e9
print(f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")

# Peak memory
peak = torch.cuda.max_memory_allocated() / 1e9
print(f"Peak: {peak:.2f} GB")
```

### Built-in Monitoring

```python
from stateset_agents.utils.monitoring import PerformanceMonitor

monitor = PerformanceMonitor()
monitor.start()

# ... training code ...

print(monitor.get_memory_stats())
```

### Memory Profiling

```bash
# Using PyTorch memory profiler
python -c "
import torch
from torch.cuda.memory import memory_stats

# Your code here

stats = memory_stats()
print(f'Peak allocated: {stats[\"allocated_bytes.all.peak\"] / 1e9:.2f} GB')
"
```

---

## Troubleshooting

### Out of Memory Errors

**Symptoms:** `CUDA out of memory` or `RuntimeError: CUDA error: out of memory`

**Solutions:**
1. Enable gradient checkpointing
2. Reduce batch size
3. Reduce sequence length
4. Use LoRA instead of full fine-tuning
5. Enable 4-bit or 8-bit quantization
6. Use gradient accumulation

### Memory Fragmentation

**Symptoms:** OOM errors despite having enough total memory

**Solutions:**
```python
# Clear cache periodically
torch.cuda.empty_cache()

# Set memory fraction limit
torch.cuda.set_per_process_memory_fraction(0.95)

# Use memory-efficient attention
config = AgentConfig(attn_implementation="flash_attention_2")
```

### Memory Leaks

**Symptoms:** Memory usage grows over time

**Solutions:**
```python
# Delete unused tensors
del tensor
torch.cuda.empty_cache()

# Use context managers
with torch.no_grad():
    # inference code

# Detach tensors from computation graph
output = model(input).detach()
```

---

## Quick Decision Guide

### "I have X GB VRAM, what can I train?"

| VRAM | Full Fine-tune | LoRA | QLoRA |
|------|----------------|------|-------|
| 8 GB | 355M | 1.3B | 3B |
| 12 GB | 774M | 3B | 7B |
| 16 GB | 1.3B | 7B | 7B |
| 24 GB | 3B | 7B | 13B |
| 40 GB | 7B | 13B | 70B (limited) |
| 80 GB | 13B | 70B | 70B |

### "I want to train X model, what do I need?"

| Model | Minimum (QLoRA) | Recommended (LoRA) | Ideal (Full FT) |
|-------|-----------------|---------------------|-----------------|
| GPT-2 (124M) | 4 GB | 8 GB | 16 GB |
| Phi-2 (2.7B) | 8 GB | 16 GB | 48 GB |
| Llama-2-7B | 12 GB | 24 GB | 80 GB |
| Llama-2-13B | 16 GB | 40 GB | 160 GB |
| Llama-2-70B | 48 GB | 80 GB | 640 GB |

---

For more optimization techniques, see [Performance Tuning Guide](PERFORMANCE_TUNING_GUIDE.md).
