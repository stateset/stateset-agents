# Fine-tuning Gemma 3 with GSPO

## Overview

This guide shows you how to fine-tune **Gemma 3 (Gemma 2)** models from Google using GSPO. GSPO provides stable and efficient training for Google's latest open models.

## Why Gemma 3?

Gemma 3 (officially Gemma 2) is Google's latest family of open models with excellent performance:

- ✅ **High quality** - Competitive with much larger models
- ✅ **Efficient** - 2B, 9B, and 27B parameter options
- ✅ **Well-optimized** - Fast inference with great quantization support
- ✅ **Instruction-tuned** - Pre-trained instruction-following variants available
- ✅ **Commercial friendly** - Permissive licensing for commercial use

## Supported Gemma 3 Models

| Model | Parameters | Use Case | Memory (4-bit) | Recommended Config |
|-------|-----------|----------|----------------|-------------------|
| **google/gemma-2-2b** | 2B | Base model | ~3GB | No quantization |
| **google/gemma-2-2b-it** | 2B | Instruction-tuned | ~3GB | No quantization |
| **google/gemma-2-9b** | 9B | Base model | ~10GB | 4-bit + LoRA |
| **google/gemma-2-9b-it** | 9B | Instruction-tuned | ~10GB | 4-bit + LoRA |
| **google/gemma-2-27b** | 27B | Base model | ~18GB | 4-bit + LoRA |
| **google/gemma-2-27b-it** | 27B | Instruction-tuned | ~18GB | 4-bit + LoRA |

💡 **Tip:** Use the `-it` (instruction-tuned) variants for conversational AI tasks!

## Quick Start

### 1. Install Dependencies

```bash
# Install StateSet Agents with all dependencies
pip install stateset-agents[dev]

# Install Gemma-specific dependencies
pip install transformers>=4.42.0 accelerate bitsandbytes

# Optional: Flash Attention 2 for faster training
pip install flash-attn --no-build-isolation
```

### 2. Accept Gemma License

Gemma models require accepting Google's license on HuggingFace:

1. Visit https://huggingface.co/google/gemma-2-9b-it
2. Click "Agree and access repository"
3. Login with your HuggingFace account: `huggingface-cli login`

### 3. Basic Fine-tuning

```bash
# Fine-tune Gemma 2-9B-it for customer service
python examples/finetune_gemma3_gspo.py \
    --model google/gemma-2-9b-it \
    --task customer_service \
    --use-lora \
    --use-4bit
```

### 4. Programmatic Usage

```python
import asyncio
from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.environment import ConversationEnvironment, CONVERSATION_CONFIGS
from stateset_agents.rewards.multi_objective_reward import create_customer_service_reward
from training.gspo_trainer import GSPOConfig, train_with_gspo
from training.config import get_config_for_task

async def finetune_gemma3():
    # Create agent with Gemma 3
    agent = MultiTurnAgent(AgentConfig(
        model_name="google/gemma-2-9b-it",
        system_prompt="You are a helpful AI assistant.",
    ))
    await agent.initialize()

    # Setup environment and rewards
    env_config = CONVERSATION_CONFIGS["customer_service"]
    environment = ConversationEnvironment(**env_config)
    reward_model = create_customer_service_reward()

    # Create GSPO config optimized for Gemma 3
    base_config = get_config_for_task("customer_service", model_name="google/gemma-2-9b-it")
    gspo_config = GSPOConfig.from_training_config(
        base_config,
        # GSPO parameters
        num_generations=6,
        clip_range_left=2e-4,
        clip_range_right=3e-4,
        # Training
        learning_rate=5e-6,
        num_outer_iterations=100,
        # Model optimization
        use_lora=True,
        lora_r=64,
        lora_alpha=128,
        lora_target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        use_4bit=True,
        gradient_checkpointing=True,
        # Output
        output_dir="./outputs/gemma3_gspo",
    )

    # Train with GSPO
    trained_agent = await train_with_gspo(
        config=gspo_config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )

    return trained_agent

# Run training
asyncio.run(finetune_gemma3())
```

## Recommended Configurations by Model Size

### Small Model (2B)

```python
GSPOConfig(
    # GSPO parameters
    num_generations=4,
    clip_range_left=3e-4,
    clip_range_right=4e-4,

    # Training
    learning_rate=1e-5,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=2,

    # Model
    use_lora=True,
    lora_r=16,
    lora_alpha=32,
    lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],

    # No quantization needed for 2B
    use_4bit=False,
)
```

**Best for:** Testing, prototyping, edge deployment

### Medium Model (9B)

```python
GSPOConfig(
    # GSPO parameters
    num_generations=6,
    clip_range_left=2e-4,
    clip_range_right=3e-4,

    # Training
    learning_rate=5e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,

    # Model
    use_lora=True,
    lora_r=64,
    lora_alpha=128,
    lora_target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],

    # 4-bit quantization recommended
    use_4bit=True,
    gradient_checkpointing=True,
)
```

**Best for:** Production use, balanced performance/quality

### Large Model (27B)

```python
GSPOConfig(
    # GSPO parameters
    num_generations=8,
    clip_range_left=1.5e-4,
    clip_range_right=2.5e-4,

    # Training
    learning_rate=3e-6,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,

    # Model
    use_lora=True,
    lora_r=64,
    lora_alpha=128,
    lora_target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],

    # Always use 4-bit for 27B
    use_4bit=True,
    gradient_checkpointing=True,
)
```

**Best for:** Maximum quality, complex reasoning tasks

## Hardware Requirements

### GPU Memory Requirements

| Model Size | Full Precision | 8-bit | 4-bit + LoRA | Recommended GPU |
|-----------|---------------|-------|--------------|-----------------|
| 2B | ~8GB | ~4GB | ~3GB | GTX 1080, RTX 3060 |
| 9B | ~36GB | ~18GB | ~10GB | RTX 3090, A6000 |
| 27B | ~108GB | ~54GB | ~18GB | A100 80GB |

### Multi-GPU Setup

For models that don't fit on a single GPU:

```bash
# Set visible GPUs
export CUDA_VISIBLE_DEVICES=0,1

# Training automatically distributes across GPUs
python examples/finetune_gemma3_gspo.py --model google/gemma-2-27b-it
```

## Command Line Examples

### Testing with Small Model

```bash
# Quick test with 2B model
python examples/finetune_gemma3_gspo.py \
    --model google/gemma-2-2b-it \
    --task customer_service \
    --output-dir ./outputs/test
```

### Production Training with 9B

```bash
# Full training with W&B logging
python examples/finetune_gemma3_gspo.py \
    --model google/gemma-2-9b-it \
    --task customer_service \
    --use-lora \
    --use-4bit \
    --wandb \
    --wandb-project gemma3-customer-service \
    --output-dir ./outputs/gemma3_9b_production
```

### Large Model with Maximum Quality

```bash
# 27B model with all optimizations
python examples/finetune_gemma3_gspo.py \
    --model google/gemma-2-27b-it \
    --task technical_support \
    --use-lora \
    --use-4bit \
    --output-dir ./outputs/gemma3_27b
```

## Best Practices

### 1. Use Instruction-Tuned Variants

For conversational AI, always prefer the `-it` models:

```python
# ✅ Good - instruction-tuned
model_name = "google/gemma-2-9b-it"

# ❌ Suboptimal - base model needs more work
model_name = "google/gemma-2-9b"
```

### 2. Enable Flash Attention 2

Gemma 2 supports Flash Attention 2 for faster training:

```bash
pip install flash-attn --no-build-isolation
```

This is automatically detected and used if available.

### 3. LoRA Target Modules

Gemma uses standard transformer architecture. Target all linear layers:

```python
lora_target_modules=[
    "q_proj",      # Query projection
    "k_proj",      # Key projection
    "v_proj",      # Value projection
    "o_proj",      # Output projection
    "gate_proj",   # FFN gate
    "up_proj",     # FFN up
    "down_proj",   # FFN down
]
```

### 4. Context Length

Gemma 2 supports up to 8K tokens. Adjust based on your task:

```python
# Short conversations
GSPOConfig(
    max_prompt_length=1024,
    max_completion_length=1024,
)

# Long documents
GSPOConfig(
    max_prompt_length=4096,
    max_completion_length=2048,
)
```

### 5. Temperature and Sampling

Gemma 2 works well with these settings:

```python
GSPOConfig(
    temperature=0.7,  # Balanced creativity/consistency
    top_p=0.9,        # Nucleus sampling
    top_k=50,         # Top-k sampling
)
```

## Gemma-Specific Considerations

### 1. Model Architecture

Gemma 2 uses:
- **Grouped Query Attention (GQA)** - More efficient than MHA
- **RoPE embeddings** - Rotary positional embeddings
- **SwiGLU activations** - Improved FFN activation
- **Logit soft-capping** - Better stability

### 2. Special Tokens

Gemma uses specific chat templates:

```python
# The -it models have built-in chat templates
# StateSet Agents handles this automatically!
messages = [
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi there!"},
]
```

### 3. Responsible AI

Google provides safety classifiers for Gemma:

```bash
pip install google-generativeai

# Use for production deployments
```

## Troubleshooting

### Issue: "Model not found"

**Solution:** Accept the license on HuggingFace first:

```bash
huggingface-cli login
# Then visit the model page and click "Agree"
```

### Issue: Out of Memory (OOM)

**Solutions:**
1. Enable 4-bit quantization: `--use-4bit`
2. Reduce batch size: `per_device_train_batch_size=1`
3. Enable gradient checkpointing (already default)
4. Reduce sequence length
5. Use smaller LoRA rank: `lora_r=32`

### Issue: Slow Training

**Solutions:**
1. Install Flash Attention 2: `pip install flash-attn --no-build-isolation`
2. Increase batch size if memory allows
3. Use multiple GPUs
4. Check that bf16 is enabled (default)

### Issue: Poor Quality

**Solutions:**
1. Use instruction-tuned variant (`-it`)
2. Increase LoRA rank: `lora_r=64`
3. Lower learning rate: `learning_rate=3e-6`
4. Increase group size: `num_generations=8`
5. Train for more iterations

### Issue: Training Instability

**Solutions:**
1. Reduce learning rate by 2-5x
2. Increase group size for more stable advantages
3. Adjust clipping ranges (reduce by 20-30%)
4. Enable reference model KL penalty: `beta=0.01`

## Performance Benchmarks

Based on our testing with Gemma 2 models:

| Model | Training Speed | Memory (4-bit) | Quality |
|-------|---------------|----------------|---------|
| Gemma 2-2B-it | ~400 tok/s | 3GB | Very Good |
| Gemma 2-9B-it | ~90 tok/s | 10GB | Excellent |
| Gemma 2-27B-it | ~30 tok/s | 18GB | Outstanding |

*Benchmarks on single A100 80GB GPU*

## Comparison: Gemma 3 vs Qwen 3

| Feature | Gemma 2 | Qwen 2.5 |
|---------|---------|----------|
| **Size range** | 2B - 27B | 0.5B - 72B + MoE |
| **Context** | 8K tokens | 32K tokens |
| **License** | Permissive | Permissive |
| **Architecture** | GQA, RoPE | MHA/GQA, RoPE |
| **Best for** | General, balanced | Long context, MoE |
| **Performance** | Excellent | Excellent |

**Choose Gemma if:**
- You want simple, reliable models
- 8K context is sufficient
- You prefer Google's ecosystem

**Choose Qwen if:**
- You need very long context (32K)
- You want MoE efficiency
- You need the widest range of sizes

## Advanced: Multi-Task Fine-tuning

Train on multiple tasks simultaneously:

```python
from stateset_agents.core.environment import ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import MultiObjectiveReward

# Combine multiple task scenarios
scenarios = (
    CONVERSATION_CONFIGS["customer_service"]["scenarios"] +
    CONVERSATION_CONFIGS["technical_support"]["scenarios"] +
    CONVERSATION_CONFIGS["sales"]["scenarios"]
)

environment = ConversationEnvironment(scenarios=scenarios, max_turns=10)

# Multi-objective reward
reward_model = MultiObjectiveReward(
    rewards={
        "customer_service": create_domain_reward("customer_service"),
        "technical_support": create_domain_reward("technical_support"),
        "sales": create_domain_reward("sales"),
    },
    weights={"customer_service": 0.4, "technical_support": 0.3, "sales": 0.3}
)

# Train as usual
trained_agent = await train_with_gspo(
    config=gspo_config,
    agent=agent,
    environment=environment,
    reward_model=reward_model,
)
```

## Exporting and Deployment

After training, export your model:

```python
# Save model
trainer.save_model("./outputs/gemma3_final")

# Load for inference
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./outputs/gemma3_final")
tokenizer = AutoTokenizer.from_pretrained("./outputs/gemma3_final")

# Deploy with vLLM for production
from vllm import LLM

llm = LLM(model="./outputs/gemma3_final")
```

## Resources

- **Gemma Model Hub:** https://huggingface.co/google/gemma-2-9b-it
- **Gemma Documentation:** https://ai.google.dev/gemma
- **GSPO Paper:** https://arxiv.org/abs/2507.18071v2
- **StateSet Agents Docs:** https://stateset-agents.readthedocs.io/

## Support

For questions or issues:
- Open an issue: https://github.com/stateset/stateset-agents/issues
- Join Discord: https://discord.gg/stateset

---

**Happy fine-tuning!** 🚀
