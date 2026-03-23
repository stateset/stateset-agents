# Kimi-K2.5 Integration

This document provides a lightweight guide for using Kimi-K2.5 with
StateSet Agents.

## Quick start

1. Build a training config:

```python
from examples.kimi_k25_config import get_kimi_k25_config

config = get_kimi_k25_config(task="customer_service")
```

2. Train with GSPO:

```python
from examples.kimi_k25_config import setup_kimi_k25_training

agent, env, reward_model, gspo_config = await setup_kimi_k25_training(
    task="customer_service",
    model_name="moonshotai/Kimi-K2.5",
)
```

## Notes

- Kimi-K2.5 is a large MoE model; LoRA and vLLM are recommended.
- Use quantization (`use_4bit=True`) on smaller GPUs.
- For long-context workloads, keep `vllm_max_model_len` aligned with 256K.
