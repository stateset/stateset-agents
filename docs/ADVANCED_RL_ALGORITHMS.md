# Advanced RL Algorithms for LLM Training

StateSet Agents now includes state-of-the-art reinforcement learning algorithms for training large language models. This guide covers the three newest additions: GEPO, DAPO, and VAPO.

## Algorithm Comparison

| Algorithm | Best For | AIME 2024 | Key Innovation |
|-----------|----------|-----------|----------------|
| **GRPO** | General dialogue | - | Group-relative advantages |
| **GSPO** | Stable training | - | Sequence-level importance weights |
| **GEPO** | Distributed/heterogeneous | - | Group-level importance weights |
| **DAPO** | Long-CoT reasoning | 50 pts | Clip-Higher + Dynamic Sampling |
| **VAPO** | Reasoning (SOTA) | 60.4 pts | Value-augmented PPO |

## GEPO: Group Expectation Policy Optimization

GEPO is designed for stable training in heterogeneous and distributed environments where network latency causes policy staleness.

### Key Innovation

GEPO uses **group-level importance weights** instead of per-token (GRPO) or per-sequence (GSPO) weights:

```
w_GEPO(y|x) = p(y|x) / E_q[q(y|x)]
```

This exponentially reduces variance under high KL divergence, making training stable even with:
- Network delays
- Heterogeneous compute resources
- Asynchronous training

### When to Use GEPO

- Training across multiple data centers
- Using heterogeneous GPU clusters
- When you see training instability with GRPO/GSPO
- Distributed training with variable latency

### Quick Start

```python
from training import GEPOConfig, train_with_gepo

# Define reward function
def reward_fn(prompt: str, response: str) -> float:
    # Your reward logic
    return score

# Configure GEPO
config = GEPOConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    group_size=8,  # Responses per prompt
    clip_eps=0.2,
    learning_rate=1e-6,
)

# Train
model, tokenizer, metrics = await train_with_gepo(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    reward_fn=reward_fn,
    train_prompts=prompts,
    config=config,
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `group_size` | 8 | Responses per prompt (G) |
| `clip_eps` | 0.2 | PPO clipping epsilon |
| `learning_rate` | 1e-6 | Learning rate |
| `warmup_ratio` | 0.03 | 3% linear warmup |

### Reference

- Paper: [GEPO: Group Expectation Policy Optimization](https://arxiv.org/abs/2508.17850)

---

## DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization

DAPO achieves 50 points on AIME 2024, optimized for long chain-of-thought (CoT) reasoning tasks.

### Four Key Techniques

#### 1. Clip-Higher (Asymmetric Clipping)

Standard PPO uses symmetric clipping `[1-e, 1+e]`. DAPO uses asymmetric:
- Lower bound: `1 - 0.2 = 0.8`
- Upper bound: `1 + 0.28 = 1.28`

This allows more exploration while maintaining stability.

#### 2. Dynamic Sampling

Filters out prompts where accuracy is 0% or 100%:
- 0% accuracy = all responses wrong = no learning signal
- 100% accuracy = all responses correct = no gradient

Only keeps samples with intermediate accuracy.

#### 3. Token-Level Loss Normalization

Normalizes loss by total token count, not sample count:

```
Loss = (1 / total_tokens) * sum(per_token_loss)
```

Prevents longer sequences from dominating gradients.

#### 4. Overlong Reward Shaping

Soft penalty as sequences approach max length:

```python
if length <= max_length - cache_length:
    penalty = 0
elif length <= max_length:
    penalty = -1 * (length - soft_start) / cache_length
else:
    penalty = -1  # Truncated
```

### When to Use DAPO

- Math and reasoning tasks
- Long chain-of-thought generation
- Tasks with verifiable answers
- When GRPO shows entropy collapse

### Quick Start

```python
from training import DAPOConfig, train_with_dapo, train_reasoning_with_dapo

# For math problems with verifiable answers
math_problems = [
    {"problem": "Solve: 2x + 3 = 7", "answer": "x = 2"},
    # ... more problems
]

model, tokenizer, metrics = await train_reasoning_with_dapo(
    model_name="Qwen/Qwen2.5-32B",
    math_problems=math_problems,
    output_dir="./outputs/dapo-math",
)

# Or with custom reward function
config = DAPOConfig(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    group_size=16,
    clip_eps_low=0.2,
    clip_eps_high=0.28,
    use_dynamic_sampling=True,
    use_overlong_shaping=True,
)

model, tokenizer, metrics = await train_with_dapo(
    model_name="Qwen/Qwen2.5-7B-Instruct",
    reward_fn=reward_fn,
    train_prompts=prompts,
    verifier_fn=verifier_fn,  # Optional: binary correctness check
    config=config,
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `clip_eps_low` | 0.2 | Lower clipping bound |
| `clip_eps_high` | 0.28 | Upper clipping bound (Clip-Higher) |
| `group_size` | 16 | Responses per prompt |
| `use_dynamic_sampling` | True | Filter trivial accuracy samples |
| `use_overlong_shaping` | True | Apply length penalty |
| `max_generation_length` | 20480 | Maximum tokens before penalty |
| `overlong_cache_length` | 4096 | Soft penalty interval |

### Reference

- Paper: [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- Code: [BytedTsinghua-SIA/DAPO](https://github.com/BytedTsinghua-SIA/DAPO)

---

## VAPO: Value-Augmented Policy Optimization

VAPO is the current state-of-the-art, achieving **60.4 on AIME 2024** (surpassing DAPO's 50).

### Seven Key Modifications to PPO

1. **Value Network Warmup**: Pretrain value network for 50 steps with Monte-Carlo returns
2. **Decoupled GAE**: Separate lambda for critic (1.0) and policy (adaptive)
3. **Length-Adaptive Lambda**: `lambda = 1 - 1/(alpha * length)`
4. **Clip-Higher**: Asymmetric clipping like DAPO
5. **Token-Level Loss**: Normalize by total tokens
6. **Positive Example LM Loss**: Add NLL loss on correct samples
7. **Group Sampling**: More samples per prompt, fewer prompts

### Key Innovation: Length-Adaptive GAE

The lambda parameter in GAE controls bias-variance tradeoff:
- High lambda (1.0) = low bias, high variance
- Low lambda (0.0) = high bias, low variance

VAPO adapts lambda based on sequence length:

```python
lambda_policy = 1 - 1 / (0.05 * sequence_length)
```

Short sequences get lower lambda (more bias, less variance).
Long sequences get higher lambda (less bias, more variance).

### When to Use VAPO

- State-of-the-art reasoning performance required
- Math, coding, and complex reasoning tasks
- When you need stable, reliable training
- Have compute for value network

### Quick Start

```python
from training import VAPOConfig, train_with_vapo

config = VAPOConfig(
    model_name="Qwen/Qwen2.5-32B",
    group_size=16,
    value_warmup_steps=50,
    actor_learning_rate=1e-6,
    critic_learning_rate=2e-6,
    use_positive_lm_loss=True,
    positive_lm_weight=0.1,
)

model, tokenizer, metrics = await train_with_vapo(
    model_name="Qwen/Qwen2.5-32B",
    reward_fn=reward_fn,
    train_prompts=prompts,
    verifier_fn=verifier_fn,
    config=config,
)
```

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `value_warmup_steps` | 50 | Steps to pretrain value network |
| `lambda_critic` | 1.0 | GAE lambda for value network |
| `lambda_policy_alpha` | 0.05 | Alpha for length-adaptive lambda |
| `actor_learning_rate` | 1e-6 | Policy learning rate |
| `critic_learning_rate` | 2e-6 | Value network learning rate |
| `use_positive_lm_loss` | True | Add NLL on correct samples |
| `positive_lm_weight` | 0.1 | Weight for positive LM loss |
| `clip_eps_low` | 0.2 | Lower clipping bound |
| `clip_eps_high` | 0.28 | Upper clipping bound |

### Reference

- Paper: [VAPO: Efficient and Reliable Reinforcement Learning for Advanced Reasoning Tasks](https://arxiv.org/abs/2504.05118)

---

## Choosing the Right Algorithm

```
                    ┌─────────────────────────────────────┐
                    │     What's your use case?           │
                    └─────────────────────────────────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    ▼               ▼               ▼
              Dialogue        Reasoning         Distributed
                    │               │               │
                    ▼               ▼               ▼
              ┌─────────┐     ┌─────────┐     ┌─────────┐
              │  GRPO   │     │  Need   │     │  GEPO   │
              │  GSPO   │     │  SOTA?  │     │         │
              └─────────┘     └────┬────┘     └─────────┘
                                   │
                         ┌─────────┴─────────┐
                         ▼                   ▼
                       Yes                  No
                         │                   │
                         ▼                   ▼
                   ┌─────────┐         ┌─────────┐
                   │  VAPO   │         │  DAPO   │
                   │ (60.4)  │         │  (50)   │
                   └─────────┘         └─────────┘
```

### Summary Recommendations

| Scenario | Recommended | Why |
|----------|-------------|-----|
| Customer service chatbot | GRPO/GSPO | Stable, well-tested |
| Math problem solving | VAPO | SOTA performance |
| Code generation | DAPO | Good for long outputs |
| Distributed training | GEPO | Handles latency |
| Quick prototyping | GRPO | Simplest |
| Production reasoning | VAPO | Most reliable |

---

## Example: Training a Math Reasoning Model

```python
import asyncio
from training import VAPOConfig, train_with_vapo

# Prepare math dataset
math_problems = [
    "What is the sum of all integers from 1 to 100?",
    "Solve for x: 3x^2 - 12x + 9 = 0",
    # ... more problems
]

# Define reward (use verifier for correctness)
def reward_fn(prompt: str, response: str) -> float:
    # In practice, use a proper math verifier
    if "final answer" in response.lower():
        return 1.0
    return 0.0

def verifier_fn(prompt: str, response: str) -> bool:
    # Check if answer is correct
    # Use symbolic computation or pattern matching
    return True  # Placeholder

async def main():
    config = VAPOConfig(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        group_size=8,
        num_episodes=1000,
        value_warmup_steps=50,
        output_dir="./outputs/math-vapo",
    )

    model, tokenizer, metrics = await train_with_vapo(
        model_name="Qwen/Qwen2.5-7B-Instruct",
        reward_fn=reward_fn,
        train_prompts=math_problems,
        verifier_fn=verifier_fn,
        config=config,
        use_wandb=True,
        wandb_project="math-reasoning",
    )

    print(f"Final accuracy: {metrics['accuracy'][-1]:.2%}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Hyperparameter Tuning Tips

### GEPO
- Start with `group_size=8` (from paper)
- Use standard `clip_eps=0.2`
- Monitor KL divergence - GEPO should stay stable

### DAPO
- Keep `clip_eps_low=0.2, clip_eps_high=0.28` (validated in paper)
- Enable all four techniques initially
- Adjust `overlong_cache_length` based on your max sequence length
- Monitor `filtered_ratio` - should be < 50%

### VAPO
- Always use value warmup (50 steps minimum)
- Critic LR should be >= Actor LR
- Start with `positive_lm_weight=0.1`
- Monitor `explained_variance` - should increase during training

---

## Troubleshooting

### Training Instability
1. Try GEPO if using distributed training
2. Reduce learning rate
3. Increase group_size for better advantage estimation
4. Check KL divergence metrics

### Low Accuracy
1. Ensure reward function is well-calibrated
2. Check if dynamic sampling is filtering too many samples
3. Try VAPO with value warmup
4. Verify verifier function is correct

### Memory Issues
1. Reduce group_size
2. Enable gradient checkpointing
3. Use LoRA/QLoRA
4. Reduce max_completion_length

---

## References

1. GEPO: [arxiv.org/abs/2508.17850](https://arxiv.org/abs/2508.17850)
2. DAPO: [arxiv.org/abs/2503.14476](https://arxiv.org/abs/2503.14476)
3. VAPO: [arxiv.org/abs/2504.05118](https://arxiv.org/abs/2504.05118)
4. GRPO: [arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300)
5. GSPO: [arxiv.org/abs/2507.18071](https://arxiv.org/abs/2507.18071)
