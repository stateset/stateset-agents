# Getting Started with Post-Training Qwen/Qwen3.5-0.8B

## Overview

This guide shows how to run a first GSPO post-training job for `Qwen/Qwen3.5-0.8B` in this repository.

The recommended starting checkpoint is `Qwen/Qwen3.5-0.8B-Base`.
Use `Qwen/Qwen3.5-0.8B` as a baseline for comparison or inference, not as the default starting point for a new post-training run.

## Why start from `-Base`

For post-training in this repo, prefer `Qwen/Qwen3.5-0.8B-Base` because:

- the base checkpoint is the one Qwen positions for fine-tuning and research workflows
- the model card notes that the chat control tokens were trained to support efficient LoRA-style PEFT without needing embedding fine-tuning
- it gives you a cleaner starting point before task-specific RL shaping

## Scope of this guide

This is a **text-only** getting-started path.

Today, the default StateSet GSPO loader initializes models through `AutoModelForCausalLM` and a tokenizer. That is a good fit for text post-training, but it is not a full multimodal post-training path for Qwen3.5's vision encoder.

If you want image-conditioned post-training, build a multimodal-specific training path first. For a first run, keep the setup text-only.

## Recommended first run

Start with these choices:

- Model: `Qwen/Qwen3.5-0.8B-Base`
- Algorithm: `GSPO`
- Task preset: `customer_service`
- Starter profile: `balanced` by default, switch to `memory` for smaller GPUs or `quality` when you have headroom
- Adaptation: `LoRA`
- Quantization: off by default on `balanced`, automatically enabled on the `memory` starter profile
- Context lengths: keep them short at first, for example `1024` prompt tokens and `768` completion tokens

## Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,training,trl]"
```

Notes:

- `trl` pulls in `bitsandbytes`, which you need for `--use-4bit`
- if `bitsandbytes` fails to import, leave quantization off and get the base run working first
- if `flash-attn` is not installed, prefer `attn_implementation="sdpa"` for the initial agent load

## Quick start with the dedicated starter path

The starter is available from the packaged CLI and the equivalent example script.
Start with a dry-run so you can inspect the resolved config without loading a model:

```bash
stateset-agents qwen3-5-0-8b --json-output
stateset-agents qwen3-5-0-8b --starter-profile memory --json-output
stateset-agents qwen3-5-0-8b --list-profiles --json-output
stateset-agents qwen3-5-0-8b --write-config ./qwen3_5_0_8b.json
stateset-agents init --preset qwen3-5-0-8b --path ./qwen3_5_0_8b.json --format json
stateset-agents init --preset qwen3-5-0-8b --starter-profile memory --path ./qwen3_5_0_8b_memory.json --format json
# equivalent example script:
python examples/finetune_qwen3_5_0_8b_gspo.py --dry-run
python examples/finetune_qwen3_5_0_8b_gspo.py --starter-profile memory --dry-run
python examples/finetune_qwen3_5_0_8b_gspo.py --list-profiles
```

Then run the recommended starter job:

```bash
stateset-agents qwen3-5-0-8b --no-dry-run --task customer_service
stateset-agents qwen3-5-0-8b --config ./qwen3_5_0_8b.json --no-dry-run
# equivalent example script:
python examples/finetune_qwen3_5_0_8b_gspo.py --task customer_service
```

If you are on a smaller CUDA GPU and need extra memory headroom:

```bash
stateset-agents qwen3-5-0-8b --no-dry-run --task customer_service --use-4bit
```

The generic family-wide script is still available at `examples/finetune_qwen3_gspo.py`, but the dedicated starter path is the cleaner way to validate an initial `0.8B` run.

### Starter profiles

Use the built-in starter profiles to match your first run to your hardware and tolerance for latency:

- `balanced`: the default path; this keeps LoRA on and uses the repo's standard first-run settings
- `memory`: enables `4-bit`, shortens context and group sizes, and lowers iteration cost for smaller GPUs
- `quality`: increases context and group sizes for a steadier run when you have more memory headroom

The starter profile is resolved into the saved config file, so once you write a config with `--write-config` or `init --preset`, the downstream run does not need to remember which profile created it.

### Inspect the profile matrix before choosing one

If you want to compare the built-in profiles before running or saving one, ask the starter to print the profile catalog:

```bash
stateset-agents qwen3-5-0-8b --task sales --list-profiles --json-output
python examples/finetune_qwen3_5_0_8b_gspo.py --task sales --list-profiles
```

The catalog includes a short description, the resolved config, validation warnings, and a summary of the important first-run knobs such as effective batch size, quantization mode, context lengths, and group sizes.

## Minimal programmatic example

```python
import asyncio

from stateset_agents import MultiTurnAgent
from stateset_agents.core.agent import AgentConfig
from stateset_agents.core.environment import CONVERSATION_CONFIGS, ConversationEnvironment
from stateset_agents.rewards.multi_objective_reward import create_customer_service_reward
from stateset_agents.training import GSPOConfig, get_config_for_task, train_with_gspo

MODEL_NAME = "Qwen/Qwen3.5-0.8B-Base"


async def main() -> None:
    agent = MultiTurnAgent(
        AgentConfig(
            model_name=MODEL_NAME,
            system_prompt="You are a helpful customer service assistant.",
            max_new_tokens=768,
            trust_remote_code=True,
            attn_implementation="sdpa",
        )
    )
    await agent.initialize()

    env_config = CONVERSATION_CONFIGS["customer_service"].copy()
    environment = ConversationEnvironment(**env_config)
    reward_model = create_customer_service_reward()

    base_config = get_config_for_task("customer_service", model_name=MODEL_NAME)
    config = GSPOConfig.from_training_config(
        base_config,
        report_to="none",
        output_dir="./outputs/qwen3_5_0_8b_gspo",
        num_outer_iterations=25,
        generations_per_iteration=32,
        num_generations=4,
        learning_rate=8e-6,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        max_prompt_length=1024,
        max_completion_length=768,
        use_lora=True,
        lora_r=32,
        lora_alpha=64,
        lora_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        gradient_checkpointing=True,
        use_4bit=False,
    )

    await train_with_gspo(
        config=config,
        agent=agent,
        environment=environment,
        reward_model=reward_model,
    )


asyncio.run(main())
```

## Why this configuration is a good starting point

- `Qwen/Qwen3.5-0.8B-Base` keeps the run aligned with a post-training workflow rather than continuing from an already post-trained assistant checkpoint.
- `GSPO` is the right default in this codebase for Qwen-style post-training.
- `LoRA` keeps the run cheap and fast enough for iteration.
- `num_generations=4` keeps the group-relative signal useful without exploding generation cost.
- shorter prompt and completion lengths reduce the chance of an early OOM and make reward iteration faster.

## What gets saved

Training outputs are written under your `output_dir`.

Look for:

- checkpoint directories such as `checkpoint-5`
- the final artifact at `output_dir/final_model`
- `training_metrics.json` alongside the saved model assets

If you train with LoRA enabled, the saved model is a PEFT-style adapter checkpoint rather than a merged standalone base model.

## Common gotchas

### 1. Use `trust_remote_code=True`

Qwen3.5 relies on custom modeling code. Set `trust_remote_code=True` on the initial `AgentConfig`.

### 2. Prefer `attn_implementation="sdpa"` for the first run

The default agent config in this repo prefers `flash_attention_2`, which may fail if the extension is not installed. `sdpa` is the safer first-run option.

### 3. Keep the first run text-only

This guide does not cover image inputs or vision-encoder training.

### 4. Turn off W&B until the run is stable

`TrainingConfig` defaults to W&B-oriented settings elsewhere in the repo. For local smoke tests, set `report_to="none"`.

### 5. Add quantization only when you need it

`Qwen/Qwen3.5-0.8B-Base` is small enough that many runs can start without 4-bit quantization. Only add `use_4bit=True` when memory pressure requires it.

## When to add SFT before GSPO

If you already have supervised task data, do a short SFT stage first and use GSPO after that for behavioral shaping, preference pressure, and reward-driven refinement.

If you do not have curated supervised data yet, starting directly with a small GSPO run is still a good way to validate your environment and reward design.

## Suggested next steps

- replace the preset customer service scenarios with your own domain prompts
- tighten or extend the reward function in `stateset_agents/rewards/multi_objective_reward.py`
- increase `num_outer_iterations` only after the small smoke test runs cleanly
- compare the resulting adapter against the untouched base model and the post-trained `Qwen/Qwen3.5-0.8B` checkpoint

## Related files

- `stateset_agents/cli.py`
- `stateset_agents/training/qwen3_5_starter.py`
- `examples/finetune_qwen3_5_0_8b_gspo.py`
- `examples/qwen3_5_config.py`
- `examples/finetune_qwen3_gspo.py`
- `docs/CLI_REFERENCE.md`
- `docs/GSPO_GUIDE.md`
- `docs/MEMORY_REQUIREMENTS_GUIDE.md`
- `stateset_agents/training/gspo_entrypoints.py`
- `stateset_agents/training/gspo_config.py`
