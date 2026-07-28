"""
Starter-project scaffolding — what a fine-tuning consultant clones on day one.

``scaffold_project`` materializes a complete project directory from a named
template. The shipped templates target the audience this framework is built
for: a developer who wants to fine-tune a model on a client's data and serve
it. Each template includes everything needed to go from ``pip install`` to a
running endpoint.

Templates ship as in-Python strings rather than file-tree assets so they're
trivially testable, embeddable, and patchable without packaging gymnastics.

Available templates (auto-discovered from ``SCAFFOLD_TEMPLATES``):

* ``customer-support`` — multi-turn dialogue agent (the framework's
  differentiator). Uses ``SupportRewardComposite``.
* ``gsm8k-math`` — single-turn math reasoner with verifiable reward. The
  cheapest path to a published number.
* ``minimal`` — bare scaffold: one trainer, one reward stub, one scenario.

Usage from Python::

    from stateset_agents.scaffolding import scaffold_project
    scaffold_project("customer-support", "./my-client-project")

Usage from CLI::

    stateset-agents starter customer-support ./my-client-project
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Template data — kept as in-source strings for testability.
# ---------------------------------------------------------------------------

# Shared files across all templates ------------------------------------------

_GITIGNORE = """# Project artifacts
outputs/
checkpoints/
wandb/
*.log
*.tar
*.tar.gz
benchmark_results/local/

# Python
__pycache__/
*.pyc
.pytest_cache/
.venv/
venv/

# Editors
.vscode/
.idea/
*.swp
.DS_Store
"""

_REQUIREMENTS = """# Pinned for reproducibility — bump with care.
# Install the source-tree version of stateset-agents for the full 0.11.6 surface:
#   pip install 'stateset-agents[training,api] @ git+https://github.com/stateset/stateset-agents@14c0e65'
stateset-agents[training,api]>=0.7.1
datasets>=2.0.0
transformers>=4.57.1
accelerate>=0.20.0
peft>=0.4.0
wandb>=0.15.0
"""


# Customer support template ---------------------------------------------------

_CS_CONFIG = """# Training configuration for a multi-turn customer-support agent.
# Edit these values to point at your own dataset, model, and reward weights.

model:
  name: Qwen/Qwen3.5-0.8B
  torch_dtype: bfloat16
  use_lora: true
  lora_r: 16
  lora_alpha: 32

training:
  algorithm: gspo
  seed: 42
  num_epochs: 3
  learning_rate: 0.000005      # 5e-6
  max_prompt_length: 512
  max_completion_length: 320
  num_generations: 4
  clip_range_left: 0.0003      # 3e-4 — see whitepaper §5.2 (tighter than PPO by design)
  clip_range_right: 0.0004     # 4e-4
  warmup_ratio: 0.1
  output_dir: outputs/customer_support_v1

environment:
  max_turns: 4
  scenarios_path: scenarios.jsonl

reward:
  intent_weight: 0.6
  brand_voice_weight: 0.3
  require_safety: true
"""

_CS_SCENARIOS = """\
{"intent": "refund", "user_query": "I want my money back for order #4521", "must_acknowledge": ["refund", "order"], "must_avoid": ["impossible"]}
{"intent": "refund", "user_query": "Cancel my subscription and refund last month", "must_acknowledge": ["refund", "cancel"], "must_avoid": ["impossible"]}
{"intent": "refund", "user_query": "My order arrived damaged — I want a refund", "must_acknowledge": ["refund", "damaged"], "must_avoid": ["impossible"]}
{"intent": "technical", "user_query": "The app crashes every time I open it", "must_acknowledge": ["app", "crash"], "must_avoid": ["your fault"]}
{"intent": "technical", "user_query": "I can't log in — keeps saying invalid password", "must_acknowledge": ["password", "login"], "must_avoid": ["your fault"]}
{"intent": "billing", "user_query": "Why is my bill higher this month?", "must_acknowledge": ["bill"], "must_avoid": ["impossible"]}
{"intent": "billing", "user_query": "I need a copy of last month's invoice", "must_acknowledge": ["invoice"], "must_avoid": ["impossible"]}
{"intent": "general", "user_query": "What are your business hours?", "must_acknowledge": ["hours"], "must_avoid": []}
"""

_CS_REWARD = '''"""Reward function for the customer-support agent.

This is a thin wrapper around the framework's ``SupportRewardComposite`` that
your team can extend with company-specific signals (brand-voice phrases,
escalation triggers, etc.). It loads weights from ``config.yaml`` so the
training script and eval script see the same definition.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from stateset_agents.data.customer_support_bench import SupportRewardComposite


def load_reward(config_path: str | Path = "config.yaml") -> SupportRewardComposite:
    """Construct the project reward from ``config.yaml``."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    r = cfg.get("reward", {})
    return SupportRewardComposite(
        intent_weight=r.get("intent_weight", 0.6),
        brand_voice_weight=r.get("brand_voice_weight", 0.3),
        require_safety=r.get("require_safety", True),
    )
'''

_CS_TRAIN = '''"""Train a multi-turn customer-support agent with GSPO.

Run this after editing ``config.yaml`` and ``scenarios.jsonl``::

    python train.py

The script:

1. Sets all RNG seeds from ``config.yaml``.
2. Loads scenarios from ``scenarios.jsonl``.
3. Builds the environment with the composite reward from ``reward.py``.
4. Fine-tunes Qwen 3.5 0.8B with GSPO (LoRA r=16).
5. Saves the LoRA adapter to ``outputs/<run_name>``.

Use ``serve.sh`` to deploy the resulting adapter.
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import yaml

from reward import load_reward

from stateset_agents.core import ConversationEnvironment, MultiTurnAgent
from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.training import GSPOConfig, GSPOTrainer
from stateset_agents.utils.reproducibility import set_all_seeds


def load_scenarios(path: Path) -> list[dict]:
    scenarios = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))
    return scenarios


async def main() -> None:
    logging.basicConfig(level=logging.INFO)

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    set_all_seeds(cfg["training"]["seed"])

    scenarios = load_scenarios(Path(cfg["environment"]["scenarios_path"]))
    print(f"Loaded {len(scenarios)} scenarios")

    # Map JSONL scenarios into the format ConversationEnvironment expects.
    env_scenarios = [
        {
            "user_query": s["user_query"],
            "intent": s.get("intent", "general"),
            "must_acknowledge": s.get("must_acknowledge", []),
            "must_avoid": s.get("must_avoid", []),
        }
        for s in scenarios
    ]

    agent = MultiTurnAgent(AgentConfig(
        model_name=cfg["model"]["name"],
        torch_dtype=cfg["model"]["torch_dtype"],
        use_peft=cfg["model"].get("use_lora", True),
        peft_config={
            "r": cfg["model"].get("lora_r", 16),
            "lora_alpha": cfg["model"].get("lora_alpha", 32),
            "lora_dropout": 0.05,
        },
    ))
    await agent.initialize()

    reward_fn = load_reward()
    env = ConversationEnvironment(
        scenarios=env_scenarios,
        reward_fn=reward_fn,
        max_turns=cfg["environment"]["max_turns"],
    )

    config = GSPOConfig(
        model_name=cfg["model"]["name"],
        num_generations=cfg["training"]["num_generations"],
        clip_range_left=cfg["training"]["clip_range_left"],
        clip_range_right=cfg["training"]["clip_range_right"],
        learning_rate=cfg["training"]["learning_rate"],
        max_prompt_length=cfg["training"]["max_prompt_length"],
        max_completion_length=cfg["training"]["max_completion_length"],
        warmup_ratio=cfg["training"]["warmup_ratio"],
        num_epochs=cfg["training"]["num_epochs"],
        output_dir=cfg["training"]["output_dir"],
        use_lora=cfg["model"].get("use_lora", True),
        lora_r=cfg["model"].get("lora_r", 16),
        lora_alpha=cfg["model"].get("lora_alpha", 32),
        gradient_checkpointing=True,
    )

    trainer = GSPOTrainer(config=config, agent=agent, environment=env)
    await trainer.train()
    print(f"\\nDone. Adapter saved to {config.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
'''

_CS_EVAL = '''"""Evaluate a trained adapter against held-out scenarios.

Run after ``train.py`` produces ``outputs/<run_name>/``::

    python eval.py --checkpoint outputs/customer_support_v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import yaml

from reward import load_reward

from stateset_agents.core.agent import Agent
from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.core.trajectory import ConversationTurn


def prompt_for(scenario: dict) -> str:
    return (
        "You are a helpful customer support agent. Respond warmly, address "
        "the user's concern, and confirm the next step.\\n\\n"
        f"User: {scenario['user_query']}\\n\\nAgent:"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", help="Path to the trained adapter directory.")
    parser.add_argument("--scenarios", default="scenarios.jsonl")
    args = parser.parse_args()

    cfg = yaml.safe_load(Path("config.yaml").read_text())
    reward = load_reward()

    config = AgentConfig(
        model_name=cfg["model"]["name"],
        torch_dtype=cfg["model"]["torch_dtype"],
        max_new_tokens=cfg["training"]["max_completion_length"],
        temperature=0.0,
        do_sample=False,
    )
    if args.checkpoint:
        config.peft_path = args.checkpoint
    agent = Agent(config=config)
    await agent.initialize()

    scenarios = [json.loads(line) for line in Path(args.scenarios).read_text().splitlines() if line.strip()]
    scores = []
    for s in scenarios:
        response = await agent.generate_response(prompt_for(s))
        result = await reward.compute_reward(
            [ConversationTurn(role="assistant", content=response)],
            context=s,
        )
        scores.append(result.score)
        print(f"[{s.get('intent', '?')}] {result.score:.2f}  {s['user_query'][:60]}")

    print(f"\\nMean composite score across {len(scores)} scenarios: {sum(scores)/max(len(scores),1):.3f}")


if __name__ == "__main__":
    asyncio.run(main())
'''

_CS_SERVE = """#!/usr/bin/env bash
# Serve the trained adapter via the FastAPI gateway.
#
# Usage:
#   ./serve.sh outputs/customer_support_v1
#
set -euo pipefail

CHECKPOINT="${1:-outputs/customer_support_v1}"

if [ ! -d "$CHECKPOINT" ]; then
    echo "Checkpoint not found: $CHECKPOINT" >&2
    echo "Run 'python train.py' first." >&2
    exit 2
fi

exec stateset-agents serve \\
    --checkpoint "$CHECKPOINT" \\
    --base-model "$(python -c 'import yaml,pathlib;print(yaml.safe_load(pathlib.Path(\"config.yaml\").read_text())[\"model\"][\"name\"])')" \\
    --port "${PORT:-8000}"
"""

_CS_README = """# {project_name} — Customer Support Agent

Fine-tune a multi-turn customer-support agent using StateSet Agents GSPO.

This project was scaffolded with `stateset-agents starter customer-support`.

## What's here

| File | Purpose |
|------|---------|
| `config.yaml` | Training + model + reward configuration. **Edit this first.** |
| `scenarios.jsonl` | Sample customer scenarios (8 across 4 intents). Replace with your client's data. |
| `reward.py` | Composite reward (intent ack + brand voice + safety). Extensible. |
| `train.py` | Runnable GSPO training script. |
| `eval.py` | Evaluate a trained adapter on held-out scenarios. |
| `serve.sh` | Serve the trained adapter via the bundled FastAPI gateway. |
| `requirements.txt` | Pinned dependencies. |

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. (Optional) Replace scenarios.jsonl with your client's data —
#    same schema: {{"intent": "...", "user_query": "...", "must_acknowledge": [...], "must_avoid": [...]}}

# 3. Train (Colab A100 recommended; ~3 hours on 16 train + 8 eval scenarios)
python train.py

# 4. Evaluate
python eval.py --checkpoint outputs/customer_support_v1

# 5. Serve
./serve.sh outputs/customer_support_v1
# Then in another shell:
curl -X POST http://localhost:8000/v1/messages \\
  -H "Content-Type: application/json" \\
  -d '{{"model":"trained","messages":[{{"role":"user","content":"I want a refund"}}]}}'
```

## Customizing the reward

The default reward combines three signals — intent acknowledgement, brand
voice, and a safety multiplier. Adjust weights in `config.yaml` or subclass
`SupportRewardComposite` in `reward.py` to add company-specific signals (e.g.,
a brand-vocabulary check, an escalation-trigger detector, an LLM judge).

## Production checklist

- [ ] Replace `scenarios.jsonl` with real customer logs (target: ≥500 scenarios)
- [ ] Add a `LLMJudgeReward` component (see `stateset_agents/rewards/llm_judge.py`)
- [ ] Run 3 seeds (42, 1337, 2026) and verify σ < 0.10 on your eval set
- [ ] Wire up W&B (`report_to: wandb` in config.yaml + `WANDB_PROJECT=...`)
- [ ] Deploy with Helm: `helm install <client> deployment/helm/ -f values-a100.yaml`

## Learn more

- [StateSet Agents whitepaper](https://github.com/stateset/stateset-agents/blob/master/docs/WHITEPAPER.md)
- [GSPO algorithm details](https://github.com/stateset/stateset-agents/blob/master/docs/GSPO_GUIDE.md)
- [Benchmark methodology](https://github.com/stateset/stateset-agents/blob/master/benchmark_results/SCHEMA.md)
"""


# GSM8K math template --------------------------------------------------------

_MATH_CONFIG = """# Training configuration for a math-reasoning agent (GSM8K-style).
model:
  name: Qwen/Qwen3.5-0.8B
  torch_dtype: bfloat16
  use_lora: true
  lora_r: 16

training:
  algorithm: gspo
  seed: 42
  num_epochs: 1
  learning_rate: 0.000005      # 5e-6
  num_generations: 4
  clip_range_left: 0.0003      # 3e-4
  clip_range_right: 0.0004     # 4e-4
  output_dir: outputs/math_v1
  num_train_examples: 200
  num_eval_examples: 100
"""

_MATH_TRAIN = '''"""Train a math-reasoning agent on GSM8K with GSPO."""

from __future__ import annotations

import asyncio
from pathlib import Path

import yaml

from stateset_agents.core import ConversationEnvironment, MultiTurnAgent
from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.data.gsm8k import GSM8KReward, load_gsm8k, make_gsm8k_scenarios
from stateset_agents.training import GSPOConfig, GSPOTrainer
from stateset_agents.utils.reproducibility import set_all_seeds


async def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    set_all_seeds(cfg["training"]["seed"])

    train, _ = load_gsm8k(limit=cfg["training"]["num_train_examples"])

    agent = MultiTurnAgent(AgentConfig(
        model_name=cfg["model"]["name"],
        torch_dtype=cfg["model"]["torch_dtype"],
        use_peft=cfg["model"]["use_lora"],
        peft_config={"r": cfg["model"]["lora_r"], "lora_alpha": 32, "lora_dropout": 0.05},
    ))
    await agent.initialize()

    env = ConversationEnvironment(
        scenarios=make_gsm8k_scenarios(train),
        reward_fn=GSM8KReward(),
        max_turns=1,
    )

    config = GSPOConfig(
        model_name=cfg["model"]["name"],
        num_generations=cfg["training"]["num_generations"],
        clip_range_left=cfg["training"]["clip_range_left"],
        clip_range_right=cfg["training"]["clip_range_right"],
        learning_rate=cfg["training"]["learning_rate"],
        num_epochs=cfg["training"]["num_epochs"],
        output_dir=cfg["training"]["output_dir"],
    )

    trainer = GSPOTrainer(config=config, agent=agent, environment=env)
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
'''

_MATH_README = """# {project_name} — Math Reasoning Agent

Fine-tune a math-reasoning agent on GSM8K-style problems with verifiable rewards.

Scaffolded with `stateset-agents starter gsm8k-math`.

```bash
pip install -r requirements.txt
python train.py
```

The reward is rule-based (numeric-answer match), so every gradient signal is
unambiguous. This is the cheapest path to a defensible "did fine-tuning
work?" number — see the whitepaper §11.7 benchmark methodology.
"""


# Tool-calling agent template ------------------------------------------------

_TOOL_CONFIG = """# Training configuration for a tool-using agent.
model:
  name: Qwen/Qwen3.5-0.8B
  torch_dtype: bfloat16
  use_lora: true
  lora_r: 16

training:
  algorithm: gspo
  seed: 42
  num_epochs: 3
  learning_rate: 0.000005      # 5e-6
  num_generations: 4
  clip_range_left: 0.0003      # 3e-4 — sequence-level ratios need tight clips
  clip_range_right: 0.0004     # 4e-4
  max_prompt_length: 768
  max_completion_length: 320
  output_dir: outputs/tool_agent_v1

environment:
  max_turns: 3
  scenarios_path: scenarios.jsonl

reward:
  # Weights for the composite tool-call reward (see reward.py).
  tool_selection_weight: 0.4   # picked the right tool for the user's query
  param_correctness_weight: 0.3   # parameters parse as JSON and match schema
  outcome_weight: 0.3          # final answer matches the expected result
"""

_TOOL_SCENARIOS = """\
{"user_query": "What's the weather in San Francisco?", "expected_tool": "get_weather", "expected_params": {"city": "San Francisco"}, "expected_outcome": "63"}
{"user_query": "Calculate 17 * 24", "expected_tool": "calculator", "expected_params": {"expression": "17 * 24"}, "expected_outcome": "408"}
{"user_query": "Look up the population of Tokyo", "expected_tool": "search", "expected_params": {"query": "population of Tokyo"}, "expected_outcome": "13.96"}
{"user_query": "Find recent papers on diffusion models", "expected_tool": "search", "expected_params": {"query": "diffusion models"}, "expected_outcome": "papers"}
{"user_query": "Calculate the square root of 144", "expected_tool": "calculator", "expected_params": {"expression": "sqrt(144)"}, "expected_outcome": "12"}
{"user_query": "What's the weather forecast for Paris tomorrow?", "expected_tool": "get_weather", "expected_params": {"city": "Paris"}, "expected_outcome": "58"}
"""

_TOOL_REWARD = '''"""Composite reward for tool-calling agents.

This file is a thin wrapper around ``stateset_agents.data.tool_calling_bench.ToolCallReward``
that loads weights from ``config.yaml``. Edit the weights in config (or
subclass ``ToolCallReward`` here) to inject company-specific signals.

The reward parses the standard StateSet Agents tool-call JSON block:

    ```json
    {"tool": "calculator", "parameters": {"expression": "17 * 24"}}
    ```

Three signals (weights in config.yaml):

* ``tool_selection`` — did the agent invoke the right tool?
* ``param_correctness`` — did the JSON parameters match the expected schema?
* ``outcome`` — does the final answer contain the expected ground-truth string?
"""

from __future__ import annotations

from pathlib import Path

import yaml

from stateset_agents.data.tool_calling_bench import ToolCallReward


def load_reward(config_path: str | Path = "config.yaml") -> ToolCallReward:
    cfg = yaml.safe_load(Path(config_path).read_text())
    r = cfg.get("reward", {})
    return ToolCallReward(
        tool_selection_weight=r.get("tool_selection_weight", 0.4),
        param_correctness_weight=r.get("param_correctness_weight", 0.3),
        outcome_weight=r.get("outcome_weight", 0.3),
    )
'''

_TOOL_TOOLS = '''"""Tool registry for the tool-calling agent.

Re-exports the framework's bundled ``SAMPLE_TOOLS`` (weather + calculator +
search stubs) and lets you append client-specific tools without forking.

Replace the stubs with calls to your real APIs (Slack, Stripe, your CRM, etc.)
by adding entries to ``CUSTOM_TOOLS``. Each entry must conform to the
``ToolAgent`` schema: ``name``, ``description``, ``parameters`` (JSON Schema).
"""

from __future__ import annotations

from stateset_agents.data.tool_calling_bench import SAMPLE_TOOLS as _BUNDLED_TOOLS


# Add your client's tools here — each entry uses the same schema as the bundled ones.
CUSTOM_TOOLS: list[dict] = [
    # Example:
    # {
    #     "name": "send_slack",
    #     "description": "Send a message to a Slack channel.",
    #     "parameters": {
    #         "type": "object",
    #         "properties": {
    #             "channel": {"type": "string"},
    #             "text": {"type": "string"},
    #         },
    #         "required": ["channel", "text"],
    #     },
    # },
]

SAMPLE_TOOLS = list(_BUNDLED_TOOLS) + CUSTOM_TOOLS
'''

_TOOL_TRAIN = '''"""Train a tool-calling agent with GSPO.

Run after editing ``config.yaml``, ``scenarios.jsonl``, and ``tools.py``::

    python train.py
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import yaml

from reward import load_reward
from tools import SAMPLE_TOOLS

from stateset_agents.core import ConversationEnvironment
from stateset_agents.core.tool_agent import ToolAgent
from stateset_agents.core.agent_config import AgentConfig
from stateset_agents.training import GSPOConfig, GSPOTrainer
from stateset_agents.utils.reproducibility import set_all_seeds


async def main() -> None:
    cfg = yaml.safe_load(Path("config.yaml").read_text())
    set_all_seeds(cfg["training"]["seed"])

    scenarios = [
        json.loads(line)
        for line in Path(cfg["environment"]["scenarios_path"]).read_text().splitlines()
        if line.strip()
    ]
    print(f"Loaded {len(scenarios)} scenarios with {len(SAMPLE_TOOLS)} tools")

    agent = ToolAgent(
        config=AgentConfig(
            model_name=cfg["model"]["name"],
            torch_dtype=cfg["model"]["torch_dtype"],
            use_peft=cfg["model"]["use_lora"],
            peft_config={"r": cfg["model"]["lora_r"], "lora_alpha": 32, "lora_dropout": 0.05},
        ),
        tools=SAMPLE_TOOLS,
    )
    await agent.initialize()

    reward_fn = load_reward()
    env = ConversationEnvironment(
        scenarios=scenarios,
        reward_fn=reward_fn,
        max_turns=cfg["environment"]["max_turns"],
    )

    config = GSPOConfig(
        model_name=cfg["model"]["name"],
        num_generations=cfg["training"]["num_generations"],
        clip_range_left=cfg["training"]["clip_range_left"],
        clip_range_right=cfg["training"]["clip_range_right"],
        learning_rate=cfg["training"]["learning_rate"],
        max_prompt_length=cfg["training"]["max_prompt_length"],
        max_completion_length=cfg["training"]["max_completion_length"],
        num_epochs=cfg["training"]["num_epochs"],
        output_dir=cfg["training"]["output_dir"],
        use_lora=cfg["model"]["use_lora"],
        lora_r=cfg["model"]["lora_r"],
        gradient_checkpointing=True,
    )

    trainer = GSPOTrainer(config=config, agent=agent, environment=env)
    await trainer.train()
    print(f"\\nDone. Adapter saved to {config.output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
'''

_TOOL_README = """# {project_name} — Tool-Calling Agent

Fine-tune an agent that learns to invoke tools (functions / APIs) on demand,
scaffolded with `stateset-agents starter tool-calling-agent`.

## What's here

| File | Purpose |
|------|---------|
| `config.yaml` | Training + reward configuration. |
| `scenarios.jsonl` | Sample user queries with expected tool + parameters + outcome. |
| `tools.py` | Tool registry (3 stub tools — weather, calculator, search). |
| `reward.py` | `ToolCallReward` — three signals: tool selection, parameter correctness, outcome match. |
| `train.py` | Runnable GSPO training script. |
| `requirements.txt` | Pinned deps. |

## Quickstart

```bash
pip install -r requirements.txt

# Replace the stub tools in tools.py with calls to your real APIs.
# Replace scenarios.jsonl with your client's user queries and expected tool calls.

python train.py
```

## How the reward works

The agent emits responses containing JSON tool-call blocks:

````
```json
{{"tool": "calculator", "parameters": {{"expression": "17 * 24"}}}}
```
````

`ToolCallReward` parses the first such block from every assistant turn,
compares the tool name and parameters against the scenario's `expected_tool`
and `expected_params`, and checks whether the final response contains
`expected_outcome` as a substring. The three sub-scores are weighted per
`config.yaml`.

## Production checklist

- [ ] Replace the stub tools with real API clients (use `httpx.AsyncClient`)
- [ ] Replace `scenarios.jsonl` with real user queries + expected tool calls
  (200+ for a useful eval signal)
- [ ] Add a `SafetyReward` component if any tool has destructive side effects
- [ ] Wire up W&B for training metrics
"""


# Minimal template -----------------------------------------------------------

_MIN_CONFIG = """# Minimal scaffold: edit to fit your task.
model:
  name: Qwen/Qwen3.5-0.8B
  torch_dtype: bfloat16

training:
  algorithm: gspo
  seed: 42
  output_dir: outputs/minimal_v1
"""

_MIN_TRAIN = '''"""Minimal trainer scaffold. Edit and extend."""

from __future__ import annotations

import asyncio


async def main() -> None:
    print("Fill in your training logic here.")
    print("See `python -m stateset_agents.cli starter --help` for richer templates.")


if __name__ == "__main__":
    asyncio.run(main())
'''

_MIN_README = """# {project_name}

Minimal StateSet Agents scaffold. Run::

    pip install -r requirements.txt
    python train.py

For a fuller starting point, regenerate with::

    stateset-agents starter customer-support ./my-project
"""


# ---------------------------------------------------------------------------
# Template registry
# ---------------------------------------------------------------------------


@dataclass
class StarterTemplate:
    """Describes one scaffold template."""

    name: str
    description: str
    files: dict[str, str]  # relative path → contents

    def render(self, project_name: str) -> dict[str, str]:
        """Substitute ``{project_name}`` into every file. Returns a dict copy."""
        return {
            path: (
                content.format(project_name=project_name)
                if "{project_name}" in content
                else content
            )
            for path, content in self.files.items()
        }


_COMMON_FILES = {
    ".gitignore": _GITIGNORE,
    "requirements.txt": _REQUIREMENTS,
}


SCAFFOLD_TEMPLATES: dict[str, StarterTemplate] = {
    "customer-support": StarterTemplate(
        name="customer-support",
        description="Multi-turn customer-support agent (the framework's differentiator).",
        files={
            **_COMMON_FILES,
            "config.yaml": _CS_CONFIG,
            "scenarios.jsonl": _CS_SCENARIOS,
            "reward.py": _CS_REWARD,
            "train.py": _CS_TRAIN,
            "eval.py": _CS_EVAL,
            "serve.sh": _CS_SERVE,
            "README.md": _CS_README,
        },
    ),
    "gsm8k-math": StarterTemplate(
        name="gsm8k-math",
        description="Single-turn math-reasoning agent with verifiable rewards.",
        files={
            **_COMMON_FILES,
            "config.yaml": _MATH_CONFIG,
            "train.py": _MATH_TRAIN,
            "README.md": _MATH_README,
        },
    ),
    "tool-calling-agent": StarterTemplate(
        name="tool-calling-agent",
        description="Agent that learns to invoke tools/APIs (weather, calculator, search).",
        files={
            **_COMMON_FILES,
            "config.yaml": _TOOL_CONFIG,
            "scenarios.jsonl": _TOOL_SCENARIOS,
            "tools.py": _TOOL_TOOLS,
            "reward.py": _TOOL_REWARD,
            "train.py": _TOOL_TRAIN,
            "README.md": _TOOL_README,
        },
    ),
    "minimal": StarterTemplate(
        name="minimal",
        description="Bare scaffold — edit everything.",
        files={
            **_COMMON_FILES,
            "config.yaml": _MIN_CONFIG,
            "train.py": _MIN_TRAIN,
            "README.md": _MIN_README,
        },
    ),
}


def list_templates() -> list[StarterTemplate]:
    """Return all available templates, sorted by name."""
    return sorted(SCAFFOLD_TEMPLATES.values(), key=lambda t: t.name)


def _apply_client_customizations(
    content: str, client_name: str, project_name: str
) -> str:
    """Patch generated content with client-specific values where it makes sense.

    Replaces output_dir paths, W&B project names, and config-yaml `wandb_project`
    fields so a freshly-scaffolded project lands ready for a named engagement.
    Idempotent: safe to call on already-customized content.
    """
    # Slugify the client name for paths (alphanumeric + underscore).
    slug = "".join(c if c.isalnum() else "_" for c in client_name.lower()).strip("_")
    if not slug:
        return content

    out = content
    # Per-template output_dir conventions: outputs/<template>_v1 -> outputs/<slug>_v1
    for default in (
        "outputs/customer_support_v1",
        "outputs/tool_agent_v1",
        "outputs/math_v1",
        "outputs/minimal_v1",
    ):
        out = out.replace(default, f"outputs/{slug}_v1")
    # Add a wandb_project line under training: if config.yaml doesn't have one.
    if "training:" in out and "wandb_project" not in out:
        out = out.replace(
            "  algorithm: gspo\n",
            f"  algorithm: gspo\n  wandb_project: {slug}\n",
            1,
        )
    return out


def scaffold_project(
    template_name: str,
    output_dir: str | Path,
    project_name: str | None = None,
    force: bool = False,
    client_name: str | None = None,
) -> list[Path]:
    """Materialize a starter template into ``output_dir``.

    Args:
        template_name: Key into ``SCAFFOLD_TEMPLATES`` (e.g. ``"customer-support"``).
        output_dir: Where to write the project. Parent directories are created
            as needed. If the directory already exists and is non-empty, the
            call raises unless ``force=True``.
        project_name: Substituted for ``{project_name}`` in templates. Defaults
            to the basename of ``output_dir``.
        force: Overwrite existing files. Use with care.
        client_name: If set, slugified and used to customize generated
            ``output_dir`` paths and the W&B project name. The customer-support
            consultant's "type this once and the whole project is named"
            convenience.

    Returns:
        The list of file paths created.

    Raises:
        KeyError: ``template_name`` is unknown.
        FileExistsError: ``output_dir`` is non-empty and ``force=False``.
    """
    if template_name not in SCAFFOLD_TEMPLATES:
        available = ", ".join(sorted(SCAFFOLD_TEMPLATES))
        raise KeyError(f"Unknown template {template_name!r}. Available: {available}")

    out = Path(output_dir)
    if out.exists() and any(out.iterdir()) and not force:
        raise FileExistsError(
            f"{out} is non-empty. Pass force=True (or --force) to overwrite."
        )
    out.mkdir(parents=True, exist_ok=True)

    name = project_name or out.name
    template = SCAFFOLD_TEMPLATES[template_name]
    files = template.render(name)

    created: list[Path] = []
    for rel_path, content in files.items():
        if client_name:
            content = _apply_client_customizations(content, client_name, name)
        dst = out / rel_path
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(content)
        if rel_path.endswith(".sh"):
            dst.chmod(0o755)
        created.append(dst)

    # Drop a small marker so the CLI can show next steps without reinventing them.
    marker: dict[str, Any] = {
        "template": template_name,
        "project_name": name,
    }
    if client_name:
        marker["client_name"] = client_name
    (out / ".stateset-agents-starter.json").write_text(json.dumps(marker, indent=2))
    created.append(out / ".stateset-agents-starter.json")

    logger.info(
        "Scaffolded %s template into %s (%d files)", template_name, out, len(created)
    )
    return created
