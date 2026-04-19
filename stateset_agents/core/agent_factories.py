"""Factory, preset, and checkpoint helpers for core agents."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .agent_config import AgentConfig

if TYPE_CHECKING:
    from .agent import Agent, MultiTurnAgent


def _load_agent_configs() -> dict[str, Any]:
    """Load agent configurations from YAML presets, with hardcoded fallbacks."""
    configs = {}

    try:
        from stateset_agents.config import get_agent_preset, list_agent_presets

        for preset_name in list_agent_presets():
            try:
                configs[preset_name] = get_agent_preset(preset_name)
            except (KeyError, ValueError, TypeError, RuntimeError):
                pass
    except ImportError:
        pass

    fallback_configs = {
        "helpful_assistant": {
            "system_prompt": "You are a helpful, harmless, and honest AI assistant. Provide clear, accurate, and helpful responses.",
            "temperature": 0.7,
            "max_new_tokens": 512,
        },
        "customer_service": {
            "system_prompt": "You are a professional customer service representative. Be polite, helpful, and solution-oriented.",
            "temperature": 0.6,
            "max_new_tokens": 256,
        },
        "tutor": {
            "system_prompt": "You are a patient and encouraging tutor. Break down complex topics and guide students step-by-step.",
            "temperature": 0.8,
            "max_new_tokens": 512,
        },
        "creative_writer": {
            "system_prompt": "You are a creative writing assistant. Help with storytelling, character development, and creative expression.",
            "temperature": 0.9,
            "max_new_tokens": 1024,
        },
    }

    for name, config in fallback_configs.items():
        if name not in configs:
            configs[name] = config

    return configs


AGENT_CONFIGS = _load_agent_configs()


def create_agent(
    agent_type: str = "multi_turn", model_name: str = "stub://default", **kwargs
) -> Agent | MultiTurnAgent:
    """Create an agent of the specified type."""
    from .agent import MultiTurnAgent

    config = AgentConfig(model_name=model_name, **kwargs)

    if agent_type == "multi_turn":
        return MultiTurnAgent(config)
    raise ValueError(f"Unknown agent type: {agent_type}")


def create_peft_agent(
    model_name: str, peft_config: dict[str, Any], **kwargs
) -> MultiTurnAgent:
    """Create a PEFT-enabled agent with LoRA."""
    from .agent import MultiTurnAgent, _load_peft

    if not _load_peft():
        raise ImportError("PEFT library not available. Install with: pip install peft")

    config = AgentConfig(
        model_name=model_name, use_peft=True, peft_config=peft_config, **kwargs
    )
    return MultiTurnAgent(config)


async def save_agent_checkpoint(
    agent: Agent, checkpoint_path: str, save_model: bool = True
) -> None:
    """Save an agent checkpoint."""
    checkpoint_dir = Path(checkpoint_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if save_model and agent.model is not None:
        agent.model.save_pretrained(checkpoint_dir)
        if agent.tokenizer is not None:
            agent.tokenizer.save_pretrained(checkpoint_dir)

    config_path = checkpoint_dir / "agent_config.json"
    with open(config_path, "w") as f:
        json.dump(asdict(agent.config), f, indent=2, default=str)


async def load_agent_from_checkpoint(
    checkpoint_path: str, load_model: bool = True
) -> MultiTurnAgent:
    """Load an agent from checkpoint."""
    from .agent import MultiTurnAgent

    checkpoint_dir = Path(checkpoint_path)
    config_path = checkpoint_dir / "agent_config.json"
    with open(config_path) as f:
        config_dict = json.load(f)

    agent = MultiTurnAgent(AgentConfig(**config_dict))
    if load_model:
        await agent.initialize()
    return agent


def get_preset_config(preset_name: str, **overrides) -> AgentConfig:
    """Build an AgentConfig from a named preset plus overrides."""
    if preset_name not in AGENT_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(AGENT_CONFIGS.keys())}"
        )

    preset = AGENT_CONFIGS[preset_name].copy()
    preset.update(overrides)
    return AgentConfig(**preset)
