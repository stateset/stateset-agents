"""Factory, preset, and checkpoint helpers for core agents."""

from __future__ import annotations

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
    """Atomically publish inference components; config-only saves use a new path.

    Stop concurrent training/generation first. This does not save conversations,
    optimizer state, or external rollout engines. Config fields must be JSON data.
    """
    from .agent_checkpoint import save_checkpoint

    save_checkpoint(
        agent, Path(checkpoint_path).expanduser().absolute(), save_model=save_model
    )


async def load_agent_from_checkpoint(
    checkpoint_path: str, load_model: bool = True
) -> MultiTurnAgent:
    """Restore saved weights; ``load_model=False`` defers component loading.

    Configuration-only snapshots deliberately initialize their original source.
    Adapter snapshots require their recorded base model to remain available.
    """
    from .agent_checkpoint import load_checkpoint

    return await load_checkpoint(Path(checkpoint_path), load_model=load_model)


def get_preset_config(preset_name: str, **overrides) -> AgentConfig:
    """Build an AgentConfig from a named preset plus overrides."""
    if preset_name not in AGENT_CONFIGS:
        raise ValueError(
            f"Unknown preset: {preset_name}. Available: {list(AGENT_CONFIGS.keys())}"
        )

    preset = AGENT_CONFIGS[preset_name].copy()
    preset.update(overrides)
    return AgentConfig(**preset)
