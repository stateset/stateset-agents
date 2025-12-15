"""
Configuration management for StateSet Agents.

This module provides utilities for loading and managing agent, environment,
and training configurations from YAML/JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)

# Default presets directory
PRESETS_DIR = Path(__file__).parent / "presets"


def _load_yaml(path: Path) -> Dict[str, Any]:
    """Load a YAML file, falling back to JSON if yaml is unavailable."""
    try:
        import yaml
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        logger.warning("PyYAML not installed, falling back to JSON parsing")
        with open(path, "r") as f:
            return json.load(f)


def _load_json(path: Path) -> Dict[str, Any]:
    """Load a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_config(path: Union[str, Path]) -> Dict[str, Any]:
    """Load a configuration file (YAML or JSON).

    Args:
        path: Path to the configuration file

    Returns:
        Dict containing the configuration

    Raises:
        FileNotFoundError: If the file doesn't exist
        ValueError: If the file format is not supported
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    elif suffix == ".json":
        return _load_json(path)
    else:
        raise ValueError(f"Unsupported configuration format: {suffix}")


def load_preset(
    category: str,
    name: str,
    presets_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """Load a preset configuration by category and name.

    Args:
        category: Configuration category (e.g., "agents", "environments")
        name: Preset name (e.g., "customer_service", "tutoring")
        presets_dir: Optional custom presets directory

    Returns:
        Dict containing the preset configuration

    Raises:
        FileNotFoundError: If the preset doesn't exist
    """
    presets_dir = presets_dir or PRESETS_DIR

    # Try YAML first, then JSON
    for ext in (".yaml", ".yml", ".json"):
        preset_path = presets_dir / category / f"{name}{ext}"
        if preset_path.exists():
            return load_config(preset_path)

    raise FileNotFoundError(
        f"Preset '{name}' not found in category '{category}'. "
        f"Looked in: {presets_dir / category}"
    )


def list_presets(
    category: str,
    presets_dir: Optional[Path] = None,
) -> list:
    """List available presets in a category.

    Args:
        category: Configuration category
        presets_dir: Optional custom presets directory

    Returns:
        List of preset names
    """
    presets_dir = presets_dir or PRESETS_DIR
    category_dir = presets_dir / category

    if not category_dir.exists():
        return []

    presets = []
    for file in category_dir.iterdir():
        if file.suffix.lower() in (".yaml", ".yml", ".json"):
            presets.append(file.stem)

    return sorted(set(presets))


def get_agent_preset(name: str) -> Dict[str, Any]:
    """Load an agent preset configuration.

    Args:
        name: Agent preset name (e.g., "customer_service", "helpful_assistant")

    Returns:
        Agent configuration dict
    """
    return load_preset("agents", name)


def get_environment_preset(name: str) -> Dict[str, Any]:
    """Load an environment preset configuration.

    Args:
        name: Environment preset name (e.g., "customer_service", "tutoring")

    Returns:
        Environment configuration dict
    """
    return load_preset("environments", name)


def list_agent_presets() -> list:
    """List available agent presets."""
    return list_presets("agents")


def list_environment_presets() -> list:
    """List available environment presets."""
    return list_presets("environments")
