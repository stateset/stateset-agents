"""The ``init`` scaffolding wizard for the StateSet Agents CLI.

Split out of ``stateset_agents/cli.py``. The command attaches to the parent
Typer app exported by ``cli``, following the sibling ``cli_chat`` /
``cli_train`` pattern.

Every preset except ``default`` resolves the same way: import a starter module,
validate ``--task`` and ``--starter-profile`` against that module's choice
tuples, then call its ``get_*_config`` factory. That shape is expressed once, as
:data:`STARTER_PRESETS`, instead of ten copies of the same twenty lines. The
``default`` preset is the odd one out: it has no starter module and its YAML is
a hand-written, commented template rather than a ``yaml.safe_dump``.
"""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app

_echo = _cli._echo
CLI_IMPORT_EXCEPTIONS = _cli.CLI_IMPORT_EXCEPTIONS


@dataclass(frozen=True)
class StarterPreset:
    """How one ``--preset`` finds its config factory and choice tuples."""

    #: Dotted path of the starter module under ``stateset_agents.training``.
    module: str
    #: Name of the ``get_*_config(task=..., starter_profile=...)`` factory.
    factory: str
    #: Name of the module's tuple of valid ``--task`` values.
    task_choices: str
    #: Name of the module's tuple of valid ``--starter-profile`` values.
    profile_choices: str
    #: Human label used in the "helpers unavailable" message.
    label: str


#: Model-specific presets, keyed by the ``--preset`` value.
STARTER_PRESETS: dict[str, StarterPreset] = {
    "qwen3-5-0-8b": StarterPreset(
        module="stateset_agents.training.qwen3_5_starter",
        factory="get_qwen3_5_config",
        task_choices="QWEN35_08B_TASK_CHOICES",
        profile_choices="QWEN35_08B_STARTER_PROFILE_CHOICES",
        label="Qwen3.5-0.8B",
    ),
    "kimi-k2-6": StarterPreset(
        module="stateset_agents.training.kimi_k2_6_starter",
        factory="get_kimi_k2_6_config",
        task_choices="KIMI_K26_TASK_CHOICES",
        profile_choices="KIMI_K26_STARTER_PROFILE_CHOICES",
        label="Kimi-K2.6",
    ),
    "kimi-k3": StarterPreset(
        module="stateset_agents.training.kimi_k3_starter",
        factory="get_kimi_k3_config",
        task_choices="KIMI_K3_TASK_CHOICES",
        profile_choices="KIMI_K3_STARTER_PROFILE_CHOICES",
        label="Kimi-K3",
    ),
    "muse-glimmer": StarterPreset(
        module="stateset_agents.training.muse_glimmer_starter",
        factory="get_muse_glimmer_config",
        task_choices="MUSE_GLIMMER_TASK_CHOICES",
        profile_choices="MUSE_GLIMMER_STARTER_PROFILE_CHOICES",
        label="Muse Glimmer",
    ),
    "nemotron-3-5": StarterPreset(
        module="stateset_agents.training.nemotron_3_5_starter",
        factory="get_nemotron_3_5_config",
        task_choices="NEMOTRON_3_5_TASK_CHOICES",
        profile_choices="NEMOTRON_3_5_STARTER_PROFILE_CHOICES",
        label="Nemotron 3.5",
    ),
    "qwen3.8-27b": StarterPreset(
        module="stateset_agents.training.qwen3_8_starter",
        factory="get_qwen3_8_config",
        task_choices="QWEN38_27B_TASK_CHOICES",
        profile_choices="QWEN38_27B_STARTER_PROFILE_CHOICES",
        label="Qwen3.8 27B",
    ),
    "qwen3-coder": StarterPreset(
        module="stateset_agents.training.qwen3_coder_starter",
        factory="get_qwen3_coder_config",
        task_choices="QWEN3_CODER_TASK_CHOICES",
        profile_choices="QWEN3_CODER_STARTER_PROFILE_CHOICES",
        label="Qwen3 Coder",
    ),
    "gpt-oss": StarterPreset(
        module="stateset_agents.training.gpt_oss_starter",
        factory="get_gpt_oss_config",
        task_choices="GPT_OSS_TASK_CHOICES",
        profile_choices="GPT_OSS_STARTER_PROFILE_CHOICES",
        label="gpt-oss",
    ),
    "deepseek-v4": StarterPreset(
        module="stateset_agents.training.deepseek_v4_starter",
        factory="get_deepseek_v4_config",
        task_choices="DEEPSEEK_V4_TASK_CHOICES",
        profile_choices="DEEPSEEK_V4_STARTER_PROFILE_CHOICES",
        label="deepseek-v4",
    ),
    "gemma-4-31b": StarterPreset(
        module="stateset_agents.training.gemma4_starter",
        factory="get_gemma4_31b_config",
        task_choices="GEMMA4_31B_TASK_CHOICES",
        profile_choices="GEMMA4_31B_STARTER_PROFILE_CHOICES",
        label="Gemma 4 31B",
    ),
}

#: Every accepted ``--preset`` value, in the order the help text lists them.
PRESET_CHOICES = (
    "default",
    "qwen3-5-0-8b",
    "kimi-k2-6",
    "kimi-k3",
    "gemma-4-31b",
    "muse-glimmer",
    "nemotron-3-5",
    "qwen3.8-27b",
    "qwen3-coder",
    "gpt-oss",
    "deepseek-v4",
)

_PRESET_LIST = ", ".join(PRESET_CHOICES)
_MODEL_PRESET_LIST = ", ".join(PRESET_CHOICES[1:-1]) + f", or {PRESET_CHOICES[-1]}"

#: The ``default`` preset's config, mirrored by the YAML template below.
DEFAULT_CONFIG: dict[str, Any] = {
    "agent": {"model_name": "gpt2", "max_new_tokens": 64, "temperature": 0.7},
    "training": {"num_episodes": 5, "max_turns": 3},
    "environment": {
        "type": "conversation",
        "scenarios": [
            {
                "id": "demo",
                "topic": "general_help",
                "context": "User needs general assistance",
                "user_responses": [
                    "Thanks! Can you elaborate?",
                    "Interesting, tell me more.",
                ],
            }
        ],
    },
}

#: Hand-written YAML for the ``default`` preset — commented and ordered for a
#: human reader, which ``yaml.safe_dump`` would not preserve.
DEFAULT_CONFIG_YAML = (
    "# StateSet Agents - Starter Config\n"
    "agent:\n"
    "  model_name: gpt2\n"
    "  max_new_tokens: 64\n"
    "  temperature: 0.7\n"
    "\n"
    "training:\n"
    "  num_episodes: 5\n"
    "  max_turns: 3\n"
    "\n"
    "environment:\n"
    "  type: conversation\n"
    "  scenarios:\n"
    "    - id: demo\n"
    "      topic: general_help\n"
    "      context: User needs general assistance\n"
    "      user_responses:\n"
    "        - Thanks! Can you elaborate?\n"
    "        - Interesting, tell me more.\n"
)


def _validate_options(
    path: str, overwrite: bool, format: str, preset: str, starter_profile: str
) -> Path:
    """Check the flag combination and return the destination path."""
    if format not in {"yaml", "yml", "json"}:
        _echo("format must be yaml or json")
        raise typer.Exit(code=2)

    if preset not in set(PRESET_CHOICES):
        _echo(f"Unsupported preset. Use one of: {_PRESET_LIST}.")
        raise typer.Exit(code=2)

    if preset == "default" and starter_profile != "balanced":
        _echo(f"`--starter-profile` only applies to --preset {_MODEL_PRESET_LIST}.")
        raise typer.Exit(code=2)

    config_path = Path(path)
    if config_path.exists() and not overwrite:
        _echo(f"Config already exists: {path}. Use --overwrite to replace it.")
        raise typer.Exit(code=2)
    return config_path


def _dump_yaml(cfg: dict[str, Any]) -> str:
    """Serialize a starter config to YAML, or exit if PyYAML is absent."""
    try:
        import yaml
    except ImportError as e:
        _echo(
            "PyYAML is required for YAML starter configs. Install with: pip install pyyaml"
        )
        raise typer.Exit(code=2) from e
    return str(yaml.safe_dump(cfg, sort_keys=False))


def _serialize(cfg: dict[str, Any], format: str) -> str:
    """Serialize a starter config in the requested ``--format``."""
    if format == "json":
        return json.dumps(cfg, indent=2) + "\n"
    return _dump_yaml(cfg)


def _starter_config(spec: StarterPreset, task: str, starter_profile: str) -> Any:
    """Import a starter module, validate the choices, and build its config."""
    try:
        module = importlib.import_module(spec.module)
        task_choices = getattr(module, spec.task_choices)
        profile_choices = getattr(module, spec.profile_choices)
        factory = getattr(module, spec.factory)
    except CLI_IMPORT_EXCEPTIONS as e:
        _echo(f"{spec.label} starter helpers unavailable. Install training extras.")
        _echo(f"Details: {e}")
        raise typer.Exit(code=2) from e

    if task not in task_choices:
        _echo(f"Unsupported task. Use one of: {', '.join(task_choices)}.")
        raise typer.Exit(code=2)
    if starter_profile not in profile_choices:
        _echo(f"Unsupported starter profile. Use one of: {', '.join(profile_choices)}.")
        raise typer.Exit(code=2)

    return factory(task=task, starter_profile=starter_profile).to_dict()


def _render_config(preset: str, format: str, task: str, starter_profile: str) -> str:
    """Produce the file body for ``--preset`` in the requested format."""
    if preset == "default":
        if format == "json":
            return json.dumps(DEFAULT_CONFIG, indent=2) + "\n"
        return DEFAULT_CONFIG_YAML

    cfg = _starter_config(STARTER_PRESETS[preset], task, starter_profile)
    return _serialize(cfg, format)


def _write_config(config_path: Path, path: str, serialized: str) -> None:
    """Write the rendered config, creating parent directories as needed."""
    try:
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(serialized, encoding="utf-8")
        _echo(f"Wrote starter config to {path}")
    except OSError as e:
        _echo(f"Failed to write config: {e}")
        raise typer.Exit(code=2) from e


@app.command()
def init(
    path: str = typer.Option(
        "./stateset_agents.yaml", help="Path for a starter config"
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing file"
    ),
    format: str = typer.Option(
        "yaml",
        "--format",
        "-f",
        help="Output format: yaml or json",
    ),
    preset: str = typer.Option(
        "default",
        "--preset",
        help="Starter preset: default, qwen3-5-0-8b, kimi-k2-6, kimi-k3, gemma-4-31b, muse-glimmer, nemotron-3-5, qwen3.8-27b, qwen3-coder, gpt-oss, or deepseek-v4",
    ),
    task: str = typer.Option(
        "customer_service",
        "--task",
        help="Task preset for model-specific starter presets.",
    ),
    starter_profile: str = typer.Option(
        "balanced",
        "--starter-profile",
        help="Starter profile for model-specific starter presets.",
    ),
) -> None:
    """Scaffold a starter config to get started."""
    config_path = _validate_options(path, overwrite, format, preset, starter_profile)
    serialized = _render_config(preset, format, task, starter_profile)
    _write_config(config_path, path, serialized)
