"""Meta-tests for the data-driven per-model training commands.

``stateset_agents/cli_train.py`` generates one Typer command per model preset
that carries a ``cli_command`` name. These tests keep the generator and the
registry in sync: every preset with a ``cli_command`` must be registered, every
generated command must map back to exactly one preset, and each command's
``--help`` must run and mention that preset's ``model_id``.
"""

from __future__ import annotations

import re

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_train
from stateset_agents.cli import app
from stateset_agents.core.model_presets import PRESETS, ModelPreset

CLI_PRESETS: dict[str, ModelPreset] = {
    preset.cli_command: preset
    for preset in PRESETS.values()
    if preset.cli_command is not None
}

EXPECTED_COMMANDS = {
    "qwen3-5-0-8b",
    "kimi-k2-6",
    "kimi-k3",
    "gemma-4-31b",
    "muse-glimmer",
    "nemotron-3-5",
    "qwen3-8-27b",
    "qwen3-coder",
    "gpt-oss",
    "deepseek-v4",
}


def _registered_names() -> set[str]:
    return {info.name for info in app.registered_commands if info.name is not None}


def test_expected_commands_have_presets() -> None:
    assert set(CLI_PRESETS) == EXPECTED_COMMANDS


def test_cli_command_names_are_unique() -> None:
    names = [p.cli_command for p in PRESETS.values() if p.cli_command is not None]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("command", sorted(EXPECTED_COMMANDS))
def test_every_cli_preset_is_registered(command: str) -> None:
    assert command in _registered_names()


def test_every_generated_command_maps_to_one_preset() -> None:
    generated = set(cli_train.GENERATED_MODEL_COMMANDS)
    assert generated == set(CLI_PRESETS)
    assert generated <= _registered_names()


@pytest.mark.parametrize("command", sorted(EXPECTED_COMMANDS))
def test_command_help_mentions_model_id(command: str) -> None:
    result = CliRunner().invoke(app, [command, "--help"], env={"COLUMNS": "200"})
    assert result.exit_code == 0, result.output
    text = re.sub(r"\s+", " ", result.output)
    assert CLI_PRESETS[command].model_id in text
