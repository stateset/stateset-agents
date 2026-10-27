"""The ``flywheel`` command's CLI surface is pinned while its body is split
into typed helpers (code-quality review): every option keeps its name, flags,
default, help text, type and requiredness, and ``FlywheelOptions`` mirrors
the command signature so no parameter is dropped on the way to the helpers.

``goldens/flywheel_cli_params.json`` was captured from the pre-refactor
command; ``{providers}`` stands for the live provider registry.
"""

from __future__ import annotations

import inspect
import json
from dataclasses import fields
from pathlib import Path

import typer
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.cli_flywheel import FlywheelOptions, flywheel
from stateset_agents.remote.registry import available_providers

GOLDEN = Path(__file__).parent / "goldens" / "flywheel_cli_params.json"
_PLAIN_TERMINAL = {
    "COLUMNS": "200",
    "NO_COLOR": "1",
    "TERM": "dumb",
    "FORCE_COLOR": "0",
}


def _command() -> typer.core.TyperCommand:
    return typer.main.get_command(app).commands["flywheel"]


def _describe(param) -> dict:
    default = param.default
    rendered = (
        f"Path({str(Path(default).as_posix())!r})"
        if isinstance(default, Path)
        else repr(default)
    )
    return {
        "default": rendered,
        "help": getattr(param, "help", None),
        "is_flag": getattr(param, "is_flag", None),
        "name": param.name,
        "opts": list(param.opts),
        "required": param.required,
        "secondary": list(param.secondary_opts),
        "type": param.type.name,
    }


def test_options_mirror_the_command_signature() -> None:
    signature = inspect.signature(flywheel).parameters
    expected = [(name, p.annotation) for name, p in signature.items() if name != "ctx"]
    assert [(f.name, f.type) for f in fields(FlywheelOptions)] == expected


def test_command_parameters_match_the_recorded_surface() -> None:
    golden = json.loads(GOLDEN.read_text())
    providers = ", ".join(available_providers())
    for item in golden["params"]:
        item["help"] = item["help"].replace("{providers}", providers)
    command = _command()
    assert command.help == golden["help"]
    assert [_describe(param) for param in command.params] == golden["params"]


def test_help_lists_every_option() -> None:
    result = CliRunner().invoke(app, ["flywheel", "--help"], env=_PLAIN_TERMINAL)
    assert result.exit_code == 0, result.output
    for param in _command().params:
        assert param.opts[0] in result.output
