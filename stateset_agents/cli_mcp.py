"""``stateset-agents mcp`` — run the StateSet Agents MCP server.

Thin CLI wrapper over ``stateset_agents.mcp_server``. Requires the optional
``mcp`` extra (``pip install stateset-agents[mcp]``); without it, this
command prints an install hint and exits with a non-zero code instead of
raising a raw traceback.

Split out of stateset_agents/cli.py following the pattern in cli_ingest.py:
this module imports the shared ``app`` and attaches its command via
``@app.command()`` at import time.
"""

from __future__ import annotations

from pathlib import Path

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app

_echo = _cli._echo


@app.command("mcp")
def mcp(
    transport: str = typer.Option(
        "stdio",
        "--transport",
        help="MCP transport to serve over (default: stdio).",
    ),
    weight_space: bool = typer.Option(
        False,
        "--weight-space",
        help=(
            "Also expose the experimental ws_* weight-space programming tools "
            "(loads models into the server process)."
        ),
    ),
    finetune_workspace: Path | None = typer.Option(
        None,
        "--finetune-workspace",
        help="Enable preparation and inspection of fine-tuning runs in this directory.",
    ),
    allow_training: bool = typer.Option(
        False,
        "--allow-training",
        help="Enable training, local evaluation and queued checkpoint comparison.",
    ),
    allow_paid: bool = typer.Option(
        False, "--allow-paid", help="Also permit launches on paid training providers."
    ),
) -> None:
    """Run the StateSet Agents MCP server (stdio transport by default).

    Exposes the grade -> curate -> retrain "improve" loop as MCP tools so
    any MCP client (Claude Code/Desktop, other agents) can drive it. See
    docs/MCP_SERVER.md for registration instructions, e.g.::

        claude mcp add stateset-agents -- stateset-agents mcp

    Real training requires --finetune-workspace and --allow-training.
    """
    if transport != "stdio":
        _echo(
            f"Unsupported --transport '{transport}'. Only 'stdio' is supported in v1.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        from stateset_agents.mcp_server import create_server

        server = create_server(
            weight_space=weight_space,
            finetune_workspace=finetune_workspace,
            allow_training=allow_training,
            allow_paid=allow_paid,
        )
    except ImportError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc
    except ValueError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc

    server.run(transport="stdio")
