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
    workspace: Path | None = typer.Option(
        None,
        "--workspace",
        "-w",
        help="Default workspace directory for 3D Weight Studio MCP tools and resources.",
    ),
) -> None:
    """Run the StateSet Agents MCP server (stdio transport by default).

    Exposes the grade -> curate -> retrain "improve" loop and the full
    3D Weight Studio (inspection, surgical editing, CAA steering, circuits,
    residual stream, SAE features, A/B Arena, Auto-Repair, 95% CI Eval Gate,
    LoRA/GGUF/HF Card/HTML PR report export, and Autonomous Self-Heal) as
    annotated MCP tools, live ``stateset://studio/*`` resources, and guided
    workflow prompts. See docs/MCP_SERVER.md for registration instructions, e.g.::

        claude mcp add stateset-agents -- stateset-agents mcp --workspace .

    No tool starts real remote GPU training (dry-run only).
    """
    if transport != "stdio":
        _echo(
            f"Unsupported --transport '{transport}'. Only 'stdio' is supported in v1.",
            err=True,
        )
        raise typer.Exit(code=2)

    try:
        from stateset_agents.mcp_server import create_server
    except ImportError as exc:
        _echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc

    server = create_server(workspace=workspace)
    server.run(transport="stdio")
