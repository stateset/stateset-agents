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
) -> None:
    """Run the StateSet Agents MCP server (stdio transport by default).

    Exposes the grade -> curate -> retrain "improve" loop as MCP tools so
    any MCP client (Claude Code/Desktop, other agents) can drive it. See
    docs/MCP_SERVER.md for registration instructions, e.g.::

        claude mcp add stateset-agents -- stateset-agents mcp

    v1 scope: no tool starts real GPU training (dry-run only).
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

    server = create_server()
    server.run(transport="stdio")
