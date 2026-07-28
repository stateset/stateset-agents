"""``stateset-agents ingest`` — bring logs from agents built elsewhere.

Thin CLI wrapper over ``stateset_agents.data.trajectory_ingest``. Converts
OpenAI chat-completions or LangChain/LangGraph message logs into
graded-history JSONL files compatible with ``scripts/grade_transcript.py``
(one ``{"role", "content"}`` object per line), so logs captured by an agent
built with any framework can be graded and folded back into training.

Split out of stateset_agents/cli.py following the pattern in cli_chat.py:
this module imports the shared ``app`` and attaches its command via
``@app.command()`` at import time.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer

from stateset_agents import cli as _cli
from stateset_agents.cli import app
from stateset_agents.data.trajectory_ingest import (
    from_langchain_json,
    from_openai_jsonl,
    to_grading_history,
)

_echo = _cli._echo


@app.command("ingest")
def ingest(
    format: str = typer.Option(
        ...,
        "--format",
        "-f",
        help="Source log format: 'openai' (chat-completions messages JSONL) "
        "or 'langchain' (LangChain/LangGraph message-dump JSON).",
    ),
    input: str = typer.Option(
        ...,
        "--input",
        "-i",
        help="Path to the source log file. For --format openai: JSONL, one "
        "conversation per line ({'messages': [...]} or a bare message list). "
        "For --format langchain: a single JSON file (see "
        "stateset_agents.data.trajectory_ingest docstring for supported shapes).",
    ),
    output: str = typer.Option(
        ...,
        "--output",
        "-o",
        help="Output path. If it ends in .jsonl, all conversations are "
        "concatenated into one graded-history JSONL file (turns from "
        "different conversations are separated by a blank line). Otherwise "
        "it is treated as a directory and one <output>/conversation_<N>.jsonl "
        "file is written per conversation — feed any of them to "
        "`python scripts/grade_transcript.py --history <file>`.",
    ),
) -> None:
    """Convert third-party conversation logs into graded-history JSONL.

    Examples::

        stateset-agents ingest --format openai --input logs.jsonl --output graded.jsonl
        stateset-agents ingest --format langchain --input lc_run.json --output out_dir/
    """
    fmt = format.strip().lower()
    if fmt not in ("openai", "langchain"):
        _echo(
            f"Unsupported --format '{format}'. Choose 'openai' or 'langchain'.",
            err=True,
        )
        raise typer.Exit(code=2)

    input_path = Path(input)
    if not input_path.exists():
        _echo(f"Input file not found: {input_path}", err=True)
        raise typer.Exit(code=2)

    try:
        if fmt == "openai":
            trajectories = from_openai_jsonl(input_path)
        else:
            trajectories = from_langchain_json(input_path)
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        _echo(f"Failed to ingest {input_path}: {exc}", err=True)
        raise typer.Exit(code=1) from exc

    if not trajectories:
        _echo(f"No conversations found in {input_path}", err=True)
        raise typer.Exit(code=1)

    output_path = Path(output)
    if output_path.suffix == ".jsonl":
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            for i, traj in enumerate(trajectories):
                if i > 0:
                    f.write("\n")
                for turn in to_grading_history(traj):
                    f.write(json.dumps(turn) + "\n")
        _echo(
            f"Wrote {len(trajectories)} conversation(s) "
            f"({sum(len(t.turns) for t in trajectories)} turns) to {output_path}"
        )
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        for i, traj in enumerate(trajectories):
            conv_path = output_path / f"conversation_{i}.jsonl"
            with open(conv_path, "w", encoding="utf-8") as f:
                for turn in to_grading_history(traj):
                    f.write(json.dumps(turn) + "\n")
        _echo(f"Wrote {len(trajectories)} conversation(s) to {output_path}/")
