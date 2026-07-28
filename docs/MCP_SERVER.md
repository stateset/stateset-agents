# StateSet Agents MCP server

`stateset-agents mcp` exposes the framework's grade → curate → retrain
"improve" loop as [MCP](https://modelcontextprotocol.io) tools, so any MCP
client — Claude Code, Claude Desktop, or another agent — can drive it
directly instead of shelling out to the CLI.

Every tool is a thin wrapper over the same, already-tested module functions
the CLI uses (`stateset_agents.cli_improve`, `stateset_agents.data.
trajectory_ingest`, `scripts/grade_transcript.py`, `examples/
model_presets.py`, `examples/finetune_gspo.py`). No grading, curation, or
training logic is reimplemented in the server.

**v1 scope: no tool starts real GPU training.** `dry_run_finetune` only ever
runs `examples/finetune_gspo.py --dry-run` (stub backend, no model download,
no training). Use the CLI (`stateset-agents fine-tune`, `examples/
finetune_gspo.py --no-dry-run`, or the packaged starters) for real training
runs.

## Install

The MCP server is an optional extra — it is not a core dependency:

```bash
pip install 'stateset-agents[mcp]'
```

## Run it

```bash
stateset-agents mcp --transport stdio   # stdio is the only supported transport in v1
```

Without the `mcp` extra installed, the command exits with a clear install
hint instead of a raw traceback:

```
$ stateset-agents mcp
The 'mcp' package is required for the StateSet Agents MCP server. Install it with: pip install stateset-agents[mcp]
```

## Register with Claude Code

```bash
claude mcp add stateset-agents -- stateset-agents mcp
```

Claude Code will then have access to the tools below in any session.

## Register with Claude Desktop

Add to your `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "stateset-agents": {
      "command": "stateset-agents",
      "args": ["mcp"]
    }
  }
}
```

## Tools

| Tool | Wraps | Purpose |
|---|---|---|
| `list_rewards()` | `cli_improve.KNOWN_REWARDS` | List the rule-based reward names (`gsm8k`, `customer_support`, `tool_calling`). No LLM-judge rewards — those need an API key and are out of scope for the offline improve loop. |
| `ingest_transcripts(input_path, format, output_dir)` | `stateset_agents.data.trajectory_ingest` | Convert an OpenAI chat-completions JSONL or LangChain/LangGraph message dump into per-conversation transcript JSONLs. |
| `grade_transcript(history_path, reward)` | `scripts/grade_transcript.py` | Score every assistant turn in a single transcript file; returns mean score + per-reward-component breakdown. |
| `improve_run(transcripts_dir, reward, output_dir, threshold=0.7, format="transcripts")` | `cli_improve.run_improve` | Grade every transcript in a directory, curate turns scoring `>= threshold` into `<output_dir>/curated.jsonl`, and write `improve_summary.json` + `next_steps.md`. Same orchestration function the CLI's `improve run` command calls — output is byte-for-byte comparable. |
| `improve_status(output_dir)` | `cli_improve.get_improve_status` | Return the summary JSON from a previous `improve_run`/`improve run`. |
| `list_model_presets()` | `examples/model_presets.py` | List model preset names and key hyperparameter fields (model id, learning rate, prompt/completion lengths, LoRA/quantization flags). |
| `dry_run_finetune(model_preset)` | `examples/finetune_gspo.py --dry-run` | Preview the resolved agent/reward/trainer config for a preset without downloading weights or training. |

Every tool validates its inputs and returns a structured dict — on failure
that's `{"error": "..."}` rather than a raised traceback, so MCP clients get
an actionable message instead of a stack trace.

## Resources

v1 ships tools only, not MCP resources. The tool surface (in particular
`improve_status`) already covers reading back a previous run's summary, and
the SDK's resource templating adds indirection this version doesn't need
yet — revisit if a client workflow specifically wants
`improve://summary/{output_dir}`-style addressing.

## A typical session

```
list_rewards()
ingest_transcripts("logs.jsonl", "openai", "transcripts/")
improve_run("transcripts/", "customer_support", "improved/", threshold=0.7)
improve_status("improved/")
list_model_presets()
dry_run_finetune("qwen3.5-0.8b")
```

Then hand off to a real training run outside the MCP server — e.g.
`stateset-agents fine-tune --input improved/curated.jsonl` or
`python examples/finetune_gspo.py --model qwen3.5-0.8b --no-dry-run`.

See also: [`docs/COOKBOOK.md`](COOKBOOK.md) — "The improvement loop in one
command" walks through the same flow via the CLI directly.
