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
| `grade_transcript(history_path, reward, context_file=None)` | `stateset_agents.data.grade_transcript` | Score the assistant turns using one optional reference object per turn. Returns per-turn grades, reasons for ungradable turns, and a mean over graded turns. |
| `improve_run(transcripts_dir, reward, output_dir, threshold=0.7, format="transcripts", context_file=None)` | `cli_improve.run_improve` | Curate passing assistant actions with their conversation histories. Pass one reference file for a single conversation or a directory of matching filenames for a batch. Writes `curated.jsonl`, `improve_summary.json` and `next_steps.md`. |
| `improve_status(output_dir)` | `cli_improve.get_improve_status` | Return the summary JSON from a previous `improve_run`/`improve run`. |
| `list_model_presets()` | `examples/model_presets.py` | List model preset names and key hyperparameter fields (model id, learning rate, prompt/completion lengths, LoRA/quantization flags). |
| `dry_run_finetune(model_preset)` | `examples/finetune_gspo.py --dry-run` | Preview the resolved agent/reward/trainer config for a preset without downloading weights or training. |
| `studio_list_models(workspace=".", refresh=False)` | `stateset_agents.weight_studio` | Discover local `.safetensors` checkpoints, sharded models, and StateSet architecture twins in the workspace. |
| `studio_inspect_model(model_id, workspace=".", checkpoint_id=None)` | `stateset_agents.weight_studio` | Inspect layer architecture, tensor counts, Frobenius norms, and active non-destructive edit history. |
| `studio_apply_edit(model_id, op, target, params=None, label="", workspace=".", checkpoint_id=None)` | `stateset_agents.weight_studio` | Apply a non-destructive surgical weight edit (`scale`, `prune`, `clamp`, `svd_truncate`, `head_edit`, `rank1_edit`, `steer_bake`, `region_edit`, `blend`, `noise`). |
| `studio_extract_steering(model_id, positive_prompt, negative_prompt, name="Steering Vector", layer_index=None, workspace=".", checkpoint_id=None)` | `stateset_agents.weight_studio` | Extract a Contrastive Activation Steering (CAA/RepE) vector and identify the optimal injection layer. |
| `studio_discover_circuit(model_id, prompt, top_k=10, system_prompt="", workspace=".", checkpoint_id=None)` | `stateset_agents.weight_studio` | Discover the Top-K causal attention heads & MLP cores and attention-flow arc count for a prompt. |
| `studio_eval_and_export(model_id, export_name="", run_merge_probe=True, require_go=True, max_cases=6, workspace=".", checkpoint_id=None)` | `stateset_agents.weight_studio` | Run the Pre-Export Batch Eval Gate (`GO`/`WARN`/`BLOCK`) and optionally write a SHA-256 reload-verified `.safetensors` checkpoint + portable recipe. |

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
