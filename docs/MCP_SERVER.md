# StateSet Agents & 3D Weight Studio MCP Server

`stateset-agents mcp` exposes both StateSet Agents' grade → curate → retrain
"improve" loop **and** the complete **3D Weight Studio** (weight inspection,
Wanda/SVD sensitivity profiling, Base-vs-Edited inference & Logit Lens, causal
circuits, residual-stream tracing, SAE sparse features, CAA behavioral
steering, non-destructive surgical editing with undo/redo/recipes, Multi-Edit
A/B Arena, Surgical Auto-Repair, 95% CI Batch Eval Gate, PEFT LoRA /
`.safetensors` / GGUF v3 + Ollama Modelfile / HF Model Card / interactive 3D
HTML PR report export, workbench sessions, and the Autonomous
Transcript-to-Surgery Self-Healing Loop) over the
[Model Context Protocol (MCP)](https://modelcontextprotocol.io).

Any MCP-capable client — **Claude Code**, **Claude Desktop**, **Cursor**,
**Antigravity**, **LangGraph**, or custom AI agents — can inspect, steer,
surgically repair, evaluate, and package models without shelling out to the CLI.

- **25 annotated tools** with MCP `ToolAnnotations` (`readOnlyHint`,
  `destructiveHint=False`, `idempotentHint`) so MCP clients can safely
  auto-approve read-only inspection and interpretability calls.
- **5 live MCP resources** (`stateset://studio/*`) giving agents instant
  context on workspace models, capabilities, steering presets, scenarios, and
  saved sessions.
- **4 guided MCP workflow prompts** so engineers and agents can launch
  end-to-end playbooks in one command.
- **Self-correcting error responses**: unknown `model_id` calls automatically
  return `available_models` and a recovery `hint` so agents self-heal in one
  turn.
- **Non-destructive by design**: original checkpoints are never overwritten,
  and no MCP tool launches real remote GPU training (`dry_run_finetune` only
  resolves preset configs).

## Install

```bash
pip install 'stateset-agents[mcp]'
```

## Run It

```bash
stateset-agents mcp --transport stdio --workspace .
```

You can also pin the default workspace via the `STATESET_STUDIO_WORKSPACE`
environment variable so agents don't need to pass `workspace` on every call.

## Register With AI Clients

### Live Streamable HTTP MCP (Real-Time 3D Studio Browser Sync!)

Launch 3D Weight Studio (`stateset-agents studio --port 8765`) and point any HTTP
MCP client directly at `http://127.0.0.1:8765/api/weight-studio/mcp`. Every
surgical edit, CAA steering bake, Auto-Heal, or Red-Team immunization executed
by an AI agent streams over `GET /api/weight-studio/events` (SSE) and animates
connected 3D WebGL2 browser viewports live!

### Claude Code

```bash
claude mcp add stateset-agents -- stateset-agents mcp --workspace /path/to/your/workspace
```

### Claude Desktop / Cursor / Antigravity (`mcpServers`)

```json
{
  "mcpServers": {
    "stateset-weight-studio": {
      "url": "http://127.0.0.1:8765/api/weight-studio/mcp",
      "transport": "streamable-http"
    },
    "stateset-agents": {
      "command": "stateset-agents",
      "args": ["mcp", "--workspace", "/path/to/your/workspace"]
    }
  }
}
```

## Tools (27 Total)

### Improve & Finetune Preview (7 Tools)

| Tool | Access | Purpose |
| :--- | :---: | :--- |
| `list_rewards()` | Read-only | List built-in offline reward names (`gsm8k`, `customer_support`, `tool_calling`, `nsr`). |
| `ingest_transcripts(input_path, format, output_dir)` | Write | Convert OpenAI JSONL or LangChain/LangGraph message dumps into per-conversation transcript JSONLs. |
| `grade_transcript(history_path, reward, context_file=None)` | Read-only | Score assistant turns with per-turn breakdown and mean score. |
| `improve_run(transcripts_dir, reward, output_dir, threshold=0.7, format="transcripts", context_file=None)` | Write | Curate passing assistant actions (`curated.jsonl`, `improve_summary.json`, `next_steps.md`). |
| `improve_status(output_dir)` | Read-only | Read the summary JSON written by a prior `improve_run`. |
| `list_model_presets()` | Read-only | List model preset names and key hyperparameter fields. |
| `dry_run_finetune(model_preset)` | Read-only | Preview the resolved GSPO/agent/reward config without downloading weights or training. |

### 3D Weight Studio & Autonomous Surgery (18 Tools)

| Tool | Access | Purpose |
| :--- | :---: | :--- |
| `studio_list_models(workspace="", refresh=False)` | Read-only | Discover local `.safetensors` checkpoints, sharded models, and StateSet architecture twins. |
| `studio_inspect_model(model_id, workspace="", checkpoint_id=None)` | Read-only | Inspect layer architecture, tensor counts, Frobenius norms, and active edit stack. |
| `studio_profile_sensitivity(model_id, target_compression=0.35, ...)` | Read-only | Profile per-layer Wanda/Fisher salience + effective SVD rank and synthesize an adaptive mixed-rank compression recipe. |
| `studio_run_inference(model_id, prompt, system_prompt="", compare_base=True, steering=None, causal_patches=None, ...)` | Read-only | Execute Base-vs-Edited forward pass with Logit Lens, activation norms, live CAA steering, and causal patching. |
| `studio_discover_circuit(model_id, prompt, top_k=10, ...)` | Read-only | Discover the Top-K causal attention heads & MLP cores driving a prompt (`Circuit 3D`). |
| `studio_residual_stream(model_id, prompt, ...)` | Read-only | Trace Base vs. Edited residual hidden-state drift, Attention/MLP contribution split, and Logit Lens top-token convergence per layer. |
| `studio_sae_analyze(model_id, prompt, layer_index=None, top_k=10, clamps=None, ...)` | Read-only | Decompose residual activations into sparse monosemantic SAE features (`L0`, `R²`) with optional feature clamping. |
| `studio_extract_steering(model_id, positive_prompt, negative_prompt, name="Steering Vector", ...)` | Read-only | Extract a Contrastive Activation Steering (CAA/RepE) vector and find the optimal injection layer. |
| `studio_apply_edit(model_id, op, target, params=None, label="", ...)` | Session | Apply a non-destructive surgical weight edit (`scale`, `prune`, `clamp`, `svd_truncate`, `head_edit`, `rank1_edit`, `steer_bake`, `region_edit`, `blend`, `noise`). |
| `studio_manage_edits(model_id, action="history", recipe=None, reset_first=True, ...)` | Session | Full edit lifecycle: `history`, `undo`, `redo`, `reset`, `export_recipe`, or `apply_recipe`. |
| `studio_arena_compare(model_id, prompt, candidates=None, reward_fn="composite", ...)` | Read-only | Non-destructively benchmark Base, Active Edited, and candidate surgical edits side-by-side on a prompt. |
| `studio_auto_repair(model_id, reward_fn="composite", max_steps=4, ...)` | Session | Automatically bisect & dial back offending surgical edits when the Batch Eval Gate returns `WARN`/`BLOCK` until `GO` is restored. |
| `studio_eval_and_export(model_id, export_name="", run_merge_probe=True, require_go=True, max_cases=6, reward_fn="composite", custom_jsonl="", ...)` | Write | Run the 95% CI Pre-Export Batch Eval Gate (`GO`/`WARN`/`BLOCK`) and optionally write a SHA-256 reload-verified `.safetensors` checkpoint + recipe. |
| `studio_export_lora(model_id, export_name="surgical-lora-v1", rank=16, alpha=32, ...)` | Write | Factorize active surgical weight diffs `ΔW = W_edited - W_base` via truncated SVD into a reload-verified PEFT LoRA adapter (`adapter_model.safetensors` + `adapter_config.json`). |
| `studio_export_artifact(model_id, format="gguf", export_name="surgical-export-v1", quant_type="Q8_0", repo_id="", include_eval=True, ...)` | Write | Export active surgical edits as `gguf` (GGUF v3 + Ollama `Modelfile`), `hf_card` (Hugging Face `README.md` Model Card), or `html_report` (offline standalone interactive 3D HTML PR report). |
| `studio_save_session(session_name, model_id, metadata=None, ...)` | Write | Persist a named 3D Weight Studio workbench session (edit recipe + steering/eval metadata) to disk. |
| `studio_sessions(action="list", session_name="", ...)` | Session | List saved workbench sessions (`action="list"`) or restore a saved session (`action="load"`). |
| `studio_auto_heal(model_id, transcripts_path="", transcripts_jsonl="", reward_fn="customer_support", export_format="lora", output_name="", lora_rank=16, lora_alpha=32, require_go=True, ...)` | Write | Run the end-to-end **Autonomous Transcript-to-Surgery Self-Healing Loop**: diagnose failing transcripts → mine contrastive pairs → CAA steering → Multi-Edit A/B Arena → bake → 95% CI Eval Gate + Auto-Repair → export verified artifact + HTML PR report. |
| `studio_redteam_audit(model_id, categories=None, probes_path="", probes_jsonl="", max_probes=10, immunize=False, export_format="none", output_name="", ...)` | Session / Write | Run the **Adversarial Red-Team Safety Audit** (prompt injection, PII exfiltration, unauthorized tool calls, policy bypass, hallucinated commitments) and optionally apply one-click **Surgical Immunization** (`immunize=True`) while verifying benign helpfulness. |
| `studio_model_tournament(model_ids=None, reward_fn="customer_support", max_cases=4, include_redteam=True, synthesize_merge=True, merge_methods=None, export_format="none", output_name="", ...)` | Session / Write | Benchmark 2+ models/checkpoints on StateSet rewards + Red-Team ASR, build a Pareto leaderboard, synthesize TIES/DARE/SLERP surgical merges of the Top-2, promote the winning champion, and optionally export a verified bundle. |

Every tool validates inputs and returns a structured JSON dict (`{"error": "...", "hint": "..."}` on failure rather than a raw traceback).

## Live MCP Resources (`stateset://studio/*`)

| Resource URI | Content |
| :--- | :--- |
| `stateset://studio/capabilities` | Supported edit ops, target selectors, reward scorers, export formats, and recommended agent playbooks. |
| `stateset://studio/models` | Live catalog of discovered `.safetensors` checkpoints, sharded models, and StateSet architecture twins. |
| `stateset://studio/steering-presets` | Built-in StateSet contrastive behavioral steering pairs ready for CAA extraction. |
| `stateset://studio/scenarios` | Built-in StateSet evaluation & interpretability test scenarios. |
| `stateset://studio/sessions` | Persisted 3D Weight Studio workbench sessions on disk. |
| `stateset://studio/redteam-probes` | Built-in 10-probe Adversarial Red-Team safety suite spanning 5 enterprise attack vectors. |

## Guided MCP Workflow Prompts

| Prompt | Arguments | Workflow |
| :--- | :--- | :--- |
| `auto_heal_model` | `model_id`, `transcripts_path=""`, `reward="customer_support"`, `export_format="lora"` | Diagnose failing production logs → Autonomous Self-Heal → inspect 95% CI Eval Gate & HTML PR report. |
| `redteam_and_immunize` | `model_id`, `export_format="lora"` | Run Adversarial Red-Team Safety Audit → surgically immunize vulnerable layers without hurting benign helpfulness → export verified bundle. |
| `model_tournament_and_merge` | `reward="customer_support"`, `export_format="lora"` | Benchmark multiple checkpoints on reward + safety → sweep TIES/DARE/SLERP surgical merges → promote & export Pareto champion. |
| `surgical_behavior_edit` | `model_id`, `behavior_goal`, `reward="composite"` | Extract CAA steering → Multi-Edit A/B Arena → bake → verify Residual Stream & Eval Gate → export PEFT LoRA + HTML report. |
| `interpretability_audit` | `model_id`, `prompt` | Full mechanistic audit across Circuit 3D, Residual Stream Logit Lens, SAE feature clamping, and Wanda/SVD sensitivity. |
| `compress_and_export_model` | `model_id`, `target_compression=0.35`, `export_format="gguf"` | Profile Wanda/SVD sensitivity → replay adaptive compression recipe → Eval Gate + Auto-Repair → export verified GGUF/LoRA. |

## Example Autonomous Agent Session

```text
# 1. Discover workspace checkpoints
studio_list_models()

# 2. Heal failing production logs into a verified PEFT LoRA + HTML PR report in one call
studio_auto_heal(
    model_id="local:outputs/support_v1",
    transcripts_path="logs/failed_tickets.jsonl",
    reward_fn="customer_support",
    export_format="lora",
    output_name="support-healed-v2"
)

# 3. Or run a custom surgical experiment with full undo & auto-repair safety
studio_profile_sensitivity(model_id="local:outputs/support_v1", target_compression=0.35)
studio_manage_edits(model_id="local:outputs/support_v1", action="apply_recipe", recipe=...)
studio_eval_and_export(model_id="local:outputs/support_v1", reward_fn="customer_support")
studio_auto_repair(model_id="local:outputs/support_v1", reward_fn="customer_support")
studio_export_artifact(model_id="local:outputs/support_v1", format="gguf", export_name="support-q8")
studio_export_artifact(model_id="local:outputs/support_v1", format="html_report", export_name="support-pr-report")
```

See also: [`docs/WEIGHT_STUDIO.md`](WEIGHT_STUDIO.md) and [`docs/COOKBOOK.md`](COOKBOOK.md).
