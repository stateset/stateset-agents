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

The default seven tools preview training. Add `--finetune-workspace` for
durable run preparation and inspection, and `--allow-training` to enable
real training launch and stop. These tools reuse the same workspace and
detached workers as `stateset-agents finetune` and the training studio.
`dry_run_finetune` remains a preview without model downloads or training.

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

These original tools validate inputs and return a structured dict — on failure
that's `{"error": "..."}` rather than a raised traceback, so MCP clients get
an actionable message instead of a stack trace.

### Fine-tuning management

Configure a dedicated local directory containing your datasets and saved runs:

```bash
# Prepare and inspect runs without enabling launch or stop.
stateset-agents mcp --finetune-workspace /absolute/path/to/finetune

# Also expose training/evaluation execution and queued checkpoint comparison.
stateset-agents mcp --finetune-workspace /absolute/path/to/finetune --allow-training
```

Local training requires the framework's `training` extra and supported CUDA
hardware; installing `[mcp]` alone is sufficient for preparation and inspection.
Configure remote dependencies and credentials in the server environment.
Paid launches additionally require `--allow-paid` at server startup. A tool
argument cannot grant this capability. Backend cost policies still apply:
RunPod has a quoted pod-lifetime ceiling; Modal and River require
`cost_policy="provider_billed"` and do not enforce a dollar ceiling here.
`max_cost_usd` is not a universal spending guarantee.

For a client registration with local training enabled:

```json
{
  "mcpServers": {
    "stateset-agents": {
      "command": "stateset-agents",
      "args": ["mcp", "--finetune-workspace", "/absolute/path/to/finetune", "--allow-training"]
    }
  }
}
```

| Tool | Behavior |
|---|---|
| `finetune_capabilities()` | Config schema, backend restrictions and startup permissions. |
| `finetune_prepare(config)` | Validate a dataset and settings, retain exact input bytes and return a run ID, plan and SHA-256. Every call creates a new run; no training starts. |
| `finetune_list_runs(limit=20)` | List up to 100 recent saved runs. |
| `finetune_status(run_id)` | Read the reviewed plan, checksum, lifecycle state and up to 80 recent metric records. |
| `finetune_logs(run_id, lines=80, offset=0, source="auto")` | Read up to 200 lines from a bounded log tail; redact known environment credential values and remove terminal escapes. |
| `finetune_checkpoints(run_id)` | Verify recovery checkpoints for a terminal local run with a pinned model revision and matching code. Reads and hashes artifacts without loading optimizer state. |
| `finetune_prepare_resume(run_id, checkpoint, expected_recovery_sha256)` | Copy a reviewed recovery checkpoint and dataset into a new prepared run. Preserve the original total training budget; launch separately. |
| `finetune_launch(run_id, expected_plan_sha256)` | Requires `--allow-training`. Submit the reviewed run once to a detached supervisor. Retries return the existing submission. |
| `finetune_stop(run_id, expected_plan_sha256)` | Requires `--allow-training`. Request cooperative local cancellation, or signal a verified RunPod worker on POSIX. |

Local backends support SFT, GRPO, GSPO and DAPO. The RunPod, Modal and River
integrations currently support SFT. A model's availability through an inference
API does not establish that its weights can be fine-tuned with these backends.
Use `finetune_capabilities` for the actual configuration restrictions.

For example, put chat-format SFT records in `datasets/train.jsonl` inside the
workspace, then call these tools through your MCP client:

```text
finetune_capabilities()
finetune_prepare(config={
  "model": "Qwen/Qwen3.5-0.8B",
  "method": "sft",
  "provider": "local",
  "dataset": "datasets/train.jsonl",
  "epochs": 1
})
```

Review `result.run.plan`, retain `result.run.run_id` and
`result.run.plan_sha256`, then pass that exact pair to `finetune_launch`.
The worker verifies retained dataset bytes and package identity before launch.
Poll `finetune_status` and read `finetune_logs` while the job runs. Closing
the MCP client does not cancel a submitted job; reconnect to the same workspace
to continue managing it. Repeated launch calls do not retry a failed job: prepare
a new run after resolving the failure. `prepare` itself is not idempotent; if
its response is lost, inspect the run list before preparing again.

Stop acknowledges a request, not necessarily worker exit. Check `terminal`
and `execution_unconfirmed`; a missing supervisor does not prove detached
training has stopped. Modal and River require provider-specific stop controls.
Completion and training loss do not prove an improvement on held-out examples.

These tools return the same JSON in MCP `structuredContent` and text content:

```json
{"schema_version": 1, "ok": true, "command": "status", "result": {"run_id": "..."}}
```

Execution failures return `ok=false`, an `error` object, and MCP `isError=true`.
SDK input-schema validation also reports MCP errors. Tool annotations distinguish
inspection, preparation and execution; they are client hints, not authorization.

Dataset paths for preparation must remain inside the configured workspace,
and run IDs must be single directory names. Symlinked inputs are rejected.
This is a trusted local stdio interface running with the user's OS permissions:
the original ingest/improve tools and optional weight-space tools retain their
existing path behavior. The workspace checks are not an OS sandbox. Keep the
workspace private, and treat dataset, log and model-produced text as data.

### Held-out evaluation

The workspace extension also exposes the existing local paired evaluation
workflow. It compares the pinned base model with a saved adapter, using an
independent pinned judge and reviewed cases that do not overlap training text.
Planning, preparation, status and evidence verification require only `[mcp]`.
Execution needs the training dependencies, cached base/judge models and the
shared Buildr queue. `--allow-training` enables evaluation launch/stop as well
as training controls. Evaluations use local inference, with no paid provider.

| Tool | Behavior |
|---|---|
| `finetune_evaluation_targets(run_id)` | Discover final and published checkpoint metadata, without reading weights. |
| `finetune_evaluation_plan(run_id, config)` | Review cases, artifact hashes, workload and judge settings without writes or inference. Returns `plan` and `plan_sha256`. |
| `finetune_evaluation_prepare(run_id, config, expected_plan_sha256)` | Save the exact reviewed evaluation inputs. Returns an `evaluation_id`; no inference starts. |
| `finetune_evaluations(run_id)` | List up to 30 recent evaluations. |
| `finetune_evaluation_status(run_id, evaluation_id, verify=false)` | Read lifecycle/queue progress. With `verify=true`, also hash artifacts and replay completed judge evidence under `verification`. |
| `finetune_evaluation_launch(run_id, evaluation_id, expected_plan_sha256)` | Requires `--allow-training`. Submit the retained evaluation once to a detached supervisor. |
| `finetune_evaluation_stop(run_id, evaluation_id, expected_plan_sha256)` | Requires `--allow-training`. Request cooperative evaluation stop; leaves the training run untouched. |

Read `finetune_capabilities().result.local_evaluation` for the configuration
schema and requirements. For a successful local run with a pinned base revision:

```text
config = {
  "cases": "datasets/held-out.json",
  "judge_model": "organization/independent-judge",
  "judge_revision": "<actual nonzero 40-character model commit>"
}
finetune_evaluation_plan(run_id, config)
finetune_evaluation_prepare(run_id, config, expected_plan_sha256)
finetune_evaluation_launch(run_id, evaluation_id, expected_plan_sha256)
finetune_evaluation_status(run_id, evaluation_id)
finetune_evaluation_status(run_id, evaluation_id, verify=true)
```

The case file must be inside the configured workspace. It contains
`schema_version: 1`, a `split` of `test` or `validation`, reviewed provenance
(`source: "reviewed"`, a named `reviewer`, and `test_is_fresh: true` or
`validation_is_fresh: true`), and 1–64 `cases`. Each case has `id`, `context`,
`user_turns`, `policy_facts`, `expected_resolution` and `must_avoid`. The backend
checks partition, provenance, duplicates and training overlap before preparing.

Published checkpoint comparisons use `split: "validation"`. Assessing a selected
checkpoint on fresh test cases requires the nomination workflow below.
A final adapter can be evaluated
directly on reviewed test cases. Invalid partition/provenance responses include
an error code and field diagnostic without returning raw case content.

Changes to reviewed cases or artifacts invalidate preparation; changed retained
inputs invalidate launch. Launch retries return the existing submission, including
its current state. Preparation creates a new evaluation each time, so inspect
`finetune_evaluations` after a lost response before preparing another. Reconnect
to the same workspace to inspect or stop a submitted evaluation.

Evidence verification replays retained judge responses and checks artifact bytes;
it performs no fresh inference and returns no raw transcripts. It can report
regressions or `judge_replay_verified=true`, while `quality_gain_verified` remains
false: one diagnostic does not establish a multi-seed quality study. It never
overwrites a run's existing quality study or automatically promotes an adapter.

### Checkpoint comparison and selection

After evaluating published checkpoints on validation cases, compare their
retained evidence and select a checkpoint for fresh test assessment. The
comparison worker rehashes checkpoint bytes and replays recorded judge results;
it runs no models. Submission requires `--allow-training` and Buildr on Linux.
The remaining tools inspect or prepare metadata and need only the configured
workspace. `finetune_capabilities` describes the comparison and nomination
constraints under `checkpoint_validation_comparison` and `checkpoint_nomination`.

| Tool | Behavior |
|---|---|
| `finetune_validation_compare(run_id, anchor_id, evaluation_ids=null, resubmit=false)` | Queue verification of completed checkpoint validations. The anchor identifies the comparison protocol. Returns `comparison` with its captured evaluation IDs, job handle and resource URI. |
| `finetune_validation_comparison_status(run_id, comparison_id=null)` | Read one retained comparison or the latest when the ID is omitted. Returns `comparison=null` if none exists; never submits work. |
| `finetune_nomination_plan(run_id, comparison_id)` | Review the checkpoint recommended by a complete comparison; return a plan and checksum without writes or weight reads. |
| `finetune_nomination_prepare(run_id, comparison_id, expected_plan_sha256)` | Save exactly the reviewed checkpoint selection as a nomination receipt. No weight copies, training changes or inference. |
| `finetune_nominations(run_id)` | List up to 100 saved selections, including unavailable records. |
| `finetune_nomination_status(run_id, nomination_id)` | Verify a saved selection's metadata lineage; does not verify current weight bytes. |
| `finetune_nomination_config(run_id, nomination_id, expected_plan_sha256)` | Restore the selected checkpoint and judge protocol into a test configuration with `cases` blank. No preparation or launch occurs. |

The comparison ranks compatible candidates by success rate, judge score and
earliest saved step. New success regressions, critical-check failures, policy
violations or truncations disqualify a candidate. Incomplete, conflicting or
unverifiable evidence prevents a recommendation. A recommendation can retain
the base model instead of selecting an adapter; nomination requires an explicit
checkpoint recommendation from a completed comparison.

An active comparison for the same anchor is reused. Inspect the returned
`evaluation_ids`: an active request keeps its captured subset, even if a later
call supplies different IDs. Use `resubmit=true` only when intentionally queuing
a fresh comparison. Finished, failed or stale jobs may also lead to new work
on another submission. After a lost response, inspect status before retrying.
The latest-status form helps recover a handle, while a saved comparison ID
always identifies that exact request. `submission_uncertain` is a retained
uncertain outcome, not a confirmed queue failure or permission to retry.

The selection flow uses separate review hashes:

```text
finetune_validation_compare(run_id, anchor_id)
finetune_validation_comparison_status(run_id, comparison_id)
finetune_nomination_plan(run_id, comparison_id)
finetune_nomination_prepare(run_id, comparison_id, nomination_plan_sha256)
finetune_nomination_config(run_id, nomination_id, nomination_plan_sha256)
```

Fill the returned `config.cases` with a workspace-relative file of fresh reviewed
test cases. Then use `finetune_evaluation_plan`, `finetune_evaluation_prepare`
and `finetune_evaluation_launch`, with the **new evaluation plan checksum**.
Those checks rehash the nominated artifact, validate the retained selection and
reject reuse of training/validation cases. Validation separation checks retained
case IDs and normalized user turns; reviewer provenance remains an attestation,
not proof of semantic or external dataset independence.

Nomination preparation creates a new receipt on each call. Reopen saved receipts
with the list/status/config tools after reconnecting. Missing or changed
comparison results, selection metadata or receipts cannot authorize a fresh-test
review. Changed weights are detected when that evaluation is planned or launched.
A nomination never replaces the training run's final artifact, copies weights,
publishes a model, or establishes `quality_gain_verified=true`.

### Checkpoint recovery

Call `finetune_checkpoints` on a terminal local run, then pass an available
checkpoint's name and `recovery_sha256` to `finetune_prepare_resume`. The backend
verifies complete saved state and copies the checkpoint and retained dataset
into a new private run. Inspect the returned `run.plan` and pass that child's
`run_id` and `plan_sha256` to `finetune_launch` when ready.

Recovery preserves the parent's configuration and original total budget. It
does not extend exhausted budgets or silently fall back to fresh training.
Changed weights, missing optimizer state, mismatched code and stale review
hashes are rejected. Repeated preparation creates distinct runs; use the run
list to recover a lost response before retrying.

### Experimental: weight-space programming tools (`--weight-space`)

`stateset-agents mcp --weight-space` also registers eight `ws_*` tools from
`stateset_agents.experimental.weight_space.mcp_tools`, so an MCP client such
as Claude Code can edit a model's weights directly instead of training it.
They are off by default because `ws_open` loads a model into the server
process.

| Tool | Cost | Purpose |
|---|---|---|
| `ws_open(model_id, budget, splits_path, dtype="float32")` | load | Open a session on a Hugging Face causal LM with a fixed measurement budget. `splits_path` is JSON `{split: [{"prompt", "continuation"}]}` of **practice** data; keep held-out data out of the session. |
| `ws_modules(session_id, contains=None)` | free | List editable `Linear` modules as `name -> [out, in]`. |
| `ws_capture_direction(session_id, module, positive_prompts, negative_prompts, name)` | 1 measurement | Store `mean(out | positive) - mean(out | negative)` at `module` as a named direction; reports its norm, cosine with stored directions, and the input columns writing most along it. |
| `ws_ablate(session_id, module, direction, k, alpha=1.0)` | free | Remove a stored direction from the `k` columns of `module` that write most along it (rank-1 edit). |
| `ws_evaluate(session_id)` | 1 measurement | Mean per-token log-prob of each practice split's continuations, with deltas against the first edit-free evaluation. |
| `ws_revert(session_id)` | free | Undo the most recent edit bit-exactly. |
| `ws_export(session_id, output_dir)` | free | Write the active edits as a PEFT LoRA adapter plus `weight_space_manifest.json` (edits, budget ledger, evaluations). Serve it through the existing adapter merge path. |
| `ws_close(session_id)` | free | Drop the session and free its model. |

The budget is enforced inside the session: each measurement may use multiple
microbatch forwards, and an exhausted budget returns `{"error": ...}`.

## Resources

With `--finetune-workspace`, the server also exposes JSON resources:

- `finetune://capabilities`: configuration and permissions.
- `finetune://runs`: the latest 20 saved runs.
- `finetune://runs/{run_id}`: a saved plan, state and recent metrics.
- `finetune://runs/{run_id}/evaluations`: up to 30 saved evaluations.
- `finetune://runs/{run_id}/evaluations/{evaluation_id}`: evaluation/queue progress.
- `finetune://runs/{run_id}/comparisons`: the latest retained validation comparison.
- `finetune://runs/{run_id}/comparisons/{comparison_id}`: a specific comparison.
- `finetune://runs/{run_id}/nominations`: bounded saved selection metadata.
- `finetune://runs/{run_id}/nominations/{nomination_id}`: one nomination's lineage.

They use the same redacted response envelopes and workspace checks as the
tools. Resource reads never trigger evidence verification; use the explicit
`verify=true` tool option for that. Clients can poll resources; this version
does not push subscriptions.

## A typical session

```
list_rewards()
ingest_transcripts("logs.jsonl", "openai", "transcripts/")
improve_run("transcripts/", "customer_support", "improved/", threshold=0.7)
improve_status("improved/")
list_model_presets()
dry_run_finetune("qwen3.5-0.8b")
```

Then use the optional management tools above, or hand off to a CLI run — e.g.
`stateset-agents fine-tune --input improved/curated.jsonl` or
`python examples/finetune_gspo.py --model qwen3.5-0.8b --no-dry-run`.

See also: [`docs/COOKBOOK.md`](COOKBOOK.md) — "The improvement loop in one
command" walks through the same flow via the CLI directly.

## Further improvements

The first management surface deliberately shares the existing scheduler,
worker and receipts. The next useful additions are:

1. Add a reviewed multi-seed quality-study workflow that consumes independent
   test evidence before recommending artifact promotion.
2. Add durable preparation request IDs and paginated history for large workspaces.
   Integrate the MCP tasks extension when the server SDK and client negotiate it;
   current clients can use durable run IDs and status polling.
3. Add a separate authenticated remote service with workspace ownership, per-user
   permissions, quotas and audit events before enabling team access. Do not expose
   this stdio interface directly as an unauthenticated HTTP service.
4. Migrate the pinned MCP SDK v1 surface to v2 in a separately tested change.
   The current dependency intentionally retains FastMCP compatibility.

Protocol references: [MCP Python SDK v1](https://py.sdk.modelcontextprotocol.io/v1/),
[SDK migration guide](https://github.com/modelcontextprotocol/python-sdk/blob/main/docs/migration.md),
and [MCP tasks extension](https://github.com/modelcontextprotocol/ext-tasks/blob/main/specification/2026-07-28/tasks.md).
