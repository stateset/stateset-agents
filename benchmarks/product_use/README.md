# StateSet product-use benchmark

This public starter set evaluates whether an agent can use the StateSet MCP
improvement loop. Version 0.3 executes submitted tool calls against the real
StateSet functions in a fresh temporary workspace, then checks returned data
and generated files. Ingestion must reproduce the source conversations, and
curated examples must match their source transcripts and saved summary. Seven
tasks cover ingestion, grading, curation, status, training preview, reward
discovery, and model preset discovery.

Run the version 0.3 demonstration to verify the harness:

```bash
python -m benchmarks.product_use.execution \
  --tasks benchmarks/product_use/tasks.v0.3.public.json \
  --submissions benchmarks/product_use/demonstrations.v0.3.json
```

A submission is a JSON array of `{ "task_id": ..., "calls": [...] }` rows.
Each call has a tool `name` and an `arguments` object. Participants receive the
task prompts and [`tools.v0.3.json`](tools.v0.3.json), then submit their chosen calls. The harness
executes only seven allowlisted MCP functions. Path arguments must be relative to
the task workspace; absolute paths and traversal are rejected. Missing tasks
score zero, and unknown task IDs are rejected. The demonstrations are public
training examples, so their scores are not evidence of model capability.

Export the demonstrations to a model-agnostic JSONL training file. The exporter
replays every trace and refuses to label a failing trace as verified:

```bash
python -m benchmarks.product_use.export_examples \
  --tasks benchmarks/product_use/tasks.v0.3.public.json \
  --demonstrations benchmarks/product_use/demonstrations.v0.3.json \
  --tools benchmarks/product_use/tools.v0.3.json \
  --output product-use-examples.jsonl
```

Each row contains the prompt, tool schemas, MCP interface, tool calls, and verified score.
The committed [`examples.v0.3.jsonl`](examples.v0.3.jsonl) contains the same
rows. Provider-specific training formats can be derived from them.
The dedicated Benchmark CI job replays both committed corpora against the real
MCP functions before running performance measurements.

The version 0.2 executable task set and version 0.1 plan scorer remain
available for reproducibility. Version 0.1 uses self-reported tool calls and
artifacts, so its score is diagnostic only.

## Generated transfer checks (version 0.4)

Generate synthetic tasks with fresh paths and a choice of two model presets.
The generator produces five task families per variant: ingestion, grading,
training preview, reward discovery before grading, and preset discovery before
preview. Store the seed outside the repository and keep it private for an
independent evaluation:

```bash
python -c 'import secrets; from pathlib import Path; Path("/tmp/product-use.seed").write_text(secrets.token_hex(32))'
python -m benchmarks.product_use.generate \
  --seed-file /tmp/product-use.seed --variants 3 \
  --tasks-output /tmp/product-use.tasks.json \
  --prompts-output /tmp/product-use.prompts.json
```

Give participants only the prompts file and
[`tools.v0.4.json`](tools.v0.4.json). Keep the full task file with the
evaluator. Participants submit the same `{ "task_id": ..., "calls": [...] }`
format used by version 0.3. Score their submission with:

```bash
python -m benchmarks.product_use.execution \
  --tasks /tmp/product-use.tasks.json --submissions participant.json
```

For a local harness check, add `--oracle-output /tmp/product-use.oracle.json`
to the generation command and replay that file as the submission. Do not give
the oracle or seed to participants in an independent evaluation. The generated
set varies task paths and preset names but reuses the same synthetic support
conversations and task families, so it is a transfer check rather than a broad
measure of product competence.

## Interactive tool sessions

An agent adapter can evaluate one task at a time over JSON lines. The runner
first writes a `task` event with the prompt and tool schemas. Send one tool-call
object per line; after each call, read the `tool_result` event before choosing
the next call. Send `{ "done": true }` to finish and read the final `report`:

```bash
python -m benchmarks.product_use.interactive \
  --tasks benchmarks/product_use/tasks.v0.3.public.json \
  --task-id discover-and-preview-qwen \
  --tools benchmarks/product_use/tools.v0.3.json
```

For example, an adapter can send `{"name":"list_model_presets","arguments":{}}`,
inspect the returned presets, then send a `dry_run_finetune` call. The same
protocol works with a private version 0.4 task file and its matching tool
catalog. Each task allows at most four calls. A rejected tool call counts
toward that limit; the agent may use the feedback to correct its next call.
The final score checks the actual tool results and workspace, not the agent's
claim of success. Interactive reports carry `evaluation_mode: interactive`;
report them separately from static trace-replay scores because interactive
agents can correct a rejected call within their budget.

This local runner is for trusted submissions. Temporary directories and path
checks confine the supported tool arguments, but they do not provide OS-level
isolation for hostile code. A hosted leaderboard must run each participant in
an isolated container or VM, apply resource limits, and use private generated
holdout tasks. Never include customer traces, credentials, or private task seeds
in public demonstrations.

The schema is deliberately compatible with the MCP tools documented in
[`docs/MCP_SERVER.md`](../../docs/MCP_SERVER.md). Future task versions must
change the benchmark version and retain the old task file for reproducibility.
