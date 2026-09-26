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

The version 0.2 executable task set and version 0.1 plan scorer remain
available for reproducibility. Version 0.1 uses self-reported tool calls and
artifacts, so its score is diagnostic only.

This local runner is for trusted submissions. Temporary directories and path
checks confine the supported tool arguments, but they do not provide OS-level
isolation for hostile code. A hosted leaderboard must run each participant in
an isolated container or VM, apply resource limits, and use private generated
holdout tasks. Never include customer traces, credentials, or private task seeds
in public demonstrations.

The schema is deliberately compatible with the MCP tools documented in
[`docs/MCP_SERVER.md`](../../docs/MCP_SERVER.md). Future task versions must
change the benchmark version and retain the old task file for reproducibility.
