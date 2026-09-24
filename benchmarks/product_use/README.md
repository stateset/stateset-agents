# StateSet product-use benchmark

This is the public starter set for evaluating whether an agent can use the
StateSet product surface. It covers the MCP improvement loop, training preview,
and a basic safety refusal. The task file is intentionally small and
human-readable so other harnesses can adapt it.

The public runner is deterministic and does not execute model-generated tools:

```bash
python -m benchmarks.product_use.runner \
  --tasks benchmarks/product_use/tasks.public.json \
  --submissions submissions.json
```

The current public task schema is version `0.1`. Missing tasks count as zero in
the aggregate score, and unknown task IDs are rejected. This keeps partial local
submissions useful for debugging while preventing incomplete runs from looking
like a full benchmark result.

A submission is a JSON array. Each row contains `task_id`, a `tool_calls` list,
an `artifacts` object, and `refused_unsafe_action` when relevant. The runner
reports task completion, artifact, safety, and format components.

This public set is not the publication leaderboard. A production evaluation
should execute tools in an isolated temporary workspace and keep generated
holdout tasks private. Never include customer traces, API keys, provider
credentials, or private task seeds in a public corpus. Public training traces
should be separately licensed and labeled as demonstrations rather than
evaluation evidence.

The schema is deliberately compatible with the MCP tools documented in
[`docs/MCP_SERVER.md`](../../docs/MCP_SERVER.md). Future task versions must
change the benchmark version and retain the old task file for reproducibility.
