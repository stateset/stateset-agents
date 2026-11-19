# StateSet Agents quickstart

Run an offline loop from conversation logs to a graded, curated training
dataset. Then preview training and serving. The commands below work from an
installed Python package without a repository checkout, GPU, or API key.

`traceforge-rl` and `import traceforge_rl` are aliases for `stateset-agents`
and `stateset_agents`. See the [rename guide](docs/RENAME_TRACEFORGE.md).

## 1. Install

Use Python 3.10 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install stateset-agents
```

The packaged grading and dataset-preparation modules described here require a
release containing these changes. To verify the current source before that
release is published, run `python -m pip install .` from its checkout, then
follow the remaining steps from any directory.

## 2. Grade and curate sample conversations

The sample contains one weak reply and two replies that pass the heuristic
customer-support rubric. The rubric scores tone and relevance; review real
examples against your own business policy before using them for training.

The following steps use Bash. Artifacts stay in the printed working directory.

<!-- quickstart-smoke:start -->
```bash
WORKDIR="$(mktemp -d -t stateset-quickstart.XXXXXX)"
export WORKDIR

python - <<'SAMPLE'
import json
import os
from pathlib import Path

conversations = [
    ("My order has not arrived. Can you help?", "No."),
    (
        "I need a refund for my damaged order.",
        "I am sorry your order arrived damaged. Thank you for letting us know. "
        "I would be happy to help with your refund. Please share your order "
        "number so I can check the return policy and explain the next steps.",
    ),
    (
        "Can you check my order status? It has been a week.",
        "Of course, happy to help! I understand waiting a week is frustrating. "
        "Please share your order number and I will look up the latest shipping "
        "status and tracking details for you.",
    ),
]
path = Path(os.environ["WORKDIR"]) / "logs.jsonl"
with path.open("w", encoding="utf-8") as handle:
    for question, answer in conversations:
        handle.write(json.dumps({"messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]}) + "\n")
print(f"Working directory: {path.parent}")
SAMPLE

python -m stateset_agents.cli ingest --format openai --input "$WORKDIR/logs.jsonl" --output "$WORKDIR/transcripts"
python -m stateset_agents.cli improve run --transcripts "$WORKDIR/transcripts" --reward customer_support --output "$WORKDIR/improved"
python -m stateset_agents.cli improve status --output "$WORKDIR/improved"
```

The output directory contains `improve_summary.json`, `curated.jsonl`, and
`next_steps.md`. Confirm that grading kept the useful examples:

```bash
python - <<'CHECK'
import json
import os
from pathlib import Path

output = Path(os.environ["WORKDIR"]) / "improved"
summary = json.loads((output / "improve_summary.json").read_text())
assert summary["transcript_count"] == 3
assert summary["assistant_turn_count"] == 3
assert summary["curated_count"] == 2
rows = [json.loads(line) for line in (output / "curated.jsonl").read_text().splitlines()]
assert len(rows) == 2
assert all(row["response"] != "No." for row in rows)
print("Graded three conversations and retained two examples.")
CHECK
```

## 3. Prepare data and preview training

Convert the curated pairs into chat-format JSONL, then validate an SFT plan:

```bash
python -m stateset_agents.data.prepare_sft_dataset --input "$WORKDIR/improved/curated.jsonl" --format chat --output "$WORKDIR/improved/sft_train.jsonl" --min-score 0.7 --dedup
python -m stateset_agents.training.sft --dataset "$WORKDIR/improved/sft_train.jsonl" --base-model Qwen/Qwen3.5-0.8B --output-dir "$WORKDIR/sft_v1" --num-epochs 3 --lora-r 16 --dry-run
python -m stateset_agents.training.gspo_preview --model qwen3.5-0.8b --task customer_support
python -m stateset_agents.cli serve --dry-run
```
<!-- quickstart-smoke:end -->

These previews download no model weights, perform no optimizer updates, and
create no trained checkpoint. The GSPO preview resolves hyperparameters; its
task label does not select a task-specific reward or environment.

For real local SFT, install `stateset-agents[training]`, use a supported GPU,
and rerun the SFT command without `--dry-run`. Review held-out quality before
serving. See [the improvement pipeline](docs/IMPROVEMENT_PIPELINE.md) for
validation/test separation and promotion criteria.

## 4. Plan remote training

Choose a provider explicitly. RunPod has retained live training evidence for
the configurations in the [RunPod guide](docs/RUNPOD_GUIDE.md). The CLI's
compatibility default remains Modal, whose live transport is unverified.
Provider support does not certify every model or hardware combination.

This command prints a local resource plan and allocates no compute:

```bash
python -m stateset_agents.cli train-remote --provider runpod --dataset "$WORKDIR/improved/sft_train.jsonl" --base-model Qwen/Qwen3.5-0.8B --output-dir "$WORKDIR/sft_v1" --plan-only
```

For a live run, follow the provider guide to configure credentials and choose
a timeout and `--max-cost` budget. Remote `--dry-run` can still rent hardware;
use `--plan-only` for a non-billable preview.

## 5. Serve an evaluated checkpoint

After real training and evaluation have produced an adapter:

```bash
python -m pip install 'stateset-agents[training,api]'
python -m stateset_agents.cli serve --checkpoint "$WORKDIR/sft_v1" --host 127.0.0.1 --port 8000
```

See the [API guide](docs/GETTING_STARTED_API.md) for authentication and production
configuration. The earlier dry runs do not produce an adapter to serve.

## Run experiments on StateSet Swarm

Fan out a GRPO run and a verified checkpoint edit to two sandboxes. The guide
includes a local replay, hosted launch commands and sample results for three RL
seeds. [Swarm getting started](examples/swarm/README.md)

## Verification

The wheel CI runs the marked offline steps above from outside the checkout,
including in a minimal environment without PyTorch. It also exercises MCP
grading and preset-preview functions against the installed wheel. This proves
the local workflow; it does not certify remote training or model-quality gains.

Continue with the [platform tour](docs/PLATFORM_TOUR.md) or
[cookbook](docs/COOKBOOK.md).
