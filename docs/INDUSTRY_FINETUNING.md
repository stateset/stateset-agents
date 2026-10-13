# Industry fine-tuning

Use the same SFT pipeline with separate data and LoRA adapters for each industry.
These recipes provide task scope, synthetic format examples, and evaluation
criteria. They are not pretrained industry agents or evidence of business outcomes.

| Recipe | Starter tasks |
| --- | --- |
| `financial-services` | Account support, onboarding, dispute intake |
| `retail` | Order tracking, returns, product assistance |
| `travel` | Booking support, itinerary changes, guest assistance |
| `healthcare` | Scheduling, patient navigation, administrative support |
| `services` | Lead qualification, estimate intake, booking |
| `media` | Subscriber onboarding, billing, subscription management |
| `telecommunications` | Troubleshooting, billing, plan support |
| `technology` | Product onboarding, technical support, account assistance |
| `public-sector` | Service navigation, application assistance, case status |

## Prepare and preview without a GPU

```bash
stateset-agents industry list
stateset-agents industry show retail
stateset-agents industry init retail ./retail-source

# Replace or expand these four synthetic demonstrations with reviewed data.
stateset-agents industry validate ./retail-source/examples.jsonl
stateset-agents industry prepare retail ./retail-source/examples.jsonl ./retail-run \
  --model qwen3.5-2b --validation-fraction 0.2 --seed 42
stateset-agents industry train ./retail-run --dry-run
```

Catalog, validation, preparation, and previews use the base installation. They
do not import Torch, Transformers, PEFT, or Datasets, or download a checkpoint.
Output directories must be new; existing files are never overwritten.

`prepare` creates `train.jsonl`, `validation.jsonl`, and `manifest.json`. The
manifest records the model ID, split seed, source and output SHA-256 hashes,
counts, exact duplicates removed, and evaluation criteria. Editing either split
after preparation causes training to fail. Prepare a new directory after edits.
For reproducible training, pass `--model-revision` with the model repository's
immutable 40-character commit hash. Both tokenizer and weights load that revision;
the adapter lineage records it, and evaluation rejects predictions declaring a
different revision. Omitting the option retains the model repository's default
revision for compatibility and does not provide an immutable model pin.

## Dataset contract and split boundaries

Every nonblank JSONL line must be an object with a `messages` list. Conversations
contain text messages using `system`, `developer`, `user`, `assistant`, and `tool`
roles and end with an assistant target. Tool declarations use the OpenAI-style
`tools` array. Assistant `tool_calls` use unique IDs and function arguments that
are objects or JSON object strings. Tool results reference a preceding call ID.
A final assistant tool call without a result is allowed as the target; its call
ID may be omitted for native formats such as FunctionGemma.

Malformed rows fail with their physical line number. The validator checks
structure and tool references; it does not prove business correctness or validate
arbitrary JSON Schema constraints. Multimodal message content is not supported by
this workflow. Rendering preserves typed tool arguments and per-record schemas.

Use a stable `group_id` for records from the same case, customer, or source
conversation. Exact duplicate records are removed. Records with the same initial
user request (ignoring case and repeated whitespace) or `group_id` stay together,
including transitive connections. At least two independent groups are required.
The holdout fraction applies to groups, so row proportions can differ. Reordering
the input does not change a split with the same seed.

This prevents those explicit overlaps, not semantic paraphrase leakage. Choose
group IDs and review holdouts to reflect the real evaluation boundary.

## Train and evaluate

```bash
pip install 'stateset-agents[small-models]'
stateset-agents industry train ./retail-run --num-epochs 3 --max-length 1024
```

Execution uses the existing SFT trainer: BF16 LoRA, rank 16, alpha 32, learning
rate `2e-5`, batch size 1, accumulation 8, and training seed 42. `--seed` on
`prepare` controls the dataset split. This workflow requires CUDA for execution
and returns an error when it is unavailable. CPU previews are explicitly marked
`planned`; they never claim to have trained an adapter. Use the existing model
starter commands for GSPO and QLoRA profiles.

The model's own chat template must support your roles and tool schemas. A
registered checkpoint ID is not a guarantee that every dataset format suits that
model; specialized tool-only models require appropriately scoped data.

The saved adapter and normal SFT lineage manifest go into `retail-run/adapter`.
`industry_run.json` records the plan, data hashes, and running/trained/failed
status. Existing adapter directories are refused to prevent accidental overwrite.
Serve the adapter through `AgentConfig.peft_path` as with any existing SFT output.

Training leaves validation data untouched. Use the paired evaluator below to
measure reference agreement before promoting an adapter. Neither successful
training nor the synthetic examples establish industry quality. Independently
review grounded responses, missing-information handling, appropriate escalation,
and the recipe's criteria. Current facts belong in tools or retrieval; training
examples teach how to use them.

## Compare the base model and adapter

The evaluator scores **every assistant turn**, including intermediate tool calls.
Both models receive the same reference conversation prefix and tool schemas for
each turn. Tool names, call order, and typed arguments must match; generated call
IDs and JSON object key order may differ. Text comparison is case-sensitive and
ignores repeated whitespace. Extra tool calls fail even if the text matches.
Malformed replies and backend errors count as failed cases, rather than being
dropped from the denominator.

This is **teacher-forced reference agreement**, not autonomous task success.
Earlier reference responses and tool results are supplied again at each turn;
generation errors do not propagate through a live conversation. Equivalent
paraphrases can fail exact text matching. The report therefore does not label
this score as hallucination detection, clinical safety, or business performance.

### Run inference through Python callbacks

`collect_industry_predictions` accepts any synchronous inference backend. Its
callback receives `(messages, tools)` and returns an object containing
`response`, an assistant message with `role`, `content`, and optional OpenAI-style
`tool_calls`. Optional `generated_tokens` and `cost_usd` are measurements for
that request. Missing measurements remain `null`, never zero. Exceptions are
retained as failures. The collector measures request latency, including callback
overhead; load models before invoking it to exclude load time.

The following function runs a complete comparison once you supply your base and
adapter callbacks. Each callback must apply the declared generation settings.

```python
import json
from pathlib import Path

from stateset_agents.evaluation.industry import (
    IndustryEvaluationPolicy,
    collect_industry_predictions,
    evaluate_industry_project,
    export_industry_evaluation,
)


def compare_models(project, infer_base, infer_adapter, revision, adapter_sha256):
    requests = export_industry_evaluation(project)
    model = {"base_model": requests["base_model"], "revision": revision}
    settings = {"do_sample": False, "max_new_tokens": 256}
    baseline = collect_industry_predictions(
        project, infer_base, variant="baseline", model=model, settings=settings,
    )
    candidate = collect_industry_predictions(
        project, infer_adapter, variant="candidate",
        model={**model, "adapter": {
            "id": str(Path(project) / "adapter"), "sha256": adapter_sha256,
        }},
        settings=settings,
    )
    # Preserve raw responses for reproducibility and independent review.
    for name, bundle in (("baseline", baseline), ("candidate", candidate)):
        with (Path(project) / f"{name}.json").open("x") as stream:
            json.dump(bundle, stream, indent=2, allow_nan=False)
    report = evaluate_industry_project(
        project, baseline, candidate,
        policy=IndustryEvaluationPolicy(min_groups=30, min_success_rate=0.9),
    )
    with (Path(project) / "evaluation.json").open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)
    return report
```

Use an immutable base checkpoint revision and a SHA-256 fingerprint of the
adapter weights. Identities and settings are caller declarations: hashing a
prediction file binds the report to those predictions, but cannot attest which
model generated them. Verify checkpoint loading and adapter effects separately.
The callbacks never receive the current target or future messages. They do
receive earlier reference turns; send each prefix independently to your backend.

### Export requests and gate saved predictions

For inference outside the Python process, export the same requests:

```bash
stateset-agents industry eval-export ./retail-run ./requests.json
```

Create one prediction bundle per model. Copy `suite_sha256` and `base_model`
from the export, and supply exactly one prediction for every exported `case_id`:

```json
{
  "schema_version": 1,
  "kind": "stateset-industry-predictions",
  "suite_sha256": "<copied from requests.json>",
  "variant": "baseline",
  "model": {"base_model": "<copied from requests.json>", "revision": "<checkpoint revision>"},
  "settings": {"do_sample": false, "max_new_tokens": 256},
  "predictions": [{
    "case_id": "<copied case_id>",
    "response": {"role": "assistant", "content": "Generated response"},
    "elapsed_seconds": 0.5,
    "generated_tokens": 12,
    "cost_usd": null
  }]
}
```

For the candidate, set `variant` to `candidate` and add
`model.adapter = {"id": "adapter identifier", "sha256": "<64 lowercase hex characters>"}`.
Its base revision and generation settings must match the baseline. A failed
request still needs its case ID, elapsed time, `response: null`, and a nonempty
`error` string. Do not omit failed cases or substitute baseline predictions.

```bash
stateset-agents industry evaluate ./retail-run \
  --baseline ./baseline.json --candidate ./candidate.json \
  --output ./evaluation.json --min-groups 30 --min-success-rate 0.9 \
  --min-improvement 0.0 --max-regression-rate 0.0
```

The default gate requires at least 30 independent prompt/source groups, at least
90% candidate group reference agreement, no decline in group agreement, and no
previously correct case becoming incorrect. A group passes only when **all** its
assistant turns match. Transitive prompt and source-ID links use the same grouping
as preparation, so repeated conversations cannot inflate the sample count.

Synthetic-marked data cannot pass the gate, even with perfect scores. The scorer
does not infer whether unmarked data is synthetic or independently reviewed.
Optional `--max-mean-latency-seconds` and `--max-mean-cost-usd` set per-request
resource limits; a required but unknown cost fails the gate. Lowering thresholds
changes the policy recorded in the report; it does not improve the evidence.
These are post-run acceptance limits, not inference spending caps. Apply request
timeouts, cancellation, and spending budgets in your inference backend.

Exit codes are `0` for a passing configured gate, `1` for a failed gate with a
saved report, and `2` for invalid inputs or output errors. Output files must be
new. Reports contain per-case failures, paired improvements/regressions, group
scores, usage, policy, and hashes of the suite and both prediction bundles.
They do not modify the prepared manifest or claim a statistical significance
test. Reserve a fresh final test set if you repeatedly tune against this holdout.

## Live pilot evidence

The [2026-10-07 Qwen3.5 2B/4B pilot](../benchmark_results/industry_pilot/20261007/README.md)
verified six optimizer updates per model, saved-adapter reload effects, artifact
hashes, and reproducible paired evaluation on real GPUs. Both synthetic-data
quality gates failed, and neither model improved its reference score. Raw
predictions, failed attempts, provider cleanup, and compute estimates are retained.

## Python API

```python
from stateset_agents.data import load_finetuning_data, split_finetuning_data
from stateset_agents.training.industry import (
    prepare_industry_training,
    train_industry_project,
)

rows = load_finetuning_data("reviewed_conversations.jsonl")
train, validation = split_finetuning_data(rows, validation_fraction=0.2, seed=42)

manifest = prepare_industry_training(
    "technology", "reviewed_conversations.jsonl", "technology-run",
    model="smollm3-3b",
)
plan = train_industry_project("technology-run")  # defaults to a preview
# On a configured CUDA host:
# result = train_industry_project("technology-run", dry_run=False)
```
