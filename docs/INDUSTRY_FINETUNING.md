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

The validation file is **held out, not automatically evaluated**. Neither a
successful training run nor the synthetic examples establish industry quality.
Compare the base model and adapter on those independent cases, including tool
selection and arguments, grounded responses, missing-information handling,
appropriate escalation, and the recipe's criteria. Current facts belong in tools
or retrieval; training examples teach how to use them.

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
