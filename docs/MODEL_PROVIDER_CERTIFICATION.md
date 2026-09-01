# Model and provider certification

StateSet uses explicit product tiers and evidence stages so that “the loader
accepts this repository” is never confused with “this combination works in
production.” The machine-readable source is
`stateset_agents.remote.model_catalog`; inspect it with:

```bash
stateset-agents model-support --json
```

## Product tiers

| Tier | Meaning |
|---|---|
| `default` | Recommended starting points with retained live training evidence |
| `frontier-preview` | Architecture support exists, but the expensive end-to-end proof is incomplete |
| `compatibility` | Supported through a starter or the generic Hugging Face path; topology may require tuning |

## Certification stages

Stages are cumulative product gates:

1. `configured` — model id, loader, tokenizer, and dependency path exist.
2. `smoke-tested` — architecture and LoRA-target tests pass without claiming a successful remote training run.
3. `training-verified` — a live bounded job returned loadable artifacts and retained cost/cleanup evidence.
4. `serving-verified` — the trained result answered held-out requests through the supported serving path.
5. `production-certified` — training and serving are periodically rechecked, recovery is exercised, and no unresolved cleanup or security gate remains.

No model/provider combination is currently labeled `production-certified`.
That designation is intentionally harder to earn than a successful demo.

## Non-billable RunPod planning

`--plan-only` resolves catalog defaults without constructing an executor,
contacting RunPod, or provisioning hardware:

```bash
stateset-agents train-remote \
  --provider runpod \
  --dataset data/train.jsonl \
  --base-model Qwen/Qwen3.8-Flash-Next \
  --plan-only
```

For measured models, omitted GPU/count/disk options are filled automatically.
Frontier and unknown models use estimated resources and require `--max-cost`
before a live run. Explicit `--gpu`, `--gpu-count`, and
`--container-disk-gb` values always win. `--no-auto-resources` restores raw
provider defaults.

`--dry-run` is different: it executes the packaged training command in dry-run
mode and a remote provider may still allocate compute. Use `--plan-only` when
the requirement is zero billing.

## Current golden paths

- Development/CI: `Qwen/Qwen3.5-0.8B` on one RunPod RTX A4000.
- Serious single-node training: `Qwen/Qwen3.8-27B` on one RunPod H100 80GB.
- Frontier qualification: Qwen3.8-Flash-Next first, then GLM-5.3-Flash, with
  explicit spend ceilings and retained train/eval/serve/cleanup evidence.
- Managed alternative: River for remote-autograd RL. Fireworks lifecycle and
  Modal transport remain certification work, not production claims.

The manual `Modal Transport Verify` workflow exercises the real image,
function, A10 GPU, local-dataset upload, mounted read, output commit, and Volume
cleanup path. The executor deletes its per-job persistent Volume on success,
training failure, dry-run completion, and transport error; the workflow asserts
that the remote log names the mounted dataset, compares pre/post account
inventories, and retains the evidence. Modal remains `transport-unverified`
until that workflow passes on a live account.

## Promotion gate

A frontier model is promoted only after the evidence artifact identifies the
commit, dataset and wheel digests, exact GPU topology, duration, spend,
evaluation assertions, returned artifact hashes, and confirmed resource
cleanup. Provider authentication canaries do not satisfy this gate by
themselves.
