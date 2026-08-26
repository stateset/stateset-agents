# Fireworks AI provider (`train-remote --provider fireworks`)

> **Not yet run against the live service.** The adapter is written against the
> real `fireworks-ai` SDK (version 1.x, the typed client generated from
> Fireworks' OpenAPI spec), so the resource names, keyword arguments, and
> response fields are taken from the installed client rather than from prose
> docs — the *shapes* are trustworthy. What is unverified is the service's
> *behaviour*: state-transition timing, whether your account can download a
> PEFT addon's weights, and the deployment/addon-load sequence. The specific
> assumptions most likely to bite are in
> [Unverified assumptions](#unverified-assumptions), each with the symptom it
> would produce.

## What Fireworks is here

Fireworks sits between the two shapes already in this package:

| | Modal / RunPod | **Fireworks** | River |
|---|---|---|---|
| Who owns the machine | you rent one | Fireworks (training) | nobody |
| What we ship | the training script | a dataset | gradients, call by call |
| `submit()` returns | after the run | as soon as the job exists | after the run |
| Result | local adapter | addon on Fireworks (± local copy) | `river://` pointer |

You upload a chat-format dataset, create a supervised fine-tuning job, and
Fireworks trains a LoRA addon on hardware it schedules itself. The job is
genuinely asynchronous: the job id stays meaningful after your process exits
and matches what you see in the Fireworks dashboard.

## Setup

```bash
pip install 'stateset-agents[fireworks]'
export FIREWORKS_API_KEY=fw_...        # app.fireworks.ai → Settings → API keys
export FIREWORKS_ACCOUNT_ID=my-org     # the account slug, not the display name
```

Both are read at submit time and never written into the job spec.

## Training

```bash
stateset-agents train-remote \
  --provider fireworks \
  --dataset outputs/improved/curated.jsonl \
  --base-model accounts/fireworks/models/qwen3p5-9b \
  --output-dir outputs/sft_fw_v1 \
  --num-epochs 2 --lora-r 16 --learning-rate 1e-4 --max-length 2048
```

Progress is only visible after the job resolves unless you ask for it live:

```bash
STATESET_FIREWORKS_VERBOSE=1 stateset-agents train-remote --provider fireworks …
```

`wait()` polls every 15 seconds rather than the framework-default 1 second — a
managed fine-tune queues and then runs for minutes to hours, so a per-second
poll of the control plane learns nothing.

### Spec fields that are ignored

`--gpu`, `--gpu-count`, `--container-disk-gb`, `--cloud-type`, and
`--network-volume-id` describe rented machines. Fireworks picks the training
hardware itself, so they are accepted and logged as ignored rather than
rejected — the same spec should be submittable to any provider. They matter
again only at `--deploy` time, which takes its accelerator explicitly.

`--resume` is likewise a no-op: there is no pod disk to resume from.

## What lands on disk

`fetch()` always writes:

- `fireworks_checkpoint.json` — the pointer: account, job name, output model
  id, base model, dataset id, hyperparameters, Fireworks' own cost estimate,
  and the OpenAI-compatible inference base URL.
- `stateset_manifest.json` — the usual provenance record, so
  `stateset-agents adapters` works exactly as it does for local runs.

It *also* attempts to download the addon's files through
`models.get_download_endpoint`. When that succeeds you get real adapter
weights and `stateset-agents serve --checkpoint <dir>` works; when it does not
(the API declines, or the download fails mid-stream) the pointer is still
written and `weights_downloaded` is `false`. A partially downloaded file is
deleted rather than left to masquerade as adapter weights.

Sampling the hosted addon directly:

```python
from openai import OpenAI

client = OpenAI(
    base_url="https://api.fireworks.ai/inference/v1",
    api_key=os.environ["FIREWORKS_API_KEY"],
)
client.chat.completions.create(
    model="accounts/my-org/models/sft-abc",   # pointer["model"]
    messages=[{"role": "user", "content": "…"}],
)
```

## Serving it on rented hardware

`--deploy` creates an on-demand Fireworks deployment of the *base* model with
addons enabled, then loads the tuned LoRA onto it:

```bash
stateset-agents train-remote --provider fireworks … \
  --deploy --deploy-accelerator NVIDIA_H100_80GB
```

This is the step that actually rents hardware, and **it bills for as long as
the deployment exists**. It is therefore a separate call, never part of
`submit()`, and `min_replica_count` defaults to 0 so an idle deployment scales
to nothing. Tear it down explicitly:

```bash
stateset-agents undeploy --deployment accounts/my-org/deployments/dep-1
```

If the deployment is created but the addon fails to load, the error names the
deployment and tells you to delete it — a half-built deployment still bills.

## Cost accounting

Every terminal job appends one line to the cost ledger
(`stateset-agents costs`). The dollar figure is Fireworks' own
`estimatedCost` from the job resource, converted from its `{units, nanos}`
form — not a price computed here. When the job reports no estimate, the ledger
records `null`: unknown, never zero, so a budget check cannot silently pass.

Deployment costs are **not** in the ledger. They accrue per hour outside any
job lifecycle, and inventing a number for them would be worse than the
explicit teardown instruction.

## Limits

- **Jobs are restart-safe on the submitting machine.** The non-secret training
  spec and Fireworks resource ids are written atomically under the StateSet
  cache directory. A later CLI process can poll or fetch the job with
  `stateset-agents remote-job --job-id <id> [--wait|--fetch]`. Moving to
  another machine still requires moving that metadata file or downloading the
  addon from the Fireworks console.
- **`logs()` are progress events, not trainer stdout.** The fine-tuning API
  exposes state and percent-complete, not the training log. Lines look like
  `fireworks job running - 40% - epoch 1 - 12403 tokens`.
- **Cancel is a delete.** The SDK exposes deletion rather than a cancel verb
  for supervised fine-tuning jobs (reinforcement jobs do have `cancel`), so
  that is what `cancel()` calls to stop the billing.
- **Reinforcement fine-tuning is not wired up.** Fireworks has an RFT API
  (`reinforcement_fine_tuning_jobs`, plus evaluators), but it needs a reward
  definition that has no equivalent in `RemoteJobSpec` today. SFT only.

## Unverified assumptions

| Assumption | Symptom if wrong |
|---|---|
| `datasets.create` accepts `format: "CHAT"` with `user_uploaded: {}` and the upload follows as a separate call | `create` or `upload` fails with an argument error before the job is created; no money spent |
| A bare dataset id (not the full `accounts/…/datasets/…` name) is accepted as the job's `dataset` | job creation fails with a not-found; fix is to pass the full resource name |
| `batch_size` on the job means per-device batch size | training runs but with a different effective batch than the spec asked for — check `batch_size` on the returned job resource |
| `JOB_STATE_EARLY_STOPPED` means a trained addon exists | a job reports success with no usable `output_model`; `fetch()` then writes a pointer with `model: null` |
| A PEFT addon is downloadable via `models.get_download_endpoint` | `weights_downloaded` is always `false` and `serve --checkpoint` never works — the pointer path is the fallback and is expected to be the common case |
| The training `--base-model` is also a valid deployment `baseModel` | `--deploy` fails with a not-found while the fine-tune itself succeeded; pass the Fireworks model id (`accounts/fireworks/models/…`) rather than the bare HF name |
| A deployment created with `enable_addons=True` accepts `lora.load(model=…, deployment=…)` | `--deploy` creates a billing deployment and then errors; the error tells you to run `undeploy` |
| Fireworks' `estimatedCost` is populated for completed SFT jobs | ledger rows show `null` cost, which is honest but less useful |

When you run this live, the fixes belong in
`tests/unit/test_remote_fireworks_executor.py` first — the fakes there are a
recording of the call sequence we believe in, so correcting them is how the
correction gets pinned down.
