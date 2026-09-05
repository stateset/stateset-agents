# StateSet Agents CLI Reference

Use `stateset-agents --help` to see the current runtime command list.

## Commands

### `stateset-agents provider-canary`

Run read-only live authentication, SDK compatibility, and cleanup checks for
River, RunPod, and Fireworks. No training job, pod, or deployment is created.

```bash
stateset-agents provider-canary --provider runpod --strict
stateset-agents provider-canary --provider river --provider fireworks \
    --strict --output provider-canary.json
```

Without `--strict`, missing credentials are reported as `skipped`. Scheduled
CI uses `--strict`, so absent credentials or leaked `stateset-canary-*`
resources fail the job. See `docs/PROVIDER_CANARIES.md`.

### `stateset-agents version`

Show the installed version and runtime details.

```bash
stateset-agents version
stateset-agents version --json
```

### `stateset-agents train`

Run a lightweight training demo by default with `--stub`, or launch the configured training flow.

```bash
stateset-agents train
stateset-agents train --stub
stateset-agents train --config ./stateset_agents.yaml --episodes 10 --profile balanced
stateset-agents train --stub --dry-run
stateset-agents train --list-objectives
stateset-agents train --objective rloo --no-dry-run
```

#### Options

- `--config PATH`: YAML or JSON config file.
- `--episodes INTEGER`: Override number of episodes (must be > 0).
- `--save PATH`: Optional checkpoint output directory.
- `--dry-run / --no-dry-run`: Validate configuration and print guidance.
- `--stub`: Run a fast stub flow with no external model downloads.
- `--profile [balanced|speed|quality]`: Training profile.
- `--objective NAME`: Policy objective preset (`grpo`, `dr_grpo`, `bnpo`, `dapo`, `gspo`, `gspo_token`, `gepo`, `rloo`, `reinforce_pp_baseline`, `cispo`, `ppo`). Defaults to the trainer's native objective; see [OBJECTIVES.md](OBJECTIVES.md).
- `--list-objectives`: Describe every objective preset and exit.

### `stateset-agents train-remote`

Run the SFT job from `improve` on local or rented GPU compute. Picks up where
`improve` leaves off — it consumes `curated.jsonl` and writes a trained adapter.

```bash
# On this machine
stateset-agents train-remote --dataset improved/curated.jsonl \
    --base-model Qwen/Qwen3.5-0.8B

# On rented GPUs
stateset-agents train-remote --provider modal --gpu A100 \
    --dataset improved/curated.jsonl --base-model Qwen/Qwen3.5-0.8B

# Optional: reference provider-side Secrets and constrain placement.
# Secret values are never serialized into the StateSet job payload.
export STATESET_MODAL_SECRET_NAMES=huggingface,weights-and-biases
export STATESET_MODAL_REGION=us-east

# Or on RunPod (GPU names are RunPod's own, e.g. "NVIDIA RTX A4000")
export RUNPOD_API_KEY=...
stateset-agents train-remote --provider runpod --gpu "NVIDIA RTX A4000" \
    --dataset improved/curated.jsonl --base-model Qwen/Qwen3.5-0.8B

# Large checkpoint on RunPod, with a base-vs-tuned comparison afterwards
stateset-agents train-remote --provider runpod --gpu "NVIDIA H100 80GB HBM3" \
    --container-disk-gb 160 --eval-prompts prompts.txt \
    --dataset improved/curated.jsonl --base-model meta-models/Muse-Glimmer-30B

# See the plan without training (works with no GPU)
stateset-agents train-remote --dataset improved/curated.jsonl \
    --base-model Qwen/Qwen3.5-0.8B --dry-run
```

The job itself is `stateset_agents.training.sft` — identical whichever provider
runs it (`scripts/sft_from_curated.py` is a thin CLI over the same code).
Remote runs install a pinned published `stateset-agents[training]` rather than
syncing your working tree, so a remote run is reproducible; the tradeoff is
that testing an unreleased change remotely needs a dev release.

A remote run succeeds only if it actually produces an adapter: a container
that exits cleanly having written nothing is reported as a failure, not a
success with an empty output directory.

##### `--provider river`

River AI is a remote *autograd* service rather than a machine you rent, so
this provider behaves differently from `modal`/`runpod` in three ways worth
knowing before you use it:

```bash
export RIVER_API_KEY=rv_...
stateset-agents train-remote --provider river \
    --dataset improved/curated.jsonl --base-model Qwen/Qwen3.5-9B --lora-r 16
```

- The trained LoRA stays on River. `--output-dir` receives a
  `river_checkpoint.json` pointer (the `river://` URI, base model, LoRA config,
  step/loss summary) and a `stateset_manifest.json`, **not** adapter weights —
  so `stateset-agents serve --checkpoint` cannot load the result.
- Hardware options (`--gpu`, `--gpu-count`, `--container-disk-gb`,
  `--cloud-type`, `--network-volume-id`) have no River equivalent and are
  ignored with a log line rather than raising.
- River bills per token and quotes no price to the SDK, so the cost ledger
  records the run with `cost_usd: null` — unknown, never zero. `--max-cost`
  therefore cannot be checked against a River run.
- LoRA rank is capped at 32 by River and is validated locally.

**Live-verified 2026-08-18**: a real run trained Qwen3.5-9B (140 rows,
3 epochs) and sampling the `river://` checkpoint answered 3/3 held-out
prompts with the trained behaviour. Note `river-client` requires Python
≥3.12 (this repo runs 3.10 — use a separate venv for River runs). See
`docs/RIVER_PROVIDER.md`.

#### The `fireworks` provider

`--provider fireworks` is Fireworks AI's **managed fine-tuning service**. Like
River it schedules its own hardware, so the same machine-shaped options are
ignored; unlike River the job is asynchronous (the job id stays valid after
your process exits) and the trained LoRA addon may be downloadable.

- `fetch()` always writes `fireworks_checkpoint.json` plus the usual manifest,
  and additionally downloads the addon's weights when the API offers them.
  `weights_downloaded` in the pointer says whether `serve --checkpoint` will
  work.
- `--deploy` (Fireworks only) creates an on-demand deployment of the base
  model with addons enabled and loads the tuned LoRA onto it, printing an
  OpenAI-compatible base URL. It rents hardware and bills until deleted —
  tear it down with `stateset-agents undeploy --deployment <name>`.
- The cost ledger records Fireworks' own `estimatedCost`, or `null` when the
  job reports none. Deployment cost is not in the ledger.
- Needs `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID`, and the
  `stateset-agents[fireworks]` extra.

**Not yet live-verified** — written against the real `fireworks-ai` 1.x SDK,
exercised only against fakes. See `docs/FIREWORKS_PROVIDER.md`.

#### Options

- `--dataset PATH`: Chat-format JSONL to train on (required).
- `--base-model TEXT`: Hugging Face base model (required).
- `--provider [coreweave|fireworks|local|modal|nebius|river|runpod]`: Where to
  run. Default
  `local`. `river` and `fireworks` are different in kind from the others —
  see the notes above.
- `--deploy` / `--deploy-accelerator TEXT`: Fireworks only — serve the tuned
  addon on an on-demand deployment after training.
- `--output-dir PATH`: Adapter output directory. Default `outputs/sft_v1`.
- `--num-epochs`, `--lora-r`, `--lora-alpha`, `--learning-rate`,
  `--max-length`, `--per-device-batch-size`,
  `--gradient-accumulation-steps`: Passed through to the training script.
- `--gpu TEXT`: GPU type to request in provider vocabulary (remote only).
  Modal defaults to `A10`; multi-GPU requests render as `GPU:count`.
- `--gpu-count INTEGER`: Number of GPUs to request on providers that expose
  topology (RunPod, CoreWeave, and Nebius). Nebius requires a matching
  `NEBIUS_PRESET`. Default `1`. With more than one, the
  training job loads the
  base model with `device_map="auto"`, sharding the checkpoint across every
  visible GPU — this is what lets a model bigger than one card train at all
  (verified live: Muse-Glimmer-30B split ~evenly across 2x H100). Single-GPU
  behavior is unchanged. Billing scales with the count.
- `--timeout INTEGER`: Job timeout in seconds. Default `3600`.
- `--package-version TEXT`: Version installed remotely. Defaults to the
  running version.
- `--container-disk-gb INTEGER`: Container/scratch disk in GB on RunPod,
  CoreWeave, and Nebius. Size it at roughly 2.5x the model download (a 30B
  BF16 checkpoint is
  ~63GB and dies mid-download on the 40GB default). Defaults to the
  executor's default (40).
- `--cloud-type [SECURE|COMMUNITY]`: RunPod only — which pod pool to rent
  from. `SECURE` (default) is reserved capacity; `COMMUNITY` is spot-priced —
  markedly cheaper, but the pod can be reclaimed mid-job. When a pod dies
  under a running job (any cloud type — observed live even on SECURE), the
  executor terminates the dead pod, provisions a fresh one, re-uploads the
  inputs, and reruns (bounded by the executor's `max_provision_attempts`,
  default 2). Without `--network-volume-id` the rerun **restarts training
  from scratch** — the dead pod's checkpoints lived on its container disk and
  died with it; with a volume attached, the rerun **resumes from the newest
  surviving checkpoint** automatically.
- `--network-volume-id TEXT`: RunPod only — id of an **existing** RunPod
  network volume, mounted at `/workspace` so checkpoints land on durable
  storage that outlives the pod. With it, the pod-died-mid-job retry re-runs
  with `--resume` and an interruption costs at most one epoch, not the whole
  run. Volumes are datacenter-scoped, so the pod is pinned to the volume's
  datacenter (make sure your `--gpu` type is available there). The volume is
  caller-managed: create it in the RunPod console or via
  `POST /v1/networkvolumes` (`{"name", "size", "dataCenterId"}`), list yours
  with `RunPodApi.list_network_volumes()`, and **delete it when done — it
  bills monthly, not hourly**.
- `--resume`: Resume from the newest `checkpoint-<N>` directory already in
  `--output-dir` when one exists; with none, the job logs it and trains
  fresh. This helps where prior checkpoints are actually visible to the
  worker — rerunning an interrupted `--provider local` job, a rerun onto a
  RunPod network volume that kept earlier checkpoints
  (`--network-volume-id`), or a manual rerun on a machine that kept its disk.
  A fresh RunPod pod without a volume starts with an empty output dir, so
  `--resume` is a harmless no-op there (see `--cloud-type` above for what
  happens on pod death).
- `--eval-prompts PATH`: Local text file of eval entries, one per line (blank
  lines skipped). After training, each prompt is answered by both the base
  model and the tuned adapter (greedy decoding), and the side-by-side
  comparison is written to `output_dir/eval_results.json` as a list of
  `{"prompt", "base", "finetuned"}`. No effect on dry runs.

  **File format** — two kinds of line:
  - A plain line is a bare prompt (compare-only, exactly as before).
  - A line that parses as a **JSON object** is a prompt spec:

    ```json
    {"prompt": "What is your return window?", "expect": ["30 days"], "forbid": ["no returns"], "judge": "customer_support", "min_judge_score": 0.5}
    ```

    Only `prompt` is required. `expect`/`forbid` are substrings matched
    **case-insensitively against the finetuned completion only**; the row
    gains `"checks": {"expect_hits", "forbid_hits", "passed"}`. `judge`
    names a domain for
    `stateset_agents.rewards.multi_objective_reward.create_domain_reward`
    (e.g. `customer_support`) — if importable on the worker it scores the
    finetuned completion into `"judge_score"`; if not, the row just lacks
    the score (a warning is logged, never a crash). `min_judge_score`
    fails the row when a judge score exists below it.

    When **any** assertion fails, the job exits non-zero **after** the
    adapter and `eval_results.json` are saved — the artifacts survive, but
    `train-remote` reports the job as failed, so CI can gate on content.
    (JSON lines that aren't objects, e.g. a bare array, are treated as
    plain prompt text.)
- `--max-cost FLOAT`: Refuse to run if the pod could cost more than this
  many dollars (its full `--timeout` at the provider's quoted hourly rate).
  Checked before any work starts; an unpriceable pod is refused rather than
  rented.
- `--parent-adapter TEXT`: Adapter this run descends from, recorded in the
  trained adapter's manifest so improvement-loop generations stay linked.
- `--dry-run`: Print the training plan without training.

#### Providers

| Provider | Needs | Transport | Notes |
|---|---|---|---|
| `local` | a GPU on this machine | none | Verified end-to-end |
| `runpod` | `RUNPOD_API_KEY`, an SSH keypair, `ssh`/`scp` on PATH | SSH/SCP to a rented pod | **Verified end-to-end on live hardware** (RTX A4000 + Qwen3.5-0.8B, ~5 min; H100 80GB + Muse-Glimmer-30B multimodal, 63GB checkpoint, `container_disk_gb=160`). GPU names are RunPod's own (`"NVIDIA RTX A4000"`) |
| `modal` | `pip install "stateset-agents[modal]"`; Modal token; optional named Secrets | Per-job Modal Volume | Local datasets are uploaded before allocation, outputs are committed/downloaded separately, and the Volume is deleted; live transport certification pending |

RunPod creates the pod with TCP 22 exposed and your public key

Use `train-remote --plan-only` for a non-billable resource plan. It never
constructs the provider executor. For RunPod, known models receive catalog
GPU/count/disk defaults; estimated frontier or unknown-model plans require an
explicit `--max-cost` before execution. `--dry-run` is a worker-level training
dry run and can still provision remote hardware.

Modal requires SDK 1.1.2 or newer. Every executor-created `stateset-sft-*`
Volume is deleted after artifact retrieval, dry-run completion, or failure;
failure to confirm deletion makes the job fail rather than silently retaining
persistent storage. Set `STATESET_MODAL_SECRET_NAMES` to a comma-separated list
of existing Modal Secret names (for example a Secret containing `HF_TOKEN` or
`WANDB_API_KEY`); only names are sent by StateSet and Modal injects their values
inside the Function. `STATESET_MODAL_REGION` optionally constrains compute
placement. Modal's standard `MODAL_ENVIRONMENT` selects the isolated resource
environment.
(`~/.ssh/id_ed25519.pub` or `id_rsa.pub`) injected, copies the dataset in,
runs the job, copies the adapter back, and **terminates the pod on every exit
path** — including failures and timeouts — so nothing keeps billing. By
default no network volume is created, so there is no storage cost after the
run; with `--network-volume-id` the (caller-managed, monthly-billed) volume
persists until you delete it.

To test an unreleased change on real hardware, point the RunPod executor at a
locally built wheel instead of PyPI (the pinned version cannot resolve before
it is published):

```python
RunPodExecutor(wheel=Path("dist/stateset_agents-0.48.0-py3-none-any.whl"))
```

### `stateset-agents undeploy`

Delete a Fireworks deployment so it stops billing. Deployments are created by
`train-remote --provider fireworks --deploy`, and they bill for as long as they
exist — nothing tears one down implicitly.

```bash
stateset-agents undeploy --deployment accounts/my-org/deployments/dep-1
```

#### Options

- `--deployment TEXT`: Deployment name or bare id, as printed by
  `train-remote --deploy` (required).

Needs `FIREWORKS_API_KEY` and `FIREWORKS_ACCOUNT_ID`. See
`docs/FIREWORKS_PROVIDER.md`.

### `stateset-agents inference-deploy`

Deploy a complete model directory to CoreWeave Dedicated Inference or a
Nebius Serverless AI endpoint. This is deliberately separate from
`train-remote`: a LoRA adapter must be merged or materialized before it can be
used as standalone BYOW weights.

```bash
stateset-agents inference-deploy --provider coreweave \
  --name support-production --model-name support-model \
  --weights-uri s3://model-weights/support-model \
  --gpu gd-8xh100ib-i128 --runtime dynamo-vllm --zone US-WEST-04A
```

The JSON result is the durable cleanup handle. Preserve it, especially
`gateway_id` and `owns_gateway` for an automatically created CoreWeave
gateway. See `docs/COREWEAVE_PROVIDER.md` and `docs/NEBIUS_PROVIDER.md` for
provider configuration and secret requirements.

### `stateset-agents inference-status`

Read a managed inference resource without changing it:

```bash
stateset-agents inference-status --provider coreweave \
  --deployment-id <id> --model-name support-model
```

### `stateset-agents inference-delete`

Delete a managed deployment. For CoreWeave, add `--gateway-id <id>
--delete-gateway` only when the deployment handle says `owns_gateway: true`.
Omitting that flag preserves shared gateways.

### `stateset-agents qwen3-5-0-8b`

Preview or run the dedicated starter path for `Qwen/Qwen3.5-0.8B`.
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents qwen3-5-0-8b
stateset-agents qwen3-5-0-8b --json-output
stateset-agents qwen3-5-0-8b --starter-profile memory --json-output
stateset-agents qwen3-5-0-8b --list-profiles --json-output
stateset-agents qwen3-5-0-8b --write-config ./qwen3_5_0_8b.json
stateset-agents qwen3-5-0-8b --config ./qwen3_5_0_8b.json --no-dry-run
stateset-agents qwen3-5-0-8b --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Qwen starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`Qwen/Qwen3.5-0.8B-Base` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents kimi-k2-6`

Preview or run the dedicated starter path for `moonshotai/Kimi-K2.6`.
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents kimi-k2-6
stateset-agents kimi-k2-6 --json-output
stateset-agents kimi-k2-6 --starter-profile memory --json-output
stateset-agents kimi-k2-6 --list-profiles --json-output
stateset-agents kimi-k2-6 --write-config ./kimi_k2_6.json
stateset-agents kimi-k2-6 --config ./kimi_k2_6.json --no-dry-run
stateset-agents kimi-k2-6 --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Kimi starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`moonshotai/Kimi-K2.6` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents kimi-k3`

Preview or run the dedicated starter path for `moonshotai/Kimi-K3`.
The command defaults to a dry-run so you can inspect the resolved config before loading a model.
Note: the `moonshotai/Kimi-K3` ID is provisional — HF weights are not yet published (as of 2026-07-16).

```bash
stateset-agents kimi-k3
stateset-agents kimi-k3 --json-output
stateset-agents kimi-k3 --starter-profile memory --json-output
stateset-agents kimi-k3 --list-profiles --json-output
stateset-agents kimi-k3 --write-config ./kimi_k3.json
stateset-agents kimi-k3 --config ./kimi_k3.json --no-dry-run
stateset-agents kimi-k3 --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Kimi starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`moonshotai/Kimi-K3` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents muse-glimmer`

Preview or run the dedicated starter path for `meta-models/Muse-Glimmer-30B`,
Meta's open agentic model (Aug 2026; dense 30B, 131K ctx, Apache-2.0).
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents muse-glimmer
stateset-agents muse-glimmer --json-output
stateset-agents muse-glimmer --starter-profile memory --json-output
stateset-agents muse-glimmer --list-profiles --json-output
stateset-agents muse-glimmer --write-config ./muse_glimmer.json
stateset-agents muse-glimmer --config ./muse_glimmer.json --no-dry-run
stateset-agents muse-glimmer --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Muse Glimmer starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`meta-models/Muse-Glimmer-30B` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents nemotron-3-5`

Preview or run the dedicated starter path for `nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16`,
NVIDIA's hybrid Mamba-2/attention/MoE open model (Aug 2026; 30B total / ~3B active, 256K ctx, OpenMDW-1.1).
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents nemotron-3-5
stateset-agents nemotron-3-5 --json-output
stateset-agents nemotron-3-5 --starter-profile memory --json-output
stateset-agents nemotron-3-5 --list-profiles --json-output
stateset-agents nemotron-3-5 --write-config ./nemotron_3_5.json
stateset-agents nemotron-3-5 --config ./nemotron_3_5.json --no-dry-run
stateset-agents nemotron-3-5 --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Nemotron 3.5 starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-BF16` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents qwen3-8-27b`

Preview or run the dedicated starter path for `Qwen/Qwen3.8-27B`,
Alibaba's multimodal hybrid-attention open model (2026-08-05; 27.8B params, 256K ctx, Apache-2.0).
LoRA targets cover standard attention, Mamba-style linear attention, and the MLP; the vision tower is excluded.
This is a ~56GB BF16 checkpoint — budget ~160GB of disk and an 80GB card (or `--gpu-count 2`).
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents qwen3-8-27b
stateset-agents qwen3-8-27b --json-output
stateset-agents qwen3-8-27b --starter-profile memory --json-output
stateset-agents qwen3-8-27b --list-profiles --json-output
stateset-agents qwen3-8-27b --write-config ./qwen3_8_27b.json
stateset-agents qwen3-8-27b --config ./qwen3_8_27b.json --no-dry-run
stateset-agents qwen3-8-27b --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Qwen3.8 27B starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`Qwen/Qwen3.8-27B` recommended; `Qwen/Qwen3.8-27B-FP8` is inference-oriented).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents qwen3-coder`

Preview or run the dedicated starter path for `Qwen/Qwen3-Coder-30B-A3B-Instruct`,
Alibaba's open coding MoE model (30B total / ~3B active, 128 experts / 8 active, 256K ctx, Apache-2.0).
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents qwen3-coder
stateset-agents qwen3-coder --json-output
stateset-agents qwen3-coder --starter-profile memory --json-output
stateset-agents qwen3-coder --list-profiles --json-output
stateset-agents qwen3-coder --write-config ./qwen3_coder.json
stateset-agents qwen3-coder --config ./qwen3_coder.json --no-dry-run
stateset-agents qwen3-coder --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved Qwen3 Coder starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`Qwen/Qwen3-Coder-30B-A3B-Instruct` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents gpt-oss`

Preview or run the dedicated starter path for `openai/gpt-oss-20b`,
OpenAI's open-weight reasoning MoE model (32 experts / 4 active, 131K ctx, Apache-2.0).
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents gpt-oss
stateset-agents gpt-oss --json-output
stateset-agents gpt-oss --starter-profile memory --json-output
stateset-agents gpt-oss --list-profiles --json-output
stateset-agents gpt-oss --write-config ./gpt_oss.json
stateset-agents gpt-oss --config ./gpt_oss.json --no-dry-run
stateset-agents gpt-oss --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved gpt-oss starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`openai/gpt-oss-20b` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents deepseek-v4`

Preview or run the dedicated starter path for `deepseek-ai/DeepSeek-V4-Flash`,
DeepSeek's large MoE with MLA attention (256 routed experts / 6 active, up to 1M positions, MIT; QLoRA-only, vLLM generation).
The command defaults to a dry-run so you can inspect the resolved config before loading a model.

```bash
stateset-agents deepseek-v4
stateset-agents deepseek-v4 --json-output
stateset-agents deepseek-v4 --starter-profile memory --json-output
stateset-agents deepseek-v4 --list-profiles --json-output
stateset-agents deepseek-v4 --write-config ./deepseek_v4.json
stateset-agents deepseek-v4 --config ./deepseek_v4.json --no-dry-run
stateset-agents deepseek-v4 --no-dry-run --task customer_service --use-4bit
```

#### Options

- `--config PATH`: Load a saved deepseek-v4 starter config file (`json` or `yaml`).
- `--task TEXT`: Starter task preset (`customer_service`, `technical_support`, `sales`, `conversational`).
- `--starter-profile TEXT`: Starter profile (`balanced`, `memory`, `quality`).
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name (`deepseek-ai/DeepSeek-V4-Flash` recommended).
- `--use-lora / --no-lora`: Override LoRA for the run.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization.
- `--output-dir PATH`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count (must be > 0).
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config PATH`: Write the resolved starter config to `json`/`yaml` and exit.
- `--dry-run / --no-dry-run`: Preview or execute the starter workflow.
- `--json-output`: Emit a machine-readable preview/result payload.

### `stateset-agents validate-config`

Validate a training config without running training.

```bash
stateset-agents validate-config --config ./stateset_agents.json
stateset-agents validate-config --config ./stateset_agents.yaml --strict --json-output
stateset-agents validate-config --config ./stateset_agents.yaml --fail-on-warnings
```

Options:

- `--config PATH`: YAML or JSON config path.
- `--strict`: Exit non-zero when validation errors are found.
- `--fail-on-warnings`: Exit non-zero when validation warnings are found.
- `--json-output`: Emit machine-readable result with `valid`, `errors`, and `warnings`.

### `stateset-agents serve`

Run the API gateway (`stateset_agents.api.main`) with Uvicorn.

```bash
stateset-agents serve
stateset-agents serve --host 0.0.0.0 --port 8000 --reload
stateset-agents serve --dry-run
```

#### Options

- `--host TEXT`: Bind host.
- `--port INTEGER`: Bind port.
- `--reload`: Enable auto-reload (development).
- `--dry-run`: Preview startup command without launching the server.

### `stateset-agents doctor`

Check common runtime dependencies.

```bash
stateset-agents doctor
stateset-agents doctor --strict
stateset-agents doctor --json-output
stateset-agents doctor --strict --json-output
```

`--strict` exits with non-zero status if required dependencies are missing.
`--json-output` writes a JSON payload with `required_dependencies` and `optional_dependencies`.

### `stateset-agents evaluate`

Run a single message through a checkpointed agent.

```bash
stateset-agents evaluate --checkpoint ./checkpoints/agent --message "Hello"
stateset-agents evaluate --dry-run --message "Hello"
```

### `stateset-agents init`

Generate a starter config (`yaml` default, `json` optional).

```bash
stateset-agents init
stateset-agents init --path ./stateset_agents.yaml --format json
stateset-agents init --path ./stateset_agents.yaml --overwrite --format yaml
stateset-agents init --preset qwen3-5-0-8b --path ./qwen3_5_0_8b.json --format json
stateset-agents init --preset qwen3-5-0-8b --starter-profile memory --path ./qwen3_5_0_8b_memory.json --format json
```

Options:

- `--path PATH`: Output config path.
- `--overwrite`: Replace an existing file.
- `--format [yaml|json]`: Output file format.
- `--preset [default|qwen3-5-0-8b|kimi-k2-6|kimi-k3|gemma-4-31b|muse-glimmer|nemotron-3-5|qwen3.8-27b|qwen3-coder|gpt-oss|deepseek-v4]`: Starter config preset.
- `--task TEXT`: Task preset for model-specific starter configs.
- `--starter-profile TEXT`: Starter profile for model-specific starter configs.

Aliases:

- `stateset-agents init-config` is equivalent to `stateset-agents init`.

### `stateset-agents advanced`

Experimental command bundle for advanced workflows:

- `debug`
- `profile`
- `validate`
- `progress`
- `tree`

This group is loaded only when optional advanced CLI dependencies are available.

### `stateset-agents preflight`

Run dependency and (optional) config checks together.

```bash
stateset-agents preflight
stateset-agents preflight --config ./stateset_agents.yaml
stateset-agents preflight --config ./stateset_agents.json --strict --json-output
```

Options:

- `--config PATH`: Validate this config as part of the preflight.
- `--strict`: Fail on missing required dependencies.
- `--fail-on-warnings`: Fail when validation warnings are present.
- `--json-output`: Return JSON payload for automation.

### `stateset-agents publish-check`

Run a preflight check plus import smoke checks before publishing.

```bash
stateset-agents publish-check
stateset-agents publish-check --config ./stateset_agents.yaml
stateset-agents publish-check --config ./stateset_agents.yaml --strict --json-output
stateset-agents publish-check --config ./stateset_agents.yaml --fail-on-warnings --json-output
```

Options:

- `--config PATH`: Validate this config as part of publish checks.
- `--strict`: Fail when required dependencies or required imports are missing.
- `--fail-on-warnings`: Fail when validation warnings are present.
- `--json-output`: Return JSON payload with dependency/import/config status.

### `stateset-agents chat`

Open an interactive REPL against an in-process agent.

```bash
stateset-agents chat
```

#### Options

- `--model, -m TEXT`: HF model name or stub://<id> for the in-process REPL.
- `--checkpoint, -c TEXT`: Path to a LoRA adapter to load on top of --model.
- `--system TEXT`: Optional system prompt prepended to every conversation.
- `--max-new-tokens INTEGER`: Generation length cap per response.
- `--history TEXT`: Path to a JSONL file to APPEND each turn (one JSON object per line). Capture interesting conversations to replay or grade later with `make grade-transcript`.
- `--replay TEXT`: Path to a JSONL transcript to replay as initial conversation context. Useful for resuming a debugging session.
- `--grade TEXT`: Score each assistant turn live with the named reward function. Options: gsm8k, customer_support, tool_calling. Mismatches between intuition and score surface reward-function bugs.

### `stateset-agents chat-remote`

Chat with a fine-tuned model on a rented RunPod GPU, ephemerally. Rents a
pod, installs the package, loads the base model plus your local LoRA adapter
there, and opens a REPL over SSH. **The pod is terminated when the session
ends** — including on errors and Ctrl+C — so there are no open ports and no
idle billing. Type `exit`/`quit` or Ctrl+D/Ctrl+C to leave.

```bash
export RUNPOD_API_KEY=...

# Interactive chat with a fine-tune too big for the local machine
stateset-agents chat-remote --base-model meta-models/Muse-Glimmer-30B \
    --adapter outputs/sft_v1

# Scripted spot-check: send prompts, print replies, exit (and kill the pod)
stateset-agents chat-remote --base-model Qwen/Qwen3.5-0.8B \
    --adapter outputs/sft_v1 \
    --prompt "what's the return policy?" --prompt "and for sale items?"
```

The remote side is `stateset_agents.remote.chat_repl`: a JSON-lines chat
server on the pod that keeps the full multi-turn history and generates
greedily through the model's chat template (the same decoding as the
`train-remote --eval-prompts` comparison). The transport is one persistent
SSH channel — no port beyond RunPod's own SSH one is exposed.

Requirements match `train-remote --provider runpod`: `RUNPOD_API_KEY`, an SSH
keypair (`~/.ssh/id_ed25519.pub` or `id_rsa.pub`), and `ssh`/`scp` on PATH.

**Every chat is training data.** On exit — including Ctrl+C and errors — the
conversation is saved by default to `./chat_transcripts/chat_<timestamp>.jsonl`
in OpenAI chat format, ready for the improvement flywheel:

```bash
stateset-agents ingest --format openai --input chat_transcripts/chat_<ts>.jsonl \
    --output graded.jsonl
# then: improve -> train-remote -> chat-remote again, on the better adapter
```

Disable with `--no-save`, or pick the destination with `--save-transcript`.

#### Options

- `--base-model TEXT`: Hugging Face base model (required).
- `--adapter PATH`: Local LoRA adapter directory (e.g. `outputs/sft_v1`),
  uploaded to the pod for the session.
- `--gpu TEXT`: RunPod GPU type to rent. Default `NVIDIA H100 80GB HBM3`.
- `--container-disk-gb INTEGER`: Container disk for the model download, in
  GB. Size it at roughly 2.5x the checkpoint. Default `160`.
- `--max-turns INTEGER`: Safety cap on interactive turns — the pod bills
  while you type. Default `50`.
- `--prompt TEXT`: Non-interactive mode; repeatable. Sends each prompt in
  order, prints each reply, and exits.
- `--save-transcript PATH`: Where to save the conversation transcript
  (OpenAI chat-format JSONL). Default
  `./chat_transcripts/chat_<timestamp>.jsonl`.
- `--save / --no-save`: Save the transcript on exit. Default on; only
  successfully answered turns are recorded, and empty sessions write
  nothing.

### `stateset-agents serve-remote`

Serve a model as a **persistent** OpenAI-compatible endpoint on a rented
RunPod GPU. Rents a pod, installs vLLM, loads the base model (plus your
local LoRA adapter, served under the model name `adapter`), and prints the
endpoint URL and a generated Bearer token. Unlike `chat-remote`, the pod
**keeps running after the command exits** — that is the point — so cost is
controlled three ways:

1. **On-pod self-destruct** (`--max-hours`, default `1.0`): a `nohup`-ed
   script on the pod sleeps for that long and then calls the RunPod DELETE
   endpoint on its own pod id. It fires even if your laptop is gone.
   **Tradeoff:** to make that possible, your `RUNPOD_API_KEY` is copied to
   the pod (`chmod 600`, root-only). Use a dedicated, revocable key if that
   matters to you.
2. `--stop <name-or-id>`: terminate a serve pod immediately.
3. `--list`: show running serve pods with status, age, and $/hr.

On any *startup* failure the pod is terminated before the error is shown.

```bash
export RUNPOD_API_KEY=...

# Serve a small model for an hour (the default cap)
stateset-agents serve-remote --base-model Qwen/Qwen3.5-0.8B

# Serve base + fine-tuned adapter for a demo afternoon
stateset-agents serve-remote --base-model Qwen/Qwen3.5-0.8B \
    --adapter outputs/sft_v1 --max-hours 4

# Call it (the command prints this, filled in)
curl http://<ip>:<port>/v1/chat/completions \
    -H "Authorization: Bearer <token>" -H "Content-Type: application/json" \
    -d '{"model": "adapter", "messages": [{"role": "user", "content": "Hi"}]}'

# Manage what's running
stateset-agents serve-remote --list
stateset-agents serve-remote --stop stateset-serve-<id>
```

The endpoint is vLLM's own OpenAI-compatible server (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`), launched with `--api-key` so every request
must carry the printed Bearer token. The URL comes from RunPod's public port
mapping for the pod's port 8000. Requirements match
`train-remote --provider runpod`: `RUNPOD_API_KEY`, an SSH keypair, and
`ssh`/`scp` on PATH.

**VRAM note:** vLLM loads the whole model into GPU memory. The default GPU
(`NVIDIA RTX A4000`, 16 GB) fits models up to ~7B at fp16; for bigger models
pick a bigger `--gpu` (e.g. `"NVIDIA H100 80GB HBM3"`) and raise
`--container-disk-gb` to ~2.5x the checkpoint for the download.

#### Options

- `--base-model TEXT`: Hugging Face base model to serve. Required unless
  `--stop` or `--list` is given.
- `--adapter PATH`: Local LoRA adapter directory to serve on top of the
  base model, as served-model name `adapter`.
- `--gpu TEXT`: RunPod GPU type. Default `NVIDIA RTX A4000` (16 GB).
- `--container-disk-gb INTEGER`: Container disk in GB — must fit the vLLM
  install (~10 GB) plus ~2.5x the model checkpoint. Default `60`.
- `--max-hours FLOAT`: On-pod self-destruct deadline in hours. Default
  `1.0`; must be positive.
- `--merge`: Fold the (single) adapter into full base weights on the pod
  and serve the merged checkpoint. REQUIRED for hybrid models
  (Qwen3.5/3.8 families) — vLLM loads their LoRA adapters without error
  and silently serves the base weights. Still answers as model
  `adapter`; needs disk for a second full copy of the weights.
- `--strict`: Fail (terminating the pod) if the adapter-effect probe
  finds an adapter's greedy completion byte-identical to the base
  model's; without it the probe warns loudly. Every adapter-serving
  start now probes its own effect; `--merge` verifies pre-vs-post-merge
  on the pod and writes `merge_probe.json`.
- `--stop TEXT`: Terminate a running serve pod by name or id, then exit.
- `--list`: List running serve pods (name, id, status, age, $/hr), then
  exit.

### `stateset-agents deploy`

Fine-tune on a rented GPU, then serve the result — one command.
`train-remote` then `serve-remote`, glued: rent, train, give the hardware
back, rent again, serve the fresh adapter as an authenticated
OpenAI-compatible endpoint, and print the URL and Bearer token. A failed
training job refuses to serve. This is the zero-to-API story of
`docs/GETTING_STARTED_API.md` as a single invocation.

```bash
stateset-agents deploy --dataset improved/curated.jsonl \
  --base-model Qwen/Qwen3.5-0.8B --max-cost 5
```

- `--dataset PATH` (required): Chat-format JSONL to train on.
- `--base-model TEXT` (required): Hugging Face base model.
- `--output-dir PATH`: Where the trained adapter is written locally
  (default `outputs/deploy_v1`).
- `--gpu TEXT`: RunPod GPU used for BOTH the training job and the endpoint
  (default `NVIDIA H100 80GB HBM3`).
- `--container-disk-gb INTEGER`: ~2.5x the checkpoint size.
- `--num-epochs INTEGER`: Training epochs (default 3).
- `--max-cost FLOAT`: Ceiling for the TRAINING job.
- `--max-hours FLOAT`: Endpoint self-destruct, armed on the serving pod
  (default 1.0).

### `stateset-agents remote-job`

Reconnect to a durable asynchronous provider job after the submitting CLI has
exited. Fireworks currently supports durable reconnects.

```bash
stateset-agents remote-job --provider fireworks --job-id JOB_ID --wait
```

- `--job-id TEXT` (required): Provider-owned job identifier.
- `--provider TEXT`: Provider that owns the job (default `fireworks`).
- `--wait`: Poll to a terminal state and fetch artifacts.
- `--fetch`: Fetch artifacts immediately; the job must already be complete.
- `--output-dir PATH`: Override the persisted artifact destination.

### `stateset-agents remote-providers`

List each remote executor's capabilities without loading provider SDKs or
requiring credentials.

```bash
stateset-agents remote-providers --json
```

- `--json`, `--json-output`: Emit machine-readable JSON.

### `stateset-agents runpod-orphans`

Inspect locally recorded cleanup leases for RunPod training pods whose
submitting process exited before cleanup. The default is read-only.

```bash
stateset-agents runpod-orphans
stateset-agents runpod-orphans --terminate
```

- `--terminate`: Terminate every leased pod and remove a lease only after the
  provider confirms deletion. Review the read-only output first.

### `stateset-agents model-support`

Show the auditable verification level for each model/provider claim. The
output distinguishes unit-tested framework registration, a live hardware
attempt, and successful inference; failed attempts are never promoted to
inference verification.

```bash
stateset-agents model-support
stateset-agents model-support --json
```

- `--json`, `--json-output`: Emit schema-versioned machine-readable evidence.

### `stateset-agents flywheel`

The improvement loop, unattended: harvest the current generation's rare
successes (best-of-N rejection sampling against objective `expect`/`forbid`
checks), train the next generation on nothing but those, measure it, and
repeat — stopping on plateau, a dry harvest (no signal to train on), a
perfect score, or when the next rental's worst case would break
`--max-cost`. Every generation leaves its harvest set, its adapter with a
lineage manifest, and `flywheel_report.json` with pass rates and dollars.
The methodology is `docs/FLYWHEEL_HEADROOM.md` (2/12 → 10/12 for $3.32).

```bash
stateset-agents flywheel --base-model meta-models/Muse-Glimmer-30B \
  --harvest-prompts harvest.json --eval-prompts eval.json \
  --gpu "NVIDIA H100 80GB HBM3" --container-disk-gb 170 --max-cost 20
```

- `--base-model TEXT` (required): Base model every generation is LoRA-tuned
  from.
- `--harvest-prompts PATH` (required): JSON list of
  `{prompt, expect, forbid}` specs sampled during harvest. The checks
  define success; they are mandatory.
- `--eval-prompts PATH` (required): JSON list of specs that score each
  generation. Keep disjoint from the harvest prompts.
- `--output-root PATH`: Where generations and the report land (default
  `outputs/flywheel`).
- `--initial-adapter PATH`: Existing adapter to start from (defaults to the
  bare base model).
- `--generations INTEGER`: Maximum NEW generations to train (default 3).
- `--best-of INTEGER`: Samples per harvest prompt (default 8).
- `--temperature FLOAT`: Harvest sampling temperature (default 0.9).
- `--max-cost FLOAT`: Hard dollar ceiling for the WHOLE run.
- `--provider TEXT`: Executor to run on (default `runpod`).
- `--gpu TEXT`: GPU type, in the provider's own vocabulary.
- `--container-disk-gb INTEGER`: Container disk per pod.
- `--num-epochs INTEGER`: Training epochs per generation (default 3).
- `--repeats INTEGER`: Run the whole loop N times and report the score
  distribution (min/mean/max); the budget is shared across repeats
  (default 1).
- `--dry-run`: Print each job's plan without renting anything.

### `stateset-agents fine-tune`

Fine-tune from a curated JSONL in one command.

```bash
stateset-agents fine-tune CURATED
```

#### Options

- `--base-model, -m TEXT`: HF base model to fine-tune.
- `--output-dir, -o TEXT`: Where the LoRA adapter is saved.
- `--min-score FLOAT`: Drop curated examples below this score before SFT.
- `--num-epochs, -e INTEGER`: Training epochs.
- `--lora-r INTEGER`: LoRA rank.
- `--dry-run`: Print the training plan without running it (forced when no GPU).

### `stateset-agents improve`

Run the grade -> curate -> retrain loop as a single command.

```bash
stateset-agents improve ACTION
```

#### Options

- `--transcripts TEXT`: For --format transcripts: a directory of transcript JSONL files (one conversation per file, {'role','content'} per line — the shape `stateset-agents chat --history` writes). For --format openai/langchain: the single source log file to ingest first.
- `--reward TEXT`: Reward function: gsm8k, customer_support, or tool_calling (rule-based, no API key required).
- `--output, -o TEXT`: Output directory for curated data + reports.
- `--threshold FLOAT`: Minimum score for curation (default: 0.7).
- `--format, -f TEXT`: Input format: 'transcripts' (already chat-history JSONL), 'openai', or 'langchain' (ingested first via stateset_agents.data.trajectory_ingest).

### `stateset-agents ingest`

Convert third-party conversation logs into graded-history JSONL.

```bash
stateset-agents ingest
```

#### Options

- `--format, -f TEXT`: Source log format: 'openai' (chat-completions messages JSONL) or 'langchain' (LangChain/LangGraph message-dump JSON).
- `--input, -i TEXT`: Path to the source log file. For --format openai: JSONL, one conversation per line ({'messages': [...]} or a bare message list). For --format langchain: a single JSON file (see stateset_agents.data.trajectory_ingest docstring for supported shapes).
- `--output, -o TEXT`: Output path. If it ends in .jsonl, all conversations are concatenated into one graded-history JSONL file (turns from different conversations are separated by a blank line — note the grader treats such a file as ONE transcript; use directory mode to grade conversations separately). Otherwise it is treated as a directory and one <output>/conversation_<N>.jsonl file is written per conversation — feed any of them to `python scripts/grade_transcript.py --history <file>`.

### `stateset-agents mcp`

Run the StateSet Agents MCP server (stdio transport by default).

```bash
stateset-agents mcp
```

#### Options

- `--transport TEXT`: MCP transport to serve over (default: stdio).

### `stateset-agents auto-research`

Run the autonomous research loop to optimize agent training.

```bash
stateset-agents auto-research
```

#### Options

- `--config, -c TEXT`: Path to auto-research config file (YAML/JSON).
- `--max-experiments, -n INTEGER`: Maximum experiments to run (0 = unlimited).
- `--time-budget, -t INTEGER`: Wall-clock seconds per experiment.
- `--proposer, -p TEXT`: Proposer strategy: perturbation, smart, adaptive, random, grid, bayesian, llm.
- `--algorithm, -a TEXT`: Training algorithm: gspo, grpo, dapo, vapo.
- `--output-dir, -o TEXT`: Directory for results and checkpoints.
- `--search-space, -s TEXT`: Search space: grpo, auto_research, quick, reward, model, multi_algorithm, full.
- `--improvement-patience INTEGER`: Stop after this many consecutive non-improvements (0 = disabled).
- `--max-wall-clock INTEGER`: Total wall-clock budget in seconds (0 = unlimited).
- `--wandb`: Log experiments to Weights & Biases.
- `--wandb-project TEXT`: W&B project name.
- `--stub`: Run with stub model for testing the loop without GPU.
- `--dry-run`: Validate config and show plan without running.

### `stateset-agents benchmark`

Run and aggregate Phase 0 / whitepaper-v1 benchmarks.

Subcommands:

- `aggregate`: Aggregate all *.json results in a directory into summary.md + summary.csv.
- `phase0`: Run a single Phase 0 benchmark and emit a schema-compliant JSON result.
- `plot`: Generate publication figures from aggregated benchmark results.
- `smoke`: Quick end-to-end smoke test of the GSM8K benchmark pipeline (no training).

```bash
stateset-agents benchmark --help
```

### `stateset-agents recipe`

Open a cookbook recipe in $PAGER, or `list` them all.

```bash
stateset-agents recipe NAME
```

### `stateset-agents starter`

Scaffold a fork-and-go fine-tuning project.

```bash
stateset-agents starter TEMPLATE OUTPUT
```

#### Options

- `--name, -n TEXT`: Project name (defaults to the basename of the output directory).
- `--force, -f`: Overwrite an existing non-empty directory.
- `--client-name TEXT`: Client name (slugified) — patches output_dir paths and the W&B project name throughout the scaffold.

### `stateset-agents tour`

Open the platform tour — the one document that walks the full developer journey.

```bash
stateset-agents tour
```

### `stateset-agents init-config`

Alias for `init`.

```bash
stateset-agents init-config
```

#### Options

- `--path TEXT`: Path for a starter config
- `--overwrite`: Overwrite existing file
- `--format, -f TEXT`: Output format: yaml or json
- `--preset TEXT`: Starter preset: default, qwen3-5-0-8b, kimi-k2-6, kimi-k3, gemma-4-31b, muse-glimmer, or nemotron-3-5
- `--task TEXT`: Task preset for model-specific starter presets.
- `--starter-profile TEXT`: Starter profile for model-specific starter presets.

### `stateset-agents gemma-4-31b`

Preview or run the dedicated Gemma 4 31B GSPO starter path.

```bash
stateset-agents gemma-4-31b
```

#### Options

- `--config, -c TEXT`: Path to a Gemma 4 31B starter config file (JSON/YAML).
- `--task TEXT`: Task preset for the Gemma 4 31B starter path.
- `--starter-profile TEXT`: Starter profile: balanced, memory, or quality.
- `--list-profiles`: Describe all built-in starter profiles and exit.
- `--model TEXT`: Model name. For post-training, use google/gemma-4-31B-it.
- `--use-lora / --no-lora`: Override LoRA usage. Defaults come from --starter-profile.
- `--use-4bit / --no-use-4bit`: Override 4-bit quantization. Defaults come from --starter-profile.
- `--use-8bit / --no-use-8bit`: Override 8-bit quantization. Defaults come from --starter-profile.
- `--output-dir TEXT`: Override the output directory for checkpoints and adapters.
- `--iterations INTEGER`: Override the outer GSPO iteration count for the starter run.
- `--wandb`: Enable Weights & Biases logging.
- `--wandb-project TEXT`: Optional W&B project name.
- `--write-config TEXT`: Write the resolved Gemma starter config to JSON/YAML and exit.
- `--dry-run / --no-dry-run`: Preview the resolved config instead of loading a model.
- `--json, --json-output`: Output machine-readable JSON.

### `stateset-agents costs`

Show what remote runs have actually cost. Every `train-remote`, `chat-remote`,
and `serve-remote` pod appends one line to a per-user cost ledger — what it
ran, on what hardware, for how long, and the dollar amount — derived from the
provider's quoted hourly rate and the measured pod lifetime.

```bash
stateset-agents costs
stateset-agents costs --limit 25
stateset-agents costs --json
```

#### Options

- `--ledger PATH`: Read a specific ledger file (default: the shared per-user ledger).
- `--limit INTEGER`: How many recent runs to list. Default `10`.
- `--json` / `--json-output`: Emit machine-readable JSON (summary + recent runs).

The ledger is advisory bookkeeping, not a bill: it counts each pod from
creation (when billing starts) rather than from job start, so it rounds
against you rather than in your favour. A run whose provider quoted no price
is recorded with an unknown cost — never as free.

### `stateset-agents adapters`

List trained adapters with their provenance and lineage. Every training run
writes `stateset_manifest.json` beside its adapter — base model, dataset path
and content hash, hyperparameters, eval outcome, and the adapter it descends
from — so an adapter directory is never anonymous tensors.

```bash
stateset-agents adapters
stateset-agents adapters --dir outputs
stateset-agents adapters --json
```

#### Options

- `--dir PATH` / `-d`: Directory to scan. Default `outputs`.
- `--json` / `--json-output`: Emit machine-readable JSON (adapters + lineage map).

Adapters trained before manifests existed still appear, marked as carrying no
provenance — an audit that hid them would be worthless. Lineage links resolve
by recorded path first and directory name second, so a manifest written on a
rented pod still links up after the adapter is fetched somewhere else.

Record a generation link by passing `--parent-adapter` to `train-remote`.

## Exit behavior

- Non-zero exit codes indicate command failures (e.g., missing modules or invalid input).
- Use `--dry-run` modes to inspect intended behavior before running heavy operations.

## Troubleshooting

- If `version`/`doctor`/`serve` fail to import modules, install optional extras as needed:
  - API extras for serving (`fastapi`, `uvicorn`)
  - Rich tooling for `advanced` workflows (`rich`, `ipython`)
- If config loading fails, check file path, extension, and YAML/JSON syntax.
