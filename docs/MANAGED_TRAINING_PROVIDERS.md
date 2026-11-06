# Managed training providers

StateSet Agents supports four additional managed substrates without flattening
their different execution models into misleading abstractions. All integrations
below are code-complete and unit-tested; none is marked live-certified until a
retained provider run passes the repository's evidence gates.

## Tinker and Inkling 2 Small

Tinker is a remote-autograd service. StateSet keeps the training loop locally,
builds assistant-masked causal-LM `Datum` batches, calls `forward_backward` and
`optim_step`, then saves both sampler weights and resumable training state.

```bash
pip install 'stateset-agents[tinker]'
export TINKER_API_KEY=...
stateset-agents train-remote \
  --provider tinker \
  --dataset data/train.jsonl \
  --base-model thinkingmachines/Inkling-2-Small \
  --output-dir outputs/inkling-2-small
```

Inkling 2 Small uses the official Tinker ID
`thinkingmachines/Inkling-2-Small` and native `tml-renderers.Inkling2` SFT
rendering, including assistant loss masks and end-of-turn boundaries. The
renderer requires Python 3.11+ and PyTorch 2.10+ on a supported Linux or macOS
platform; these optional dependencies are included in the `tinker` extra on
Python 3.11+. Text conversations and tool-call messages are supported, including
top-level OpenAI `tools` declarations. Image/audio training is not implemented
in this adapter. The earlier `thinkingmachines/Inkling-Small` ID remains usable.
See the official [Inkling guide](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/)
and [renderer documentation](https://tinker-docs.thinkingmachines.ai/cookbook/inkling/tml-renderers/).
StateSet lists Inkling 2 Small as `configured` / `frontier-preview`. Offline
integration checks with Tinker 0.38.0 and tml-renderers 0.4.0 verify assistant
and tool-result masking, multi-turn conversations, real SDK `Datum` creation,
and checkpoint handoff with RPC doubles. A live provider training run is still
pending. No local weights or RunPod hardware sizing are assumed.

Run the offline SDK checks after installing the `tinker` and `dev` extras:

```bash
pytest -q tests/integration/test_inkling2_sdk.py
```

These tests skip when the optional SDK or renderer is absent. They do not need
an API key or call Tinker's hosted service. Empty rendered datasets fail before
a remote training client is created.

The output directory contains `tinker_checkpoint.json` with `tinker://`
sampler and state URIs. Tinker selects the hardware; GPU flags do not provision
a machine. `--job-kind rl` accepts token-aligned JSONL rows containing
`input_ids`, `target_tokens`, rollout `logprobs`, and `advantages`, and uses
Tinker's `importance_sampling` loss. This explicit contract prevents stale or
invented on-policy probabilities from entering the update.

## Prime Intellect Lab, verifiers, and OpenEnv

Prime Lab consumes its native TOML training configuration. StateSet renders
that configuration and runs the documented `prime train run` workflow. Supply
the verifiers/OpenEnv environment through provider options:

```bash
uv tool install prime
prime login
stateset-agents train-remote \
  --provider prime --job-kind rl \
  --dataset data/prompts.jsonl \
  --base-model Qwen/Qwen3.5-9B \
  --provider-options-json '{"environment":"stateset/support","max_steps":100,"rollouts_per_example":8}'
```

Optional keys are `harness`, `runtime`, `max_tokens`, and `temperature`. The
result contains `prime_training_run.json`; adapters remain governed by the
Prime run and dashboard lifecycle.

## Hugging Face Jobs and Inference Endpoints

Jobs mount the local dataset through a Hub bucket and mount a writable output
path. Set a bucket before submitting:

```bash
pip install 'stateset-agents[huggingface]'
export HF_TOKEN=...
export HF_JOBS_BUCKET=my-org/stateset-jobs
export HF_JOBS_NAMESPACE=my-org       # optional
stateset-agents train-remote --provider huggingface \
  --dataset data/train.jsonl --base-model Qwen/Qwen3.5-0.8B
```

`huggingface_job.json` records the durable job URL and bucket artifact URI.
Dedicated endpoints use the existing provider-neutral commands:

```bash
stateset-agents inference-deploy --provider huggingface \
  --name support --model-name support \
  --weights-uri my-org/support-model --gpu nvidia-a10g
```

## Together AI

Together accepts chat JSONL for managed LoRA supervised fine-tuning:

```bash
pip install 'stateset-agents[together]'
export TOGETHER_API_KEY=...
stateset-agents train-remote --provider together \
  --dataset data/train.jsonl --base-model <together-finetunable-model>
```

The durable job can be re-polled with `stateset-agents remote-job`; successful
fetches write `together_checkpoint.json` with the hosted tuned-model identity.

## Read-only credential checks

These probes authenticate but do not create billable resources:

```bash
stateset-agents provider-canary --provider tinker
stateset-agents provider-canary --provider huggingface
stateset-agents provider-canary --provider together
```

Prime authentication is owned by `prime login`; use `prime whoami` before a
live run. The StateSet Prime adapter intentionally does not scrape or persist
CLI credentials.
