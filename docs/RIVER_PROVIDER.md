# River AI provider (`train-remote --provider river`)

> **Python version**: `river-client` requires Python ≥3.12 while this
> repo supports 3.10+. For River runs, create a side venv:
> `uv venv --python 3.12 .river-venv && uv pip install --python .river-venv/bin/python river-client -e .`
> then run `.river-venv/bin/python -m stateset_agents.cli train-remote --provider river …`.
> Verified live 2026-08-18 (training + checkpoint sampling).

> **Live verification:** on 2026-08-18 this executor trained Qwen3.5-9B and
> sampling the saved `river://` checkpoint reproduced the target behaviour on
> 3/3 held-out prompts. Offline behavioural goldens still pin every submit
> mode, and `provider-canary.yml` now provides the recurring authentication,
> health, and capability check once the repository secret is configured.

## What River is

River is a **remote autograd and optimizer service**. Unlike Modal or RunPod —
which rent you a machine and let us ship `stateset_agents.training.sft` to it —
River never gives you a machine. It gives you gradients:

```python
import river_client as river

client  = river.Client(api_key="rv_...")
session = client.create_session()
model   = session.create_model(
    base_model="Qwen/Qwen3.5-9B",
    lora=river.LoraConfig(rank=16, train_attn=True, train_mlp=True,
                          train_unembed=False),
)

for epoch in range(num_epochs):
    for batch in batches:
        model.forward_backward(batch, loss_fn="cross_entropy")
        model.optim_step(lr=2e-5)

checkpoint = model.save_weights("my-adapter", mode="inference")  # -> river://...
```

**You own the training loop.** That is the whole shape of this provider: the
executor is not a job submitter, it *is* the loop, with the tensor math
happening at `api.river.ai`.

Other documented surface: `client.sample(prompt, base_model=..., max_tokens=...)`
and `session.sample(..., checkpoint=...)` for inference,
`client.get_capabilities()` for the models your account may use, and
`client.health_check()`. Calls can be run asynchronously — submit returns a
`request_id` you poll with `.result()` — but the high-level helpers this
integration uses poll internally.

Supported loss functions: `cross_entropy` (SFT), and `importance_sampling`,
`ppo`, `cispo`, `dro` for RL.

## What this integration does

- `stateset_agents/remote/river_batches.py` — pure batch construction. No SDK
  import, no network, no state. Converts our chat-format JSONL rows into
  River's SFT data, and our trajectories into River's RL data. This is the part
  that has to be right, so it is the part that is isolated and heavily tested.
- `stateset_agents/remote/river.py` — `RiverExecutor`, implementing the same
  `submit` / `status` / `logs` / `fetch` / `cancel` contract as `local`,
  `modal`, and `runpod`, and inheriting `wait`.
- Registered in the provider registry, so `--provider river` resolves and
  `stateset-agents train-remote --provider river ...` runs.

### Batch formats

River takes token ids, not text — so *we* tokenize. One datum per row:

| Task | Keys |
| --- | --- |
| SFT | `input_ids`, `target_tokens`, `weights` |
| RL  | `input_ids`, `old_logprobs`, `advantages`, `attention_mask` |

`weights` (SFT) and `advantages` (RL) scale the per-token loss; `0.0` excludes
a token entirely. We use that to mask the prompt: **system/user/tool tokens and
the generation prefix get weight `0.0`; assistant tokens get `1.0`**, so loss is
computed only on what the model should say. In a multi-turn conversation
*every* assistant turn is weighted, not just the last.

## What this integration does **not** do

- **It does not download weights.** `save_weights` returns a `river://` URI and
  the trained LoRA stays on River's servers. `fetch()` therefore writes a
  *pointer*, `river_checkpoint.json` (checkpoint URI, base model, LoRA config,
  step/loss summary), plus the standard `stateset_manifest.json` so provenance,
  `stateset-agents adapters`, and lineage work identically to a local adapter.
  It does **not** fabricate `adapter_model.safetensors`. Consequently
  `stateset-agents serve --checkpoint <dir>` **cannot load a River result** —
  sample it through River's API instead.
- **It ignores machine-shaped options.** `--gpu`, `--gpu-count`,
  `--container-disk-gb`, `--cloud-type`, and `--network-volume-id` describe
  rented hardware, and River exposes none. They are logged as ignored rather
  than raising, so one spec remains submittable to any provider.
- **It does not price the run.** River bills per token and the SDK publishes no
  price to us, so the cost ledger records `cost_usd: null` — *unknown*, never
  `0`. A zero would silently under-report `stateset-agents costs` and let any
  `--max-cost` check pass. Token counts are recorded when the response exposes
  them, so spend can be reconstructed from River's price list.
- **It does not drive an RL loop end to end.** `build_rl_batch` exists and is
  tested, but no CLI command wires it to `forward_backward(loss_fn="ppo")` yet.
- **It does not call `get_capabilities()` or `health_check()`.** Both would be
  useful preflight checks; neither can be written responsibly without a key to
  check the response shape against.

## Unverified assumptions

### 1. The causal shift (the big one)

River's SFT datum is documented as `{input_ids, target_tokens, weights}`
without stating who performs the next-token shift. **We assume the caller
does.** For a tokenized conversation `t[0..n-1]` we emit:

```
input_ids     = t[0 : n-1]
target_tokens = t[1 : n]
weights       = w[1 : n]     # weight of the TARGET, not of the input
```

All three lists have length `n-1` and are index-aligned.

*If this is wrong* (River shifts internally), the symptom is unmistakable on
the first real run: loss stays high and flat, and sampled continuations look
off-by-one — the model reproducing the token it was just given. The fix is one
line, in `_shift_for_causal_lm()` in `river_batches.py`, which is the only
place the assumption is implemented.

### 2. Prefix-stable tokenization

Assistant spans are located by tokenizing successively longer chat-template
renderings and diffing their lengths. That is exact for BPE/SentencePiece in
practice, but a tokenizer that re-segments across a message boundary could
shift a span by a token or two. The ids stay correct; only the mask edges
would be slightly off.

### 3. Call names and signatures

`create_session()`, `create_model(base_model=, lora=)`,
`forward_backward(batch, loss_fn=)`, `optim_step(lr=)`,
`save_weights(name, mode=)`, and `river.LoraConfig(...)` are taken verbatim
from the docs. A client without `create_session` is used directly as the
session, which is the most likely benign variation.

### 4. Response shapes

Loss and token counts are read defensively — attribute *or* mapping access,
several candidate names (`loss`; `num_tokens`/`tokens`/`total_tokens`), and a
missing value never fails a run that otherwise succeeded. Similarly,
`save_weights` may return a string or an object; both are accepted.

### 5. Base-model list

The documented base models are recorded in `DOCUMENTED_BASE_MODELS`, but
`validate_base_model()` **warns rather than fails** for an unlisted name.
River authorizes models per account (`get_capabilities()`), so refusing an
unlisted-but-authorized model would be worse than a warning followed by
River's own authoritative error.

LoRA rank, by contrast, is a hard documented limit (1–32) and *is* enforced
locally — cheap to check, and a clearer message than a remote 400.

## Turning it on

```bash
pip install river-client            # not available from PyPI at time of writing
export RIVER_API_KEY=rv_...

stateset-agents train-remote --provider river \
    --dataset improved/curated.jsonl \
    --base-model Qwen/Qwen3.5-9B \
    --lora-r 16 --num-epochs 3
```

Both the key and the SDK are checked at submit time, each with a message
naming exactly what is missing and how to supply it. Neither is needed to list
providers or construct the executor — the SDK is imported lazily.

To test the plumbing without a key, inject a client:

```python
from stateset_agents.remote.river import RiverExecutor

executor = RiverExecutor(client=my_fake_client, tokenizer=my_tokenizer)
```

`client` and `tokenizer` are the two seams; with both injected, nothing in this
module touches the network. That is how the entire test suite runs.

## First real run: what to check

1. Does `create_model` accept our `LoraConfig` kwargs? (Wrong keyword → clear
   `TypeError` from the SDK.)
2. Does loss *decrease*? A flat, high loss means assumption 1 is wrong.
3. Does `save_weights` return something `river://`-shaped? If not,
   `river_checkpoint.json` will show whatever it did return —
   `_as_uri()` stringifies rather than crashing.
4. Does sampling with the returned checkpoint reflect the training data? That
   is the real end-to-end proof, and no unit test here can substitute for it.
5. Does the response expose token counts? If so, the ledger's `tokens` field
   starts populating and per-token cost becomes reconstructible.

## What we confirmed against the live API

Live training and checkpoint sampling established the full executor path.
Earlier direct REST probing also established three details the SDK docs did not:

- There **is** a plain REST surface (`GET /v1/models` answers), despite the
  documentation presenting an SDK-only interface.
- It authenticates with `Authorization: Bearer rv_...` — unauthenticated
  requests answer `401`.
- Errors come back in an OpenAI-shaped envelope. An unfunded account answers
  `402` with `{"error":{"message":"Billing: insufficient_funds",...}}`.

The executor translates those two account states into named, actionable
errors rather than a generic training failure, because they are the first
thing a new user hits and no amount of retrying fixes either one.

Everything else here remains **unverified**: no fine-tune has run, and the
batch-shape assumptions above stand until a funded account confirms them.
