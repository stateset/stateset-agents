# Zero to calling your own fine-tuned model

Start with nothing but conversation logs. Finish with an API endpoint that
answers in your agent's voice, called the same way you call any other model —
OpenAI-compatible `chat/completions` or Anthropic-style `messages`.

The whole path costs about a dollar of rented GPU and takes under an hour,
most of it waiting for downloads.

```
your logs ──▶ ingest ──▶ improve ──▶ train-remote ──▶ adapter ──▶ serve ──▶ API call
                                     (rents a GPU,              (local or
                                      gives it back)             rented)
```

---

## Step 0 — Install

```bash
pip install "stateset-agents[remote]"

export RUNPOD_API_KEY=...              # console.runpod.io/user/settings
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
```

No GPU needed on your machine for any step. See
[`RUNPOD_GUIDE.md`](RUNPOD_GUIDE.md) for the details of renting.

---

## Step 1 — Turn logs into training data

If you already have agent logs in OpenAI chat format (or LangChain traces):

```bash
stateset-agents ingest --format openai --input my_agent_logs.jsonl --output transcripts/

stateset-agents improve run \
  --transcripts transcripts/ \
  --reward customer_support \
  --output improved/
```

`improve` grades every turn and keeps the good ones in `improved/curated.jsonl`.
That file is your training set.

**No logs yet?** Write 100–200 examples by hand in the same format — that is
genuinely enough to teach a voice:

```json
{"messages": [{"role": "user", "content": "Where is my order #10021?"},
              {"role": "assistant", "content": "Thanks for reaching out to Acme Support! I checked right away: order #10021 left our warehouse and arrives in 2 business days. — Robin @ Acme"}]}
```

The measured result below came from 140 such examples.

---

## Step 2 — Fine-tune on a rented GPU

```bash
cat > held_out.txt <<'EOF'
{"prompt": "Where is my order #77701?", "expect": ["Acme Support", "77701"]}
EOF

stateset-agents train-remote --provider runpod \
  --gpu "NVIDIA H100 80GB HBM3" --container-disk-gb 160 \
  --dataset improved/curated.jsonl \
  --base-model meta-models/Muse-Glimmer-30B \
  --output-dir outputs/support_v1 \
  --num-epochs 3 --lora-r 16 \
  --eval-prompts held_out.txt --eval-max-new-tokens 300 \
  --max-cost 5
```

A pod is created, trains, copies the adapter to `outputs/support_v1/`, and is
**terminated** — on success, failure, or timeout.

Two files matter when it finishes:

- `adapter_model.safetensors` — the thing you trained (a few hundred MB)
- `eval_results.json` — the base model's answers beside the tuned model's, on
  prompts it never trained on, with pass/fail for your assertions

Check the bill: `stateset-agents costs`.

Start smaller if you prefer — `Qwen/Qwen3.5-0.8B` on an `NVIDIA RTX A4000`
trains in about five minutes for well under a dollar, and every step below is
identical.

---

## Step 3a — Serve it locally, call it over HTTP

If you have any GPU (or are happy with CPU for a small model), this is the
shortest path to an endpoint and needs no further rentals:

```bash
pip install "stateset-agents[api]"

stateset-agents serve \
  --checkpoint outputs/support_v1 \
  --base-model meta-models/Muse-Glimmer-30B \
  --port 8000
```

The server exposes both API shapes on the same port:

**OpenAI-compatible** — point any OpenAI client at it:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "support_v1",
    "messages": [{"role": "user", "content": "Where is my order #77701?"}],
    "max_tokens": 200
  }'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
reply = client.chat.completions.create(
    model="support_v1",
    messages=[{"role": "user", "content": "Where is my order #77701?"}],
)
print(reply.choices[0].message.content)
```

**Anthropic-style messages** — same server, `/v1/messages`:

```bash
curl http://localhost:8000/v1/messages \
  -H "Content-Type: application/json" \
  -d '{
    "model": "support_v1",
    "max_tokens": 200,
    "messages": [{"role": "user", "content": "Where is my order #77701?"}]
  }'
```

`GET /v1/models` lists what is loaded. Auth is off by default for local use —
see [`API_AUTH.md`](API_AUTH.md) if present, and turn it on before exposing the
port anywhere.

---

## Step 3b — Serve it on a rented GPU

> **Hybrid models (Qwen3.5/3.8 families): add `--merge`.** vLLM silently
> ignores their LoRA adapters (it loads them, then serves base weights) —
> `--merge` folds the adapter into full weights on the pod and the serve
> self-verifies its effect. All adapter serves probe adapter-vs-base at
> startup and warn (or fail, with `--strict`) on identical output.
> Prefer one command end to end? `stateset-agents deploy` = train + serve.

When the model is too big for your hardware, rent the inference too. This
starts a vLLM OpenAI-compatible server on a pod and prints its URL and a
generated token:

```bash
stateset-agents serve-remote \
  --base-model meta-models/Muse-Glimmer-30B \
  --adapter outputs/support_v1 \
  --gpu "NVIDIA H100 80GB HBM3" \
  --max-hours 1
```

```bash
curl $ENDPOINT/chat/completions \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"model": "adapter", "messages": [{"role": "user", "content": "Where is my order #77701?"}]}'
```

Unlike training, **this pod stays alive after the command exits** — that is the
point of a server. Three things keep that from becoming an expensive mistake:

- `--max-hours` arms a self-destruct on the pod itself, so it dies even if your
  laptop does
- `stateset-agents serve-remote --list` shows what is running, with age and
  hourly cost
- `stateset-agents serve-remote --stop <name>` terminates it now

> **Status:** the pod, its ports, and the self-destruct are verified on real
> hardware (pods terminated correctly even when the local process was killed).
> Endpoint bring-up has repeatedly been blocked by GPU capacity during
> verification rather than by a defect. If it fails for you, Step 3a is the
> proven path and the same client code works against it.

---

## Step 4 — Just chat with it

For a conversation rather than an endpoint:

```bash
stateset-agents chat-remote \
  --base-model meta-models/Muse-Glimmer-30B \
  --adapter outputs/support_v1
```

Multi-turn, pod dies on exit, and every conversation is saved to
`chat_transcripts/` in the format Step 1 accepts — which is how you get a
second generation:

```bash
stateset-agents ingest --format openai --input chat_transcripts/chat_*.jsonl --output transcripts2/
stateset-agents improve run --transcripts transcripts2/ --reward customer_support --output improved2/
stateset-agents train-remote ... --dataset improved2/curated.jsonl --parent-adapter outputs/support_v1
```

`stateset-agents adapters` then shows the family tree, and
`stateset-agents costs` shows what the whole exercise cost.

---

## What this looks like when it works

From a real run — 140 examples, 3 epochs, one H100, about a dollar — answering
an order number that appeared in no training example:

**Base model:**

> `to=self` We need to respond. No context. Probably we don't have access to
> order tracking. We should ask for clarification…

**After fine-tuning:**

> Thanks for reaching out to StateSet Support! I checked right away: your order
> #77701 is on the way — it left our warehouse and should arrive within 3
> business days. Anything else I can help with? — Astra @ StateSet

The base model never answers the customer. The tuned one resolves the request,
carries the unseen order number, and signs off in the trained voice. That
difference is what `eval_results.json` records for you on every run, and what
`expect`/`forbid` assertions turn into a pass or a failed build.

---

## Where to go next

- [`RUNPOD_GUIDE.md`](RUNPOD_GUIDE.md) — GPU/disk sizing, spot pricing, multi-GPU, troubleshooting
- [`CLI_REFERENCE.md`](CLI_REFERENCE.md) — every command and flag
- [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) — first-class starters (Muse Glimmer, Nemotron 3.5, Qwen3-Coder, gpt-oss, DeepSeek V4, and more)
- [`FLYWHEEL_EXPERIMENT.md`](FLYWHEEL_EXPERIMENT.md) — what the improvement loop has and has not been measured to do
