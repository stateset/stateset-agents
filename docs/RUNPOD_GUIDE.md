# Fine-tuning on RunPod

Everything in this guide has been run against real rented hardware. Where a
number appears — a disk size, a cost, a model that fits — it came from a run
that either worked or failed instructively, not from a spec sheet.

The shape of it: you have conversation logs and no GPU. You rent one for
twenty minutes, train an adapter, get proof it learned, and give the hardware
back. A 30B model costs about a dollar.

---

## 1. Setup (once)

**An API key.** Create one at [console.runpod.io](https://console.runpod.io/user/settings):

```bash
export RUNPOD_API_KEY=...        # or put it in ~/.runpod/config.toml
```

**An SSH keypair.** RunPod pods are reached over SSH; the executor injects your
public key at pod creation and needs no password anywhere:

```bash
ls ~/.ssh/id_ed25519.pub || ssh-keygen -t ed25519
```

**The package**, with the remote extra:

```bash
pip install "stateset-agents[remote]"
```

That is the whole setup. There is no RunPod SDK dependency, no daemon, and no
account linking — the executor talks to RunPod's REST API and moves files with
`scp`.

---

## 2. Your first run (about $0.30)

Start small enough that mistakes are cheap. A tiny model on a $0.25/hr card:

```bash
stateset-agents train-remote --provider runpod \
  --gpu "NVIDIA RTX A4000" \
  --dataset improved/curated.jsonl \
  --base-model Qwen/Qwen3.5-0.8B \
  --max-cost 2
```

What happens, in order: a pod is created, your dataset is copied to it, the
published package is installed, training runs, the adapter is copied back to
`outputs/`, and **the pod is terminated** — on success, on failure, and on
timeout. Training cleanup runs in a `finally`, so a hard kill of the caller or
a laptop power loss can bypass it; check the RunPod console after that kind of
failure and terminate any orphan. (`serve-remote` separately arms an in-pod
watchdog.)

Training now writes a local cleanup lease immediately after each pod is
created. After a caller crash, inspect those leases without touching remote
state, then explicitly terminate them:

```bash
stateset-agents runpod-orphans
stateset-agents runpod-orphans --terminate
```

Leases are removed only after the RunPod API confirms termination. The command
never deletes network volumes.

Check what it cost:

```bash
stateset-agents costs
```

---

## 3. Choosing a GPU and a disk

These two choices cause most first-run failures.

**Disk: roughly 2.5× the model download.** The checkpoint is downloaded, then
unpacked into memory-mapped files; the default 40GB is fine for models up to
about 7B and nothing larger. A 63GB checkpoint on a 40GB disk dies mid-download
with an opaque HuggingFace cache error (`File reconstruction error: Background
writer channel closed`) — that exact failure cost a real run here.

| Model size (BF16) | `--container-disk-gb` |
|---|---|
| ≤ 7B (~15GB) | 40 (default) |
| 14B (~30GB) | 90 |
| 30B (~60GB) | 160 |

**GPU: enough VRAM for the weights plus training overhead.** LoRA keeps the
optimizer state small, so the weights dominate:

| Model | Card that works | Notes |
|---|---|---|
| Qwen3.5-0.8B | RTX A4000 (16GB) | ~$0.25/hr |
| Qwen2.5-14B | L40S (48GB) | ~$0.90/hr |
| Muse-Glimmer-30B (63GB) | H100 80GB | ~$3.30/hr, verified |
| Muse-Glimmer-30B (63GB) | 2× L40S with `--gpu-count 2` | sharded, ~$1.80/hr, verified |

GPU names are RunPod's own vocabulary, exactly as their console spells them
(`"NVIDIA RTX A4000"`, `"NVIDIA H100 80GB HBM3"`). A wrong name and an
out-of-capacity pool look identical from the outside — both surface as a `500`
from the API — so if a type you believe exists keeps failing, try another
before doubting the name.

**Capacity is real and varies by hour.** Multi-GPU configurations and specific
datacenters are scarcest. Scripting a small fallback list is normal practice:

```python
for gpu in ["NVIDIA L40S", "NVIDIA RTX A6000", "NVIDIA RTX A5000"]:
    ...  # try, catch the provider error, move on
```

---

## 4. Proving the model actually learned

A training run that exits zero has proved that training *ran*. To find out
whether it *worked*, hand it held-out prompts:

```bash
cat > held_out.txt <<'EOF'
Where is my order #77701?
{"prompt": "How do I cancel order #90210?", "expect": ["Astra @ StateSet", "90210"], "forbid": ["thinking process"]}
EOF

stateset-agents train-remote --provider runpod --gpu "NVIDIA H100 80GB HBM3" \
  --container-disk-gb 160 \
  --dataset improved/curated.jsonl \
  --base-model meta-models/Muse-Glimmer-30B \
  --eval-prompts held_out.txt --eval-max-new-tokens 300
```

Plain lines are prompts. A line that parses as a JSON object is an assertion
spec: `expect` substrings that must appear in the fine-tuned answer, `forbid`
substrings that must not. The job writes `eval_results.json` beside the
adapter — the base model's answer next to the tuned model's, for every prompt —
and **exits non-zero if an assertion fails**, after saving the adapter, so a
failed check never destroys what you paid to train.

Two lessons from real runs:

- **Reasoning models need a bigger budget.** Nemotron 3.5 Lightning spends its
  first tokens thinking; at the default 90 tokens the eval captured only the
  preamble and looked like a failure. `--eval-max-new-tokens 300` fixed it.
- **Fidelity takes more epochs than fluency.** Both Muse Glimmer and Nemotron
  learned the *shape* of a support reply in 3 epochs, but Nemotron invented a
  brand name ("Astra @ Heyday Support") until epoch 8. An `expect` assertion on
  your brand catches exactly this.

---

## 5. Talking to what you trained

```bash
stateset-agents chat-remote \
  --base-model meta-models/Muse-Glimmer-30B \
  --adapter outputs/sft_v1
```

Rents a pod, loads base + adapter, and gives you a multi-turn conversation;
the pod dies when you exit. Every conversation is saved to
`chat_transcripts/` in the format `ingest` accepts, which is how the loop
closes:

```bash
stateset-agents ingest --format openai --input chat_transcripts/chat_*.jsonl --output transcripts/
stateset-agents improve run --transcripts transcripts/ --reward customer_support --output improved2/
stateset-agents train-remote ... --dataset improved2/curated.jsonl --parent-adapter outputs/sft_v1
```

`--parent-adapter` records the generation link in the new adapter's manifest,
so `stateset-agents adapters` can show you the family tree later.

---

## 5b. Serving: merge, verify, or both

Two hard-won rules for serving what you trained:

- **Hybrid model families (Qwen3.5/3.8) need `--merge`.** vLLM loads
  their LoRA adapters without error and silently serves the base weights
  (proven by byte-identical greedy completions — see the DISPROVEN row in
  [`PROOFS.md`](PROOFS.md)). `serve-remote --merge` folds the adapter into
  full weights on the pod and serves the merged checkpoint; the merge
  verifies its own effect (`merge_probe.json`) and refuses to serve a
  no-op.
- **Every adapter serve now self-verifies**: after readiness the endpoint
  is probed greedy adapter-vs-base; identical output warns loudly, or
  fails and terminates the pod with `--strict`. Multiple adapters can ride
  one endpoint for A/B: `--adapter champion=... --adapter challenger=...`.

And the one-command path: `stateset-agents deploy --dataset ...
--base-model ...` rents, trains, releases the hardware, and serves the
fresh adapter with URL + token printed.

## 5c. Self-improvement on pods

The flywheel (`stateset-agents flywheel --provider runpod ...`) runs the
harvest → curate → train → measure loop on rented GPUs — same command as
the zero-infrastructure River version, same stopping rules (plateau,
perfection, dry harvest, `--max-cost`). Live results and the measured
operating regime are in [`FLYWHEEL_DOMAIN2.md`](FLYWHEEL_DOMAIN2.md) and
[`rl-vibe.md`](rl-vibe.md); difficulty-parameterized eval kits come from
`python -m stateset_agents.training.eval_ladder`.

## 6. Spending less

**Community cloud** is spot-priced and interruptible — roughly half the cost of
secure cloud:

```bash
stateset-agents train-remote ... --cloud-type COMMUNITY
```

Interruption is survivable: if the pod dies mid-run, the executor provisions a
fresh one and retries. By default the retry restarts training, because the dead
pod's checkpoints died with its container disk. Attach a network volume and
they survive:

```bash
stateset-agents train-remote ... --network-volume-id <id> --cloud-type COMMUNITY
```

The volume mounts at `/workspace`, so checkpoints outlive the pod and the retry
resumes from the newest one instead of starting over. Volumes are
datacenter-scoped (the pod is automatically pinned to the volume's datacenter,
which narrows your GPU choices) and **bill monthly whether or not you use
them** — delete them when done.

**Ceilings.** `--max-cost N` refuses to start a run whose worst case (the full
`--timeout` at the quoted hourly rate, multiplied by `--gpu-count`) would
exceed N dollars. A pod the provider will not price is refused rather than
rented.

---

## 7. Bigger than one card

```bash
stateset-agents train-remote ... --gpu-count 2 --gpu "NVIDIA L40S"
```

With more than one GPU visible, the checkpoint loads with `device_map="auto"`
and shards across them; the logs record the split
(`Model sharded across devices: 0=24 module(s), 1=36 module(s)`), so you can
confirm both cards were actually used rather than merely rented. Verified with
a 63GB checkpoint across two 48GB cards.

---

## 8. When something goes wrong

| Symptom | Cause | Fix |
|---|---|---|
| `500 Server Error` on pod creation | No capacity for that GPU type (or a wrong type name) | Try another GPU type; multi-GPU and specific datacenters are scarcest |
| `NET_003 … never became reachable` | Pod provisioned but never got networking | Retry; if a whole pool does this repeatedly, switch pools |
| `File reconstruction error` mid-download | Container disk too small | `--container-disk-gb` ≈ 2.5× the checkpoint |
| Job dies right after "Loading tokenizer and model" | Out of VRAM for the weights | Bigger card, or `--gpu-count 2` |
| `serve-remote` pod RUNNING with no IP/ports | That host will never publish networking | Fixed: networking now fails after 5 min (not 30) and a fresh pod is tried |
| Run hangs with the pod alive but idle | Was: dead SSH peer. Fixed — keepalives now detect a lost pod in ~2 minutes | Update to ≥ 0.25.0 |
| `TrainingArguments … unexpected keyword` | transformers drift on the pod | Fixed in ≥ 0.24.0 (arguments are filtered against the installed signature) |
| Eval output is all reasoning, no answer | Reasoning model, budget too small | `--eval-max-new-tokens 300` |

**Check for orphans** any time you are unsure — the executor is careful, but
scripts you write yourself may not be:

```bash
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/pods
curl -s -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/networkvolumes
```

Pods bill by the second; volumes bill monthly. In this project's own testing,
every executor-managed pod terminated correctly — including through three
agent crashes — while the one resource that leaked was a volume created by a
hand-written script outside the executor's lifecycle. Create resources through
the executor where you can.

---

## 9. Verified configurations

Each row is a run that actually happened here:

| Model | Hardware | Result |
|---|---|---|
| Qwen3.5-0.8B | 1× RTX A4000 | ~5 min, adapter returned |
| Qwen3.5-0.8B | 1× A4000, COMMUNITY cloud | spot pricing works |
| Qwen3.5-0.8B | 1× card + 25GB network volume | checkpoints on durable storage |
| Qwen2.5-14B (30GB) | 2× L40S | $0.73, 11 min |
| Muse-Glimmer-30B (63GB) | 1× H100 80GB | 258MB adapter; persona learned from 140 examples |
| Muse-Glimmer-30B (63GB) | 2× L40S, `--gpu-count 2` | sharded 24/36 modules, $0.35 |
| Nemotron-3.5-Lightning-30B | 1× H100 80GB | 8 epochs to hold a brand name |
| GSPO convergence (tiny model) | 1× RTX A4500 | target-token probability 2.8e-05 → 0.125 |

---

## See also

- [`CLI_REFERENCE.md`](CLI_REFERENCE.md) — every flag for `train-remote`, `chat-remote`, `serve-remote`, `costs`, `adapters`
- [`FLYWHEEL_EXPERIMENT.md`](FLYWHEEL_EXPERIMENT.md) — measured curation quality, and what the loop does and does not yet prove
- [`SUPPORTED_MODELS.md`](SUPPORTED_MODELS.md) — first-class starters and their presets
