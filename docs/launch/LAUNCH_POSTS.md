# Launch posts — ready to adapt and publish

> **Gate: do not post any of these until the flagship benchmark exists.**
> Run `make flagship-benchmark-all` (see `benchmarks/FLAGSHIP.md`), then replace
> every `[NUMBER]` placeholder with the real 3-seed mean ± std and link the
> committed result JSONs. You get one launch; spend it with receipts.

---

## 1. Show HN (post the repo link; this is the top comment)

**Title:** Show HN: StateSet Agents – RL fine-tune your AI agent from its own production logs

Hi HN — we built an open-source RL framework with one specific opinion: the
unit of training should be the *conversation*, not the single response, and
the training data should be the logs your agent already produces.

The loop, concretely:

    pip install stateset-agents
    stateset-agents ingest --format openai my_agent_logs.jsonl -o transcripts/
    stateset-agents improve run --transcripts transcripts/ --reward customer_support -o improved/

That grades every conversation (rule-based rewards or LLM-judge), curates the
turns above threshold into a training set, and emits the exact fine-tune
command to run next (GSPO — sequence-level GRPO). The whole demo runs offline
on CPU in about five minutes; the fine-tune step wants a GPU.

Some things we did that we think are unusual:

- Our CI trains a real (tiny) model nightly and asserts the target-token
  probability strictly increases — verified against zero-signal and
  reversed-reward controls. If the trainer stops learning, the build goes red.
- Every suggested command the tool prints is regression-tested against the
  real CLI parsers, because we got burned by our own docs.
- There's an MCP server (`stateset-agents mcp`), so Claude Code/Desktop or any
  MCP client can drive the improve loop conversationally.
- Result on the flagship benchmark: [NUMBER] judge-score improvement, 3 seeds,
  8B model, multi-turn customer support, receipts in the repo.

Five group-based trainers (GRPO/GSPO/GEPO/DAPO/VAPO), offline RL for logged
conversations, vLLM rollouts, a Rust GAE kernel, FastAPI serving with
OpenAI/Anthropic-compatible endpoints.

Honest caveats: single-node scale (verl/OpenRLHF beat us on distributed
throughput today), and the license is BUSL-1.1 [UPDATE IF CHANGED].

Repo: https://github.com/stateset/stateset-agents — happy to answer anything.

---

## 2. X/Twitter thread

**1/** Your AI agent produces logs all day. They're a training set.

We built an OSS framework that turns them into a better agent:
ingest → grade → curate → RL fine-tune. One command each.

[NUMBER] judge improvement on an 8B model, 3 seeds, receipts in repo. 🧵

**2/** The differentiator vs TRL/verl: conversations are the RL episode.
Multi-turn credit assignment, group-relative advantages (GRPO/GSPO family),
rewards that score whole dialogues — not single completions.

**3/** `stateset-agents ingest --format openai logs.jsonl`
`stateset-agents improve run --reward customer_support`

Works on OpenAI-format and LangChain traces. Five minutes, CPU only, offline.
[ATTACH: demo GIF/video]

**4/** We made the CI prove the RL actually works: a nightly job trains a tiny
model and asserts P(target token) strictly increases — and fails under
zero-signal or reversed rewards. Your move, other frameworks. 😄

**5/** It speaks MCP. `claude mcp add stateset-agents -- stateset-agents mcp`
and Claude can grade your transcripts, curate a dataset, and dry-run a
fine-tune config from chat.

**6/** The flagship result: [NUMBER] on multi-turn customer support, 8B model,
3 seeds, publish-gated (no seed selection, judge-stability precheck, negative
results committed too). Methodology + JSONs: [LINK]

**7/** pip install stateset-agents
Repo: https://github.com/stateset/stateset-agents
Whitepaper, cookbook, and a 10-example getting-started ladder inside. ⭐️s welcome, issues answered fast.

---

## 3. r/LocalLLaMA

**Title:** We open-sourced an RL framework that fine-tunes your agent from its own conversation logs (GSPO/GRPO, multi-turn, runs the demo on CPU)

Most RLHF tooling assumes single-turn prompt/response pairs. Agents don't work
that way — the thing you care about is whole conversations. StateSet Agents
treats the conversation as the RL episode:

- **Bring your own logs**: `ingest` converts OpenAI chat format or LangChain
  dumps into trajectories.
- **`improve run`**: grades every transcript (rule-based or LLM-judge
  rewards), curates the best turns into a training set, prints the exact
  GSPO/SFT command to run next.
- **Trainers**: GRPO, GSPO (sequence-level — the one that behaves on long
  outputs/MoE), DAPO, VAPO, GEPO, plus offline RL (BCQ/CQL/IQL/DT) for pure
  log-based training. LoRA + 4-bit throughout, vLLM for rollouts.
- **Receipts**: [NUMBER] judge improvement, 8B, 3 seeds, committed JSONs.
  Nightly CI literally trains a tiny model and asserts it learns.

Demo is CPU-only and offline (`bash examples/five_minute_demo.sh`). MCP server
included if you want Claude to drive it.

What we'd love feedback on: the reward-function API, the curation thresholds,
and what model families you'd want presets for (12 included: Qwen3.5, GLM-5.x,
Kimi, Llama-3, Mistral, Gemma).

---

## 4. One-paragraph blurb (directories, newsletters, awesome-lists)

StateSet Agents is an open-source RL framework for improving conversational AI
agents from their own logs: ingest OpenAI/LangChain conversation traces, grade
them with composable rewards or an LLM judge, curate the best turns, and
fine-tune with multi-turn group-based RL (GRPO/GSPO/DAPO/VAPO). Ships a
one-command improve loop, an MCP server so agents can drive their own
improvement, 12 model presets, vLLM/LoRA support, and a CI pipeline that
nightly proves the trainer still learns. `pip install stateset-agents`.

---

## Posting order & notes

1. Flagship benchmark first (the gate above). 2. Record the demo GIF
   (`examples/five_minute_demo.sh` under asciinema/terminalizer). 3. Show HN
   Tue–Thu ~14:00 UTC; reply to every comment for the first 3 hours.
   4. X thread same day, Reddit the day after (different framing, don't
   cross-link HN). 5. Submit the MCP server to registries the same week.
   6. LangChain/LangGraph docs-example PR after the first wave (link the
   launch discussion for credibility).
