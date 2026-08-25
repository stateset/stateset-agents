# rl-vibe.md — the state of stateset-agents, told straight

*A field report on what we built, what we proved, what broke, and what we
know now that we didn't two weeks ago. Every number below is a run that
happened; the receipts live in [`PROOFS.md`](PROOFS.md) and the retained
reports.*

---

## The one-paragraph vibe

We set out to build an RL framework for multi-turn conversational agents
and ended up with something more specific and more useful: a
**self-improvement loop whose epistemology is the product**. It raises a
model's ceiling on checkable tasks, knows when it can't, catches itself
cheating, finds each model's honest capability boundary, and writes all of
it down. It runs on your GPU, on rented pods, or on a remote-autograd
service with zero machines at all — and the same JSON spec drives all
three.

## The headline numbers

| metric | value |
|---|---|
| Ceiling-raises demonstrated | **7** (across 4 models, 3 providers, 4 domains) |
| Perfect scores (12/12) reached by the loop | **5** |
| Largest single lift | **2/12 → 12/12** (naive 35B, two wheel-turns) |
| Multi-turn lift (the founding claim) | 8/12 → **12/12** on episodes requiring cross-turn memory |
| Honest plateaus / walls (the loop refusing false wins) | 4, all autonomous |
| Reward hacks caught live and fixed by A/B | 1 (metric ↑ 0.67→0.84 while truth ↓ 6→4/12) |
| Shipped claims later DISPROVEN and recorded as such | 1 (vLLM hybrid-LoRA silent no-op) |
| Live-only bugs found on real hardware/services | **25+**, each now a tested fix |
| Releases in the sprint (v0.21.0 → v0.38.0) | 25, all on PyPI, ~140 commits |
| Unit test files | 189 |
| Total compute spend for every result above | **≈ $40** (RunPod) + River tokens |

## What the framework is today

**Training** — LoRA SFT that runs identically on `local`, `modal`,
`runpod` (SSH'd pods with self-destructs, cost ceilings, adapter lineage
manifests, pod-death retry, multi-GPU sharding), and `river` (remote
autograd: session → train_step → `river://` checkpoints, transient
recovery per their own taxonomy, live step/loss streaming via
`STATESET_RIVER_VERBOSE`).

**Evaluation** — objective `{prompt, expect, forbid}` specs, LLM judges
with thresholds, and the **difficulty ladder**: declarative domain specs
that generate train/harvest/eval kits at parameterized compound depth,
with refusal prompts (declined remedy → forbidden token) and N-turn
episode scripts whose final turn demands a summary of everything done.
Difficulty is a knob, not an authoring accident.

**The flywheel** — `stateset-agents flywheel`: best-of-N rejection
sampling against the checks, train the next generation on survivors,
measure greedily, repeat. Stops itself on plateau, perfection, dry
harvest, or a hard dollar ceiling checked *before* each rental.
`--repeats N` turns results into distributions. Multi-turn episodes roll
out branch-parallel and train on whole conversations.

**RL** — `--algorithm cispo`: every sample graded (expected-token
fraction + completeness bonus − refusal violation), group-relative
advantages, River's clipped-IS losses on their sampler's own logprobs.
Works single-turn and multi-turn (episode-level advantages broadcast
across every assistant turn). Verified live; at parity with SFT where
both were tried; its Goodhart failure mode observed, diagnosed, and
fixed.

**Serving** — `serve-remote`: vLLM OpenAI endpoints on pods with bearer
auth, SSE streaming, multi-adapter A/B under named models, `--merge` for
hybrid architectures (whose LoRA vLLM silently ignores — proven, then
fixed via full-weight merging with an on-pod effect probe), and
**self-verification**: every adapter serve greedy-probes adapter-vs-base
and refuses (or warns) on identical output. `deploy` = train-then-serve
in one command.

**Trust infrastructure** — [`PROOFS.md`](PROOFS.md) (21 claim rows:
live-verified with dates, unit-pinned, automation-pending, or DISPROVEN),
cost ledger where unknown ≠ free, lineage manifests with dataset hashes,
release gates that run before file mutation, weekly GPU/flywheel smokes
(armed, awaiting one repo secret).

## The experimental arc, compressed

**The mechanism** (proven 7×): where a model succeeds *rarely but
sometimes* at something checkable, one flywheel turn converts rarity into
reliability. Muse-30B support: 2/12→10/12 (reproduced ×2). Qwen-0.8B IT:
0→7 and 0→**11**/12. Qwen-9B via River, zero infra: 7→11/12. 35B travel:
7→**12/12** in one turn.

**The operating regime** (measured, twice from each side): the loop lifts
when the temperature harvest rate sits near ~60% with greedy headroom;
it honestly stalls when success is too common to be informative (65%,
83% harvests → plateaus). The harvest rate is the leading indicator —
watch it arc (23% → 93% across the 35B's two turns).

**Multi-turn** (the founding promise, closed): episode scripts where the
user never repeats the account reference make context carryover an
objective score. Rung 3: 8→12/12. Rung 4: the rung-3 adapter transferred
*upward* (11/12) and topped off at 12/12 — the ladder is a curriculum.
Rung 5: the 9B walled at 9/12 — confirmed from both directions (SFT
regressed and was auto-discarded; RL held flat with reward pinned at
1.70/2.0 and nothing to push against).

**Scale** (the finale): the same rung-5 wall fell to an episode-naive
35B in two wheel-turns, 2/12 → 11/12 → **12/12** — including the refusal
episode, the universal last-standing failure of every smaller model.
The ladder finds each model's ceiling; the flywheel drives to it; scale
moves it.

**The safety story** (lived, not claimed): the first shaped reward
Goodharted — mean reward climbed four rounds straight while the real
eval fell, the model having learned to resolve one issue confidently and
drop the rest. The objective/eval separation caught it the same hour;
the failure anatomy named the exploit; a completeness bonus fixed it on
an A/B rerun from the same checkpoint. This is why the objective and the
eval are different code paths.

## What broke (and what it taught)

Provider chaos absorbed into policy: RunPod pods dying mid-job, REST
500s, http-proxy ports that never publish TCP mappings, a pod that hung
its client for 28 minutes on a shell-precedence bug; River losing
in-flight requests (`NOT_FOUND`) near run-ends, a 4× pool slowdown, and
one ~12-hour full outage that the capacity experiment survived via an
auto-resuming serving probe. Silent failure is the real enemy: three
separate incidents were invisible until we built live instrumentation —
now everything streams.

The deepest lesson, twice-learned: **a "successful" verification that
doesn't check the effect is worse than none** (vLLM "loaded" adapters it
never applied; peft "loaded" mismatched keys with delta exactly 0.0).
Every serving path now proves its own effect or refuses.

## What's honestly not done

- **Real-transcript training.** Everything above is synthetic domains
  with substring/judge checks. The judge-gated path is built and tested;
  the pilot on genuine conversation logs is the thesis-deciding
  experiment, still unrun.
- **RL superiority.** Verified, safety-tested, never yet *better* than
  imitation. Knobs untuned; the theory says its regime is walls with
  variance, and the one wall we found had none.
- **Two five-minute switches** (PyPI trusted publisher, `RUNPOD_API_KEY`
  secret) between "verified this sprint" and "re-verifies itself weekly."
- **The rarity controller** — auto-tuning temperature/difficulty into the
  measured operating window — designed, unbuilt.
- Single-maintainer bus factor, mitigated only by the discipline that
  every incident becomes a test, a doc, or a proof row.

## The vibe, restated

Ship nothing you didn't watch work. Grade every claim. Let the loop stop
itself. Spend dollars, not weeks. Write down the failures with the same
pride as the wins — they're the moat. And when the eval says 12/12,
don't celebrate: build a harder rung.
