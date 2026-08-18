# The flywheel replicates: second domain, 35× smaller model, one command

**Result: 0/12 → 6/12 → 7/12 on out-of-distribution compound requests,
with the harvest rate rising 5.8% → 42.7% between generations — produced
by a single `stateset-agents flywheel` invocation for $2.60.**

This replicates [`FLYWHEEL_HEADROOM.md`](FLYWHEEL_HEADROOM.md) (Muse-Glimmer-30B,
customer support, 2/12 → 10/12) in a different domain (IT helpdesk), on a
different base model (Qwen/Qwen3.5-0.8B — 35× smaller), and — the product
point — through the shipped autonomous command rather than hand-driven
scripts. Run 2026-08-17.

## Setup

- **Persona**: "Byte @ TechNest" IT helpdesk. Seven issue types (password,
  VPN, printer, disk, email sync, 2FA, software install), each with a
  canonical resolution sentence containing an objective proof token
  ("reset link", "vpn profile", "print spooler", "update cache",
  "sync profile", "bypass code", "software center").
- **Gen-1**: LoRA on 140 synthetic *single-issue* tickets, 8 epochs
  (3 epochs left the canonical wording unanchored — see "what it took"),
  $0.58.
- **Eval**: 12 *compound* tickets (two issues per message, tickets 77xxx),
  pass = both proof tokens + the ticket number. Gen-1 never saw a compound
  ticket.
- **Harvest**: 30 disjoint compound tickets (88xxx), best-of-16 at
  temperature 0.9, only samples passing their own checks kept.

## The run (one command)

```bash
stateset-agents flywheel --base-model Qwen/Qwen3.5-0.8B \
  --initial-adapter outputs/domain2_gen1_e8 \
  --harvest-prompts harvest_prompts.json --eval-prompts eval_prompts.json \
  --gpu "NVIDIA H100 80GB HBM3" --generations 2 --best-of 16 \
  --num-epochs 8 --max-cost 6
```

| generation | eval (greedy) | harvest rate | cost |
|---|---|---|---|
| gen-1 (start) | **0/12** | 28/480 (5.8%) | — |
| gen-2 (trained on the 28) | **6/12** | 205/480 (**42.7%**) | $1.29 |
| gen-3 (trained on the 205) | **7/12** | — | $1.31 |

The structure is the headroom experiment's exactly: greedy decoding never
succeeds, temperature sampling occasionally does, curation keeps only the
successes, and one training generation converts a 5.8% latent capability
into majority-reliable behaviour. The harvest-rate jump (5.8% → 42.7%) is
the cleanest single number: the same 480-sample procedure against the same
checks, before and after one turn of the wheel.

## What it took (the honest part)

The first three spins failed, each finding a real bug now fixed with tests:

1. The pod installed the PyPI release, which predated the harvest module
   (`STATESET_AGENTS_WHEEL` seam added).
2. The harvest generated on CPU with an H100 at 0% for an hour — single-GPU
   loads land on CPU and nothing moved the model (same fix as sft's eval
   path).
3. A gate-failed job's artifacts died with its pod: the executor skipped its
   download step on non-zero exits, defeating `wait()`'s fetch-on-failure —
   and the flywheel's eval reader parsed an imagined format besides. Failed
   jobs now salvage artifacts before termination.

Two experiment-design lessons, both mirrored from earlier live runs: 3
epochs does not anchor canonical wording at 0.8B (8 does — the Nemotron
lesson), and run 2's plateau (11 harvested rows, 3 epochs → 0/12) shows the
training dose matters as much as the harvest quality.

## Limitations

- Single seed, one run of this exact configuration (the headroom result was
  reproduced twice; this one has not been re-run yet).
- Proof-token checks measure phrasing-anchored resolution, not semantic
  correctness; gen-1 at 3 epochs *paraphrased* both resolutions while
  passing nothing, so the metric undercounts loose-but-correct behaviour.
- Synthetic tickets; seven issue types; 0.8B model. The claim is the
  *mechanism* replicates, not that these numbers generalize.
- 7/12 is not 12/12: the remaining failures are wording near-misses, the
  same tail the headroom run saw.

## Cost of the whole campaign

Gen-1 training (twice: 3 then 8 epochs) $1.02; three failed spins $4.04
(the CPU-grinding hour is $3.60 of that); the successful run $2.60 —
**$7.66 total**, every line in the cost ledger.
