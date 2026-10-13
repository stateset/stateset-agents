# 1.5B GSPO: StateSet vs direct TRL — protocol v5.1 (pre-release harness)

**Verdict: parity at 1.5B.** On identical work per step (4 prompts × 4
completions) and identical sampling (temperature 0.7, top-p 0.9, no top-k,
no repetition penalty, no stop strings), StateSet's native GSPO and TRL 1.12's
GRPOTrainer (GSPO sequence-level ratio) are indistinguishable in quality and
throughput:

| Seed | StateSet final (gain) | TRL final (gain) | Paired difference |
|---:|---:|---:|---:|
| 42 | 32.4% (+3.1) | 28.9% (−0.4) | +3.5 |
| 1337 | 29.7% (+0.4) | 32.4% (+3.1) | −2.7 |
| 2026 | 32.0% (+2.7) | 32.0% (+2.7) | 0.0 |

Mean gain +2.08 vs +1.82 points; paired difference +0.26 points, 95% CI
[−7.5, +8.0] (t, n=3). Throughput 0.993 vs 0.999 samples/s; peak VRAM 10.2 vs
10.5 GiB. Baseline 29.3% on 256 held-out GSM8K problems. **No quality
advantage is claimed in either direction.**

**This is not `benchmarks/a_plus_gate.py` evidence.** The harness commit
(`b006aab`, v0.64.2 + PR #99 + batched GSPO) predates the frozen release
commit the A+ gate requires; the comparison must be rerun at that commit.

## How this protocol got here

- v4 was unmatched: the TRL adapter passed the prompt count as TRL's batch,
  which TRL counts in completions (1 prompt/step vs the native leg's 4), and
  overstated TRL throughput 4×. Stopped after one native run.
- v5 fixed the batch but the native leg still sampled with `AgentConfig`
  defaults (temperature 0.8, top-k 50, repetition penalty 1.1, a stop at a
  blank line that truncated chain-of-thought). Stopped after one pair.
- v5.1 (this run) matches sampling on both legs and records it in each native
  run's `algorithm_config.sampling`. Batched GSPO generation (PR #101) took
  native training from 6,032 s to 1,357 s for 200 steps.

## Files

- `evidence/*.json` — per-seed measured evidence (schema v2), validated by
  `benchmarks/framework_comparison.py`.
- `report/` — the validator's report.
- `accounting/shootout-summary.json`, `runpod-provider.json` — pod lifecycle,
  authoritative price, termination confirmation. Cost $5.97 (5.5 h, 1× L40S).

Raw run directories (logs and LoRA adapters, 332 MB) are not committed. The
report was generated with them present, which re-hashes every artifact the
evidence names; reproduce with the commands below.

```bash
python benchmarks/runpod_shootout.py benchmarks/runpod_shootout_v5_1_manifest.json \
  --execute --confirm-max-cost-usd 25 --output-dir <dir>
python benchmarks/framework_comparison.py <dir>/evidence --output-dir <dir>/report
```
