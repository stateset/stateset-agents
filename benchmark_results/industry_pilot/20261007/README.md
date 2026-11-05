# Qwen3.5 industry pilot — 2026-10-07

This is an SFT pipeline check with synthetic lookup conversations. It does not
establish domain quality or an advantage for reinforcement learning.

## Protocol

- One GPU per attempt, BF16 LoRA, one epoch, seed 42. The completed 2B run
  used an NVIDIA A40; the completed 4B run used an NVIDIA L40S.
- Each model has 48 training source groups and 16 held-out source groups.
- Evaluate both assistant turns separately: 32 requests per baseline/candidate.
  Each request includes the reference conversation history; this is not an
  autonomous tool-execution evaluation.
- Greedy decoding, thinking disabled, at most 96 generated tokens; identical
  settings and pinned base revision for baseline and candidate.
- Check six optimizer updates, nonzero saved LoRA B tensors, and a nonzero
  adapter-on/off logit difference after reloading the saved adapter.
- Exact tool names and typed arguments are scored independently of strict
  whitespace-normalized text agreement. Text paraphrases can fail even when
  they communicate the correct status.
- Synthetic data and fewer than 30 independent held-out groups must fail the
  default quality gate, regardless of score.

The wheel SHA-256 is
`078ce9f745205323e4be32140c34bbadf08659ae33d3e6c352f89728a12e1268`,
built from the packaged code at `f6ac0f227be26ad0833ff801e05d72ea8f9ee4fb`.
The corrected harness is at `71d4bd4d5dc07d98e7d5aa9a28c8e4d6dcfcb334`;
that commit changes only the benchmark and its tests. Each attempt manifest
records both immutable Hugging Face revisions and the exact harness digest.

## Attempts

1. `attempt1`: both model loads were rejected by a harness provenance check.
   Transformers extracts a text configuration without the parent configuration's
   commit metadata. No training result is claimed. Pod termination was confirmed.
2. `attempt2`: 2B training and reload checks completed. During 4B training, the
   launcher's byte-based log tail split a UTF-8 progress character and raised a
   decoding error. Its cleanup path retained partial artifacts and terminated
   the pod. No completed 4B result is claimed from this attempt.
3. `attempt3`: retried only 4B with line-based log tailing, but the A40 host
   never exposed SSH within 600 seconds. No training occurred; termination was
   confirmed.
4. `attempt4`: A6000 allocation failed with provider HTTP 500 after five API
   attempts. No pod ID was returned, and a subsequent account listing was empty.
5. `attempt5`: 4B training, saved-adapter reload, and held-out evaluation
   completed on an L40S. Startup was allowed 900 seconds within the same
   one-hour total lifetime.

## Results

| Model / industry | GPU | Optimizer updates | Peak allocated training GiB | Training seconds, including load/save | Reload max absolute logit delta | Tool exact match, base → adapter | Text exact match, base → adapter |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3.5 2B / retail | A40 | 6 | 8.25 | 45.38 | 0.75 | 16/16 → 16/16 | 0/16 → 0/16 |
| Qwen3.5 4B / technology | L40S | 6 | 17.36 | 79.07 | 0.71875 | 16/16 → 16/16 | 16/16 → 16/16 |

Timing and memory measurements are descriptive single-run observations. GPU
types differ between completed model runs, so these are not controlled model
speed comparisons.

Both quality gates failed because the data is synthetic and there are only 16
held-out source groups. The 2B run additionally failed the minimum group
reference-agreement floor. No reference-agreement improvement was observed:
4B already matched every reference before training, while 2B preserved correct
tool calls but used paraphrases instead of the required exact answer format.
The results establish working training and adapter reload, not quality lift.
Both retained prediction bundles reproduce their GPU `evaluation.json` exactly
when rescored locally, and both downloaded adapters match their recorded
SHA-256 digests. See [summary.json](summary.json).

## Reproduce scoring without a GPU

From the repository root, with the feature build installed:

```bash
stateset-agents industry evaluate \
  benchmark_results/industry_pilot/20261007/attempt2/qwen3.5-2b/prepared \
  --baseline benchmark_results/industry_pilot/20261007/attempt2/qwen3.5-2b/baseline.json \
  --candidate benchmark_results/industry_pilot/20261007/attempt2/qwen3.5-2b/candidate.json \
  --output /tmp/industry-pilot-rescored.json
```

Expected exit status: **1**, because the configured quality gate fails. Use a
new output path on each invocation; reports are never silently overwritten.
The GPU harness can be rerun with `python benchmarks/industry_pilot.py MANIFEST
OUTPUT_DIRECTORY` in an environment matching the retained dependency versions.
The archived launcher source documents the actual bounded provisioning run; its
local staging paths are machine-specific.

## Local verification

The isolated feature build passed **6,094 tests**, with 18 skips and **65.14%**
package coverage. All **346** packaged modules passed type checking. The
subsequent harness-only revision fix passed 17 focused tests. Raw full-suite and
type-check logs are retained under `local-verification/`.

## Spend and cleanup

The user authorized a $25 total pilot budget. Every pod has a one-hour lifetime
limit and a $5 per-pod ceiling. The A40 provider rate was $0.49/hour; other GPU
rates are recorded with their attempts. Provider records retain estimated compute
spend and termination confirmation. Estimates
use the provider rate and observed lifetime; they are not invoices and exclude
separate storage/network charges.

Across five attempts, four pods were allocated and all four were terminated.
The fifth attempt returned no pod ID. The final provider listing contained
**zero active pods**. Total estimated compute spend was **$0.4246**; the raw
per-attempt provider records and [final check](final-provider-check.json) are
retained.

Model weights and optimizer tensors remain in the local `/tmp` artifact
directories and are not included in this evidence bundle. Adapter hashes,
configuration, training state, raw predictions, and prepared data are retained.

## What this leaves unproved

- Representative industry performance: reviewed data, a fresh final test set,
  task-specific outcome checks, and uncertainty estimates are still needed.
- RL advantage: this pilot performs supervised fine-tuning. The corrected,
  matched native-GSPO-versus-TRL comparison still needs multiple seeds.
- Fleet reliability: startup and allocation failures in this pilot are retained;
  a successful model run does not erase them.

The latest scheduled GPU workflow at the time of this pilot was
[2026-10-05](https://github.com/stateset/stateset-agents/actions/runs/37317451465).
Its SFT job passed, but its RL job failed at pod allocation with HTTP 500 after
five retries. It supplies no new RL convergence result.
