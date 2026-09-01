# Flagship benchmark — the one number that sells the framework

**Goal:** a reproducible, headline-grade result on a real (7–8B) model on the
framework's differentiator task — multi-turn customer support — with receipts
that anyone can re-run from a fresh clone. The existing whitepaper result
(judge +0.079 on a 0.5B model, §11.7) proves the pipeline; this proves the
*product*.

## The run

Three seeds × GSPO on an immutable 7–9B instruct checkpoint, multi-turn
customer-support composite reward, and cross-family judge-scored evaluation
before/after. Copy `flagship_manifest.example.json`, replace every placeholder,
and use one command for the complete roster:

```bash
cp benchmarks/flagship_manifest.example.json benchmarks/flagship_manifest.json
# Fill immutable revisions, exact hardware, provider billing source, cost limit,
# cross-family judge, and the provider driver before continuing.

make benchmark-flagship-run \
  MANIFEST=benchmarks/flagship_manifest.json \
  OUTPUT_DIR=benchmark_results/flagship_v1/preflight \
  EXTRA_ARGS=--preflight

make benchmark-flagship-run \
  MANIFEST=benchmarks/flagship_manifest.json \
  OUTPUT_DIR=benchmark_results/flagship_v1/measured
```

The driver is invoked as an argv list with `shell=False`. It must report the
provider-derived cost and create its policy artifact inside the runner-owned
directory. The outer runner measures wall time, hashes the artifact itself,
retains stdout/stderr and failed attempts, and refuses partial matrices.

## Publish gates (do not publish a number that fails these)

1. **Three seeds, all reported** — no seed selection. Mean ± std in the
   headline; per-seed JSONs committed under `benchmark_results/flagship_v1/`
   (the schema in `benchmark_results/SCHEMA.md` applies).
2. **Judge stability** — every seed records the cross-family judge's measured
   self-disagreement; any value above 0.05 invalidates the matrix.
3. **Baseline parity** — the pre-training eval uses the identical prompt
   template, decoding params, and judge as the post-training eval
   (`--skip-baseline` is forbidden for the flagship).
4. **Provenance** — harness commit, manifest/config digests, model, dataset and
   judge revisions, exact GPU/CUDA/driver, policy digest, external wall time,
   and provider-derived cost are bound to the evidence.
5. **A negative or null result still gets committed** to
   `benchmark_results/flagship_v1/` — the credibility of every other number
   in this repo depends on not silently discarding runs.

## Where the number goes once it exists

- README badge + "What's new" headline (replace the §11.7 badge).
- `docs/WHITEPAPER.md` new section, anchored to the exact commit.
- The PyPI project description (long_description picks up README).

## Hardware notes

- The flagship gate accepts only 7–9B models. A smaller diagnostic is useful,
  but must use a separate output directory and cannot satisfy this gate.
- CI never runs this (GPU); the nightly `-m slow` convergence test
  (`tests/e2e/test_gspo_convergence_tiny.py`) is the automated proxy that the
  training loop still learns.
