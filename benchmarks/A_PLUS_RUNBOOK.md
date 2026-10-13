# A+ evidence runbook

`benchmarks/a_plus_gate.py` passes only when every gate's evidence was
collected from **one clean commit** at the installed package version. Collect
paid evidence only after the code is frozen; anything collected earlier must be
re-run. `benchmarks/a_plus_evidence.py` keeps the bundle in one layout:

```bash
ROOT=benchmark_results/a_plus/$(git rev-parse HEAD)
python benchmarks/a_plus_evidence.py status --root "$ROOT"   # what is missing
```

## 0. Freeze

1. Merge the release branch to `master` and tag it. Every evidence run must
   check out that exact commit (`git status` clean, `pip install -e .` at it).
2. Publish `stateset/customer-support-multiturn-v1` on the Hugging Face Hub
   (`python scripts/build_support_dataset.py --output <dir>`, then upload) and
   pin its commit as `dataset_revision` in `benchmarks/flagship_manifest.json`.
   Pinning is a commit, so it happens before the freeze tag.

## 1. Free gates

```bash
python benchmarks/a_plus_evidence.py collect-free --root "$ROOT"
```

Runs the reliability matrix on CPU (3 faults x 3 seeds) and
`scripts/publish_readiness.sh` (tests with coverage, Bandit, pip-audit, build).

## 2. Paid gates

Every driver plans for free first, refuses to run without an explicit spend
ceiling at or above its worst case, and terminates what it provisions.

| Gate | Command | Hardware | Worst case |
|---|---|---|---|
| Flagship preflight (diagnostic) | `run_flagship_matrix.py flagship_preflight_manifest.json --preflight` | 1x H100 | ~$13.30 |
| Flagship (3 seeds) | `run_flagship_matrix.py flagship_manifest.json` | 1x H100 per seed | <= $75 |
| Competitive frameworks (5 x 3 seeds) | `runpod_shootout.py runpod_shootout_v6_manifest.json --execute` | 1x L40S | ~$52 |
| Scaling 1/2/4/8 GPUs x 3 seeds | `run_scaling_matrix.py --launcher-manifest scaling_launcher_runpod.example.json` | A40 pods, 2 nodes for 8 GPUs | ~$10-25 |
| Scaling image attestation | `scripts/build_scaling_image.py --execute --confirm-push ...` | registry push | free |
| Distributed async (4 scenarios x 3 seeds, 12 h soak) | `run_distributed_async_matrix.py distributed_async_manifest.json` | 2 CPU pods | ~$10.36 |
| Agent quality (tau3-bench + BFCL-v4) | `run_agent_quality_matrix.py` | GPU endpoint + user-simulator API | not yet priced |
| Providers (River, RunPod, Fireworks, CoreWeave, Nebius) | provider canaries | read-only | free; needs credentials |

Copy each run's evidence into the `a_plus_evidence.py` layout (`status` lists
the exact paths).

## 3. Gate

```bash
python benchmarks/a_plus_evidence.py gate --root "$ROOT"
```

The gate re-validates every raw artifact and writes `$ROOT/report.json`. It
also requires StateSet throughput within 10% of the fastest framework and a
flagship improvement of at least +0.03 with a positive paired 95% bound;
those are results, not paperwork, and may fail.
