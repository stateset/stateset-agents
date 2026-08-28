# Live provider canaries

StateSet runs a non-billable compatibility probe for River, RunPod, and
Fireworks through the same SDK/API clients used by remote training:

```bash
stateset-agents provider-canary --provider river --strict
stateset-agents provider-canary --provider runpod --strict
stateset-agents provider-canary --provider fireworks --strict
```

Repeat `--provider` to build a matrix, or omit it to probe all three. Add
`--output canary.json` to retain schema-versioned JSON evidence.

## What it proves

- **River:** the key authenticates, service health responds, and account
  capabilities can be read.
- **RunPod:** the account pod inventory is readable, no local cleanup lease is
  outstanding, and no `stateset-canary-*` pod was leaked.
- **Fireworks:** models, fine-tuning jobs, and deployments are readable, and
  no `stateset-canary-*` job or deployment was leaked.
- Every report states `billable_resources_created: 0`. Credential values are
  redacted from provider errors.

The read-only canary does not replace lifecycle tests. RunPod's weekly
`gpu-verify.yml` workflow still creates a cheap pod, trains, retrieves the
adapter, asserts the training effect, and terminates the pod. Fireworks and
River training-effect evidence remains the responsibility of their separately
budgeted lifecycle jobs.

## Automation

`.github/workflows/provider-canary.yml` runs weekly and on demand. Configure:

- `RIVER_API_KEY`
- `RUNPOD_API_KEY`
- `FIREWORKS_API_KEY`
- `FIREWORKS_ACCOUNT_ID`

Missing credentials fail the strict canary rather than producing a misleading
green check. JSON evidence is retained as a workflow artifact for 30 days.

`.github/workflows/slow-e2e.yml` separately runs the opt-in slow test lane with
per-test and job-level time limits, including subprocess and protocol tests
that are intentionally excluded from pull-request latency.
