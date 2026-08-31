# Live provider canaries

StateSet runs a non-billable compatibility probe through the same SDK/API
clients used by remote training:

```bash
stateset-agents provider-canary --provider river --strict
stateset-agents provider-canary --provider runpod --strict
stateset-agents provider-canary --provider fireworks --strict
stateset-agents provider-canary --provider coreweave --strict
stateset-agents provider-canary --provider nebius --strict
```

Repeat `--provider` to build a matrix. Omitting it probes the established
River/RunPod/Fireworks scheduled set. Add
`--output canary.json` to retain schema-versioned JSON evidence.

## What it proves

- **River:** the key authenticates, service health responds, and account
  capabilities can be read.
- **RunPod:** the account pod inventory is readable, no local cleanup lease is
  outstanding, and no `stateset-canary-*` pod was leaked.
- **Fireworks:** models, fine-tuning jobs, and deployments are readable, and
  no `stateset-canary-*` job or deployment was leaked.
- **CoreWeave:** the configured kubeconfig is accepted and its identity may
  create Kubernetes Jobs; no workload is created.
- **Nebius:** the selected official CLI profile can list Serverless AI jobs;
  no workload is created.
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

`.github/workflows/cloud-provider-verify.yml` is the separate, protected
CoreWeave/Nebius certification lane. `canary` mode requires the exact `READ
ONLY` acknowledgement. Billable `sft` mode requires `CLOUD GPU SPEND APPROVED`,
a published `stateset-agents` version for the remote container, provider
credentials, and approval through the `cloud-provider-verification`
environment. The Nebius installer is verified against the environment's
`NEBIUS_CLI_INSTALL_SHA256` secret before execution.
