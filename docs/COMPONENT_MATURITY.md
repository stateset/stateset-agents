# Component maturity policy

StateSet Agents publishes an evidence-backed maturity level for every canonical
product domain. The machine-readable source of truth is
[`contracts/component_maturity_v1.json`](../contracts/component_maturity_v1.json),
and `make release-governance` validates it on every pull request and release.

## Levels

- **Stable** components are covered by the v1 public contract, tests,
  documentation, and retained production or benchmark evidence. Breaking
  changes require a new versioned surface.
- **Beta** components are functionally complete and tested, but have published
  limitations and measurable graduation criteria. Compatible migrations are
  documented when their configuration or behavior changes.
- **Experimental** components are research surfaces, are not recommended for
  production, and may change in patch releases. Their limitations and promotion
  criteria must still be explicit.

## Current inventory

| Component ID | Maturity |
|---|---|
| `api-serving` | Stable |
| `bayesian-reward-model` | Experimental |
| `core-agents` | Stable |
| `deployment-assets` | Beta |
| `distillation` | Beta |
| `distributed-rollouts` | Beta |
| `environments-rewards` | Stable |
| `external-training-backends` | Beta |
| `gspo-dapo-gepo` | Beta |
| `grpo-training` | Stable |
| `memory` | Beta |
| `offline-rl` | Experimental |
| `ppo-rlaif-trainers` | Experimental |
| `preference-optimization` | Beta |
| `provider-control-plane` | Beta |
| `research-modules` | Experimental |
| `reward-learning-rlaif` | Beta |
| `rust-acceleration` | Beta |
| `sim-to-real` | Experimental |
| `tool-use-rl` | Beta |
| `vapo` | Experimental |

Since v0.64.0 the code for the `offline-rl`, `ppo-rlaif-trainers`,
`sim-to-real`, and `bayesian-reward-model` components lives under the
`stateset_agents.experimental` namespace, so the import path itself signals
that no stability guarantee applies. The previous `stateset_agents.training`,
`stateset_agents.evaluation`, `stateset_agents.environments`, and
`stateset_agents.rewards` paths were deprecated in v0.64.0 and removed in
v0.65.0. The four root sim-to-real exports (`stateset_agents.SimToRealMetrics`,
`SimToRealEvaluator`, `ConversationSimulator`, `ConversationSimulatorConfig`)
were deprecated in v0.64.0 and, after their v0.66.0 deadline, removed from the
v1 contract; import them from `stateset_agents.experimental.sim_to_real`.

The JSON manifest supplies each component's owner, documentation, tests,
retained evidence, limitations, graduation criteria, and stable-contract
references. Promotion is a reviewed evidence change, not a marketing decision.
The checker rejects a stable component without valid v1 references, and rejects
beta or experimental entries that omit limitations or measurable graduation
criteria.
