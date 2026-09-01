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
| `core-agents` | Stable |
| `deployment-assets` | Beta |
| `distributed-rollouts` | Beta |
| `environments-rewards` | Stable |
| `external-training-backends` | Beta |
| `gspo-dapo-gepo` | Beta |
| `grpo-training` | Stable |
| `memory` | Beta |
| `offline-rl` | Experimental |
| `provider-control-plane` | Beta |
| `research-modules` | Experimental |
| `reward-learning-rlaif` | Beta |
| `rust-acceleration` | Beta |
| `sim-to-real` | Beta |
| `vapo` | Experimental |

The JSON manifest supplies each component's owner, documentation, tests,
retained evidence, limitations, graduation criteria, and stable-contract
references. Promotion is a reviewed evidence change, not a marketing decision.
The checker rejects a stable component without valid v1 references, and rejects
beta or experimental entries that omit limitations or measurable graduation
criteria.
