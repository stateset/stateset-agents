# Kubernetes scaling billing evidence

CoreWeave and Nebius cluster capacity can outlive an individual Kubernetes
Job, so pod resource requests and public list prices are not accepted as A+
billing evidence. Supply one `*.billing.json` envelope for every retained
scaling lifecycle record and retain the provider's raw billing export beside
it.

Start with `kubernetes_billing.example.billing.json`. Placeholder and zero
values are intentionally invalid. Each envelope must contain:

- the exact provider, harness commit, Kubernetes Job, namespace, GPU count,
  and seed from its lifecycle record;
- a provider export ID and a unique per-Job allocation ID;
- a UTC billing window that fully contains the recorded Job lifecycle;
- USD line items with globally unique provider line-item IDs, positive
  quantities, finite non-negative costs, and a total equal to their sum; and
- a bundle-relative raw export path plus its nonzero SHA-256 digest.

The raw export must be retained as UTF-8 CSV or JSON. Do not normalize or
rewrite it after download. The gate requires the Job, export, allocation, and
line-item identifiers to occur in those raw bytes; hashing an unrelated bill
cannot validate an operator-authored envelope. One raw export may back multiple Job
envelopes when the provider delivers a consolidated statement, but every
attributed allocation and line-item ID must remain unique so the matrix cannot
double-count one charge.

Pass the directory containing these envelopes as
`SCALING_BILLING_EVIDENCE`. The A+ gate requires exact coverage of all 12
topology/seed Jobs, re-hashes every raw export, verifies billing and lifecycle
windows, and reports total scaling cost and cost per measured optimizer step.

`status` may be `provider-exported` for a finalized provider usage export or
`settled` for an invoice. Estimates, Kubernetes request summaries, and copied
pricing-page values do not satisfy this contract.
