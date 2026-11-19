# Private improvement workflow

The `stateset-agents workflow` commands connect dataset intake, River training,
paired evaluation, staging approval, deployment monitoring, and rollback. The SDK
entry point is `stateset_agents.improvement`.

**Customer data belongs outside the public repository.** The workflow rejects
customer input, review, dataset, experiment, deployment, and payment journal paths
inside Git checkouts, including linked worktrees. Ignoring a directory in Git is
not sufficient. Generated workspace directories use mode `0700`, and workflow
records use `0600`. Symbolic links and parent traversal are rejected. This version
requires a POSIX filesystem with advisory locks and atomic hard links.

Use a private directory such as `$HOME/.local/share/stateset-private`. Restrict the
original input files and backups too. The workflow does not change their existing
permissions or provide encryption. Do not attach these directories to a public PR,
CI artifact upload, issue, or support ticket. `scripts/check_private_workflow_data.py`
rejects tracked JSON explicitly marked as customer/private workflow data; it is an
accidental-commit guard, not a complete PII detector.

CLI status and comparison output contains fixed labels, booleans and aggregate
counts. Detailed reports, reviews, identities, traces, checkpoints, and approvals
stay private. The public evidence catalog contains only explicitly selected,
already-public synthetic studies. No command scans private workspaces for export.

## Try the workflow without spending money

Install the ordinary package and generate fictional examples outside the checkout:

```bash
stateset-agents workflow synthetic-fixture /tmp/stateset-workflow-example
stateset-agents workflow evidence
```

The fixture executes a deterministic fictional refund policy locally. It produces
32 records per split, spanning eight scenario families. It makes no model/provider
calls and establishes no learning or customer-quality evidence.

Create a private JSON configuration file with an explicit parent checkpoint:

```json
{
  "base_model": "Qwen/Qwen3.5-9B",
  "parent_checkpoint": "river://YOUR-EXISTING-PARENT-CHECKPOINT",
  "seeds": [42],
  "min_cases": 32,
  "min_gain": 0.03,
  "epochs": 1
}
```

Then prepare the experiment:

```bash
stateset-agents workflow plan /tmp/stateset-workflow-run \
  /tmp/stateset-workflow-example/dataset /path/outside/git/config.json
stateset-agents workflow status /tmp/stateset-workflow-run
```

Planning makes no provider calls. It snapshots the dataset, package source hashes,
training seeds, evaluation settings, thresholds, and first-seed candidate selection.
Changing package sources invalidates further experiment execution. Run experiments
from an immutable installation. Existing completed stages are reused only after
verification; a sealed attempt with no result is an unknown outcome, not permission
to repeat work.

## Reviewed customer data

Do not put customer data into the synthetic tutorial or source control. Use a
separate private JSONL source and a private independent-review attestation.
Each customer record requires:

- `record_id`: a unique record identifier.
- `source_id`: the merchant or other grouping unit that must not cross partitions.
- `source_kind`: `customer`.
- `provenance_id`: a private reference to the approved source.
- `timestamp`: a timezone-aware ISO timestamp.
- `messages`: text chat messages with at least one assistant response.
- `evaluation_case`: a structured case with a `family` for held-out evaluation.

The review JSON requires schema version `1`, nonempty `reviewer`, `collector` and
`rubric_id`, distinct reviewer/collector identities, timezone-aware `reviewed_at`,
`approved_for_training: true`, `approved_for_evaluation: true`, `source_sha256` of
the exact input bytes, and `record_ids` covering every input record exactly once.
These are local attestations, not independent verification of identity or consent.

```bash
stateset-agents workflow dataset /private/customer-source.jsonl /private/dataset \
  --review /private/review.json \
  --validation-after 2026-08-01T00:00:00Z \
  --test-after 2026-09-01T00:00:00Z
```

Existing curation redacts known contact/credential patterns, deduplicates messages,
and assigns an entire source group to its newest temporal partition. This can leave
a split empty; intake rejects that result. Choose source groups and periods that
support meaningful isolation. Redaction does not make customer data publishable.
Synthetic inputs require `source_kind: synthetic` on every row and cannot carry a
customer review attestation.

**The built-in River evaluator uses a fictional refund policy.** It does not
establish compatibility with an actual merchant's policy, authorization rules,
payments, or conversations. Its cases must satisfy `RefundPolicyEnvironment`,
`record_id` must equal `evaluation_case.order_id`, and all training/held-out cases
must carry their facts so cross-partition fact overlap can be rejected. Customer
quality needs a reviewed merchant-specific `ExperimentBackend`, a versioned
`adapter_id`, and corresponding cases; that integration requires the private
source and policy owner. `evaluation_scope` preserves this distinction.

## Execute a River experiment

River SDK/model access and `RIVER_API_KEY` must already be available. The native
rollout engine may impose a newer Python requirement than the base package; its
runtime preflight checks this before sampling. The evaluator currently pins the
audited River 0.11.0 rollout-engine source hash and checks cross-run sampler seed
overlap; a changed SDK engine needs review before this adapter can sample. Never put the credential in a
configuration, command argument, or tracked file.

Training uploads the private training split to River. Evaluation uploads prompts.
Both CLI commands require explicit billing and provider-upload acknowledgment:

```bash
stateset-agents workflow train /private/experiment \
  --acknowledge-river-billing --allow-provider-upload
stateset-agents workflow evaluate /private/experiment --split validation \
  --acknowledge-river-billing --allow-provider-upload
stateset-agents workflow compare /private/experiment --split validation
stateset-agents workflow evaluate /private/experiment --split test \
  --acknowledge-river-billing --allow-provider-upload
stateset-agents workflow compare /private/experiment
stateset-agents workflow approve /private/experiment --reviewer operator-name
```

These examples authorize execution only when an operator runs them. Planning or
installing the package never launches paid work. Token/step counts and per-request
timeouts do **not** enforce a dollar ceiling or a whole-job shutdown deadline.
River billed costs remain `null` until independently reconciled.

Each seed starts from the specified parent weights with a fresh optimizer. SDK
submission retries and executor session restarts are disabled for this workflow.
An interrupted training attempt remains sealed; inspect retained checkpoint
artifacts and reconcile provider activity before preparing a new experiment. This
is not optimizer-state resume, and a saved hosted pointer does not establish it.

Native evaluation admits each trajectory against a conservative output-token
allowance and independently replays tool traces before accepting a report.
Validation must pass before test sampling. A test-use claim in the original
dataset directory prevents another experiment from sampling that same dataset
through this API. Keep that directory and ledger; copying or manually inspecting
data is outside this local guarantee. Both arms must match exact planned
cases, families, model, settings, source identity, seeds, and checkpoint selection.
Comparison requires improvement in every seed, the planned mean gain, no family
regression, and zero candidate policy violations or truncations. A tie does not
prove improvement. For a repeated-training study set at least six distinct seeds
and `require_significance: true`; the conditional one-sided seed sign test is an
engineering gate, not independent production certification.

Approval is **staging only**. River checkpoint URIs can be republished and do not
attest immutable weight bytes. The existing production promotion requirements
remain separate; this workflow cannot satisfy them with a pointer or synthetic
passing report.

## Deploy, monitor, and roll back

River dedicated deployments require a team key with deployment/model access.
They bill separately from training. First prepare a plan, identifying an existing
healthy deployment if rollback is required:

```bash
stateset-agents workflow prepare-deployment /private/deployment /private/experiment \
  --previous-deployment-id EXISTING_DEPLOYMENT_ID
stateset-agents workflow deployment deploy /private/deployment --acknowledge-river-billing
stateset-agents workflow deployment monitor /private/deployment
stateset-agents workflow deployment activate /private/deployment
```

Creation retains an idempotency key before the request. If the response is lost,
`deployment reconcile ... --acknowledge-river-billing` reuses that exact key.
Ordinary `deploy` refuses to resubmit an unknown outcome. Activation requires
phase `ready` and every desired replica ready; degraded and zero-replica states
are not healthy. The deployment uses one unified replica, not a selected GPU type.

Applications must obtain their endpoint from the local routing journal:

```python
from pathlib import Path
from stateset_agents.improvement import Deployment

base_url = Deployment(Path('/private/deployment')).endpoint()
```

This does not update an external load balancer. Applications must refresh the
endpoint after a route change. Monitoring is one observation per call; schedule it
with your service supervisor. Provider readiness is not an application-quality,
latency, or error-rate SLO check; maintain those checks in your service as well.

```bash
stateset-agents workflow deployment monitor /private/deployment --rollback-on-failure
stateset-agents workflow deployment rollback /private/deployment
stateset-agents workflow deployment stop /private/deployment
stateset-agents workflow deployment monitor /private/deployment
```

Rollback records the healthy previous endpoint before requesting candidate
scale-to-zero. A lost shutdown response reuses the same shutdown key. Stopping an
active candidate invalidates its local route. Shutdown acknowledgment means
requested, not stopped: monitor for `scaled_to_zero`. Cleanup remains available
if package source changes invalidate the experiment approval. No automatic
background shutdown or deployment billing cap is claimed.

## Deterministic merchant tools

`DurableRefundResolver` exposes only `resolve_refund(intent="resolve_refund")`.
A trusted `RefundBackend` owns authorization, order lookup, versioned deterministic
policy, atomic commit, and reconciliation. Facts use integer cents. Bind the
principal, order and operation key in trusted application code, never model output.

The resolver authorizes every request, including cached receipts. It durably seals
the exact facts/action before commit, serializes processes, and refuses another
commit after timeout, cancellation, or process death. `reconcile()` only queries the
existing operation. Receipts must bind the exact request hash and transaction ID.
The merchant backend must atomically revalidate authorization/facts/policy and
provide durable idempotency. A local journal cannot provide exactly-once payments
by itself. The existing `SyntheticRefundResolver` remains a fictional fixture.

## Evidence and support boundary

`workflow evidence --repository /path/to/checkout` verifies the historical public
artifact hashes. The catalog includes successful mechanics, pilot improvements,
failed quality comparisons, and the failed six-seed RL replication. Audit success,
experiment acceptance, source identity, source kind, and unknown costs remain
separate. Historical live evidence does not validate new deployment/recovery code.

The supported orchestration surface is `prepare_dataset`, `verify_dataset`,
`ExperimentConfig`, `Experiment`, `Deployment`, `DurableRefundResolver`, and
`evidence_catalog`. Custom provider/merchant adapters and native River execution
require their own live integration verification. Existing experimental trainers
and advanced CLI commands retain their previous status. Scripted tests establish
local contracts only; they do not replace live failure injection, multi-node
recovery, a long-duration soak, or reviewed merchant evaluation.

The Semgrep/MCP development-tool audit blocker documented in `SECURITY.md` remains
open. Do not bypass it with an unsupported dependency override or advisory waiver.
