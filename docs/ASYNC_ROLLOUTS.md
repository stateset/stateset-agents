# Asynchronous rollout control plane

StateSet's native async rollout coordinator decouples rollout producers from a
learner without hiding policy staleness. It is a transport-neutral foundation
for local tasks, Ray actors, Kubernetes workers, or provider-hosted rollout
services.

The coordinator enforces six invariants:

1. Every rollout identifies the exact policy version that sampled it.
2. Every rollout retains sampler token log-probabilities for importance
   correction.
3. Future-policy samples fail closed instead of entering the learner queue.
4. Samples beyond `max_policy_lag` are discarded before training.
5. A bounded queue applies backpressure so producers cannot exhaust memory.
6. Rollout IDs remain deduplicated across consumed batches and checkpoint
   restore, preventing producer retries from causing duplicate updates.

## Minimal producer/learner loop

```python
import asyncio

from stateset_agents.training import (
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
    RolloutRecord,
    compute_importance_weights,
)


async def main() -> None:
    coordinator = AsyncRolloutCoordinator(
        AsyncRolloutConfig(
            queue_capacity=128,
            min_batch_size=8,
            max_batch_size=32,
            max_policy_lag=1,
            max_importance_weight=2.0,
        )
    )

    # A rollout worker snapshots learner policy v0, samples, and retains the
    # sampler log-probability of every optimized token.
    await coordinator.submit(
        RolloutRecord(
            rollout_id="worker-3/request-17/sample-2",
            policy_version=0,
            sampler_log_probs=(-0.42, -0.31, -0.77),
            payload={
                "input_ids": [101, 202, 303],
                "advantages": [0.8, 0.8, 0.8],
            },
        )
    )

    batch = await coordinator.next_batch(min_size=1)

    # After the learner forward pass, correct each token against its actual
    # behavior policy. Ratios are calculated in log space and clipped to
    # [1 / max_importance_weight, max_importance_weight].
    weights = compute_importance_weights(
        learner_log_probs=(-0.40, -0.35, -0.70),
        sampler_log_probs=batch.records[0].sampler_log_probs,
        max_weight=coordinator.config.max_importance_weight,
    )
    assert len(weights) == 3

    # Publish the completed optimizer update. Queued v0 samples remain valid
    # at lag 1; another update will evict them before they can train.
    await coordinator.advance_policy()
    await coordinator.close()


asyncio.run(main())
```

## Managed producer/learner runtime

`AsyncRolloutRuntime` turns the coordinator into a bounded executable loop. It
starts independent producer tasks, serializes learner updates, and publishes
each completed policy snapshot before exposing its version to producers.

```python
from stateset_agents.training import (
    AsyncRolloutConfig,
    AsyncRolloutCoordinator,
    AsyncRolloutRuntime,
    AsyncRolloutRuntimeConfig,
    RolloutBatch,
    RolloutRecord,
)


async def produce(worker_id: int, policy_version: int) -> RolloutRecord:
    sample = await sample_with_policy(worker_id, policy_version)
    return RolloutRecord(
        rollout_id=sample.id,
        policy_version=policy_version,
        sampler_log_probs=tuple(sample.log_probs),
        payload=sample.payload,
    )


async def learn(batch: RolloutBatch) -> dict[str, float]:
    loss = await optimizer_step(batch)
    return {"loss": loss}


async def publish(policy_version: int) -> None:
    await rollout_service.publish_weights(policy_version)


coordinator = AsyncRolloutCoordinator(AsyncRolloutConfig(min_batch_size=8))
runtime = AsyncRolloutRuntime(
    coordinator=coordinator,
    producer=produce,
    learner_step=learn,
    publish_policy=publish,
    config=AsyncRolloutRuntimeConfig(producer_count=8, max_updates=100),
)
result = await runtime.run()
```

Producer results must match the requested policy version. Worker exceptions,
invalid learner metrics, publication failures, and bounded-wait timeouts fail
the run and close its coordinator. Internal waiters are cancellation-safe, and
even a producer callback that delays cancellation cannot hold the caller beyond
the configured shutdown deadline. The initial policy is published before any
producer starts, and a newer version becomes visible only after its publication
callback succeeds. `AsyncRolloutRunResult` retains per-update learner metrics
and final coordinator counters for the evidence bundle.

## Backpressure and shutdown

`submit(..., timeout_seconds=N)` and `next_batch(..., timeout_seconds=N)` use
bounded waits and raise `AsyncRolloutTimeout` when their deadline expires.
Closing a coordinator immediately releases blocked producers. Consumers may
drain one final partial batch, after which `next_batch()` raises
`AsyncRolloutClosed`.

`coordinator.stats()` exposes submitted, consumed, stale-drop, future-reject,
duplicate-reject, queue-depth, and maximum-observed-lag counters without
exposing rollout payloads. These counters should be attached to training
evidence and alerts. `await coordinator.state_dict()` captures the queue,
policy version, counters, and deduplication ledger for restart; restore it with
`AsyncRolloutCoordinator.from_state_dict(...)` through the same checkpoint
mechanism used for learner and optimizer state.

## Remote worker control plane

`DistributedRolloutControlPlane` adds the worker lifecycle that a remote
transport needs without coupling correctness to HTTP, Ray, Kubernetes, or a
specific GPU provider. A registration returns a renewable `WorkerLease` with a
generation, deadline, and exact policy assignment. Re-registering the same
worker fences its prior generation, so a delayed process cannot submit after a
replacement takes ownership.

```python
import time

from stateset_agents.training import (
    DistributedRolloutConfig,
    DistributedRolloutControlPlane,
    PolicyArtifact,
)

control = DistributedRolloutControlPlane(
    coordinator=coordinator,
    config=DistributedRolloutConfig(
        lease_ttl_seconds=30,
        max_workers=1_024,
        worker_history_capacity=1_000_000,
        policy_artifact_capacity=64,
        require_policy_artifact=True,
    ),
)

await control.register_initial_policy_artifact(
    PolicyArtifact(
        policy_version=0,
        uri="s3://my-policy-bucket/run-42/policy-0.safetensors",
        sha256="<64 lowercase hexadecimal characters>",
        size_bytes=12_345_678,
        published_at=time.time(),
    )
)

lease = await control.register("run-42/worker-3")
lease_artifact = await control.policy_artifact(lease.policy_version)
assert lease_artifact is not None

# A transport should authenticate this call before delegating to the control
# plane. The heartbeat renews the lease and returns the current assignment.
lease = await control.heartbeat(lease.worker_id, lease.lease_id)
sample = await sample_with_policy(lease.policy_version)
await control.submit(
    lease.worker_id,
    lease.lease_id,
    RolloutRecord(
        rollout_id=sample.id,
        policy_version=lease.policy_version,
        sampler_log_probs=tuple(sample.log_probs),
        payload=sample.payload,
        policy_artifact_sha256=lease_artifact.sha256,
    ),
)
```

Admission checks both the active worker generation and the assigned policy
snapshot before the coordinator applies its usual lag, backpressure, and
deduplication rules. Both live capacity and retained generation history are
bounded to prevent identity churn from creating unbounded controller state.
`health()` returns sorted, lease-ID-free worker status for
metrics endpoints; `stats()` distinguishes unknown, expired, stale-generation,
and wrong-policy rejections. `state_dict()` checkpoints leases, generation
fences, counters, and the coordinator atomically from the control plane's point
of view. Restore discards leases whose wall-clock deadline passed while the
controller was unavailable, while preserving their generation fence.

Lease IDs prevent stale process generations from acting as the current worker;
they are not authentication credentials. Checkpoint files contain active lease
IDs and must receive the same access controls as optimizer and model state.

## Content-addressed weight synchronization

`PolicyArtifact` binds every policy version to an immutable URI, exact byte
count, SHA-256 digest, and publication timestamp. Artifact URIs reject embedded
credentials, query strings, and fragments so signed secrets cannot leak into
checkpoints, health responses, or evidence bundles. Supported descriptor
schemes are `https`, `s3`, `gs`, `az`, and absolute `file` URIs.

After an optimizer step, upload and durably commit the weights before calling:

```python
await control.publish_policy_artifact(
    PolicyArtifact(
        policy_version=coordinator.current_policy_version + 1,
        uri="s3://my-policy-bucket/run-42/policy-1.safetensors",
        sha256=uploaded_sha256,
        size_bytes=uploaded_size,
        published_at=time.time(),
    )
)
```

This records the artifact before advancing the coordinator while holding the
same condition used by worker heartbeats. A worker cannot observe policy
version N without also receiving N's artifact descriptor. Failed or cancelled
advancement rolls back the staged descriptor. Distributed publishers must use
this method instead of calling `coordinator.advance_policy()` directly.

Workers download the assigned URI, call
`verify_policy_artifact(local_path, assignment)`, load the weights, and include
`policy_artifact_sha256` in every `RolloutRecord`. Artifact-required control
planes reject rollouts that omit the digest or claim different bytes. Artifact
history is bounded and checkpointed alongside leases and the rollout queue.

## Authenticated HTTP transport

The FastAPI gateway mounts a typed transport at `/api/v1/rollouts`. A training
process owns the coordinator and attaches its control plane to the application;
the gateway intentionally returns `503` instead of silently creating a queue
that is disconnected from the learner.

```python
from stateset_agents.api import attach_distributed_rollout_control_plane, create_app

app = create_app()
attach_distributed_rollout_control_plane(app, control)
```

Configure a dedicated worker credential and keep global authentication enabled
in deployed environments:

```bash
export API_REQUIRE_AUTH=true
export API_KEYS='replace-with-a-long-random-key:rollout_worker,replace-with-long-admin-key:admin'
export API_MAX_REQUEST_SIZE_MB=10
uvicorn my_training_gateway:app --host 0.0.0.0 --port 8000
```

The rollout routes require explicit API-key or JWT authentication even when a
development gateway sets `API_REQUIRE_AUTH=false`. Roles `rollout_worker`,
`trainer`, and `admin` may operate workers; only `admin` may inspect fleet-wide
health and counters. Each external worker ID is namespaced to an opaque
fingerprint of the authenticated principal, preventing one credential from
renewing, submitting through, or unregistering another credential's worker.

| Method | Route | Purpose |
|---|---|---|
| `POST` | `/api/v1/rollouts/workers/{id}/register` | Create or replace a worker generation and fetch its weights |
| `POST` | `/api/v1/rollouts/workers/{id}/heartbeat` | Renew its lease and fetch exact version, URI, size, and digest |
| `POST` | `/api/v1/rollouts/workers/{id}/submit` | Submit one typed rollout through all admission gates |
| `DELETE` | `/api/v1/rollouts/workers/{id}` | Release the current generation |
| `GET` | `/api/v1/rollouts/workers` | Read lease-ID-free fleet health as an administrator |
| `GET` | `/api/v1/rollouts/stats` | Read lifecycle and rejection counters as an administrator |

The gateway maps expired leases to `410`, fenced or wrong-policy submissions to
`409`, capacity exhaustion to `429`, and closed/backpressured queues to `503`.
The global request-size middleware enforces `API_MAX_REQUEST_SIZE_MB` against
both declared `Content-Length` and chunked bodies before JSON parsing. Deploying
TLS and network-level denial-of-service protection remains the responsibility
of the ingress or service mesh.

## Evidence boundary

These components establish deterministic scheduling, artifact-before-version
publication ordering, content-addressed weight assignments, failure
propagation, staleness semantics, remote-worker fencing, lease expiry, and
control-plane checkpoint recovery. They do **not** by themselves provide a
non-HTTP transport or prove measured multi-node throughput, worker download
latency, process-level crash recovery, or learning-quality parity. StateSet
will only claim those properties after retained multi-node GPU runs exercise
the same policy-version, artifact, lease, HTTP, and counter contract.
