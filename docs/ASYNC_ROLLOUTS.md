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

## Evidence boundary

These components establish deterministic in-process scheduling, publication
ordering, failure propagation, and staleness semantics. They do **not** by
themselves prove multi-node throughput, remote weight synchronization, crash
recovery, or learning-quality parity. StateSet will only claim those properties
after retained multi-node GPU runs exercise the same policy-version and counter
contract.
