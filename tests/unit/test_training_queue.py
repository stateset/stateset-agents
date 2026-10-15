"""The same durable queue contract runs against SQLite and optional PostgreSQL."""

import os
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import pytest

from stateset_agents.api.training_queue import TrainingQueue


@pytest.fixture(params=["sqlite", "postgres"])
def queue(request, tmp_path):
    if request.param == "postgres":
        dsn = os.environ.get("STATESET_TEST_QUEUE_DSN")
        if not dsn:
            pytest.skip(
                "Set STATESET_TEST_QUEUE_DSN to an isolated disposable database"
            )
        value = TrainingQueue(dsn, max_concurrent_jobs=2)
        with value._transaction() as connection:
            connection.execute("DELETE FROM stateset_training_jobs")
        return value
    return TrainingQueue(tmp_path / "queue.sqlite3", max_concurrent_jobs=2)


def submit(queue, owner="owner", key=None):
    return queue.submit({"num_episodes": 2}, owner, key)


def test_inputs_and_idempotency_survive_reopening(queue):
    first = submit(queue, key="request")
    reopened = TrainingQueue(queue.postgres_dsn or queue.path, max_concurrent_jobs=2)
    assert submit(reopened, key="request") == first
    with pytest.raises(ValueError, match="different"):
        reopened.submit({"num_episodes": 3}, "owner", "request")
    assert reopened.get(first, "someone-else") is None
    assert reopened.get(first, "owner")["status"] == "queued"


def test_claims_are_atomic_and_capacity_is_global(queue):
    for _ in range(8):
        submit(queue)
    with ThreadPoolExecutor(max_workers=8) as pool:
        claims = list(pool.map(lambda _: queue.claim(), range(8)))
    claimed = [item for item in claims if item is not None]
    assert len(claimed) == 2 and len({item[0] for item in claimed}) == 2
    with pytest.raises(ValueError, match="capacity"):
        TrainingQueue(queue.postgres_dsn or queue.path, max_concurrent_jobs=1)


def test_cancellation_and_expired_worker_writes_are_fenced(queue):
    queued = submit(queue)
    assert not queue.cancel(queued, "someone-else")
    assert queue.cancel(queued, "owner")
    assert queue.get(queued)["status"] == "cancelled"
    running = submit(queue)
    job_id, token, snapshot = queue.claim()
    assert job_id == running
    assert queue.cancel(job_id)
    assert queue.heartbeat(job_id, token)
    with queue._transaction() as connection:
        connection.execute(
            "UPDATE stateset_training_jobs SET lease_until=0 WHERE id=?", (job_id,)
        )
    assert queue.claim() is None
    assert queue.get(job_id)["status"] == "failed"
    with pytest.raises(RuntimeError, match="lease"):
        queue.heartbeat(job_id, token)
    snapshot.update(status="completed", completed_at=datetime.utcnow())
    with pytest.raises(RuntimeError, match="lease"):
        queue.finish(job_id, token, snapshot)


def test_late_cancellation_cannot_be_overwritten_by_success(queue):
    submit(queue)
    job_id, token, snapshot = queue.claim()
    queue.cancel(job_id)
    snapshot.update(status="completed", completed_at=datetime.utcnow())
    queue.finish(job_id, token, snapshot)
    assert queue.get(job_id)["status"] == "cancelled"
