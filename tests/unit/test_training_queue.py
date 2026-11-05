"""The same durable queue contract runs against SQLite and optional PostgreSQL."""

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from datetime import datetime
from types import SimpleNamespace

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


@pytest.mark.parametrize("operation", ["claim", "heartbeat"])
@pytest.mark.parametrize(
    "duration",
    [True, False, None, "60", 0, -1, 3601, 10**400, float("inf"), float("nan")],
)
def test_lease_duration_is_validated_without_mutating_jobs(queue, operation, duration):
    job_id = submit(queue)
    if operation == "heartbeat":
        _, token, _ = queue.claim()
    before = queue.get(job_id)
    with pytest.raises(ValueError, match="lease_seconds"):
        if operation == "heartbeat":
            queue.heartbeat(job_id, token, lease_seconds=duration)
        else:
            queue.claim(lease_seconds=duration)
    assert queue.get(job_id) == before


def _advance_clock_after_lock(queue, monkeypatch, clock, after):
    """Model time spent waiting for the transaction's exclusive lock."""
    transaction = queue._transaction

    @contextmanager
    def delayed_transaction():
        with transaction() as connection:
            clock[0] = after
            yield connection

    monkeypatch.setattr(queue, "_transaction", delayed_transaction)


def test_claim_lease_starts_after_acquiring_database_lock(tmp_path, monkeypatch):
    import stateset_agents.api.training_queue as module

    value = TrainingQueue(tmp_path / "queue.sqlite3")
    clock = [1000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: clock[0]))
    job_id = submit(value)
    _advance_clock_after_lock(value, monkeypatch, clock, after=1100.0)
    assert value.claim(lease_seconds=60)[0] == job_id
    with value._connection() as connection:
        row = connection.execute(
            "SELECT lease_until FROM stateset_training_jobs WHERE id=?", (job_id,)
        ).fetchone()
    assert row["lease_until"] == 1160.0


def test_heartbeat_cannot_revive_a_lease_that_expired_while_waiting_for_lock(
    tmp_path, monkeypatch
):
    import stateset_agents.api.training_queue as module

    value = TrainingQueue(tmp_path / "queue.sqlite3")
    clock = [1000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: clock[0]))
    job_id = submit(value)
    _, token, _ = value.claim(lease_seconds=60)
    _advance_clock_after_lock(value, monkeypatch, clock, after=1061.0)
    with pytest.raises(RuntimeError, match="lease"):
        value.heartbeat(job_id, token)
    assert value.claim() is None
    assert value.get(job_id)["status"] == "failed"


def test_lease_is_expired_at_its_deadline(tmp_path, monkeypatch):
    import stateset_agents.api.training_queue as module

    value = TrainingQueue(tmp_path / "queue.sqlite3")
    clock = [1000.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(time=lambda: clock[0]))
    job_id = submit(value)
    _, token, snapshot = value.claim(lease_seconds=60)
    clock[0] = 1060.0
    with pytest.raises(RuntimeError, match="lease"):
        value.heartbeat(job_id, token)
    snapshot.update(status="completed", completed_at=datetime.utcnow())
    with pytest.raises(RuntimeError, match="lease"):
        value.finish(job_id, token, snapshot)
    assert value.claim() is None
    assert value.get(job_id)["status"] == "failed"


def test_postgres_uses_database_clock_instead_of_worker_host_clock(queue, monkeypatch):
    if queue.postgres_dsn is None:
        pytest.skip("PostgreSQL clock contract")
    import stateset_agents.api.training_queue as module

    def forbidden_host_clock():
        raise AssertionError("PostgreSQL lease decisions must use the database clock")

    monkeypatch.setattr(module, "time", SimpleNamespace(time=forbidden_host_clock))
    job_id = submit(queue)
    claimed, token, snapshot = queue.claim()
    assert claimed == job_id
    assert not queue.heartbeat(job_id, token)
    snapshot.update(status="completed", completed_at=datetime.utcnow())
    queue.finish(job_id, token, snapshot)
    assert queue.get(job_id)["status"] == "completed"


def test_page_filters_owners_and_status_before_decoding_and_orders_ties(
    queue, monkeypatch
):
    visible = [submit(queue, owner="owner") for _ in range(3)]
    visible.append(submit(queue, owner=None))
    cancelled = visible.pop()
    queue.cancel(cancelled)
    foreign = submit(queue, owner="someone-else")
    with queue._transaction() as connection:
        connection.execute("UPDATE stateset_training_jobs SET created=100")
        # A different owner's corrupt payload must not be loaded by this listing.
        connection.execute(
            "UPDATE stateset_training_jobs SET snapshot='invalid-json' WHERE id=?",
            (foreign,),
        )
    decoded = []
    original_decode = queue._decode

    def decode(value):
        decoded.append(value)
        return original_decode(value)

    monkeypatch.setattr(queue, "_decode", decode)
    items, total = queue.page(owner="owner", status="queued", limit=1, offset=1)
    assert (
        total == 3 and [item[0] for item in items] == sorted(visible, reverse=True)[1:2]
    )
    assert len(decoded) == 1
    items[0][1]["metrics"]["caller-change"] = True
    assert not queue.get(items[0][0])["metrics"]


@pytest.mark.parametrize("status", [None, "", "queued", "cancelled", "' OR 1=1 --"])
def test_page_keeps_hostile_owner_and_status_values_as_bound_parameters(queue, status):
    hostile_owner = "' OR 1=1; DROP TABLE stateset_training_jobs; --"
    owned = submit(queue, owner=hostile_owner)
    public = submit(queue, owner=None)
    foreign = submit(queue, owner="foreign")
    queue.cancel(public)
    expected = (
        {owned, public}
        if not status
        else (
            {owned}
            if status == "queued"
            else {public} if status == "cancelled" else set()
        )
    )
    rows, total = queue.page(owner=hostile_owner, status=status)
    assert {job_id for job_id, _ in rows} == expected
    assert total == len(expected)
    all_rows, all_total = queue.page()
    assert all_total == 3 and {job_id for job_id, _ in all_rows} == {
        owned,
        public,
        foreign,
    }


def test_page_counts_public_jobs_and_supports_arbitrarily_large_offsets(queue):
    public = submit(queue, owner=None)
    submit(queue, owner="someone-else")
    items, total = queue.page(owner="owner")
    assert total == 1 and items[0][0] == public
    assert queue.page(owner="owner", offset=10**100) == ([], 1)
    items, total = queue.page()
    assert len(items) == total == 2


@pytest.mark.parametrize(
    "options",
    [{"limit": 0}, {"limit": 101}, {"limit": True}, {"offset": -1}, {"offset": False}],
)
def test_page_rejects_invalid_bounds(queue, options):
    with pytest.raises(ValueError):
        queue.page(**options)


def test_page_count_and_rows_share_a_snapshot_while_another_replica_writes(
    queue, monkeypatch
):
    first = submit(queue)
    other = TrainingQueue(queue.postgres_dsn or queue.path, max_concurrent_jobs=2)
    original_connection = queue._connection
    changed = []

    class Connection:
        def __init__(self, connection):
            self.connection = connection

        def execute(self, statement, parameters=()):
            if statement.startswith("SELECT id,snapshot") and not changed:
                changed.append(True)
                other.cancel(first)
                submit(other)
                submit(other)
            return self.connection.execute(statement, parameters)

        def rollback(self):
            self.connection.rollback()

    @contextmanager
    def connection():
        with original_connection() as value:
            yield Connection(value)

    monkeypatch.setattr(queue, "_connection", connection)
    items, total = queue.page(owner="owner", status="queued")
    assert total == 1 and items[0][0] == first
    assert items[0][1]["status"] == "queued"
    assert queue.get(first)["status"] == "cancelled"


def test_pending_capacity_is_atomic_and_idempotent_retries_still_work(queue):
    from stateset_agents.api.training_queue import TrainingQueueFullError

    first = submit(queue, key="accepted")
    with queue._transaction() as connection:
        for index in range(998):
            connection.execute(
                "INSERT INTO stateset_training_jobs(id,owner,fingerprint,status,snapshot,created) "
                "SELECT ?,owner,fingerprint,status,snapshot,created FROM stateset_training_jobs WHERE id=?",
                (f"prefill-{index}", first),
            )

    def attempt(index):
        try:
            return submit(queue, key=f"concurrent-{index}")
        except TrainingQueueFullError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(attempt, range(4)))
    assert sum(isinstance(result, str) for result in results) == 1
    assert sum(isinstance(result, TrainingQueueFullError) for result in results) == 3
    assert submit(queue, key="accepted") == first
