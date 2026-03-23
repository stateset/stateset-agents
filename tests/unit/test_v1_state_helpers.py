from stateset_agents.api.routers import v1


def test_resolve_idempotency_job_id_handles_string_and_dict_values():
    idempotency = {"plain": "job-1", "wrapped": {"job_id": "job-2"}}

    assert v1._resolve_idempotency_job_id(idempotency, "plain") == "job-1"
    assert v1._resolve_idempotency_job_id(idempotency, "wrapped") == "job-2"
    assert v1._resolve_idempotency_job_id(idempotency, "missing") is None


def test_resolve_idempotency_job_id_rejects_invalid_wrapped_values():
    idempotency = {"empty": {}, "bad": {"job_id": 123}, "none": {"job_id": None}}

    assert v1._resolve_idempotency_job_id(idempotency, "empty") is None
    assert v1._resolve_idempotency_job_id(idempotency, "bad") is None
    assert v1._resolve_idempotency_job_id(idempotency, "none") is None


def test_cleanup_timed_state_removes_expired_and_caps_to_max():
    values = {
        "old": {"status": "running"},
        "mid": {"status": "running"},
        "new": {"status": "running"},
    }
    metadata = {
        "old": 1000.0,
        "mid": 1004.0,
        "new": 1008.0,
    }
    now = 1010.0

    v1._cleanup_timed_state(
        values,
        metadata,
        ttl_seconds=5,
        now=now,
        max_items=2,
    )

    assert set(values.keys()) == {"mid", "new"}
    assert set(metadata.keys()) == {"mid", "new"}


def test_cleanup_timed_state_seeds_missing_metadata_for_legacy_values():
    values = {"legacy": {"status": "running"}}
    metadata = {}

    v1._cleanup_timed_state(
        values,
        metadata,
        ttl_seconds=60,
        now=500.0,
        max_items=10,
    )

    assert metadata == {"legacy": 500.0}
