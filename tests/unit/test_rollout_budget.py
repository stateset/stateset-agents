"""Admission budgets must survive uncertainty and reject work before sampling."""

import asyncio
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from stateset_agents.evaluation.agent_runs import content_hash
from stateset_agents.remote.river_environment import river_environment_factory
from stateset_agents.remote.rollout_budget import (
    RolloutAdmissionBudget,
    RolloutBudgetExceeded,
)
from stateset_agents.training.river_refund import (
    campaign,
    execute_run,
    prepare_run,
    run_manifest,
)


def budget(path, *, create=False, limit=2048):
    return RolloutAdmissionBudget(
        path, limit=limit, trajectory_tokens=1024, fingerprint="run", create=create
    )


@pytest.mark.asyncio
async def test_native_evaluation_reserves_before_mock_sampling(tmp_path, monkeypatch):
    @dataclass
    class Checkpoint:
        path: str

    sampled = []

    class Engine:
        def __init__(self, model, **kwargs):
            self.factory = kwargs["env"]

        async def rollout(self, rows, **kwargs):
            for row in rows:
                env = self.factory()
                await env.reset(row)
                sampled.append(row["order_id"])
                trajectory = SimpleNamespace(
                    metrics={},
                    truncated="generated_tokens",
                    generated_tokens=5,
                    elapsed=0.1,
                )
                trajectory.reward = await env.on_truncated(
                    trajectory, row, "generated_tokens"
                )
                yield [trajectory]

    rl = SimpleNamespace(
        Env=object,
        Budget=lambda **kw: kw,
        Schedule=lambda **kw: kw,
        GroupCompletion=lambda **kw: kw,
        RolloutEngine=Engine,
        CheckpointSampler=lambda *args, **kw: SimpleNamespace(**kw),
    )
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=rl, Checkpoint=Checkpoint)
    )
    args = SimpleNamespace(
        output=tmp_path,
        base_model="base",
        seed=42,
        steps=2,
        concurrency=8,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=None,
        evaluate_only=True,
        dry_run=True,
        rollout_token_budget=1024,
    )
    splits = {"test": [{"order_id": "A", "amount_cents": 100, "eligible": True}]}
    prepare_run(args, splits)
    args.dry_run = False
    model = SimpleNamespace(save_weights=lambda *args, **kw: Checkpoint("river://base"))
    renderer = SimpleNamespace(tokenizer=object())
    await campaign(model, object(), renderer, args, splits)
    report = json.loads((tmp_path / "test_results.json").read_text())
    assert report["rollout_budget"]["reserved_generated_tokens"] == 1024
    assert report["generated_tokens"] == 5
    # The short response does not refund its full allowance to a later engine.
    with pytest.raises(ValueError, match="sealed"):
        await campaign(model, object(), renderer, args, splits)
    assert sampled == ["A"]


def test_reservations_survive_restart_and_never_refund(tmp_path):
    path = tmp_path / "budget.json"
    first = budget(path, create=True)
    first.reserve()
    restarted = budget(path)
    restarted.reserve()
    assert restarted.snapshot()["reserved_generated_tokens"] == 2048
    with pytest.raises(RolloutBudgetExceeded):
        restarted.reserve()
    with pytest.raises(RolloutBudgetExceeded):
        restarted.ensure_available()
    with pytest.raises(ValueError, match="reset"):
        budget(path, create=True)


def test_concurrent_admission_never_oversubscribes(tmp_path):
    ledger = budget(tmp_path / "budget.json", create=True, limit=8 * 1024)

    def attempt(_):
        try:
            ledger.reserve()
        except RolloutBudgetExceeded:
            return False
        return True

    with ThreadPoolExecutor(max_workers=16) as pool:
        assert sum(pool.map(attempt, range(64))) == 8
    assert ledger.snapshot()["admitted_trajectories"] == 8


@pytest.mark.parametrize(
    "field,value",
    [
        ("fingerprint", "other"),
        ("limit", 4096),
        ("trajectory_tokens", 512),
        ("admitted_trajectories", -1),
        ("admitted_trajectories", True),
        ("reserved_generated_tokens", 1),
        ("reserved_generated_tokens", 3072),
    ],
)
def test_corrupt_or_rebound_budget_fails_closed(tmp_path, field, value):
    path = tmp_path / "budget.json"
    ledger = budget(path, create=True)
    state = ledger.snapshot()
    state[field] = value
    path.write_text(json.dumps(state))
    with pytest.raises(ValueError):
        ledger.reserve()


def test_deleted_state_is_not_recreated(tmp_path):
    path = tmp_path / "budget.json"
    ledger = budget(path, create=True)
    ledger.reserve()
    path.unlink()
    with pytest.raises(FileNotFoundError):
        ledger.reserve()
    assert not path.exists()


@pytest.mark.parametrize("limit", [True, 0, 1023, 1024.0])
def test_invalid_budget_rejected_without_writing(tmp_path, limit):
    path = tmp_path / "budget.json"
    with pytest.raises(ValueError):
        budget(path, create=True, limit=limit)
    assert not path.exists()


def test_admission_precedes_reset_and_retains_uncertain_work(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=SimpleNamespace(Env=object))
    )
    ledger = budget(tmp_path / "budget.json", create=True)
    resets = []

    class BrokenEnvironment:
        async def reset(self, row):
            resets.append(row)
            raise ConnectionError("uncertain environment operation")

    factory = river_environment_factory(BrokenEnvironment, before_reset=ledger.reserve)
    for _ in range(2):
        with pytest.raises(ConnectionError):
            asyncio.run(factory().reset({"case": 1}))
    with pytest.raises(RolloutBudgetExceeded):
        asyncio.run(factory().reset({"case": 1}))
    assert len(resets) == 2
    assert ledger.snapshot()["reserved_generated_tokens"] == 2048


def test_persistence_failure_prevents_environment_work(tmp_path, monkeypatch):
    monkeypatch.setitem(
        sys.modules, "river_client", SimpleNamespace(rl=SimpleNamespace(Env=object))
    )
    ledger = budget(tmp_path / "budget.json", create=True)

    def fail_write(*args):
        raise OSError("disk full")

    monkeypatch.setattr("stateset_agents.remote.rollout_budget.atomic_json", fail_write)

    class Environment:
        async def reset(self, row):
            pytest.fail("Admission must persist before reset")

    factory = river_environment_factory(Environment, before_reset=ledger.reserve)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(factory().reset({}))
    assert ledger.snapshot()["admitted_trajectories"] == 0


def test_native_preflight_preserves_budget_and_rejects_before_provider_import(
    tmp_path, monkeypatch
):
    args = SimpleNamespace(
        output=tmp_path,
        base_model="base",
        seed=42,
        steps=2,
        concurrency=8,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=None,
        evaluate_only=True,
        dry_run=True,
        rollout_token_budget=1024,
    )
    splits = {"test": [{"order_id": "test-A", "amount_cents": 100, "eligible": True}]}
    prepare_run(args, splits)
    ledger = RolloutAdmissionBudget(
        tmp_path / "rollout_budget.json",
        limit=1024,
        trajectory_tokens=1024,
        fingerprint=content_hash(run_manifest(args, splits)),
    )
    ledger.reserve()
    args.dry_run = False
    prepare_run(args, splits)
    monkeypatch.setitem(sys.modules, "river_client", None)
    with pytest.raises(RolloutBudgetExceeded):
        execute_run(args, splits)
    (tmp_path / "rollout_budget.json").unlink()
    with pytest.raises(FileNotFoundError):
        prepare_run(args, splits)
