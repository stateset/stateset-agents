"""Private workflow contracts, using synthetic data and scripted providers only."""

from __future__ import annotations

import asyncio
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents.cli_workflow import app
from stateset_agents.core.refund_tools import RefundOrderFacts, RefundResolutionRequest
from stateset_agents.evaluation.agent_runs import content_hash, evaluation_report
from stateset_agents.improvement import (
    Deployment,
    DurableRefundResolver,
    Experiment,
    ExperimentConfig,
    evidence_catalog,
    prepare_dataset,
    verify_dataset,
)
from stateset_agents.improvement.deployments import RiverDeploymentProvider
from stateset_agents.improvement.storage import (
    private_directory,
    private_path,
    read_json,
    write_json,
)


@pytest.fixture
def source(tmp_path):
    rows = [
        {
            "record_id": f"case-{i}",
            "source_id": f"merchant-{i}",
            "source_kind": "customer",
            "provenance_id": f"private-source-{i}",
            "timestamp": f"2026-0{i + 1}-01T00:00:00Z",
            "messages": [
                {"role": "user", "content": f"PRIVATE_CONVERSATION_{i}"},
                {"role": "assistant", "content": f"Private answer {i}"},
            ],
            "evaluation_case": {
                "order_id": f"case-{i}",
                "family": "refund",
                "index": i,
            },
        }
        for i in range(3)
    ]
    path = tmp_path / "source.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows))
    return path


def prepare(source, destination, **kwargs):
    return prepare_dataset(
        source,
        destination,
        validation_after="2026-02-01T00:00:00Z",
        test_after="2026-03-01T00:00:00Z",
        **kwargs,
    )


@pytest.fixture
def dataset(source, tmp_path):
    destination = tmp_path / "dataset"
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row in rows:
        row["source_kind"] = "synthetic"
    source.write_text("\n".join(json.dumps(row) for row in rows))
    prepare(source, destination, synthetic=True)
    return destination


class ScriptedBackend:
    evidence_mode = "scripted"
    adapter_id = "scripted-contract-v1"

    def __init__(self):
        self.trained = []
        self.evaluated = []
        self.fail_training = False
        self.candidate_success = True
        self.violation = 0

    def train(self, config, dataset, output, seed):
        self.trained.append(seed)
        if self.fail_training:
            raise TimeoutError("private provider response contains SECRET")
        return {
            "base_model": config.base_model,
            "checkpoint": f"river://candidate/{seed}",
        }

    def evaluate(self, config, cases, checkpoint, seed, split, implementation, output):
        self.evaluated.append((checkpoint, split))
        candidate = checkpoint != config.parent_checkpoint
        return evaluation_report(
            environment=config.environment,
            base_model=config.base_model,
            seed=seed,
            cases=cases,
            settings=config.evaluation,
            checkpoint={"path": checkpoint},
            outcomes=[
                {
                    "case_id": key,
                    "success": candidate
                    and self.candidate_success
                    and not self.violation,
                    "policy_violations": self.violation,
                    "tool_calls": 1,
                    "generated_tokens": 10,
                    "elapsed_seconds": 0.1,
                    "reward": int(candidate),
                    "truncated": None,
                }
                for key in cases
            ],
            case_families={key: case["family"] for key, case in cases.items()},
            implementation=implementation,
            split=split,
        )


@pytest.fixture
def experiment(dataset, tmp_path):
    return Experiment.create(
        tmp_path / "experiment",
        dataset,
        ExperimentConfig(
            base_model="scripted/model",
            parent_checkpoint="river://parent",
            evidence_mode="scripted",
            adapter_id="scripted-contract-v1",
            min_cases=1,
        ),
    )


def complete(experiment, backend=None):
    backend = backend or ScriptedBackend()
    experiment.train(backend)
    experiment.evaluate(backend)
    experiment.evaluate(backend, split="test")
    experiment.approve(reviewer="PRIVATE_REVIEWER")
    return backend


def test_private_workflow_end_to_end_and_content_free_status(experiment):
    backend = complete(experiment)
    assert experiment.compare()["passed"]
    assert experiment.verified_approval()["scope"] == "staging_only"
    assert not experiment.verified_approval()["production_quality_established"]
    experiment.train(backend)
    experiment.evaluate(backend, split="test")
    assert backend.trained == [42]
    assert len(backend.evaluated) == 4
    result = CliRunner().invoke(app, ["status", str(experiment.root)])
    assert result.exit_code == 0, result.output
    assert "PRIVATE_" not in result.output
    assert "river://" not in result.output
    assert str(experiment.root) not in result.output
    status = json.loads(result.output)
    assert status["completed_stages"] == 5
    assert status["cost_usd"] is None
    assert status["evidence_mode"] == "scripted"


@pytest.mark.parametrize("marker", ["directory", "worktree_file"])
def test_rejects_customer_artifacts_in_git_even_if_ignored(tmp_path, source, marker):
    repo = tmp_path / "public"
    repo.mkdir()
    if marker == "directory":
        (repo / ".git").mkdir()
        (repo / ".git" / "HEAD").write_text("ref: refs/heads/main")
    else:
        (repo / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(ValueError, match="outside Git"):
        prepare(source, repo / "ignored-data", synthetic=True)
    copied = repo / "source.jsonl"
    copied.write_bytes(source.read_bytes())
    with pytest.raises(ValueError, match="outside Git"):
        prepare(copied, tmp_path / "private-data")


def test_symlink_and_parent_traversal_are_rejected(tmp_path):
    target = tmp_path / "target"
    target.mkdir()
    link = tmp_path / "link"
    link.symlink_to(target, target_is_directory=True)
    with pytest.raises(ValueError, match="symbolic"):
        private_path(link / "data.json")
    with pytest.raises(ValueError, match="parent traversal"):
        private_path(tmp_path / "target" / ".." / "outside")


def test_owner_only_and_exclusive_artifacts(dataset):
    assert dataset.stat().st_mode & 0o077 == 0
    assert all(p.stat().st_mode & 0o077 == 0 for p in dataset.iterdir())
    with pytest.raises(FileExistsError):
        write_json(dataset / "dataset.json", {"overwrite": True})
    assert verify_dataset(dataset)["source_kind"] == "synthetic"


def test_customer_review_bound_to_exact_source_and_independent_reviewer(
    source, tmp_path
):
    review = {
        "schema_version": 1,
        "reviewer": "reviewer",
        "collector": "collector",
        "rubric_id": "merchant-policy-v1",
        "approved_for_training": True,
        "approved_for_evaluation": True,
        "reviewed_at": "2026-04-01T00:00:00Z",
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "record_ids": [f"case-{i}" for i in range(3)],
    }
    path = tmp_path / "review.json"
    path.write_text(json.dumps(review))
    manifest = prepare(source, tmp_path / "reviewed", review=path)
    assert manifest["review_status"] == "attested_independent_review"
    assert verify_dataset(tmp_path / "reviewed") == manifest
    review["reviewer"] = "collector"
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="independent"):
        prepare(source, tmp_path / "bad-review", review=path)
    review["reviewer"] = "reviewer"
    review["source_sha256"] = "0" * 64
    path.write_text(json.dumps(review))
    with pytest.raises(ValueError, match="source bytes"):
        prepare(source, tmp_path / "changed", review=path)


def test_customer_review_required(source, tmp_path):
    with pytest.raises(ValueError, match="attestation"):
        prepare(source, tmp_path / "unreviewed")


def test_duplicate_json_fields_rejected(source, tmp_path):
    source.write_text('{"record_id":"a","record_id":"b"}\n')
    with pytest.raises(ValueError, match="Duplicate"):
        prepare(source, tmp_path / "ambiguous", synthetic=True)


def test_dataset_tampering_invalidates_experiment(experiment):
    path = experiment.root / "dataset" / "test.json"
    path.write_text("[]")
    with pytest.raises(ValueError, match="changed"):
        experiment.verify()


def test_cross_partition_sources_rejected_even_with_rehashed_split(dataset):
    validation = read_json(dataset / "validation.json")
    test = read_json(dataset / "test.json")
    test[0]["source_group"] = validation[0]["source_group"]
    (dataset / "test.json").write_text(json.dumps(test))
    manifest = read_json(dataset / "dataset.json")
    manifest["split_sha256"]["test"] = content_hash(test)
    (dataset / "dataset.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="leakage"):
        verify_dataset(dataset)


def test_lost_training_acknowledgment_is_not_retried_after_restart(experiment):
    backend = ScriptedBackend()
    backend.fail_training = True
    with pytest.raises(TimeoutError):
        experiment.train(backend)
    backend.fail_training = False
    restarted = Experiment(experiment.root)
    with pytest.raises(FileExistsError):
        restarted.train(backend)
    assert backend.trained == [42]
    assert restarted.status()["uncertain_stages"] == 1


def test_source_change_blocks_further_training(experiment, monkeypatch):
    monkeypatch.setattr(
        "stateset_agents.improvement.workflow.package_implementation", lambda: {}
    )
    with pytest.raises(ValueError, match="sources changed"):
        experiment.train(ScriptedBackend())


def test_scripted_backend_cannot_supply_live_evidence(experiment):
    backend = ScriptedBackend()
    backend.evidence_mode = "live"
    with pytest.raises(ValueError, match="Backend"):
        experiment.train(backend)
    assert not backend.trained


def test_failed_validation_keeps_test_unopened(experiment):
    backend = ScriptedBackend()
    backend.candidate_success = False
    experiment.train(backend)
    experiment.evaluate(backend)
    with pytest.raises(ValueError, match="Validation gate failed"):
        experiment.evaluate(backend, split="test")
    assert all(split == "validation" for _, split in backend.evaluated)
    assert not list(experiment.root.glob("test-*"))


def test_changed_approval_cannot_change_selected_checkpoint(experiment):
    complete(experiment)
    path = experiment.root / "approval.json"
    approval = read_json(path)
    approval["checkpoint"] = "river://other"
    path.write_text(json.dumps(approval))
    with pytest.raises(ValueError, match="approval"):
        experiment.verified_approval()


def test_evidence_catalog_preserves_negative_replication_and_unknown_costs():
    catalog = evidence_catalog(repository=Path(__file__).resolve().parents[2])
    replication = next(
        e for e in catalog["entries"] if e["id"].endswith("river-recovery-replication")
    )
    assert replication["audit_passed"] is True
    assert replication["experiment_accepted"] is False
    assert all(
        e["source_kind"] == "synthetic" and e["cost_usd"] is None
        for e in catalog["entries"]
    )


class ScriptedDeployments:
    evidence_mode = "scripted"

    def __init__(self):
        self.creates = []
        self.stops = []
        self.lose_create = False
        self.lose_stop = False
        self.candidate_ready = True
        self.previous_ready = True

    def create(self, checkpoint, key):
        self.creates.append(key)
        if self.lose_create:
            self.lose_create = False
            raise TimeoutError("response lost after creation")
        return self.get("candidate")

    def get(self, identity):
        ready = self.candidate_ready if identity == "candidate" else self.previous_ready
        return {
            "id": identity,
            "checkpoint": (
                "river://candidate/42" if identity == "candidate" else "river://old"
            ),
            "endpoint": f"https://{identity}.example/v1",
            "ready": ready,
            "phase": "ready" if ready else "degraded",
        }

    def stop(self, identity, key):
        self.stops.append((identity, key))
        if self.lose_stop:
            self.lose_stop = False
            raise TimeoutError("stop response lost")


@pytest.fixture
def deployment(experiment, tmp_path):
    complete(experiment)
    return Deployment.prepare(
        tmp_path / "deployment", experiment, previous_deployment_id="previous"
    )


def test_deployment_reconciliation_reuses_provider_idempotency_key(deployment):
    provider = ScriptedDeployments()
    provider.lose_create = True
    with pytest.raises(TimeoutError):
        deployment.deploy(provider)
    with pytest.raises(ValueError, match="unknown"):
        Deployment(deployment.root).deploy(provider)
    Deployment(deployment.root).deploy(provider, reconcile=True)
    assert len(provider.creates) == 2
    assert provider.creates[0] == provider.creates[1]


def test_monitor_rolls_back_before_uncertain_stop_and_reuses_stop_key(deployment):
    provider = ScriptedDeployments()
    deployment.deploy(provider)
    deployment.activate(provider)
    assert deployment.endpoint() == "https://candidate.example/v1"
    provider.candidate_ready = False
    provider.lose_stop = True
    with pytest.raises(TimeoutError):
        deployment.monitor(provider, rollback_on_failure=True)
    assert deployment.endpoint() == "https://previous.example/v1"
    Deployment(deployment.root).rollback(provider)
    assert len(provider.stops) == 2
    assert provider.stops[0] == provider.stops[1]
    with pytest.raises(ValueError, match="shutdown"):
        deployment.activate(provider)


def test_unhealthy_previous_does_not_switch_route_or_stop_candidate(deployment):
    provider = ScriptedDeployments()
    deployment.deploy(provider)
    deployment.activate(provider)
    provider.previous_ready = False
    with pytest.raises(ValueError, match="Previous deployment"):
        deployment.rollback(provider)
    assert deployment.endpoint() == "https://candidate.example/v1"
    assert not provider.stops


def test_cleanup_remains_available_after_source_changes(deployment, monkeypatch):
    provider = ScriptedDeployments()
    deployment.deploy(provider)
    monkeypatch.setattr(
        "stateset_agents.improvement.workflow.package_implementation", lambda: {}
    )
    deployment.stop(provider)
    assert len(provider.stops) == 1


def test_river_readiness_requires_all_desired_replicas():
    deployment = SimpleNamespace(
        id="d",
        checkpoint="river://c",
        base_url="https://example/v1",
        phase="ready",
        replicas={"unified": SimpleNamespace(desired=2, ready=1)},
    )
    assert not RiverDeploymentProvider._snapshot(deployment)["ready"]
    deployment.replicas["unified"].ready = 2
    assert RiverDeploymentProvider._snapshot(deployment)["ready"]
    deployment.phase = "degraded"
    assert not RiverDeploymentProvider._snapshot(deployment)["ready"]


class TrustedBackend:
    policy_version = "fictional-fixture-v1"

    def __init__(self):
        self.commits = 0
        self.allowed = True
        self.lost_response = False
        self.receipt = None

    async def authorize(self, principal, order_id):
        if not self.allowed:
            raise PermissionError("authorization revoked")

    async def lookup(self, order_id):
        return RefundOrderFacts(order_id, 1099, 101, 2, 30, "delivered", False)

    def decide(self, facts):
        return RefundResolutionRequest(facts, "fixture").action()

    async def commit(self, request):
        self.commits += 1
        assert request["action"]["args"]["amount_cents"] == 998
        self.receipt = {
            "applied": True,
            "transaction_id": "tx",
            "request_sha256": content_hash(request),
        }
        if self.lost_response:
            raise TimeoutError("payment committed, response lost")
        return self.receipt

    async def reconcile(self, request):
        return self.receipt


def resolver(root, backend):
    return DurableRefundResolver(
        root,
        principal="principal",
        order_id="order",
        idempotency_key="logical-operation",
        backend=backend,
    )


@pytest.mark.asyncio
async def test_durable_refund_resolver_never_repeats_uncertain_commit(tmp_path):
    backend = TrustedBackend()
    backend.lost_response = True
    operation = resolver(tmp_path / "journal", backend)
    with pytest.raises(TimeoutError):
        await operation.resolve_refund(intent="resolve_refund")
    restarted = resolver(tmp_path / "journal", backend)
    with pytest.raises(ValueError, match="uncertain"):
        await restarted.resolve_refund(intent="resolve_refund")
    receipt = await restarted.reconcile()
    assert receipt["applied"]
    assert await restarted.resolve_refund(intent="resolve_refund") == receipt
    assert backend.commits == 1
    backend.allowed = False
    with pytest.raises(PermissionError):
        await restarted.resolve_refund(intent="resolve_refund")


@pytest.mark.asyncio
async def test_concurrent_refund_intents_share_one_receipt(tmp_path):
    backend = TrustedBackend()
    operation = resolver(tmp_path / "journal", backend)
    receipts = await asyncio.gather(
        *(operation.resolve_refund(intent="resolve_refund") for _ in range(5))
    )
    assert all(r == receipts[0] for r in receipts)
    assert backend.commits == 1
    assert set(operation.as_function().parameters["properties"]) == {"intent"}


@pytest.mark.asyncio
async def test_process_death_leaves_attempt_sealed_and_lock_released(tmp_path):
    root = private_directory(tmp_path / "journal")
    code = """
import asyncio, os, sys
from pathlib import Path
from stateset_agents.improvement.tools import DurableRefundResolver
from stateset_agents.core.refund_tools import RefundOrderFacts
class Backend:
    policy_version = 'fixture'
    async def authorize(self, *args): pass
    async def lookup(self, order): return RefundOrderFacts(order, 100, 0, 1, 30, 'delivered', False)
    def decide(self, facts): return {'tool': 'refund', 'args': {'amount_cents': 100}}
    async def commit(self, request): os._exit(17)
operation = DurableRefundResolver(Path(sys.argv[1]), principal='principal', order_id='order', idempotency_key='logical-operation', backend=Backend())
asyncio.run(operation.resolve_refund(intent='resolve_refund'))
"""
    process = subprocess.run(
        [sys.executable, "-c", code, str(root)], capture_output=True, timeout=30
    )
    assert process.returncode == 17, process.stderr
    backend = TrustedBackend()
    operation = resolver(root, backend)
    with pytest.raises(ValueError, match="uncertain"):
        await operation.resolve_refund(intent="resolve_refund")
    with pytest.raises(ValueError, match="not confirmed"):
        await operation.reconcile()
    assert backend.commits == 0


def test_cli_requires_explicit_paid_execution(experiment):
    result = CliRunner().invoke(app, ["train", str(experiment.root)])
    assert result.exit_code == 2
    assert "--acknowledge-river-billing" in result.output
    assert not list(experiment.root.glob("train-*"))


def test_cli_sanitizes_parser_errors(tmp_path):
    root = private_directory(tmp_path / "invalid")
    (root / "plan.json").write_text("PRIVATE_SECRET invalid json")
    result = CliRunner().invoke(app, ["status", str(root)])
    assert result.exit_code == 2
    assert "PRIVATE_SECRET" not in result.output


def test_real_records_cannot_be_relabelled_with_synthetic_flag(source, tmp_path):
    with pytest.raises(ValueError, match="source_kind=synthetic"):
        prepare(source, tmp_path / "mislabelled", synthetic=True)


def test_synthetic_native_fixture_plans_without_provider_calls(tmp_path):
    from stateset_agents.improvement.fixtures import create_synthetic_fixture

    dataset = create_synthetic_fixture(tmp_path / "fixture", count=8)
    experiment = Experiment.create(
        tmp_path / "native-plan",
        dataset,
        ExperimentConfig("Qwen/Qwen3.5-9B", "river://explicit-parent", min_cases=8),
    )
    status = experiment.status()
    assert status["dataset_counts"] == {"train": 8, "validation": 8, "test": 8}
    assert status["evaluation_scope"] == "synthetic_policy_replay"
    assert status["completed_stages"] == 0


def test_stopping_active_deployment_invalidates_route(deployment):
    provider = ScriptedDeployments()
    deployment.deploy(provider)
    deployment.activate(provider)
    deployment.stop(provider)
    with pytest.raises(ValueError, match="No serving route"):
        deployment.endpoint()
    with pytest.raises(ValueError, match="shutdown"):
        deployment.activate(provider)


def test_private_data_commit_guard_never_prints_payload(tmp_path):
    from scripts.check_private_workflow_data import check_paths

    (tmp_path / "customer.jsonl").write_text(
        json.dumps({"source_kind": "customer", "messages": "PRIVATE_CONTENT"})
    )
    (tmp_path / "synthetic.json").write_text(json.dumps({"source_kind": "synthetic"}))
    assert check_paths(tmp_path, ["customer.jsonl", "synthetic.json"]) == [
        "customer.jsonl"
    ]


def test_test_split_cannot_be_reused_for_a_new_experiment(
    experiment, dataset, tmp_path
):
    complete(experiment)
    second = Experiment.create(
        tmp_path / "second",
        dataset,
        ExperimentConfig(
            base_model="scripted/model",
            parent_checkpoint="river://parent",
            evidence_mode="scripted",
            adapter_id="scripted-contract-v1",
            min_cases=1,
        ),
    )
    backend = ScriptedBackend()
    second.train(backend)
    second.evaluate(backend)
    with pytest.raises(ValueError, match="already consumed"):
        second.evaluate(backend, split="test")
    assert all(split == "validation" for _, split in backend.evaluated)


def test_report_checkpoint_mismatch_is_rejected_before_comparison(experiment):
    class WrongCheckpoint(ScriptedBackend):
        def evaluate(self, *args, **kwargs):
            report = super().evaluate(*args, **kwargs)
            report["selected_checkpoint"]["path"] = "river://wrong"
            return report

    backend = WrongCheckpoint()
    experiment.train(backend)
    with pytest.raises(ValueError, match="does not match"):
        experiment.evaluate(backend)
    assert not list(experiment.root.glob("test-*"))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seeds": (42, 42)},
        {"seeds": (True,)},
        {"min_cases": 0},
        {"min_gain": float("nan")},
        {"learning_rate": 0},
        {"require_significance": True},
    ],
)
def test_invalid_protocol_fails_before_provider_calls(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig("model", "river://parent", **kwargs)


def test_unconfirmed_candidate_policy_violation_fails_safety_gate(experiment):
    backend = ScriptedBackend()
    backend.violation = 1
    experiment.train(backend)
    experiment.evaluate(backend)
    result = experiment.compare(split="validation")
    assert result["gates"]["zero_candidate_safety_failures"] is False
    assert result["passed"] is False


@pytest.mark.asyncio
async def test_wrong_payment_receipt_cannot_be_acknowledged(tmp_path):
    class WrongReceipt(TrustedBackend):
        async def commit(self, request):
            self.commits += 1
            return {
                "applied": True,
                "transaction_id": "another-operation",
                "request_sha256": "0" * 64,
            }

    backend = WrongReceipt()
    operation = resolver(tmp_path / "journal", backend)
    with pytest.raises(ValueError, match="exact request"):
        await operation.resolve_refund(intent="resolve_refund")
    with pytest.raises(ValueError, match="uncertain"):
        await resolver(tmp_path / "journal", backend).resolve_refund(
            intent="resolve_refund"
        )
    assert backend.commits == 1


def test_river_training_adapter_exports_valid_train_only_jsonl(
    dataset, tmp_path, monkeypatch
):
    from stateset_agents.improvement import river
    from stateset_agents.training.sft import parse_chat_dataset

    assert river._SingleAttemptExecutor.MAX_TRANSIENT_ATTEMPTS == 1
    seen = []
    client = SimpleNamespace(close=lambda: seen.append("closed"))

    class RecordingExecutor:
        def __init__(self, client):
            self.client = client

        def submit(self, spec):
            rows = parse_chat_dataset(spec.dataset.read_bytes())
            assert len(rows) == 1
            assert rows[0]["split"] == "train"
            assert spec.dataset.stat().st_mode & 0o077 == 0
            assert spec.harvest["checkpoint"] == "river://parent"
            assert spec.seed == 42
            seen.append(spec)
            return "handle"

        def fetch(self, handle, artifacts):
            assert handle == "handle"
            write_json(
                artifacts / "river_checkpoint.json",
                {"checkpoint": "river://new", "base_model": "model"},
            )

    monkeypatch.setattr(river, "river_client", lambda timeout: client)
    monkeypatch.setattr(river, "_SingleAttemptExecutor", RecordingExecutor)
    output = private_directory(tmp_path / "attempt")
    pointer = river.RiverBackend().train(
        ExperimentConfig("model", "river://parent"), dataset / "train.json", output, 42
    )
    assert pointer["checkpoint"] == "river://new"
    assert seen[-1] == "closed"


def test_native_sampler_schedule_change_is_rejected(monkeypatch, tmp_path):
    from stateset_agents.improvement import river

    engine = tmp_path / "engine.py"
    engine.write_text("# changed seed schedule")
    monkeypatch.setitem(
        sys.modules,
        "river_client",
        SimpleNamespace(rl=SimpleNamespace(RolloutEngine=object)),
    )
    monkeypatch.setattr(river.inspect, "getfile", lambda cls: str(engine))
    with pytest.raises(ValueError, match="seed schedule"):
        river.verify_sampler_runtime()


def test_cli_sanitizes_nonstandard_provider_exceptions(experiment, monkeypatch):
    class ProviderError(Exception):
        pass

    def failed_status(self):
        raise ProviderError("PRIVATE_PROVIDER_SECRET")

    monkeypatch.setattr(Experiment, "status", failed_status)
    result = CliRunner().invoke(app, ["status", str(experiment.root)])
    assert result.exit_code == 2
    assert "PRIVATE_PROVIDER_SECRET" not in result.output


@pytest.mark.asyncio
@pytest.mark.parametrize("corrupt", [False, True])
async def test_native_evaluation_wrapper_replays_provider_outcomes(
    monkeypatch, corrupt
):
    from dataclasses import dataclass

    from stateset_agents.core.environments.refund_policy_environment import (
        refund_policy_benchmark,
    )
    from stateset_agents.evaluation.implementation import package_implementation
    from stateset_agents.improvement.river import RiverBackend
    from tests.unit.test_river_refund_delegation import execute

    @dataclass
    class Checkpoint:
        path: str
        step: int
        checkpoint_type: str

    admitted = []

    class Engine:
        def __init__(self, sampler, **kwargs):
            self.factory = kwargs["env"]
            assert kwargs["schedule"].concurrency == 1
            assert kwargs["budget"].max_generated_tokens == 1024

        async def rollout(self, rows, **kwargs):
            assert kwargs["group_size"] == 1
            for row in rows:
                _, trajectory = await execute(self.factory, row)
                trajectory.generated_tokens = 12
                trajectory.elapsed = 0.1
                if corrupt:
                    trajectory.metrics["executed_tool_calls"] = 999
                yield [trajectory]

    sdk = SimpleNamespace(
        Checkpoint=Checkpoint,
        rl=SimpleNamespace(
            Env=object,
            RolloutEngine=Engine,
            CheckpointSampler=lambda *args, **kwargs: SimpleNamespace(**kwargs),
            Budget=SimpleNamespace,
            Schedule=SimpleNamespace,
            GroupCompletion=SimpleNamespace,
        ),
    )
    monkeypatch.setitem(sys.modules, "river_client", sdk)
    case = refund_policy_benchmark("validation", count=8, seed=501)[0]
    call = RiverBackend()._evaluate(
        ExperimentConfig("model", "river://parent"),
        {case["order_id"]: case},
        "river://parent",
        42,
        "validation",
        package_implementation(),
        object(),
        SimpleNamespace(tokenizer=object()),
        SimpleNamespace(reserve=lambda: admitted.append(True)),
    )
    if corrupt:
        with pytest.raises(ValueError):
            await call
    else:
        report = await call
        assert report["outcomes"][0]["success"] is True
        assert report["outcomes"][0]["executed_tool_calls"] == 3
    assert admitted == [True]
