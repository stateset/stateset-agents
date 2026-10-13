"""Study preparation is offline, repeatable, and preserves existing evidence."""

import asyncio
import hashlib
import json
from importlib import metadata
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.evaluation import study_preflight as module
from stateset_agents.evaluation.agent_study import StudyConfig, prepare_study
from stateset_agents.remote import river_runtime as runtime


@pytest.fixture
def study(tmp_path):
    config = StudyConfig(train_count=8, validation_count=8, test_count=8, steps=2)
    prepare_study(tmp_path, config)
    return tmp_path, config


def ready_runtime():
    return {
        "python": {"version": "3.12.12", "passed": True},
        "river_sdk": {"version": "0.11.0", "passed": True},
        "credentials": {"configured": True},
    }


def test_preparation_creates_all_datasets_and_reuses_without_modification(
    study, monkeypatch
):
    root, config = study
    monkeypatch.delenv("RIVER_API_KEY", raising=False)
    monkeypatch.setattr(
        module, "_runtime_checks", lambda: pytest.fail("SDK probe during preparation")
    )
    first = asyncio.run(module.prepare_study_data(root))
    assert first["provider_requests"] == 0
    assert [row["seed"] for row in first["datasets"]] == list(config.seeds)
    assert all(
        row["status"] == "created"
        and row["train_examples"] == row["replayed_examples"] == 8
        for row in first["datasets"]
    )
    before = {p: p.read_bytes() for p in root.rglob("*.json*")}
    second = asyncio.run(module.prepare_study_data(root))
    assert all(row["status"] == "reused" for row in second["datasets"])
    assert {p: p.read_bytes() for p in root.rglob("*.json*")} == before


def test_existing_corrupt_bundle_blocks_new_preparation_without_overwrite(study):
    root, config = study
    destination = root / "seed-47" / "data"
    asyncio.run(
        module.prepare_refund_data(
            destination, seed=47, train_count=8, validation_count=8, test_count=8
        )
    )
    damaged = destination / "train.jsonl"
    damaged.write_text("damaged evidence")
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(module.prepare_study_data(root))
    assert damaged.read_text() == "damaged evidence"
    assert not (root / "seed-42").exists()


@pytest.mark.parametrize("change", ["command", "source"])
def test_stale_or_altered_plan_fails_before_creating_datasets(study, change):
    root, _ = study
    path = root / "study_plan.json"
    plan = json.loads(path.read_text())
    if change == "command":
        plan["stages"][0]["argv"] = ["sh", "-c", "false"]
    else:
        plan["implementation"]["files"]["__init__.py"] = "0" * 64
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="changed"):
        asyncio.run(module.prepare_study_data(root))
    with pytest.raises(ValueError, match="changed"):
        module.preflight_study(root)
    assert not list(root.glob("seed-*"))


def test_interrupted_preparation_reuses_completed_bundles(study, monkeypatch):
    root, _ = study
    original = module.prepare_refund_data
    calls = []

    async def interrupted(destination, **kwargs):
        calls.append(kwargs["seed"])
        if len(calls) == 2:
            raise OSError("interrupted preparation")
        return await original(destination, **kwargs)

    monkeypatch.setattr(module, "prepare_refund_data", interrupted)
    with pytest.raises(OSError):
        asyncio.run(module.prepare_study_data(root))
    first = (root / "seed-42/data/train.jsonl").read_bytes()
    monkeypatch.setattr(module, "prepare_refund_data", original)
    report = asyncio.run(module.prepare_study_data(root))
    assert report["datasets"][0]["status"] == "reused"
    assert sum(row["status"] == "created" for row in report["datasets"]) == 5
    assert (root / "seed-42/data/train.jsonl").read_bytes() == first


def test_preflight_reports_missing_inputs_then_local_readiness(study, monkeypatch):
    root, _ = study
    monkeypatch.setattr(module, "_runtime_checks", ready_runtime)
    report = module.preflight_study(root)
    assert not report["local_checks_passed"] and len(report["issues"]) == 6
    asyncio.run(module.prepare_study_data(root))
    report = module.preflight_study(root)
    assert report["local_checks_passed"]
    assert report["stages"] == {"total": 54, "paid": 42}
    assert report["provider_requests"] == 0 and report["unverified"]
    assert report["limits"]["dollar_cap"] is None
    holdout = root / "seed-42/base/test_results.json"
    holdout.parent.mkdir()
    holdout.write_text("unreadable outcome payload")
    assert module.preflight_study(root)["local_checks_passed"]
    (root / "seed-42/data/test.json").write_text("[]")
    assert not module.preflight_study(root)["local_checks_passed"]


@pytest.mark.parametrize("corruption", ["selected", "duplicate", "heldout"])
def test_training_rows_must_match_planned_count_and_cases(
    study, monkeypatch, corruption
):
    root, _ = study
    asyncio.run(module.prepare_study_data(root))
    monkeypatch.setattr(module, "_runtime_checks", ready_runtime)
    data = root / "seed-42/data"
    manifest_path = data / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if corruption == "selected":
        manifest["selected"] = 1
    else:
        path = data / "train.jsonl"
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        if corruption == "duplicate":
            rows[1] = rows[0]
        else:
            rows[0]["metadata"]["split"] = "test"
        path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
        manifest["artifacts"]["train.jsonl"] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    report = module.preflight_study(root)
    assert not report["local_checks_passed"]
    assert not report["datasets"][0]["passed"]
    with pytest.raises(ValueError, match="training example"):
        asyncio.run(module.prepare_study_data(root))


def test_cli_prepares_and_writes_strict_preflight_report(study, monkeypatch):
    root, _ = study
    runtime = ready_runtime()
    runtime["credentials"]["configured"] = False
    monkeypatch.setattr(module, "_runtime_checks", lambda: runtime)
    runner = CliRunner()
    result = runner.invoke(
        app, ["benchmark", "prepare-agent-study-data", "--study-dir", str(root)]
    )
    assert result.exit_code == 0, result.output
    assert "6 created" in result.output
    result = runner.invoke(
        app,
        ["benchmark", "preflight-agent-study", "--study-dir", str(root), "--strict"],
    )
    assert result.exit_code == 1, result.output
    report = json.loads((root / "study_preflight.json").read_text())
    assert not report["local_checks_passed"]
    assert len(report["issues"]) == 1 and "RIVER_API_KEY" in report["issues"][0]
    assert (
        json.loads((root / "data_preparation.json").read_text())["provider_requests"]
        == 0
    )


def rewrite_training_rows(data, mutate):
    """Simulate self-consistent damaged content, beyond a byte-hash mismatch."""
    path = data / "train.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    mutate(rows)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = data / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"]["train.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(manifest))


@pytest.mark.parametrize(
    "corruption",
    ["prompt", "observation", "action", "incomplete", "extra", "missing", "family"],
)
def test_preflight_replays_transcripts_even_when_artifact_hashes_match(
    study, monkeypatch, corruption
):
    root, _ = study
    asyncio.run(module.prepare_study_data(root))
    monkeypatch.setattr(module, "_runtime_checks", ready_runtime)

    def corrupt(rows):
        row = rows[0]
        if corruption == "family":
            row["metadata"]["family"] = "wrong-family"
        elif corruption == "missing":
            del row["messages"]
        elif corruption == "prompt":
            row["messages"][0]["content"] = "different task"
        elif corruption == "observation":
            observation = next(m for m in row["messages"][2:] if m["role"] == "user")
            observation["content"] = "forged facts"
        elif corruption == "action":
            action = next(m for m in row["messages"] if m["role"] == "assistant")
            action["content"] = "not JSON"
        elif corruption == "incomplete":
            row["messages"].pop()
        else:
            row["messages"].append({"role": "assistant", "content": "extra turn"})

    rewrite_training_rows(root / "seed-42/data", corrupt)
    before = {p: p.read_bytes() for p in root.rglob("*.json*")}
    report = module.preflight_study(root)
    assert not report["local_checks_passed"] and report["provider_requests"] == 0
    failed = report["datasets"][0]
    assert not failed["passed"] and "training example" in failed["reason"]
    assert all(row["replayed_examples"] == 8 for row in report["datasets"][1:])
    with pytest.raises(ValueError, match="training example"):
        asyncio.run(module.prepare_study_data(root))
    assert {p: p.read_bytes() for p in root.rglob("*.json*")} == before
    result = CliRunner().invoke(
        app,
        ["benchmark", "preflight-agent-study", "--study-dir", str(root), "--strict"],
    )
    assert result.exit_code == 1, result.output


def test_rehashed_invalid_later_seed_blocks_creation_of_earlier_missing_bundles(study):
    root, _ = study
    data = root / "seed-47/data"
    asyncio.run(
        module.prepare_refund_data(
            data, seed=47, train_count=8, validation_count=8, test_count=8
        )
    )
    rewrite_training_rows(data, lambda rows: rows[0].update(messages=[]))
    with pytest.raises(ValueError, match="sandbox replay"):
        asyncio.run(module.prepare_study_data(root))
    assert not (root / "seed-42").exists()


@pytest.mark.asyncio
async def test_async_preflight_replays_examples_inside_existing_event_loop(
    study, monkeypatch
):
    root, _ = study
    await module.prepare_study_data(root)
    monkeypatch.setattr(module, "_runtime_checks", ready_runtime)
    report = await module.preflight_study_async(root)
    assert report["local_checks_passed"]
    assert sum(row["replayed_examples"] for row in report["datasets"]) == 48
    with pytest.raises(RuntimeError, match="Await preflight_study_async"):
        module.preflight_study(root)


def test_runtime_inspection_does_not_construct_client_or_renderer_or_expose_key(
    monkeypatch,
):
    def forbidden(*args, **kwargs):
        pytest.fail("Runtime inspection must not open clients or load tokenizers")

    class Trainer:
        async def run(self, data, *, after_recovery=None):
            pass

    required = (
        "Env",
        "RolloutEngine",
        "CheckpointSampler",
        "Budget",
        "Schedule",
        "Adam",
        "GroupCentered",
        "Truncation",
        "GroupCompletion",
        "Checkpointing",
        "Evaluator",
    )
    sdk_modules = {
        "river_client": SimpleNamespace(
            Client=forbidden, LoraConfig=forbidden, Checkpoint=forbidden
        ),
        "river_client.rl": SimpleNamespace(
            AsyncTrainer=Trainer, **dict.fromkeys(required, forbidden)
        ),
        "river_client.renderers": SimpleNamespace(get_renderer=forbidden),
    }
    monkeypatch.setattr(
        runtime,
        "metadata",
        SimpleNamespace(
            version=lambda name: "0.11.0",
            PackageNotFoundError=metadata.PackageNotFoundError,
        ),
    )
    monkeypatch.setattr(
        runtime, "importlib", SimpleNamespace(import_module=sdk_modules.__getitem__)
    )
    monkeypatch.setenv("RIVER_API_KEY", "private-test-credential")
    result = runtime.inspect_native_runtime()
    assert result["river_sdk"]["passed"]
    assert result["credentials"]["configured"]
    assert "private-test-credential" not in json.dumps(result)
    for namespace, capability in (
        ("river_client.rl", "Truncation"),
        ("river_client.rl", "Env"),
        ("river_client", "LoraConfig"),
        ("river_client", "Checkpoint"),
    ):
        delattr(sdk_modules[namespace], capability)
        failure = runtime.inspect_native_runtime()["river_sdk"]
        assert not failure["passed"]
        assert capability in failure["reason"]
        setattr(sdk_modules[namespace], capability, forbidden)
    Trainer.run = lambda self, data: None
    assert (
        "recovery callback" in runtime.inspect_native_runtime()["river_sdk"]["reason"]
    )
    Trainer.run = lambda self, data, after_recovery, /: None
    assert (
        "keyword recovery callback"
        in runtime.inspect_native_runtime()["river_sdk"]["reason"]
    )


def test_sdk_import_errors_do_not_expose_exception_text(monkeypatch):
    def fail(name):
        raise RuntimeError("private-test-credential")

    monkeypatch.setattr(
        runtime,
        "metadata",
        SimpleNamespace(
            version=lambda name: "0.11.0",
            PackageNotFoundError=metadata.PackageNotFoundError,
        ),
    )
    monkeypatch.setattr(runtime, "importlib", SimpleNamespace(import_module=fail))
    report = runtime.inspect_native_runtime()
    assert not report["river_sdk"]["passed"]
    assert "RuntimeError" in report["river_sdk"]["reason"]
    assert "private-test-credential" not in json.dumps(report)


@pytest.mark.parametrize("version", [None, "0.10.0", "0.12.0", "invalid"])
def test_unavailable_or_unsupported_sdk_never_passes_or_imports(version, monkeypatch):
    def installed(name):
        if version is None:
            raise metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(
        runtime,
        "metadata",
        SimpleNamespace(
            version=installed, PackageNotFoundError=metadata.PackageNotFoundError
        ),
    )
    monkeypatch.setattr(
        runtime,
        "importlib",
        SimpleNamespace(
            import_module=lambda name: pytest.fail("unsupported SDK imported")
        ),
    )
    monkeypatch.setenv("RIVER_API_KEY", " ")
    report = runtime.inspect_native_runtime()
    assert not report["river_sdk"]["passed"]
    assert not report["credentials"]["configured"]
