"""Held-out declarations must fail before training or provider side effects."""

from __future__ import annotations

import builtins
import hashlib
import importlib
import json

import pytest

from stateset_agents.data.finetuning import (
    load_finetuning_data,
    validate_finetuning_split,
)
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.job import RemoteJobSpec
from stateset_agents.training import sft
from stateset_agents.training.industry import (
    load_industry_project,
    prepare_industry_training,
    train_industry_project,
)
from stateset_agents.verticals.industry_synthetic import generate_industry_datasets


def chat(split=None, number=0):
    row = {
        "messages": [
            {"role": "user", "content": f"request {number}"},
            {"role": "assistant", "content": "response"},
        ]
    }
    if split is not None:
        row["metadata"] = {"split": split}
    return row


def write_rows(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.mark.parametrize("split", ["validation", "test"])
def test_mixed_input_fails_whole_worker_before_gpu_detection(
    tmp_path, monkeypatch, split
):
    path = write_rows(tmp_path / "renamed-train.jsonl", [chat("train"), chat(split, 1)])
    monkeypatch.setattr(
        sft, "gpu_available", lambda: pytest.fail("GPU detection reached")
    )
    result = sft.run_sft_job(
        {"dataset": str(path), "output_dir": str(tmp_path / "out"), "dry_run": False}
    )
    assert result["returncode"] == 1
    assert any(
        "line 2" in line and "requested train split" in line for line in result["logs"]
    )
    assert not (tmp_path / "out").exists()


def test_in_memory_training_rejects_before_optional_ml_imports(tmp_path, monkeypatch):
    original_import = builtins.__import__

    def guarded_import(name, *args, **kwargs):
        if name.split(".")[0] in {"datasets", "peft", "transformers"}:
            pytest.fail("ML import reached before split check")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    with pytest.raises(ValueError, match="row 2.*requested train split"):
        sft.run_sft(
            rows=[chat("train"), chat("test", 1)],
            base_model="unused",
            output_dir=tmp_path / "adapter",
            num_epochs=1,
            lora_r=16,
            lora_alpha=32,
            learning_rate=2e-5,
            max_length=1024,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
        )
    assert not (tmp_path / "adapter").exists()


@pytest.mark.parametrize(
    "declaration",
    [None, "private-value-never-echo", [], {}, True, "Train"],
)
def test_invalid_declarations_fail_without_echoing_values(declaration):
    with pytest.raises(ValueError) as error:
        validate_finetuning_split(
            {"metadata": {"split": declaration}}, expected_split="train"
        )
    assert "private-value-never-echo" not in str(error.value)


@pytest.mark.parametrize(
    "row",
    [
        {"split": "test"},
        {"split": "train", "metadata": {"split": "test"}},
        {"metadata": {"split": "test"}},
    ],
)
def test_all_supported_declaration_locations_are_checked(row):
    with pytest.raises(ValueError):
        validate_finetuning_split(row, expected_split="train")


@pytest.mark.parametrize(
    "raw",
    [
        '{"messages":[],"metadata":{"split":"test","split":"train"}}',
        '{"messages":[],"metadata":{"split":"test"},"metadata":{"split":"train"}}',
        '{"split":"test","split":"train","messages":[]}',
    ],
)
def test_duplicate_keys_fail_in_both_dataset_loaders(tmp_path, raw):
    path = tmp_path / "ambiguous.jsonl"
    path.write_text(raw + "\n", encoding="utf-8")
    for load in (sft.load_chat_dataset, load_finetuning_data):
        with pytest.raises(ValueError, match="line 1.*Duplicate JSON"):
            load(path)


def test_generic_reads_allow_explicit_evaluation_role(tmp_path):
    path = write_rows(tmp_path / "test.jsonl", [chat("test")])
    assert load_finetuning_data(path) == [chat("test")]
    assert load_finetuning_data(path, expected_split="test") == [chat("test")]
    with pytest.raises(ValueError, match="requested validation split"):
        load_finetuning_data(path, expected_split="validation")


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", ["ecommerce", "logistics", "saas"])
async def test_generated_projects_train_and_validate_with_their_declared_roles(
    tmp_path, domain
):
    root = tmp_path / "data"
    await generate_industry_datasets(
        root, domains=(domain,), train_count=8, validation_count=8, test_count=8
    )
    project = root / domain
    source = (project / "train.jsonl").read_bytes()
    assert len(sft.load_chat_dataset(project / "train.jsonl")) == 8
    assert train_industry_project(project)["status"] == "planned"
    assert (project / "train.jsonl").read_bytes() == source
    # Rehashing a renamed held-out file cannot make it eligible for training.
    held_out = (project / "test.jsonl").read_bytes()
    (project / "train.jsonl").write_bytes(held_out)
    manifest_path = project / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"]["train.jsonl"] = hashlib.sha256(held_out).hexdigest()
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="requested train split"):
        train_industry_project(project)
    assert not (project / "adapter").exists()


def test_prepare_only_resplits_training_eligible_rows_and_keeps_source(tmp_path):
    source = write_rows(tmp_path / "source.jsonl", [chat("train", i) for i in range(8)])
    before = source.read_bytes()
    project = tmp_path / "prepared"
    prepare_industry_training("retail", source, project, validation_fraction=0.25)
    _, train, validation = load_industry_project(project)
    assert len(train) == 6 and len(validation) == 2
    assert all(row["metadata"]["split"] == "train" for row in train)
    assert all(row["metadata"]["split"] == "validation" for row in validation)
    assert source.read_bytes() == before
    for split in ("validation", "test"):
        write_rows(source, [chat(split, i) for i in range(8)])
        destination = tmp_path / split
        with pytest.raises(ValueError, match="requested train split"):
            prepare_industry_training("retail", source, destination)
        assert not destination.exists()


class NoProviderCalls:
    def __getattr__(self, name):
        pytest.fail(f"Provider boundary reached: {name}")


PROVIDERS = [
    ("local", "LocalExecutor", {}),
    (
        "river",
        "RiverExecutor",
        {"client": NoProviderCalls(), "tokenizer": NoProviderCalls()},
    ),
    (
        "runpod",
        "RunPodExecutor",
        {
            "api": NoProviderCalls(),
            "ssh": NoProviderCalls(),
            "allow_environment_wheel": False,
        },
    ),
    ("modal", "ModalExecutor", {}),
    (
        "coreweave",
        "CoreWeaveExecutor",
        {"kube": NoProviderCalls(), "artifact_store": NoProviderCalls()},
    ),
    (
        "nebius",
        "NebiusExecutor",
        {"cli": NoProviderCalls(), "artifact_store": NoProviderCalls()},
    ),
    ("huggingface", "HuggingFaceJobsExecutor", {"api": NoProviderCalls()}),
    ("fireworks", "FireworksExecutor", {"client": NoProviderCalls()}),
]


@pytest.mark.parametrize(
    "module,class_name,options", PROVIDERS, ids=[p[0] for p in PROVIDERS]
)
@pytest.mark.parametrize("kind", ["validation", "test", "mixed", "duplicate"])
def test_provider_submit_rejects_before_any_external_boundary(
    tmp_path, monkeypatch, module, class_name, options, kind
):
    dataset = tmp_path / "renamed-train.jsonl"
    if kind == "duplicate":
        dataset.write_text('{"split":"test","split":"train","messages":[]}\n')
    else:
        rows = [chat("train"), chat("test", 1)] if kind == "mixed" else [chat(kind)]
        write_rows(dataset, rows)
    imported = importlib.import_module(f"stateset_agents.remote.{module}")
    executor = getattr(imported, class_name)(**options)

    def forbidden(*args, **kwargs):
        pytest.fail("External setup reached before source split check")

    if module == "local":
        monkeypatch.setattr(imported.subprocess, "run", forbidden)
    if module == "modal":
        monkeypatch.setattr(executor, "_require_sdk", forbidden)
    if module == "river":
        monkeypatch.setattr(executor, "_checked_client", forbidden)
        monkeypatch.setattr(executor, "_get_tokenizer", forbidden)
    spec = RemoteJobSpec(
        dataset=dataset, base_model="Qwen/Qwen3.5-9B", output_dir=tmp_path / "out"
    )
    with pytest.raises(RemoteExecutionError, match="Invalid SFT training source"):
        executor.submit(spec)
    assert not (tmp_path / "out").exists()
    assert not getattr(executor, "_jobs", {})
    assert not getattr(executor, "_specs", {})
    assert sorted(p.name for p in tmp_path.iterdir()) == [dataset.name]
