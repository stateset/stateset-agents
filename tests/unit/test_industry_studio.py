"""Preserve industry provenance through Studio while keeping holdouts out of SFT."""

import json

import pytest

from stateset_agents import tui
from stateset_agents.data.text_sft import DatasetFormatError, load_text_sft_snapshot
from stateset_agents.finetune_inspection import inspect_training_data
from stateset_agents.finetune_workspace import FinetuneConfig, plan_run
from stateset_agents.verticals.industry_synthetic import (
    DOMAINS,
    generate_industry_datasets,
)


@pytest.mark.asyncio
@pytest.mark.parametrize("domain", DOMAINS)
async def test_source_provenance_survives_inspection_plan_and_training_snapshot(
    tmp_path, domain
):
    root = tmp_path / "data"
    await generate_industry_datasets(
        root, domains=(domain,), train_count=8, validation_count=8, test_count=8
    )
    train, validation = (
        root / domain / "train.jsonl",
        root / domain / "validation.jsonl",
    )
    snapshot = load_text_sft_snapshot(str(train), str(validation))
    assert snapshot.provenance["train"]["synthetic_examples"] == 8
    assert snapshot.provenance["validation"]["synthetic_examples"] == 8
    assert snapshot.provenance["train"]["industry_domains"] == [domain]
    assert all(set(row) == {"messages"} for row in snapshot.train)
    description = snapshot.describe()
    description["dataset_provenance"]["train"]["industry_domains"].clear()
    assert snapshot.provenance["train"]["industry_domains"] == [domain]
    report = inspect_training_data(train)
    assert report["dataset_provenance"]["synthetic_examples"] == 8
    assert any("synthetic" in note for note in report["warnings"])
    assert "samples" not in report
    plan = plan_run(FinetuneConfig(dataset=str(train)))
    assert plan.examples == 8 and plan.data == train.read_bytes()
    assert (
        "synthetic" in plan.notes[0] and "not independently reviewed" in plan.notes[0]
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("split", ["validation", "test"])
@pytest.mark.parametrize("domain", DOMAINS)
async def test_renamed_held_out_file_cannot_reach_studio_training(
    tmp_path, split, domain
):
    root = tmp_path / "data"
    await generate_industry_datasets(
        root, domains=(domain,), train_count=8, validation_count=8, test_count=8
    )
    path = tmp_path / "renamed-train.jsonl"
    path.write_bytes((root / domain / f"{split}.jsonl").read_bytes())
    for operation in (
        lambda: load_text_sft_snapshot(str(path)),
        lambda: inspect_training_data(path),
        lambda: plan_run(FinetuneConfig(dataset=str(path))),
    ):
        with pytest.raises(DatasetFormatError) as error:
            operation()
        assert error.value.diagnostic["code"] == "held_out_split"
    assert not (tmp_path / "runs").exists()


@pytest.mark.asyncio
async def test_palette_and_review_keep_generated_data_without_launch(
    tmp_path, monkeypatch, settle_preview
):
    root = tmp_path / "data"
    await generate_industry_datasets(
        root, domains=("logistics",), train_count=8, validation_count=8, test_count=8
    )
    studio = tui.Studio(
        tmp_path / "runs", FinetuneConfig(dataset=str(root / "logistics/train.jsonl"))
    )
    monkeypatch.setattr(
        tui, "launch_run", lambda *_: pytest.fail("Review must not launch")
    )
    studio.handle("COMMANDS")
    for char in "training launch":
        studio.handle(char)
    studio.handle("ENTER")
    settle_preview(studio)
    assert studio.overlay == "review" and studio.plan.examples == 8
    assert any("synthetic" in note for note in studio._review_lines())
    assert not studio.workspace.exists()


def test_duplicate_keys_cannot_hide_a_split_declaration(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text('{"messages":[],"metadata":{"split":"test","split":"train"}}\n')
    with pytest.raises(DatasetFormatError) as error:
        load_text_sft_snapshot(str(path))
    assert error.value.diagnostic["code"] == "invalid_json"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", ["reviewed", "synthetic_false", "extra", "private_domain"]
)
async def test_invalid_provenance_never_becomes_a_claim_of_reviewed_data(
    tmp_path, change
):
    root = tmp_path / "data"
    await generate_industry_datasets(
        root, domains=("saas",), train_count=8, validation_count=8, test_count=8
    )
    row = json.loads((root / "saas/train.jsonl").read_text().splitlines()[0])
    if change == "reviewed":
        row["metadata"]["review_status"] = "independently_reviewed"
    elif change == "synthetic_false":
        row["synthetic"] = False
    elif change == "extra":
        row["metadata"]["private"] = "private-value-never-echo"
    else:
        row["metadata"]["industry_domain"] = "private-value-never-echo"
    path = tmp_path / "altered.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(DatasetFormatError) as error:
        inspect_training_data(path)
    assert "private-value-never-echo" not in str(error.value)
