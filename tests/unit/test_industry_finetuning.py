"""Industry data preparation, split integrity, and honest execution contracts."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents.data.finetuning import (
    check_finetuning_overlap,
    load_finetuning_data,
    split_finetuning_data,
    validate_finetuning_row,
)
from stateset_agents.training.industry import (
    industry_examples,
    init_industry_project,
    list_industry_recipes,
    prepare_industry_training,
    train_industry_project,
)


def chat(prompt, answer="response", **extra):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        **extra,
    }


@pytest.mark.parametrize("recipe", list_industry_recipes(), ids=lambda r: r.name)
def test_each_industry_round_trips_through_preparation(recipe, tmp_path):
    source = init_industry_project(recipe.name, tmp_path / "source")
    rows = load_finetuning_data(source / "examples.jsonl")
    assert len(rows) == 4
    assert all(row["synthetic"] for row in rows)
    assert any(m.get("tool_calls") for row in rows for m in row["messages"])
    project = tmp_path / "prepared"
    manifest = prepare_industry_training(
        recipe.name, source / "examples.jsonl", project
    )
    assert manifest["source_rows"] == 4
    assert manifest["duplicates_removed"] == 0
    plan = train_industry_project(project)
    assert plan["status"] == "planned"
    assert plan["train_rows"] == 3
    assert plan["validation_rows"] == 1
    assert plan["validation_status"] == "held_out_not_evaluated"
    assert not (project / "adapter").exists()


@pytest.mark.parametrize(
    "value", [None, 1, [], {}, {"messages": []}, chat("", "answer"), chat("hi", "")]
)
def test_invalid_rows_are_rejected(value):
    with pytest.raises(ValueError):
        validate_finetuning_row(value)


def test_loader_reports_physical_line_and_does_not_skip_errors(tmp_path):
    path = tmp_path / "bad.jsonl"
    path.write_text(json.dumps(chat("valid")) + "\n\nnull\n")
    with pytest.raises(ValueError, match="line 3"):
        load_finetuning_data(path)
    path.write_text(json.dumps(chat("valid")) + '\n{"messages": NaN}\n')
    with pytest.raises(ValueError, match="line 2.*Non-finite"):
        load_finetuning_data(path)


@pytest.mark.parametrize(
    "mutation", ["undeclared", "orphan", "missing", "arguments", "duplicate", "role"]
)
def test_tool_conversation_integrity(mutation):
    row = industry_examples("retail")[0]
    call = row["messages"][2]["tool_calls"][0]
    if mutation == "undeclared":
        call["function"]["name"] = "delete_account"
    elif mutation == "orphan":
        row["messages"][3]["tool_call_id"] = "unknown"
    elif mutation == "missing":
        del row["messages"][3]
    elif mutation == "arguments":
        call["function"]["arguments"] = "[1,2]"
    elif mutation == "duplicate":
        row["messages"][2]["tool_calls"].append(copy.deepcopy(call))
    else:
        row["messages"][2]["role"] = "user"
    with pytest.raises(ValueError):
        validate_finetuning_row(row)


def test_terminal_tool_targets_and_argument_types_are_preserved():
    row = industry_examples("retail")[0]
    row["messages"] = row["messages"][:3]
    arguments = {"record_id": "DEMO", "nested": {"count": 3, "enabled": True}}
    row["messages"][-1]["tool_calls"][0]["function"]["arguments"] = arguments
    assert validate_finetuning_row(row) is row
    assert row["messages"][-1]["tool_calls"][0]["function"]["arguments"] == arguments


def test_functiongemma_targets_without_call_ids_are_supported():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples/data/functiongemma_commerce.jsonl"
    )
    assert len(load_finetuning_data(path)) == 2


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("files", []),
        ("model_preset", []),
        ("industry", None),
        ("seed", True),
        ("train_rows", 99),
        ("validation_rows", True),
    ],
)
def test_malformed_manifest_fails_before_execution(prepared, key, value):
    path = prepared / "manifest.json"
    data = json.loads(path.read_text())
    data[key] = value
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        train_industry_project(prepared)
    assert not (prepared / "adapter").exists()


def test_deduplication_and_transitive_group_split_are_order_independent():
    rows = [
        chat("Request A", group_id="case-1"),
        chat("Request B", group_id="case-1"),
        chat(" request   b ", "alternative", group_id="case-2"),
        chat("Request C", group_id="case-2"),
        chat("Independent D"),
        chat("Independent E"),
    ]
    train, validation = split_finetuning_data(rows + [copy.deepcopy(rows[0])])
    assert len(train) + len(validation) == len(rows)
    assert (train, validation) == split_finetuning_data(list(reversed(rows)))
    check_finetuning_overlap(train, validation)
    assert all(row in train for row in rows[:4]) or all(
        row in validation for row in rows[:4]
    )


@pytest.mark.parametrize("fraction", [0, 1, -0.1, float("nan"), float("inf")])
def test_invalid_split_fraction(fraction):
    with pytest.raises(ValueError, match="validation_fraction"):
        split_finetuning_data([chat("A"), chat("B")], fraction)


def test_unseparable_data_and_cross_split_leakage_fail():
    with pytest.raises(ValueError, match="independent"):
        split_finetuning_data([chat("Same", "a"), chat("same", "b")])
    with pytest.raises(ValueError, match="overlap"):
        check_finetuning_overlap(
            [chat("A", group_id="case")], [chat("B", group_id="case")]
        )


@pytest.fixture
def prepared(tmp_path):
    source = init_industry_project("retail", tmp_path / "source")
    project = tmp_path / "prepared"
    prepare_industry_training("retail", source / "examples.jsonl", project)
    return project


def test_preparation_refuses_overwrite_and_unknown_models(tmp_path):
    source = init_industry_project("retail", tmp_path / "source")
    original = (source / "examples.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        init_industry_project("retail", source)
    assert (source / "examples.jsonl").read_bytes() == original
    with pytest.raises(ValueError, match="preset"):
        prepare_industry_training(
            "retail", source / "examples.jsonl", tmp_path / "bad", model="typo"
        )
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize("filename", ["train.jsonl", "validation.jsonl"])
def test_changed_data_cannot_reach_trainer(prepared, filename):
    with (prepared / filename).open("a") as stream:
        stream.write(json.dumps(chat("extra")) + "\n")
    with pytest.raises(ValueError, match="changed after preparation"):
        train_industry_project(prepared)


def test_no_gpu_is_an_execution_error_not_a_successful_preview(prepared, monkeypatch):
    from stateset_agents.training import sft

    monkeypatch.setattr(sft, "gpu_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA"):
        train_industry_project(prepared, dry_run=False)
    assert not (prepared / "adapter").exists()


@pytest.mark.parametrize("fail", [False, True])
def test_training_only_receives_training_split_and_records_outcome(
    prepared, monkeypatch, fail
):
    from stateset_agents.training import sft

    pytest.importorskip("transformers")
    monkeypatch.setattr(sft, "gpu_available", lambda: True)
    captured = {}

    def train(**kwargs):
        captured.update(kwargs)
        if fail:
            raise RuntimeError("trainer failed")
        return kwargs["output_dir"]

    monkeypatch.setattr(sft, "run_sft", train)
    if fail:
        with pytest.raises(RuntimeError, match="trainer failed"):
            train_industry_project(prepared, dry_run=False)
    else:
        assert train_industry_project(prepared, dry_run=False)["status"] == "trained"
    assert captured["rows"] == load_finetuning_data(prepared / "train.jsonl")
    check_finetuning_overlap(
        captured["rows"], load_finetuning_data(prepared / "validation.jsonl")
    )
    run = json.loads((prepared / "adapter/industry_run.json").read_text())
    assert run["status"] == ("failed" if fail else "trained")


def test_preview_and_catalog_do_not_import_ml_stack(prepared):
    code = """
import sys
from stateset_agents.training.industry import list_industry_recipes, train_industry_project
assert len(list_industry_recipes()) == 9
assert train_industry_project(sys.argv[1])['status'] == 'planned'
assert not {'torch', 'transformers', 'peft', 'datasets'} & set(sys.modules)
"""
    subprocess.run([sys.executable, "-c", code, str(prepared)], check=True, timeout=20)


def test_cli_end_to_end(tmp_path):
    from stateset_agents.cli import app

    runner = CliRunner()
    source, project = tmp_path / "source", tmp_path / "prepared"
    assert len(json.loads(runner.invoke(app, ["industry", "list"]).stdout)) == 9
    for args in (
        ["industry", "init", "healthcare", str(source)],
        ["industry", "validate", str(source / "examples.jsonl")],
        [
            "industry",
            "prepare",
            "healthcare",
            str(source / "examples.jsonl"),
            str(project),
        ],
        ["industry", "train", str(project), "--dry-run"],
    ):
        result = runner.invoke(app, args)
        assert result.exit_code == 0, result.output
    assert json.loads(result.stdout)["status"] == "planned"
    assert runner.invoke(app, ["industry", "show", "unknown"]).exit_code == 2


@pytest.mark.parametrize("revision", ["main", "", True, "a" * 39, "G" * 40])
def test_preparation_rejects_mutable_or_malformed_model_revisions(tmp_path, revision):
    source = init_industry_project("retail", tmp_path / "source")
    with pytest.raises(ValueError, match="model_revision"):
        prepare_industry_training(
            "retail",
            source / "examples.jsonl",
            tmp_path / "project",
            model_revision=revision,
        )
    assert not (tmp_path / "project").exists()


def test_pinned_revision_reaches_trainer_and_plan(tmp_path, monkeypatch):
    from stateset_agents.training import sft

    source = init_industry_project("retail", tmp_path / "source")
    project = tmp_path / "project"
    prepare_industry_training(
        "retail", source / "examples.jsonl", project, model_revision="a" * 40
    )
    assert train_industry_project(project)["model_revision"] == "a" * 40
    captured = {}
    monkeypatch.setattr(sft, "gpu_available", lambda: True)
    monkeypatch.setattr(sft, "run_sft", lambda **kwargs: captured.update(kwargs))
    train_industry_project(project, dry_run=False)
    assert captured["model_revision"] == "a" * 40


def test_sft_model_loader_forwards_revision(monkeypatch):
    from stateset_agents.core import transformers_compat
    from stateset_agents.training import sft

    captured = {}
    sentinel = object()

    def load(cls, name, kwargs):
        captured.update(kwargs)
        return sentinel, cls

    monkeypatch.setattr(transformers_compat, "load_generation_model", load)
    assert sft.load_base_model_for_sft("Qwen/Qwen3.5-2B", revision="a" * 40) is sentinel
    assert captured["revision"] == "a" * 40


def test_sft_tokenizer_receives_same_revision(tmp_path, monkeypatch):
    from transformers import AutoTokenizer

    from stateset_agents.training import sft

    captured = {}

    class StopBeforeWeights(Exception):
        pass

    def load(name, **kwargs):
        captured.update(kwargs)
        raise StopBeforeWeights

    monkeypatch.setattr(AutoTokenizer, "from_pretrained", load)
    with pytest.raises(StopBeforeWeights):
        sft.run_sft(
            [],
            "Qwen/Qwen3.5-2B",
            tmp_path,
            1,
            16,
            32,
            2e-5,
            1024,
            1,
            8,
            model_revision="a" * 40,
        )
    assert captured["revision"] == "a" * 40


@pytest.mark.parametrize("synthetic", ["true", "false", 0, 1, None])
def test_synthetic_metadata_cannot_bypass_the_evaluation_gate(synthetic):
    with pytest.raises(ValueError, match="synthetic"):
        validate_finetuning_row(chat("Question", synthetic=synthetic))
