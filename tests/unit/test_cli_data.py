"""``stateset-agents data decontaminate``: report, filter, flag and exit codes."""

from __future__ import annotations

import json

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app

QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half "
    "as many clips in May. How many clips did Natalia sell altogether?"
)


def _write(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return path


@pytest.fixture()
def files(tmp_path):
    train = _write(
        tmp_path / "train.jsonl",
        [
            {
                "messages": [
                    {"role": "user", "content": QUESTION},
                    {"role": "assistant", "content": "72"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Reset my router please"},
                    {"role": "assistant", "content": "Done."},
                ]
            },
        ],
    )
    held_out = _write(
        tmp_path / "eval.jsonl",
        [
            {"question": QUESTION},
            {"question": "Plan a three day trip to Lisbon in spring"},
        ],
    )
    return tmp_path, train, held_out


def invoke(*args):
    return CliRunner().invoke(app, ["data", "decontaminate", *map(str, args)])


def test_report_action_writes_evidence_and_summary(files):
    tmp_path, train, held_out = files
    report = tmp_path / "out" / "report.json"
    result = invoke("--train", train, "--eval", held_out, "--report", report)
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["summary"]["contaminated_eval_rows"] == 1
    assert summary["eval_sets"]["eval.jsonl"]["contamination_rate"] == 0.5
    written = json.loads(report.read_text(encoding="utf-8"))
    entry = written["eval_sets"]["eval.jsonl"]["contaminated"][0]
    assert entry["row"] == "eval.jsonl:1" and entry["train_examples"] == [
        "train.jsonl:1"
    ]


def test_filter_and_flag_write_cleaned_training_rows(files):
    tmp_path, train, held_out = files
    filtered = tmp_path / "clean.jsonl"
    result = invoke(
        "--train", train, "--eval", held_out, "--action", "filter", "-o", filtered
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["action"]["affected_rows"] == 1
    rows = [json.loads(line) for line in filtered.read_text().splitlines()]
    assert [row["messages"][0]["content"] for row in rows] == ["Reset my router please"]

    flagged = tmp_path / "flagged.json"
    result = invoke(
        "--train", train, "--eval", held_out, "--action", "flag", "-o", flagged
    )
    assert result.exit_code == 0, result.output
    rows = json.loads(flagged.read_text())
    assert rows[0]["metadata"]["contamination"]["eval_sets"] == ["eval.jsonl"]
    assert "metadata" not in rows[1]


def test_target_eval_writes_a_clean_eval_subset(files):
    tmp_path, train, held_out = files
    clean = tmp_path / "eval.clean.jsonl"
    result = invoke(
        "--train",
        train,
        "--eval",
        held_out,
        "--action",
        "filter",
        "--target",
        "eval",
        "-o",
        clean,
    )
    assert result.exit_code == 0, result.output
    rows = [json.loads(line) for line in clean.read_text().splitlines()]
    assert rows == [{"question": "Plan a three day trip to Lisbon in spring"}]


def test_fail_on_contamination_exits_one_after_writing_outputs(files):
    tmp_path, train, held_out = files
    report = tmp_path / "report.json"
    result = invoke(
        "--train",
        train,
        "--eval",
        held_out,
        "--report",
        report,
        "--fail-on-contamination",
    )
    assert result.exit_code == 1
    assert report.exists()
    clean_train = _write(
        tmp_path / "other.jsonl", [{"prompt": "Reset my router please"}]
    )
    assert (
        invoke(
            "--train", clean_train, "--eval", held_out, "--fail-on-contamination"
        ).exit_code
        == 0
    )


def test_builtin_eval_sets_and_near_duplicate_options(tmp_path):
    train = _write(
        tmp_path / "train.jsonl", [{"prompt": "The app crashes every time I open it"}]
    )
    result = invoke(
        "--train",
        train,
        "--eval",
        "builtin:support-bench",
        "--near-duplicates",
        "--threshold",
        "0.7",
    )
    assert result.exit_code == 0, result.output
    summary = json.loads(result.output)
    assert summary["eval_sets"]["support-bench"]["contaminated_rows"] == 1
    assert "near duplicates" in summary["method"]


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--action", "filter"], "needs --output"),
        (["--action", "drop", "-o", "x.jsonl"], "--action must be one of"),
        (["--target", "both"], "--target must be one of"),
        (["-o", "x.jsonl"], "--output needs --action filter or flag"),
        (["--eval", "builtin:mmlu"], "Unknown built-in eval set"),
    ],
)
def test_invalid_options_exit_two(files, extra, message):
    _tmp_path, train, held_out = files
    result = invoke("--train", train, "--eval", held_out, *extra)
    assert result.exit_code == 2
    assert message in result.output


def test_outputs_are_never_overwritten_and_text_is_required(files):
    tmp_path, train, held_out = files
    existing = tmp_path / "exists.json"
    existing.write_text("{}", encoding="utf-8")
    result = invoke("--train", train, "--eval", held_out, "--report", existing)
    assert result.exit_code == 2 and "Refusing to overwrite" in result.output
    assert existing.read_text(encoding="utf-8") == "{}"
    empty = _write(tmp_path / "empty.jsonl", [{"id": 1}])
    result = invoke("--train", empty, "--eval", held_out)
    assert result.exit_code == 2 and "No training row has text" in result.output
    result = invoke(
        "--train",
        train,
        "--train",
        empty,
        "--eval",
        held_out,
        "--action",
        "filter",
        "-o",
        tmp_path / "o.jsonl",
    )
    assert result.exit_code == 2 and "exactly one --train" in result.output


def test_builtin_training_sets_are_accepted_for_reports(tmp_path):
    held_out = _write(
        tmp_path / "eval.jsonl", [{"question": "What are your business hours?"}]
    )
    result = invoke("--train", "builtin:support-bench", "--eval", held_out)
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["summary"]["contaminated_eval_rows"] == 1
    result = invoke(
        "--train",
        "builtin:support-bench",
        "--eval",
        held_out,
        "--action",
        "filter",
        "-o",
        tmp_path / "x.jsonl",
    )
    assert result.exit_code == 2 and "--train file" in result.output
