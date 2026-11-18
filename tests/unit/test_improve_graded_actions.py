"""A scored assistant action keeps the evidence used to grade it through SFT."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.cli_improve import ImproveDataError, run_improve
from stateset_agents.data import grade_transcript as grading
from stateset_agents.data import prepare_sft_dataset as prepare
from stateset_agents.mcp_server import grade_transcript as mcp_grade


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.fixture
def refund(tmp_path):
    source = tmp_path / "refund.jsonl"
    action = {"name": "refund", "arguments": '{"order":42}'}
    messages = [
        {"role": "system", "content": "Follow refund policy."},
        {"role": "user", "content": "Refund order 42."},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"type": "function", "id": "refund-42", "function": action}],
        },
        {"role": "tool", "tool_call_id": "refund-42", "content": '{"approved":true}'},
        {"role": "assistant", "content": "I issued your refund."},
        {"role": "user", "content": "Future private question"},
    ]
    tools = [
        {
            "type": "function",
            "function": {"name": "refund", "parameters": {"type": "object"}},
        }
    ]
    write_jsonl(source, [{"tools": tools, "messages": messages}])
    reference = {
        "expected_tool": "refund",
        "expected_params": {"order": 42},
        "expected_tool_result": {"approved": True},
        "expected_outcome": "issued",
    }
    contexts = tmp_path / "references.jsonl"
    write_jsonl(contexts, [{}, reference])
    return source, contexts, reference


def test_refund_cli_to_supervised_tokens(refund, tmp_path, monkeypatch):
    from stateset_agents.training.sft import load_chat_dataset, tokenize_sft_row
    from tests.unit.test_loss_computation_real_tokenizer_masks import _byte_tokenizer
    from tests.unit.test_river_sft_masking import QWEN3_LIKE

    source, contexts, _ = refund
    output = tmp_path / "output"
    result = CliRunner().invoke(
        app,
        [
            "improve",
            "run",
            "--format",
            "openai",
            "--transcripts",
            str(source),
            "--context-file",
            str(contexts),
            "--reward",
            "tool_calling",
            "--threshold",
            "0",
            "--output",
            str(output),
        ],
    )
    assert result.exit_code == 0, result.output
    summary = json.loads((output / "improve_summary.json").read_text())
    assert (summary["graded_turn_count"], summary["ungradable_turn_count"]) == (1, 1)
    assert summary["mean_score"] == 1.0
    (entry,) = prepare.load_curated(output / "curated.jsonl")
    assert entry["prompt"] == "Refund order 42."
    converted = tmp_path / "train.jsonl"
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare_sft_dataset",
            "--input",
            str(output / "curated.jsonl"),
            "--format",
            "chat",
            "--output",
            str(converted),
            "--dedup",
        ],
    )
    assert prepare.main() == 0
    (row,) = load_chat_dataset(converted)
    messages = row["messages"]
    assert len(messages) == 5
    assert messages[2]["weight"] == 0 and messages[4]["weight"] == 1
    (call,) = messages[2]["tool_calls"]
    assert call["function"] == {"name": "refund", "arguments": {"order": 42}}
    assert messages[3]["tool_call_id"] == call["id"] == "refund-42"
    assert row["tools"][0]["function"]["name"] == "refund"
    assert "expected_tool" not in converted.read_text()

    tok = _byte_tokenizer(QWEN3_LIKE)
    encoded = tokenize_sft_row(tok, row, max_length=4096)
    assert encoded is not None
    supervised = tok.decode([label for label in encoded["labels"] if label != -100])
    full = tok.decode(encoded["input_ids"])
    assert "I issued your refund." in supervised
    assert "Follow refund policy." in full and "Follow refund policy." not in supervised
    assert "<tool_call>refund" in full and "<tool_call>refund" not in supervised
    assert '"approved":true' in full and '"approved":true' not in supervised
    assert "Future private question" not in full


@pytest.mark.parametrize(
    ("change", "expected_score"),
    [
        ("wrong_result", 0),
        ("wrong_parameter", 0),
        ("numeric_boolean", 0),
        ("unlinked", None),
    ],
)
def test_refund_grade_requires_the_matching_observation(
    refund, tmp_path, change, expected_score
):
    source, contexts, reference = refund
    row = json.loads(source.read_text())
    if change == "wrong_result":
        row["messages"][3]["content"] = '{"approved":false}'
    elif change == "wrong_parameter":
        reference["expected_params"] = {"order": "42"}
    elif change == "numeric_boolean":
        reference["expected_tool_result"] = {"approved": 1}
    else:
        row["messages"][3]["tool_call_id"] = "other-call"
    write_jsonl(source, [row])
    write_jsonl(contexts, [{}, reference])
    result = run_improve(
        transcripts=str(source),
        format="openai",
        reward="tool_calling",
        context_file=str(contexts),
        output=str(tmp_path / "out"),
    )
    assert result["mean_score"] == expected_score
    assert result["curated_count"] == 0


@pytest.mark.asyncio
async def test_text_judge_only_scores_verified_text(refund, monkeypatch):
    from stateset_agents.cli_improve import _build_reward
    from stateset_agents.data.trajectory_ingest import (
        from_openai_jsonl,
        to_grading_history,
    )
    from stateset_agents.rewards import llm_judge

    judge = AsyncMock()
    judge.evaluate.return_value = 1.0
    monkeypatch.setattr(
        llm_judge, "JudgeConfig", lambda: SimpleNamespace(api_key="test")
    )
    monkeypatch.setattr(llm_judge, "LLMJudge", lambda config: judge)
    reward = _build_reward("tool_calling", None, True, lambda msg: None)
    source, _, reference = refund
    (trajectory,) = from_openai_jsonl(source)
    history = to_grading_history(trajectory)
    contexts = [
        {"expected_tool": "refund", "expected_params": {"order": 42.0}},
        reference,
    ]
    valid = await grading.grade_transcript(
        history, contexts, reward, reward_name="tool_calling"
    )
    assert [row["score"] for row in valid] == [1, 1]
    history[3]["content"] = '{"approved":false}'
    denied = await grading.grade_transcript(
        history, contexts, reward, reward_name="tool_calling"
    )
    assert denied[1]["score"] == 0
    history[3]["tool_call_id"] = "unknown"
    unlinked = await grading.grade_transcript(
        history, contexts, reward, reward_name="tool_calling"
    )
    assert unlinked[1]["score"] is None
    judge.evaluate.assert_awaited_once()


@pytest.mark.asyncio
async def test_reference_file_and_missing_gold_have_cli_mcp_parity(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    path = history / "math.jsonl"
    write_jsonl(
        path,
        [
            {"role": "user", "content": "Two plus two?"},
            {"role": "assistant", "content": "#### 4"},
        ],
    )
    references = tmp_path / "gold"
    references.mkdir()
    write_jsonl(references / path.name, [{"gold_answer": 4}])
    from stateset_agents.cli_improve import run_improve_async

    for has_gold in (False, True):
        context_file = str(references) if has_gold else None
        summary = await run_improve_async(
            transcripts=str(history),
            reward="gsm8k",
            threshold=0,
            output=str(tmp_path / "out"),
            context_file=context_file,
        )
        report = await mcp_grade(
            str(path), "gsm8k", str(references / path.name) if has_gold else None
        )
        assert (
            summary["mean_score"] == report["mean_score"] == (1.0 if has_gold else None)
        )
        assert summary["curated_count"] == int(has_gold)
        if not has_gold:
            assert report["rows"][0]["status"] == "ungradable"
            assert (
                "gold_answer"
                in summary["transcripts"][0]["ungradable_turns"][0]["reason"]
            )


def test_context_files_must_align_before_curation(refund, tmp_path):
    source, contexts, _ = refund
    output = tmp_path / "out"
    output.mkdir()
    old = output / "curated.jsonl"
    old.write_text("retained\n")
    write_jsonl(contexts, [{"expected_tool": "refund"}])
    with pytest.raises(ImproveDataError, match="expected 2 assistant-turn contexts"):
        run_improve(
            transcripts=str(source),
            format="openai",
            reward="tool_calling",
            output=str(output),
            context_file=str(contexts),
        )
    assert old.read_text() == "retained\n"


def test_dedup_retains_distinct_context_and_reads_legacy_rows(tmp_path):
    path = tmp_path / "curated.jsonl"
    write_jsonl(path, [{"prompt": "Status?", "response": "Done."}])
    pair = [
        {"role": "user", "content": "Status?"},
        {"role": "assistant", "content": "Done."},
    ]
    for prefix in (
        [],
        [{"role": "system", "content": "Account A"}],
        [{"role": "system", "content": "Account B"}],
    ):
        grading.write_curated_examples(path, prefix + pair, [{"score": 1}], 0.7, path)
    entries = prepare.filter_entries(prepare.load_curated(path), dedup=True)
    assert len(entries) == 3
    assert prepare.to_chat(entries[0]) == {"messages": pair}
    assert "Account A" in prepare.to_hf_trainer(entries[1])["text"]
    assert "Account B" in prepare.to_axolotl(entries[2])["instruction"]


def test_batch_references_grade_langchain_calls_by_conversation(tmp_path):
    from stateset_agents.training.sft import tokenize_sft_row
    from tests.unit.test_loss_computation_real_tokenizer_masks import _byte_tokenizer
    from tests.unit.test_river_sft_masking import QWEN3_LIKE

    source = tmp_path / "calls.json"
    references = tmp_path / "references"
    references.mkdir()
    conversations = []
    for index, (actual, expected) in enumerate([(42, 42), (19, 20)]):
        conversations.append(
            [
                {"type": "human", "content": f"Refund order {expected}."},
                {
                    "type": "ai",
                    "content": "",
                    "tool_calls": [
                        {"id": "call-1", "name": "refund", "args": {"order": actual}}
                    ],
                },
            ]
        )
        write_jsonl(
            references / f"conversation_{index}.jsonl",
            [{"expected_tool": "refund", "expected_params": {"order": expected}}],
        )
    source.write_text(json.dumps(conversations))
    output = tmp_path / "curated"
    summary = run_improve(
        transcripts=str(source),
        format="langchain",
        reward="tool_calling",
        context_file=str(references),
        output=str(output),
    )
    assert summary["graded_turn_count"] == 2
    assert [t["mean_score"] for t in summary["transcripts"]] == [1, 0]
    (entry,) = prepare.load_curated(output / "curated.jsonl")
    row = prepare.to_chat(entry)
    target = row["messages"][-1]
    assert target["weight"] == 1 and target["content"] == ""
    assert target["tool_calls"][0]["function"] == {
        "name": "refund",
        "arguments": {"order": 42},
    }
    tok = _byte_tokenizer(QWEN3_LIKE)
    encoded = tokenize_sft_row(tok, row, max_length=4096)
    assert encoded is not None
    supervised = tok.decode([label for label in encoded["labels"] if label != -100])
    assert "<tool_call>refund</tool_call>" in supervised
    assert "Refund order 42" not in supervised
