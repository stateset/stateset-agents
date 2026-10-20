"""Temporal splits must keep duplicate-connected sources out of training."""

import copy

import pytest

from stateset_agents.data.curation import curate_conversations, redact_record


def row(source, timestamp, answer="done", reward=1):
    return {
        "source_id": source,
        "timestamp": timestamp,
        "reward": reward,
        "messages": [
            {"role": "user", "content": "help"},
            {"role": "assistant", "content": answer},
        ],
    }


def test_duplicates_bridge_sources_and_push_entire_group_to_test():
    records = [
        row("a", "2026-01-01T00:00:00Z"),
        row("b", "2026-02-01T00:00:00Z"),
        row("b", "2026-04-01T00:00:00Z", "new"),
        row("c", "2026-01-01T00:00:00Z", "distinct"),
    ]
    original = copy.deepcopy(records)
    result = curate_conversations(
        records,
        validation_after="2026-02-01T00:00:00Z",
        test_after="2026-03-01T00:00:00Z",
    )
    assert len(result["splits"]["train"]) == 1 and len(result["splits"]["test"]) == 2
    assert len({item["source_group"] for item in result["splits"]["test"]}) == 1
    assert result["manifest"]["duplicate_rows"] == 1 and records == original


def test_redaction_reaches_tool_arguments_and_credentials():
    result = redact_record(
        {
            "tool_calls": [
                {
                    "args": {
                        "email": "private@example.com",
                        "note": "Call +1 (415) 555-0100",
                    }
                }
            ],
            "authorization": "Bearer abcdefghijklmnop",
        }
    )
    assert "private@example.com" not in str(result) and "555-0100" not in str(result)
    assert "abcdefghijklmnop" not in str(result)


def test_quality_filter_does_not_remove_failed_held_out_examples():
    result = curate_conversations(
        [
            row("a", "2026-01-01T00:00:00Z", "train", 0),
            row("b", "2026-04-01T00:00:00Z", "failed", 0),
        ],
        validation_after="2026-02-01T00:00:00Z",
        test_after="2026-03-01T00:00:00Z",
        minimum_reward=0.5,
    )
    assert not result["splits"]["train"] and len(result["splits"]["test"]) == 1


@pytest.mark.parametrize("timestamp", ["2026-01-01", None, "wrong"])
def test_bad_timestamp_rejected(timestamp):
    with pytest.raises(ValueError):
        curate_conversations(
            [row("a", timestamp)],
            validation_after="2026-02-01T00:00:00Z",
            test_after="2026-03-01T00:00:00Z",
        )
