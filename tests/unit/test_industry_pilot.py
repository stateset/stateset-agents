"""Check pilot inputs and native tool decoding before renting a GPU."""

import pytest

from benchmarks.industry_pilot import (
    parse_qwen_response,
    synthetic_lookup_rows,
    validate_manifest,
)
from stateset_agents.data.finetuning import (
    group_finetuning_data,
    validate_finetuning_row,
)


def test_pilot_rows_are_valid_independent_and_explicitly_synthetic():
    rows = synthetic_lookup_rows("retail")
    assert len(rows) == 64
    assert len(group_finetuning_data(rows)) == 64
    for row in rows:
        assert row["synthetic"] is True
        validate_finetuning_row(row)


def test_native_tool_decode_preserves_declared_string_values():
    row = synthetic_lookup_rows("retail", 1)[0]
    response = parse_qwen_response(
        "<tool_call>\n<function=get_order>\n<parameter=record_id>\n001\n</parameter>\n</function>\n</tool_call>",
        row["tools"],
    )
    assert response["tool_calls"][0]["function"] == {
        "name": "get_order",
        "arguments": {"record_id": "001"},
    }
    assert response["content"] is None


@pytest.mark.parametrize(
    "text",
    [
        "<tool_call>unfinished",
        "<tool_call><function=delete_order></function></tool_call>",
        "<tool_call><function=get_order>unparsed</function></tool_call>",
        "<tool_call><function=get_order><parameter=unknown>x</parameter></function></tool_call>",
        "<tool_call><function=get_order><parameter=record_id>x</parameter><parameter=record_id>y</parameter></function></tool_call>",
    ],
)
def test_native_tool_decode_never_drops_malformed_calls(text):
    with pytest.raises(ValueError):
        parse_qwen_response(text, synthetic_lookup_rows("retail", 1)[0]["tools"])


def test_native_decode_retains_unexpected_prose_for_scoring():
    row = synthetic_lookup_rows("retail", 1)[0]
    text = "Done! <tool_call><function=get_order><parameter=record_id>x</parameter></function></tool_call>"
    assert parse_qwen_response(text, row["tools"])["content"] == "Done!"
    assert (
        parse_qwen_response("An ordinary answer.", row["tools"])["content"]
        == "An ordinary answer."
    )


@pytest.mark.parametrize(
    "preset,revision",
    [("../outside", "a" * 40), ("qwen3.5-2b", "main"), ("qwen3.5-4b", None)],
)
def test_pilot_refuses_unsupported_or_unpinned_models(preset, revision):
    with pytest.raises(ValueError):
        validate_manifest(
            {
                "schema_version": 1,
                "models": [
                    {"preset": preset, "revision": revision, "industry": "retail"}
                ],
            }
        )


def test_pilot_accepts_pinned_models_and_refuses_duplicate_output_paths():
    spec = {"preset": "qwen3.5-2b", "revision": "a" * 40, "industry": "retail"}
    validate_manifest({"schema_version": 1, "models": [spec]})
    with pytest.raises(ValueError, match="Duplicate"):
        validate_manifest({"schema_version": 1, "models": [spec, spec]})
