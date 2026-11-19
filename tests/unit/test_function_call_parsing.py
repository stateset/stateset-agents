"""Parse complete tool responses before dispatching any handler."""

import json

import pytest

from stateset_agents.core.function_calling import FunctionCallingMixin


@pytest.mark.parametrize(
    "prefix,suffix", [("", ""), ("```json\n", "\n```"), ("Use this:\n", " Done.")]
)
def test_nested_arguments_and_escaped_strings_round_trip(prefix, suffix):
    arguments = {"nested": {"items": [1, 2]}, "text": '雪 "braces } { and \\ quotes'}
    document = {
        "tool_calls": [
            {
                "id": "one",
                "type": "function",
                "function": {"name": "lookup", "arguments": json.dumps(arguments)},
            },
            {"id": "two", "function": {"name": "next", "arguments": "{}"}},
        ]
    }
    calls = FunctionCallingMixin()._parse_tool_calls(
        prefix + json.dumps(document) + suffix
    )
    assert [call.id for call in calls] == ["one", "two"]
    assert calls[0].parsed_arguments() == arguments
    assert calls[1].parsed_arguments() == {}


def test_standalone_function_objects_are_decoded_without_matching_quoted_fragments():
    text = json.dumps(
        {
            "name": "lookup",
            "arguments": json.dumps({"quoted": '{"name":"fake","arguments":"{}"}'}),
        }
    )
    calls = FunctionCallingMixin()._parse_tool_calls(text)
    assert len(calls) == 1
    assert calls[0].name == "lookup"
    assert calls[0].parsed_arguments()["quoted"].startswith('{"name":"fake"')


@pytest.mark.parametrize(
    "response",
    [
        '{"tool_calls":[{"function":{"name":"run","arguments":"{}"}}',
        '{bad outer: {"name":"run","arguments":"{}"}}',
        '{"name":"run","arguments":"{}"} {incomplete',
        "ordinary response without tools",
    ],
)
def test_incomplete_or_malformed_enclosing_response_cannot_authorize_fragments(
    response,
):
    assert FunctionCallingMixin()._parse_tool_calls(response) == []


@pytest.mark.parametrize(
    "document",
    [
        {"tool_calls": {}},
        {"tool_calls": [None]},
        {"tool_calls": [{"type": "other", "function": {"name": "run"}}]},
        {"tool_calls": [{"function": {"name": "run", "arguments": {}}}]},
        {"tool_calls": [{"id": "", "function": {"name": "run"}}]},
        {
            "tool_calls": [
                {"id": "one", "function": {"name": "run"}},
                {"id": "one", "function": {"name": "run"}},
            ]
        },
    ],
)
def test_ambiguous_tool_envelopes_fail_before_dispatch(document):
    with pytest.raises(ValueError):
        FunctionCallingMixin()._parse_tool_calls(json.dumps(document))


def test_duplicate_envelope_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        FunctionCallingMixin()._parse_tool_calls('{"tool_calls":[],"tool_calls":[]}')


@pytest.mark.parametrize("with_id", [False, True])
def test_single_openai_call_envelope_preserves_legacy_support(with_id):
    document = {"type": "function", "function": {"name": "lookup", "arguments": "{}"}}
    if with_id:
        document["id"] = "single"
    calls = FunctionCallingMixin()._parse_tool_calls(json.dumps(document))
    assert len(calls) == 1
    assert calls[0].name == "lookup"
    assert calls[0].parsed_arguments() == {}
    assert calls[0].id == "single" if with_id else calls[0].id.startswith("call_")
