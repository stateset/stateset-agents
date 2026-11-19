"""Regression coverage for grading an assistant's parallel tool calls and receipts."""

import json

import pytest

from stateset_agents.cli_improve import run_improve
from stateset_agents.data import grade_transcript as grading
from stateset_agents.data import prepare_sft_dataset as prepare
from stateset_agents.mcp_server import grade_transcript as mcp_grade


@pytest.fixture
def orders():
    calls = [
        {
            "id": f"lookup-{order}",
            "type": "function",
            "function": {
                "name": "lookup_order",
                "arguments": json.dumps({"order": order}),
            },
        }
        for order in (42, 19)
    ]
    history = [
        {"role": "user", "content": "Check both orders."},
        {"role": "assistant", "content": "", "tool_calls": calls},
        {"role": "tool", "tool_call_id": "lookup-19", "content": '{"state":"pending"}'},
        {"role": "tool", "tool_call_id": "lookup-42", "content": '{"state":"shipped"}'},
        {"role": "assistant", "content": "42 shipped and 19 is pending."},
    ]
    expected = [
        {"tool": "lookup_order", "args": {"order": 19}, "result": {"state": "pending"}},
        {"tool": "lookup_order", "args": {"order": 42}, "result": {"state": "shipped"}},
    ]
    return history, expected


@pytest.mark.asyncio
async def test_parallel_order_lookups_reach_curation_and_supervised_tokens(
    orders, tmp_path
):
    from stateset_agents.cli_improve import run_improve_async
    from stateset_agents.remote.river_batches import build_sft_batch
    from stateset_agents.training.sft import tokenize_sft_row
    from tests.unit.test_loss_computation_real_tokenizer_masks import _byte_tokenizer
    from tests.unit.test_river_sft_masking import QWEN3_LIKE

    history, expected = orders
    source = tmp_path / "orders.jsonl"
    source.write_text(json.dumps({"messages": history}) + "\n")
    refs = [
        {
            "expected_tool_calls": [
                {"tool": r["tool"], "args": r["args"]} for r in expected
            ]
        },
        {"expected_tool_calls": expected, "expected_outcome": "42 shipped"},
    ]
    context = tmp_path / "references.jsonl"
    context.write_text("".join(json.dumps(ref) + "\n" for ref in refs))
    output = tmp_path / "out"
    summary = await run_improve_async(
        transcripts=str(source),
        format="openai",
        reward="tool_calling",
        context_file=str(context),
        output=str(output),
    )
    report = await mcp_grade(
        str(output / "ingested" / "conversation_0.jsonl"), "tool_calling", str(context)
    )
    assert summary["curated_count"] == 2
    assert [r["score"] for r in report["rows"]] == [1, 1]
    action, reply = [
        prepare.to_chat(row) for row in prepare.load_curated(output / "curated.jsonl")
    ]
    assert len(action["messages"]) == 2
    assert reply["messages"][1]["weight"] == 0
    assert "expected_tool_calls" not in json.dumps(reply)

    template = (
        "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}"
        "{% for c in m.tool_calls | default([]) %}<call>{{ c.function.name }} "
        "{{ c.function.arguments.order }}</call>{% endfor %}<|im_end|>\n{% endfor %}"
        "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}"
    )
    tok = _byte_tokenizer(template)

    def supervised(row):
        encoded = tokenize_sft_row(tok, row, max_length=4096)
        assert encoded is not None
        return tok.decode([t for t in encoded["labels"] if t != -100])

    calls = supervised(action)
    assert (
        "<call>lookup_order 42</call>" in calls
        and "<call>lookup_order 19</call>" in calls
    )
    rendered = tok.apply_chat_template(action["messages"], tokenize=False)
    first_call_end = rendered.index("</call>") + len("</call>")
    cutoff = len(tok.encode(rendered[:first_call_end], add_special_tokens=False))
    assert tokenize_sft_row(tok, action, max_length=cutoff) is None
    for shifted in (False, True):
        assert (
            build_sft_batch([action], tok, max_length=cutoff, shift_targets=shifted)
            == []
        )
    reply_labels = supervised(reply)
    assert (
        "42 shipped and 19 is pending." in reply_labels and "<call>" not in reply_labels
    )

    first_call_only = _byte_tokenizer(QWEN3_LIKE)
    for row in (action, reply):
        with pytest.raises(ValueError, match="omits a parallel tool call"):
            tokenize_sft_row(first_call_only, row, max_length=4096)

    # A template may reject a comparison that leaves an orphaned tool result.
    strict = (
        "{% for m in messages if m.role == 'tool' %}"
        "{% if m.tool_call_id not in messages[1].tool_calls | map(attribute='id') | list %}"
        "{{ raise_exception('orphaned result') }}{% endif %}{% endfor %}"
    ) + template
    assert tokenize_sft_row(_byte_tokenizer(strict), reply, max_length=4096) is not None


@pytest.mark.parametrize(
    ("change", "score"),
    [
        ("swapped_results", 0),
        ("wrong_type", 0),
        ("extra_call", 0),
        ("missing_result", None),
        ("orphan_result", None),
        ("duplicate_result", None),
        ("duplicate_call_id", None),
        ("missing_call_id", None),
        ("intervening_user", None),
        ("previous_batch", None),
    ],
)
@pytest.mark.asyncio
async def test_complete_mismatch_vs_incomplete_evidence(orders, change, score):
    history, expected = orders
    if change == "swapped_results":
        history[2]["content"], history[3]["content"] = (
            history[3]["content"],
            history[2]["content"],
        )
    elif change == "wrong_type":
        expected[0]["args"]["order"] = "19"
    elif change == "extra_call":
        expected.pop()
    elif change == "missing_result":
        history.pop(2)
    elif change == "orphan_result":
        history[2]["tool_call_id"] = "another-run"
    elif change == "duplicate_result":
        history[2]["tool_call_id"] = history[3]["tool_call_id"]
    elif change == "duplicate_call_id":
        history[1]["tool_calls"][0]["id"] = "lookup-19"
    elif change == "missing_call_id":
        history[1]["tool_calls"][0].pop("id")
    elif change == "intervening_user":
        history.insert(-1, {"role": "user", "content": "Start a different order."})
    elif change == "previous_batch":
        history.insert(
            -1,
            {
                "role": "assistant",
                "content": "",
                "tool_calls": history[1]["tool_calls"],
            },
        )
    contexts = [{} for m in history if m["role"] == "assistant"]
    contexts[-1] = {"expected_tool_calls": expected}
    rows = await grading.grade_transcript(
        history, contexts, grading.get_reward("tool_calling")
    )
    assert rows[-1]["score"] == score
    assert rows[-1]["status"] == ("ungradable" if score is None else "graded")


@pytest.mark.parametrize(
    "context",
    [
        {"expected_tool_calls": []},
        {"expected_tool_calls": [{"tool": "lookup_order", "args": []}]},
        {
            "expected_tool_calls": [
                {"tool": "a", "args": {}, "result": 0},
                {"tool": "b", "args": {}},
            ]
        },
        {
            "expected_tool_calls": [{"tool": "a", "args": {}}],
            "expected_tool_result": None,
        },
        {"expected_tool_calls": [{"tool": "a", "args": {}, "reslt": 1}]},
    ],
)
def test_invalid_batch_reference_is_ungradable_at_zero_threshold(
    orders, tmp_path, context
):
    history, _ = orders
    source = tmp_path / "orders.jsonl"
    source.write_text(json.dumps({"messages": history}))
    refs = tmp_path / "refs.jsonl"
    refs.write_text("{}\n" + json.dumps(context))
    summary = run_improve(
        transcripts=str(source),
        format="openai",
        reward="tool_calling",
        threshold=0,
        context_file=str(refs),
        output=str(tmp_path / "out"),
    )
    assert summary["curated_count"] == 0
    assert summary["ungradable_turn_count"] == 2
    assert summary["mean_score"] is None


@pytest.mark.asyncio
async def test_repeated_identical_calls_require_distinct_receipts(orders):
    history, expected = orders
    expected[0]["args"] = expected[1]["args"]
    history[1]["tool_calls"][1]["function"]["arguments"] = '{"order":42}'
    # The parsed number can match either receipt. The literal text fits only one.
    expected[0]["result"], expected[1]["result"] = 42, "42"
    history[2]["content"], history[3]["content"] = "42", "42.0"
    refs = [{}, {"expected_tool_calls": expected}]
    reward = grading.get_reward("tool_calling")
    assert (await grading.grade_transcript(history, refs, reward))[-1]["score"] == 1
    expected[0]["result"] = "42"
    assert (await grading.grade_transcript(history, refs, reward))[-1]["score"] == 0
