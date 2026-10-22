"""Every JSONL reader keeps records whose strings contain Unicode line breaks.

``json.dumps(..., ensure_ascii=False)`` — how the repo writes JSONL — leaves
U+2028, U+2029 and U+0085 unescaped; ``str.splitlines`` split records on them,
so readers dropped (or rejected) valid rows and row counts in manifests were
wrong. JSON Lines separates records with ``"\\n"`` only.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from stateset_agents.data.conversation_dataset import (  # noqa: E402
    ConversationDataset,
    ConversationDatasetConfig,
)
from stateset_agents.data.finetuning import load_finetuning_data  # noqa: E402
from stateset_agents.data.jsonl import jsonl_lines  # noqa: E402
from stateset_agents.data.source import DatasetSourceIdentity  # noqa: E402
from stateset_agents.data.text_sft import parse_text_sft_records  # noqa: E402
from stateset_agents.training.lineage import hash_dataset  # noqa: E402
from stateset_agents.training.sft import load_sft_dataset  # noqa: E402

SCRIPT_DIR = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

import grade_transcript as gt  # noqa: E402
import prepare_sft_dataset as prep  # noqa: E402

#: Every character ``str.splitlines`` breaks on that JSON may leave raw.
SEPARATORS = (" ", " ", "\x85")
SETTINGS = settings(
    max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow]
)

texts = st.lists(
    st.one_of(st.sampled_from(SEPARATORS), st.sampled_from(["order", "42", "é", " "])),
    min_size=1,
    max_size=8,
).map(lambda parts: "x" + "".join(parts) + "y")


def _write(path: Path, records: list[dict], newline: str) -> bytes:
    raw = "".join(json.dumps(r, ensure_ascii=False) + newline for r in records)
    path.write_bytes(raw.encode("utf-8"))
    return raw.encode("utf-8")


def _chat(user: str, reply: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": reply},
        ]
    }


@SETTINGS
@given(
    st.lists(st.tuples(texts, texts), min_size=1, max_size=4),
    st.sampled_from(["\n", "\r\n"]),
)
def test_every_reader_keeps_every_record(tmp_path_factory, pairs, newline):
    tmp = tmp_path_factory.mktemp("jsonl")
    n = len(pairs)

    chat = tmp / "chat.jsonl"
    raw = _write(chat, [_chat(u, a) for u, a in pairs], newline)
    rows, identity = load_sft_dataset(chat)
    assert [r["messages"][0]["content"] for r in rows] == [u for u, _ in pairs]
    assert identity.source.rows == n
    assert DatasetSourceIdentity.from_bytes(chat, raw).rows == n
    assert hash_dataset(chat)[1] == n
    assert len(parse_text_sft_records(raw, str(chat))[0]) == n
    assert len(load_finetuning_data(chat)) == n

    curated = tmp / "curated.jsonl"
    _write(
        curated,
        [{"prompt": u, "response": a, "score": 0.9} for u, a in pairs],
        newline,
    )
    assert [e["prompt"] for e in prep.load_curated(curated)] == [u for u, _ in pairs]
    assert len(gt._existing_curated_keys(curated)[1]) == len(set(pairs))

    transcript = tmp / "transcript.jsonl"
    turns = [t for u, a in pairs for t in _chat(u, a)["messages"]]
    _write(transcript, turns, newline)
    assert gt.load_transcript(transcript) == turns
    contexts = tmp / "contexts.jsonl"
    _write(contexts, [{"gold": u} for u, _ in pairs], newline)
    assert [c["gold"] for c in gt.load_contexts(contexts)] == [u for u, _ in pairs]

    trajectories = tmp / "trajectories.jsonl"
    _write(
        trajectories,
        [{"turns": _chat(u, a)["messages"], "total_reward": 1.0} for u, a in pairs],
        newline,
    )
    config = ConversationDatasetConfig(normalize_rewards=False, compute_returns=False)
    assert len(ConversationDataset.from_jsonl(str(trajectories), config)) == n


def test_jsonl_lines_splits_on_newline_only_and_numbers_physical_lines():
    text = "a b\r\n\n  \nc\x85d\n"
    assert jsonl_lines(text) == [(1, "a b"), (4, "c\x85d")]
