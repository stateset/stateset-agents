"""Source-bound text SFT data, including file replacement during setup."""

import hashlib
import json
from pathlib import Path

import pytest

from stateset_agents.data.text_sft import load_text_sft_data, load_text_sft_snapshot


def _row(question="question", answer="answer", **extras):
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        **extras,
    }


def test_snapshot_hashes_exact_bytes_used_once_even_if_source_changes(
    tmp_path, monkeypatch
):
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    source = {
        train: (
            json.dumps(
                _row(answer="Merci à bientôt\u2028again", id="train"),
                ensure_ascii=False,
            )
            + "\r\n"
        ).encode(),
        validation: (
            json.dumps(_row(question="different", id="validation")) + "\n"
        ).encode(),
    }
    for path, raw in source.items():
        path.write_bytes(raw)
    reads = []
    original = Path.read_bytes

    def replace_after_read(path):
        raw = original(path)
        reads.append(path)
        if path in source:
            path.write_text(json.dumps(_row(question="replacement")) + "\n")
        return raw

    monkeypatch.setattr(Path, "read_bytes", replace_after_read)
    snapshot = load_text_sft_snapshot(str(train), str(validation))
    assert reads == [train, validation]
    assert snapshot.train[0]["messages"][1]["content"] == "Merci à bientôt\u2028again"
    assert snapshot.validation[0]["messages"][0]["content"] == "different"
    for name, path in (("train", train), ("validation", validation)):
        assert snapshot.dataset_sha256[name] == hashlib.sha256(source[path]).hexdigest()
    normalized = [{"messages": _row(answer="Merci à bientôt\u2028again")["messages"]}]
    assert (
        snapshot.content_sha256["train"]
        == hashlib.sha256(
            json.dumps(normalized, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()
    )
    preview = snapshot.describe()
    preview["dataset_sha256"]["train"] = "changed by caller"
    assert snapshot.dataset_sha256["train"] != "changed by caller"


def test_snapshot_semantic_identity_ignores_ids_and_whitespace(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text(json.dumps(_row(id="first")) + "\n")
    first = load_text_sft_snapshot(str(path))
    path.write_text(
        json.dumps(_row(id="second"), indent=None, separators=(",", ":")) + "\n\n"
    )
    second = load_text_sft_snapshot(str(path))
    assert first.dataset_sha256 != second.dataset_sha256
    assert first.content_sha256 == second.content_sha256
    assert first.validation is None
    assert first.describe()["validation_examples"] == 0


def test_snapshot_checks_overlap_on_captured_data(tmp_path):
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train.write_text(json.dumps(_row(id="train")) + "\n")
    validation.write_text(json.dumps(_row(id="validation")) + "\n")
    with pytest.raises(ValueError, match="overlap"):
        load_text_sft_snapshot(str(train), str(validation))


@pytest.mark.parametrize(
    "validation_question",
    [
        "where is my  ORDER 42?",
        " Where is my order 42?\n",
        "Where is my order\u00a042?",
        "\uff37here is my order 42?",  # full-width W
    ],
)
def test_overlap_check_normalizes_the_initial_request(tmp_path, validation_question):
    """Only byte-identical conversations were caught; a whitespace, case or
    Unicode-compatibility variant of a training request leaked into validation."""
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train.write_text(json.dumps(_row("Where is my order 42?", "It shipped.")) + "\n")
    validation.write_text(
        json.dumps(_row(validation_question, "It shipped today. ")) + "\n"
    )
    with pytest.raises(ValueError, match="overlap"):
        load_text_sft_snapshot(str(train), str(validation))


def test_distinct_requests_do_not_overlap(tmp_path):
    train, validation = tmp_path / "train.jsonl", tmp_path / "validation.jsonl"
    train.write_text(json.dumps(_row("Where is my order 42?")) + "\n")
    validation.write_text(json.dumps(_row("Where is my order 43?")) + "\n")
    assert load_text_sft_snapshot(str(train), str(validation)).validation


@pytest.mark.parametrize(
    "raw,reason",
    [
        (b"\n", "empty"),
        (b"\n{}\n", ":2:"),
        (b'{"messages":[{"role":"assistant","content":"a"}]}\n', "alternate"),
        (b'{"messages":[{"role":"user","content":["image"]}]}\n', "nonempty text"),
    ],
)
def test_invalid_snapshot_retains_validation(raw, reason, tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_bytes(raw)
    with pytest.raises(ValueError, match=reason):
        load_text_sft_data(str(path))
