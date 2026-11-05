"""Preference records, JSONL loading, prompt-disjoint splits and encoding."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from stateset_agents.data.identity import prompt_identity
from stateset_agents.training.preference.data import (
    PreferenceDataset,
    PreferencePair,
    UnpairedExample,
    check_preference_overlap,
    load_preference_jsonl,
    parse_preference_record,
    resolve_splits,
    split_preference_dataset,
    write_preference_jsonl,
)
from stateset_agents.training.preference.encoding import (
    PreferenceEncoder,
    render_plain_prompt,
)


def _write(path: Path, rows: list[object]) -> Path:
    path.write_text(
        "".join(
            (row if isinstance(row, str) else json.dumps(row, ensure_ascii=False))
            + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )
    return path


# ----------------------------------------------------------------- records


def test_record_shapes() -> None:
    raw = parse_preference_record({"prompt": "Q", "chosen": "a", "rejected": "b"})
    assert isinstance(raw, PreferencePair) and raw.prompt == "Q"
    chat = parse_preference_record(
        {
            "messages": [
                {"role": "system", "content": "s"},
                {"role": "user", "content": "q"},
            ],
            "chosen": {"role": "assistant", "content": "a"},
            "rejected": [{"role": "assistant", "content": "b"}],
            "margin": 1,
            "weight": 2,
            "group_id": "case-1",
        }
    )
    assert isinstance(chat, PreferencePair)
    assert chat.identity_text == "q"
    assert (chat.chosen_text, chat.rejected_text, chat.margin, chat.weight) == (
        "a",
        "b",
        1.0,
        2.0,
    )
    implicit = parse_preference_record(
        {
            "chosen": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a"},
            ],
            "rejected": [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "b"},
            ],
        }
    )
    assert isinstance(implicit, PreferencePair)
    assert implicit.prompt_value == ({"role": "user", "content": "q"},)
    unpaired = parse_preference_record(
        {"prompt": "Q", "completion": "a", "label": False}
    )
    assert isinstance(unpaired, UnpairedExample) and unpaired.label is False


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"prompt": "Q", "chosen": "a"}, "chosen and rejected"),
        ({"prompt": "Q", "chosen": "a", "rejected": "a"}, "identical"),
        ({"prompt": " ", "chosen": "a", "rejected": "b"}, "nonempty"),
        (
            {
                "prompt": [{"role": "system", "content": "s"}],
                "chosen": "a",
                "rejected": "b",
            },
            "user",
        ),
        (
            {
                "prompt": [
                    {"role": "user", "content": "q"},
                    {"role": "assistant", "content": "x"},
                ],
                "chosen": "a",
                "rejected": "b",
            },
            "assistant turn",
        ),
        ({"prompt": "Q", "chosen": "a", "rejected": "b", "weight": 0}, "positive"),
        ({"prompt": "Q", "chosen": "a", "rejected": "b", "weight": True}, "number"),
        ({"prompt": "Q", "chosen": "a", "rejected": "b", "margin": "1"}, "number"),
        ({"prompt": "Q", "chosen": "a", "rejected": "b", "group_id": ""}, "group_id"),
        ({"prompt": "Q", "messages": [], "chosen": "a", "rejected": "b"}, "not both"),
        ({"prompt": "Q", "completion": "a", "label": 1}, "boolean"),
        ({"prompt": "Q", "completion": "a", "label": True, "chosen": "b"}, "either"),
        (
            {
                "prompt": "Q",
                "chosen": {"role": "user", "content": "a"},
                "rejected": "b",
            },
            "assistant message",
        ),
        ({"prompt": "Q", "chosen": "a", "rejected": "b", "split": "dev"}, "split"),
    ],
)
def test_invalid_records_are_rejected(row: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_preference_record(row)


def test_records_are_private_copies() -> None:
    messages = [{"role": "user", "content": "q"}]
    pair = PreferencePair(prompt=messages, chosen="a", rejected="b")
    messages[0]["content"] = "changed"
    assert pair.identity_text == "q"


def test_kto_examples_unpair_every_pair() -> None:
    dataset = PreferenceDataset.from_records(
        [
            {"prompt": "Q", "chosen": "a", "rejected": "b", "weight": 2.0},
            {"prompt": "R", "completion": "c", "label": True},
        ]
    )
    examples = dataset.kto_examples()
    assert [(e.completion, e.label, e.weight) for e in examples] == [
        ("c", True, 1.0),
        ("a", True, 2.0),
        ("b", False, 2.0),
    ]


# ---------------------------------------------------------------- loading


def test_jsonl_round_trip(tmp_path: Path) -> None:
    rows = [
        {
            "prompt": "Q1",
            "chosen": "a",
            "rejected": "b",
            "margin": 0.5,
            "source_id": "s",
        },
        {
            "prompt": [{"role": "user", "content": "Q2 more"}],
            "chosen": "a",
            "rejected": "b",
            "metadata": {"rater": 3},
        },
        {"prompt": "Q3", "completion": "c", "label": True, "weight": 2.0},
    ]
    dataset = load_preference_jsonl(_write(tmp_path / "in.jsonl", rows))
    assert len(dataset) == 3 and len(dataset.unpaired) == 1
    assert dataset.pairs[1].identity_text == "Q2 more"  # not split on U+2028
    copy = load_preference_jsonl(
        write_preference_jsonl(dataset, tmp_path / "out.jsonl")
    )
    assert copy == dataset


@pytest.mark.parametrize(
    ("line", "message"),
    [
        ('{"prompt": "Q", "chosen": "a", "rejected": NaN}', "Non-finite"),
        ('{"prompt": "Q", "prompt": "R", "chosen": "a", "rejected": "b"}', "Duplicate"),
        ("[1, 2]", "object"),
        ("{not json", "line 2"),
    ],
)
def test_loader_is_strict_and_reports_lines(
    tmp_path: Path, line: str, message: str
) -> None:
    path = _write(
        tmp_path / "bad.jsonl", [{"prompt": "Q0", "chosen": "a", "rejected": "b"}, line]
    )
    with pytest.raises(ValueError, match=message):
        load_preference_jsonl(path)


def test_loader_enforces_declared_split(tmp_path: Path) -> None:
    path = _write(
        tmp_path / "train.jsonl",
        [{"prompt": "Q", "chosen": "a", "rejected": "b", "split": "validation"}],
    )
    with pytest.raises(ValueError, match="does not match"):
        load_preference_jsonl(path, expected_split="train")
    with pytest.raises(ValueError, match="empty"):
        load_preference_jsonl(_write(tmp_path / "empty.jsonl", ["", "  "]))


# ---------------------------------------------------------------- splits


def _identities(dataset: PreferenceDataset) -> set[str]:
    return {prompt_identity(record.identity_text) for record in dataset.records}


def _many(seed: int, count: int = 60) -> list[dict]:
    rng = random.Random(seed)
    rows = []
    for index in range(count):
        prompt = f"Request {rng.randrange(25)}"
        # case/whitespace variants are the same request
        if rng.random() < 0.3:
            prompt = "  " + prompt.upper() + "  "
        row: dict = {
            "prompt": [{"role": "user", "content": prompt}],
            "chosen": f"a{index}",
            "rejected": f"b{index}",
        }
        if rng.random() < 0.2:
            row["group_id"] = f"g{rng.randrange(5)}"
        rows.append(row)
    return rows


@pytest.mark.parametrize("seed", range(5))
def test_split_never_shares_a_prompt_group_or_source(seed: int) -> None:
    dataset = PreferenceDataset.from_records(_many(seed))
    train, evaluation = split_preference_dataset(dataset, 0.3, seed=seed)
    assert len(train) and len(evaluation)
    assert not _identities(train) & _identities(evaluation)

    def groups(dataset: PreferenceDataset) -> set[str | None]:
        return {r.group_id for r in dataset.records if r.group_id}

    assert not groups(train) & groups(evaluation)
    assert len(train) + len(evaluation) == len(dataset)


def test_split_is_deterministic_and_order_independent() -> None:
    rows = _many(11)
    first = split_preference_dataset(PreferenceDataset.from_records(rows), 0.25, seed=3)
    shuffled = list(rows)
    random.Random(0).shuffle(shuffled)
    second = split_preference_dataset(
        PreferenceDataset.from_records(shuffled), 0.25, seed=3
    )

    def key(dataset: PreferenceDataset) -> list[str]:
        return sorted(json.dumps(r.to_dict(), sort_keys=True) for r in dataset.records)

    assert key(first[0]) == key(second[0]) and key(first[1]) == key(second[1])


def test_split_links_records_transitively() -> None:
    dataset = PreferenceDataset.from_records(
        [
            {"prompt": "A", "chosen": "1", "rejected": "2", "source_id": "s1"},
            {
                "prompt": "B",
                "chosen": "1",
                "rejected": "2",
                "source_id": "s1",
                "group_id": "g",
            },
            {"prompt": "C", "chosen": "1", "rejected": "2", "group_id": "g"},
            {"prompt": "D", "chosen": "1", "rejected": "2"},
        ]
    )
    train, evaluation = split_preference_dataset(dataset, 0.5, seed=0)
    sides = [{r.identity_text for r in part.records} for part in (train, evaluation)]
    assert {"A", "B", "C"} in sides and {"D"} in sides


def test_split_requires_two_groups_and_valid_arguments() -> None:
    same = PreferenceDataset.from_records(
        [{"prompt": "Same", "chosen": str(i), "rejected": "x"} for i in range(4)]
    )
    with pytest.raises(ValueError, match="two independent"):
        split_preference_dataset(same, 0.5)
    with pytest.raises(ValueError, match="validation_fraction"):
        split_preference_dataset(same, 1.0)
    with pytest.raises(ValueError, match="seed"):
        split_preference_dataset(same, 0.5, seed=1.5)  # type: ignore[arg-type]


def test_overlap_check_and_resolve_splits(tmp_path: Path) -> None:
    train = PreferenceDataset.from_records(
        [{"prompt": "Hello  World", "chosen": "a", "rejected": "b"}]
    )
    leaked = PreferenceDataset.from_records(
        [{"prompt": "hello world", "chosen": "c", "rejected": "d"}]
    )
    with pytest.raises(ValueError, match="overlap"):
        check_preference_overlap(train, leaked)
    with pytest.raises(ValueError, match="overlap"):
        resolve_splits(train, leaked, 0.0, 0)
    path = _write(tmp_path / "d.jsonl", _many(2))
    full, held = resolve_splits(path, None, 0.2, 0)
    assert held is not None and not _identities(full) & _identities(held)
    assert resolve_splits(path, None, 0.0, 0)[1] is None


# --------------------------------------------------------------- encoding


@pytest.fixture(scope="module")
def tokenizer():
    pytest.importorskip("transformers")
    from tests._tiny_tokenizer import tiny_tokenizer

    return tiny_tokenizer(103)


def _ids(tokenizer, text: str) -> list[int]:
    return list(tokenizer(text, add_special_tokens=False)["input_ids"])


def test_plain_format_matches_the_agent_prompt(tokenizer) -> None:
    encoder = PreferenceEncoder(tokenizer, max_length=128)
    messages = [
        {"role": "system", "content": "Be brief"},
        {"role": "user", "content": "hi"},
    ]
    sequence = encoder.encode(messages, "42")
    assert render_plain_prompt(messages) == "System: Be brief\nUser: hi\nAssistant: "
    assert list(sequence.prompt_ids) == _ids(
        tokenizer, "System: Be brief\nUser: hi\nAssistant: "
    )
    assert list(sequence.response_ids) == _ids(tokenizer, "42") + [
        tokenizer.eos_token_id
    ]
    assert sequence.response_mask == [0] * len(sequence.prompt_ids) + [1] * 3
    with pytest.raises(ValueError, match="plain prompt format"):
        encoder.encode(
            [{"role": "user", "content": "q"}, {"role": "tool", "content": "x"}], "a"
        )


def test_raw_format_and_eos_toggle(tokenizer) -> None:
    sequence = PreferenceEncoder(tokenizer, append_eos=False).encode("Q: 1+1=", "2")
    assert list(sequence.prompt_ids) == _ids(tokenizer, "Q: 1+1=")
    assert list(sequence.response_ids) == _ids(tokenizer, "2")
    with pytest.raises(ValueError, match="raw prompt strings"):
        PreferenceEncoder(tokenizer, prompt_format="raw").encode(
            [{"role": "user", "content": "q"}], "a"
        )


def test_chat_template_masks_only_the_final_assistant_turn(tokenizer) -> None:
    import copy

    templated = copy.deepcopy(tokenizer)
    templated.chat_template = (
        "{% for m in messages %}<{{ m['role'] }}>{{ m['content'] }}<eos>{% endfor %}"
        "{% if add_generation_prompt %}<assistant>{% endif %}"
    )
    encoder = PreferenceEncoder(templated)
    sequence = encoder.encode([{"role": "user", "content": "hi"}], "ok")
    assert (
        encoder.resolve_format([{"role": "user", "content": "hi"}]) == "chat_template"
    )
    eos = templated.eos_token_id
    assert list(sequence.prompt_ids) == _ids(templated, "<user>hi") + [eos] + _ids(
        templated, "<assistant>"
    )
    assert list(sequence.response_ids) == _ids(templated, "ok") + [eos]


def test_truncation_keeps_the_prompt_end_and_response_start(tokenizer) -> None:
    encoder = PreferenceEncoder(
        tokenizer, max_length=8, max_prompt_length=4, prompt_format="raw"
    )
    sequence = encoder.encode("abcdefgh", "123456")
    assert sequence.truncated
    assert list(sequence.prompt_ids) == _ids(tokenizer, "efgh")
    assert list(sequence.response_ids) == _ids(tokenizer, "1234")
    tight = PreferenceEncoder(tokenizer, max_length=3, prompt_format="raw").encode(
        "abcdefgh", "12"
    )
    assert (
        list(tight.prompt_ids) == _ids(tokenizer, "gh") and len(tight.response_ids) == 1
    )
    with pytest.raises(ValueError, match="smaller"):
        PreferenceEncoder(tokenizer, max_length=4, max_prompt_length=4)
    with pytest.raises(ValueError, match="prompt_format"):
        PreferenceEncoder(tokenizer, prompt_format="jinja")
    with pytest.raises(ValueError, match="chat template"):
        PreferenceEncoder(tokenizer, prompt_format="chat_template")
