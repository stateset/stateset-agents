"""Split integrity, checked with Hypothesis.

Every place that separates training data from the data that measures it —
``ConversationDataset.split``, ``split_finetuning_data``, text SFT snapshots,
and eval-ladder / flywheel prompt generation — must be deterministic for a
seed, independent of input order, disjoint under the repo's normalization
(NFKC, casefold and whitespace collapsing, plus digit masking where
reference numbers are the only difference), and must keep linked records
(content duplicates, ``group_id`` / ``source_id`` groups, issue
combinations) on one side.
"""

from __future__ import annotations

import itertools
import json
import random
import re
import tempfile
import unicodedata
from pathlib import Path
from typing import Any

import pytest

hypothesis = pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from stateset_agents.data.conversation_dataset import (  # noqa: E402
    ConversationDataset,
    ConversationDatasetConfig,
)
from stateset_agents.data.finetuning import (  # noqa: E402
    group_finetuning_data,
    split_finetuning_data,
)
from stateset_agents.data.identity import prompt_identity  # noqa: E402
from stateset_agents.data.text_sft import load_text_sft_snapshot  # noqa: E402
from stateset_agents.flywheel import FlywheelConfig  # noqa: E402
from stateset_agents.training.eval_ladder import (  # noqa: E402
    DomainSpec,
    Issue,
    build_episode_ladder,
    build_ladder,
)

SETTINGS = settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
WORDS = ("refund", "late", "broken", "missing", "charge", "address", "login")
POOLS = ("alpha", "bravo", "charlie")


def norm(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def masked(text: str) -> str:
    return re.sub(r"\d+", "#", norm(text))


@st.composite
def variant(draw: Any, text: str) -> str:
    """``text`` as a user might type it again: case, spacing, Unicode forms."""
    out = draw(st.sampled_from([text, text.upper(), text.title()]))
    if draw(st.booleans()):
        out = out.replace(" ", draw(st.sampled_from(["  ", " ", "\t", " \n"])))
    if draw(st.booleans()):
        out = "".join(chr(ord(c) + 0xFEE0) if "a" <= c <= "z" else c for c in out)
    return draw(st.sampled_from(["", " ", "　"])) + out


@st.composite
def requests(draw: Any) -> str:
    words = draw(st.lists(st.sampled_from(WORDS), min_size=1, max_size=3))
    return draw(variant(" ".join(words)))


# --------------------------------------------------------------------------- #
# ConversationDataset.split
# --------------------------------------------------------------------------- #


@st.composite
def trajectories(draw: Any) -> list[dict[str, Any]]:
    rows = []
    for i in range(draw(st.integers(min_value=1, max_value=14))):
        metadata = {}
        if draw(st.booleans()):
            metadata["group_id"] = draw(st.sampled_from(POOLS))
        if draw(st.booleans()):
            metadata["source_id"] = draw(variant(draw(st.sampled_from(POOLS))))
        rows.append(
            {
                "trajectory_id": f"t{i}",
                "turns": [
                    {"role": "user", "content": draw(requests())},
                    {"role": "assistant", "content": draw(requests())},
                ],
                "total_reward": draw(st.floats(0, 1)),
                "metadata": metadata,
            }
        )
    return rows


def _dataset(rows: list[dict[str, Any]]) -> ConversationDataset:
    config = ConversationDatasetConfig(normalize_rewards=False, compute_returns=False)
    return ConversationDataset.from_dict({"trajectories": rows}, config)


def _ids(subset: ConversationDataset) -> list[str]:
    return [t.trajectory_id for t in subset.trajectories]


def _content_key(row: dict[str, Any]) -> str:
    return json.dumps([(t["role"], norm(t["content"])) for t in row["turns"]])


@SETTINGS
@given(
    trajectories(),
    st.integers(0, 2**16),
    st.sampled_from([0.2, 0.5, 0.8]),
    st.randoms(use_true_random=False),
)
def test_conversation_split_is_reproducible_disjoint_and_group_preserving(
    rows: list[dict[str, Any]], seed: int, ratio: float, rng: random.Random
) -> None:
    train, val = _dataset(rows).split(train_ratio=ratio, seed=seed)
    again = _dataset(rows).split(train_ratio=ratio, seed=seed)
    assert (_ids(train), _ids(val)) == (_ids(again[0]), _ids(again[1]))

    shuffled = list(rows)
    rng.shuffle(shuffled)
    other = _dataset(shuffled).split(train_ratio=ratio, seed=seed)
    assert set(_ids(train)) == set(_ids(other[0]))
    assert set(_ids(val)) == set(_ids(other[1]))

    by_id = {row["trajectory_id"]: row for row in rows}
    assert sorted(_ids(train) + _ids(val)) == sorted(by_id)
    for key in (
        _content_key,
        lambda row: prompt_identity(row["turns"][0]["content"]),
        lambda row: row["metadata"].get("group_id"),
        lambda row: norm(row["metadata"].get("source_id", "")) or None,
    ):
        train_keys = {key(by_id[i]) for i in _ids(train)} - {None}
        val_keys = {key(by_id[i]) for i in _ids(val)} - {None}
        assert not train_keys & val_keys

    unlinked = not any(row["metadata"] for row in rows)
    # Without metadata links, groups are exactly the distinct requests.
    requests = {prompt_identity(row["turns"][0]["content"]) for row in rows}
    if unlinked and len(requests) >= 2:
        assert _ids(train) and _ids(val)


def test_conversation_split_default_seed_is_fixed() -> None:
    rows = [
        {
            "trajectory_id": f"t{i}",
            "turns": [
                {"role": "user", "content": f"question {i}"},
                {"role": "assistant", "content": "answer"},
            ],
        }
        for i in range(30)
    ]
    first = _dataset(rows).split()
    assert all(
        (_ids(first[0]), _ids(first[1])) == tuple(map(_ids, _dataset(rows).split()))
        for _ in range(3)
    )


# --------------------------------------------------------------------------- #
# split_finetuning_data and text SFT snapshots
# --------------------------------------------------------------------------- #


@st.composite
def chat_rows(draw: Any) -> list[dict[str, Any]]:
    rows = []
    for _ in range(draw(st.integers(min_value=2, max_value=14))):
        row: dict[str, Any] = {
            "messages": [
                {"role": "user", "content": draw(requests())},
                {"role": "assistant", "content": draw(requests())},
            ]
        }
        if draw(st.integers(0, 3)) == 0:
            row["group_id"] = draw(st.sampled_from(POOLS))
        rows.append(row)
    return rows


def _prompt(row: dict[str, Any]) -> str:
    return norm(row["messages"][0]["content"])


def _canon(rows: list[dict[str, Any]]) -> list[str]:
    return [json.dumps(row, sort_keys=True) for row in rows]


@SETTINGS
@given(chat_rows(), st.integers(0, 2**16), st.randoms(use_true_random=False))
def test_finetuning_split_is_reproducible_disjoint_and_group_preserving(
    rows: list[dict[str, Any]], seed: int, rng: random.Random
) -> None:
    assume(len(group_finetuning_data(rows)) >= 2)
    train, val = split_finetuning_data(rows, validation_fraction=0.3, seed=seed)
    assert (_canon(train), _canon(val)) == tuple(
        map(_canon, split_finetuning_data(rows, validation_fraction=0.3, seed=seed))
    )
    shuffled = list(rows)
    rng.shuffle(shuffled)
    assert (_canon(train), _canon(val)) == tuple(
        map(
            _canon,
            split_finetuning_data(shuffled, validation_fraction=0.3, seed=seed),
        )
    )
    assert train and val
    assert sorted(_canon(train) + _canon(val)) == sorted(set(_canon(rows)))
    assert not {_prompt(r) for r in train} & {_prompt(r) for r in val}
    assert not {r.get("group_id") for r in train} & {r.get("group_id") for r in val} - {
        None
    }


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps({"messages": r["messages"]}) + "\n" for r in rows),
        encoding="utf-8",
    )


@SETTINGS
@given(chat_rows(), st.integers(0, 2**16))
def test_text_sft_accepts_every_split_the_splitter_produces(
    rows: list[dict[str, Any]], seed: int
) -> None:
    """The splitter and the snapshot leakage check share one identity."""
    assume(len(group_finetuning_data(rows)) >= 2)
    train, val = split_finetuning_data(rows, validation_fraction=0.3, seed=seed)
    with tempfile.TemporaryDirectory() as tmp:
        train_path, val_path = Path(tmp, "train.jsonl"), Path(tmp, "val.jsonl")
        _write(train_path, train)
        _write(val_path, val)
        snapshot = load_text_sft_snapshot(str(train_path), str(val_path))
        again = load_text_sft_snapshot(str(train_path), str(val_path))
    assert snapshot.content_sha256 == again.content_sha256
    assert len(snapshot.train) == len(train) and len(snapshot.validation) == len(val)


@SETTINGS
@given(chat_rows(), st.data())
def test_text_sft_rejects_a_normalized_repeat_of_a_training_request(
    rows: list[dict[str, Any]], data: Any
) -> None:
    source = data.draw(st.sampled_from(rows))
    repeat = {
        "messages": [
            {
                "role": "user",
                "content": data.draw(variant(_prompt(source))),
            },
            {"role": "assistant", "content": "a different answer"},
        ]
    }
    with tempfile.TemporaryDirectory() as tmp:
        train_path, val_path = Path(tmp, "train.jsonl"), Path(tmp, "val.jsonl")
        _write(train_path, rows)
        _write(val_path, [repeat])
        with pytest.raises(ValueError, match="overlap"):
            load_text_sft_snapshot(str(train_path), str(val_path))


# --------------------------------------------------------------------------- #
# Eval ladder and flywheel prompt sets
# --------------------------------------------------------------------------- #

ISSUE_NAMES = ("bag", "seat", "meal", "visa", "wifi", "gate", "fare")


@st.composite
def domains(draw: Any) -> DomainSpec:
    names = draw(
        st.lists(
            st.sampled_from(ISSUE_NAMES),
            min_size=2,
            max_size=len(ISSUE_NAMES),
            unique=True,
        )
    )
    return DomainSpec(
        persona="Ava",
        ref_label="Booking {ref}",
        issues={
            name: Issue(
                phrasing=f"my {name} is wrong",
                resolution=f"I fixed your {name}.",
                token=f"{name} fix",
                refusal=f"leave my {name} alone",
            )
            for name in names
        },
    )


def _reordered(spec: DomainSpec, rng: random.Random) -> DomainSpec:
    items = list(spec.issues.items())
    rng.shuffle(items)
    return DomainSpec(
        persona=spec.persona, issues=dict(items), ref_label=spec.ref_label
    )


@SETTINGS
@given(
    domains(),
    st.data(),
    st.integers(0, 2**16),
    st.randoms(use_true_random=False),
)
def test_ladder_eval_and_harvest_are_disjoint_and_reproducible(
    spec: DomainSpec, data: Any, seed: int, rng: random.Random
) -> None:
    n = len(spec.issues)
    depth = data.draw(st.integers(2, min(4, n)))
    refusals = data.draw(st.sampled_from([0.0, 0.5, 1.0])) if n > depth else 0.0
    counts = {
        "eval_count": data.draw(st.integers(1, 30)),
        "harvest_count": data.draw(st.integers(1, 40)),
        "train_count": data.draw(st.integers(0, 40)),
    }
    kit = build_ladder(
        spec, depth=depth, refusal_fraction=refusals, seed=seed, **counts
    )
    assert kit == build_ladder(
        spec, depth=depth, refusal_fraction=refusals, seed=seed, **counts
    )
    assert kit == build_ladder(
        _reordered(spec, rng),
        depth=depth,
        refusal_fraction=refusals,
        seed=seed,
        **counts,
    )

    # Train, eval and harvest are mutually disjoint under the shared request
    # identity with reference digits masked.
    eval_keys = {masked(p["prompt"]) for p in kit["eval"]}
    harvest_keys = {masked(p["prompt"]) for p in kit["harvest"]}
    train_keys = {masked(r["messages"][0]["content"]) for r in kit["train"]}
    assert not eval_keys & harvest_keys
    assert not eval_keys & train_keys
    assert not harvest_keys & train_keys
    if len(list(itertools.combinations(spec.issues, depth))) >= 2:
        # Group-preserving: an issue combination lives on one side only.
        eval_sets = {frozenset(p["expect"][:-1]) for p in kit["eval"]}
        assert not eval_sets & {frozenset(p["expect"][:-1]) for p in kit["harvest"]}

    eval_refs = {p["expect"][-1] for p in kit["eval"]}
    harvest_refs = {p["expect"][-1] for p in kit["harvest"]}
    train_refs = {
        re.search(r"Booking (\d+)", r["messages"][0]["content"]).group(1)
        for r in kit["train"]
    }
    assert len(eval_refs) == counts["eval_count"]
    assert len(harvest_refs) == counts["harvest_count"]
    assert not (eval_refs & harvest_refs or train_refs & (eval_refs | harvest_refs))

    config = FlywheelConfig(
        base_model="base/model",
        harvest_prompts=kit["harvest"],
        eval_prompts=kit["eval"],
    )
    assert config.eval_prompts == kit["eval"]


@SETTINGS
@given(domains(), st.data(), st.integers(0, 2**16))
def test_episode_ladder_eval_and_harvest_are_disjoint_and_reproducible(
    spec: DomainSpec, data: Any, seed: int
) -> None:
    turns = data.draw(st.integers(2, min(4, len(spec.issues))))
    counts = {
        "eval_count": data.draw(st.integers(1, 20)),
        "harvest_count": data.draw(st.integers(1, 30)),
    }
    refusals = data.draw(st.sampled_from([0.0, 0.5]))
    kit = build_episode_ladder(
        spec, turns=turns, refusal_fraction=refusals, seed=seed, **counts
    )
    assert kit == build_episode_ladder(
        spec, turns=turns, refusal_fraction=refusals, seed=seed, **counts
    )

    def key(script: dict[str, Any]) -> tuple[str, ...]:
        return tuple(masked(turn) for turn in script["turns"])

    assert not {key(s) for s in kit["eval"]} & {key(s) for s in kit["harvest"]}
    eval_refs = {s["turn_expect"][0][-1] for s in kit["eval"]}
    harvest_refs = {s["turn_expect"][0][-1] for s in kit["harvest"]}
    assert len(eval_refs) == counts["eval_count"]
    assert not eval_refs & harvest_refs
    FlywheelConfig(
        base_model="base/model",
        harvest_prompts=kit["harvest"],
        eval_prompts=kit["eval"],
    )
