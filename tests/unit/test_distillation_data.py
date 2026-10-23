"""Distillation data: strict records, JSONL round trips, prompt-disjoint splits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from stateset_agents.training.distillation import (
    DistillationDataset,
    DistillationExample,
    load_distillation_jsonl,
    parse_distillation_record,
    resolve_distillation_splits,
    split_distillation_dataset,
    write_distillation_jsonl,
)


def test_record_shapes() -> None:
    plain = parse_distillation_record({"prompt": "Q: 2+2=", "completion": "4"})
    assert plain.prompt == "Q: 2+2=" and plain.completion == "4"
    chat = parse_distillation_record(
        {
            "messages": [
                {"role": "system", "content": "Be brief."},
                {"role": "user", "content": "Refund?"},
                {"role": "assistant", "content": "Approved."},
            ],
            "weight": 2.0,
            "group_id": "case-1",
        }
    )
    assert [m["role"] for m in chat.prompt] == ["system", "user"]
    assert chat.completion == {"role": "assistant", "content": "Approved."}
    assert chat.weight == 2.0 and chat.group_id == "case-1"
    prompt_only = parse_distillation_record(
        {"messages": [{"role": "user", "content": "Hi"}]}
    )
    assert prompt_only.completion is None


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"prompt": "Q", "chosen": "a", "rejected": "b"}, "preference field"),
        ({"prompt": "Q", "completion": "a", "label": True}, "preference field"),
        ({"prompt": "Q", "messages": []}, "either prompt or messages"),
        ({"completion": "a"}, "prompt or messages"),
        ({"messages": [{"role": "user", "content": "Q"}], "completion": "a"}, "own"),
        ({"prompt": "", "completion": "a"}, "nonempty"),
        ({"prompt": "Q", "completion": "a", "weight": 0}, "weight"),
        ({"prompt": "Q", "completion": {"role": "user", "content": "a"}}, "assistant"),
        ({"prompt": "Q", "split": "dev"}, "split"),
        ([1, 2], "JSON object"),
    ],
)
def test_invalid_records_raise(row: object, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_distillation_record(row)


def test_jsonl_round_trip_and_strict_parsing(tmp_path: Path) -> None:
    dataset = DistillationDataset.from_records(
        [
            {"prompt": "a", "completion": "1"},
            {"prompt": [{"role": "user", "content": "b"}], "metadata": {"k": 1}},
            DistillationExample(prompt="c", completion="3", source_id="s"),
        ]
    )
    path = write_distillation_jsonl(dataset, tmp_path / "data.jsonl")
    assert load_distillation_jsonl(path) == dataset
    assert dataset.summary() == {
        "examples": 3,
        "with_completion": 2,
        "prompt_groups": 3,
    }
    assert not dataset.has_completions
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"prompt": "a", "completion": NaN}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_distillation_jsonl(bad)
    duplicate = tmp_path / "dup.jsonl"
    duplicate.write_text('{"prompt": "a", "prompt": "b"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_distillation_jsonl(duplicate)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_distillation_jsonl(empty)


def _records(count: int) -> list[dict]:
    return [
        {
            "prompt": [{"role": "user", "content": f"question {i % 6}"}],
            "completion": str(i),
        }
        for i in range(count)
    ]


def test_split_is_prompt_disjoint_deterministic_and_order_independent() -> None:
    dataset = DistillationDataset.from_records(_records(18))
    train, evaluation = split_distillation_dataset(dataset, 0.34, seed=3)
    train_ids = {example.identity_text for example in train.examples}
    eval_ids = {example.identity_text for example in evaluation.examples}
    assert not train_ids & eval_ids and len(train) + len(evaluation) == 18
    shuffled = DistillationDataset(tuple(reversed(dataset.examples)))
    assert split_distillation_dataset(shuffled, 0.34, seed=3) == (train, evaluation)
    with pytest.raises(ValueError, match="two independent"):
        split_distillation_dataset(DistillationDataset.from_records(_records(1)), 0.5)


def test_explicit_eval_data_must_not_overlap() -> None:
    train = _records(6)
    with pytest.raises(ValueError, match="overlap"):
        resolve_distillation_splits(train, [{"prompt": "QUESTION 1"}], 0.0, 0)
    split_train, split_eval = resolve_distillation_splits(train, None, 0.3, 0)
    assert split_eval is not None and len(split_train) + len(split_eval) == 6
    assert resolve_distillation_splits(train, None, 0.0, 0)[1] is None


def test_config_validation() -> None:
    from stateset_agents.training.distillation import DistillationConfig

    DistillationConfig().validate()
    assert DistillationConfig(divergence="reverse_kl").beta == 1.0
    assert DistillationConfig(divergence="jsd", jsd_beta=0.3).beta == 0.3
    cases = [
        ({"divergence": "kl"}, "divergence"),
        ({"divergence": "jsd", "jsd_beta": 0.0}, "jsd_beta"),
        ({"lmbda": 1.5}, "lmbda"),
        ({"on_policy_loss": "sampled"}, "lmbda > 0"),
        (
            {"lmbda": 1.0, "on_policy_loss": "sampled", "sampling_temperature": 0.7},
            "sampling_temperature",
        ),
        ({"lmbda": 1.0, "on_policy_loss": "sampled", "top_p": 0.9}, "top_p"),
        ({"teacher_top_k": 8, "lmbda": 0.5}, "live teacher"),
        ({"teacher_top_k": 0}, "teacher_top_k"),
        ({"vocab_alignment": "loose"}, "vocab_alignment"),
        ({"temperature": 0}, "temperature"),
        ({"top_p": 1.5}, "top_p"),
        ({"lora_target_modules": []}, "lora_target_modules"),
    ]
    for overrides, message in cases:
        with pytest.raises(ValueError, match=message):
            DistillationConfig(**overrides).validate()
    with pytest.raises(ValueError, match="unknown"):
        DistillationConfig.from_dict({"beta": 0.5})
    assert json.loads(json.dumps(DistillationConfig().to_dict()))["lmbda"] == 0.0
