"""Training identity binds tokenization, configuration and code without Torch."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from stateset_agents.training.distributed_preflight import (
    fingerprint_encoded_rows,
    sft_preflight_identity,
    verify_sft_rank_consensus,
)


def _tokenizer(serialized=None):
    return SimpleNamespace(
        backend_tokenizer=SimpleNamespace(
            to_str=lambda: serialized
            or '{"model":{"vocab":{"a":0,"b":1}},"normalizer":null}'
        ),
        chat_template="{{ messages }}",
        special_tokens_map={"eos_token": "[EOS]"},
        padding_side="right",
        truncation_side="right",
    )


def _identity(tokenizer=None, **overrides):
    settings = {
        "model_id": "local/model",
        "revision": "a" * 40,
        "training_config": {"seed": 42, "learning_rate": 1e-5},
        "dataset": {"dataset_sha256": {"train": "b" * 64}, "train_examples": 1},
        "tokenizer": tokenizer or _tokenizer(),
        "encoded_data": {
            "train": fingerprint_encoded_rows(
                [{"input_ids": [1, 2], "labels": [-100, 2]}]
            )
        },
        "runtime_versions": {"torch": "2.11.0"},
        "implementation_modules": ["stateset_agents.data.text_sft"],
    }
    settings.update(overrides)
    return sft_preflight_identity(**settings)


def test_identity_construction_and_import_are_torch_free():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from stateset_agents.training.distributed_preflight import fingerprint_encoded_rows; import sys; assert fingerprint_encoded_rows([]); assert 'torch' not in sys.modules",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    identity = _identity()
    assert len(identity["implementation"]) == 2
    assert all(len(value) == 64 for value in identity["implementation"].values())


def test_tokenizer_identity_canonicalizes_object_key_order_but_detects_id_changes():
    first = _identity()
    same = _identity(_tokenizer('{"normalizer":null,"model":{"vocab":{"b":1,"a":0}}}'))
    different = _identity(
        _tokenizer('{"model":{"vocab":{"a":1,"b":0}},"normalizer":null}')
    )
    assert first == same
    assert first["tokenizer"] != different["tokenizer"]
    template = _tokenizer()
    template.chat_template = "different template"
    assert _identity(template)["tokenizer"] != first["tokenizer"]


def test_encoded_identity_binds_order_labels_and_record_boundaries():
    rows = [
        {"input_ids": [1, 2], "labels": [-100, 2]},
        {"input_ids": [3, 4], "labels": [-100, 4]},
    ]
    digest = fingerprint_encoded_rows(rows)
    assert digest == fingerprint_encoded_rows(iter(rows))
    assert digest != fingerprint_encoded_rows(reversed(rows))
    changed = copy.deepcopy(rows)
    changed[0]["labels"][0] = 1
    assert digest != fingerprint_encoded_rows(changed)
    assert fingerprint_encoded_rows([{}, {}]) != fingerprint_encoded_rows([{}])
    with pytest.raises(ValueError):
        fingerprint_encoded_rows([{"input_ids": [float("nan")]}])


def test_identity_captures_values_without_source_or_output_paths():
    identity = _identity()
    assert "path" not in json.dumps(identity)
    assert identity["model"] == "local/model"
    assert identity["training_config"] == {"seed": 42, "learning_rate": 1e-5}
    with pytest.raises(ValueError, match="serialized native fast tokenizer"):
        _identity(SimpleNamespace(backend_tokenizer=None, chat_template="template"))
    with pytest.raises((ValueError, ModuleNotFoundError)):
        _identity(implementation_modules=["stateset_agents.module_that_does_not_exist"])


def test_consensus_never_silently_falls_back_without_a_process_group(monkeypatch):
    dist = pytest.importorskip("torch.distributed")
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    with pytest.raises(RuntimeError, match="initialized process group"):
        verify_sft_rank_consensus(_identity(), world_size=2)
