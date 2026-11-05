"""JSON digests go through ``utils.provenance.json_sha256`` byte for byte.

Hashes of manifests, dataset and case fingerprints, transcript identities and
experiment digests are persisted and compared later, so consolidating the
hand-rolled ``hashlib.sha256(json.dumps(...))`` copies onto one helper must
not change a single digest. Each call site keeps the serialization it always
used (sorted or not, compact or not, ``allow_nan``, ``ensure_ascii``); this
file pins every combination in use against the expression it replaced, and
pins the call sites themselves.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

import pytest

from stateset_agents.utils.provenance import canonical_json, content_hash, json_sha256

#: Values that separate the settings: key order, nesting, non-ASCII text,
#: floats, integers that look like floats, empty containers, tuples.
SAMPLES: list[Any] = [
    {"b": 1, "a": [1, 2.5, {"z": None, "y": True}], "é": "naïve ☃"},
    [{"role": "user", "content": "Où est ma commande ?"}, {"role": "assistant"}],
    {"nested": {"k": [1.0, 2, 3e-9], "emoji": "\U0001f600"}, "": ""},
    [[1, 2, 3], [4, 5, 6]],
    ("tuple", 1, {"x": -0.0}),
    {},
    [],
    "plain",
    12345678901234567890,
]


def _legacy(**kwargs: Any) -> Callable[[Any], str]:
    def digest(value: Any) -> str:
        return hashlib.sha256(json.dumps(value, **kwargs).encode()).hexdigest()

    return digest


#: (json_sha256 settings, the expression call sites used before)
VARIANTS: list[tuple[dict[str, Any], Callable[[Any], str], str]] = [
    (
        {},
        _legacy(sort_keys=True, separators=(",", ":"), allow_nan=False),
        "content_hash, refund_policy_environment.case_fingerprint",
    ),
    (
        {"allow_nan": True},
        _legacy(sort_keys=True, separators=(",", ":")),
        "training.adapters nemo_rl/openrlhf/verl experiment digests",
    ),
    (
        {"compact": False, "allow_nan": True},
        _legacy(sort_keys=True),
        "remote.river_rl.run_fingerprint, data.refund_recovery transcript_hash",
    ),
    (
        {"compact": False, "allow_nan": True, "ensure_ascii": False},
        _legacy(sort_keys=True, ensure_ascii=False),
        "data.decontamination builtin sets, data.text_sft content_sha256",
    ),
    (
        {"compact": False, "ensure_ascii": False},
        _legacy(sort_keys=True, allow_nan=False, ensure_ascii=False),
        "data.curation._digest, data.source.SFTDatasetIdentity.from_rows",
    ),
    (
        {
            "sort_keys": False,
            "compact": False,
            "allow_nan": True,
            "ensure_ascii": False,
        },
        _legacy(ensure_ascii=False),
        "data.conversation_dataset._content_digest",
    ),
    (
        {"sort_keys": False, "compact": False, "allow_nan": True},
        _legacy(),
        "data.conversation_dataset._group_digest, distillation.cache.sequence_key",
    ),
]


@pytest.mark.parametrize(
    "settings,legacy,sites", VARIANTS, ids=[v[2].split(",")[0] for v in VARIANTS]
)
@pytest.mark.parametrize("value", SAMPLES, ids=range(len(SAMPLES)))
def test_every_setting_in_use_is_byte_identical(
    settings: dict[str, Any], legacy: Callable[[Any], str], sites: str, value: Any
) -> None:
    assert json_sha256(value, **settings) == legacy(value), sites


def test_content_hash_is_the_canonical_default() -> None:
    value = SAMPLES[0]
    assert content_hash(value) == json_sha256(value)
    assert canonical_json(value) == json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    # Pinned bytes: a change here silently invalidates every persisted identity.
    assert (
        content_hash({"b": 1, "a": [1, 2.5, None]})
        == hashlib.sha256(b'{"a":[1,2.5,null],"b":1}').hexdigest()
    )


def test_default_rejects_nonfinite_numbers_and_legacy_settings_keep_them() -> None:
    with pytest.raises(ValueError):
        content_hash({"x": float("nan")})
    assert json_sha256([float("inf")], allow_nan=True) == _legacy()([float("inf")])


def test_old_import_paths_re_export_the_canonical_helpers() -> None:
    from stateset_agents.evaluation import agent_runs, implementation
    from stateset_agents.remote import river_rl
    from stateset_agents.utils import atomic_io, provenance

    assert agent_runs.content_hash is provenance.content_hash
    assert implementation.package_implementation is provenance.package_implementation
    assert river_rl.atomic_json is atomic_io.atomic_json
    assert river_rl.exclusive_run is atomic_io.exclusive_run


# --- the call sites themselves ------------------------------------------------


def test_refund_case_fingerprint_is_unchanged() -> None:
    from stateset_agents.core.environments.refund_policy_environment import (
        _ORDER_FIELDS,
        RefundPolicyEnvironment,
        refund_policy_benchmark,
    )

    row = refund_policy_benchmark("validation", 1, 0)[0]
    facts = {key: row[key] for key in _ORDER_FIELDS if key != "order_id"}
    expected = hashlib.sha256(
        json.dumps(
            facts, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()
    assert RefundPolicyEnvironment.case_fingerprint(row) == expected


def test_curation_and_source_digests_are_unchanged() -> None:
    from stateset_agents.data import curation
    from stateset_agents.data.source import SFTDatasetIdentity

    rows = [{"messages": [{"role": "user", "content": "héllo"}], "id": 1}]
    legacy = hashlib.sha256(
        json.dumps(rows, sort_keys=True, allow_nan=False, ensure_ascii=False).encode()
    ).hexdigest()
    assert curation._digest(rows) == legacy
    assert SFTDatasetIdentity.from_rows(rows).content_sha256 == legacy
    with pytest.raises(ValueError, match="finite JSON"):
        SFTDatasetIdentity.from_rows([{"x": float("nan")}])


def test_river_run_fingerprint_is_unchanged(tmp_path) -> None:
    from types import SimpleNamespace

    from stateset_agents.remote.river_rl import RiverRLConfig, run_fingerprint

    dataset = tmp_path / "train.jsonl"
    dataset.write_text('{"messages": []}\n', encoding="utf-8")
    spec = SimpleNamespace(
        dataset=str(dataset),
        eval_prompts=["q"],
        base_model="m",
        lora_r=8,
        learning_rate=1e-4,
        eval_max_new_tokens=32,
    )
    config = RiverRLConfig()
    from dataclasses import asdict

    payload = {
        "dataset": hashlib.sha256(dataset.read_bytes()).hexdigest(),
        "eval": spec.eval_prompts,
        "model": spec.base_model,
        "rank": spec.lora_r,
        "lr": spec.learning_rate,
        "eval_max_new_tokens": spec.eval_max_new_tokens,
        "initial_adapter": None,
        "config": asdict(config),
        "scorer": "s",
    }
    expected = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    assert run_fingerprint(spec, config, "s", initial_checkpoint=None) == expected


def test_distillation_sequence_key_is_unchanged() -> None:
    from stateset_agents.training.distillation.cache import sequence_key

    class Sequence:
        prompt_ids = (1, 2, 3)
        response_ids = [4, 5]

    payload = json.dumps([[1, 2, 3], [4, 5]])
    assert (
        sequence_key(Sequence())  # type: ignore[arg-type]
        == hashlib.sha256(payload.encode()).hexdigest()
    )
