"""Mutable pointer paths must not conceal a change to an RL starting model."""

import asyncio
import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.river import (
    CHECKPOINT_POINTER_NAME,
    _checkpoint_from_pointer,
)
from tests.unit.test_river_rl_contract import _executor
from tests.unit.test_river_submit_golden import stub_river_renderers  # noqa: F401
from tests.unit.test_river_submit_golden import (
    scenario_episode_harvest,
    scenario_episode_rl,
    scenario_harvest,
    scenario_rl,
)


def write_pointer(path, value):
    path.mkdir(exist_ok=True)
    (path / CHECKPOINT_POINTER_NAME).write_text(json.dumps(value))
    return str(path)


@pytest.mark.parametrize(
    "value",
    [
        None,
        3,
        [],
        {},
        {"checkpoint": None},
        {"checkpoint": "https://example.com/model"},
        {"checkpoint": "river://"},
        {"checkpoint": "river://bad model"},
        {"checkpoint": "river://bad\u0000model"},
        {"checkpoint": "river://valid", "provider": "other"},
        {"checkpoint": "river://valid", "base_model": "different"},
        {"checkpoint": "river://valid", "base_model": None},
        {"checkpoint": "river://valid", "lora": None},
        {"checkpoint": "river://valid", "lora": {"rank": True}},
        {"checkpoint": "river://valid", "lora": {"rank": 32}},
    ],
)
def test_invalid_pointer_is_rejected_without_modifying_it(tmp_path, value):
    reference = write_pointer(tmp_path / "pointer", value)
    path = tmp_path / "pointer" / CHECKPOINT_POINTER_NAME
    before = path.read_bytes()
    with pytest.raises(RemoteExecutionError):
        _checkpoint_from_pointer(reference, base_model="base", lora_rank=16)
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    "raw",
    [
        '{"checkpoint":"river://first","checkpoint":"river://second"}',
        '{"checkpoint":"river://valid","lora":{"rank":16,"rank":32}}',
    ],
)
def test_duplicate_pointer_fields_are_rejected(tmp_path, raw):
    (tmp_path / CHECKPOINT_POINTER_NAME).write_text(raw)
    with pytest.raises(RemoteExecutionError, match="unambiguous"):
        _checkpoint_from_pointer(tmp_path)


@pytest.mark.parametrize(
    "reference", ["", " ", False, 0, [], "river://", "river://x\n", "river://x\x7f"]
)
def test_invalid_direct_reference_cannot_fall_back_to_base(reference):
    with pytest.raises(RemoteExecutionError):
        _checkpoint_from_pointer(reference)


def test_direct_and_legacy_pointers_remain_supported(tmp_path):
    assert _checkpoint_from_pointer(None) is None
    uri = "river://run/sampler_weights/model"
    assert _checkpoint_from_pointer(uri, base_model="base", lora_rank=16) == uri
    reference = write_pointer(tmp_path / "legacy", {"checkpoint": uri})
    assert _checkpoint_from_pointer(reference, base_model="base", lora_rank=16) == uri
    reference = write_pointer(
        tmp_path / "modern",
        {
            "checkpoint": uri,
            "base_model": "base",
            "provider": "river",
            "lora": {"rank": 16},
        },
    )
    assert _checkpoint_from_pointer(reference, base_model="base", lora_rank=16) == uri


@pytest.mark.parametrize(
    "scenario",
    [scenario_rl, scenario_episode_rl, scenario_harvest, scenario_episode_harvest],
)
def test_wrong_model_fails_before_client_initialization(
    tmp_path, monkeypatch, scenario
):
    spec, client = scenario(tmp_path)
    reference = write_pointer(
        tmp_path / "pointer",
        {
            "checkpoint": "river://original",
            "base_model": "different-model",
        },
    )
    executor = _executor(tmp_path, client)
    monkeypatch.setattr(executor, "_get_client", lambda: pytest.fail("Opened client"))
    with pytest.raises(RemoteExecutionError, match="base model"):
        executor.submit(
            replace(spec, harvest={**spec.harvest, "adapter_dir": reference})
        )
    assert not client.calls


@pytest.mark.parametrize("scenario", [scenario_rl, scenario_episode_rl])
def test_changed_pointer_cannot_resume_an_existing_run(tmp_path, scenario):
    spec, client = scenario(tmp_path)
    reference = write_pointer(tmp_path / "pointer", {"checkpoint": "river://original"})
    spec = replace(spec, harvest={**spec.harvest, "adapter_dir": reference})
    executor = _executor(tmp_path, client)
    executor.submit(spec)
    before = {p: p.read_bytes() for p in spec.output_dir.rglob("*.json")}
    calls = list(client.calls)
    write_pointer(tmp_path / "pointer", {"checkpoint": "river://replacement"})
    with pytest.raises(ValueError, match="differs from the saved run"):
        executor.submit(replace(spec, resume=True))
    assert client.calls == calls
    assert {p: p.read_bytes() for p in spec.output_dir.rglob("*.json")} == before


def test_same_checkpoint_can_resume_through_a_different_pointer_path(tmp_path):
    spec, client = scenario_rl(tmp_path)
    value = {"checkpoint": "river://original"}
    first = write_pointer(tmp_path / "pointer", value)
    second = write_pointer(tmp_path / "alias", value)
    executor = _executor(tmp_path, client)
    executor.submit(replace(spec, harvest={**spec.harvest, "adapter_dir": first}))
    before = (spec.output_dir / "rl_state.json").read_bytes()
    calls = list(client.calls)
    executor.submit(
        replace(spec, resume=True, harvest={**spec.harvest, "adapter_dir": second})
    )
    assert client.calls == calls
    assert (spec.output_dir / "rl_state.json").read_bytes() == before


def test_pointer_is_resolved_once_for_both_sampling_and_run_identity(
    tmp_path, monkeypatch
):
    spec, client = scenario_rl(tmp_path)
    original = {"checkpoint": "river://original"}
    reference = write_pointer(tmp_path / "pointer", original)
    spec = replace(spec, harvest={**spec.harvest, "adapter_dir": reference})
    executor = _executor(tmp_path, client)

    def open_client():
        write_pointer(tmp_path / "pointer", {"checkpoint": "river://replacement"})
        return client

    monkeypatch.setattr(executor, "_get_client", open_client)
    executor.submit(spec)
    created = [call for call in client.calls if call["call"] == "create_model"]
    assert "river://original" in created[0]["checkpoint"]
    write_pointer(tmp_path / "pointer", original)
    calls = list(client.calls)
    executor.submit(replace(spec, resume=True))
    assert client.calls == calls


@pytest.mark.parametrize("metadata", [{"base_model": "other"}, {"lora": {"rank": 32}}])
def test_native_cli_rejects_incompatible_pointer_before_preparing_run(
    tmp_path, monkeypatch, metadata
):
    from stateset_agents.training import river_refund

    reference = write_pointer(
        tmp_path / "pointer", {"checkpoint": "river://original", **metadata}
    )
    output = tmp_path / "run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "river-refund",
            "--base-model",
            "base",
            "--checkpoint",
            reference,
            "--output",
            str(output),
            "--dry-run",
        ],
    )
    monkeypatch.setattr(
        river_refund, "require_native_runtime", lambda: pytest.fail("SDK inspection")
    )
    with pytest.raises(RemoteExecutionError):
        river_refund.main()
    assert not output.exists()


def test_programmatic_native_run_binds_prepared_pointer_uri(tmp_path, monkeypatch):
    from stateset_agents.training import river_refund as native

    reference = write_pointer(tmp_path / "pointer", {"checkpoint": "river://original"})
    output = tmp_path / "run"
    output.mkdir()
    args = SimpleNamespace(
        output=output,
        base_model="base",
        seed=42,
        steps=2,
        concurrency=8,
        max_staleness=0,
        learning_rate=1e-5,
        checkpoint=reference,
        evaluate_only=True,
        dry_run=False,
    )
    splits = {"test": [{"order_id": "test-A", "amount_cents": 100, "eligible": True}]}
    native.prepare_run(args, splits)
    manifest = json.loads((output / "run_manifest.json").read_text())
    assert manifest["checkpoint"] == "river://original"
    assert args.checkpoint == reference
    before = {p: p.read_bytes() for p in output.iterdir()}
    write_pointer(tmp_path / "pointer", {"checkpoint": "river://replacement"})
    monkeypatch.setitem(sys.modules, "river_client", None)
    monkeypatch.setattr(
        native, "require_native_runtime", lambda: pytest.fail("SDK inspection")
    )
    with pytest.raises(ValueError, match="configuration or cases differ"):
        native.execute_run(args, splits)
    with pytest.raises(ValueError, match="configuration or cases differ"):
        asyncio.run(native.campaign(object(), object(), object(), args, splits))
    assert {p: p.read_bytes() for p in output.iterdir()} == before
