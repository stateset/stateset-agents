"""Bounded agent JSON records reject special files without starting work."""

import builtins
import hashlib
import io
import json
import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from stateset_agents import cli_finetune as api
from stateset_agents import finetune_workspace as workspace
from stateset_agents.finetune_workspace import write_json


@pytest.fixture
def no_work(monkeypatch):
    calls = []

    def forbid(name):
        def called(*args, **kwargs):
            calls.append(name)
            raise AssertionError(f"Record inspection must not call {name}")

        return called

    for name in (
        "plan_run",
        "create_run",
        "launch_run",
        "verified_run_settings",
        "require_backend_credentials",
        "inspect_training_data",
        "inspect_readiness",
    ):
        monkeypatch.setattr(api, name, forbid(name))
    monkeypatch.setattr(workspace.subprocess, "Popen", forbid("Popen"))
    yield calls
    assert calls == []


def invoke(*args):
    result = CliRunner().invoke(api.app, list(map(str, args)))
    response = json.loads(result.stdout)
    assert response["schema_version"] == 1
    assert response["ok"] == (result.exit_code == 0)
    assert len(result.stdout.splitlines()) == 1
    return result, response


def retained_run(tmp_path):
    root = tmp_path / "workspace with spaces"
    run = root / "retained-run"
    run.mkdir(parents=True)
    write_json(run / "plan.json", {"config": {"dataset": "never-read.jsonl"}})
    write_json(run / "state.json", {"status": "prepared"})
    return root, run


def nonregular(path, kind, tmp_path):
    if path.exists():
        path.unlink()
    target = None
    if kind == "fifo":
        if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
            pytest.skip("Nonblocking named pipes are unavailable")
        os.mkfifo(path)
    elif kind == "directory":
        path.mkdir()
    else:
        target = tmp_path / "private-target.json"
        target.write_text('{"dataset":"PRIVATE-RECORD-CONTENT"}')
        path.symlink_to(target)
    return target


def guard_record_open(monkeypatch, path):
    """Fail before a regressed blocking FIFO open could hang the test process."""
    original_open = os.open
    original_path_open = Path.open
    original_builtin_open = builtins.open
    original_io_open = io.open
    expected = os.path.abspath(os.fspath(path))

    def matches(candidate):
        return not isinstance(candidate, int) and (
            os.path.abspath(os.fspath(candidate)) == expected
        )

    def guarded_open(candidate, flags, *args, **kwargs):
        if matches(candidate):
            if hasattr(os, "O_NONBLOCK"):
                assert flags & os.O_NONBLOCK, "Record opens must be nonblocking"
            if hasattr(os, "O_NOFOLLOW"):
                assert flags & os.O_NOFOLLOW, "Record opens must not follow links"
        return original_open(candidate, flags, *args, **kwargs)

    def guarded_path_open(candidate, *args, **kwargs):
        assert not matches(candidate), "Record reads need descriptor admission"
        return original_path_open(candidate, *args, **kwargs)

    def guarded_builtin_open(candidate, *args, **kwargs):
        assert not matches(candidate), "Record reads need descriptor admission"
        return original_builtin_open(candidate, *args, **kwargs)

    def guarded_io_open(candidate, *args, **kwargs):
        assert not matches(candidate), "Record reads need descriptor admission"
        return original_io_open(candidate, *args, **kwargs)

    monkeypatch.setattr(api.os, "open", guarded_open)
    monkeypatch.setattr(Path, "open", guarded_path_open)
    monkeypatch.setattr(builtins, "open", guarded_builtin_open)
    monkeypatch.setattr(io, "open", guarded_io_open)


@pytest.mark.parametrize("command", ["validate", "prepare"])
@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink"])
def test_config_special_files_return_json_without_reading_data_or_starting_work(
    tmp_path, monkeypatch, no_work, command, kind
):
    config = tmp_path / "config.json"
    target = nonregular(config, kind, tmp_path)
    guard_record_open(monkeypatch, config)
    root = tmp_path / "not-created"
    args = [command, "--config", config]
    if command == "prepare":
        args += ["--workspace", root]
    result, response = invoke(*args)
    assert not response["ok"]
    assert response["error"]["code"] in {"invalid_record", "io_error"}
    assert "PRIVATE-RECORD-CONTENT" not in result.stdout
    assert not root.exists() and no_work == []
    if target is not None:
        assert target.read_text() == '{"dataset":"PRIVATE-RECORD-CONTENT"}'


@pytest.mark.parametrize("listing", [False, True])
@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink"])
def test_plan_special_files_do_not_hang_targeted_status_or_history(
    tmp_path, monkeypatch, no_work, listing, kind
):
    root, run = retained_run(tmp_path)
    target = nonregular(run / "plan.json", kind, tmp_path)
    guard_record_open(monkeypatch, run / "plan.json")
    before = set(run.iterdir())
    args = ["status"] + ([] if listing else [run.name])
    result, response = invoke(*args, "--workspace", root)
    assert "PRIVATE-RECORD-CONTENT" not in result.stdout
    if kind == "symlink":
        if listing:
            assert response["ok"] and response["result"]["runs"] == []
        else:
            assert response["error"]["code"] == "run_not_found"
    else:
        assert response["ok"]
        summary = (
            response["result"]["runs"][0] if listing else response["result"]["run"]
        )
        assert summary["run_id"] == run.name
        assert summary["plan_sha256"] is None
        assert summary["config"] == {}
        assert summary["unreadable_records"] == ["plan.json"]
    assert set(run.iterdir()) == before and no_work == []
    if target is not None:
        assert target.read_text() == '{"dataset":"PRIVATE-RECORD-CONTENT"}'


@pytest.mark.parametrize("listing", [False, True])
@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink"])
@pytest.mark.parametrize(
    "filename, projected",
    [
        ("provider-job.json", "provider"),
        ("provider-result.json", "provider_result"),
        ("completion.json", "completion"),
    ],
)
def test_optional_special_records_remain_visible_as_unreadable(
    tmp_path, monkeypatch, no_work, listing, kind, filename, projected
):
    root, run = retained_run(tmp_path)
    target = nonregular(run / filename, kind, tmp_path)
    guard_record_open(monkeypatch, run / filename)
    before = set(run.iterdir())
    args = ["status"] + ([] if listing else [run.name])
    result, response = invoke(*args, "--workspace", root)
    assert response["ok"]
    summary = response["result"]["runs"][0] if listing else response["result"]["run"]
    assert summary["unreadable_records"] == [filename]
    assert projected not in summary
    assert summary["state"]["status"] == "prepared"
    assert "PRIVATE-RECORD-CONTENT" not in result.stdout
    assert set(run.iterdir()) == before and no_work == []
    if target is not None:
        assert target.read_text() == '{"dataset":"PRIVATE-RECORD-CONTENT"}'


def test_regular_record_keeps_exact_original_byte_digest(tmp_path):
    path = tmp_path / "record.json"
    raw = (
        ' {\n "nested": {"count": 2, "ok": true},'
        ' "ratios": [1.25, 1e308, -2.5e-4, -0.0],'
        ' "label": "snowman \u2603"\n}\n'
    ).encode("utf-8")
    path.write_bytes(raw)
    value, digest = api._object(path)
    assert value == {
        "nested": {"count": 2, "ok": True},
        "ratios": [1.25, 1e308, -2.5e-4, -0.0],
        "label": "snowman \u2603",
    }
    assert digest == hashlib.sha256(raw).hexdigest()
    assert digest != hashlib.sha256(json.dumps(value).encode()).hexdigest()


@pytest.mark.parametrize(
    "raw",
    [
        b"[]",
        b"null",
        b"invalid json",
        b'{"value": NaN}',
        b'{"value": Infinity}',
        b'{"nested": {"value": -Infinity}}',
        b'{"nested": {"value": 1e9999}}',
        b'{"nested": [{"value": -1e9999}]}',
    ],
)
def test_invalid_and_nonfinite_records_keep_the_invalid_record_error(tmp_path, raw):
    path = tmp_path / "record.json"
    path.write_bytes(raw)
    with pytest.raises(api._RequestError) as error:
        api._object(path)
    assert error.value.code == "invalid_record"


def test_regular_record_byte_limit_includes_trailing_whitespace(tmp_path):
    path = tmp_path / "record.json"
    raw = b'{"v":1}' + b" " * (32 - len(b'{"v":1}'))
    path.write_bytes(raw)
    assert api._object(path, limit=32) == ({"v": 1}, hashlib.sha256(raw).hexdigest())
    path.write_bytes(raw + b" ")
    with pytest.raises(api._RequestError) as error:
        api._object(path, limit=32)
    assert error.value.code == "invalid_record"


def test_record_that_grows_after_descriptor_admission_still_respects_limit(
    tmp_path, monkeypatch
):
    path = tmp_path / "record.json"
    path.write_bytes(b'{"v":1}')
    original_fdopen = os.fdopen
    read_sizes = []

    class GrowingReader:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            self.stream.__enter__()
            return self

        def __exit__(self, *args):
            return self.stream.__exit__(*args)

        def fileno(self):
            return self.stream.fileno()

        def read(self, size):
            read_sizes.append(size)
            with path.open("ab") as writer:
                writer.write(b" " * 64)
            return self.stream.read(size)

    monkeypatch.setattr(
        api.os,
        "fdopen",
        lambda *args, **kwargs: GrowingReader(original_fdopen(*args, **kwargs)),
    )
    with pytest.raises(api._RequestError) as error:
        api._object(path, limit=32)
    assert error.value.code == "invalid_record" and read_sizes == [33]


def test_replacement_symlink_between_check_and_open_is_not_followed(
    tmp_path, monkeypatch, no_work
):
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("Atomic no-follow opens are unavailable")
    path = tmp_path / "config.json"
    path.write_text("{}")
    target = tmp_path / "private-target.json"
    target.write_text('{"dataset":"PRIVATE-RECORD-CONTENT"}')
    original_open = os.open

    def replaced_open(candidate, flags, *args, **kwargs):
        if os.fspath(candidate) == os.fspath(path):
            assert flags & os.O_NOFOLLOW
            assert flags & os.O_NONBLOCK
            path.unlink()
            path.symlink_to(target)
        return original_open(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(api.os, "open", replaced_open)
    result, response = invoke("validate", "--config", path)
    assert not response["ok"] and response["error"]["code"] == "io_error"
    assert "PRIVATE-RECORD-CONTENT" not in result.stdout and no_work == []
    assert target.read_text() == '{"dataset":"PRIVATE-RECORD-CONTENT"}'


def test_missing_config_keeps_io_error_without_creating_workspace(tmp_path, no_work):
    root = tmp_path / "not-created"
    _, response = invoke(
        "prepare", "--config", tmp_path / "missing.json", "--workspace", root
    )
    assert not response["ok"] and response["error"]["code"] == "io_error"
    assert not root.exists() and no_work == []
