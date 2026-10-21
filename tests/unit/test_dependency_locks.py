"""Resolver failures and cancellation must leave working dependency locks intact."""

from __future__ import annotations

import json
import os
import shutil
import sys
from pathlib import Path

import pytest

from scripts import check_dependency_locks as locks

pytestmark = pytest.mark.skipif(
    os.name != "posix", reason="Resolver subprocess fixture uses a POSIX executable"
)


@pytest.fixture
def repository(tmp_path: Path) -> Path:
    root = tmp_path / "checkout with spaces"
    root.mkdir()
    (root / "pyproject.toml").write_text('[project]\nname="fixture"\nversion="1.0"\n')
    for name, _ in locks.LOCKS:
        (root / name).write_text("# original header\nnumpy==2.4.6\n")
    return root


@pytest.fixture
def compiler(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[str, Path]:
    log = tmp_path / "calls.jsonl"
    monkeypatch.setenv("STATESET_LOCK_TEST_LOG", str(log))
    program = tmp_path / "fake compiler"
    program.write_text(
        f"#!{sys.executable}\n"
        "import json, os, pathlib, sys, time\n"
        "output = pathlib.Path(next(a.split('=', 1)[1] for a in sys.argv if a.startswith('--output-file=')))\n"
        "with open(os.environ['STATESET_LOCK_TEST_LOG'], 'a') as stream:\n"
        "    stream.write(json.dumps({'output': str(output), 'args': sys.argv[1:], 'cwd': os.getcwd()}) + '\\n')\n"
        "mode = os.environ.get('STATESET_LOCK_TEST_MODE', '')\n"
        "selected = os.environ.get('STATESET_LOCK_TEST_TARGET', output.name)\n"
        "if output.name == selected:\n"
        "    if mode == 'fail':\n"
        "        output.write_text('partial output\\n')\n"
        "        sys.exit(7)\n"
        "    if mode == 'stale':\n"
        "        output.write_text('numpy==2.4.7\\n')\n"
        "    elif mode == 'missing':\n"
        "        output.unlink()\n"
        "    elif mode == 'empty':\n"
        "        output.write_text('# no resolved packages\\n')\n"
        "    elif mode == 'timeout':\n"
        "        time.sleep(30)\n"
        "    elif mode == 'concurrent':\n"
        "        source = pathlib.Path(os.environ['STATESET_LOCK_TEST_CHANGED'])\n"
        "        source.write_bytes(source.read_bytes() + b'# concurrent edit\\n')\n"
    )
    program.chmod(0o700)
    return str(program), log


def snapshot(root: Path) -> dict[str, tuple[bytes, int, int, int]]:
    result = {}
    for path in root.iterdir():
        stat = path.stat()
        result[path.name] = (
            path.read_bytes(),
            stat.st_ino,
            stat.st_mtime_ns,
            stat.st_mode,
        )
    return result


def test_success_checks_all_three_locks_without_replacing_working_files(
    repository, compiler
):
    program, log = compiler
    before = snapshot(repository)
    assert locks.check_locks(repository, compiler=program) == 0
    assert snapshot(repository) == before
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert [Path(row["output"]).name for row in calls] == [
        name for name, _ in locks.LOCKS
    ]
    for row, (_, options) in zip(calls, locks.LOCKS, strict=True):
        assert row["cwd"] == str(repository)
        output = Path(row["output"])
        assert not output.is_relative_to(repository)
        assert not output.parent.exists()
        assert all(option in row["args"] for option in options)


@pytest.mark.parametrize("name", [name for name, _ in locks.LOCKS])
def test_each_resolver_failure_preserves_inputs(
    repository, compiler, monkeypatch, name
):
    program, log = compiler
    before = snapshot(repository)
    monkeypatch.setenv("STATESET_LOCK_TEST_MODE", "fail")
    monkeypatch.setenv("STATESET_LOCK_TEST_TARGET", name)
    assert locks.check_locks(repository, compiler=program) == 1
    assert snapshot(repository) == before
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    assert Path(calls[-1]["output"]).name == name
    assert not Path(calls[-1]["output"]).parent.exists()


@pytest.mark.parametrize("mode", ["stale", "missing", "empty"])
def test_invalid_resolved_output_cannot_pass(repository, compiler, monkeypatch, mode):
    program, _ = compiler
    before = snapshot(repository)
    monkeypatch.setenv("STATESET_LOCK_TEST_MODE", mode)
    assert locks.check_locks(repository, compiler=program) != 0
    assert snapshot(repository) == before


@pytest.mark.parametrize("name", ["pyproject.toml", *(name for name, _ in locks.LOCKS)])
def test_concurrent_edit_is_rejected_and_preserved(
    repository, compiler, monkeypatch, name
):
    program, _ = compiler
    before = snapshot(repository)
    monkeypatch.setenv("STATESET_LOCK_TEST_MODE", "concurrent")
    monkeypatch.setenv("STATESET_LOCK_TEST_CHANGED", str(repository / name))
    assert locks.check_locks(repository, compiler=program) == 1
    assert (repository / name).read_bytes() == before[name][0] + b"# concurrent edit\n"
    for other in set(before) - {name}:
        assert snapshot(repository)[other] == before[other]


def test_timeout_preserves_inputs(repository, compiler, monkeypatch):
    program, log = compiler
    before = snapshot(repository)
    monkeypatch.setenv("STATESET_LOCK_TEST_MODE", "timeout")
    assert locks.check_locks(repository, compiler=program, timeout=1) == 1
    assert snapshot(repository) == before
    output = Path(json.loads(log.read_text().splitlines()[0])["output"])
    assert not output.parent.exists()


def test_interrupt_cleans_temporary_outputs_without_undoing_inputs(
    repository, compiler, monkeypatch
):
    program, _ = compiler
    before = snapshot(repository)
    outputs = []

    def interrupt(command, **kwargs):
        outputs.append(
            Path(
                next(
                    a.split("=", 1)[1]
                    for a in command
                    if a.startswith("--output-file=")
                )
            )
        )
        outputs[-1].write_text("interrupted partial output\n")
        raise KeyboardInterrupt

    monkeypatch.setattr(locks.subprocess, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        locks.check_locks(repository, compiler=program)
    assert snapshot(repository) == before
    assert outputs and not outputs[0].parent.exists()


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo"])
def test_nonregular_inputs_reject_before_resolving(
    repository, compiler, tmp_path, kind
):
    program, log = compiler
    path = repository / "requirements-lock.txt"
    content = path.read_bytes()
    path.unlink()
    if kind == "symlink":
        outside = tmp_path / "outside.txt"
        outside.write_bytes(content)
        path.symlink_to(outside)
    elif kind == "directory":
        path.mkdir()
    else:
        os.mkfifo(path)
    assert locks.check_locks(repository, compiler=program) == 2
    assert not log.exists()


@pytest.mark.parametrize("timeout", [0, -1, True])
def test_invalid_timeout_never_resolves(repository, compiler, timeout):
    program, log = compiler
    before = snapshot(repository)
    assert locks.check_locks(repository, compiler=program, timeout=timeout) == 2
    assert snapshot(repository) == before
    assert not log.exists()


def test_cli_interrupt_has_nonzero_exit(repository, monkeypatch):
    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(locks, "check_locks", interrupt)
    assert locks.main(["--root", str(repository)]) == 130


def test_default_compiler_uses_the_selected_python_environment(
    repository, compiler, tmp_path, monkeypatch
):
    program, log = compiler
    binaries = tmp_path / "virtualenv/bin"
    binaries.mkdir(parents=True)
    sibling = binaries / "pip-compile"
    shutil.copy2(program, sibling)
    monkeypatch.setattr(locks.sys, "executable", str(binaries / "python"))
    assert locks.check_locks(repository) == 0
    assert len(log.read_text().splitlines()) == 3


def test_missing_compiler_preserves_inputs(repository, monkeypatch):
    before = snapshot(repository)
    monkeypatch.setattr(locks.shutil, "which", lambda *args: None)
    assert locks.check_locks(repository) == 2
    assert snapshot(repository) == before
