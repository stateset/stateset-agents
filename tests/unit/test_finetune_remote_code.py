"""Reviewed remote package bytes stay private, bounded and independent of env drift."""

import builtins
import hashlib
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from stateset_agents import finetune_remote_code as code

VERSION = "0.66.1"
WHEEL_NAME = "stateset_agents-0.66.1-py3-none-any.whl"


@pytest.fixture
def wheel(tmp_path):
    path = tmp_path / WHEEL_NAME
    path.write_bytes(b"reviewed-package-archive")
    return path


@pytest.fixture
def run(tmp_path):
    path = tmp_path / "private-run"
    path.mkdir(mode=0o700)
    return path


def capture(wheel):
    return code.capture_remote_code(
        "runpod", VERSION, environ={code.WHEEL_ENV: str(wheel)}
    )


def guard_nonblocking(monkeypatch, path):
    """Reject an unsafe open before an actual FIFO open could hang pytest."""
    originals = (os.open, Path.open, builtins.open, io.open)

    def matches(candidate):
        return not isinstance(candidate, int) and os.fspath(candidate) == os.fspath(
            path
        )

    def opened(candidate, flags, *args, **kwargs):
        if matches(candidate):
            if hasattr(os, "O_NONBLOCK"):
                assert flags & os.O_NONBLOCK
            if hasattr(os, "O_NOFOLLOW"):
                assert flags & os.O_NOFOLLOW
        return originals[0](candidate, flags, *args, **kwargs)

    def guarded(original):
        def checked(candidate, *args, **kwargs):
            assert not matches(candidate), "A wheel requires descriptor admission"
            return original(candidate, *args, **kwargs)

        return checked

    monkeypatch.setattr(code.os, "open", opened)
    monkeypatch.setattr(Path, "open", guarded(originals[1]))
    monkeypatch.setattr(builtins, "open", guarded(originals[2]))
    monkeypatch.setattr(io, "open", guarded(originals[3]))


def replace_special(path, kind, tmp_path):
    path.unlink(missing_ok=True)
    if kind == "fifo":
        if not hasattr(os, "mkfifo") or not hasattr(os, "O_NONBLOCK"):
            pytest.skip("Nonblocking named pipes are unavailable")
        os.mkfifo(path, 0o600)
    elif kind == "directory":
        path.mkdir()
    else:
        outside = tmp_path / "outside.whl"
        outside.write_bytes(b"outside-unreviewed-archive")
        path.symlink_to(outside)


def test_release_mode_is_an_explicit_version_pin_without_wheel_io(monkeypatch, run):
    def forbidden(*args, **kwargs):
        raise AssertionError("Release planning must not inspect any wheel")

    monkeypatch.setattr(code, "_wheel_identity", forbidden)
    selection = code.capture_remote_code("runpod", VERSION, environ={})
    assert selection.describe() == {
        "schema_version": 1,
        "provider": "runpod",
        "mode": "release",
        "package_version": VERSION,
    }
    monkeypatch.setenv(code.WHEEL_ENV, "/missing/unreviewed.whl")
    record = code.snapshot_remote_code(run, selection)
    assert code.verify_remote_code(run, record, package_version=VERSION) is None
    assert list(run.iterdir()) == []


@pytest.mark.parametrize("provider", ["local", "modal", "river"])
def test_other_providers_do_not_read_wheel_environment(provider):
    class NoEnvironment:
        def get(self, *args):
            raise AssertionError("Unrelated provider read the environment")

    assert code.capture_remote_code(provider, VERSION, environ=NoEnvironment()) is None


def test_capture_reads_only_the_wheel_override_and_does_not_expose_source_path(wheel):
    reads = []

    class OnlyWheel(dict):
        def get(self, key, default=None):
            reads.append(key)
            assert key == code.WHEEL_ENV
            return super().get(key, default)

    selection = code.capture_remote_code(
        "runpod", VERSION, environ=OnlyWheel({code.WHEEL_ENV: str(wheel)})
    )
    record = selection.describe()
    assert reads == [code.WHEEL_ENV]
    assert record["wheel_sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert record["wheel_bytes"] == wheel.stat().st_size
    assert record["wheel_filename"] == WHEEL_NAME
    assert str(wheel) not in json.dumps(record) and str(wheel) not in repr(selection)
    assert "source_path" not in record and selection.source_path == wheel


def test_wheel_capture_preserves_binary_bytes(wheel):
    raw = b"PK\x03\x04\r\n\x1a\x00\xff\r\nend"
    wheel.write_bytes(raw)
    record = capture(wheel).describe()
    assert record["wheel_bytes"] == len(raw)
    assert record["wheel_sha256"] == hashlib.sha256(raw).hexdigest()


@pytest.mark.parametrize("change", [None, "descriptor", "path"])
def test_cross_api_ctime_difference_preserves_each_api_change_guard(
    wheel, monkeypatch, change
):
    """Model Windows creation/change-time differences without dropping either guard."""
    real_fstat, real_lstat = os.fstat, Path.lstat
    calls = {"descriptor": 0, "path": 0}

    def stamp(info, ctime):
        return SimpleNamespace(
            st_mode=info.st_mode,
            st_dev=info.st_dev,
            st_ino=info.st_ino,
            st_size=info.st_size,
            st_mtime_ns=info.st_mtime_ns,
            st_ctime_ns=ctime,
        )

    def descriptor_stat(descriptor):
        info = real_fstat(descriptor)
        calls["descriptor"] += 1
        return stamp(
            info,
            info.st_ctime_ns
            + 10
            + int(change == "descriptor" and calls["descriptor"] > 1),
        )

    def path_stat(path, *args, **kwargs):
        info = real_lstat(path, *args, **kwargs)
        if path != wheel:
            return info
        calls["path"] += 1
        return stamp(
            info, info.st_ctime_ns + int(change == "path" and calls["path"] > 2)
        )

    monkeypatch.setattr(code.os, "fstat", descriptor_stat)
    monkeypatch.setattr(Path, "lstat", path_stat)
    if change is None:
        assert (
            capture(wheel).describe()["wheel_sha256"]
            == hashlib.sha256(wheel.read_bytes()).hexdigest()
        )
    else:
        with pytest.raises(ValueError, match="changed during inspection"):
            capture(wheel)


def test_relative_override_is_bound_to_review_cwd(wheel):
    selection = code.capture_remote_code(
        "runpod", VERSION, environ={code.WHEEL_ENV: WHEEL_NAME}, cwd=wheel.parent
    )
    assert selection.source_path == wheel


def test_private_snapshot_is_independent_of_source_and_ambient_environment(
    wheel, run, tmp_path, monkeypatch
):
    selection = capture(wheel)
    record = code.snapshot_remote_code(run, selection)
    path = code.verify_remote_code(run, record, package_version=VERSION)
    assert path == run / code.SNAPSHOT_DIRECTORY / WHEEL_NAME
    assert path.read_bytes() == wheel.read_bytes() and path != wheel
    wheel.unlink()
    monkeypatch.setenv(code.WHEEL_ENV, str(tmp_path / "later-unreviewed.whl"))
    assert code.verify_remote_code(run, record) == path
    if os.name != "nt":
        assert path.stat().st_mode & 0o777 == 0o400
        assert path.parent.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink"])
def test_nonregular_override_fails_without_blocking(wheel, tmp_path, monkeypatch, kind):
    replace_special(wheel, kind, tmp_path)
    guard_nonblocking(monkeypatch, wheel)
    with pytest.raises((OSError, ValueError)):
        capture(wheel)


@pytest.mark.parametrize("kind", ["fifo", "directory", "symlink"])
def test_nonregular_snapshot_fails_without_blocking(
    wheel, run, tmp_path, monkeypatch, kind
):
    record = code.snapshot_remote_code(run, capture(wheel))
    path = run / code.SNAPSHOT_DIRECTORY / WHEEL_NAME
    replace_special(path, kind, tmp_path)
    guard_nonblocking(monkeypatch, path)
    with pytest.raises((OSError, ValueError)):
        code.verify_remote_code(run, record)


@pytest.mark.parametrize(
    "name", ["archive.tar", "archive.WHL", "bad name.whl", ".hidden.whl"]
)
def test_override_requires_plain_wheel_filename(tmp_path, name):
    path = tmp_path / name
    path.write_bytes(b"reviewed")
    with pytest.raises(ValueError, match=".whl"):
        capture(path)


def test_byte_limit_and_empty_file_fail_before_any_content_read(wheel, monkeypatch):
    monkeypatch.setattr(code, "MAX_WHEEL_BYTES", 4)
    monkeypatch.setattr(
        code.os, "read", lambda *args: pytest.fail("Invalid size was read")
    )
    with pytest.raises(ValueError, match="64 MiB"):
        capture(wheel)
    wheel.write_bytes(b"")
    with pytest.raises(ValueError, match="64 MiB"):
        capture(wheel)


def test_exact_byte_limit_is_accepted(wheel, monkeypatch):
    monkeypatch.setattr(code, "MAX_WHEEL_BYTES", wheel.stat().st_size)
    assert capture(wheel).describe()["wheel_bytes"] == wheel.stat().st_size


def test_mutation_between_review_and_prepare_is_rejected_and_partial_copy_removed(
    wheel, run
):
    selection = capture(wheel)
    wheel.write_bytes(b"X" * selection.wheel_bytes)
    with pytest.raises(ValueError, match="reviewed bytes"):
        code.snapshot_remote_code(run, selection)
    assert list(run.iterdir()) == []


@pytest.mark.parametrize("mutation", ["grow", "overwrite", "replace"])
def test_wheel_mutation_during_stream_is_rejected(wheel, monkeypatch, mutation):
    original = os.read
    changed = False

    def read_and_mutate(descriptor, size):
        nonlocal changed
        chunk = original(descriptor, size)
        if chunk and not changed:
            changed = True
            if mutation == "grow":
                with wheel.open("ab") as stream:
                    stream.write(b"extra")
            elif mutation == "overwrite":
                before = wheel.stat()
                wheel.write_bytes(b"X" * before.st_size)
                os.utime(
                    wheel, ns=(before.st_atime_ns, before.st_mtime_ns + 1000000000)
                )
            else:
                replacement = wheel.with_name("replacement.whl")
                replacement.write_bytes(b"replacement")
                try:
                    replacement.replace(wheel)
                except PermissionError:
                    if os.name != "nt":
                        raise
                    # Windows can refuse replacement of an open file. That
                    # prevents this mutation before the reader needs to reject it.
                    return chunk
        return chunk

    monkeypatch.setattr(code.os, "read", read_and_mutate)
    try:
        selection = capture(wheel)
    except ValueError as error:
        assert "changed during inspection" in str(error)
    else:
        assert os.name == "nt" and mutation == "replace"
        assert (
            selection.describe()["wheel_sha256"]
            == hashlib.sha256(wheel.read_bytes()).hexdigest()
        )
    assert changed


def test_descriptor_is_closed_when_content_read_fails(wheel, monkeypatch):
    original = os.open
    descriptors = []

    def opened(*args, **kwargs):
        descriptor = original(*args, **kwargs)
        descriptors.append(descriptor)
        return descriptor

    def failed(*args):
        raise OSError("injected read error")

    monkeypatch.setattr(code.os, "open", opened)
    monkeypatch.setattr(code.os, "read", failed)
    with pytest.raises(OSError, match="injected"):
        capture(wheel)
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])


def test_mutated_private_snapshot_cannot_be_rebound_to_source(wheel, run):
    record = code.snapshot_remote_code(run, capture(wheel))
    path = run / code.SNAPSHOT_DIRECTORY / WHEEL_NAME
    path.chmod(0o600)
    path.write_bytes(b"X" * record["wheel_bytes"])
    with pytest.raises(ValueError, match="reviewed bytes"):
        code.verify_remote_code(run, record)
    assert wheel.read_bytes() != path.read_bytes()


@pytest.mark.parametrize("component", ["run", "directory", "wheel"])
def test_snapshot_private_permissions_are_revalidated(wheel, run, component):
    if os.name == "nt":
        pytest.skip("POSIX private-mode checks")
    record = code.snapshot_remote_code(run, capture(wheel))
    directory = run / code.SNAPSHOT_DIRECTORY
    path = directory / WHEEL_NAME
    {"run": run, "directory": directory, "wheel": path}[component].chmod(
        0o755 if component != "wheel" else 0o644
    )
    with pytest.raises(ValueError, match="private"):
        code.verify_remote_code(run, record)


@pytest.mark.parametrize("component", ["run", "directory"])
def test_snapshot_does_not_follow_directory_links(wheel, run, tmp_path, component):
    record = code.snapshot_remote_code(run, capture(wheel))
    if component == "run":
        link = tmp_path / "linked-run"
        link.symlink_to(run, target_is_directory=True)
        run = link
    else:
        directory = run / code.SNAPSHOT_DIRECTORY
        moved = run / "old-remote-code"
        directory.rename(moved)
        directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ValueError, match="private regular directories"):
        code.verify_remote_code(run, record)


def test_existing_snapshot_is_never_overwritten_or_cleaned(wheel, run):
    selection = capture(wheel)
    record = code.snapshot_remote_code(run, selection)
    path = code.verify_remote_code(run, record)
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        code.snapshot_remote_code(run, selection)
    assert path.read_bytes() == before


def test_failed_flush_removes_only_new_snapshot(wheel, run, monkeypatch):
    selection = capture(wheel)
    marker = run / "existing-record.json"
    marker.write_text("unchanged")

    def failed(*args):
        raise OSError("injected fsync failure")

    monkeypatch.setattr(code.os, "fsync", failed)
    with pytest.raises(OSError, match="injected"):
        code.snapshot_remote_code(run, selection)
    assert list(run.iterdir()) == [marker] and marker.read_text() == "unchanged"


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", True),
        ("provider", "modal"),
        ("mode", "unexpected"),
        ("mode", []),
        ("package_version", ""),
        ("package_version", "0.66.1; injected"),
        ("wheel_filename", "../outside.whl"),
        ("wheel_filename", "outside\\\\wheel.whl"),
        ("wheel_filename", "artifact.tar"),
        ("wheel_sha256", "g" * 64),
        ("wheel_bytes", True),
        ("wheel_bytes", 0),
        ("wheel_bytes", 64 * 1024 * 1024 + 1),
        ("source_path", "/must-not-be-public"),
    ],
)
def test_record_schema_rejects_unsafe_or_unreviewed_fields(wheel, field, value):
    record = capture(wheel).describe()
    record[field] = value
    with pytest.raises(ValueError):
        code.normalize_remote_code(record)


def test_missing_review_requires_fresh_review_instead_of_environment_default(
    run, monkeypatch
):
    monkeypatch.setenv(code.WHEEL_ENV, "/must-not-be-used.whl")
    with pytest.raises(ValueError, match="fresh run"):
        code.verify_remote_code(run, None)


def test_package_version_binding_is_checked_for_wheel_and_release(wheel, run):
    for selection in (
        capture(wheel),
        code.capture_remote_code("runpod", VERSION, environ={}),
    ):
        with pytest.raises(ValueError, match="framework"):
            code.verify_remote_code(run, selection.describe(), package_version="0.0.0")


def test_normalized_records_are_independent_copies(wheel):
    selection = capture(wheel)
    first = selection.describe()
    second = code.normalize_remote_code(first)
    second["wheel_sha256"] = "0" * 64
    assert first == selection.describe() and first != second


def test_source_replaced_by_a_link_between_check_and_open_is_rejected(
    wheel, tmp_path, monkeypatch
):
    if not hasattr(os, "O_NOFOLLOW"):
        pytest.skip("Atomic no-follow opens are unavailable")
    outside = tmp_path / "outside.whl"
    outside.write_bytes(b"outside-unreviewed-archive")
    original = os.open

    def replaced(candidate, flags, *args, **kwargs):
        if os.fspath(candidate) == os.fspath(wheel):
            assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK
            wheel.unlink()
            wheel.symlink_to(outside)
        return original(candidate, flags, *args, **kwargs)

    monkeypatch.setattr(code.os, "open", replaced)
    with pytest.raises(OSError):
        capture(wheel)
    assert outside.read_bytes() == b"outside-unreviewed-archive"
