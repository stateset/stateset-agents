"""Checkpoint publication survives write failures and real process exits."""

import subprocess
import sys
from pathlib import Path

import pytest

from stateset_agents.training import checkpoint_publication as publication


def save(path, value):
    with publication.checkpoint_write(path) as staging:
        (staging / "training_state.pt").write_bytes(value)
        (staging / "model.safetensors").write_bytes(value * 2)


def read(path):
    with publication.checkpoint_read(path):
        assert publication.validate_checkpoint_publication(path, required=True)
        return (path / "training_state.pt").read_bytes()


@pytest.mark.parametrize("overwrite", [False, True])
@pytest.mark.parametrize("failure", ["artifact", "marker", "journal", "publish"])
@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt])
def test_failed_save_retains_previous_bytes_or_no_checkpoint(
    tmp_path, monkeypatch, overwrite, failure, error_type
):
    path = tmp_path / "saved"
    if overwrite:
        save(path, b"original")
    before = (
        {item.name: item.read_bytes() for item in path.iterdir()} if overwrite else {}
    )
    error = error_type("injected write failure")
    replace = publication.os.replace
    write_json = publication._write_json

    def fail_replace(source, target):
        if failure == "publish" and Path(target) == path:
            if ".stateset-stage-" in Path(source).name:
                raise error
        return replace(source, target)

    def fail_json(destination, value):
        if failure == "marker" and destination.name == publication.COMPLETION_FILE:
            raise error
        if failure == "journal" and destination.name == ".publication-journal":
            raise error
        return write_json(destination, value)

    monkeypatch.setattr(publication.os, "replace", fail_replace)
    monkeypatch.setattr(publication, "_write_json", fail_json)
    with pytest.raises(error_type) as caught:
        with publication.checkpoint_write(path) as staging:
            (staging / "training_state.pt").write_bytes(b"new")
            if failure == "artifact":
                raise error
    assert caught.value is error
    if overwrite:
        assert {item.name: item.read_bytes() for item in path.iterdir()} == before
        assert read(path) == b"original"
    else:
        assert not path.exists()
    assert not list(tmp_path.glob("*.stateset-stage-*"))
    assert not (tmp_path / ".saved.stateset-publication.json").exists()


@pytest.mark.parametrize("phase", ["old_moved", "new_published"])
def test_real_process_exit_recovers_complete_old_or_new_checkpoint(tmp_path, phase):
    path = tmp_path / "saved"
    save(path, b"original")
    script = """
import os
import sys
from pathlib import Path
from stateset_agents.training import checkpoint_publication as publication
path = Path(sys.argv[1])
phase = sys.argv[2]
replace = os.replace
def crash(source, target):
    replace(source, target)
    if ((phase == "old_moved" and Path(source) == path)
        or (phase == "new_published" and Path(target) == path)):
        os._exit(72)
publication.os.replace = crash
with publication.checkpoint_write(path) as staging:
    (staging / "training_state.pt").write_bytes(b"replacement")
    (staging / "model.safetensors").write_bytes(b"weights")
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path), phase],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 72, result.stderr.decode()
    assert (tmp_path / ".saved.stateset-publication.json").exists()
    expected = b"original" if phase == "old_moved" else b"replacement"
    assert read(path) == expected
    assert not (tmp_path / ".saved.stateset-previous").exists()
    assert not (tmp_path / ".saved.stateset-publication.json").exists()
    save(path, b"next")
    assert read(path) == b"next"


@pytest.mark.parametrize(
    "mutation", ["changed", "missing", "extra", "symlink", "marker"]
)
def test_inventory_rejects_corrupt_checkpoint(tmp_path, mutation):
    path = tmp_path / "saved"
    save(path, b"original")
    model = path / "model.safetensors"
    if mutation == "changed":
        model.write_bytes(b"corruption")
    elif mutation == "missing":
        model.unlink()
    elif mutation == "extra":
        (path / "unexpected.bin").write_bytes(b"old weights")
    elif mutation == "symlink":
        model.unlink()
        model.symlink_to(path / "training_state.pt")
    else:
        (path / publication.COMPLETION_FILE).write_text("{}")
    with pytest.raises(ValueError):
        read(path)


def test_reader_excludes_writer_without_overwriting_checkpoint(tmp_path):
    path = tmp_path / "saved"
    save(path, b"original")
    with publication.checkpoint_read(path):
        with pytest.raises(ValueError, match="busy"):
            save(path, b"new")
        with pytest.raises(ValueError, match="busy"):
            read(path)
    assert read(path) == b"original"


def test_new_directory_replaces_stale_weight_layout(tmp_path):
    path = tmp_path / "saved"
    save(path, b"original")
    with publication.checkpoint_write(path) as staging:
        (staging / "training_state.pt").write_bytes(b"new")
        (staging / "pytorch_model.bin").write_bytes(b"new weights")
    assert read(path) == b"new"
    assert not (path / "model.safetensors").exists()


def test_unrelated_destination_and_unowned_backup_are_preserved(tmp_path):
    path = tmp_path / "saved"
    path.mkdir()
    (path / "notes.txt").write_text("user content")
    with pytest.raises(ValueError, match="not a checkpoint"):
        save(path, b"new")
    assert (path / "notes.txt").read_text() == "user content"
    other = tmp_path / "other"
    previous = tmp_path / ".other.stateset-previous"
    previous.mkdir()
    (previous / "notes.txt").write_text("unowned backup")
    with pytest.raises(ValueError, match="without a publication journal"):
        save(other, b"new")
    assert (previous / "notes.txt").read_text() == "unowned backup"


def test_invalid_journal_cannot_remove_another_directory(tmp_path):
    path = tmp_path / "saved"
    victim = tmp_path / "victim"
    victim.mkdir()
    (victim / "keep").write_text("keep")
    (tmp_path / ".saved.stateset-publication.json").write_text(
        '{"schema_version":1,"staging":"../victim"}'
    )
    with pytest.raises(ValueError, match="journal"):
        read(path)
    assert (victim / "keep").read_text() == "keep"


@pytest.mark.parametrize("overwrite", [False, True])
def test_sync_failure_after_publish_rolls_back(tmp_path, monkeypatch, overwrite):
    path = tmp_path / "saved"
    if overwrite:
        save(path, b"original")
    sync = publication._sync_directory
    failed = False

    def fail_once(directory):
        nonlocal failed
        if (
            directory == tmp_path
            and path.exists()
            and (path / "training_state.pt").read_bytes() == b"new"
            and not failed
        ):
            failed = True
            raise OSError("directory sync failed")
        return sync(directory)

    monkeypatch.setattr(publication, "_sync_directory", fail_once)
    with pytest.raises(OSError, match="directory sync failed"):
        save(path, b"new")
    assert failed
    if overwrite:
        assert read(path) == b"original"
    else:
        assert not path.exists()


def test_failed_rollback_keeps_journal_for_next_access(tmp_path, monkeypatch):
    path = tmp_path / "saved"
    save(path, b"original")
    replace = publication.os.replace

    def fail(source, target):
        if Path(target) == path:
            raise OSError("destination temporarily unavailable")
        return replace(source, target)

    monkeypatch.setattr(publication.os, "replace", fail)
    with pytest.raises(OSError, match="temporarily unavailable"):
        save(path, b"new")
    assert (tmp_path / ".saved.stateset-publication.json").exists()
    assert (tmp_path / ".saved.stateset-previous").exists()
    monkeypatch.setattr(publication.os, "replace", replace)
    assert read(path) == b"original"


def test_crash_during_recovery_cleanup_preserves_legacy_checkpoint(tmp_path):
    path = tmp_path / "saved"
    previous = tmp_path / ".saved.stateset-previous"
    previous.mkdir()
    (previous / "training_state.pt").write_bytes(b"legacy")
    staging = tmp_path / ".saved.stateset-stage-interrupted"
    staging.mkdir()
    (staging / "training_state.pt").write_bytes(b"unfinished")
    publication._write_json(
        tmp_path / ".saved.stateset-publication.json",
        {"schema_version": 1, "staging": staging.name},
    )
    script = """
import os
import sys
from pathlib import Path
from stateset_agents.training import checkpoint_publication as publication
remove = publication.shutil.rmtree
def crash(path):
    remove(path)
    os._exit(72)
publication.shutil.rmtree = crash
with publication.checkpoint_read(Path(sys.argv[1])):
    pass
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(path)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 72, result.stderr.decode()
    assert not (tmp_path / ".saved.stateset-publication.json").exists()
    with publication.checkpoint_read(path):
        assert (path / "training_state.pt").read_bytes() == b"legacy"
