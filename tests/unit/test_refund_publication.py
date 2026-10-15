"""A failed dataset write must never expose a partial training bundle."""

import asyncio
import os
import selectors
import subprocess
import sys
from pathlib import Path

import pytest

from stateset_agents.data import refund_demonstrations as module
from stateset_agents.evaluation.agent_study import StudyConfig, prepare_study
from stateset_agents.evaluation.study_preflight import prepare_study_data
from stateset_agents.remote.river_rl import exclusive_run


async def prepare(destination):
    return await module.prepare_refund_data(
        destination, train_count=8, validation_count=8, test_count=8
    )


@pytest.mark.parametrize(
    "failed_file", ["train.json", "validation.json", "data_manifest.json"]
)
def test_failed_publication_leaves_destination_empty_and_retryable(
    tmp_path, monkeypatch, failed_file
):
    output = tmp_path / "data"
    original = module.atomic_json

    def write(path, value):
        original(path, value)
        assert not list(output.iterdir()), "partial data became visible before commit"
        if path.name == failed_file:
            raise OSError("disk full")

    monkeypatch.setattr(module, "atomic_json", write)
    with pytest.raises(OSError, match="disk full"):
        asyncio.run(prepare(output))
    assert output.exists() and not list(output.iterdir())
    monkeypatch.setattr(module, "atomic_json", original)
    asyncio.run(prepare(output))
    manifest, _ = module._load_data(output)
    assert manifest["selected"] == 8


def test_destination_filled_during_staging_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "data"
    original = module.atomic_json

    def write(path, value):
        original(path, value)
        if path.name == "data_manifest.json":
            (output / "user-evidence.txt").write_text("keep this")

    monkeypatch.setattr(module, "atomic_json", write)
    with pytest.raises(OSError):
        asyncio.run(prepare(output))
    assert (output / "user-evidence.txt").read_text() == "keep this"
    assert not (output / "data_manifest.json").exists()


def test_sibling_lock_prevents_concurrent_publication(tmp_path):
    output = tmp_path / "data"
    with exclusive_run(tmp_path / ".data.publication"):
        with pytest.raises(ValueError, match="Another driver"):
            asyncio.run(prepare(output))
    assert not output.exists()
    asyncio.run(prepare(output))


@pytest.mark.skipif(
    os.name == "nt", reason="POSIX pipe readiness and abrupt process death"
)
def test_killed_publisher_leaves_no_partial_bundle_and_os_releases_lock(tmp_path):
    output = tmp_path / "data"
    script = """
import asyncio, sys, time
from pathlib import Path
from stateset_agents.data import refund_demonstrations as module
original = module.atomic_json
def write(path, value):
    original(path, value)
    if path.name == 'train.json':
        print('staged', flush=True)
        time.sleep(60)
module.atomic_json = write
asyncio.run(module.prepare_refund_data(Path(sys.argv[1]), train_count=8, validation_count=8, test_count=8))
"""
    process = subprocess.Popen(
        [sys.executable, "-c", script, str(output)],
        cwd=Path(__file__).resolve().parents[2],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            assert selector.select(timeout=30), "publisher did not reach staging"
            assert process.stdout.readline().strip() == "staged"
        assert output.exists() and not list(output.iterdir())
        process.kill()
        process.communicate(timeout=10)
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=10)
    orphaned = list((tmp_path / ".data.publication").glob("stage-*"))
    assert len(orphaned) == 1 and (orphaned[0] / "train.json").exists()
    asyncio.run(prepare(output))
    assert module._load_data(output)[0]["selected"] == 8
    assert orphaned[
        0
    ].exists()  # Uncommitted staging remains separate from published data.


def test_study_preparation_recovers_after_failure_inside_bundle(tmp_path, monkeypatch):
    prepare_study(
        tmp_path, StudyConfig(train_count=8, validation_count=8, test_count=8, steps=2)
    )
    original = module.atomic_json

    def write(path, value):
        if path.name == "validation.json":
            raise OSError("interrupted dataset write")
        original(path, value)

    monkeypatch.setattr(module, "atomic_json", write)
    with pytest.raises(OSError):
        asyncio.run(prepare_study_data(tmp_path))
    monkeypatch.setattr(module, "atomic_json", original)
    report = asyncio.run(prepare_study_data(tmp_path))
    assert all(row["status"] == "created" for row in report["datasets"])


def test_failure_after_directory_commit_preserves_bundle_for_reuse(
    tmp_path, monkeypatch
):
    prepare_study(
        tmp_path, StudyConfig(train_count=8, validation_count=8, test_count=8, steps=2)
    )
    original = module.os.rename

    def rename(source, destination):
        original(source, destination)
        raise OSError("interrupted after directory commit")

    monkeypatch.setattr(module.os, "rename", rename)
    with pytest.raises(OSError, match="after directory commit"):
        asyncio.run(prepare_study_data(tmp_path))
    bundle = tmp_path / "seed-42/data"
    assert module._load_data(bundle)[0]["selected"] == 8
    before = {path.name: path.read_bytes() for path in bundle.iterdir()}
    monkeypatch.setattr(module.os, "rename", original)
    report = asyncio.run(prepare_study_data(tmp_path))
    assert report["datasets"][0]["status"] == "reused"
    assert sum(row["status"] == "created" for row in report["datasets"]) == 5
    assert {path.name: path.read_bytes() for path in bundle.iterdir()} == before
