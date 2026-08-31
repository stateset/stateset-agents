"""Tests for the provider-neutral S3 artifact transport and worker."""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from stateset_agents.remote.artifacts import S3ArtifactStore, parse_s3_uri
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.worker import decode_spec, run_object_store_job


class _Paginator:
    def __init__(self, client):
        self.client = client

    def paginate(self, *, Bucket, Prefix):
        yield {
            "Contents": [
                {"Key": key}
                for (bucket, key) in sorted(self.client.objects)
                if bucket == Bucket and key.startswith(Prefix)
            ]
        }


class FakeS3:
    def __init__(self):
        self.objects: dict[tuple[str, str], bytes] = {}

    def upload_file(self, source, bucket, key):
        self.objects[(bucket, key)] = Path(source).read_bytes()

    def download_file(self, bucket, key, destination):
        Path(destination).write_bytes(self.objects[(bucket, key)])

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return _Paginator(self)

    def delete_objects(self, *, Bucket, Delete):
        for row in Delete["Objects"]:
            self.objects.pop((Bucket, row["Key"]), None)


def test_parse_s3_uri_rejects_non_s3() -> None:
    assert parse_s3_uri("s3://bucket/path") == ("bucket", "path")
    with pytest.raises(ValueError):
        parse_s3_uri("https://bucket/path")


@pytest.mark.parametrize(
    "uri",
    [
        "s3://bucket/path?credential=secret",
        "s3://bucket/path#fragment",
        "s3://bucket/../outside",
    ],
)
def test_parse_s3_uri_rejects_ambiguous_or_traversing_uris(uri) -> None:
    with pytest.raises(ValueError):
        parse_s3_uri(uri)


def test_delete_refuses_bucket_root() -> None:
    with pytest.raises(ValueError, match="entire artifact bucket"):
        S3ArtifactStore(FakeS3()).delete_prefix("s3://bucket")


def test_directory_round_trip_and_delete(tmp_path) -> None:
    client = FakeS3()
    store = S3ArtifactStore(client)
    source = tmp_path / "source"
    source.mkdir()
    (source / "adapter.json").write_text("{}")
    (source / "nested").mkdir()
    (source / "nested" / "weights.bin").write_bytes(b"weights")

    uploaded = store.upload_directory(source, "s3://bucket/jobs/1/output")
    destination = tmp_path / "download"
    written = store.download_prefix("s3://bucket/jobs/1/output", destination)

    assert len(uploaded) == 2
    assert (destination / "adapter.json").read_text() == "{}"
    assert (destination / "nested" / "weights.bin").read_bytes() == b"weights"
    assert len(written) == 2
    store.delete_prefix("s3://bucket/jobs/1")
    assert not client.objects


def test_download_rejects_traversal_key(tmp_path) -> None:
    client = FakeS3()
    client.objects[("bucket", "jobs/1/output/../secret")] = b"no"
    with pytest.raises(RemoteExecutionError, match="unsafe artifact key"):
        S3ArtifactStore(client).download_prefix(
            "s3://bucket/jobs/1/output", tmp_path / "out"
        )


def test_worker_executes_dry_run_without_gpu(tmp_path) -> None:
    client = FakeS3()
    client.objects[("bucket", "input/data.jsonl")] = (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ]
            }
        )
        + "\n"
    ).encode()
    payload = {
        "dataset": "ignored",
        "base_model": "Qwen/test",
        "output_dir": "ignored",
        "dry_run": True,
        "num_epochs": 1,
        "lora_r": 8,
        "lora_alpha": 16,
        "learning_rate": 0.0001,
        "max_length": 64,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
    }
    encoded = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode()

    outcome = run_object_store_job(
        encoded,
        "s3://bucket/input/data.jsonl",
        "s3://bucket/output",
        store=S3ArtifactStore(client),
    )

    assert outcome["returncode"] == 0
    assert outcome["output_dir"] == "s3://bucket/output"
    assert decode_spec(encoded)["base_model"] == "Qwen/test"
