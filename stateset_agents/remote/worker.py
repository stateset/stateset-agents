"""Container entrypoint shared by object-store-backed remote providers."""

from __future__ import annotations

import argparse
import base64
import json
import tempfile
from pathlib import Path
from typing import Any

from stateset_agents.remote.artifacts import S3ArtifactStore
from stateset_agents.training.sft import run_sft_job

RESULT_MARKER = "STATESET_REMOTE_RESULT="


def decode_spec(encoded: str) -> dict[str, Any]:
    """Decode the URL-safe base64 JSON job envelope."""
    try:
        payload = base64.urlsafe_b64decode(encoded.encode("ascii"))
        value = json.loads(payload)
    except (ValueError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid remote job specification") from exc
    if not isinstance(value, dict):
        raise ValueError("remote job specification must be a JSON object")
    return value


def run_object_store_job(
    encoded_spec: str,
    input_uri: str,
    output_uri: str,
    *,
    store: S3ArtifactStore | None = None,
) -> dict[str, Any]:
    """Download an input, execute the packaged job, and upload all outputs."""
    artifact_store = store or S3ArtifactStore()
    payload = decode_spec(encoded_spec)
    with tempfile.TemporaryDirectory(prefix="stateset-job-") as temporary:
        root = Path(temporary)
        dataset = root / "input" / "dataset.jsonl"
        output = root / "output"
        artifact_store.download_file(input_uri, dataset)
        payload["dataset"] = str(dataset)
        payload["output_dir"] = str(output)
        outcome = run_sft_job(payload)
        uploaded = (
            artifact_store.upload_directory(output, output_uri)
            if output.exists()
            else []
        )
        outcome["uploaded_artifacts"] = uploaded
        # Avoid returning a container-local path to the caller.
        outcome["output_dir"] = output_uri
        return outcome


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint used inside provider containers."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec-b64", required=True)
    parser.add_argument("--input-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--s3-endpoint-url")
    args = parser.parse_args(argv)
    store = S3ArtifactStore(endpoint_url=args.s3_endpoint_url)
    try:
        outcome = run_object_store_job(
            args.spec_b64, args.input_uri, args.output_uri, store=store
        )
    except Exception as exc:  # noqa: BLE001 - must report container failures
        outcome = {
            "returncode": 1,
            "logs": [f"remote worker failed: {exc}"],
            "output_dir": args.output_uri,
            "uploaded_artifacts": [],
        }
    print(RESULT_MARKER + json.dumps(outcome, sort_keys=True))
    return int(outcome.get("returncode", 1))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
