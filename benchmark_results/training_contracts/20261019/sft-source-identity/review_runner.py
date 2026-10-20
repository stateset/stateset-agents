"""Check full industry corpora through captured-byte readers without training."""

import hashlib
import json
from pathlib import Path

from stateset_agents.data.source import DatasetSourceIdentity
from stateset_agents.remote.executor import RemoteExecutionError
from stateset_agents.remote.fireworks import _read_chat_rows
from stateset_agents.training.sft import parse_chat_dataset

ROOT = Path("outputs/sft-source-identity-review-20261019")
DATA = Path("outputs/industry-datasets-20261019")


def main():
    manifest_raw = (DATA / "dataset_manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    reports = []
    for domain in ("ecommerce", "logistics", "saas"):
        path = DATA / domain / "train.jsonl"
        raw = path.read_bytes()
        identity = DatasetSourceIdentity.from_bytes(path, raw)
        assert identity.sha256 == manifest["datasets"][domain]["files"]["train.jsonl"]
        generic_rows = parse_chat_dataset(raw)
        upload_rows = _read_chat_rows(path, data=raw)
        assert len(generic_rows) == len(upload_rows) == identity.rows == 512
        assert generic_rows == upload_rows
        assert all(row.get("synthetic") is True for row in generic_rows)
        rejections = []
        for split in ("validation", "test"):
            held_out = path.with_name(f"{split}.jsonl")
            captured = held_out.read_bytes()
            for name, read in (
                ("river_sft_reader", lambda raw=captured: parse_chat_dataset(raw)),
                (
                    "fireworks_upload_reader",
                    lambda path=held_out, raw=captured: _read_chat_rows(path, data=raw),
                ),
            ):
                try:
                    read()
                except (ValueError, RemoteExecutionError) as error:
                    assert "requested train split" in str(error)
                    rejections.append({"split": split, "reader": name})
                else:
                    raise AssertionError("Held-out corpus reached training reader")
        assert hashlib.sha256(path.read_bytes()).hexdigest() == identity.sha256
        reports.append(
            {
                "domain": domain,
                "source_identity": identity.to_dict(),
                "generic_reader_rows": len(generic_rows),
                "fireworks_reader_rows": len(upload_rows),
                "held_out_rejections": rejections,
            }
        )
    assert (DATA / "dataset_manifest.json").read_bytes() == manifest_raw
    result = {
        "scope": "Captured-byte parsing and source identities; no provider, tokenizer or training calls",
        "datasets": reports,
        "dataset_manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "training_source_rows": sum(r["source_identity"]["rows"] for r in reports),
        "provider_calls": 0,
        "model_updates": 0,
        "quality_certification": False,
    }
    (ROOT / "review-summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "training_source_rows": result["training_source_rows"],
                "held_out_rejections": sum(
                    len(r["held_out_rejections"]) for r in reports
                ),
                "provider_calls": 0,
                "model_updates": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
