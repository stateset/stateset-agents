"""Check source and normalized training identities on the retained industry data."""

import json
from pathlib import Path

from stateset_agents.data.text_sft import load_text_sft_snapshot
from stateset_agents.training.sft import load_sft_dataset

ROOT = Path("outputs/local-sft-identity-review-20261019")
DATA = Path("outputs/industry-datasets-20261019")


def main():
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    results = []
    for domain in ("ecommerce", "logistics", "saas"):
        path = DATA / domain / "train.jsonl"
        expected = manifest["datasets"][domain]["files"]["train.jsonl"]
        generic, source = load_sft_dataset(path, expected_sha256=expected)
        normalized, text_source = load_sft_dataset(
            path, text_only=True, expected_sha256=expected
        )
        text_snapshot = load_text_sft_snapshot(str(path))
        source.verify_rows(generic)
        text_source.verify_rows(normalized)
        assert (
            len(generic)
            == len(normalized)
            == source.examples
            == text_source.examples
            == 512
        )
        assert source.source == text_source.source
        assert normalized == text_snapshot.train
        assert text_source.content_sha256 == text_snapshot.content_sha256["train"]
        assert source.source.sha256 == text_snapshot.dataset_sha256["train"] == expected
        assert all(row.get("synthetic") is True for row in generic)
        assert all(set(row) == {"messages"} for row in normalized)
        rejections = []
        for split in ("validation", "test"):
            for text_only in (False, True):
                try:
                    load_sft_dataset(
                        path.with_name(f"{split}.jsonl"), text_only=text_only
                    )
                except ValueError:
                    rejections.append({"split": split, "text_only": text_only})
                else:
                    raise AssertionError("Held-out data reached local SFT loading")
        results.append(
            {
                "domain": domain,
                "source_identity": source.source.to_dict(),
                "parsed_training_rows": source.examples,
                "parsed_content_sha256": source.content_sha256,
                "normalized_training_rows": text_source.examples,
                "normalized_content_sha256": text_source.content_sha256,
                "held_out_rejections": rejections,
            }
        )
    report = {
        "scope": "Industry corpus source/row identity capture; no model loading or training",
        "datasets": results,
        "training_source_rows": 1536,
        "provider_calls": 0,
        "target_model_updates": 0,
        "quality_certification": False,
    }
    (ROOT / "review-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "source_rows": 1536,
                "held_out_rejections": 12,
                "provider_calls": 0,
                "target_model_updates": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
