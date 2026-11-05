"""Verify all generated corpora through the real Studio inspection and plan path."""

import asyncio
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

from stateset_agents.data.industry_synthetic import audit_industry_datasets
from stateset_agents.data.text_sft import DatasetFormatError, load_text_sft_snapshot
from stateset_agents.finetune_inspection import inspect_training_data
from stateset_agents.finetune_workspace import FinetuneConfig, plan_run
from stateset_agents.tui import Studio

ROOT = Path("outputs/industry-studio-review-20261019")
DATA = Path("outputs/industry-datasets-20261019").resolve()


def save(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


async def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    audit = await audit_industry_datasets(DATA)
    save("dataset-audit.json", audit)
    manifest = json.loads((DATA / "dataset_manifest.json").read_text())
    results = []
    for domain in manifest["domains"]:
        directory = DATA / domain
        config = FinetuneConfig(dataset=str(directory / "train.jsonl"), model=manifest["datasets"][domain]["base_model"])
        save(f"{domain}-config.json", asdict(config))
        snapshot = load_text_sft_snapshot(config.dataset, str(directory / "validation.jsonl"))
        report = inspect_training_data(Path(config.dataset))
        plan = plan_run(config)
        assert plan.examples == 512
        assert snapshot.provenance["train"]["synthetic_examples"] == 512
        assert report["dataset_provenance"]["synthetic_examples"] == 512
        assert snapshot.dataset_sha256["train"] == manifest["datasets"][domain]["files"]["train.jsonl"]
        assert any("not independently reviewed" in note for note in plan.notes)
        assert all(set(row) == {"messages"} for row in snapshot.train)
        rejected = []
        for split in ("validation", "test"):
            try:
                plan_run(FinetuneConfig(dataset=str(directory / f"{split}.jsonl")))
            except DatasetFormatError as error:
                assert error.diagnostic["code"] == "held_out_split"
                rejected.append(split)
            else:
                raise AssertionError("Held-out file reached launch planning")
        studio = Studio(ROOT / "unused-workspace", config)
        studio.data_report = report
        studio.data_signature = studio._data_settings_signature()
        studio.overlay = "data"
        (ROOT / f"{domain}-inspector.txt").write_text("\n".join(studio.lines(140, 40)) + "\n")
        studio.plan = plan
        studio.overlay = "review"
        (ROOT / f"{domain}-review.txt").write_text("\n".join(studio.lines(140, 40)) + "\n")
        assert not studio.workspace.exists()
        save(f"{domain}-snapshot.json", snapshot.describe())
        save(f"{domain}-inspection.json", report)
        save(f"{domain}-plan.json", plan.describe())
        results.append({"domain": domain, "training_examples": plan.examples,
            "synthetic_examples": report["dataset_provenance"]["synthetic_examples"],
            "held_out_rejections": rejected, "source_hash": hashlib.sha256(plan.data).hexdigest()})
    save("review-summary.json", {"complete": True, "datasets": results, "dataset_audit": audit,
        "provider_calls": 0, "model_updates": 0, "runs_prepared": 0,
        "quality_certification": False,
        "scope": "Studio parsing, provenance, inspection, snapshot and launch-review planning; no training or model evaluation"})
    print(json.dumps({"datasets": results, "verified_demonstrations": audit["verified_demonstrations"], "runs_prepared": 0}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
