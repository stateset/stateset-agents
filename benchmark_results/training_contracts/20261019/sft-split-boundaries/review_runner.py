"""Check retained industry corpora through SFT split boundaries without training."""

import asyncio
import hashlib
import json
from pathlib import Path

from stateset_agents.data.finetuning import (
    load_finetuning_data,
    validate_finetuning_source_split,
)
from stateset_agents.data.industry_synthetic import audit_industry_datasets
from stateset_agents.training.industry import train_industry_project
from stateset_agents.training.sft import load_chat_dataset, run_sft_job

ROOT = Path("outputs/sft-split-review-20261019")
DATA = Path("outputs/industry-datasets-20261019")


def payload(dataset, output):
    return {
        "dataset": str(dataset),
        "output_dir": str(output),
        "base_model": "Qwen/Qwen3.5-2B",
        "dry_run": True,
        "num_epochs": 1,
        "lora_r": 16,
        "learning_rate": 2e-5,
        "max_length": 1024,
    }


async def main():
    audit = await audit_industry_datasets(DATA)
    reports = []
    for domain in ("ecommerce", "logistics", "saas"):
        project = DATA / domain
        train = project / "train.jsonl"
        output = ROOT / f"unused-{domain}-output"
        digest = hashlib.sha256(train.read_bytes()).hexdigest()
        validate_finetuning_source_split(train)
        rows = load_chat_dataset(train)
        assert len(rows) == 512
        validation = load_finetuning_data(
            project / "validation.jsonl", expected_split="validation"
        )
        test = load_finetuning_data(project / "test.jsonl", expected_split="test")
        assert len(validation) == 64 and len(test) == 128
        assert train_industry_project(project)["status"] == "planned"
        result = run_sft_job(payload(train, output))
        assert result["returncode"] == 0, result
        failures = []
        for split in ("validation", "test"):
            source = project / f"{split}.jsonl"
            for name, reader in (
                ("provider_preflight", validate_finetuning_source_split),
                ("sft_loader", load_chat_dataset),
                (
                    "industry_training_read",
                    lambda p: load_finetuning_data(p, expected_split="train"),
                ),
            ):
                try:
                    reader(source)
                except ValueError as exc:
                    assert "requested train split" in str(exc)
                    failures.append({"split": split, "boundary": name})
                else:
                    raise AssertionError("Held-out source passed training boundary")
            blocked = run_sft_job(payload(source, output))
            assert blocked["returncode"] == 1
            failures.append({"split": split, "boundary": "packaged_worker"})
        assert not output.exists()
        assert digest == hashlib.sha256(train.read_bytes()).hexdigest()
        reports.append(
            {
                "domain": domain,
                "train_rows": len(rows),
                "validation_rows": len(validation),
                "test_rows": len(test),
                "train_sha256": digest,
                "declared_synthetic_rows": sum(
                    r.get("synthetic") is True for r in rows
                ),
                "worker_dry_run_returncode": result["returncode"],
                "held_out_rejections": failures,
                "output_created": False,
            }
        )
    report = {
        "audit": audit,
        "domains": reports,
        "dataset_manifest_sha256": hashlib.sha256(
            (DATA / "dataset_manifest.json").read_bytes()
        ).hexdigest(),
        "provider_calls": 0,
        "model_updates": 0,
        "quality_certification": False,
        "scope": "Declared split checks, dataset reconstruction, CPU SFT previews; no model evaluation",
    }
    (ROOT / "review-summary.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(
        json.dumps(
            {
                "audited": audit["verified_demonstrations"],
                "training_rows": 1536,
                "held_out_boundary_rejections": sum(
                    len(r["held_out_rejections"]) for r in reports
                ),
                "provider_calls": 0,
                "model_updates": 0,
            }
        )
    )


if __name__ == "__main__":
    asyncio.run(main())
