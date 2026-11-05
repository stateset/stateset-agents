"""Reinspect actual tiny CPU exports from the passing local test suite."""

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from stateset_agents.data.source import SFTDatasetIdentity
from stateset_agents.training.sft_artifacts import verify_sft_adapter

ROOT = Path("outputs/sft-export-review-20261019")
FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "tokenizer_config.json",
    "tokenizer.json",
    "special_tokens_map.json",
    "stateset_manifest.json",
)


def locate(root, pattern):
    found = {
        path.resolve()
        for path in root.glob(pattern)
        if not path.parent.is_symlink()
        and path.resolve().is_relative_to(root.resolve())
    }
    assert len(found) == 1, found
    return found.pop()


def retain(source, destination):
    destination.mkdir(exist_ok=False)
    for name in FILES:
        path = source / name
        if not path.exists():
            continue
        assert (
            path.is_file()
            and not path.is_symlink()
            and path.stat().st_size < 16 * 1024 * 1024
        )
        shutil.copyfile(path, destination / name)


def main(pytest_root):
    successful = locate(pytest_root, "popen-gw*/test_real_cpu_in_memory_sft_*/adapter")
    incomplete = locate(pytest_root, "popen-gw*/test_real_cpu_sft_does_not_s*/adapter")
    good, bad = ROOT / "successful-tiny-export", ROOT / "missing-lineage-export"
    retain(successful, good)
    retain(incomplete, bad)
    rows = [
        {
            "messages": [
                {"role": "user", "content": "request"},
                {"role": "assistant", "content": "response"},
            ]
        }
    ]
    identity = SFTDatasetIdentity.from_rows(rows)
    options = {
        "base_model": "local/random-tiny",
        "lora_r": 2,
        "dataset_identity": identity,
    }
    summary = verify_sft_adapter(good, **options)
    try:
        verify_sft_adapter(bad, **options)
    except ValueError as error:
        assert "stateset_manifest.json" in str(error)
        rejected = str(error)
    else:
        raise AssertionError("Missing-lineage export passed")
    assert (bad / "adapter_model.safetensors").is_file()
    with tempfile.TemporaryDirectory(prefix="sft-export-corrupt-") as staging:
        damaged = Path(staging) / "adapter"
        shutil.copytree(good, damaged)
        (damaged / "adapter_model.safetensors").write_bytes(b"broken")
        try:
            verify_sft_adapter(damaged, **options)
        except ValueError:
            pass
        else:
            raise AssertionError("Corrupt export passed")
    report = {
        "scope": "Structural reinspection of two actual tiny CPU test exports; no training or model loading",
        "successful_test_export": str(successful),
        "failed_lineage_test_export": str(incomplete),
        "verified_export": summary,
        "missing_lineage_rejected": rejected,
        "failed_run_weights_preserved": True,
        "corrupt_copy_rejected": True,
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
                "verified_tiny_exports": 1,
                "rejected_exports": 2,
                "provider_calls": 0,
                "target_model_updates": 0,
            }
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pytest-root", type=Path, required=True)
    main(parser.parse_args().pytest_root)
