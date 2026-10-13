"""The direct-TRL shootout leg does the same work per step as the native leg.

TRL's GRPO batch is counted in completions; the shootout manifest (and the
native trainers) count prompts. v4 passed the prompt count straight through,
so TRL trained on one prompt per step against the native leg's four.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]


def _adapter():
    spec = importlib.util.spec_from_file_location(
        "trl_grpo_adapter", ROOT / "benchmarks" / "adapters" / "trl_grpo.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "manifest", ["shootout_manifest_v4.json", "shootout_manifest_v5.json"]
)
def test_trl_prompts_per_step_match_the_manifest(manifest):
    config = json.loads((ROOT / "benchmarks" / manifest).read_text())["config"]
    batch = _adapter().trl_batch_arguments(config)
    completions = (
        batch["per_device_train_batch_size"] * batch["gradient_accumulation_steps"]
    )
    # TRL requires the effective batch to divide into whole prompt groups.
    assert completions % config["num_generations"] == 0
    prompts = completions // config["num_generations"]
    assert prompts == (
        config["per_device_train_batch_size"] * config["gradient_accumulation_steps"]
    )


def test_samples_processed_counts_completions_actually_trained():
    config = {
        "max_steps": 200,
        "per_device_train_batch_size": 4,
        "gradient_accumulation_steps": 1,
        "num_generations": 4,
    }
    adapter = _adapter()
    batch = adapter.trl_batch_arguments(config)
    trained = (
        config["max_steps"]
        * batch["per_device_train_batch_size"]
        * batch["gradient_accumulation_steps"]
    )
    assert adapter.completions_per_run(config) == trained == 3200
