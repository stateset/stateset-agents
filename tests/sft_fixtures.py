"""Small structural PEFT export fixtures; these do not represent trained models."""

import json
from pathlib import Path
from typing import Any

from stateset_agents.data.source import SFTDatasetIdentity
from stateset_agents.training.lineage import AdapterManifest, write_manifest


def write_sft_export(
    output: Path,
    *,
    base_model: str,
    lora_r: int,
    dataset_identity: SFTDatasetIdentity,
    model_revision: str | None = None,
    **_: Any,
) -> None:
    """Write a complete, tiny safe LoRA fixture with captured input lineage."""
    import numpy as np
    from safetensors.numpy import save_file

    output.mkdir(parents=True, exist_ok=True)
    (output / "adapter_config.json").write_text(
        json.dumps({"peft_type": "LORA", "r": lora_r})
    )
    (output / "tokenizer_config.json").write_text(
        json.dumps({"tokenizer_class": "fixture"})
    )
    save_file(
        {
            "base_model.model.fixture.lora_A.weight": np.ones(
                (lora_r, 2), dtype=np.float32
            ),
            "base_model.model.fixture.lora_B.weight": np.ones(
                (2, lora_r), dtype=np.float32
            ),
        },
        str(output / "adapter_model.safetensors"),
    )
    source = dataset_identity.source
    write_manifest(
        output,
        AdapterManifest(
            base_model=base_model,
            dataset_path=source.path if source else None,
            dataset_sha256=source.sha256 if source else None,
            dataset_rows=source.rows if source else dataset_identity.examples,
            hyperparameters={
                "model_revision": model_revision,
                "training_content_sha256": dataset_identity.content_sha256,
                "training_input_rows": dataset_identity.examples,
                "effective_training_rows": dataset_identity.examples,
            },
        ),
    )
