"""Small structural PEFT export fixtures; these do not represent trained models."""

import json
from pathlib import Path
from typing import Any

from stateset_agents.data.source import SFTDatasetIdentity
from stateset_agents.training.lineage import AdapterManifest, write_manifest


def write_tiny_text_policy(snapshot: Path, *, reply_token_id: int = 5) -> Any:
    """Save a real GPT-2 whose greedy output does not depend on random weights.

    Residual blocks are zero and layer norms are identity-scaled. The final
    prompt token and reply embedding share a direction, with a strictly larger
    output projection for the reply. Special-token replies exercise rejection.
    """
    import torch
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    tokenizer = Tokenizer(
        WordLevel(
            {"<unk>": 0, "<eos>": 1, "<pad>": 2, "user": 3, "assistant": 4, "reply": 5},
            unk_token="<unk>",
        )
    )
    tokenizer.pre_tokenizer = Whitespace()
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        eos_token="<eos>",
        pad_token="<pad>",
    )
    fast.chat_template = "{% for message in messages %}{{ message['role'] + ' ' + message['content'] + ' ' }}{% endfor %}assistant "
    snapshot.mkdir(parents=True, exist_ok=True)
    fast.save_pretrained(snapshot)
    model = GPT2LMHeadModel(
        GPT2Config(
            n_layer=1,
            n_head=1,
            n_embd=16,
            n_positions=128,
            vocab_size=6,
            bos_token_id=1,
            eos_token_id=1,
            pad_token_id=2,
        )
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        for module in model.modules():
            if isinstance(module, torch.nn.LayerNorm):
                module.weight.fill_(1)
        model.transformer.wte.weight[4, :2] = torch.tensor([1.0, -1.0])
        model.transformer.wte.weight[reply_token_id, :2] = torch.tensor([2.0, -2.0])
    model.save_pretrained(snapshot)
    return model


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
