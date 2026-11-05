"""Multimodal fine-tuning contracts without downloading Nemotron weights."""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from typer.testing import CliRunner

from stateset_agents.cli import app
from stateset_agents.data.nemotron_vl import NemotronVLCollator, NemotronVLDataset
from stateset_agents.training.nemotron_vl import (
    NEMOTRON_VL_LORA_TARGETS,
    NEMOTRON_VL_MODELS,
    NEMOTRON_VL_REVISIONS,
    NemotronVLConfig,
    describe_nemotron_vl,
    train_nemotron_vl,
)

Image = pytest.importorskip(
    "PIL.Image", reason="install the optional nemotron-vl extra"
)


def write_data(directory: Path, *, name: str = "train.jsonl") -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (2, 2), (100, 60, 20)).save(directory / "receipt.png")
    row = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": "receipt.png"},
                    {"type": "text", "text": "Total?"},
                ],
            },
            {"role": "assistant", "content": "20 dollars"},
        ]
    }
    path = directory / name
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


class Processor:
    """A deterministic pixel-preserving processor oracle, not a Nemotron model."""

    image_token = "<image>"

    def __init__(self):
        self.tokenizer = self

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        return "".join(f"{m['role']}:{m['content']}<eot>" for m in messages) + (
            "assistant:" if add_generation_prompt else ""
        )

    def __call__(self, *, text, images=None, **kwargs):
        expanded = text.replace("<image>", "<img>@@@</img>")
        ids = torch.tensor([[ord(c) for c in expanded]])
        result = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
        if images:
            result["pixel_values"] = torch.tensor(
                [[list(image.getpixel((0, 0)))] for image in images],
                dtype=torch.float32,
            ).reshape(len(images), 3, 1, 1)
        return result


@pytest.mark.parametrize("model", ["super-vl", "bolt"])
def test_official_bf16_pins_and_profiles(model):
    config = NemotronVLConfig(model=model)
    assert config.model_id == NEMOTRON_VL_MODELS[model]
    assert config.model_revision == NEMOTRON_VL_REVISIONS[model]
    assert config.model_id.endswith("-BF16")
    assert "24GB" not in config.model_id
    assert describe_nemotron_vl(config)["gpu_training_verified"] is False
    assert (
        config.model_revision
        != NemotronVLConfig(model=model, revision="a" * 40).model_revision
    )


@pytest.mark.parametrize(
    "override",
    [
        {"model": "lightning"},
        {"revision": "main"},
        {"revision": "0" * 40},
        {"max_steps": 0},
        {"max_steps": True},
        {"max_length": 1},
        {"learning_rate": float("nan")},
        {"learning_rate": float("inf")},
        {"lora_rank": 0},
        {"gradient_accumulation_steps": 0},
        {"output_dir": " "},
        {"use_4bit": True},
        {"model_name": "nvidia/fake-24GB"},
    ],
)
def test_config_rejects_invalid_and_inference_settings(override):
    with pytest.raises(ValueError):
        NemotronVLConfig.model_validate(override)


def test_dataset_preserves_images_and_multiturn(tmp_path):
    path = write_data(tmp_path)
    dataset = NemotronVLDataset(path)
    assert len(dataset) == 1
    image = dataset[0]["conversation"][0]["content"][0]["image"]
    assert image.mode == "RGB" and image.getpixel((0, 0)) == (100, 60, 20)
    assert isinstance(dataset.records[0]["conversation"][0]["content"][0]["image"], str)
    sample = dataset[0]
    sample["conversation"].extend(
        [
            {"role": "user", "content": [{"type": "text", "text": "Currency?"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "USD"}]},
        ]
    )
    batch = NemotronVLCollator(Processor())([sample])
    labels = batch["labels"][0]
    supervised = "".join(chr(int(x)) for x in labels if x != -100)
    assert supervised == "20 dollars<eot>USD<eot>"
    assert batch["pixel_values"].dtype == torch.bfloat16
    assert batch["pixel_values"][0, 0, 0, 0] == 100
    assert batch["labels"].shape == batch["input_ids"].shape
    assert "@@@" not in supervised and "Total?" not in supervised


@pytest.mark.parametrize(
    "mutation",
    [
        "video",
        "remote",
        "missing",
        "escape",
        "no_image",
        "no_assistant",
        "tool",
        "control_token",
        "unknown_field",
        "assistant_image",
        "empty_text",
    ],
)
def test_dataset_rejects_silent_media_loss_and_bad_rows(tmp_path, mutation):
    path = write_data(tmp_path)
    row = json.loads(path.read_text())
    user, assistant = row["messages"]
    image = user["content"][0]
    if mutation == "video":
        image.update(type="video")
    elif mutation == "remote":
        image["image"] = "https://example.com/image.png"
    elif mutation == "missing":
        image["image"] = "absent.png"
    elif mutation == "escape":
        outside = tmp_path.parent / f"{tmp_path.name}-outside.png"
        Image.new("RGB", (1, 1)).save(outside)
        image["image"] = str(outside)
    elif mutation == "no_image":
        user["content"] = "Total?"
    elif mutation == "no_assistant":
        row["messages"].pop()
    elif mutation == "tool":
        assistant["role"] = "tool"
    elif mutation == "control_token":
        assistant["content"] = "<image>"
    elif mutation == "unknown_field":
        user["ignored_video"] = "video.mp4"
    elif mutation == "assistant_image":
        assistant["content"] = [image]
    else:
        assistant["content"] = " "
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="train.jsonl:1"):
        NemotronVLDataset(path)


def test_dataset_checks_empty_invalid_json_and_symlinks(tmp_path):
    path = tmp_path / "train.jsonl"
    for contents in ("\n", "[]\n", "{broken\n"):
        path.write_text(contents)
        with pytest.raises(ValueError):
            NemotronVLDataset(path)
    path = write_data(tmp_path)
    original = tmp_path / "receipt.png"
    outside = tmp_path.parent / f"{tmp_path.name}-symlink.png"
    Image.new("RGB", (1, 1)).save(outside)
    original.unlink()
    original.symlink_to(outside)
    with pytest.raises(ValueError, match="inside image_root"):
        NemotronVLDataset(path)


def test_multiimage_padding_preserves_content_and_original_images(tmp_path):
    sample = NemotronVLDataset(write_data(tmp_path))[0]
    first = sample["conversation"][0]["content"][0]["image"]
    second = Image.new("RGB", (4, 2), "white")
    sample["conversation"][0]["content"].insert(1, {"type": "image", "image": second})

    class Observe(Processor):
        def __call__(self, *, images=None, **kwargs):
            if images:
                assert all(image.size == (4, 2) for image in images)
                assert images[0].getpixel((1, 0)) == (100, 60, 20)
                assert images[0].getpixel((0, 0)) == (0, 0, 0)
            return super().__call__(images=images, **kwargs)

    batch = NemotronVLCollator(Observe())([sample])
    assert batch["pixel_values"].shape[0] == 2
    assert first.size == (2, 2) and second.size == (4, 2)


def test_collator_refuses_truncation_missing_pixels_and_prefix_drift(tmp_path):
    sample = NemotronVLDataset(write_data(tmp_path))[0]
    with pytest.raises(ValueError, match="exceeds max_length"):
        NemotronVLCollator(Processor(), max_length=4)([sample])
    with pytest.raises(ValueError, match="batch size 1"):
        NemotronVLCollator(Processor())([sample, sample])

    class NoPixels(Processor):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result.pop("pixel_values", None)
            return result

    with pytest.raises(ValueError, match="discarded image"):
        NemotronVLCollator(NoPixels())([sample])

    class Drift(Processor):
        def __call__(self, **kwargs):
            result = super().__call__(**kwargs)
            result["input_ids"][0, 0] = len(kwargs["text"])
            return result

    with pytest.raises(ValueError, match="prefix mismatch"):
        NemotronVLCollator(Drift())([sample])


def test_real_lora_step_consumes_pixels_and_keeps_vision_frozen(tmp_path):
    import copy

    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import PretrainedConfig, Trainer, TrainingArguments

    class TinyVisionLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = PretrainedConfig()
            self.config.model_type = "tiny-nemotron-contract"
            self.language_model = torch.nn.Module()
            self.language_model.attn = torch.nn.Module()
            self.language_model.attn.q_proj = torch.nn.Linear(8, 256)
            self.embedding = torch.nn.Embedding(256, 8)
            self.vision_model = torch.nn.Linear(3, 8)

        def forward(self, input_ids, pixel_values, labels, **kwargs):
            pixels = pixel_values.float().flatten(1) / 255
            hidden = self.embedding(input_ids) + self.vision_model(pixels).unsqueeze(1)
            logits = self.language_model.attn.q_proj(hidden)
            loss = torch.nn.functional.cross_entropy(
                logits[:, :-1].reshape(-1, 256),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            return {"loss": loss, "logits": logits}

    torch.manual_seed(4)
    base = TinyVisionLM()
    initial = copy.deepcopy(base)
    vision_before = base.vision_model.weight.detach().clone()
    model = get_peft_model(
        base, LoraConfig(r=2, target_modules=NEMOTRON_VL_LORA_TARGETS)
    )
    sample = NemotronVLDataset(write_data(tmp_path))[0]
    batch = NemotronVLCollator(Processor())([sample])
    before = {
        k: v.detach().clone() for k, v in model.named_parameters() if v.requires_grad
    }
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=0.01
    )
    loss = model(**batch)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()
    assert any(
        not torch.equal(before[k], v)
        for k, v in model.named_parameters()
        if k in before
    )
    assert torch.equal(vision_before, base.vision_model.weight)
    assert base.vision_model.weight.grad is None
    model.save_pretrained(tmp_path / "adapter", safe_serialization=True)
    assert (tmp_path / "adapter" / "adapter_model.safetensors").is_file()
    restored = PeftModel.from_pretrained(initial, tmp_path / "adapter")
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.equal(model(**batch)["loss"], restored(**batch)["loss"])
        blank = dict(batch, pixel_values=torch.zeros_like(batch["pixel_values"]))
        assert not torch.equal(restored(**batch)["loss"], restored(**blank)["loss"])
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(tmp_path / "trainer"),
            use_cpu=True,
            max_steps=2,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            remove_unused_columns=False,
            label_names=["labels"],
            report_to=[],
            save_strategy="no",
            disable_tqdm=True,
        ),
        train_dataset=NemotronVLDataset(tmp_path / "train.jsonl"),
        data_collator=NemotronVLCollator(Processor()),
    )
    trainer.model_accepts_loss_kwargs = False
    trainer.train()
    assert trainer.state.global_step == 2
    trainer.save_model(str(tmp_path / "trainer"))
    assert (tmp_path / "trainer" / "adapter_model.safetensors").is_file()


@pytest.mark.parametrize("model", ["super-vl", "bolt"])
@pytest.mark.parametrize("fsdp", [False, True])
def test_training_wiring_pins_processor_model_and_exports_adapters(
    tmp_path, monkeypatch, model, fsdp
):
    import peft
    import transformers

    path = write_data(tmp_path)
    output = tmp_path / "output"
    config = NemotronVLConfig(
        model=model,
        train_data=str(path),
        output_dir=str(output),
        max_steps=1,
        fsdp=fsdp,
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setattr(transformers, "set_seed", lambda seed: None)
    monkeypatch.setenv("WORLD_SIZE", "2" if fsdp else "1")
    processor = MagicMock()
    processor_loader = MagicMock(return_value=processor)
    model_loader = MagicMock(return_value=MagicMock())
    settings = SimpleNamespace(llm_config=SimpleNamespace())
    monkeypatch.setattr(transformers.AutoProcessor, "from_pretrained", processor_loader)
    monkeypatch.setattr(
        transformers.AutoConfig, "from_pretrained", MagicMock(return_value=settings)
    )
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText, "from_pretrained", model_loader
    )
    parameter = SimpleNamespace(requires_grad=True)
    wrapped = MagicMock()
    wrapped.named_parameters.return_value = [
        ("base_model.model.language_model.attn.q_proj.lora_A.weight", parameter)
    ]
    monkeypatch.setattr(peft, "get_peft_model", MagicMock(return_value=wrapped))
    arguments = MagicMock()
    monkeypatch.setattr(transformers, "TrainingArguments", arguments)
    trainer = MagicMock()
    trainer.state.global_step = 1
    trainer.train.return_value.metrics = {"train_loss": 0.5}
    trainer.is_world_process_zero.return_value = True
    factory = MagicMock(return_value=trainer)
    monkeypatch.setattr(transformers, "Trainer", factory)
    result = train_nemotron_vl(config)
    assert result["global_step"] == 1
    for loader in (
        processor_loader,
        model_loader,
        transformers.AutoConfig.from_pretrained,
    ):
        assert loader.call_args.args[0] == config.model_id
        assert loader.call_args.kwargs["revision"] == config.model_revision
        assert loader.call_args.kwargs["code_revision"] == config.model_revision
        assert loader.call_args.kwargs["trust_remote_code"] is True
    assert "device_map" not in model_loader.call_args.kwargs
    assert settings.llm_config.use_cache is False
    assert settings.llm_config.num_logits_to_keep == 0
    assert settings.video_pruning_rate == 0
    if fsdp:
        fsdp_config = arguments.call_args.kwargs["fsdp_config"]
        assert fsdp_config["version"] == 2
        assert fsdp_config["reshard_after_forward"] is True
        assert fsdp_config["activation_checkpointing"] is True
        assert fsdp_config["transformer_layer_cls_to_wrap"] == [
            "NemotronHPuzzleBlock" if model == "bolt" else "NemotronHBlock"
        ]
    assert isinstance(factory.call_args.kwargs["data_collator"], NemotronVLCollator)
    trainer.save_model.assert_called_once_with(str(output))
    processor.save_pretrained.assert_called_once_with(str(output))
    assert (
        json.loads((output / "stateset_training.json").read_text())["global_step"] == 1
    )


def test_preflight_checks_data_before_weights_and_cuda(tmp_path, monkeypatch):
    import transformers

    loader = MagicMock()
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText, "from_pretrained", loader
    )
    with pytest.raises(FileNotFoundError):
        train_nemotron_vl(NemotronVLConfig(train_data=str(tmp_path / "missing.jsonl")))
    loader.assert_not_called()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="BF16-capable CUDA"):
        train_nemotron_vl(NemotronVLConfig(train_data=str(write_data(tmp_path))))
    loader.assert_not_called()


@pytest.mark.parametrize("fsdp,world", [(False, "2"), (True, "1")])
def test_invalid_distributed_launch_fails_before_loading(
    tmp_path, monkeypatch, fsdp, world
):
    import transformers

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_bf16_supported", lambda: True)
    monkeypatch.setenv("WORLD_SIZE", world)
    loader = MagicMock()
    monkeypatch.setattr(
        transformers.AutoModelForImageTextToText, "from_pretrained", loader
    )
    with pytest.raises(ValueError, match="FSDP|fsdp"):
        train_nemotron_vl(
            NemotronVLConfig(train_data=str(write_data(tmp_path)), fsdp=fsdp)
        )
    loader.assert_not_called()


@pytest.mark.parametrize("model", ["super-vl", "bolt"])
def test_cli_config_and_dry_run(model, tmp_path):
    runner = CliRunner()
    command = f"nemotron-3-5-{model}"
    config_path = tmp_path / "config.json"
    result = runner.invoke(app, [command, "--write-config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(config_path.read_text())["model"] == model
    assert (
        json.loads(config_path.read_text())["revision"] == NEMOTRON_VL_REVISIONS[model]
    )
    result = runner.invoke(app, [command, "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["model_revision"] == NEMOTRON_VL_REVISIONS[model]
    result = runner.invoke(
        app, [command, "--train-data", str(write_data(tmp_path)), "--check-data"]
    )
    assert result.exit_code == 0, result.output
    assert json.loads(result.output)["train_examples"] == 1
    for flag, enabled in (("--fsdp", True), ("--no-fsdp", False)):
        result = runner.invoke(
            app,
            [
                command,
                "--config",
                str(config_path),
                flag,
                "--write-config",
                str(config_path),
            ],
        )
        assert result.exit_code == 0, result.output
        assert json.loads(config_path.read_text())["fsdp"] is enabled
    wrong = "bolt" if model == "super-vl" else "super-vl"
    config_path.write_text(json.dumps({"model": wrong}))
    result = runner.invoke(app, [command, "--config", str(config_path)])
    assert result.exit_code == 1 and "requires model=" in result.output


@pytest.mark.parametrize("model", ["super-vl", "bolt"])
def test_cli_executes_selected_training_path(monkeypatch, model):
    training = importlib.import_module("stateset_agents.training.nemotron_vl")
    run = MagicMock(return_value={"completed": True})
    monkeypatch.setattr(training, "train_nemotron_vl", run)
    result = CliRunner().invoke(
        app, [f"nemotron-3-5-{model}", "--no-dry-run", "--max-steps", "2"]
    )
    assert result.exit_code == 0, result.output
    assert run.call_args.args[0].model == model
    assert run.call_args.args[0].max_steps == 2


def test_lazy_exports_and_dry_run_do_not_import_gpu_dependencies():
    code = """
import json, sys
from stateset_agents.training import NemotronVLConfig, describe_nemotron_vl
from stateset_agents.cli import app
from typer.testing import CliRunner
assert 'torch' not in sys.modules
result = CliRunner().invoke(app, ['nemotron-3-5-bolt'])
assert result.exit_code == 0, result.output
assert json.loads(result.output)['precision'] == 'bfloat16'
assert not {'torch', 'peft', 'transformers', 'PIL'} & set(sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30
    )
    assert result.returncode == 0, result.stderr
    training = importlib.import_module("stateset_agents.training")
    assert training.NemotronVLConfig is NemotronVLConfig


def test_validation_cannot_reuse_train_file(tmp_path):
    path = write_data(tmp_path)
    config = NemotronVLConfig(train_data=str(path), validation_data=str(path))
    with pytest.raises(ValueError, match="separate dataset files"):
        describe_nemotron_vl(config, check_data=True)
