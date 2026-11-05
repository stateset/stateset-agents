"""Completion requires a complete safe LoRA export and matching input lineage."""

import json

import pytest

from stateset_agents.data.source import DatasetSourceIdentity, SFTDatasetIdentity
from stateset_agents.training import industry, sft
from stateset_agents.training.sft_artifacts import verify_sft_adapter
from tests.sft_fixtures import write_sft_export
from tests.unit.test_industry_finetuning import prepared as prepared
from tests.unit.test_sft_dataset_identity import forbid_ml_import as forbid_ml_import
from tests.unit.test_sft_dataset_identity import tiny_cpu_sft as tiny_cpu_sft
from tests.unit.test_sft_dataset_identity import train_options


@pytest.fixture
def exported(tmp_path):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "request"},
                {"role": "assistant", "content": "response"},
            ]
        }
    ]
    data = (json.dumps(rows[0]) + "\n").encode()
    source = DatasetSourceIdentity.from_bytes(tmp_path / "data.jsonl", data)
    identity = SFTDatasetIdentity.from_rows(rows, source=source)
    output = tmp_path / "adapter"
    write_sft_export(
        output, base_model="fixture/model", lora_r=2, dataset_identity=identity
    )
    return output, identity


def verify(output, identity, **kwargs):
    return verify_sft_adapter(
        output,
        base_model="fixture/model",
        lora_r=2,
        dataset_identity=identity,
        **kwargs,
    )


def test_complete_export_is_structurally_verified_without_quality_claim(
    exported, forbid_ml_import
):
    output, identity = exported
    summary = verify(output, identity)
    assert summary["tensor_count"] == 2 and summary["lora_pairs"] == 1
    assert summary["weights"] == ["adapter_model.safetensors"]
    assert summary["quality_gain_verified"] is False


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "corrupt",
        "config",
        "tokenizer",
        "lineage",
        "source",
        "content",
        "boolean_count",
        "rank",
    ],
)
def test_incomplete_or_mislabeled_export_is_rejected(exported, kind):
    output, identity = exported
    if kind in {"empty", "corrupt"}:
        (output / "adapter_model.safetensors").write_bytes(
            b"" if kind == "empty" else b"not tensor data"
        )
    elif kind in {"config", "tokenizer", "lineage"}:
        name = {
            "config": "adapter_config.json",
            "tokenizer": "tokenizer_config.json",
            "lineage": "stateset_manifest.json",
        }[kind]
        (output / name).unlink()
    elif kind == "rank":
        (output / "adapter_config.json").write_text('{"peft_type":"LORA","r":true}')
    else:
        path = output / "stateset_manifest.json"
        manifest = json.loads(path.read_text())
        if kind == "source":
            manifest["dataset_sha256"] = "0" * 64
        elif kind == "content":
            manifest["hyperparameters"]["training_content_sha256"] = "0" * 64
        else:
            manifest["hyperparameters"]["effective_training_rows"] = True
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        verify(output, identity)


@pytest.mark.parametrize("kind", ["weights", "config", "lineage", "fifo"])
def test_special_or_linked_artifacts_are_rejected_without_following(
    exported, tmp_path, kind
):
    output, identity = exported
    name = {
        "weights": "adapter_model.safetensors",
        "config": "adapter_config.json",
        "lineage": "stateset_manifest.json",
        "fifo": "tokenizer_config.json",
    }[kind]
    path = output / name
    outside = tmp_path / "outside"
    outside.write_bytes(path.read_bytes())
    path.unlink()
    if kind == "fifo":
        import os

        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO check requires POSIX")
        os.mkfifo(path)
    else:
        path.symlink_to(outside)
    with pytest.raises(ValueError):
        verify(output, identity)


@pytest.mark.parametrize("kind", ["no_pairs", "missing_b", "wrong_shape", "wrong_rank"])
def test_tensor_headers_require_complete_compatible_lora_pairs(exported, kind):
    import numpy as np
    from safetensors.numpy import load_file, save_file

    output, identity = exported
    path = output / "adapter_model.safetensors"
    tensors = load_file(path)
    if kind == "no_pairs":
        tensors = {"unrelated.weight": np.ones((2, 2), dtype=np.float32)}
    else:
        name = next(name for name in tensors if "lora_B" in name)
        if kind == "missing_b":
            del tensors[name]
        elif kind == "wrong_rank":
            tensors[name] = np.ones((2, 3), dtype=np.float32)
            other = next(name for name in tensors if "lora_A" in name)
            tensors[other] = np.ones((3, 2), dtype=np.float32)
        else:
            tensors[name] = np.ones((2, 3), dtype=np.float32)
    save_file(tensors, str(path))
    with pytest.raises(ValueError, match="LoRA tensor pairs"):
        verify(output, identity)


def test_sharded_export_checks_all_declared_tensors(exported):
    from safetensors.numpy import load_file, save_file

    output, identity = exported
    single = output / "adapter_model.safetensors"
    tensors = load_file(single)
    single.unlink()
    mapping = {}
    for number, (name, value) in enumerate(tensors.items()):
        shard = f"adapter-{number}.safetensors"
        save_file({name: value}, str(output / shard))
        mapping[name] = shard
    index = output / "adapter_model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": mapping}))
    assert len(verify(output, identity)["weights"]) == 2
    mapping[next(iter(mapping))] = "../outside.safetensors"
    index.write_text(json.dumps({"weight_map": mapping}))
    with pytest.raises(ValueError, match="shard reference"):
        verify(output, identity)


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_mixed_precision_headers_do_not_require_materializing_tensors(exported, dtype):
    import torch
    from safetensors.torch import save_file

    output, identity = exported
    save_file(
        {
            "base_model.model.fixture.lora_A.weight": torch.ones(
                (2, 2), dtype=getattr(torch, dtype)
            ),
            "base_model.model.fixture.lora_B.weight": torch.ones(
                (2, 2), dtype=getattr(torch, dtype)
            ),
        },
        str(output / "adapter_model.safetensors"),
    )
    assert verify(output, identity)["lora_pairs"] == 1


def test_stale_lineage_cannot_mask_failed_persistence(exported):
    output, identity = exported
    expected = json.loads((output / "stateset_manifest.json").read_text())
    expected["created_at"] = "new run"
    with pytest.raises(ValueError, match="not persisted"):
        verify(output, identity, expected_lineage=expected)


@pytest.mark.parametrize("kind", ["noop", "empty", "corrupt", "lineage"])
def test_industry_cannot_report_trained_after_incomplete_export(
    prepared, monkeypatch, kind
):
    def train(**kwargs):
        output = kwargs["output_dir"]
        if kind == "noop":
            return output
        write_sft_export(
            output,
            **{key: value for key, value in kwargs.items() if key != "output_dir"},
        )
        if kind == "lineage":
            (output / "stateset_manifest.json").unlink()
        else:
            (output / "adapter_model.safetensors").write_bytes(
                b"" if kind == "empty" else b"broken"
            )
        return output

    monkeypatch.setattr(sft, "gpu_available", lambda: True)
    monkeypatch.setattr(sft, "run_sft", train)
    with pytest.raises(ValueError):
        industry.train_industry_project(prepared, dry_run=False)
    run = json.loads((prepared / "adapter/industry_run.json").read_text())
    assert run["status"] == "failed"
    assert run["validation_status"] == "held_out_not_evaluated"


def test_packaged_job_rejects_noop_trainer(tmp_path, monkeypatch):
    path = tmp_path / "data.jsonl"
    path.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "request"},
                    {"role": "assistant", "content": "response"},
                ]
            }
        )
        + "\n"
    )
    monkeypatch.setattr(sft, "gpu_available", lambda: True)
    monkeypatch.setattr(sft, "run_sft", lambda **kwargs: kwargs["output_dir"])
    outcome = sft.run_sft_job(
        {**train_options(tmp_path / "empty"), "dataset": str(path), "dry_run": False}
    )
    assert outcome["returncode"] == 1


def test_real_cpu_sft_does_not_succeed_without_persisted_lineage(
    tmp_path, tiny_cpu_sft, monkeypatch
):
    rows = [
        {
            "messages": [
                {"role": "user", "content": "request"},
                {"role": "assistant", "content": "response"},
            ]
        }
    ]
    output = tmp_path / "adapter"
    monkeypatch.setattr(sft, "write_manifest", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="metadata"):
        sft.run_sft(rows=rows, **train_options(output))
    assert (output / "adapter_model.safetensors").is_file()
