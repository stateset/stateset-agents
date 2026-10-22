"""Local SFT binds parsed rows to source bytes before model loading."""

from __future__ import annotations

import builtins
import copy
import hashlib
import json

import pytest

from stateset_agents import finetune_worker as worker
from stateset_agents import finetune_workspace as workspace
from stateset_agents.data.source import SFTDatasetIdentity
from stateset_agents.training import industry, sft


def chat(prompt="request", answer="response", **extra):
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": answer},
        ],
        **extra,
    }


@pytest.fixture
def dataset(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_bytes((json.dumps(chat()) + "\r\n\r\n").encode())
    return path


def train_options(output):
    return {
        "base_model": "local/random-tiny",
        "output_dir": output,
        "num_epochs": 1,
        "lora_r": 2,
        "lora_alpha": 4,
        "learning_rate": 1e-4,
        "max_length": 64,
        "per_device_batch_size": 1,
        "gradient_accumulation_steps": 1,
    }


@pytest.fixture
def forbid_ml_import(monkeypatch):
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".")[0] in {"datasets", "peft", "transformers", "torch"}:
            pytest.fail("ML stack reached before dataset identity validation")
        return original(name, *args, **kwargs)

    with monkeypatch.context() as scoped:
        scoped.setattr(builtins, "__import__", guarded)
        yield


def test_file_loader_binds_raw_bytes_and_normalized_rows(dataset):
    raw = dataset.read_bytes()
    rows, identity = sft.load_sft_dataset(dataset)
    assert rows == [chat()]
    assert identity.source.sha256 == hashlib.sha256(raw).hexdigest()
    assert identity.source.rows == identity.examples == 1
    identity.verify_rows(rows)


@pytest.mark.parametrize("kind", ["answer", "tool_argument", "append"])
def test_changed_rows_cannot_reuse_a_captured_identity(
    tmp_path, dataset, forbid_ml_import, kind
):
    if kind == "tool_argument":
        row = chat()
        row["messages"][-1]["tool_calls"] = [{"function": {"arguments": {"count": 1}}}]
        dataset.write_text(json.dumps(row) + "\n")
    rows, identity = sft.load_sft_dataset(dataset)
    if kind == "append":
        rows.append(chat("other request"))
    elif kind == "tool_argument":
        rows[0]["messages"][-1]["tool_calls"][0]["function"]["arguments"][
            "count"
        ] = True
    else:
        rows[0]["messages"][-1]["content"] = "PRIVATE CHANGED ANSWER"
    with pytest.raises(ValueError, match="differ from the captured SFT input") as error:
        sft.run_sft(
            rows=rows,
            dataset_path=dataset,
            dataset_identity=identity,
            **train_options(tmp_path / "out"),
        )
    assert "PRIVATE CHANGED ANSWER" not in str(error.value)
    assert not (tmp_path / "out").exists()


def test_direct_file_call_checks_rows_before_loading_a_model(
    dataset, tmp_path, forbid_ml_import
):
    with pytest.raises(ValueError, match="differ from the captured SFT input"):
        sft.run_sft(
            rows=[chat(answer="different")],
            dataset_path=dataset,
            **train_options(tmp_path / "out"),
        )


def test_captured_identity_cannot_label_a_different_file(
    dataset, tmp_path, forbid_ml_import
):
    rows, identity = sft.load_sft_dataset(dataset)
    with pytest.raises(ValueError, match="path differs"):
        sft.run_sft(
            rows=rows,
            dataset_identity=identity,
            dataset_path=tmp_path / "other",
            **train_options(tmp_path / "out"),
        )


def test_studio_rejects_changed_reviewed_data_before_ml_loading(
    dataset, tmp_path, forbid_ml_import
):
    config = workspace.FinetuneConfig(dataset=str(dataset))
    run = workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))
    (run / "dataset").write_text(json.dumps(chat(answer="changed after review")) + "\n")
    with pytest.raises(ValueError, match="changed from the reviewed source"):
        worker._train_sft(run, config)
    assert not (run / "artifacts").exists()


def test_industry_rechecks_manifest_before_gpu_or_output_setup(
    tmp_path, monkeypatch, forbid_ml_import
):
    source = industry.init_industry_project("retail", tmp_path / "source")
    project = tmp_path / "prepared"
    industry.prepare_industry_training("retail", source / "examples.jsonl", project)
    validate = industry.load_industry_project

    def change_after_validation(path):
        result = validate(path)
        (project / "train.jsonl").write_text(json.dumps(chat(answer="changed")) + "\n")
        return result

    monkeypatch.setattr(industry, "load_industry_project", change_after_validation)
    monkeypatch.setattr(
        sft, "gpu_available", lambda: pytest.fail("GPU detection reached")
    )
    with pytest.raises(ValueError, match="changed from the reviewed source"):
        industry.train_industry_project(project, dry_run=False)
    assert not (project / "adapter").exists()


def test_packaged_job_passes_identity_from_before_gpu_detection(
    dataset, tmp_path, monkeypatch
):
    original = dataset.read_bytes()
    received = {}

    def gpu():
        dataset.unlink()
        return True

    monkeypatch.setattr(sft, "gpu_available", gpu)

    def train(**kwargs):
        from tests.sft_fixtures import write_sft_export

        received.update(kwargs)
        write_sft_export(
            kwargs["output_dir"],
            **{key: value for key, value in kwargs.items() if key != "output_dir"},
        )

    monkeypatch.setattr(sft, "run_sft", train)
    result = sft.run_sft_job(
        {
            **train_options(tmp_path / "out"),
            "dataset": str(dataset),
            "dry_run": False,
        }
    )
    assert result["returncode"] == 0
    assert (
        received["dataset_identity"].source.sha256
        == hashlib.sha256(original).hexdigest()
    )
    received["dataset_identity"].verify_rows(received["rows"])


def test_studio_passes_normalized_rows_with_original_source_identity(
    dataset, tmp_path, monkeypatch
):
    dataset.write_text(json.dumps(chat(id="source-only-id")) + "\n")
    config = workspace.FinetuneConfig(dataset=str(dataset))
    run = workspace.create_run(tmp_path / "workspace", workspace.plan_run(config))
    captured = {}
    monkeypatch.setattr(sft, "run_sft", lambda **kwargs: captured.update(kwargs))
    worker._train_sft(run, config)
    assert captured["rows"] == [chat()]
    assert (
        captured["dataset_identity"].source.sha256
        == hashlib.sha256((run / "dataset").read_bytes()).hexdigest()
    )
    captured["dataset_identity"].verify_rows(captured["rows"])


@pytest.fixture
def tiny_cpu_sft(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("datasets")
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    backend = Tokenizer(
        models.WordLevel(
            {
                "[UNK]": 0,
                "[PAD]": 1,
                "[EOS]": 2,
                "user": 3,
                "assistant": 4,
                "request": 5,
                "response": 6,
            },
            unk_token="[UNK]",
        )
    )
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        unk_token="[UNK]",
        pad_token="[PAD]",
        eos_token="[EOS]",
    )
    tokenizer.chat_template = (
        "{% for m in messages %}{{ m['role'] + ' ' + m['content'] + ' ' }}{% endfor %}"
    )
    torch.manual_seed(42)
    template = GPT2LMHeadModel(
        GPT2Config(
            vocab_size=7,
            n_positions=64,
            n_embd=8,
            n_layer=1,
            n_head=1,
            bos_token_id=2,
            eos_token_id=2,
            pad_token_id=1,
        )
    )
    monkeypatch.setattr(
        sft, "load_base_model_for_sft", lambda *args, **kwargs: copy.deepcopy(template)
    )
    monkeypatch.setattr(
        "transformers.AutoTokenizer.from_pretrained",
        lambda *args, **kwargs: copy.deepcopy(tokenizer),
    )
    build = sft.build_training_arguments

    def cpu_arguments(cls, **kwargs):
        return build(
            cls, **{**kwargs, "bf16": False, "use_cpu": True, "disable_tqdm": True}
        )

    monkeypatch.setattr(sft, "build_training_arguments", cpu_arguments)
    try:
        yield
    finally:
        torch.set_num_threads(old_threads)


@pytest.mark.parametrize("change", ["replace", "delete"])
@pytest.mark.parametrize("identity_mode", ["captured", "direct_file"])
def test_real_cpu_lora_retains_captured_source_after_training_side_effects(
    dataset, tmp_path, tiny_cpu_sft, change, identity_mode
):
    from transformers import TrainerCallback

    rows, identity = sft.load_sft_dataset(dataset)
    output = tmp_path / "adapter"

    class ChangeSource(TrainerCallback):
        def on_train_begin(self, args, state, control, **kwargs):
            if change == "delete":
                dataset.unlink()
            else:
                dataset.write_text(json.dumps(chat(answer="different")) + "\n")

    sft.run_sft(
        rows=rows,
        dataset_path=dataset,
        dataset_identity=identity if identity_mode == "captured" else None,
        callbacks=[ChangeSource()],
        **train_options(output),
    )
    manifest = json.loads((output / "stateset_manifest.json").read_text())
    assert (output / "adapter_model.safetensors").is_file()
    assert manifest["dataset_sha256"] == identity.source.sha256
    assert manifest["dataset_path"] == str(dataset)
    assert manifest["dataset_rows"] == 1
    assert (
        manifest["hyperparameters"]["training_content_sha256"]
        == identity.content_sha256
    )
    assert manifest["hyperparameters"]["training_input_rows"] == 1
    assert manifest["hyperparameters"]["effective_training_rows"] == 1
    assert manifest["hyperparameters"]["final_turn_only_rows"] == 0
    assert manifest["hyperparameters"]["unmaskable_rows_dropped"] == 0
    assert manifest["hyperparameters"]["truncated_rows_dropped"] == 0
    assert manifest["hyperparameters"]["partially_truncated_rows"] == 0
    assert manifest["eval_judge_gates_skipped"] is None


def test_real_cpu_sft_records_rows_supervised_on_their_final_turn_only(
    tmp_path, tiny_cpu_sft, monkeypatch
):
    """A template that rewrites history used to train on every token of the
    row with no record of it; the manifest now counts the fallback rows."""
    rows = [chat(), chat(answer="response response")]
    rewritten = rows[1]["messages"]
    real_mask = sft.assistant_token_mask

    def rewrites_second_row(tokenizer, messages, **kwargs):
        if messages == rewritten:
            return None
        return real_mask(tokenizer, messages, **kwargs)

    monkeypatch.setattr(sft, "assistant_token_mask", rewrites_second_row)
    output = tmp_path / "adapter"
    sft.run_sft(rows=rows, **train_options(output))
    hyper = json.loads((output / "stateset_manifest.json").read_text())[
        "hyperparameters"
    ]
    assert hyper["effective_training_rows"] == 2
    assert hyper["final_turn_only_rows"] == 1
    assert hyper["unmaskable_rows_dropped"] == 0


def test_real_cpu_sft_manifest_counts_skipped_judge_gates(
    tmp_path, tiny_cpu_sft, monkeypatch
):
    def unavailable(name):
        raise ImportError("no reward stack on this worker")

    monkeypatch.setattr(sft, "_create_domain_reward", unavailable)
    output = tmp_path / "adapter"
    sft.run_sft(
        rows=[chat()],
        eval_prompts=[{"prompt": "request", "judge": "cs", "min_judge_score": 0.5}],
        eval_max_new_tokens=2,
        **train_options(output),
    )
    manifest = json.loads((output / "stateset_manifest.json").read_text())
    assert manifest["eval_judge_gates_skipped"] == 1
    (row,) = json.loads((output / "eval_results.json").read_text())
    assert row["judge_unavailable"] is True


def test_real_cpu_in_memory_sft_keeps_source_unbound(tmp_path, tiny_cpu_sft):
    rows = [chat()]
    output = tmp_path / "adapter"
    sft.run_sft(rows=rows, **train_options(output))
    manifest = json.loads((output / "stateset_manifest.json").read_text())
    assert manifest["dataset_sha256"] is None and manifest["dataset_path"] is None
    assert manifest["hyperparameters"]["dataset_provenance"] == "in_memory_unbound"
    assert (
        manifest["hyperparameters"]["training_content_sha256"]
        == SFTDatasetIdentity.from_rows(rows).content_sha256
    )
