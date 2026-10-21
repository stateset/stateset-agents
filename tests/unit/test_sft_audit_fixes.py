"""Remote SFT honours the job contract: accumulation, shuffling, sharded merges."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.remote.job import JobStatus, RemoteJobSpec
from stateset_agents.remote.river import RiverExecutor, _RiverJob
from stateset_agents.remote.tinker import TinkerExecutor

# --- River: gradient accumulation and LoRA alpha ------------------------------


class _RiverModel:
    def __init__(self, calls: list[tuple], fused: bool) -> None:
        self._calls = calls
        if fused:
            self.train_step = self._train_step

    def _train_step(self, data, lr, loss_fn="cross_entropy"):
        self._calls.append(("train_step", len(data)))
        return {"loss_mean": 0.5}, {"ok": True}

    def forward_backward(self, batch, loss_fn="cross_entropy"):
        self._calls.append(("fb", len(batch)))
        return {"loss": 0.5}

    def optim_step(self, lr):
        self._calls.append(("opt",))

    def save_weights(self, name, mode="inference"):
        return f"river://{name}"


class _RiverClient:
    def __init__(self, fused: bool) -> None:
        self.calls: list[tuple] = []
        self.loras: list[dict[str, Any]] = []
        self._fused = fused

    def LoraConfig(self, **kwargs):  # noqa: N802 - mirrors the SDK name
        self.loras.append(kwargs)
        return kwargs

    def session(self, project=None):
        client = self

        class _Session:
            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def create_model(self, base_model, lora=None):
                return _RiverModel(client.calls, client._fused)

        return _Session()


def _river_train(tmp_path: Path, rows: int, *, fused: bool, **spec_overrides):
    dataset = tmp_path / "rows.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    spec = RemoteJobSpec(
        dataset=dataset,
        base_model="Qwen/Qwen3.5-9B",
        output_dir=tmp_path / "out",
        num_epochs=1,
        per_device_batch_size=2,
        gradient_accumulation_steps=4,
        **spec_overrides,
    )
    client = _RiverClient(fused)
    job = _RiverJob(spec=spec, status=JobStatus.RUNNING)
    data = [{"input_ids": [1, 2]} for _ in range(rows)]
    RiverExecutor(client=client)._train(client, spec, data, job)
    return client, job


def test_river_accumulates_micro_batches_into_one_optimizer_step(tmp_path):
    client, job = _river_train(tmp_path, 8, fused=False)
    assert client.calls == [("fb", 2), ("fb", 2), ("fb", 2), ("fb", 2), ("opt",)]
    assert job.steps == 1


def test_river_flushes_a_partial_accumulation_window_at_epoch_end(tmp_path):
    client, job = _river_train(tmp_path, 6, fused=False)
    assert client.calls == [("fb", 2), ("fb", 2), ("fb", 2), ("opt",)]
    assert job.steps == 1


def test_river_fused_train_step_covers_the_effective_batch(tmp_path):
    client, job = _river_train(tmp_path, 8, fused=True)
    assert client.calls == [("train_step", 8)]
    assert job.steps == 1


def test_river_reports_that_lora_alpha_is_not_applied(tmp_path):
    client, job = _river_train(tmp_path, 2, fused=True, lora_alpha=64)
    assert all("alpha" not in kwargs for kwargs in client.loras)
    assert any("lora_alpha" in line for line in job.logs)


def test_river_manifest_records_gradient_accumulation(tmp_path):
    from stateset_agents.remote.river import _sft_hyperparameters

    dataset = tmp_path / "d.jsonl"
    dataset.write_text("{}\n", encoding="utf-8")
    spec = RemoteJobSpec(
        dataset=dataset,
        base_model="Qwen/Qwen3.5-9B",
        gradient_accumulation_steps=3,
    )
    params = _sft_hyperparameters(spec)
    assert params["gradient_accumulation_steps"] == 3


# --- Tinker: shuffled epochs -------------------------------------------------------


class _Immediate:
    def __init__(self, value=None) -> None:
        self.value = value

    def result(self):
        return self.value


class _Tokenizer:
    def apply_chat_template(self, messages, tokenize=False, **kwargs):
        return "".join(f"<{item['role']}>{item['content']}" for item in messages)

    def encode(self, text, add_special_tokens=False):
        return list(range(1, len(text) + 1))


class _TinkerTraining:
    def __init__(self) -> None:
        self.order: list[int] = []

    def get_tokenizer(self):
        return _Tokenizer()

    def forward_backward(self, batch, loss_fn):
        self.order.extend(len(d.kwargs["model_input"]) for d in batch)
        return _Immediate()

    def optim_step(self, params):
        return _Immediate()

    def save_weights_for_sampler(self, name):
        return _Immediate(SimpleNamespace(path="tinker://sampler"))

    def save_state(self, name):
        return _Immediate(SimpleNamespace(path="tinker://state"))


class _TinkerModule:
    class ModelInput:
        from_ints = staticmethod(lambda values: values)

    class TensorData:
        from_torch = staticmethod(lambda value: value.tolist())

    class Datum:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs

    class AdamParams:
        def __init__(self, learning_rate) -> None:
            self.learning_rate = learning_rate


def _tinker_order(tmp_path: Path, rows: int = 10, epochs: int = 2) -> list[int]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "rows.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                {
                    "messages": [
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a" * (i + 1)},
                    ]
                }
            )
            for i in range(rows)
        )
        + "\n",
        encoding="utf-8",
    )
    training = _TinkerTraining()
    service = SimpleNamespace(create_lora_training_client=lambda **kwargs: training)
    executor = TinkerExecutor(service, tinker_module=_TinkerModule())
    handle = executor.submit(
        RemoteJobSpec(
            dataset=path,
            base_model="thinkingmachines/Inkling-Small",
            output_dir=tmp_path / "out",
            num_epochs=epochs,
            per_device_batch_size=1,
            gradient_accumulation_steps=1,
        )
    )
    assert executor.wait(handle, poll_interval_s=0.001).succeeded
    return training.order


def test_tinker_shuffles_each_epoch_deterministically(tmp_path):
    order = _tinker_order(tmp_path / "a")
    first, second = order[:10], order[10:]
    assert sorted(first) == sorted(second)
    assert first != sorted(first), "epoch 1 replayed file order"
    assert first != second, "every epoch saw the same order"
    assert order == _tinker_order(tmp_path / "b"), "shuffle is not seeded"


# --- merge_adapter: sharded models and pinned revisions ----------------------------


class _ShardedModel:
    hf_device_map = {"layer.0": 0, "layer.1": 1}

    def to(self, device):  # pragma: no cover - must never be called
        raise AssertionError("a sharded model was collapsed onto one device")


def test_merge_keeps_sharded_models_in_place(monkeypatch):
    from stateset_agents.training import merge_adapter as merge

    monkeypatch.setattr(merge, "gpu_available", lambda: True)
    model = _ShardedModel()
    assert merge._place_on_gpu(model) is model


def test_merge_moves_single_device_models_to_gpu(monkeypatch):
    from stateset_agents.training import merge_adapter as merge

    monkeypatch.setattr(merge, "gpu_available", lambda: True)
    moved = SimpleNamespace(hf_device_map=None)
    model = SimpleNamespace(hf_device_map=None, to=lambda device: moved)
    assert merge._place_on_gpu(model) is moved


def test_merge_loads_the_pinned_base_revision(monkeypatch):
    from stateset_agents.training import merge_adapter as merge

    seen: dict[str, Any] = {}

    class _Config:
        architectures = ["Qwen2ForCausalLM"]

    def fake_config(name, **kwargs):
        seen["config"] = kwargs.get("revision")
        return _Config()

    def fake_load(name, *, revision=None):
        seen["model"] = revision
        return "model"

    import transformers

    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", fake_config)
    monkeypatch.setattr(merge, "load_base_model_for_sft", fake_load)
    assert merge._load_full_checkpoint("base", revision="abc123") == "model"
    assert seen == {"config": "abc123", "model": "abc123"}


def test_merge_cli_accepts_a_revision():
    from stateset_agents.training.merge_adapter import build_parser

    args = build_parser().parse_args(
        ["--base-model", "b", "--adapter", "a", "--output-dir", "o", "--revision", "r"]
    )
    assert args.revision == "r"


@pytest.mark.parametrize("revision", [None, "abc"])
def test_merge_adapter_signature_accepts_revision(revision):
    import inspect

    from stateset_agents.training.merge_adapter import merge_adapter

    parameter = inspect.signature(merge_adapter).parameters["revision"]
    assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
    assert parameter.default is None
