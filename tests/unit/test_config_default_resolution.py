"""Trainer reads settle on the dataclass defaults; dead fields are rejected.

Each inline ``getattr(config, name, default)`` restated a default that had
drifted from ``TrainingConfig``'s, so a legacy duck-typed config trained
differently from a typed one:

* ``entropy_coef`` -- 0.0 inline vs 0.01 in the dataclass (the GRPO losses);
* ``generation_batch_size`` -- 0 ("one chunk") vs 4 (old-log-prob chunks);
* ``max_completion_length`` -- 0/None vs 512 (the objective's normaliser);
* the TRL bridge sized its generation batch from ``gradient_accumulation_
  steps`` 1 and ``num_generations`` 1 while handing TRL the config's own
  values (TrainingConfig: 4 and 16).

The dataclass is the source of truth (``config_value``); the TRL bridge reads
the same attributes it passes to TRL. Dead algorithm-config fields are
rejected when set to a non-default value, with a dated deprecation, and
``validate()``/``validate_for_training()`` agree on bf16 + fp16.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from stateset_agents.core.errors import ConfigurationError
from stateset_agents.training.config import TrainingConfig
from stateset_agents.training.dapo_config import DAPOConfig
from stateset_agents.training.vapo_config import VAPOConfig

DEFAULTS = TrainingConfig()


def test_grpo_losses_use_the_dataclass_entropy_coef() -> None:
    from stateset_agents.training.loss_computation import _token_objective_settings

    _objective, entropy = _token_objective_settings(SimpleNamespace())
    assert entropy == DEFAULTS.entropy_coef == 0.01


def test_objective_normaliser_uses_the_dataclass_completion_length() -> None:
    from stateset_agents.training.loss_computation import (
        _grpo_objective,
        _token_objective,
    )

    for build in (_grpo_objective, _token_objective):
        assert build(SimpleNamespace()).max_completion_length == 512


def test_old_logprob_chunks_use_the_dataclass_generation_batch_size() -> None:
    from stateset_agents.training.loss_computation import _old_logprob_chunk

    assert _old_logprob_chunk(SimpleNamespace(), largest_group=9) == 4
    assert _old_logprob_chunk(SimpleNamespace(generation_batch_size=2), 9) == 2
    # An explicit 0 still means one chunk per group.
    assert _old_logprob_chunk(SimpleNamespace(generation_batch_size=0), 9) == 9


def test_trl_generation_batch_reads_what_trl_is_given() -> None:
    pytest.importorskip("torch")
    from stateset_agents.training.trl_grpo_config import TRLGRPOConfig
    from stateset_agents.training.trl_grpo_trainer import _trl_generation_batch_kwargs

    config = TRLGRPOConfig(
        per_device_train_batch_size=3, gradient_accumulation_steps=1, num_generations=4
    )
    assert _trl_generation_batch_kwargs(config) == {"generation_batch_size": 12}
    # A legacy config without the fields TRL is given fails here, as building
    # TRL's arguments would, instead of sizing the batch from invented values.
    with pytest.raises(AttributeError, match="gradient_accumulation_steps"):
        _trl_generation_batch_kwargs(SimpleNamespace(per_device_train_batch_size=1))


DEAD: list[tuple[Any, str, Any, str]] = [
    (DAPOConfig, "prompt_batch_size", 64, "per_device_train_batch_size"),
    (DAPOConfig, "eval_repeats", 4, "nothing"),
    (VAPOConfig, "num_prompts_per_batch", 64, "per_device_train_batch_size"),
    (VAPOConfig, "num_iterations", 2, "num_gradient_updates"),
]


def _gepo(**values: Any) -> Any:
    from stateset_agents.training.gepo_trainer import GEPOConfig

    return GEPOConfig(**values)


@pytest.mark.parametrize(
    "factory,field,value,replacement",
    [*DEAD, (_gepo, "use_group_baseline", False, "nothing")],
    ids=lambda v: getattr(v, "__name__", str(v)),
)
def test_dead_fields_are_rejected_with_a_dated_deprecation(
    factory: Any, field: str, value: Any, replacement: str
) -> None:
    factory().validate_for_training()  # the default is accepted
    with pytest.raises(ConfigurationError, match=field) as caught:
        factory(**{field: value}).validate_for_training()
    message = str(caught.value)
    assert "removed in v0.71.0" in message
    if replacement != "nothing":
        assert replacement in message


def test_bf16_and_fp16_validators_agree() -> None:
    config = TrainingConfig(bf16=True, fp16=True)
    warnings = [w for w in config.validate() if "fp16" in w]
    assert warnings and "mutually exclusive" in warnings[0]
    assert "precedence" not in warnings[0]
    with pytest.raises(ConfigurationError, match="mutually exclusive"):
        config.validate_for_training()


@pytest.mark.parametrize("bits", [4, 8])
def test_trl_bridge_loads_quantized_models_with_a_quantization_config(
    monkeypatch: pytest.MonkeyPatch, bits: int
) -> None:
    """Transformers 5 removed ``from_pretrained(load_in_8bit/4bit=...)``; the
    TRL bridge still passed them. It now uses training.quantization like SFT
    and GSPO."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from stateset_agents.training import trl_grpo_trainer as trl
    from stateset_agents.training.trl_grpo_config import TRLGRPOConfig

    loaded: list[dict[str, Any]] = []

    class Model:
        config = SimpleNamespace(use_cache=True)

        @classmethod
        def from_pretrained(cls, name: str, **kwargs: Any) -> Any:
            loaded.append(kwargs)
            return cls()

    tokenizer = SimpleNamespace(pad_token="<p>", eos_token="</s>")
    monkeypatch.setattr(trl, "_require_transformers", lambda: None)
    monkeypatch.setattr(trl, "_require_bitsandbytes", lambda: None)
    monkeypatch.setattr(trl, "_require_kbit_training_support", lambda: None)
    monkeypatch.setattr(trl, "prepare_model_for_kbit_training", lambda model: model)
    monkeypatch.setattr(trl, "AutoModelForCausalLM", Model)
    monkeypatch.setattr(
        trl,
        "AutoTokenizer",
        SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer),
    )
    monkeypatch.setattr(transformers, "BitsAndBytesConfig", lambda **kw: kw)
    config = TRLGRPOConfig(
        model_name="tiny",
        use_lora=False,
        gradient_checkpointing=False,
        use_4bit=bits == 4,
        use_8bit=bits == 8,
        bf16=True,
        fp16=False,
    )
    trl.ModelManager(config).load_model_and_tokenizer()
    [kwargs] = loaded
    assert "load_in_4bit" not in kwargs and "load_in_8bit" not in kwargs
    quant = kwargs["quantization_config"]
    assert quant[f"load_in_{bits}bit"] is True
    if bits == 4:
        assert quant["bnb_4bit_compute_dtype"] == torch.bfloat16
