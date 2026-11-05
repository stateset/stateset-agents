"""4-bit and 8-bit quantized base-model loading for LoRA (QLoRA), shared by trainers.

QLoRA loads the frozen base model through bitsandbytes (4-bit NF4 with double
quantization and a bf16 compute dtype by default, or 8-bit LLM.int8) and
trains only the LoRA adapter on top. This module holds the pieces every
trainer needs, so the SFT job and the RL trainers build the same
``BitsAndBytesConfig`` and fail the same way:

* :class:`QuantizationConfig` — validated settings (``mode`` none/4bit/8bit);
* :func:`require_bitsandbytes` — a CUDA GPU and an importable ``bitsandbytes``
  are both required; each missing piece raises a message that says what to
  do, including on CPU-only machines where previews keep working;
* :func:`build_bnb_config` — the ``transformers.BitsAndBytesConfig``;
* :func:`quantized_load_kwargs` — ``from_pretrained`` kwargs with the config
  and a device map (k-bit weights cannot be moved after loading);
* :func:`prepare_quantized_model` — ``peft.prepare_model_for_kbit_training``.

Nothing heavy is imported at module import time. The GSPO trainer's original
inline version (``gspo_trainer._prepare_model_kwargs``) builds the same 4-bit
config; :meth:`QuantizationConfig.from_flags` mirrors its ``use_4bit``/
``use_8bit`` flags (8-bit wins when both are set) so it can switch over
without a behavior change.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from stateset_agents.exceptions import IMPORT_EXCEPTIONS

__all__ = [
    "COMPUTE_DTYPES",
    "QUANTIZATION_MODES",
    "QuantizationConfig",
    "build_bnb_config",
    "prepare_quantized_model",
    "quantized_load_kwargs",
    "require_bitsandbytes",
]

QUANTIZATION_MODES = ("none", "4bit", "8bit")
FOUR_BIT_TYPES = ("nf4", "fp4")
COMPUTE_DTYPES = ("bfloat16", "float16", "float32")

INSTALL_HINT = (
    "Install it with `pip install 'stateset-agents[qlora]'` "
    "(or `pip install bitsandbytes`) on a CUDA machine."
)


@dataclass(frozen=True)
class QuantizationConfig:
    """How the frozen base model is quantized while LoRA trains on top."""

    mode: str = "none"
    #: 4-bit storage type: NF4 (QLoRA default) or FP4.
    quant_type: str = "nf4"
    #: Quantize the 4-bit quantization constants too (QLoRA's double quant).
    double_quant: bool = True
    #: Dtype of the matmuls the 4-bit layers dequantize into.
    compute_dtype: str = "bfloat16"

    def __post_init__(self) -> None:
        if self.mode not in QUANTIZATION_MODES:
            raise ValueError(
                f"quantization must be one of {', '.join(QUANTIZATION_MODES)}, "
                f"got {self.mode!r}"
            )
        if self.quant_type not in FOUR_BIT_TYPES:
            raise ValueError(
                f"4-bit quant type must be one of {', '.join(FOUR_BIT_TYPES)}"
            )
        if type(self.double_quant) is not bool:
            raise ValueError("double_quant must be a boolean")
        if self.compute_dtype not in COMPUTE_DTYPES:
            raise ValueError(
                f"compute dtype must be one of {', '.join(COMPUTE_DTYPES)}"
            )

    @property
    def enabled(self) -> bool:
        return self.mode != "none"

    @classmethod
    def from_flags(
        cls,
        *,
        use_4bit: bool = False,
        use_8bit: bool = False,
        compute_dtype: str = "bfloat16",
    ) -> QuantizationConfig:
        """Map trainer ``use_4bit``/``use_8bit`` flags; 8-bit wins if both are set.

        That precedence is what ``GSPOTrainer`` has always done
        (``load_in_4bit=use_4bit and not use_8bit``).
        """
        mode = "8bit" if use_8bit else "4bit" if use_4bit else "none"
        return cls(mode=mode, compute_dtype=compute_dtype)

    @classmethod
    def from_trainer_flags(cls, config: Any, load_dtype: Any) -> QuantizationConfig:
        """:meth:`from_flags` for a trainer config's ``use_4bit``/``use_8bit``,
        computing 4-bit layers in the model's load dtype (a ``torch.dtype``
        such as ``torch.bfloat16``, or its name)."""
        name = str(load_dtype).rsplit(".", 1)[-1]
        return cls.from_flags(
            use_4bit=bool(config.use_4bit),
            use_8bit=bool(config.use_8bit),
            compute_dtype=name if name in COMPUTE_DTYPES else "bfloat16",
        )

    @classmethod
    def coerce(cls, value: Any) -> QuantizationConfig:
        """Accept ``None``, a mode string, a mapping, or a config."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            return cls(mode=value)
        if isinstance(value, dict):
            unknown = set(value) - {
                "mode",
                "quant_type",
                "double_quant",
                "compute_dtype",
            }
            if unknown:
                raise ValueError(f"Unknown quantization settings: {sorted(unknown)}")
            return cls(**value)
        raise ValueError("quantization must be a mode string or a settings object")

    def to_dict(self) -> dict[str, Any]:
        """Settings for manifests; 4-bit details only when they apply."""
        if self.mode == "4bit":
            return {
                "mode": self.mode,
                "quant_type": self.quant_type,
                "double_quant": self.double_quant,
                "compute_dtype": self.compute_dtype,
            }
        return {"mode": self.mode}


def _cuda_available() -> bool:
    try:
        import torch
    except ImportError:
        return False
    return bool(torch.cuda.is_available())


def require_bitsandbytes(cuda_available: Callable[[], bool] | None = None) -> None:
    """Fail early, with the fix in the message, when k-bit loading cannot run.

    bitsandbytes' k-bit kernels need a CUDA GPU, so a CPU-only host gets a
    ``RuntimeError`` that says so (rather than a crash deep in model loading);
    a missing or broken install gets an ``ImportError`` naming the extra.
    """
    if not (cuda_available or _cuda_available)():
        raise RuntimeError(
            "4-bit/8-bit (QLoRA) loading needs a CUDA GPU: bitsandbytes k-bit "
            "kernels do not run on CPU. Train without quantization here, or run "
            "on a CUDA host; dry runs and training plans work without it."
        )
    if importlib.util.find_spec("bitsandbytes") is None:
        raise ImportError(
            "bitsandbytes is required for 4-bit/8-bit (QLoRA) loading. " + INSTALL_HINT
        )
    try:
        import bitsandbytes  # noqa: F401
    except IMPORT_EXCEPTIONS as exc:
        raise ImportError(
            "bitsandbytes is installed but failed to import (CUDA/driver "
            f"mismatch?): {exc}. " + INSTALL_HINT
        ) from exc


def _torch_dtype(name: str) -> Any:
    import torch

    return getattr(torch, name)


def build_bnb_config(
    config: QuantizationConfig,
    *,
    bnb_config_cls: Any = None,
    dtype_resolver: Callable[[str], Any] | None = None,
) -> Any:
    """The ``transformers.BitsAndBytesConfig`` for ``config``.

    ``bnb_config_cls``/``dtype_resolver`` exist so tests can check the exact
    arguments without transformers or torch. Transformers 5 removed the
    legacy ``load_in_*`` ``from_pretrained`` kwargs, so the quantization
    config object is the only supported route.
    """
    if not config.enabled:
        raise ValueError("No quantization requested")
    if bnb_config_cls is None:
        from transformers import BitsAndBytesConfig

        bnb_config_cls = BitsAndBytesConfig
    if config.mode == "8bit":
        return bnb_config_cls(load_in_8bit=True)
    resolve = dtype_resolver or _torch_dtype
    return bnb_config_cls(
        load_in_4bit=True,
        bnb_4bit_quant_type=config.quant_type,
        bnb_4bit_use_double_quant=config.double_quant,
        bnb_4bit_compute_dtype=resolve(config.compute_dtype),
    )


def quantized_load_kwargs(
    config: QuantizationConfig,
    base_kwargs: dict[str, Any],
    *,
    bnb_config_cls: Any = None,
    dtype_resolver: Callable[[str], Any] | None = None,
) -> dict[str, Any]:
    """``from_pretrained`` kwargs that load the base model quantized.

    Unquantized configs return a copy of ``base_kwargs``. Quantized loading
    sets ``quantization_config`` and, unless the caller already chose one, a
    ``device_map`` of ``"auto"``: k-bit weights are placed at load time and
    cannot be moved by the Trainer afterwards. The compute dtype also becomes
    the load dtype of the layers bitsandbytes leaves unquantized.
    """
    kwargs = dict(base_kwargs)
    if not config.enabled:
        return kwargs
    kwargs["quantization_config"] = build_bnb_config(
        config, bnb_config_cls=bnb_config_cls, dtype_resolver=dtype_resolver
    )
    kwargs.setdefault("device_map", "auto")
    if config.mode == "4bit":
        kwargs["torch_dtype"] = config.compute_dtype
    return kwargs


def prepare_quantized_model(
    model: Any,
    *,
    gradient_checkpointing: bool = True,
    prepare: Callable[..., Any] | None = None,
) -> Any:
    """Freeze the k-bit base, upcast norms, and enable input grads for LoRA."""
    if prepare is None:
        from peft import prepare_model_for_kbit_training

        prepare = prepare_model_for_kbit_training
    return prepare(model, use_gradient_checkpointing=gradient_checkpointing)
