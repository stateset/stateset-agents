"""Compatibility helpers for causal and composite generation checkpoints."""

from __future__ import annotations

import importlib
from typing import Any


def generation_compat_kwargs(model: Any) -> dict[str, Any]:
    """Disable the broken hybrid cache for attention-only Granite checkpoints.

    Transformers 5.14 builds a recurrent mask for dense Granite Micro, then
    asks an attention-only cache for Mamba state and raises ValueError. An
    uncached forward avoids that path; hybrid Granite models keep their cache.
    """
    config = getattr(model, "config", None)
    layers = getattr(config, "layer_types", None)
    if (
        getattr(config, "model_type", None) == "granitemoehybrid"
        and isinstance(layers, (list, tuple))
        and layers
        and all(layer in {"attention", "full_attention"} for layer in layers)
    ):
        return {"use_cache": False}
    return {}


def load_generation_model(
    causal_model_cls: Any,
    model_name: str,
    model_kwargs: dict[str, Any],
) -> tuple[Any, Any]:
    """Load a causal LM, falling back to Transformers' multimodal auto classes.

    Native multimodal repositories such as ``zai-org/GLM-5.3-Flash`` expose a
    causal text stack through a composite conditional-generation model.  They
    are intentionally absent from ``AutoModelForCausalLM``'s mapping, even
    though text-only generation and post-training remain valid.

    Returns the loaded model and the auto-model class that accepted it.  The
    latter lets RL trainers load a matching frozen reference model.
    """
    try:
        return (
            causal_model_cls.from_pretrained(model_name, **model_kwargs),
            causal_model_cls,
        )
    except ValueError as causal_exc:
        transformers = importlib.import_module("transformers")
        for class_name in (
            "AutoModelForMultimodalLM",
            "AutoModelForImageTextToText",
        ):
            model_cls = getattr(transformers, class_name, None)
            if model_cls is None:
                continue
            try:
                return model_cls.from_pretrained(model_name, **model_kwargs), model_cls
            except ValueError:
                continue
        raise causal_exc from None


__all__ = ["generation_compat_kwargs", "load_generation_model"]
