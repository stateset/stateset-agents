"""Optional dependencies of the dedicated trainers, imported on first use.

DAPO, GEPO and VAPO each carried a copy of the same lazy loaders
(``_load_transformers_<algo>``, ``_require_peft``, ``_require_wandb``) and
the copies drifted. They now share these three:

* :func:`transformers_components` -- ``(AutoTokenizer, AutoModelForCausalLM)``;
* :func:`peft_components` -- ``(LoraConfig, TaskType, get_peft_model)``;
* :func:`require_wandb` -- the ``wandb`` module (W&B reporting goes through
  :class:`~stateset_agents.training.dedicated_reporting.WandbRun`).

Each raises ``ImportError`` naming the algorithm and the extra to install.
The resolved objects are cached in this module's globals, so tests replace
them with ``monkeypatch.setattr(dedicated_deps, "AutoTokenizer", fake)``;
``wandb`` is looked up through ``sys.modules`` on every call, so a test
installs a fake with ``monkeypatch.setitem(sys.modules, "wandb", fake)``.

Importing this module imports none of them.
"""

from __future__ import annotations

import importlib
from typing import Any

#: ``transformers`` classes, resolved by :func:`transformers_components`.
AutoTokenizer: Any | None = None
AutoModelForCausalLM: Any | None = None
#: ``peft`` exports, resolved by :func:`peft_components`.
LoraConfig: Any | None = None
TaskType: Any | None = None
get_peft_model: Any | None = None

_INSTALL_HINT = (
    "Install with `pip install stateset-agents[training]` or `pip install {}`."
)


def transformers_components(algorithm: str) -> tuple[Any, Any]:
    """``(AutoTokenizer, AutoModelForCausalLM)``, importing transformers once."""
    global AutoTokenizer, AutoModelForCausalLM
    if AutoTokenizer is None or AutoModelForCausalLM is None:
        try:
            transformers = importlib.import_module("transformers")
            tokenizer_cls = transformers.AutoTokenizer
            model_cls = transformers.AutoModelForCausalLM
        except (ImportError, RuntimeError, AttributeError) as error:
            raise ImportError(
                f"transformers is required for {algorithm} training. "
                + _INSTALL_HINT.format("transformers")
            ) from error
        # Fill only the empty slots: a test may have replaced one of them.
        if AutoTokenizer is None:
            AutoTokenizer = tokenizer_cls
        if AutoModelForCausalLM is None:
            AutoModelForCausalLM = model_cls
    return AutoTokenizer, AutoModelForCausalLM


def peft_components(algorithm: str) -> tuple[Any, Any, Any]:
    """``(LoraConfig, TaskType, get_peft_model)``, importing PEFT once (only
    LoRA model loading needs it)."""
    global LoraConfig, TaskType, get_peft_model
    if LoraConfig is None or TaskType is None or get_peft_model is None:
        try:
            peft = importlib.import_module("peft")
            components = (peft.LoraConfig, peft.TaskType, peft.get_peft_model)
        except (ImportError, RuntimeError, AttributeError) as error:
            raise ImportError(
                f"PEFT is required for {algorithm} LoRA training. "
                + _INSTALL_HINT.format("peft")
            ) from error
        # Fill only the empty slots: a test may have replaced one of them.
        if LoraConfig is None:
            LoraConfig = components[0]
        if TaskType is None:
            TaskType = components[1]
        if get_peft_model is None:
            get_peft_model = components[2]
    return LoraConfig, TaskType, get_peft_model


def require_wandb(algorithm: str) -> Any:
    """The ``wandb`` module (only when experiment logging is enabled)."""
    try:
        return importlib.import_module("wandb")
    except ImportError as error:
        raise ImportError(
            f"wandb is required for {algorithm} logging. "
            + _INSTALL_HINT.format("wandb")
        ) from error


__all__ = [
    "peft_components",
    "require_wandb",
    "transformers_components",
]
