"""
Training infrastructure for GRPO Agent Framework

Includes state-of-the-art RL algorithms:
- GRPO: Group Relative Policy Optimization
- GSPO: Group Sequence Policy Optimization
- GEPO: Group Expectation Policy Optimization (best for heterogeneous/distributed)
- DAPO: Decoupled Clip and Dynamic Sampling Policy Optimization (best for reasoning)
- VAPO: Value-Augmented Policy Optimization (SOTA: 60.4 on AIME 2024)

Offline RL algorithms:
- CQL: Conservative Q-Learning
- IQL: Implicit Q-Learning
- BCQ: Batch-Constrained Q-Learning
- BEAR: Bootstrapping Error Accumulation Reduction
- Decision Transformer: Sequence modeling approach

Sim-to-Real Transfer:
- Domain Randomization: Persona, topic, and style randomization
- System Identification: Learn user behavior models
- Progressive Transfer: Gradual sim-to-real transition

Generation backends:
- vLLM: 5-20x faster generation with automatic log probability extraction
- HuggingFace: Standard generation fallback
"""

import importlib
import importlib.util
import logging
from types import ModuleType
from typing import Any

from ._registry import OPTIONAL_EXPORTS as _OPTIONAL_EXPORTS
from ._registry import PUBLIC_NAMES

logger = logging.getLogger(__name__)

# vLLM backend for fast generation - use lazy import to avoid torchvision issues
# Don't import at module level, let users import when needed
VLLM_BACKEND_AVAILABLE = False
VLLM_AVAILABLE = False

try:
    # Just check if vllm_backend module exists without importing it
    spec = importlib.util.find_spec(".vllm_backend", package=__name__)
    if spec is not None:
        VLLM_BACKEND_AVAILABLE = True
except (ImportError, ValueError):
    pass

# Optional trainer/algorithm availability is computed without importing heavy deps.


def _has_spec(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


# Lightweight dependency checks.
#
# ``TRL_AVAILABLE`` is resolved lazily (see ``__getattr__``) because importing
# ``trl`` pulls in ``torch``, and importing this package must stay torch-free.
def _detect_trl() -> bool:  # pragma: no cover - depends on the environment
    try:
        import trl
    except ImportError:
        return False

    # `hasattr` only swallows AttributeError. trl's own lazy-module loader
    # (trl._lazy_module) wraps *any* failure while resolving an attribute
    # -- including a transitively optional, environment-dependent one like
    # an unusable/absent vllm integration -- in a bare RuntimeError, not
    # AttributeError. A plain `hasattr(trl, "GRPOConfig")` would let that
    # RuntimeError escape and crash this call instead of falling back to
    # False as intended.
    try:
        return hasattr(trl, "GRPOConfig")
    except Exception:  # noqa: BLE001 - trl's lazy-attr errors are not typed
        return False


_TORCH_AVAILABLE = _has_spec("torch")
GSPO_AVAILABLE = _TORCH_AVAILABLE
GEPO_AVAILABLE = _TORCH_AVAILABLE
DAPO_AVAILABLE = _TORCH_AVAILABLE
VAPO_AVAILABLE = _TORCH_AVAILABLE
PPO_AVAILABLE = _TORCH_AVAILABLE
KL_CONTROLLERS_AVAILABLE = _TORCH_AVAILABLE
EMA_AVAILABLE = _TORCH_AVAILABLE
RLAIF_AVAILABLE = _TORCH_AVAILABLE

# Offline RL and Sim-to-Real availability
OFFLINE_RL_AVAILABLE = _TORCH_AVAILABLE
BCQ_AVAILABLE = _TORCH_AVAILABLE
BEAR_AVAILABLE = _TORCH_AVAILABLE
DECISION_TRANSFORMER_AVAILABLE = _TORCH_AVAILABLE
SIM_TO_REAL_AVAILABLE = _TORCH_AVAILABLE


AUTO_RESEARCH_AVAILABLE = True


def _maybe_import_submodule(name: str) -> ModuleType | None:
    """Import real submodules like ``stateset_agents.training.trainer`` on demand."""
    module_name = f"{__name__}.{name}"
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        spec = None

    if spec is None:
        return None

    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        # The module exists but its dependencies (usually torch) do not.
        # Report it as absent so ``hasattr`` stays False instead of raising.
        logger.debug(
            "optional training submodule %s is unimportable: %s", module_name, exc
        )
        return None
    globals()[name] = module
    return module


def __getattr__(name: str) -> Any:
    if name == "TRL_AVAILABLE":
        value = _detect_trl()
        globals()[name] = value
        return value
    if name in _OPTIONAL_EXPORTS:
        module_name, attr_name = _OPTIONAL_EXPORTS[name]
        module = importlib.import_module(module_name)
        value = getattr(module, attr_name)
        globals()[name] = value
        return value
    submodule = _maybe_import_submodule(name)
    if submodule is not None:
        return submodule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = list(PUBLIC_NAMES)
