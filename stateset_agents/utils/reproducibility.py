"""
Central reproducibility controls for StateSet Agents.

This module provides a single entry point — ``set_all_seeds`` — that seeds
every source of randomness the framework can touch: Python's ``random``,
NumPy, PyTorch (CPU and CUDA), and Hugging Face Transformers.

Use this at the top of every benchmark, example, and notebook that needs
deterministic-up-to-floating-point behavior. The framework's published
empirical results all call this before any other framework code.

Example:

    from stateset_agents.utils.reproducibility import set_all_seeds

    set_all_seeds(42)  # call once, before anything else

    # ... rest of script ...

The function records the seed in ``REPRODUCIBILITY_STATE`` so other modules
(e.g., W&B loggers) can pick it up without re-importing.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np


@dataclass
class ReproducibilityState:
    """Tracks the seeds applied in the current process."""

    seed: int | None = None
    deterministic_cuda: bool = False
    components_seeded: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": self.seed,
            "deterministic_cuda": self.deterministic_cuda,
            "components_seeded": list(self.components_seeded),
        }


REPRODUCIBILITY_STATE = ReproducibilityState()


def set_all_seeds(
    seed: int = 42, deterministic_cuda: bool = False
) -> ReproducibilityState:
    """Seed every RNG the framework can reach.

    Args:
        seed: Integer seed. The framework's canonical published-results seed is 42.
        deterministic_cuda: If True, configure CUDA for fully deterministic kernels.
            This is slower and not always available — only enable when reproducing
            published numbers, not for production training.

    Returns:
        The populated ``ReproducibilityState`` (also accessible via the module-level
        ``REPRODUCIBILITY_STATE``).
    """
    REPRODUCIBILITY_STATE.seed = seed
    REPRODUCIBILITY_STATE.deterministic_cuda = deterministic_cuda
    REPRODUCIBILITY_STATE.components_seeded = []

    random.seed(seed)
    REPRODUCIBILITY_STATE.components_seeded.append("random")

    np.random.seed(seed)
    REPRODUCIBILITY_STATE.components_seeded.append("numpy")

    os.environ["PYTHONHASHSEED"] = str(seed)
    REPRODUCIBILITY_STATE.components_seeded.append("pythonhashseed")

    try:
        import torch

        torch.manual_seed(seed)
        REPRODUCIBILITY_STATE.components_seeded.append("torch")

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            REPRODUCIBILITY_STATE.components_seeded.append("torch.cuda")

            if deterministic_cuda:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
                torch.use_deterministic_algorithms(True, warn_only=True)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
                REPRODUCIBILITY_STATE.components_seeded.append("cuda_deterministic")
    except ImportError:
        pass

    try:
        from transformers import set_seed as hf_set_seed

        hf_set_seed(seed)
        REPRODUCIBILITY_STATE.components_seeded.append("transformers")
    except ImportError:
        pass

    return REPRODUCIBILITY_STATE


def get_seed() -> int | None:
    """Return the seed last applied via ``set_all_seeds``, or None if unset."""
    return REPRODUCIBILITY_STATE.seed


def capture_rng(torch: Any) -> dict[str, Any]:
    """Capture generators using only tensors and weights-only-safe Python data.

    Python, NumPy, CPU Torch and every available CUDA generator. ``torch`` is
    passed in so this module never imports it. Matches
    ``training.runtime_state.capture_rng`` (pinned by
    ``tests/unit/test_layering_bridges.py``); evaluation isolation uses this
    copy so ``evaluation`` never imports ``training``.
    """
    numpy_state = cast(tuple[Any, ...], np.random.get_state())
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state: dict[str, Any], torch: Any) -> None:
    """Restore RNGs captured by :func:`capture_rng`.

    Raises ``ValueError`` when the CUDA generator topology differs from the
    one captured.
    """
    random.setstate(state["python"])
    values = state["numpy"]
    np.random.set_state(
        (values[0], np.asarray(values[1], dtype=np.uint32), *values[2:])
    )
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if (
            not torch.cuda.is_available()
            or len(state["cuda"]) != torch.cuda.device_count()
        ):
            raise ValueError("Checkpoint CUDA RNG topology differs from this runtime")
        # A CUDA map_location also moves serialized generator states onto the
        # GPU. PyTorch requires CPU ByteTensors even for CUDA generators.
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


__all__ = [
    "REPRODUCIBILITY_STATE",
    "ReproducibilityState",
    "capture_rng",
    "get_seed",
    "restore_rng",
    "set_all_seeds",
]
