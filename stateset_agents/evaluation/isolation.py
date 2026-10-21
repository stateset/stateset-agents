"""Reproducible local evaluation without advancing training random streams."""

from __future__ import annotations

import asyncio
import random
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from stateset_agents.core.generation_mode import inference_mode
from stateset_agents.evaluation.agent_runs import content_hash

EVALUATION_RNG_PROTOCOL = "sha256-case-v1"
_ownership_lock = threading.Lock()
_active_owner: tuple[int, int | None] | None = None
_owner_depth = 0


def _validate_seed(seed: int) -> None:
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Evaluation seed must be an integer in [0, 2**64)")


def paired_evaluation_seed(seed: int, case_id: str, case_hash: str) -> int:
    """Derive a stable 64-bit seed from run seed and exact case identity.

    Case ordering and preceding episode lengths do not affect the seed. Matching
    policies use the same starting stream per case, without claiming identical
    random choices after their execution paths diverge.
    """
    _validate_seed(seed)
    if not isinstance(case_id, str) or not case_id.strip():
        raise ValueError("Evaluation case identifier must be nonempty")
    if (
        not isinstance(case_hash, str)
        or len(case_hash) != 64
        or any(char not in "0123456789abcdef" for char in case_hash)
    ):
        raise ValueError("Evaluation case hash must be a SHA-256 digest")
    digest = content_hash(
        {
            "protocol": EVALUATION_RNG_PROTOCOL,
            "seed": seed,
            "case_id": case_id,
            "case_hash": case_hash,
        }
    )
    return int(digest[:16], 16)


@contextmanager
def _exclusive_rng_owner() -> Iterator[None]:
    """Reject overlapping callers before touching process-global generators."""
    global _active_owner, _owner_depth
    try:
        task = asyncio.current_task()
    except RuntimeError:
        task = None
    owner = (threading.get_ident(), id(task) if task is not None else None)
    with _ownership_lock:
        if _active_owner is not None and _active_owner != owner:
            raise RuntimeError(
                "Isolated evaluations must run serially in one task and thread"
            )
        _active_owner = owner
        _owner_depth += 1
    try:
        yield
    finally:
        with _ownership_lock:
            _owner_depth -= 1
            if _owner_depth == 0:
                _active_owner = None


@contextmanager
def isolated_evaluation(*, seed: int, model: Any = None) -> Iterator[None]:
    """Seed local evaluation and restore RNGs, grad mode and all module modes.

    NumPy and Torch are imported only when entering the context. Python, NumPy,
    CPU Torch and every available CUDA generator are restored on every exit,
    including cancellation. Generation runs without gradients, in eval mode,
    with the KV cache enabled when supported.

    These generators are process-global: serialize evaluation with training and
    all other RNG consumers. Overlapping isolated contexts from different tasks
    or threads raise; nesting in the owning task is supported. Private agent or
    provider generators are outside this local context's contract.
    """
    _validate_seed(seed)
    with _exclusive_rng_owner():
        import numpy as np
        import torch

        from stateset_agents.training.runtime_state import capture_rng, restore_rng

        state = capture_rng(torch)
        modes = []
        try:
            modules = (
                model.modules()
                if callable(getattr(model, "modules", None))
                else [model]
            )
            modes = [
                (module, module.training)
                for module in modules
                if hasattr(module, "training")
            ]
            random.seed(seed)
            np.random.seed(seed % 2**32)
            torch.manual_seed(seed)
            with torch.no_grad(), inference_mode(model):
                yield
        finally:
            try:
                for module, training in modes:
                    module.training = training
            finally:
                restore_rng(state, torch)
