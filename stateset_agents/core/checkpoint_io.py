"""Safe checkpoint loading.

Every ``torch.load`` in this package goes through :func:`load_checkpoint_file`.
It pins ``weights_only=True`` by default so a checkpoint from an untrusted
source cannot execute arbitrary code while being unpickled, and it turns
torch's generic "Weights only load failed" message into a
:class:`~stateset_agents.core.errors.ModelError` that says what to do about it.

``tests/unit/test_checkpoint_trust.py`` asserts that no other module in the
package calls ``torch.load`` directly.

This lives in ``core`` rather than ``training`` because it depends only on
``core.errors``; ``stateset_agents.training.checkpoint_io`` re-exports it for
the training call sites.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

from .errors import ErrorCode, ModelError

__all__ = ["load_checkpoint_file"]


def load_checkpoint_file(
    path: str | Path,
    *,
    map_location: Any = "cpu",
    trusted: bool = False,
    torch_module: Any = None,
) -> Any:
    """Load a torch checkpoint, refusing arbitrary pickled objects by default.

    Args:
        path: Checkpoint file to read.
        map_location: Passed straight through to ``torch.load``.
        trusted: When ``False`` (the default) the file is unpickled with
            ``weights_only=True``, so only tensors and plain data are restored.
            Pass ``True`` only for a checkpoint whose source you control.
        torch_module: Torch module to use; imported lazily when omitted. Callers
            that already hold a torch handle (or inject one in tests) pass it in.

    Raises:
        ModelError: The checkpoint holds pickled objects and ``trusted`` is
            ``False``.
        ImportError: Torch is not installed and no ``torch_module`` was given.
    """
    if torch_module is not None:
        torch = torch_module
    else:
        # Imported here, not at module scope: torch is an optional dependency
        # and importing ``stateset_agents.core`` must not require it.
        try:
            import torch  # type: ignore[no-redef]
        except ImportError as exc:  # pragma: no cover - import guarding
            raise ImportError(
                "PyTorch is required to load checkpoints. "
                "Install the 'training' extra: pip install stateset-agents[training]"
            ) from exc
    try:
        return torch.load(path, map_location=map_location, weights_only=not trusted)
    except pickle.UnpicklingError as exc:
        if trusted:
            raise
        raise ModelError(
            f"checkpoint {path} contains pickled objects (likely a config "
            "dataclass from before the plain-config format); pass trusted=True "
            "only if you trust its source",
            code=ErrorCode.MDL_LOAD_FAILED,
            checkpoint_path=str(path),
        ) from exc
