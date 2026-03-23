"""
Canonical exception tuples for StateSet Agents.

This module defines shared exception groups used throughout the codebase for
catching expected failures in specific contexts. Import from here instead of
defining ad-hoc tuples in each module.

Usage::

    from stateset_agents.exceptions import IMPORT_EXCEPTIONS, INFERENCE_EXCEPTIONS

    try:
        import optional_lib
    except IMPORT_EXCEPTIONS:
        optional_lib = None
"""

from __future__ import annotations

import asyncio

# ---------------------------------------------------------------------------
# Canonical exception groups
# ---------------------------------------------------------------------------

#: Optional/dynamic imports (ImportError, OSError from missing .so, RuntimeError
#: from version mismatches).
IMPORT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ImportError,
    OSError,
    RuntimeError,
)

#: GPU property detection — attribute lookups on devices that may not exist.
GPU_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    OSError,
    RuntimeError,
)

#: Model I/O: loading weights, saving checkpoints, quantisation setup.
MODEL_IO_EXCEPTIONS: tuple[type[BaseException], ...] = (
    ImportError,
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

#: Inference / forward-pass / generation failures.
INFERENCE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    TypeError,
    ValueError,
)

#: Attribute / value look-ups on tensors, configs, or dicts.
ATTRIBUTE_VALUE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    RuntimeError,
    TypeError,
    ValueError,
)

#: Network calls — includes async timeout.
NETWORK_EXCEPTIONS: tuple[type[BaseException], ...] = (
    OSError,
    RuntimeError,
    asyncio.TimeoutError,
)

#: Serialisation / deserialisation (JSON, pickle, HF parse).
SERIALIZATION_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    KeyError,
    TypeError,
    ValueError,
)

#: Device-placement helpers (model.to(), tensor.device checks).
MODEL_DEVICE_EXCEPTIONS: tuple[type[BaseException], ...] = (
    AttributeError,
    RuntimeError,
    TypeError,
)
