"""Deprecated forwarder for the Kimi-K2.5 GSPO finetune script.

``examples/finetune_kimi_k2_5_gspo.py`` and ``examples/finetune_kimi_k25_gspo.py``
were two independently-maintained scripts for the same model
(``moonshotai/Kimi-K2.5``). ``finetune_kimi_k25_gspo.py`` is the newer,
strictly richer script (adds ``--system-prompt``, ``--use-vllm``,
``--export-merged``, ``--iterations``, and more ``--task`` choices) and every
CLI flag this module used to support is still accepted there.

This module now forwards to ``examples/finetune_kimi_k25_gspo.py`` for one
release; it will be removed afterwards. Update any scripts/CI that invoke
``examples/finetune_kimi_k2_5_gspo.py`` directly to call
``examples/finetune_kimi_k25_gspo.py`` instead.
"""

# ruff: noqa: E402

from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = str(Path(__file__).resolve().parents[1])
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.finetune_kimi_k25_gspo import main as _finetune_kimi_k25_main

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    logger.warning(
        "examples/finetune_kimi_k2_5_gspo.py is deprecated and will be "
        "removed in a future release. Use "
        "examples/finetune_kimi_k25_gspo.py instead -- it accepts the same "
        "flags plus --system-prompt, --use-vllm, --export-merged, and "
        "--iterations."
    )
    _finetune_kimi_k25_main()


if __name__ == "__main__":
    main()


__all__ = ["main"]
