"""Compatibility entrypoint for the packaged River refund benchmark.

Run offline: python examples/river_refund_rl.py --dry-run --output outputs/refund
Installed package: python -m stateset_agents.training.river_refund --help
"""

import logging

from stateset_agents.training.river_refund import (
    BENCHMARKS,
    EVALUATION_SETTINGS,
    campaign,
    execute_run,
    main,
    prepare_run,
)

__all__ = [
    "BENCHMARKS",
    "EVALUATION_SETTINGS",
    "campaign",
    "execute_run",
    "main",
    "prepare_run",
]

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
