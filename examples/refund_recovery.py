"""Prepare or audit synthetic recovery fine-tuning data without a provider.

python -m examples.refund_recovery --output /tmp/refund-recovery --seed 42
python -m examples.refund_recovery --output /tmp/refund-recovery --audit
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from stateset_agents.data.refund_recovery import (
    audit_refund_recovery_data,
    prepare_refund_recovery_data,
)


async def main() -> None:
    """Publish a new immutable bundle or replay-check an existing one."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--audit", action="store_true")
    args = parser.parse_args()
    if not args.audit:
        await prepare_refund_recovery_data(args.output, seed=args.seed)
    print(json.dumps(await audit_refund_recovery_data(args.output), indent=2))


if __name__ == "__main__":
    asyncio.run(main())
