"""Thin forwarder for the DeepSeek V4 Flash GSPO finetune script.

Every flag this script exposes (--starter-profile, --config,
--write-config, --list-profiles, --use-lora/--no-lora, --use-4bit,
--use-8bit, --wandb, --output-dir, --task, --dry-run) is provided by
the unified driver in ``examples/finetune_gspo.py``. Use that instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.finetune_gspo import main as _driver_main  # noqa: E402

print(
    "examples/finetune_deepseek_v4_gspo.py is a forwarder; use "
    "'python examples/finetune_gspo.py --model deepseek-v4' instead.",
    file=sys.stderr,
)
sys.exit(_driver_main(["--model", "deepseek-v4", *sys.argv[1:]]))
