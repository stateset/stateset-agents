"""Deprecated forwarder for the Gemma 4 31B GSPO finetune script.

Every flag this script used to expose (--starter-profile, --config,
--write-config, --list-profiles, --use-lora/--no-lora, --use-4bit,
--use-8bit, --wandb, --output-dir, --task, --dry-run) is now reproduced by
the unified driver in ``examples/finetune_gspo.py``. Use that instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from examples.finetune_gspo import main as _driver_main  # noqa: E402

print(
    "examples/finetune_gemma4_31b_gspo.py is deprecated; use "
    "'python examples/finetune_gspo.py --model gemma4-31b' instead.",
    file=sys.stderr,
)
sys.exit(_driver_main(["--model", "gemma4-31b", *sys.argv[1:]]))
