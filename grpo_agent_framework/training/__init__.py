# Proxy subpackage mapping to top-level training package
import sys as _sys
from importlib import import_module

_training = import_module("training")

for _name in dir(_training):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_training, _name)

# Map common submodules
for _sub in (
    "trainer",
    "config",
    "train",
    "distributed",
    "distributed_trainer",
    "advanced_training_orchestrator",
):
    try:
        _sys.modules[__name__ + f".{_sub}"] = import_module(f"training.{_sub}")
    except Exception:
        pass
