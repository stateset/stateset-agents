"""Interactive 3D Weight Visualization, Surgical Editing, Finetuning & Inference Studio."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .editor import WeightEditEngine
    from .finetune_engine import FinetuneStudioEngine
    from .inference_engine import InferenceStudioEngine
    from .inspector import WeightInspector
    from .router import StudioRuntime, create_studio_app, mount_studio_ui
    from .surgery_loop import AutonomousSurgeryLoop, run_autonomous_heal

__all__ = [
    "AutonomousSurgeryLoop",
    "FinetuneStudioEngine",
    "InferenceStudioEngine",
    "StudioRuntime",
    "WeightEditEngine",
    "WeightInspector",
    "create_studio_app",
    "mount_studio_ui",
    "run_autonomous_heal",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AutonomousSurgeryLoop": (".surgery_loop", "AutonomousSurgeryLoop"),
    "run_autonomous_heal": (".surgery_loop", "run_autonomous_heal"),
    "WeightInspector": (".inspector", "WeightInspector"),
    "WeightEditEngine": (".editor", "WeightEditEngine"),
    "FinetuneStudioEngine": (".finetune_engine", "FinetuneStudioEngine"),
    "InferenceStudioEngine": (".inference_engine", "InferenceStudioEngine"),
    "StudioRuntime": (".router", "StudioRuntime"),
    "create_studio_app": (".router", "create_studio_app"),
    "mount_studio_ui": (".router", "mount_studio_ui"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY_EXPORTS:
        import importlib

        mod_name, attr_name = _LAZY_EXPORTS[name]
        mod = importlib.import_module(mod_name, __name__)
        return getattr(mod, attr_name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
