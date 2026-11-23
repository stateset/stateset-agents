"""Interactive 3D Weight Visualization, Surgical Editing, Finetuning & Inference Studio."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ._stats import (
        SlidingWindowRateLimiter,
        authorize_mcp_request,
        get_studio_ingress_config,
        resolve_confined_path,
    )
    from .canary_engine import CanaryRolloutEngine
    from .editor import WeightEditEngine
    from .finetune_engine import FinetuneStudioEngine
    from .gpu_surgery import (
        bake_torch_caa_steering,
        extract_torch_caa_vector,
        find_transformer_layers,
    )
    from .inference_engine import InferenceStudioEngine
    from .inspector import WeightInspector
    from .judge import PluggableRewardJudge
    from .ledger_store import StudioSQLiteStore
    from .live_serving import LiveServingBridge
    from .mcp_bridge import StudioEventBus, handle_mcp_jsonrpc
    from .redteam_engine import RedTeamSurgeryEngine
    from .router import StudioRuntime, create_studio_app, mount_studio_ui
    from .surgery_loop import AutonomousSurgeryLoop, run_autonomous_heal
    from .telemetry_ingest import LiveTraceCollector
    from .tournament_engine import ModelTournamentEngine

__all__ = [
    "AutonomousSurgeryLoop",
    "CanaryRolloutEngine",
    "FinetuneStudioEngine",
    "InferenceStudioEngine",
    "LiveServingBridge",
    "LiveTraceCollector",
    "ModelTournamentEngine",
    "PluggableRewardJudge",
    "RedTeamSurgeryEngine",
    "SlidingWindowRateLimiter",
    "StudioEventBus",
    "StudioRuntime",
    "StudioSQLiteStore",
    "WeightEditEngine",
    "WeightInspector",
    "authorize_mcp_request",
    "bake_torch_caa_steering",
    "create_studio_app",
    "extract_torch_caa_vector",
    "find_transformer_layers",
    "get_studio_ingress_config",
    "handle_mcp_jsonrpc",
    "mount_studio_ui",
    "resolve_confined_path",
    "run_autonomous_heal",
]

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "AutonomousSurgeryLoop": (".surgery_loop", "AutonomousSurgeryLoop"),
    "CanaryRolloutEngine": (".canary_engine", "CanaryRolloutEngine"),
    "LiveServingBridge": (".live_serving", "LiveServingBridge"),
    "LiveTraceCollector": (".telemetry_ingest", "LiveTraceCollector"),
    "ModelTournamentEngine": (".tournament_engine", "ModelTournamentEngine"),
    "PluggableRewardJudge": (".judge", "PluggableRewardJudge"),
    "RedTeamSurgeryEngine": (".redteam_engine", "RedTeamSurgeryEngine"),
    "SlidingWindowRateLimiter": ("._stats", "SlidingWindowRateLimiter"),
    "StudioEventBus": (".mcp_bridge", "StudioEventBus"),
    "StudioSQLiteStore": (".ledger_store", "StudioSQLiteStore"),
    "authorize_mcp_request": ("._stats", "authorize_mcp_request"),
    "bake_torch_caa_steering": (".gpu_surgery", "bake_torch_caa_steering"),
    "extract_torch_caa_vector": (".gpu_surgery", "extract_torch_caa_vector"),
    "find_transformer_layers": (".gpu_surgery", "find_transformer_layers"),
    "get_studio_ingress_config": ("._stats", "get_studio_ingress_config"),
    "handle_mcp_jsonrpc": (".mcp_bridge", "handle_mcp_jsonrpc"),
    "resolve_confined_path": ("._stats", "resolve_confined_path"),
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
