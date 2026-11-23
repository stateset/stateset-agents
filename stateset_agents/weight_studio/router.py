"""FastAPI router and standalone app for the StateSet 3D Weight Studio."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from fastapi import (
    APIRouter,
    FastAPI,
    HTTPException,
    Query,
    Request,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from ._stats import (
    SlidingWindowRateLimiter,
    authorize_mcp_request,
    get_studio_ingress_config,
)
from .canary_engine import CanaryRolloutEngine
from .editor import WeightEditEngine
from .finetune_engine import FinetuneStudioEngine
from .inference_engine import InferenceStudioEngine
from .inspector import WeightInspector
from .judge import PluggableRewardJudge
from .ledger_store import StudioSQLiteStore
from .live_serving import LiveServingBridge
from .mcp_bridge import StudioEventBus, handle_mcp_jsonrpc
from .redteam_engine import RedTeamSurgeryEngine
from .surgery_loop import AutonomousSurgeryLoop
from .telemetry_ingest import LiveTraceCollector
from .tournament_engine import ModelTournamentEngine

STATIC_DIR = Path(__file__).resolve().parent / "static"


class StudioRuntime:
    """Shared singleton container for Weight Studio engines."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.inspector = WeightInspector(workspace_root=workspace_root)
        self.store = StudioSQLiteStore(self.inspector.workspace_root)
        self.editor = WeightEditEngine(self.inspector)
        self.finetuner = FinetuneStudioEngine(self.inspector, self.editor)
        self.inference = InferenceStudioEngine(self.inspector, self.editor)
        self.healer = AutonomousSurgeryLoop(self.inspector, self.editor, self.inference)
        self.redteam = RedTeamSurgeryEngine(self.inspector, self.editor, self.inference)
        self.tournament = ModelTournamentEngine(
            self.inspector, self.editor, self.inference, self.redteam
        )
        self.serving = LiveServingBridge(self.inspector.workspace_root)
        self.canary = CanaryRolloutEngine(
            self.inspector,
            self.editor,
            self.inference,
            self.healer,
            self.redteam,
            store=self.store,
            serving=self.serving,
        )
        self.traces = LiveTraceCollector(
            self.inspector.workspace_root, store=self.store
        )
        self.events = StudioEventBus()
        self.rl_telemetry: list[dict[str, Any]] = []


_DEFAULT_RUNTIME: StudioRuntime | None = None


def get_studio_runtime(workspace_root: Path | None = None) -> StudioRuntime:
    global _DEFAULT_RUNTIME
    if _DEFAULT_RUNTIME is None or (
        workspace_root is not None
        and _DEFAULT_RUNTIME.inspector.workspace_root != workspace_root.resolve()
    ):
        _DEFAULT_RUNTIME = StudioRuntime(workspace_root=workspace_root)
    return _DEFAULT_RUNTIME


# ---------------------------------------------------------------------------
# Request Schemas
# ---------------------------------------------------------------------------


class ApplyEditRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    op: str = Field(
        ...,
        pattern="^(scale|prune|svd_truncate|clamp|noise|region_edit|blend|head_edit|rank1_edit|steer_bake|quantize)$",
    )
    target: str
    params: dict[str, Any] = Field(default_factory=dict)
    label: str = ""


class RegisterHfRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    repo_id: str = Field(..., min_length=3, max_length=200)
    revision: str = "main"
    token: str = ""


class SensitivityRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    target_compression: float = Field(0.35, ge=0.1, le=0.65)


class SyncTelemetryRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    path: str = Field(..., min_length=1, max_length=400)


class ReportHtmlRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    tensor_name: str | None = None
    include_eval: bool = True
    include_sensitivity: bool = True


class ApplyRecipeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    recipe: dict[str, Any]
    reset_first: bool = True


class HistoryActionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    edit_id: str | None = None


class ExportCheckpointRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    export_name: str = Field(..., min_length=1, max_length=120)
    run_merge_probe: bool = False


class FinetuneConfigRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    config: dict[str, Any] = Field(default_factory=dict)


class InferenceRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    prompt: str = Field(..., min_length=1, max_length=8000)
    system_prompt: str = ""
    temperature: float = Field(0.7, ge=0.05, le=2.0)
    top_p: float = Field(0.9, ge=0.05, le=1.0)
    max_tokens: int = Field(48, ge=4, le=256)
    seed: int = 42
    compare_base: bool = True
    causal_patches: list[dict[str, Any]] = Field(default_factory=list)
    steering: dict[str, Any] | None = None
    endpoint_url: str = ""
    api_key: str = ""


class CircuitRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    prompt: str = Field(..., min_length=1, max_length=8000)
    system_prompt: str = ""
    top_k: int = Field(10, ge=3, le=24)


class SteeringExtractRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    name: str = Field("Custom Steering Vector", min_length=1, max_length=120)
    positive_prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = Field(..., min_length=1, max_length=4000)
    layer_index: int | None = None


class SteeringBakeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    vector_id: str
    strength: float = Field(0.65, ge=-2.5, le=2.5)
    target_layer: int | None = None


class BatchEvalRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    dataset_path: str | None = None
    custom_jsonl: str | None = None
    reward_fn: str = Field(
        "composite", pattern="^(composite|customer_support|tool_calling|gsm8k)$"
    )
    max_cases: int = Field(6, ge=2, le=12)
    causal_patches: list[dict[str, Any]] = Field(default_factory=list)
    steering: dict[str, Any] | None = None


class ExportLoraRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    export_name: str = Field(..., min_length=1, max_length=120)
    rank: int = Field(16, ge=1, le=128)
    alpha: int = Field(32, ge=1, le=512)


class ExportGgufRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    export_name: str = Field(..., min_length=1, max_length=120)
    quant_type: str = Field("Q8_0", pattern="^(F32|F16|Q8_0)$")


class ArenaCompareRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    prompt: str = Field(..., min_length=1, max_length=8000)
    system_prompt: str = ""
    reward_fn: str = Field(
        "composite", pattern="^(composite|customer_support|tool_calling|gsm8k)$"
    )
    candidates: list[dict[str, Any]] = Field(default_factory=list)


class AttentionMapRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    prompt: str = Field(..., min_length=1, max_length=8000)
    system_prompt: str = ""
    layer_index: int = Field(0, ge=0, le=256)
    num_heads: int = Field(8, ge=2, le=12)
    max_tokens: int = Field(12, ge=4, le=14)


class LoadSaeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    source_id: str = Field("stateset_builtin", min_length=1, max_length=120)
    local_path: str = ""
    repo_id: str = ""


class ResidualStreamRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    prompt: str = Field(..., min_length=1, max_length=8000)
    system_prompt: str = ""
    causal_patches: list[dict[str, Any]] = Field(default_factory=list)
    steering: dict[str, Any] | None = None


class AutoRepairRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    reward_fn: str = Field(
        "composite", pattern="^(composite|customer_support|tool_calling|gsm8k)$"
    )
    max_steps: int = Field(4, ge=1, le=12)


class HfCardRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    export_name: str = Field(..., min_length=1, max_length=120)
    repo_id: str = ""
    include_eval: bool = True
    push_to_hub: bool = False
    token: str = ""


class SaveSessionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    session_name: str = Field(..., min_length=1, max_length=120)
    model_id: str
    checkpoint_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoadSessionRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    session_name: str = Field(..., min_length=1, max_length=120)


class HealRunRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    transcripts_path: str | None = None
    transcripts_jsonl: str | None = None
    reward_fn: str = Field(
        "customer_support",
        pattern="^(composite|customer_support|tool_calling|gsm8k)$",
    )
    export_format: str = Field("none", pattern="^(none|lora|safetensors|gguf|hf_card)$")
    output_name: str = ""
    lora_rank: int = Field(16, ge=1, le=128)
    lora_alpha: int = Field(32, ge=1, le=512)
    require_go: bool = True


class RedTeamRunRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    categories: list[str] | None = None
    probes_path: str | None = None
    probes_jsonl: str | None = None
    max_probes: int = Field(10, ge=1, le=50)
    immunize: bool = False
    export_format: str = Field("none", pattern="^(none|lora|safetensors|gguf|hf_card)$")
    output_name: str = ""
    lora_rank: int = Field(16, ge=1, le=128)
    lora_alpha: int = Field(32, ge=1, le=512)
    live_endpoint_url: str = ""
    live_model: str = ""
    live_api_key: str = ""


class TournamentRunRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_ids: list[str] | None = None
    reward_fn: str = Field(
        "customer_support",
        pattern="^(composite|customer_support|tool_calling|gsm8k)$",
    )
    max_cases: int = Field(4, ge=1, le=12)
    include_redteam: bool = True
    synthesize_merge: bool = True
    merge_methods: list[str] | None = None
    export_format: str = Field("none", pattern="^(none|lora|safetensors|gguf|hf_card)$")
    output_name: str = ""
    lora_rank: int = Field(16, ge=1, le=128)
    lora_alpha: int = Field(32, ge=1, le=512)


class CanaryGateRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    stage: str = Field("canary_10", pattern="^(shadow|canary_10|canary_50|production)$")
    traffic_path: str | None = None
    traffic_jsonl: str | None = None
    reward_fn: str = Field(
        "composite",
        pattern="^(composite|customer_support|tool_calling|gsm8k)$",
    )
    max_requests: int = Field(6, ge=1, le=20)
    max_asr: float = Field(0.25, ge=0.0, le=1.0)
    record_ledger: bool = True
    live_endpoint_url: str = ""
    live_baseline_model: str = ""
    live_candidate_model: str = ""
    live_api_key: str = ""


class AutopilotReleaseRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    transcripts_path: str | None = None
    transcripts_jsonl: str | None = None
    reward_fn: str = Field(
        "customer_support",
        pattern="^(composite|customer_support|tool_calling|gsm8k)$",
    )
    immunize_redteam: bool = True
    stage: str = Field("canary_10", pattern="^(shadow|canary_10|canary_50|production)$")
    export_format: str = Field("lora", pattern="^(none|lora|safetensors|gguf|hf_card)$")
    output_name: str = ""
    lora_rank: int = Field(16, ge=1, le=128)
    lora_alpha: int = Field(32, ge=1, le=512)


class LoadLoraAdapterRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    endpoint_url: str = Field(..., min_length=7, max_length=300)
    lora_name: str = Field(..., min_length=1, max_length=120)
    lora_path: str = Field(..., min_length=1, max_length=400)
    api_key: str = ""


class LiveCompareRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    endpoint_url: str = Field(..., min_length=7, max_length=300)
    baseline_model: str = Field(..., min_length=1, max_length=200)
    candidate_model: str = Field(..., min_length=1, max_length=200)
    prompts: list[str] | None = None
    system_prompt: str = "You are a helpful, policy-grounded StateSet agent."
    reward_fn: str = Field(
        "customer_support",
        pattern="^(composite|customer_support|tool_calling|gsm8k)$",
    )
    max_tokens: int = Field(128, ge=8, le=1024)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    api_key: str = ""
    candidate_endpoint_url: str = ""
    candidate_api_key: str = ""


class TraceIngestRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    payload: dict[str, Any]
    reward_fn: str = Field(
        "customer_support",
        pattern="^(composite|customer_support|tool_calling|gsm8k)$",
    )
    fail_threshold: float = Field(0.72, ge=0.0, le=1.0)
    judge_mode: str = Field("heuristic", pattern="^(heuristic|llm_judge|hybrid)$")
    judge_endpoint_url: str = ""
    judge_model: str = ""
    judge_api_key: str = ""
    judge_weight: float = Field(0.65, ge=0.0, le=1.0)


class CalibrationAuditRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    samples: list[dict[str, Any]] | None = None
    reward_fn: str = Field(
        "customer_support",
        pattern="^(composite|customer_support|tool_calling|gsm8k)$",
    )


class TraceAutoHealRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    reward_fn: str = Field(
        "customer_support",
        pattern="^(composite|customer_support|tool_calling|gsm8k)$",
    )
    export_format: str = Field("none", pattern="^(none|lora|safetensors|gguf|hf_card)$")
    output_name: str = ""


class SaeAnalyzeRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    prompt: str = Field(..., min_length=1, max_length=8000)
    system_prompt: str = ""
    layer_index: int | None = None
    top_k: int = Field(10, ge=4, le=24)
    clamps: list[dict[str, Any]] = Field(default_factory=list)


class DispatchRunRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    model_id: str
    checkpoint_id: str | None = None
    mode: str = Field("dry_run", pattern="^(dry_run|local_stub)$")
    config: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Router Factory
# ---------------------------------------------------------------------------


def build_studio_router(runtime: StudioRuntime | None = None) -> APIRouter:
    """Create the API router for Weight Studio endpoints."""
    api = APIRouter(prefix="/api/weight-studio", tags=["weight-studio"])

    def _rt() -> StudioRuntime:
        return runtime or get_studio_runtime()

    @api.get("/health")
    async def studio_health() -> dict[str, Any]:
        from stateset_agents import __version__ as pkg_version

        rt = _rt()
        return {
            "status": "ok",
            "version": pkg_version,
            "workspace": str(rt.inspector.workspace_root),
            "static_ready": (STATIC_DIR / "index.html").is_file(),
            "mcp_http_endpoint": "/api/weight-studio/mcp",
            "events_sse_endpoint": "/api/weight-studio/events",
        }

    @api.get("/models")
    async def list_models(refresh: bool = False) -> dict[str, Any]:
        rt = _rt()
        models = (
            rt.inspector.refresh_catalog() if refresh else rt.inspector.list_models()
        )
        return {"models": models, "count": len(models)}

    @api.get("/models/structure")
    async def model_structure(
        model_id: str = Query(...),
        checkpoint_id: str | None = Query(None),
    ) -> dict[str, Any]:
        rt = _rt()
        try:
            overrides = rt.editor.get_overrides(model_id, checkpoint_id)
            struct = rt.inspector.get_model_structure(
                model_id, checkpoint_id, overrides=overrides if overrides else None
            )
            struct["edit_history"] = rt.editor.get_history(model_id, checkpoint_id)
            return struct
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.get("/models/tensor")
    async def tensor_detail(
        model_id: str = Query(...),
        tensor_name: str = Query(...),
        checkpoint_id: str | None = Query(None),
        resolution: int = Query(64, ge=8, le=128),
        include_svd: bool = Query(True),
        secondary_model_id: str | None = Query(None),
        secondary_checkpoint_id: str | None = Query(None),
        r0: int | None = Query(None),
        r1: int | None = Query(None),
        c0: int | None = Query(None),
        c1: int | None = Query(None),
    ) -> dict[str, Any]:
        rt = _rt()
        crop = (
            (r0, r1, c0, c1)
            if r0 is not None and r1 is not None and c0 is not None and c1 is not None
            else None
        )
        try:
            edited = rt.editor.get_edited_tensor(model_id, tensor_name, checkpoint_id)
            return rt.inspector.inspect_tensor_detail(
                model_id,
                tensor_name,
                checkpoint_id=checkpoint_id,
                resolution=resolution,
                crop=crop,
                edited_array=edited,
                secondary_model_id=secondary_model_id,
                secondary_checkpoint_id=secondary_checkpoint_id,
                include_svd=include_svd,
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.get("/models/tensor.bin")
    async def tensor_binary(
        model_id: str = Query(...),
        tensor_name: str = Query(...),
        checkpoint_id: str | None = Query(None),
        resolution: int = Query(128, ge=8, le=128),
        secondary_model_id: str | None = Query(None),
        secondary_checkpoint_id: str | None = Query(None),
        r0: int | None = Query(None),
        r1: int | None = Query(None),
        c0: int | None = Query(None),
        c1: int | None = Query(None),
    ) -> Response:
        """High-speed binary Float32Array transport for up to 128x128 (16K crystal) Microscope grids."""
        rt = _rt()
        crop = (
            (r0, r1, c0, c1)
            if r0 is not None and r1 is not None and c0 is not None and c1 is not None
            else None
        )
        try:
            edited = rt.editor.get_edited_tensor(model_id, tensor_name, checkpoint_id)
            raw_bytes, meta = rt.inspector.build_binary_tensor_payload(
                model_id,
                tensor_name,
                checkpoint_id=checkpoint_id,
                resolution=resolution,
                crop=crop,
                edited_array=edited,
                secondary_model_id=secondary_model_id,
                secondary_checkpoint_id=secondary_checkpoint_id,
            )
            return Response(
                content=raw_bytes,
                media_type="application/octet-stream",
                headers={"X-Tensor-Meta": json.dumps(meta)},
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.get("/models/trajectory")
    async def model_trajectory(model_id: str = Query(...)) -> dict[str, Any]:
        rt = _rt()
        try:
            return rt.inspector.compute_trajectory_evolution(model_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # --- Surgical Weight Editing ---

    @api.post("/edits/apply")
    async def apply_edit(req: ApplyEditRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            return rt.editor.apply_edit(
                req.model_id,
                req.op,
                req.target,
                req.params,
                checkpoint_id=req.checkpoint_id,
                label=req.label,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/edits/undo")
    async def undo_edit(req: HistoryActionRequest) -> dict[str, Any]:
        return _rt().editor.undo(req.model_id, req.checkpoint_id)

    @api.post("/edits/redo")
    async def redo_edit(req: HistoryActionRequest) -> dict[str, Any]:
        return _rt().editor.redo(req.model_id, req.checkpoint_id)

    @api.post("/edits/toggle")
    async def toggle_edit(req: HistoryActionRequest) -> dict[str, Any]:
        if not req.edit_id:
            raise HTTPException(status_code=400, detail="edit_id is required")
        return _rt().editor.toggle_edit(req.model_id, req.edit_id, req.checkpoint_id)

    @api.post("/edits/reset")
    async def reset_edits(req: HistoryActionRequest) -> dict[str, Any]:
        return _rt().editor.reset(req.model_id, req.checkpoint_id)

    @api.post("/edits/export")
    async def export_checkpoint(req: ExportCheckpointRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            return rt.editor.export_verified_checkpoint(
                req.model_id,
                req.export_name,
                checkpoint_id=req.checkpoint_id,
                run_merge_probe=req.run_merge_probe,
            )
        except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/edits/recipe")
    async def export_edit_recipe(
        model_id: str = Query(...),
        checkpoint_id: str | None = Query(None),
    ) -> dict[str, Any]:
        try:
            return _rt().editor.export_recipe(model_id, checkpoint_id)
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/edits/recipe/apply")
    async def apply_edit_recipe(req: ApplyRecipeRequest) -> dict[str, Any]:
        try:
            return _rt().editor.apply_recipe(
                req.model_id,
                req.recipe,
                checkpoint_id=req.checkpoint_id,
                reset_first=req.reset_first,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Finetuning Workbench & Curated Datasets ---

    @api.get("/finetune/catalog")
    async def finetune_catalog() -> dict[str, Any]:
        return _rt().finetuner.get_catalog()

    @api.get("/finetune/datasets")
    async def finetune_datasets() -> dict[str, Any]:
        ds = _rt().finetuner.list_datasets()
        return {"datasets": ds, "count": len(ds)}

    @api.get("/finetune/datasets/inspect")
    async def finetune_dataset_inspect(path: str = Query(...)) -> dict[str, Any]:
        try:
            return _rt().finetuner.inspect_dataset(path)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.get("/finetune/workspace-runs")
    async def finetune_workspace_runs() -> dict[str, Any]:
        return {"runs": _rt().finetuner.list_workspace_runs()}

    @api.post("/finetune/estimate")
    async def finetune_estimate(req: FinetuneConfigRequest) -> dict[str, Any]:
        try:
            return _rt().finetuner.estimate(req.model_id, req.config)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/finetune/start")
    async def finetune_start(req: FinetuneConfigRequest) -> dict[str, Any]:
        try:
            return await _rt().finetuner.start_job(
                req.model_id, req.config, checkpoint_id=req.checkpoint_id
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/finetune/jobs")
    async def finetune_jobs() -> dict[str, Any]:
        return {"jobs": _rt().finetuner.list_jobs()}

    @api.get("/finetune/jobs/{job_id}")
    async def finetune_job_status(job_id: str) -> dict[str, Any]:
        try:
            return _rt().finetuner.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/finetune/jobs/{job_id}/pause")
    async def finetune_job_pause(job_id: str) -> dict[str, Any]:
        try:
            return _rt().finetuner.pause_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/finetune/jobs/{job_id}/resume")
    async def finetune_job_resume(job_id: str) -> dict[str, Any]:
        try:
            return _rt().finetuner.resume_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/finetune/jobs/{job_id}/stop")
    async def finetune_job_stop(job_id: str) -> dict[str, Any]:
        try:
            return _rt().finetuner.stop_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.get("/finetune/stream/{job_id}")
    async def finetune_sse_stream(job_id: str) -> StreamingResponse:
        rt = _rt()
        try:
            snapshot = rt.finetuner.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def _events():
            queue = rt.finetuner.subscribe(job_id)
            try:
                yield f"data: {json.dumps({'type': 'snapshot', 'job': snapshot})}\n\n"
                if snapshot.get("status") in {"completed", "failed", "stopped"}:
                    return
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"data: {json.dumps(event)}\n\n"
                        if event.get("type") in {"completed", "failed"}:
                            break
                    except asyncio.TimeoutError:
                        yield ": keepalive\n\n"
            finally:
                rt.finetuner.unsubscribe(job_id, queue)

        return StreamingResponse(_events(), media_type="text/event-stream")

    @api.websocket("/finetune/ws/{job_id}")
    async def finetune_ws(websocket: WebSocket, job_id: str) -> None:
        await websocket.accept()
        rt = _rt()
        queue = rt.finetuner.subscribe(job_id)
        try:
            try:
                await websocket.send_json(
                    {"type": "snapshot", "job": rt.finetuner.get_job(job_id)}
                )
            except KeyError:
                await websocket.send_json({"type": "error", "detail": "Job not found"})
                return
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15.0)
                    await websocket.send_json(event)
                    if event.get("type") in {"completed", "failed"}:
                        break
                except asyncio.TimeoutError:
                    await websocket.send_json({"type": "ping"})
        except WebSocketDisconnect:
            pass
        finally:
            rt.finetuner.unsubscribe(job_id, queue)

    # --- Interactive Model Inference Playground ---

    @api.get("/inference/scenarios")
    async def inference_scenarios() -> dict[str, Any]:
        return {"scenarios": _rt().inference.list_scenarios()}

    @api.post("/inference/run")
    async def run_inference(req: InferenceRequest) -> dict[str, Any]:
        try:
            return _rt().inference.run_inference(
                req.model_id,
                req.prompt,
                checkpoint_id=req.checkpoint_id,
                system_prompt=req.system_prompt,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
                seed=req.seed,
                compare_base=req.compare_base,
                causal_patches=req.causal_patches,
                steering=req.steering,
                endpoint_url=req.endpoint_url,
                api_key=req.api_key,
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/inference/circuit")
    async def discover_circuit(req: CircuitRequest) -> dict[str, Any]:
        try:
            return _rt().inference.discover_causal_circuit(
                req.model_id,
                req.prompt,
                checkpoint_id=req.checkpoint_id,
                system_prompt=req.system_prompt,
                top_k=req.top_k,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Contrastive Activation Steering (CAA / RepE) & Weight Baking ---

    @api.get("/steering/presets")
    async def steering_presets() -> dict[str, Any]:
        return {"presets": _rt().inference.list_steering_presets()}

    @api.post("/steering/extract")
    async def steering_extract(req: SteeringExtractRequest) -> dict[str, Any]:
        try:
            return _rt().inference.extract_steering_vector(
                req.model_id,
                req.positive_prompt,
                req.negative_prompt,
                checkpoint_id=req.checkpoint_id,
                name=req.name,
                layer_index=req.layer_index,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/steering/bake")
    async def steering_bake(req: SteeringBakeRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            vec_rec = rt.inference.get_steering_vector(req.vector_id)
            layer_idx = (
                int(req.target_layer)
                if req.target_layer is not None
                else int(vec_rec["recommended_layer"])
            )
            layer_vecs = vec_rec.get("layer_vectors", {})
            direction = layer_vecs.get(layer_idx) or vec_rec["direction"]
            return rt.editor.apply_edit(
                req.model_id,
                "steer_bake",
                f"layer:{layer_idx}",
                {
                    "concept": vec_rec["name"],
                    "strength": req.strength,
                    "direction": direction,
                },
                checkpoint_id=req.checkpoint_id,
                label=f"Bake CAA '{vec_rec['name'][:18]}' (L{layer_idx:02d}, α={req.strength:+.2f})",
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Batch Eval Gate (Pre-Export Regression Scorecard + Custom JSONL & Rewards) ---

    @api.get("/eval/rewards")
    async def list_eval_rewards() -> dict[str, Any]:
        return {"rewards": _rt().inference.list_eval_rewards()}

    @api.post("/eval/run")
    async def run_batch_eval(req: BatchEvalRequest) -> dict[str, Any]:
        try:
            return _rt().inference.run_batch_eval(
                req.model_id,
                checkpoint_id=req.checkpoint_id,
                dataset_path=req.dataset_path,
                custom_jsonl=req.custom_jsonl,
                reward_fn=req.reward_fn,
                max_cases=req.max_cases,
                causal_patches=req.causal_patches,
                steering=req.steering,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Studio v6: Diff-as-LoRA & GGUF v3 / Ollama Export ---

    @api.post("/export/lora")
    async def export_diff_lora(req: ExportLoraRequest) -> dict[str, Any]:
        try:
            return _rt().editor.export_diff_as_lora(
                req.model_id,
                req.export_name,
                rank=req.rank,
                alpha=req.alpha,
                checkpoint_id=req.checkpoint_id,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/export/gguf")
    async def export_gguf(req: ExportGgufRequest) -> dict[str, Any]:
        try:
            return _rt().editor.export_gguf_package(
                req.model_id,
                req.export_name,
                quant_type=req.quant_type,
                checkpoint_id=req.checkpoint_id,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Studio v6: Multi-Edit A/B Arena ---

    @api.post("/arena/compare")
    async def arena_compare(req: ArenaCompareRequest) -> dict[str, Any]:
        try:
            return _rt().inference.compare_arena_candidates(
                req.model_id,
                req.prompt,
                req.candidates or None,
                system_prompt=req.system_prompt,
                checkpoint_id=req.checkpoint_id,
                reward_fn=req.reward_fn,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Studio v6: Multi-Head Causal Attention 3D Matrix ---

    @api.post("/inference/attention")
    async def inference_attention(req: AttentionMapRequest) -> dict[str, Any]:
        try:
            return _rt().inference.compute_attention_map(
                req.model_id,
                req.prompt,
                checkpoint_id=req.checkpoint_id,
                system_prompt=req.system_prompt,
                layer_index=req.layer_index,
                num_heads=req.num_heads,
                max_tokens=req.max_tokens,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Sparse Autoencoder (SAE) Monosemantic Feature Explorer & Pretrained Loader ---

    @api.get("/sae/dictionaries")
    async def sae_dictionaries() -> dict[str, Any]:
        return {"dictionaries": _rt().inference.list_sae_dictionaries()}

    @api.post("/sae/load")
    async def sae_load(req: LoadSaeRequest) -> dict[str, Any]:
        try:
            return _rt().inference.load_sae_dictionary(
                req.source_id,
                local_path=req.local_path,
                repo_id=req.repo_id,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Studio v7: Residual Stream 3D, Auto-Repair, HF Model Card & Sessions ---

    @api.post("/inference/residual")
    async def inference_residual(req: ResidualStreamRequest) -> dict[str, Any]:
        try:
            return _rt().inference.compute_residual_stream_3d(
                req.model_id,
                req.prompt,
                checkpoint_id=req.checkpoint_id,
                system_prompt=req.system_prompt,
                causal_patches=req.causal_patches,
                steering=req.steering,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/edits/auto-repair")
    async def auto_repair_edits(req: AutoRepairRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            return rt.editor.auto_repair_edits(
                req.model_id,
                rt.inference,
                checkpoint_id=req.checkpoint_id,
                reward_fn=req.reward_fn,
                max_steps=req.max_steps,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/export/hf-card")
    async def export_hf_card(req: HfCardRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            eval_rep = (
                rt.inference.run_batch_eval(
                    req.model_id, checkpoint_id=req.checkpoint_id, max_cases=6
                )
                if req.include_eval
                else None
            )
            return rt.editor.export_hf_model_card(
                req.model_id,
                req.export_name,
                repo_id=req.repo_id,
                eval_report=eval_rep,
                checkpoint_id=req.checkpoint_id,
                push_to_hub=req.push_to_hub,
                token=req.token,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/sessions")
    async def list_sessions() -> dict[str, Any]:
        sessions = _rt().editor.list_sessions()
        return {"sessions": sessions, "count": len(sessions)}

    @api.post("/sessions/save")
    async def save_session(req: SaveSessionRequest) -> dict[str, Any]:
        try:
            return _rt().editor.save_session(
                req.session_name,
                model_id=req.model_id,
                checkpoint_id=req.checkpoint_id,
                metadata=req.metadata,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/sessions/load")
    async def load_session(req: LoadSessionRequest) -> dict[str, Any]:
        try:
            return _rt().editor.load_session(req.session_name)
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/heal/run")
    async def run_autonomous_heal_endpoint(req: HealRunRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            res = await asyncio.to_thread(
                rt.healer.run_healing_loop,
                req.model_id,
                checkpoint_id=req.checkpoint_id,
                transcripts_path=req.transcripts_path,
                transcripts_jsonl=req.transcripts_jsonl,
                reward_fn=req.reward_fn,
                export_format=req.export_format,
                output_name=req.output_name,
                lora_rank=req.lora_rank,
                lora_alpha=req.lora_alpha,
                require_go=req.require_go,
            )
            rt.events.publish(
                "studio_heal",
                source="studio_api",
                tool="studio_auto_heal",
                model_id=req.model_id,
                summary=f"Auto-Heal → {(res.get('eval_gate') or {}).get('verdict', 'GO')}",
                mutated=True,
                pulse="backprop",
            )
            return res
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/redteam/run")
    async def run_redteam_audit_endpoint(req: RedTeamRunRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            res = await asyncio.to_thread(
                rt.redteam.run_redteam_audit,
                req.model_id,
                checkpoint_id=req.checkpoint_id,
                categories=req.categories,
                probes_path=req.probes_path,
                probes_jsonl=req.probes_jsonl,
                max_probes=req.max_probes,
                immunize=req.immunize,
                export_format=req.export_format,
                output_name=req.output_name,
                lora_rank=req.lora_rank,
                lora_alpha=req.lora_alpha,
                live_endpoint_url=req.live_endpoint_url,
                live_model=req.live_model,
                live_api_key=req.live_api_key,
            )
            rt.events.publish(
                "studio_redteam",
                source="studio_api",
                tool="studio_redteam_audit",
                model_id=req.model_id,
                summary=(
                    f"{'Immunized' if req.immunize else 'Red-Team Audit'} → "
                    f"{res['verdict']} (ASR {res['attack_success_rate'] * 100:.0f}%)"
                ),
                mutated=req.immunize,
                pulse="backprop" if req.immunize else "forward",
            )
            return res
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/tournament/run")
    async def run_tournament_endpoint(req: TournamentRunRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            res = await asyncio.to_thread(
                rt.tournament.run_tournament,
                model_ids=req.model_ids,
                reward_fn=req.reward_fn,
                max_cases=req.max_cases,
                include_redteam=req.include_redteam,
                synthesize_merge=req.synthesize_merge,
                merge_methods=req.merge_methods,
                export_format=req.export_format,
                output_name=req.output_name,
                lora_rank=req.lora_rank,
                lora_alpha=req.lora_alpha,
            )
            champ = res.get("champion") or {}
            mutated = bool(res.get("promoted_merge"))
            rt.events.publish(
                "studio_tournament",
                source="studio_api",
                tool="studio_model_tournament",
                model_id=str(champ.get("model_id") or ""),
                summary=(
                    f"Tournament Champion: {champ.get('label', '?')} "
                    f"(fit {champ.get('fitness', 0):.3f})"
                ),
                mutated=mutated,
                pulse="backprop" if mutated else "forward",
            )
            return res
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/canary/gate")
    async def run_canary_gate_endpoint(req: CanaryGateRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            res = await asyncio.to_thread(
                rt.canary.run_canary_gate,
                req.model_id,
                checkpoint_id=req.checkpoint_id,
                stage=req.stage,
                traffic_path=req.traffic_path,
                traffic_jsonl=req.traffic_jsonl,
                reward_fn=req.reward_fn,
                max_requests=req.max_requests,
                max_asr=req.max_asr,
                record_ledger=req.record_ledger,
                live_endpoint_url=req.live_endpoint_url,
                live_baseline_model=req.live_baseline_model,
                live_candidate_model=req.live_candidate_model,
                live_api_key=req.live_api_key,
            )
            rt.events.publish(
                "studio_canary",
                source="studio_api",
                tool="studio_canary_gate",
                model_id=req.model_id,
                summary=f"Canary Gate [{req.stage}] → {res['decision']} ({res['recommended_traffic_pct']}%)",
                mutated=False,
                pulse="forward",
            )
            return res
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/canary/ledger")
    async def list_canary_ledger(
        limit: int = Query(25, ge=1, le=100),
    ) -> dict[str, Any]:
        rollouts = _rt().canary.list_rollouts(limit=limit)
        return {"rollouts": rollouts, "count": len(rollouts)}

    @api.post("/autopilot/run")
    async def run_autopilot_endpoint(req: AutopilotReleaseRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            res = await asyncio.to_thread(
                rt.canary.run_autopilot_release,
                req.model_id,
                checkpoint_id=req.checkpoint_id,
                transcripts_path=req.transcripts_path,
                transcripts_jsonl=req.transcripts_jsonl,
                reward_fn=req.reward_fn,
                immunize_redteam=req.immunize_redteam,
                stage=req.stage,
                export_format=req.export_format,
                output_name=req.output_name,
                lora_rank=req.lora_rank,
                lora_alpha=req.lora_alpha,
            )
            rt.events.publish(
                "studio_autopilot",
                source="studio_api",
                tool="studio_autopilot_release",
                model_id=req.model_id,
                summary=f"Autopilot → {res['decision']} (traffic {res['recommended_traffic_pct']}%)",
                mutated=True,
                pulse="backprop",
            )
            return res
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Live vLLM/Ollama LoRA Hot-Swap & Real-Endpoint Shadow A/B ---

    @api.post("/serving/load-lora")
    async def serving_load_lora(req: LoadLoraAdapterRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            res = await asyncio.to_thread(
                rt.serving.load_lora_adapter,
                req.endpoint_url,
                req.lora_name,
                req.lora_path,
                api_key=req.api_key,
            )
            rt.events.publish(
                "serving_lora_loaded",
                source="studio_api",
                tool="serving_load_lora",
                summary=f"LoRA '{req.lora_name}' -> {req.endpoint_url} ({res['status_code']})",
                mutated=False,
                pulse="forward",
            )
            return res
        except (ValueError, FileNotFoundError, PermissionError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.post("/serving/compare-live")
    async def serving_compare_live(req: LiveCompareRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            res = await asyncio.to_thread(
                rt.serving.compare_live_endpoints,
                req.endpoint_url,
                req.baseline_model,
                req.candidate_model,
                prompts=req.prompts,
                system_prompt=req.system_prompt,
                reward_fn=req.reward_fn,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                api_key=req.api_key,
                candidate_endpoint_url=req.candidate_endpoint_url,
                candidate_api_key=req.candidate_api_key,
            )
            rt.events.publish(
                "serving_live_ab",
                source="studio_api",
                tool="studio_live_serving_ab",
                summary=f"Live A/B ({req.baseline_model} vs {req.candidate_model}) → {res['decision']}",
                mutated=False,
                pulse="forward",
            )
            return res
        except (ValueError, RuntimeError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Live Production Trace Collector (OTLP/JSON + Webhook) & Calibration ---

    @api.post("/traces/ingest")
    async def traces_ingest(req: TraceIngestRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            judge = (
                PluggableRewardJudge(
                    mode=req.judge_mode,
                    judge_endpoint_url=req.judge_endpoint_url,
                    judge_model=req.judge_model,
                    judge_api_key=req.judge_api_key,
                    judge_weight=req.judge_weight,
                )
                if req.judge_mode != "heuristic"
                else None
            )
            res = await asyncio.to_thread(
                rt.traces.ingest_payload,
                req.payload,
                reward_fn=req.reward_fn,
                fail_threshold=req.fail_threshold,
                judge=judge,
            )
            rt.events.publish(
                "trace_ingested",
                source="otlp_webhook",
                tool="studio_ingest_trace",
                summary=f"Ingested {res['ingested_count']} trace(s) ({res['failing_count']} failing)",
                mutated=False,
                pulse="forward",
            )
            return res
        except (ValueError, OSError, RuntimeError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/traces/summary")
    async def traces_summary() -> dict[str, Any]:
        return _rt().traces.summary()

    @api.get("/store/traces")
    async def query_store_traces(
        failed_only: bool = False,
        source: str | None = None,
        search: str | None = None,
        labeled_only: bool = False,
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = _rt().traces.query_traces(
            failed_only=failed_only,
            source=source,
            search=search,
            labeled_only=labeled_only,
            limit=limit,
        )
        return {"traces": rows, "count": len(rows)}

    @api.get("/store/rollouts")
    async def query_store_rollouts(
        decision: str | None = None,
        model_id: str | None = None,
        stage: str | None = None,
        limit: int = Query(50, ge=1, le=500),
    ) -> dict[str, Any]:
        rows = _rt().canary.query_rollouts(
            decision=decision,
            model_id=model_id,
            stage=stage,
            limit=limit,
        )
        return {"rollouts": rows, "count": len(rows)}

    @api.get("/store/stats")
    async def store_stats() -> dict[str, Any]:
        return _rt().store.stats()

    # --- Live RL Training Cockpit & Online Reward-Hacking Guard Stream ---

    @api.post("/rl/telemetry")
    async def ingest_rl_telemetry(request: Request) -> dict[str, Any]:
        rt = _rt()
        try:
            body = await request.json()
        except Exception as exc:
            raise HTTPException(status_code=400, detail=f"Invalid JSON: {exc}") from exc
        if not isinstance(body, dict):
            raise HTTPException(
                status_code=400, detail="Payload must be a JSON object."
            )

        record = dict(body)
        rt.rl_telemetry.append(record)
        if len(rt.rl_telemetry) > 200:
            del rt.rl_telemetry[:-200]

        step = record.get("step", len(rt.rl_telemetry))
        run_name = str(record.get("run_name") or "rl-run")
        reward = record.get("reward")
        loss = record.get("loss")
        entropy = record.get("entropy")
        guard_action = str(record.get("guard_action") or "CONTINUE")
        summary_parts = [f"{run_name} step {step}"]
        if reward is not None:
            summary_parts.append(f"R={float(reward):.3f}")
        if loss is not None:
            summary_parts.append(f"L={float(loss):.4f}")
        if entropy is not None:
            summary_parts.append(f"H={float(entropy):.3f}")
        summary_parts.append(f"Guard={guard_action}")

        rt.events.publish(
            "rl_step",
            source="rl_trainer",
            tool="studio_rl_telemetry",
            summary=" · ".join(summary_parts),
            mutated=False,
            pulse="backprop",
            payload=record,
        )
        return {
            "accepted": True,
            "buffered_steps": len(rt.rl_telemetry),
            "guard_action": guard_action,
        }

    @api.get("/rl/telemetry")
    async def get_rl_telemetry(limit: int = Query(50, ge=1, le=200)) -> dict[str, Any]:
        rt = _rt()
        rows = rt.rl_telemetry[-limit:]
        latest = rows[-1] if rows else None
        return {
            "count": len(rows),
            "latest": latest,
            "steps": rows,
        }

    @api.post("/traces/calibration-audit")
    async def traces_calibration_audit(req: CalibrationAuditRequest) -> dict[str, Any]:
        rt = _rt()
        return await asyncio.to_thread(
            rt.traces.audit_reward_calibration,
            samples=req.samples,
            reward_fn=req.reward_fn,
        )

    @api.post("/traces/auto-heal")
    async def traces_auto_heal(req: TraceAutoHealRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            failing = rt.traces.list_traces(limit=100, failing_only=True)
            jsonl_blob = "\n".join(
                json.dumps({"prompt": r["prompt"], "response": r["response"]})
                for r in failing
            )
            res = await asyncio.to_thread(
                rt.healer.run_healing_loop,
                req.model_id,
                checkpoint_id=req.checkpoint_id,
                transcripts_jsonl=jsonl_blob or None,
                reward_fn=req.reward_fn,
                export_format=req.export_format,
                output_name=req.output_name,
            )
            res["ingested_failing_traces"] = len(failing)
            rt.events.publish(
                "trace_auto_heal",
                source="otlp_webhook",
                tool="studio_auto_heal",
                model_id=req.model_id,
                summary=f"Healed {len(failing)} live traces → {(res.get('eval_gate') or {}).get('verdict', 'GO')}",
                mutated=True,
                pulse="backprop",
            )
            return res
        except (KeyError, ValueError, FileNotFoundError, PermissionError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Live Streamable HTTP MCP Endpoint + Real-Time AI-Agent SSE Bus ---

    @api.post("/mcp")
    async def studio_mcp_http(request: Request) -> Response:
        ok, status_code, granted_scope, err_msg = authorize_mcp_request(
            request.headers.get("authorization"), required_scope="read"
        )
        if not ok:
            return JSONResponse(
                status_code=status_code,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32001, "message": err_msg},
                },
            )

        rt = _rt()
        try:
            body = await request.json()
        except Exception as exc:
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32700, "message": f"Parse error: {exc}"},
                },
            )

        if isinstance(body, list):
            responses = []
            for item in body:
                if isinstance(item, dict):
                    resp = await asyncio.to_thread(
                        handle_mcp_jsonrpc, rt, item, granted_scope=granted_scope
                    )
                    if resp is not None:
                        responses.append(resp)
            return JSONResponse(content=responses)

        if not isinstance(body, dict):
            return JSONResponse(
                status_code=400,
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {"code": -32600, "message": "Invalid JSON-RPC request"},
                },
            )

        resp = await asyncio.to_thread(
            handle_mcp_jsonrpc, rt, body, granted_scope=granted_scope
        )
        if resp is None:
            return Response(status_code=202)
        return JSONResponse(content=resp)

    @api.get("/events/recent")
    async def studio_recent_events(
        request: Request,
        limit: int = Query(20, ge=1, le=50),
    ) -> dict[str, Any]:
        ok, status_code, _, err_msg = authorize_mcp_request(
            request.headers.get("authorization"), required_scope="read"
        )
        if not ok:
            raise HTTPException(status_code=status_code, detail=err_msg)
        items = _rt().events.recent(limit=limit)
        return {"events": items, "count": len(items)}

    @api.get("/events")
    async def studio_event_stream(request: Request) -> StreamingResponse:
        ok, status_code, _, err_msg = authorize_mcp_request(
            request.headers.get("authorization"), required_scope="read"
        )
        if not ok:
            raise HTTPException(status_code=status_code, detail=err_msg)
        rt = _rt()
        q = rt.events.subscribe()

        async def _stream():
            try:
                yield f"data: {json.dumps({'type': 'connected', 'workspace': str(rt.inspector.workspace_root)})}\n\n"
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(q.get(), timeout=15.0)
                        yield f"data: {json.dumps(event)}\n\n"
                    except asyncio.TimeoutError:
                        yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"
            finally:
                rt.events.unsubscribe(q)

        return StreamingResponse(_stream(), media_type="text/event-stream")

    @api.post("/sae/analyze")
    async def sae_analyze(req: SaeAnalyzeRequest) -> dict[str, Any]:
        try:
            return _rt().inference.analyze_sae_features(
                req.model_id,
                req.prompt,
                checkpoint_id=req.checkpoint_id,
                system_prompt=req.system_prompt,
                layer_index=req.layer_index,
                top_k=req.top_k,
                clamps=req.clamps,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Managed CLI / Workspace Training Dispatch + Hot-Reload ---

    @api.post("/finetune/dispatch")
    async def finetune_dispatch(req: DispatchRunRequest) -> dict[str, Any]:
        try:
            return await _rt().finetuner.dispatch_training_run(
                req.model_id,
                req.config,
                mode=req.mode,
                checkpoint_id=req.checkpoint_id,
            )
        except (KeyError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    @api.get("/finetune/dispatch")
    async def finetune_dispatch_list() -> dict[str, Any]:
        return {"runs": _rt().finetuner.list_dispatch_runs()}

    @api.get("/finetune/dispatch/{run_id}")
    async def finetune_dispatch_status(run_id: str) -> dict[str, Any]:
        try:
            return _rt().finetuner.get_dispatch_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    @api.post("/finetune/dispatch/{run_id}/stop")
    async def finetune_dispatch_stop(run_id: str) -> dict[str, Any]:
        try:
            return _rt().finetuner.stop_dispatch_run(run_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # --- Zero-Download Hugging Face HTTP-Range Model Registration ---

    @api.post("/models/register-hf")
    async def register_hf_model(req: RegisterHfRequest) -> dict[str, Any]:
        try:
            summary = _rt().inspector.register_remote_hf_model(
                req.repo_id, revision=req.revision, token=req.token
            )
            return {"model": summary}
        except (ValueError, FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Auto-Sensitivity Profiler & Adaptive Rank/Sparsity Optimizer ---

    @api.post("/models/sensitivity")
    async def model_sensitivity(req: SensitivityRequest) -> dict[str, Any]:
        rt = _rt()
        try:
            overrides = rt.editor.get_overrides(req.model_id, req.checkpoint_id)
            return rt.inspector.profile_layer_sensitivity(
                req.model_id,
                checkpoint_id=req.checkpoint_id,
                overrides=overrides if overrides else None,
                target_compression=req.target_compression,
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Live Token-by-Token SSE Inference Streaming ---

    @api.post("/inference/stream")
    async def stream_inference(req: InferenceRequest) -> StreamingResponse:
        rt = _rt()
        try:
            result = rt.inference.run_inference(
                req.model_id,
                req.prompt,
                checkpoint_id=req.checkpoint_id,
                system_prompt=req.system_prompt,
                temperature=req.temperature,
                top_p=req.top_p,
                max_tokens=req.max_tokens,
                seed=req.seed,
                compare_base=req.compare_base,
                causal_patches=req.causal_patches,
                steering=req.steering,
                endpoint_url=req.endpoint_url,
                api_key=req.api_key,
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def _tok_stream():
            tokens = result["active"].get("tokens") or []
            for tok in tokens:
                yield f"data: {json.dumps({'type': 'token', 'token': tok})}\n\n"
                await asyncio.sleep(0.018)
            yield f"data: {json.dumps({'type': 'done', 'result': result})}\n\n"

        return StreamingResponse(_tok_stream(), media_type="text/event-stream")

    # --- External Trainer Telemetry Sync (trainer_state.json / W&B / JSONL) ---

    @api.get("/finetune/telemetry-sources")
    async def telemetry_sources() -> dict[str, Any]:
        srcs = _rt().finetuner.list_telemetry_sources()
        return {"sources": srcs, "count": len(srcs)}

    @api.post("/finetune/sync-telemetry")
    async def sync_telemetry(req: SyncTelemetryRequest) -> dict[str, Any]:
        try:
            return _rt().finetuner.sync_external_telemetry(req.path)
        except (FileNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

    # --- Shareable Single-File Interactive HTML PR Report ---

    @api.post("/report/html")
    async def export_html_report(req: ReportHtmlRequest) -> Response:
        from .report_exporter import build_standalone_html_report

        rt = _rt()
        try:
            overrides = rt.editor.get_overrides(req.model_id, req.checkpoint_id)
            struct = rt.inspector.get_model_structure(
                req.model_id,
                req.checkpoint_id,
                overrides=overrides if overrides else None,
            )
            history = rt.editor.get_history(req.model_id, req.checkpoint_id)
            recipe = rt.editor.export_recipe(req.model_id, req.checkpoint_id)
            eval_rep = (
                rt.inference.run_batch_eval(
                    req.model_id, checkpoint_id=req.checkpoint_id, max_cases=6
                )
                if req.include_eval
                else None
            )
            sens_rep = (
                rt.inspector.profile_layer_sensitivity(
                    req.model_id,
                    checkpoint_id=req.checkpoint_id,
                    overrides=overrides if overrides else None,
                )
                if req.include_sensitivity
                else None
            )
            t_detail = None
            if req.tensor_name:
                try:
                    edited = rt.editor.get_edited_tensor(
                        req.model_id, req.tensor_name, req.checkpoint_id
                    )
                    t_detail = rt.inspector.inspect_tensor_detail(
                        req.model_id,
                        req.tensor_name,
                        checkpoint_id=req.checkpoint_id,
                        resolution=32,
                        edited_array=edited,
                        include_svd=False,
                    )
                except (KeyError, FileNotFoundError):
                    t_detail = None

            html_doc = build_standalone_html_report(
                model_summary=struct["model"],
                structure_summary=struct["summary"],
                layers=struct["layers"],
                edit_history=history,
                recipe=recipe,
                eval_report=eval_rep,
                sensitivity_report=sens_rep,
                tensor_detail=t_detail,
            )
            return Response(
                content=html_doc,
                media_type="text/html; charset=utf-8",
                headers={
                    "Content-Disposition": 'attachment; filename="stateset-weight-studio-report.html"'
                },
            )
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    return api


def mount_studio_ui(app: FastAPI, *, runtime: StudioRuntime | None = None) -> None:
    """Mount the Weight Studio API router, ingress security middleware, and 3D browser UI."""
    ingress_cfg = get_studio_ingress_config()
    if ingress_cfg["cors_origins"]:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=list(ingress_cfg["cors_origins"]),
            allow_credentials=True,
            allow_methods=["GET", "POST", "OPTIONS"],
            allow_headers=["Authorization", "Content-Type", "Mcp-Session-Id"],
        )

    limiter = SlidingWindowRateLimiter(window_seconds=60.0)

    @app.middleware("http")
    async def studio_ingress_guard(request: Request, call_next: Any) -> Any:
        if request.url.path.startswith("/api/weight-studio"):
            cfg = get_studio_ingress_config()
            content_len_raw = request.headers.get("content-length")
            if content_len_raw:
                try:
                    if int(content_len_raw) > int(cfg["max_body_bytes"]):
                        return JSONResponse(
                            status_code=413,
                            content={
                                "detail": (
                                    f"Request body exceeds limit of {cfg['max_body_bytes']} bytes "
                                    "(configure STATESET_STUDIO_MAX_BODY_BYTES)."
                                )
                            },
                        )
                except ValueError:
                    pass

            rpm = int(cfg["rate_limit_rpm"])
            if rpm > 0:
                client_ip = (
                    request.client.host if request.client is not None else "unknown"
                )
                allowed, retry_after = limiter.allow(client_ip, rpm)
                if not allowed:
                    return JSONResponse(
                        status_code=429,
                        headers={"Retry-After": str(retry_after)},
                        content={
                            "detail": (
                                f"Rate limit exceeded ({rpm} req/min). "
                                f"Retry after {retry_after}s."
                            )
                        },
                    )
        return await call_next(request)

    app.include_router(build_studio_router(runtime))

    if STATIC_DIR.is_dir():
        app.mount(
            "/studio/static",
            StaticFiles(directory=str(STATIC_DIR)),
            name="weight-studio-static",
        )

        @app.get("/studio", include_in_schema=False)
        @app.get("/studio/", include_in_schema=False)
        async def studio_index() -> FileResponse:
            return FileResponse(str(STATIC_DIR / "index.html"))


def create_studio_app(workspace_root: Path | None = None) -> FastAPI:
    """Create a standalone FastAPI application serving the 3D Weight Studio."""
    runtime = StudioRuntime(workspace_root=workspace_root)
    app = FastAPI(
        title="StateSet 3D Weight Studio",
        description="Interactive 3D neural weight explorer, surgical editor, finetuning workbench & inference playground.",
        version="2.0.0",
    )
    mount_studio_ui(app, runtime=runtime)

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app
