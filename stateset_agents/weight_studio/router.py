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
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .editor import WeightEditEngine
from .finetune_engine import FinetuneStudioEngine
from .inference_engine import InferenceStudioEngine
from .inspector import WeightInspector
from .surgery_loop import AutonomousSurgeryLoop

STATIC_DIR = Path(__file__).resolve().parent / "static"


class StudioRuntime:
    """Shared singleton container for Weight Studio engines."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.inspector = WeightInspector(workspace_root=workspace_root)
        self.editor = WeightEditEngine(self.inspector)
        self.finetuner = FinetuneStudioEngine(self.inspector, self.editor)
        self.inference = InferenceStudioEngine(self.inspector, self.editor)
        self.healer = AutonomousSurgeryLoop(self.inspector, self.editor, self.inference)


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
        rt = _rt()
        return {
            "status": "ok",
            "workspace": str(rt.inspector.workspace_root),
            "static_ready": (STATIC_DIR / "index.html").is_file(),
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
        try:
            return _rt().healer.run_healing_loop(
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
        except (KeyError, ValueError, FileNotFoundError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

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
    """Mount the Weight Studio API router and static 3D browser app onto a FastAPI app."""
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
