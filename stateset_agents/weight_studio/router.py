"""FastAPI router and standalone app for the 3D Weight Studio."""

from __future__ import annotations

import asyncio
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
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .editor import WeightEditEngine
from .finetune_engine import FinetuneStudioEngine
from .inference_engine import InferenceStudioEngine
from .inspector import WeightInspector

STATIC_DIR = Path(__file__).resolve().parent / "static"


class StudioRuntime:
    """Shared singleton container for Weight Studio engines."""

    def __init__(self, workspace_root: Path | None = None) -> None:
        self.inspector = WeightInspector(workspace_root=workspace_root)
        self.editor = WeightEditEngine(self.inspector)
        self.finetuner = FinetuneStudioEngine(self.inspector, self.editor)
        self.inference = InferenceStudioEngine(self.inspector, self.editor)


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
        ..., pattern="^(scale|prune|svd_truncate|clamp|noise|region_edit|blend)$"
    )
    target: str
    params: dict[str, Any] = Field(default_factory=dict)
    label: str = ""


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
        resolution: int = Query(64, ge=8, le=96),
        include_svd: bool = Query(True),
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
                include_svd=include_svd,
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
                req.model_id, req.export_name, checkpoint_id=req.checkpoint_id
            )
        except (KeyError, ValueError, FileNotFoundError, OSError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

    # --- Finetuning Workbench ---

    @api.get("/finetune/catalog")
    async def finetune_catalog() -> dict[str, Any]:
        return _rt().finetuner.get_catalog()

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
        import json as _json

        rt = _rt()
        try:
            snapshot = rt.finetuner.get_job(job_id)
        except KeyError as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

        async def _events():
            queue = rt.finetuner.subscribe(job_id)
            try:
                yield f"data: {_json.dumps({'type': 'snapshot', 'job': snapshot})}\n\n"
                if snapshot.get("status") in {"completed", "failed", "stopped"}:
                    return
                while True:
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=15.0)
                        yield f"data: {_json.dumps(event)}\n\n"
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
            # Send current snapshot immediately
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
            )
        except (KeyError, FileNotFoundError) as exc:
            raise HTTPException(status_code=404, detail=str(exc)) from exc

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
        version="1.0.0",
    )
    mount_studio_ui(app, runtime=runtime)

    @app.get("/", include_in_schema=False)
    async def root_redirect() -> FileResponse:
        return FileResponse(str(STATIC_DIR / "index.html"))

    return app
