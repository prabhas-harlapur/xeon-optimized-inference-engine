from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .runtime import InferenceEngine


class LoadModelRequest(BaseModel):
    model_id: str
    alias: str
    dtype: str = Field(default="bfloat16")


class UnloadModelRequest(BaseModel):
    alias: str


class SelectModelRequest(BaseModel):
    alias: str


def build_control_router(engine: InferenceEngine) -> APIRouter:
    router = APIRouter(prefix="/control", tags=["control"])

    @router.get("/system")
    def system():
        return engine.system_info()

    @router.get("/models")
    def models():
        return {"models": engine.list_models(), "active": engine.backend.active_alias()}

    @router.post("/models/load")
    def load_model(req: LoadModelRequest):
        try:
            return engine.load_model(model_id=req.model_id, alias=req.alias, dtype=req.dtype)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.post("/models/unload")
    def unload_model(req: UnloadModelRequest):
        try:
            engine.unload_model(req.alias)
            return {"status": "ok", "unloaded": req.alias}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    @router.post("/models/select")
    def select_model(req: SelectModelRequest):
        try:
            engine.set_active_model(req.alias)
            return {"status": "ok", "active": req.alias}
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

    return router
