from __future__ import annotations

from fastapi import FastAPI
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.responses import Response

from .control_api import build_control_router
from .metrics import HostSampler
from .openai_api import build_openai_router
from .runtime import InferenceEngine

engine = InferenceEngine()
app = FastAPI(title="Xeon Optimized Inference Engine", version="0.1.0")
app.include_router(build_openai_router(engine))
app.include_router(build_control_router(engine))
sampler = HostSampler(interval_seconds=2.0)


@app.on_event("startup")
def startup() -> None:
    sampler.start()


@app.on_event("shutdown")
def shutdown() -> None:
    sampler.stop()


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/metrics")
def metrics() -> Response:
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)
