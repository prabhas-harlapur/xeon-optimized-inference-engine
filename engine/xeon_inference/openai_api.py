from __future__ import annotations

import time
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .metrics import LATENCY_SECONDS, REQUESTS_TOTAL, TOKENS_IN, TOKENS_OUT
from .runtime import InferenceEngine
from .scheduler import RequestRecord


class CompletionRequest(BaseModel):
    model: Optional[str] = None
    prompt: str
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    feature_flags: Optional[dict[str, bool]] = None


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: int = Field(default=256, ge=1)
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    top_p: float = Field(default=1.0, ge=0.0, le=1.0)
    feature_flags: Optional[dict[str, bool]] = None


def _estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def _to_prompt(messages: list[ChatMessage]) -> str:
    return "\n".join([f"{m.role}: {m.content}" for m in messages]) + "\nassistant:"


def build_openai_router(engine: InferenceEngine) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["openai"])

    @router.post("/completions")
    def completions(req: CompletionRequest):
        req_id = f"cmpl-{uuid.uuid4().hex}"
        try:
            record = RequestRecord(
                request_id=req_id,
                prompt=req.prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                model_alias=req.model,
                feature_flags=req.feature_flags,
            )
            output, latency = engine.infer(record)
            model = req.model or engine.backend.active_alias() or "unknown"
            REQUESTS_TOTAL.labels(endpoint="completions", model=model).inc()
            TOKENS_IN.labels(model=model).inc(_estimate_tokens(req.prompt))
            TOKENS_OUT.labels(model=model).inc(_estimate_tokens(output))
            LATENCY_SECONDS.labels(endpoint="completions", model=model).observe(latency)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        created = int(time.time())
        return {
            "id": req_id,
            "object": "text_completion",
            "created": created,
            "model": model,
            "choices": [{"index": 0, "text": output, "finish_reason": "stop"}],
            "usage": {
                "prompt_tokens": _estimate_tokens(req.prompt),
                "completion_tokens": _estimate_tokens(output),
                "total_tokens": _estimate_tokens(req.prompt) + _estimate_tokens(output),
            },
        }

    @router.post("/chat/completions")
    def chat_completions(req: ChatCompletionRequest):
        prompt = _to_prompt(req.messages)
        req_id = f"chatcmpl-{uuid.uuid4().hex}"
        try:
            record = RequestRecord(
                request_id=req_id,
                prompt=prompt,
                max_new_tokens=req.max_tokens,
                temperature=req.temperature,
                top_p=req.top_p,
                model_alias=req.model,
                feature_flags=req.feature_flags,
            )
            output, latency = engine.infer(record)
            model = req.model or engine.backend.active_alias() or "unknown"
            REQUESTS_TOTAL.labels(endpoint="chat_completions", model=model).inc()
            TOKENS_IN.labels(model=model).inc(_estimate_tokens(prompt))
            TOKENS_OUT.labels(model=model).inc(_estimate_tokens(output))
            LATENCY_SECONDS.labels(endpoint="chat_completions", model=model).observe(latency)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e)) from e

        created = int(time.time())
        return {
            "id": req_id,
            "object": "chat.completion",
            "created": created,
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": output},
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": _estimate_tokens(prompt),
                "completion_tokens": _estimate_tokens(output),
                "total_tokens": _estimate_tokens(prompt) + _estimate_tokens(output),
            },
        }

    return router
