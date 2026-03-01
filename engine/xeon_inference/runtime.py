from __future__ import annotations

import time
from dataclasses import asdict
from threading import RLock
from typing import Optional

import torch

from .config import EngineConfig
from .model_registry import HFBackend
from .optimizations.amx_avx import host_summary, resolve_profile
from .scheduler import RequestRecord


class InferenceEngine:
    def __init__(self, config: Optional[EngineConfig] = None) -> None:
        self.config = config or EngineConfig()
        self.backend = HFBackend()
        self.profile = resolve_profile()
        self._lock = RLock()
        torch.set_num_threads(self.profile.intra_op_threads)
        torch.set_num_interop_threads(self.profile.inter_op_threads)

    def system_info(self) -> dict:
        return {
            "host": host_summary(),
            "profile": asdict(self.profile),
            "engine": self.config.model_dump(),
            "active_model": self.backend.active_alias(),
        }

    def load_model(self, model_id: str, alias: str, dtype: str) -> dict:
        return self.backend.load_model(model_id=model_id, alias=alias, dtype=dtype)

    def unload_model(self, alias: str) -> None:
        self.backend.unload_model(alias)

    def list_models(self) -> list[dict[str, str]]:
        return self.backend.list_models()

    def set_active_model(self, alias: str) -> None:
        self.backend.set_active(alias)

    def infer(self, request: RequestRecord) -> tuple[str, float]:
        t0 = time.perf_counter()
        text = self.backend.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            model_alias=request.model_alias,
            feature_flags=request.feature_flags,
        )
        return text, time.perf_counter() - t0
