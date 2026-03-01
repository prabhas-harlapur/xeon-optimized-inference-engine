from __future__ import annotations

import gc
import threading
import time
from dataclasses import dataclass
from typing import Any, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class LoadedModel:
    alias: str
    model_id: str
    tokenizer: Any
    model: Any
    dtype: str


class HFBackend:
    def __init__(self) -> None:
        self._models: dict[str, LoadedModel] = {}
        self._active_alias: Optional[str] = None
        self._lock = threading.RLock()

    def list_models(self) -> list[dict[str, str]]:
        with self._lock:
            return [
                {
                    "alias": m.alias,
                    "model_id": m.model_id,
                    "dtype": m.dtype,
                    "active": str(m.alias == self._active_alias).lower(),
                }
                for m in self._models.values()
            ]

    def load_model(self, model_id: str, alias: str, dtype: str = "bfloat16") -> dict[str, str]:
        t0 = time.perf_counter()
        with self._lock:
            if alias in self._models:
                raise ValueError(f"alias already exists: {alias}")

            torch_dtype = {
                "float32": torch.float32,
                "bfloat16": torch.bfloat16,
                "float16": torch.float16,
            }.get(dtype, torch.bfloat16)

            tokenizer = AutoTokenizer.from_pretrained(model_id)
            model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch_dtype)
            model.eval()
            try:
                import intel_extension_for_pytorch as ipex
                model = ipex.optimize(model, dtype=torch_dtype, inplace=True)
            except Exception:
                pass

            loaded = LoadedModel(alias=alias, model_id=model_id, tokenizer=tokenizer, model=model, dtype=dtype)
            self._models[alias] = loaded
            if self._active_alias is None:
                self._active_alias = alias

        return {
            "alias": alias,
            "model_id": model_id,
            "dtype": dtype,
            "load_seconds": f"{time.perf_counter() - t0:.3f}",
        }

    def unload_model(self, alias: str) -> None:
        with self._lock:
            if alias not in self._models:
                raise ValueError(f"unknown alias: {alias}")
            del self._models[alias]
            if self._active_alias == alias:
                self._active_alias = next(iter(self._models.keys()), None)
        gc.collect()

    def set_active(self, alias: str) -> None:
        with self._lock:
            if alias not in self._models:
                raise ValueError(f"unknown alias: {alias}")
            self._active_alias = alias

    def active_alias(self) -> Optional[str]:
        return self._active_alias

    def generate(
        self,
        prompt: str,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        model_alias: Optional[str] = None,
        feature_flags: Optional[dict[str, bool]] = None,
    ) -> str:
        with self._lock:
            alias = model_alias or self._active_alias
            if alias is None or alias not in self._models:
                raise ValueError("no active model")
            m = self._models[alias]
            tokenizer = m.tokenizer
            model = m.model

        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=temperature > 0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        return tokenizer.decode(out[0], skip_special_tokens=True)
