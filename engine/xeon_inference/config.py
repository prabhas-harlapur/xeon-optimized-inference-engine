from pydantic import BaseModel, Field


class EngineConfig(BaseModel):
    max_batch_size: int = Field(default=32, ge=1)
    max_queued_requests: int = Field(default=2048, ge=1)
    default_max_new_tokens: int = Field(default=256, ge=1)
    tokenizer_workers: int = Field(default=8, ge=1)
    scheduler_tick_ms: int = Field(default=3, ge=1)
