from dataclasses import dataclass
from typing import Optional


@dataclass
class RequestRecord:
    request_id: str
    prompt: str
    max_new_tokens: int
    temperature: float
    top_p: float
    model_alias: Optional[str] = None
    feature_flags: Optional[dict[str, bool]] = None
