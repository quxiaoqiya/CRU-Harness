from __future__ import annotations
from dataclasses import dataclass, asdict
import hashlib
import json

@dataclass(frozen=True)
class ModelCapability:
    context_window: int = 8192
    supports_vision: bool = False
    supports_structured_output: bool = True
    supports_reasoning: bool = False
    max_output_tokens: int = 2048
    max_concurrency: int = 1
    name: str | None = None

    @property
    def capability_digest(self) -> str:
        payload = asdict(self); payload.pop("name", None)
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]

