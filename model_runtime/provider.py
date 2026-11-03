from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Mapping, Protocol
from .capabilities import ModelCapability

class Provider(Protocol):
    capability: ModelCapability
    def complete(self, task: str, prompt: str, *, schema: Mapping[str, Any] | None = None, **kwargs: Any) -> Any: ...

@dataclass(frozen=True)
class ProviderResponse:
    value: Any
    capability_digest: str
    provider_digest: str
    cached: bool = False

