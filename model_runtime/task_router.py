from __future__ import annotations
from typing import Any, Mapping
from .provider import Provider

class TaskRouter:
    def __init__(self, providers: Mapping[str, Provider], fallback: Provider | None = None):
        self.providers = dict(providers); self.fallback = fallback
    def provider_for(self, task: str) -> Provider:
        if task in self.providers: return self.providers[task]
        if self.fallback is not None: return self.fallback
        raise KeyError(f"no capability provider for task {task!r}")
    def complete(self, task: str, prompt: str, *, schema: Mapping[str, Any] | None = None, **kwargs: Any) -> Any:
        return self.provider_for(task).complete(task, prompt, schema=schema, **kwargs)

