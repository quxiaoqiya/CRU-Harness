from __future__ import annotations
import hashlib
import json
from typing import Any, Callable, Mapping
from .capabilities import ModelCapability
from .structured_output import parse_structured_output

class MockProvider:
    """Offline provider. A handler may be injected for deterministic fixtures."""
    def __init__(self, capability: ModelCapability | None = None, handler: Callable[..., Any] | None = None, deterministic_only: bool = True):
        self.capability = capability or ModelCapability(name="mock")
        self.handler = handler
        self.deterministic_only = deterministic_only
        self.calls: list[dict[str, Any]] = []

    def complete(self, task: str, prompt: str, *, schema: Mapping[str, Any] | None = None, **kwargs: Any) -> dict[str, Any]:
        self.calls.append({"task": task, "prompt": prompt, "schema": schema})
        if self.handler:
            result = self.handler(task=task, prompt=prompt, schema=schema, **kwargs)
        elif self.deterministic_only:
            digest = hashlib.sha256(f"{task}\n{prompt}".encode()).hexdigest()
            result = {"task": task, "digest": digest, "facts": [], "evidence_refs": []}
        else:
            result = {"task": task, "text": prompt[: self.capability.max_output_tokens]}
        return parse_structured_output(result, schema)

