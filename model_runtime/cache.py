from __future__ import annotations
import hashlib
import json
from dataclasses import dataclass
from typing import Any
from .capabilities import ModelCapability

@dataclass(frozen=True)
class CacheIdentity:
    content_hash: str
    prompt_hash: str
    schema_version: str
    capability_digest: str
    behavior_hash: str
    def key(self) -> str:
        raw = "|".join((self.content_hash, self.prompt_hash, self.schema_version, self.capability_digest, self.behavior_hash))
        return hashlib.sha256(raw.encode()).hexdigest()
    @classmethod
    def build(cls, content: str | bytes, prompt: str, schema_version: str, capability: ModelCapability, behavior: Any = "deterministic") -> "CacheIdentity":
        data = content if isinstance(content, bytes) else content.encode()
        return cls(hashlib.sha256(data).hexdigest(), hashlib.sha256(prompt.encode()).hexdigest(), schema_version, capability.capability_digest, hashlib.sha256(json.dumps(behavior, sort_keys=True, default=str).encode()).hexdigest())

