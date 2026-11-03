from __future__ import annotations
from dataclasses import dataclass, field
import os
from typing import Any
from .capabilities import ModelCapability
from .mock_provider import MockProvider

@dataclass
class RuntimeConfig:
    mode: str = "deterministic-only"
    capabilities: dict[str, ModelCapability] = field(default_factory=dict)
    @classmethod
    def from_env(cls) -> "RuntimeConfig":
        mode = os.getenv("AGENT_MODEL_MODE", "deterministic-only").strip().lower()
        return cls(mode=mode)
    def provider(self, task: str) -> MockProvider:
        return MockProvider(self.capabilities.get(task, ModelCapability(name="mock")), deterministic_only=self.mode == "deterministic-only")

