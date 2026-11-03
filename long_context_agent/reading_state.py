from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
class AgentStatus(str, Enum): INIT="init"; INDEXED="indexed"; READING="reading"; RECONCILING="reconciling"; VALIDATING="validating"; COMPLETE="complete"; PAUSED="paused"; FAILED="failed"
@dataclass
class Budget:
    max_model_calls: int = 64; max_vision_calls: int = 16; max_chunks: int = 10000; max_memory_points: int = 10000; max_seconds: float = 900.0; model_calls: int = 0; vision_calls: int = 0; chunk_reads: int = 0
    def can_call(self, vision=False): return (self.vision_calls < self.max_vision_calls if vision else self.model_calls < self.max_model_calls)
    def consume(self, vision=False):
        if not self.can_call(vision): return False
        if vision: self.vision_calls += 1
        else: self.model_calls += 1
        return True
    def consume_chunk(self):
        if self.chunk_reads >= self.max_chunks: return False
        self.chunk_reads += 1
        return True
@dataclass
class ReadingState:
    document_id: str
    status: AgentStatus = AgentStatus.INIT
    action_index: int = 0
    visited_sections: list[str] = field(default_factory=list)
    completed_chunks: list[str] = field(default_factory=list)
    unresolved: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    required_fields: list[str] = field(default_factory=list)
    required_any_of: list[list[str]] = field(default_factory=list)
    actions: list[dict[str, Any]] = field(default_factory=list)
    budget: Budget = field(default_factory=Budget)
    error: str | None = None
    def next_action(self):
        if self.action_index >= len(self.actions): return None
        a=self.actions[self.action_index]; self.action_index += 1; return a
    def to_dict(self):
        d=asdict(self); d["status"]=self.status.value; d["budget"]=asdict(self.budget); return d
    @classmethod
    def from_dict(cls,d):
        x=dict(d); x["status"]=AgentStatus(x.get("status", "init")); x["budget"]=Budget(**x.get("budget",{})); return cls(**x)
