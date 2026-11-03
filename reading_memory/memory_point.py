"""Chunk-level memory records; never a product record."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable
import hashlib, json
from .anchor_schema import ContextAnchor, EvidenceRef

@dataclass
class MemoryPoint:
    document_id: str
    level: str = "L1"
    chunk_id: str = ""
    summary: str = ""
    anchors: list[ContextAnchor] = field(default_factory=list)
    relations: list[dict[str, Any]] = field(default_factory=list)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    uncertainty: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    source_hash: str = ""
    memory_version: int = 1
    def __post_init__(self):
        if not self.chunk_id:
            body = json.dumps([self.document_id, self.level, self.summary, [a.to_dict() for a in self.anchors]], sort_keys=True, default=str)
            self.chunk_id = "mem_" + hashlib.sha256(body.encode()).hexdigest()[:24]
        if not self.source_hash:
            self.source_hash = hashlib.sha256(self.summary.encode()).hexdigest()
        self.evidence_refs = _merge_evidence(self.evidence_refs + [e for a in self.anchors for e in a.evidence_refs])
    def to_dict(self):
        d = asdict(self); d["anchors"] = [a.to_dict() for a in self.anchors]; d["evidence_refs"] = [e.to_dict() for e in self.evidence_refs]; return d
    @classmethod
    def from_dict(cls, d):
        x = dict(d); x["anchors"] = [ContextAnchor.from_dict(a) for a in x.get("anchors", [])]; x["evidence_refs"] = [EvidenceRef(**e) if not isinstance(e, EvidenceRef) else e for e in x.get("evidence_refs", [])]; return cls(**x)

def _merge_evidence(refs: Iterable[EvidenceRef]) -> list[EvidenceRef]:
    out=[]; seen=set()
    for e in refs:
        k=e.key()
        if k not in seen: seen.add(k); out.append(e)
    return out
