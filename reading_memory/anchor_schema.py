"""Stable, evidence-linked context anchors. Domain fields are open-world."""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Iterable, Mapping
import hashlib, json
from .anchor_types import AnchorType

def stable_anchor_id(document_id: str, anchor_type: str, normalized_value: Any, section_ids: Iterable[str] = ()) -> str:
    payload = json.dumps([document_id, str(anchor_type), normalized_value, sorted(section_ids)], ensure_ascii=False, sort_keys=True, default=str)
    return "anc_" + hashlib.sha256(payload.encode()).hexdigest()[:24]

@dataclass
class EvidenceRef:
    block_id: str
    page: int | None = None
    start: int | None = None
    end: int | None = None
    token_start: int | None = None
    token_end: int | None = None
    figure_id: str | None = None
    table_id: str | None = None
    source_sha256: str | None = None
    parser_version: str | None = None
    def key(self) -> tuple: return (self.block_id, self.page, self.start, self.end, self.figure_id, self.table_id)
    def to_dict(self): return asdict(self)

@dataclass
class ContextAnchor:
    anchor_type: str
    label: str
    normalized_value: Any = None
    document_id: str = ""
    anchor_id: str = ""
    related_anchor_ids: list[str] = field(default_factory=list)
    section_ids: list[str] = field(default_factory=list)
    evidence_refs: list[EvidenceRef] = field(default_factory=list)
    confidence: float = 0.0
    uncertainty: str | None = None
    salience: float = 0.5
    status: str = "candidate"
    memory_version: int = 1
    negated: bool = False
    attributes: dict[str, Any] = field(default_factory=dict)
    def __post_init__(self):
        self.anchor_type = str(self.anchor_type.value if isinstance(self.anchor_type, AnchorType) else self.anchor_type)
        if not self.anchor_id:
            self.anchor_id = stable_anchor_id(self.document_id, self.anchor_type, self.normalized_value if self.normalized_value is not None else self.label, self.section_ids)
        self.confidence = max(0.0, min(1.0, float(self.confidence)))
        self.salience = max(0.0, min(1.0, float(self.salience)))
    def dedupe_key(self) -> tuple: return (self.anchor_type, str(self.normalized_value if self.normalized_value is not None else self.label).strip().casefold())
    def to_dict(self):
        d = asdict(self); d["evidence_refs"] = [e.to_dict() for e in self.evidence_refs]; return d
    @classmethod
    def from_dict(cls, data: Mapping[str, Any]):
        d = dict(data); d["evidence_refs"] = [e if isinstance(e, EvidenceRef) else EvidenceRef(**e) for e in d.get("evidence_refs", [])]; return cls(**d)
