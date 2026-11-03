from __future__ import annotations
from dataclasses import dataclass, field
from .memory_point import MemoryPoint
from .anchor_schema import ContextAnchor
@dataclass
class SectionMemory:
    document_id: str
    section_id: str
    title: str = ""
    points: list[MemoryPoint] = field(default_factory=list)
    anchors: list[ContextAnchor] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    memory_version: int = 1
    def add(self, point: MemoryPoint):
        self.points.append(point)
        by_key={a.dedupe_key():a for a in self.anchors}
        for a in point.anchors:
            if a.dedupe_key() in by_key:
                old=by_key[a.dedupe_key()]; old.evidence_refs.extend(a.evidence_refs); old.confidence=max(old.confidence,a.confidence); old.related_anchor_ids=list(dict.fromkeys(old.related_anchor_ids+a.related_anchor_ids))
            else: self.anchors.append(a); by_key[a.dedupe_key()]=a
        self.conflicts.extend(x for x in point.conflicts if x not in self.conflicts); self.missing_fields.extend(x for x in point.missing_fields if x not in self.missing_fields)
    def to_dict(self): return {"document_id":self.document_id,"section_id":self.section_id,"title":self.title,"points":[p.to_dict() for p in self.points],"anchors":[a.to_dict() for a in self.anchors],"conflicts":self.conflicts,"missing_fields":self.missing_fields,"memory_version":self.memory_version}
