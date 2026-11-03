from __future__ import annotations
from dataclasses import dataclass, field
from .section_memory import SectionMemory
from .memory_point import MemoryPoint
from .anchor_schema import ContextAnchor
from .conflict_tracker import ConflictTracker
@dataclass
class DocumentMemory:
    document_id: str
    source_sha256: str = ""
    sections: dict[str, SectionMemory] = field(default_factory=dict)
    anchors: list[ContextAnchor] = field(default_factory=list)
    conflicts: ConflictTracker = field(default_factory=ConflictTracker)
    missing_fields: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)
    completed_chunks: set[str] = field(default_factory=set)
    memory_version: int = 1
    # A private completion marker.  It is set only after the reader reaches
    # validation; product_projection uses it as a fail-closed gate.
    complete: bool = False
    def add_point(self, point: MemoryPoint, section_id: str = "") -> bool:
        if point.chunk_id in self.completed_chunks: return False
        sid=section_id or (point.anchors[0].section_ids[0] if point.anchors and point.anchors[0].section_ids else "unsectioned")
        sec=self.sections.setdefault(sid, SectionMemory(self.document_id,sid))
        sec.add(point); self.completed_chunks.add(point.chunk_id)
        by_key={a.dedupe_key():a for a in self.anchors}
        for a in point.anchors:
            if a.dedupe_key() in by_key:
                old=by_key[a.dedupe_key()]; old.evidence_refs.extend(a.evidence_refs); old.confidence=max(old.confidence,a.confidence); old.negated = old.negated or a.negated
            else: self.anchors.append(a); by_key[a.dedupe_key()]=a
            self.conflicts.observe(a)
        self.missing_fields=list(dict.fromkeys(self.missing_fields+point.missing_fields)); self.pending_tasks=list(dict.fromkeys(self.pending_tasks+point.conflicts)); return True
    def query(self, *, entity=None, event=None, field=None, evidence=None, conflict=False, missing=None):
        vals=self.anchors
        if entity: vals=[a for a in vals if a.anchor_type in {"entity","material","sample","device"} and str(entity).casefold() in (a.label+" "+str(a.normalized_value)).casefold()]
        if event: vals=[a for a in vals if a.anchor_type in {"event","synthesis","preparation","test","characterization"} and str(event).casefold() in (a.label+" "+str(a.normalized_value)).casefold()]
        if field: vals=[a for a in vals if a.attributes.get("field")==field or a.anchor_type==field]
        if evidence: vals=[a for a in vals if any(e.block_id==evidence for e in a.evidence_refs)]
        if conflict: return self.conflicts.unresolved()
        if missing: return [a for a in vals if a.status=="missing"]
        return vals
    def to_dict(self): return {"document_id":self.document_id,"source_sha256":self.source_sha256,"sections":{k:v.to_dict() for k,v in self.sections.items()},"anchors":[a.to_dict() for a in self.anchors],"conflicts":self.conflicts.to_dict(),"missing_fields":self.missing_fields,"pending_tasks":self.pending_tasks,"completed_chunks":sorted(self.completed_chunks),"memory_version":self.memory_version,"complete":self.complete}
