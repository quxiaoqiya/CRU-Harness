"""Evidence-preserving hierarchical compaction."""
from __future__ import annotations
from .memory_point import MemoryPoint
from .document_memory import DocumentMemory
from .anchor_schema import EvidenceRef
class MemoryCompactor:
    def __init__(self, max_points: int = 256, max_summary_chars: int = 12000): self.max_points=max_points; self.max_summary_chars=max_summary_chars
    def compact_points(self, points: list[MemoryPoint]) -> MemoryPoint:
        if not points: return MemoryPoint(document_id="", summary="")
        anchors=[]; seen={}
        evidence=[]; uncertain=[]; conflicts=[]; missing=[]; summaries=[]
        for p in points:
            if p.summary: summaries.append(p.summary)
            evidence.extend(p.evidence_refs); uncertain.extend(p.uncertainty); conflicts.extend(p.conflicts); missing.extend(p.missing_fields)
            for a in p.anchors:
                if a.uncertainty:
                    uncertain.append(str(a.uncertainty))
                k=a.dedupe_key()
                if k not in seen: seen[k]=a; anchors.append(a)
                else:
                    seen[k].evidence_refs.extend(a.evidence_refs); seen[k].confidence=max(seen[k].confidence,a.confidence); seen[k].negated |= a.negated
        summary=" ".join(dict.fromkeys(summaries))[:self.max_summary_chars]
        return MemoryPoint(document_id=points[0].document_id, level="L2", summary=summary, anchors=anchors, evidence_refs=evidence, uncertainty=list(dict.fromkeys(uncertain)), conflicts=list(dict.fromkeys(conflicts)), missing_fields=list(dict.fromkeys(missing)))
    def compact_document(self, memory: DocumentMemory) -> DocumentMemory:
        # Keep anchors/evidence and unresolved metadata; drop duplicate chunk points from active representation.
        for sec in memory.sections.values():
            if len(sec.points)>self.max_points:
                merged=self.compact_points(sec.points); sec.points=[merged]
        memory.memory_version += 1; return memory
