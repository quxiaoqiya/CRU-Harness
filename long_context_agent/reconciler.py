from __future__ import annotations
from reading_memory import DocumentMemory
class Reconciler:
    def reconcile(self, memory: DocumentMemory):
        return [a for a in memory.anchors if a.status in {"candidate","validated"} and a.confidence >= .5 and a.evidence_refs and not a.negated]
    def validate(self, memory, required_fields=(), required_any_of=()):
        present={a.attributes.get("field",a.anchor_type) for a in memory.anchors if a.status != "missing"}
        missing=[f for f in required_fields if f not in present and not any(f.casefold() in str(a.normalized_value).casefold() for a in memory.anchors)]
        groups=[group for group in required_any_of if not any(x in present for x in group)]
        return not missing and not groups, missing, groups
