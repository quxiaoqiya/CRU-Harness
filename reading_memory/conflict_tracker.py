from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import hashlib
from .anchor_schema import ContextAnchor, EvidenceRef
@dataclass
class Conflict:
    conflict_id: str; key: str; values: list[Any] = field(default_factory=list); evidence_refs: list[EvidenceRef] = field(default_factory=list); status: str = "open"; note: str = ""
class ConflictTracker:
    def __init__(self): self._items: dict[str, Conflict] = {}; self._observed: dict[str, dict[str, tuple[Any,list[EvidenceRef]]]] = {}
    def observe(self, anchor: ContextAnchor) -> Conflict | None:
        # Group by semantic label/field, then compare normalized values. This preserves disagreement evidence.
        group = f"{anchor.anchor_type}:{str(anchor.attributes.get('field', anchor.label)).strip().casefold()}"
        value = anchor.normalized_value if anchor.normalized_value is not None else anchor.label
        digest = repr(value).casefold(); vals = self._observed.setdefault(group, {})
        if digest not in vals: vals[digest] = (value, list(anchor.evidence_refs))
        else: vals[digest][1].extend(anchor.evidence_refs)
        if len(vals) < 2 and anchor.anchor_type != "conflict" and not anchor.attributes.get("conflict"): return None
        key=group; cid="conf_"+hashlib.sha256(key.encode()).hexdigest()[:20]; c=self._items.setdefault(cid, Conflict(cid,key))
        c.values=list(v[0] for v in vals.values()); c.evidence_refs=[]
        for _,(_,refs) in vals.items(): c.evidence_refs.extend(refs)
        if anchor.anchor_type == "conflict" or anchor.attributes.get("conflict"): c.values.append(value)
        return c
    def add(self, key, values, evidence_refs=None, note=""):
        cid="conf_"+hashlib.sha256(key.encode()).hexdigest()[:20]; c=self._items.setdefault(cid,Conflict(cid,key)); c.values.extend(v for v in values if v not in c.values); c.evidence_refs.extend(evidence_refs or []); c.note=note or c.note; return c
    def unresolved(self): return [c for c in self._items.values() if c.status != "resolved"]
    def resolve(self, conflict_id):
        if conflict_id in self._items: self._items[conflict_id].status="resolved"
    def to_dict(self): return {k:{**vars(v),"evidence_refs":[e.to_dict() for e in v.evidence_refs]} for k,v in self._items.items()}
    @classmethod
    def from_dict(cls,data):
        x=cls()
        for k,v in (data or {}).items():
            refs=[e if isinstance(e,EvidenceRef) else EvidenceRef(**e) for e in v.get("evidence_refs",[])]; x._items[k]=Conflict(v["conflict_id"],v["key"],v.get("values",[]),refs,v.get("status","open"),v.get("note",""))
        return x
