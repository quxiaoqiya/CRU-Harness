from __future__ import annotations
import re
from reading_memory import ContextAnchor, EvidenceRef, MemoryPoint
class SectionReader:
    def __init__(self, provider=None): self.provider=provider
    def read(self, document_id, section_id, text, *, source_sha256="", fields=None):
        fields=fields or []
        # Providers are optional. If supplied, accept only structured data and retain
        # deterministic evidence refs; malformed responses fall back to local parsing.
        if self.provider is not None and hasattr(self.provider, "complete"):
            try:
                response = self.provider.complete("text_reader", text, schema={"type":"object"})
                value = getattr(response, "value", response)
                if isinstance(value, dict):
                    anchors=[]
                    for item in value.get("anchors", []):
                        if not isinstance(item, dict): continue
                        item=dict(item); item.setdefault("document_id", document_id); item.setdefault("section_ids", [section_id]); item.setdefault("confidence", .5); item.setdefault("anchor_type", "entity"); item.setdefault("label", str(item.get("normalized_value", "")) or "unlabelled")
                        item["evidence_refs"]=[EvidenceRef(**e) if isinstance(e,dict) else e for e in item.get("evidence_refs", [])]
                        if not item["evidence_refs"]:
                            item["evidence_refs"]=[EvidenceRef(block_id=f"{section_id}:model", start=0, end=len(text), source_sha256=source_sha256)]
                        anchors.append(ContextAnchor(**item))
                    return MemoryPoint(document_id=document_id,level="L1",chunk_id=f"{document_id}:{section_id}",summary=str(value.get("summary", ""))[:1000],anchors=anchors,conflicts=list(value.get("conflicts", [])),missing_fields=list(value.get("missing_fields", [])),source_hash=source_sha256)
            except Exception:
                pass
        anchors=[]
        for m in re.finditer(r"\b(?:[A-Z][A-Za-z0-9-]{2,}|[A-Z][a-z]+\s+[A-Z][a-z]+)\b", text):
            label=m.group(0)
            if label.lower() in {"The","This","Figure","Table","However"}: continue
            ev=EvidenceRef(block_id=f"{section_id}:span:{m.start()}:{m.end()}",start=m.start(),end=m.end(),source_sha256=source_sha256)
            anchors.append(ContextAnchor(anchor_type="entity",label=label,normalized_value=label,document_id=document_id,section_ids=[section_id],evidence_refs=[ev],confidence=.35,salience=.25))
        for field in fields:
            if field.casefold() not in text.casefold(): anchors.append(ContextAnchor(anchor_type="missing",label=field,document_id=document_id,section_ids=[section_id],status="missing",attributes={"field":field}))
        return MemoryPoint(document_id=document_id,level="L1",chunk_id=f"{document_id}:{section_id}",summary=" ".join(text.split())[:1000],anchors=anchors,missing_fields=[a.label for a in anchors if a.status=="missing"],source_hash=source_sha256)
