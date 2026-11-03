from __future__ import annotations
class ContextBuilder:
    def __init__(self, max_chars=24000): self.max_chars=max_chars
    def build(self, document, action, memory):
        sid=action.get("section_id"); blocks=getattr(document,"blocks",[]) or []
        if sid:
            sections=getattr(document,"sections",{}) or {}; sec=sections.get(sid) if isinstance(sections,dict) else next((s for s in sections if getattr(s,"section_id",getattr(s,"id",None))==sid),None)
            ids=set(getattr(sec,"block_ids",[]) or []) if sec else set()
            if ids: blocks=[b for b in blocks if getattr(b,"block_id",getattr(b,"id",None)) in ids]
        return "\n".join(getattr(b,"text",str(b)) for b in blocks)[:self.max_chars]
