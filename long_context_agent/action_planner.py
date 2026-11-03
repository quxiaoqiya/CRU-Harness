from __future__ import annotations
class ActionPlanner:
    def __init__(self, *, required_fields=None, required_any_of=None): self.required_fields=list(required_fields or []); self.required_any_of=[list(x) for x in (required_any_of or [])]
    def initial(self, document): return [{"type":"get_section","section_id":sid} for sid in _sections(document)]
    def next_actions(self, state, memory, document):
        actions=[]
        if memory.conflicts.unresolved(): actions += [{"type":"search_evidence","query":c.key} for c in memory.conflicts.unresolved()]
        if memory.missing_fields: actions += [{"type":"search_anchor","query":f} for f in memory.missing_fields]
        if not actions: actions += [{"type":"get_section","section_id":sid} for sid in _sections(document) if sid not in state.visited_sections]
        return actions

def _sections(document):
    sections=getattr(document,"sections",None) or []
    if isinstance(sections,dict): return list(sections.keys())
    return [getattr(s,"section_id",getattr(s,"id",str(i))) for i,s in enumerate(sections)]
