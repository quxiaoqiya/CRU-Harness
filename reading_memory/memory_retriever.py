from __future__ import annotations
from .document_memory import DocumentMemory
class MemoryRetriever:
    def __init__(self, memory: DocumentMemory): self.memory=memory
    def search(self, query: str, limit: int = 10):
        q=query.casefold(); scored=[]
        for a in self.memory.anchors:
            text=(a.label+" "+str(a.normalized_value)+" "+" ".join(a.attributes.keys())).casefold(); score=sum(1 for t in q.split() if t in text)+a.salience*.1
            if score: scored.append((score,a))
        return [a for _,a in sorted(scored,key=lambda x:x[0],reverse=True)[:limit]]
    def by_entity(self, value, limit=20): return self.memory.query(entity=value)[:limit]
    def by_event(self, value, limit=20): return self.memory.query(event=value)[:limit]
    def by_field(self, value, limit=20): return self.memory.query(field=value)[:limit]
    def by_evidence(self, block_id): return self.memory.query(evidence=block_id)
    def unresolved(self): return self.memory.conflicts.unresolved()
    def missing(self): return self.memory.missing_fields
