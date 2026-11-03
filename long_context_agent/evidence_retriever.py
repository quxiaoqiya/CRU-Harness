from __future__ import annotations
from reading_memory import MemoryRetriever
class EvidenceRetriever:
    def __init__(self, document, memory): self.document=document; self.memory=memory; self.retriever=MemoryRetriever(memory)
    def search(self, query, limit=8):
        hits=self.retriever.search(query,limit)
        if hits: return hits
        q=str(query).casefold(); return [b for b in (getattr(self.document,"blocks",[]) or []) if q in getattr(b,"text",str(b)).casefold()][:limit]
