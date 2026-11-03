from __future__ import annotations
import json, os, tempfile, time
from pathlib import Path
from .document_memory import DocumentMemory
from .anchor_schema import ContextAnchor
from .section_memory import SectionMemory
from .memory_point import MemoryPoint
from .conflict_tracker import ConflictTracker
class MemoryStateStore:
    """Atomic local checkpoint store; deliberately separate from product databases."""
    def __init__(self, root: str | os.PathLike = ".memory_checkpoints"): self.root=Path(root); self.root.mkdir(parents=True,exist_ok=True)
    def path(self, document_id): return self.root/(document_id+".json")
    def save(self, memory: DocumentMemory, state: dict | None = None):
        data={"memory":memory.to_dict(),"state":state or {}}; p=self.path(memory.document_id); fd,tmp=tempfile.mkstemp(dir=self.root,prefix=p.name,suffix=".tmp")
        with os.fdopen(fd,"w",encoding="utf-8") as f: json.dump(data,f,ensure_ascii=False,indent=2)
        os.replace(tmp,p); return p
    def load(self, document_id):
        p=self.path(document_id)
        if not p.exists(): return None
        with p.open(encoding="utf-8") as f: data=json.load(f)
        md=data["memory"]; m=DocumentMemory(document_id=md["document_id"],source_sha256=md.get("source_sha256", ""),memory_version=md.get("memory_version",1))
        m.anchors=[ContextAnchor.from_dict(a) for a in md.get("anchors",[])]
        m.sections={sid:SectionMemory(document_id,sid,sec.get("title", ""),[MemoryPoint.from_dict(x) for x in sec.get("points",[])],[ContextAnchor.from_dict(a) for a in sec.get("anchors",[])],sec.get("conflicts",[]),sec.get("missing_fields",[]),sec.get("memory_version",1)) for sid,sec in md.get("sections",{}).items()}
        m.conflicts=ConflictTracker.from_dict(md.get("conflicts",{})); m.missing_fields=md.get("missing_fields",[]); m.pending_tasks=md.get("pending_tasks",[]); m.completed_chunks=set(md.get("completed_chunks",[])); m.complete=bool(md.get("complete", False)); return m,data.get("state",{})
    def delete(self, document_id): self.path(document_id).unlink(missing_ok=True)
    def cleanup(self, ttl_seconds: float, *, now: float | None = None) -> int:
        """Remove only expired checkpoint files; product stores are untouched."""
        cutoff = (time.time() if now is None else now) - max(0.0, float(ttl_seconds))
        removed = 0
        for path in self.root.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except FileNotFoundError:
                pass
        return removed
