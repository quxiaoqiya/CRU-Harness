from dataclasses import dataclass
from reading_memory import ContextAnchor, DocumentMemory, EvidenceRef, MemoryCompactor, MemoryPoint, MemoryStateStore
from long_context_agent import ReaderAgent, AgentStatus
@dataclass
class Block: block_id: str; text: str
@dataclass
class Section: section_id: str; block_ids: list[str]
@dataclass
class Doc:
    document_id: str; source_sha256: str; sections: list[Section]; blocks: list[Block]
def test_anchor_dedup_and_evidence_preserved():
    e=EvidenceRef("b1",page=1); a=ContextAnchor("entity","A",normalized_value="a",document_id="d",evidence_refs=[e]); m=DocumentMemory("d")
    assert m.add_point(MemoryPoint("d",chunk_id="c1",anchors=[a]))
    assert not m.add_point(MemoryPoint("d",chunk_id="c1",anchors=[a]))
    assert m.anchors[0].evidence_refs[0].block_id == "b1"
def test_reader_checkpoint_and_all_sections(tmp_path):
    d=Doc("d", "f"*64,[Section("s1",["b1"]),Section("s2",["b2"])],[Block("b1","Material A"),Block("b2","Result stable")])
    _,mem,state=ReaderAgent(state_store=MemoryStateStore(tmp_path)).read(d)
    assert state.status == AgentStatus.COMPLETE
    assert set(state.visited_sections)=={"s1","s2"}
    assert MemoryStateStore(tmp_path).load("d") is not None
def test_compactor_keeps_evidence():
    e=EvidenceRef("b"); p=MemoryPoint("d",anchors=[ContextAnchor("entity","A",document_id="d",evidence_refs=[e])])
    c=MemoryCompactor().compact_points([p,p])
    assert c.evidence_refs and c.evidence_refs[0].block_id == "b"


def test_checkpoint_cleanup_is_local_only(tmp_path):
    store = MemoryStateStore(tmp_path)
    store.save(DocumentMemory("ttl-doc"))
    path = store.path("ttl-doc")
    path.touch()
    assert store.cleanup(0, now=path.stat().st_mtime + 1) == 1
    assert not path.exists()
