from __future__ import annotations
from reading_memory import DocumentMemory, MemoryStateStore
from .reading_state import ReadingState, AgentStatus
from .action_planner import ActionPlanner
from .context_builder import ContextBuilder
from .section_reader import SectionReader
from .reconciler import Reconciler
from .evidence_retriever import EvidenceRetriever
import time
class ReaderAgent:
    """Offline-first reader; checkpoints memory separately from product projection."""
    def __init__(self, parser=None, provider=None, state_store=None, *, max_context_chars=24000):
        self.parser=parser; self.provider=provider; self.state_store=state_store or MemoryStateStore(); self.context_builder=ContextBuilder(max_context_chars); self.reader=SectionReader(provider); self.reconciler=Reconciler(); self.state=None
    def read(self, source, *, required_fields=None, required_any_of=None, resume=True):
        doc=source if hasattr(source,"document_id") else self._parse(source); document_id=getattr(doc,"document_id","document")
        loaded=self.state_store.load(document_id) if resume else None
        if loaded:
            memory,state_data=loaded; state=ReadingState.from_dict(state_data)
            if state.status == AgentStatus.COMPLETE and memory.complete:
                self.state = state
                return doc, memory, state
            state.status=AgentStatus.READING
        else:
            memory=DocumentMemory(document_id,getattr(doc,"source_sha256","")); state=ReadingState(document_id,required_fields=list(required_fields or []),required_any_of=[list(x) for x in (required_any_of or [])]); state.actions=ActionPlanner(required_fields=required_fields,required_any_of=required_any_of).initial(doc); state.status=AgentStatus.INDEXED
        planner=ActionPlanner(required_fields=state.required_fields,required_any_of=state.required_any_of)
        evidence_retriever = EvidenceRetriever(doc, memory)
        started = time.monotonic()
        all_sections = {a.get("section_id") for a in state.actions if a.get("type") == "get_section"}
        for _ in range(256):
            valid,missing,groups=self.reconciler.validate(memory,state.required_fields,state.required_any_of); memory.missing_fields=list(dict.fromkeys(missing+[f for g in groups for f in g]))
            action=state.next_action()
            if action is None:
                state.actions.extend(planner.next_actions(state,memory,doc)); action=state.next_action()
            if action is None: state.status=AgentStatus.COMPLETE if valid else AgentStatus.PAUSED; break
            kind = action.get("type")
            if kind in {"search_anchor", "search_evidence"}:
                # Convert a retrieval request into bounded section reads. This
                # keeps unresolved/conflict-driven follow-up evidence-aware.
                hits = evidence_retriever.search(action.get("query", ""))
                section_ids = []
                for hit in hits:
                    sid = getattr(hit, "section_id", None)
                    if sid is None and hasattr(hit, "section_ids"):
                        section_ids.extend(getattr(hit, "section_ids", []))
                    elif sid:
                        section_ids.append(sid)
                for sid in dict.fromkeys(section_ids):
                    if sid not in state.visited_sections:
                        state.actions.insert(state.action_index, {"type": "get_section", "section_id": sid})
                continue
            if kind in {"inspect_figure", "inspect_table"}:
                # Captions/table rows are already represented in the IR. Add a
                # bounded synthetic section action for the relevant evidence.
                target = action.get("figure_id") or action.get("table_id")
                for block in getattr(doc, "blocks", []) or []:
                    refs = getattr(block, "figure_refs", []) + getattr(block, "table_refs", [])
                    if target and target in refs and getattr(block, "block_id", "") not in state.completed_chunks:
                        state.actions.insert(state.action_index, {"type": "get_block_window", "section_id": getattr(block, "section_id", "")})
                        break
                continue
            if kind in {"resolve_entity", "reconcile_experiment", "validate_candidate", "commit_product_projection"}:
                continue
            if kind not in {"get_section","get_block_window"}: continue
            if not state.budget.consume_chunk():
                state.status=AgentStatus.PAUSED; state.error="chunk_budget_exhausted"; break
            # A local deterministic reader does not consume model-call budget.
            # Provider-backed section reading consumes one model call here.
            if self.provider is not None and not state.budget.consume(vision=kind.startswith("inspect_")):
                state.status=AgentStatus.PAUSED; state.error="model_budget_exhausted"; break
            if time.monotonic() - started > state.budget.max_seconds:
                state.status=AgentStatus.PAUSED; state.error="time_budget_exhausted"; break
            sid=action.get("section_id","unsectioned")
            if sid in state.visited_sections: continue
            text=self.context_builder.build(doc,action,memory); point=self.reader.read(document_id,sid,text,source_sha256=getattr(doc,"source_sha256",""),fields=state.required_fields); memory.add_point(point,sid); state.completed_chunks.append(point.chunk_id); state.visited_sections.append(sid); state.status=AgentStatus.READING; self.state_store.save(memory,state.to_dict())
            if valid and not memory.conflicts.unresolved() and not memory.missing_fields and all_sections.issubset(set(state.visited_sections)): state.status=AgentStatus.COMPLETE; memory.complete=True; break
        if state.status == AgentStatus.COMPLETE:
            memory.complete = True
        self.state_store.save(memory,state.to_dict()); self.state=state; return doc,memory,state
    def _parse(self, source):
        if self.parser is None:
            from independent_ingestion import parse
            return parse(source)
        return self.parser.parse(source)
    def product_projection(self, memory, *, required_fields=()):
        valid,missing,_=self.reconciler.validate(memory,required_fields,())
        if not valid: raise ValueError(f"projection blocked by missing fields: {missing}")
        from product_projection import project_document_memory
        result = project_document_memory(memory)
        if not result.ready:
            raise ValueError(f"projection blocked: {list(result.blockers)}")
        return result.for_duckdb()
