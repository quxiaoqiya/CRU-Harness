import pytest
from long_context_agent.trajectory_memory_compact import render_source_range,restore_source_view,compact_evidence_part
from long_context_agent.adaptive_evidence import EvidenceWindow


def test_display_roundtrip_handles_quotes_indents_and_escaped_backslashes():
    raw=r'{"tree":"Root\n\tlink \"Observed\"","literal":"C:\\new\\test"}'
    view,edits=render_source_range(raw,0,len(raw))
    assert '\n\t' in view and 'link "Observed"' in view
    assert r'C:\\new\\test' in view
    assert restore_source_view(view,edits)==raw


@pytest.mark.parametrize('start',range(6))
def test_window_boundary_never_decodes_an_escape_from_outside(start):
    raw='abc'+r'\\\n\t"label"'+r'\\n'
    view,edits=render_source_range(raw,start,len(raw))
    assert restore_source_view(view,edits)==raw[start:]


def test_source_headers_keep_original_addresses_and_reject_invented_raw_text():
    source=r'prefix Root\n\tlink "Observed" suffix'
    w=EvidenceWindow(7,len(source)-7,source[7:-7],'q',1)
    text=w.render();view,receipts=compact_evidence_part(text,source)
    assert view.startswith(f'[{w.span} | gap q]')
    assert receipts[0]['source_span']==[7,len(source)-7] and receipts[0]['reversible']
    with pytest.raises(ValueError,match='not the original'):
        compact_evidence_part(text.replace('Root','Fake'),source)


def test_overlapping_ranges_merge_by_position_without_losing_source_union():
    source='abcdefghijklmnop'
    a=EvidenceWindow(0,10,source[:10],'left',1);b=EvidenceWindow(6,16,source[6:],'right',1)
    view,receipts=compact_evidence_part(a.render()+'\n\n'+b.render(),source)
    assert len(receipts)==1 and receipts[0]['source_span']==[0,16]
    assert receipts[0]['duplicate_source_chars_removed']==4
    assert receipts[0]['original_source_spans']==[[0,10],[6,16]]
    assert view.endswith(source)


def test_identical_text_at_distinct_positions_is_not_content_deduplicated():
    source='sameXsame'
    a=EvidenceWindow(0,4,source[:4],'one',1);b=EvidenceWindow(5,9,source[5:],'two',1)
    view,receipts=compact_evidence_part(a.render()+'\n\n'+b.render(),source)
    assert len(receipts)==2 and all(r['duplicate_source_chars_removed']==0 for r in receipts)
    assert [r['source_span'] for r in receipts]==[[0,4],[5,9]]


def test_review_is_unchanged_and_final_compaction_keeps_source_receipts(tmp_path):
    from long_context_agent.trajectory_memory import TrajectoryStore,TrajectoryEvidenceClient
    from long_context_agent.trajectory_memory_completion import PerCallBudgetedClient
    from long_context_agent.trajectory_memory_compact import CompactCompletionClient
    from long_context_agent.experiment_budget import ExperimentBudget
    from long_context_agent.adaptive_evidence import _messages
    class Tokenizer:
        model='test'
        def count(self,**kwargs):
            text=''.join(m['content'] if isinstance(m['content'],str) else ''.join(p.get('text','') for p in m['content']) for m in kwargs['messages'])
            return len(text)//4
    store=TrajectoryStore(tmp_path);store.insert({'id':'t','states':[{'accessibility_tree':('Root\n\tlink "Observed"\n'*400)}]});store.finalize()
    raw=PerCallBudgetedClient(Tokenizer(),ExperimentBudget(16384,65536,8,4096))
    c=CompactCompletionClient(raw,store,domain_system='official',question_text='q')
    base=TrajectoryEvidenceClient(raw,store,domain_system='official',question_text='q')
    w=EvidenceWindow(0,len(store.context),store.context,'q',1)
    review=_messages('q',[w]);assert c.prepare(review)==base.prepare(review)
    final,_=c.prepare(_messages('q',[w],final=True))
    assert c._display_receipts and '\n\tlink "Observed"' in final[-1]['content'][1]['text']
    assert all(r['reversible'] for v in c._display_receipts.values() for r in v['windows'])
