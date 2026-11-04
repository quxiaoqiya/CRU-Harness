from pathlib import Path
from types import SimpleNamespace
import pytest

from scripts import benchmark_longmemeval_v2 as bench
from long_context_agent.experiment_budget import ProviderTransport


def test_scoring_cannot_open_labels_before_all_registered_terminals(tmp_path,monkeypatch):
    monkeypatch.setattr(bench,'check',lambda work:{})
    monkeypatch.setattr(bench,'read_rows',lambda path:[{'id':str(i)} for i in range(20)])
    original=Path.read_text
    def guarded(path,*args,**kwargs):
        if 'private_labels' in str(path):raise AssertionError('Gold opened before completion')
        return original(path,*args,**kwargs)
    monkeypatch.setattr(Path,'read_text',guarded)
    with pytest.raises(ValueError,match='All sixty terminals'):
        bench.score(tmp_path,{})


def test_headtail_preserves_exact_raw_spans_and_full_direct_is_lossless():
    store=SimpleNamespace(context='0123456789'*100)
    windows=bench.headtail_windows(store,101)
    assert sum(w.end-w.start for w in windows)==101
    assert all(w.text==store.context[w.start:w.end] for w in windows)
    assert windows[0].start==0 and windows[-1].end==len(store.context)
    whole=bench.headtail_windows(store,10000)
    assert len(whole)==1 and whole[0].text==store.context


def test_wrong_provider_model_is_not_silently_accepted(monkeypatch):
    monkeypatch.setattr(ProviderTransport,'request',lambda *args,**kwargs:{
        'model':'different-model','usage':{'prompt_tokens':100,'completion_tokens':3},'choices':[{'finish_reason':'stop'}]})
    transport=bench.AuditedTransport(base_url='http://example.test/v1',api_key='test',model=bench.MODEL)
    with pytest.raises(ValueError,match='model identity differs'):
        transport.request('/chat/completions',{})
    assert transport.response_events[-1]['actual_model']=='different-model'
