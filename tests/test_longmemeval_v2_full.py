import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import benchmark_longmemeval_v2_full as full


def test_full_job_plan_has_no_duplicate_physical_jobs():
    rows=[{'id':str(i),'image':'current.png' if i<29 else None} for i in range(451)]
    jobs=full.physical_jobs(rows)
    assert len(jobs)==len({(q['id'],a) for q,a in jobs})==931
    assert sum(a=='direct_headtail_native' for _,a in jobs)==451
    assert sum(a=='harness_visual_anchor' for _,a in jobs)==451
    assert sum(a=='harness_visual_anchor_v1' for _,a in jobs)==29
    assert full.physical_jobs(rows)==jobs
    assert full.CHANNELS['direct']+full.CHANNELS['harness']==6


def test_alias_is_explicit_immutable_and_disallowed_for_visual(tmp_path):
    record={'id':'q','arm':'harness_visual_anchor','question_image':False,
            'ledger':[{'charged_tokens':123}],'answer':'original'}
    source=tmp_path/'q__harness_visual_anchor.json';source.write_text(json.dumps(record))
    alias=full.alias_record(record,source)
    assert alias['arm']=='harness_visual_anchor_v1'
    assert alias['shared_physical_record']['additional_generation_calls']==0
    assert alias['shared_physical_record']['additional_tokens']==0
    assert alias['shared_physical_record']['sha256']==full.b.digest(source)
    alias['ledger'][0]['charged_tokens']=9
    assert record['ledger'][0]['charged_tokens']==123
    with pytest.raises(ValueError):full.alias_record({**record,'question_image':True},source)


def test_both_no_image_apis_have_identical_completion_fallback(monkeypatch):
    import long_context_agent.visual_anchor_memory as v1
    import long_context_agent.visual_anchor_memory_v3 as v3
    calls=[]
    def adapt(*args,**kwargs):
        return {'args':args,'kwargs':kwargs}
    def run(store,question,client):
        calls.append((store,question,client));return {'answer':'same'}
    for module in (v1,v3):
        monkeypatch.setattr(module,'CompletionEvidenceClient',adapt)
        monkeypatch.setattr(module,'run_completion_harness',run)
    index=SimpleNamespace(store='same store',anchor_config=v1.AnchorConfig())
    for module in (v1,v3):
        assert module.run_visual_anchor(index,'same question','same budget',domain_system='same policy',query_image=None)=={'answer':'same'}
    assert calls[0]==calls[1]


def test_failure_and_unknown_remain_in_planned_denominator():
    rows=[{'correct':True,'unknown':False,'judge_error':None,'status':'returned','usage':{'charged_tokens':100,'physical_requests':1}},
          {'correct':False,'unknown':True,'judge_error':None,'status':'returned','usage':{'charged_tokens':90,'physical_requests':1}},
          {'correct':None,'unknown':False,'judge_error':'parse','status':'error','usage':{'charged_tokens':80,'physical_requests':1}}]
    s=full.summarize(rows)
    assert s['n']==3 and s['correct']==1 and s['accuracy_all_planned']==1/3
    assert s['unknown']==s['judge_failures']==1
    assert s['method_assigned_tokens']==270
