"""Verify public full-run arithmetic and source fingerprints, offline."""
from collections import Counter
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    x=json.loads((ROOT/'benchmarks/longmemeval_v2_full_20261104.json').read_text());rows=x['rows']
    assert x['n_questions']==451 and len(rows)==x['logical_records']==1353
    assert len({(r['id'],r['arm']) for r in rows})==1353
    assert len({r['id'] for r in rows})==451 and x['haystack_tier']=='small'
    assert sum(r['physical_compute_shared'] for r in rows)==422
    physical=[r for r in rows if not r['physical_compute_shared']]
    assert len(physical)==x['physical_records']==931
    assert sum(r['method_assigned_tokens'] for r in physical)==x['physical_tokens']
    assert sum(r['method_assigned_generations'] for r in physical)==x['physical_generations']
    assert x['offline_audit']['terminal_records']==1353
    assert not x['offline_audit']['violations'] and x['offline_audit']['provider_or_tokenizer_calls_made']==0
    for name,expected in x['code_sha256'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==expected,name
    for arm,summary in x['summary'].items():
        group=[r for r in rows if r['arm']==arm];s=summary['overall']
        assert len(group)==s['n']==451
        assert sum(r['correct'] is True for r in group)==s['correct']
        assert s['accuracy_all_planned']==s['correct']/451
        assert sum(r['method_assigned_tokens'] for r in group)==s['method_assigned_tokens']
        assert Counter(r['question_type'] for r in group)==Counter({k:v['n'] for k,v in summary['by_question_type'].items()})
        assert summary['by_cohort']['unused_question']['n']==375
        assert summary['by_cohort']['previously_consumed']['n']==76
        assert summary['visual']['n']==29 and summary['no_query_image']['n']==422
        for r in group:
            assert r['within_budget']
            assert r['method_assigned_tokens']<=(262144 if arm=='direct_headtail_native' else 65536)
            assert r['method_assigned_generations']<=(1 if arm=='direct_headtail_native' else 8)
    by={(r['id'],r['arm']):r for r in rows}
    for r in rows:
        if r['physical_compute_shared']:
            assert r['arm']=='harness_visual_anchor_v1' and not r['question_image']
            origin=by[r['id'],'harness_visual_anchor']
            assert r['correct'] is origin['correct'] and r['method_assigned_tokens']==origin['method_assigned_tokens']
    for key in ('paired_vs_v1','paired_vs_direct'):
        pair=x[key];assert pair['n']==451
        assert sum(pair[k] for k in ('both_correct','new_gain','new_loss','both_wrong'))==451
    print('PASS: 451 questions, 1353 logical results, 931 physical tasks, 422 explicit shared paths; arithmetic/budgets/source integrity only.')


if __name__=='__main__':main()
