"""Export aggregate/paired V3 evidence handoff results without raw observations."""
from collections import Counter
import json
from pathlib import Path
import re

from scripts import benchmark_longmemeval_v2_visual_handoff as b

ROOT = b.ROOT


def public_row(r):
    x = r['result']
    rounds = x.get('trace', [])
    latest = rounds[-1] if rounds else {}
    fresh = set(latest.get('fresh_ids', []))
    selected = set(latest.get('selected_ids', []))
    final = set(x.get('final_evidence_ids', []))
    refusal = bool(re.search(r"\b(?:i (?:do not|don't) know|cannot determine|insufficient evidence|unknown)\b",
                             r['parsed_answer'], re.I))
    return {'id':r['id'],'domain':r['domain'],'arm':r['arm'],'status':r['status'],
        'correct':r['correct'],'official_unknown':r['unknown'],
        'declared_abstention_by_text_heuristic':refusal,
        'scoring_function':r['scoring_function'], 'judge_error':r['judge_error'],
        'calls':r['usage']['physical_requests'],'charged_tokens':r['usage']['charged_tokens'],
        'within_budget':r['usage']['within_budget'], 'rounds':len(rounds),
        'new_selected_last_round':len(fresh & selected),
        'new_selected_last_round_in_final':len(fresh & selected & final),
        'selected_omitted':len(x.get('pinned_omitted', [])),
        'selected_visual_omitted':len(x.get('final_visual_omitted', [])),
        'final_required_gap_count':len(x.get('latest_review', {}).get('required_gaps', [])),
        'final_unresolved_conflict_count':len(x.get('latest_review', {}).get('conflicts', []))}


def pairs(rows, left, right):
    a = {r['id']:r for r in rows if r['arm'] == left}
    c = {r['id']:r for r in rows if r['arm'] == right}
    assert set(a) == set(c)
    return {'n':len(a),'both_correct':sum(a[k]['correct'] is True and c[k]['correct'] is True for k in a),
        'new_gain':sum(a[k]['correct'] is not True and c[k]['correct'] is True for k in a),
        'new_loss':sum(a[k]['correct'] is True and c[k]['correct'] is not True for k in a),
        'both_wrong':sum(a[k]['correct'] is not True and c[k]['correct'] is not True for k in a)}


def export():
    experiments = {}
    scoring_started, inference_completed = [], []
    for stage in ('dev','holdout'):
        work = ROOT/f'work/longmemeval_v2_visual_handoff_{stage}_20261104'
        freeze = b.check(work)
        audit = json.loads((work/'formal/audit.json').read_text())
        scoring_started.append(json.loads((work/'formal/scoring_access.json').read_text())['opened_at_unix'])
        inference_completed.extend(r['completed_at_unix'] for r in audit['scored'])
        offline = json.loads((work/'offline_audit.json').read_text())
        assert offline['status'] == 'pass'
        assert offline['terminal_records'] == offline['replay_passed'] == freeze['expected_terminal_records']
        rows = [public_row(r) for r in audit['scored']]
        experiments[stage] = {'freeze_sha256':b.digest(work/'formal/freeze.json'),
            'selection':json.loads((work/'selection_manifest.json').read_text()),
            'source_sha256':freeze['source_sha256'], 'code_sha256':freeze['code_sha256'],
            'summary':audit['summary'], 'rows':rows,
            'offline_audit':{k:v for k,v in offline.items() if k not in {'receipts'}},
            'offline_audit_sha256':b.digest(work/'offline_audit.json'),
            'correct_declared_abstentions':dict(Counter(r['arm'] for r in rows if r['correct'] and r['declared_abstention_by_text_heuristic']))}
    # Historical V1 predictions are never re-generated or re-judged to improve
    # the development comparison. They are explicitly a cached reference.
    old = []
    for directory in ('longmemeval_v2_visual_anchor_dev_20261103','longmemeval_v2_visual_anchor_holdout_20261103'):
        audit = json.loads((ROOT/'work'/directory/'formal/audit.json').read_text())
        old += [public_row(r) for r in audit['scored'] if r['arm']=='harness_visual_anchor']
    assert len(old)==20
    dev = experiments['dev']['rows']
    for r in old:r['arm']='historical_v1'
    experiments['dev']['historical_v1_summary']={'n':20,'correct':sum(r['correct'] is True for r in old),
        'charged_tokens':sum(r['charged_tokens'] for r in old),'cached_predictions_and_judges':True}
    experiments['dev']['paired_vs_historical_v1']=pairs(old+dev,'historical_v1','harness_visual_anchor')
    experiments['dev']['historical_v1_rows']=old
    rows = experiments['holdout']['rows']
    assert min(scoring_started) > max(inference_completed), 'Both inference groups must be fixed before any scoring'
    experiments['holdout']['paired_vs_v1']=pairs(rows,'harness_visual_anchor_v1','harness_visual_anchor')
    experiments['holdout']['paired_vs_direct']=pairs(rows,'direct_headtail_native','harness_visual_anchor')
    result={'schema_version':1,'date':'2026-11-04','model':b.MODEL,
        'upstream_code_commit':'2cc8c540bdb87fe6761629b585e727e1c4704520',
        'dataset_revision':'f152293e235517d504809563c833d7190b8c713b',
        'official_leaderboard_submission':False,'judge_substitution_from_official_default':True,
        'same_compute_comparison':False,'sources_reused_question_disjoint_only':True,
        'no_per_question_oracle_routing':True,'default_controller_changed':False,
        'both_inference_groups_fixed_before_any_scoring':True,
        'regression_test_count':56,
        'action_coverage':json.loads((ROOT/'work/longmemeval_v2_visual_handoff_dev_20261104/action_coverage_v3.json').read_text()),
        'experiments':experiments}
    target=ROOT/'benchmarks/longmemeval_v2_visual_handoff_20261104.json'
    b.atomic(target,result)
    print(json.dumps({s:experiments[s]['summary'] for s in experiments},indent=2))


if __name__=='__main__':export()
