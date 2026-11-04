"""Offline public arithmetic/provenance verification, not semantic regrading."""
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


def main():
    data=json.loads((ROOT/'benchmarks/longmemeval_v2_visual_handoff_20261104.json').read_text())
    assert not data['official_leaderboard_submission'] and not data['same_compute_comparison']
    assert data['judge_substitution_from_official_default'] and not data['default_controller_changed']
    assert data['regression_test_count']==56
    for stage,ex in data['experiments'].items():
        rows=ex['rows'];n=20 if stage=='dev' else 8
        assert len({r['id'] for r in rows})==n
        assert len({(r['id'],r['arm']) for r in rows})==len(rows)
        assert len(rows)==(20 if stage=='dev' else 24)
        assert ex['offline_audit']['status']=='pass'
        assert ex['offline_audit']['replay_passed']==len(rows)
        assert ex['offline_audit']['provider_or_tokenizer_calls_made']==0
        for name,digest in ex['code_sha256'].items():
            assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest, name
        for arm,s in ex['summary'].items():
            group=[r for r in rows if r['arm']==arm]
            assert len(group)==s['n']==n
            assert sum(r['correct'] is True for r in group)==s['correct']
            assert s['task_success_rate_all_planned']==s['correct']/n
            assert sum(r['calls'] for r in group)==s['generation_calls']
            assert sum(r['charged_tokens'] for r in group)==s['charged_or_reserved_tokens']
        for r in rows:
            assert r['within_budget']
            assert r['charged_tokens'] <= (262144 if r['arm']=='direct_headtail_native' else 65536)
            assert r['calls'] <= (1 if r['arm']=='direct_headtail_native' else 8)
            assert 0<=r['new_selected_last_round_in_final']<=r['new_selected_last_round']
        for key,pair in ex.items():
            if key.startswith('paired_'):
                assert pair['n']==n
                assert sum(pair[k] for k in ('both_correct','new_gain','new_loss','both_wrong'))==n
        if stage=='holdout':
            assert ex['selection']['prior_question_count']==68
            assert ex['selection']['question_id_overlap_with_prior68']==0
            assert ex['selection']['normalized_question_hash_overlap_with_prior68']==0
            assert ex['selection']['query_image_sha_overlap_with_prior68']==0
    print('PASS: 44 outcomes; paired arithmetic, token/call budgets, frozen code and audit receipts. Not semantic regrading.')


if __name__=='__main__':main()
