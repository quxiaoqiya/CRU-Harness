"""Verify public LongMemEval-V2 arithmetic, budgets and fingerprints offline.

Standard library only. This does not reconstruct unpublished answers or prove
semantic entailment. The separately hashed local audit replays private records.
"""
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
ID = 'longmemeval_v2_qwen_20261103'
ARMS = ('direct_full', 'direct_headtail_native', 'harness')
FORBIDDEN = {'question', 'answer', 'gold', 'labels', 'content', 'reasoning',
             'trace', 'facts', 'evidence_text', 'authorization', 'api_key', 'headers'}
PRIVATE = re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}|(?i:Bearer\s+\S+)|'
                     r'/(?:Users|home|root|private|uniparser|mnt)/|data:image/|'
                     r'\b192\.168\.\d+\.\d+\b')


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan(value):
    if isinstance(value, dict):
        check(not FORBIDDEN.intersection(value), 'Raw-content key in public metadata')
        for key, item in value.items():
            if key.endswith('_sha256') and isinstance(item, str):
                check(re.fullmatch('[a-f0-9]{64}', item) is not None, 'Malformed SHA-256')
            scan(item)
    elif isinstance(value, list):
        for item in value:
            scan(item)
    elif isinstance(value, str):
        check(not PRIVATE.search(value), 'Private material in public metadata')


def ledger(entries, budget, declared):
    total = inp = out = unknown = 0
    check(len(entries) <= budget['max_calls'], 'Generation call ceiling exceeded')
    for number, item in enumerate(entries, 1):
        check(item['physical_request'] == number, 'Ledger order mismatch')
        admission, output = item['admission_prompt_tokens'], item['requested_output_tokens']
        check(type(admission) is int and admission >= 0, 'Invalid token admission')
        check(type(output) is int and 0 < output <= budget['max_output_tokens'], 'Output ceiling exceeded')
        check(admission + output <= budget['context_tokens'], 'Request budget exceeded')
        check(total + admission + output <= budget['total_tokens'], 'Task reservation exceeded')
        if item['usage_known']:
            i, o = item['known_prompt_tokens'], item['known_completion_tokens']
            check(type(i) is int and 0 <= i <= admission, 'Input usage exceeds admission')
            check(type(o) is int and 0 <= o <= output, 'Output usage exceeds reservation')
            check(item['charged_tokens'] == i + o, 'Known-usage conservation failed')
            inp += i
            out += o
        else:
            check(item['charged_tokens'] == admission + output, 'Unknown usage not fully reserved')
            unknown += 1
        total += item['charged_tokens']
    check(total <= budget['total_tokens'], 'Task ceiling exceeded')
    check(declared == {'physical_generations': len(entries), 'known_input_tokens': inp,
                       'known_output_tokens': out, 'charged_or_reserved_tokens': total,
                       'unknown_usage_requests': unknown}, 'Usage aggregate mismatch')


def main():
    p = json.loads((ROOT/'benchmarks'/f'{ID}.json').read_text())
    a_path = ROOT/'benchmarks'/f'{ID}_audit.json'
    a = json.loads(a_path.read_text())
    scan(p)
    scan(a)
    check(p['n_questions'] == 20 and len(set(p['sample_ids'])) == 20, 'Question coverage mismatch')
    check((p['n_unique_haystacks'], p['n_unique_trajectories'], p['n_history_screenshots'],
           p['n_question_screenshots']) == (2, 200, 5095, 4), 'Pilot scope mismatch')
    check(p['official_code_commit'] == '2cc8c540bdb87fe6761629b585e727e1c4704520', 'Code pin differs')
    check(p['data_revision'] == 'f152293e235517d504809563c833d7190b8c713b', 'Data pin differs')
    check(p['freeze_sha256'] == 'c9f2d218c783f4f6e71d0278c898007d3469bb34019703e77424fb14fe6cec25', 'Freeze differs')
    check(p['model'] == 'Qwen3.8-Flash-Next-FP8', 'Model mismatch')
    check(p['official_leaderboard_submission'] is False and p['official_LAFS_score_measured'] is False,
          'Official result scope changed')
    check(p['same_compute_comparison'] is False, 'Unequal compute not disclosed')
    check(p['all60_fixed_before_scoring_label_access'] and p['first_scored_at_unix'] > p['frozen_at_unix'], 'Gold gate mismatch')
    check(len(p['code_sha256']) == 49, 'Frozen code set differs')
    for path, expected in p['code_sha256'].items():
        check(digest(ROOT/path) == expected, 'Frozen source changed: '+path)
    check(digest(ROOT/'scripts/audits/audit_longmemeval_v2.py') == p['auditor_sha256'], 'Auditor changed')
    check(digest(a_path) == p['offline_audit_sha256'], 'Offline audit changed')
    check(a['auditor_sha256'] == p['auditor_sha256'] and a['freeze_sha256'] == p['freeze_sha256'], 'Audit binding failed')
    check(a['status'] == 'pass' and a['terminal_records'] == a['ledgers_passed'] == a['scores_recomputed'] == a['trajectory_passed'] == 60,
          'Incomplete audit')
    check(a['trajectory_unavailable_or_mismatched'] == a['provider_or_tokenizer_calls_made'] == 0, 'Replay gap or I/O')
    check(a['violations'] == [] and a['semantic_entailment_or_gold_truth_independently_verified'] is False,
          'Audit semantic scope changed')
    rows = p['results']
    by = {(r['id'], r['arm']): r for r in rows}
    check(len(rows) == len(by) == 60 and set(by) == {(q, arm) for q in p['sample_ids'] for arm in ARMS}, 'Missing or duplicate arm')
    sources = {s['source_sha256']: s for s in p['source_text_measurements']}
    check(len(sources) == 2 and sorted(s['canonical_text_tokens'] for s in sources.values()) == [31476162,34235595], 'Shared-source count differs')
    receipts = {(r['id'], r['arm']): r for r in a['receipts']}
    for r in rows:
        check(type(r['correct']) is bool and r['judge_error'] is None, 'Unevaluable score')
        check(r['model'] == p['model'] and all(m == p['model'] for m in r['actual_models']), 'Actual model mismatch')
        check(r['completed_at_unix'] <= p['first_scored_at_unix'], 'Post-gold inference')
        check(r['trajectory_count'] == 100 and r['source_text_tokens_exact'] == sources[r['source_sha256']]['canonical_text_tokens'], 'Source count mismatch')
        check(r['source_chars'] == sources[r['source_sha256']]['source_chars'] and r['source_state_count'] == sources[r['source_sha256']]['state_count'], 'Source metadata mismatch')
        ledger(r['ledger_receipts'], p['budgets'][r['arm']], r['usage'])
        ledger(r['judge_receipts'], p['judge_budget'], r['judge_usage'])
        rec = receipts[r['id'], r['arm']]
        check(rec['record_sha256'] == r['record_sha256'] and rec['answer_sha256'] == r['answer_sha256'], 'Private-record commitment mismatch')
        check(rec['trajectory_replay'] == 'passed' and rec['image_occurrences_verified'] == r['image_occurrences'], 'Replay/image receipt mismatch')
        if r['arm'] == 'direct_full':
            check(r['status'] == 'native_capacity_denied' and not r['correct'] and not r['ledger_receipts'], 'Full-direct outcome mismatch')
            check(r['source_text_tokens_exact'] > p['budgets'][r['arm']]['context_tokens'] + 4096, 'Capacity denial lacks lower bound')
        else:
            check(r['status'] == 'returned' and not r['transport_errors'] and r['terminal_error'] is None, 'Unexpected delivery failure')
    for arm in ARMS:
        rr = [r for r in rows if r['arm'] == arm]
        s = p['summary'][arm]
        check(s['n'] == 20 and s['correct'] == sum(r['correct'] for r in rr), 'Accuracy numerator mismatch')
        check(s['task_success_rate'] == s['correct']/20, 'Task rate mismatch')
        check(s['returned'] == sum(r['status'] == 'returned' for r in rr), 'Delivery count mismatch')
        check(s['semantic_accuracy_among_returned'] == (s['correct']/s['returned'] if s['returned'] else None), 'Semantic scope mismatch')
        check(s['unknown'] == sum(r['unknown'] for r in rr) and s['status_counts'] == dict(Counter(r['status'] for r in rr)), 'Status mismatch')
        for field, declared in s['usage'].items():
            check(declared == sum(r['usage'][field] for r in rr), 'Usage summary mismatch')
        check(math.isclose(s['mean_charged_or_reserved_tokens'],s['usage']['charged_or_reserved_tokens']/20), 'Mean usage mismatch')
        check(math.isclose(s['mean_seconds'],statistics.mean(r['seconds'] for r in rr)), 'Mean time mismatch')
        check(math.isclose(s['median_seconds'],statistics.median(r['seconds'] for r in rr)), 'Median time mismatch')
        check(s['max_correct_source_text_tokens'] == max((r['source_text_tokens_exact'] for r in rr if r['correct']), default=None), 'Correct source extent mismatch')
        for prefix, condition in [('deterministic',lambda r:not r['scoring_function'].startswith('llm_')),
                                  ('llm_judged',lambda r:r['scoring_function'].startswith('llm_')),
                                  ('question_image',lambda r:r['question_image'])]:
            sub = [r for r in rr if condition(r)]
            check(s[prefix+'_n'] == len(sub) and s[prefix+'_correct'] == sum(r['correct'] for r in sub), 'Subset mismatch')
        for key in ('ability','domain'):
            for name, v in s[key].items():
                sub = [r for r in rr if r[key] == name]
                check(v == {'n':len(sub),'correct':sum(r['correct'] for r in sub)}, 'Group mismatch')
    pairs = [(by[q,'direct_headtail_native']['correct'], by[q,'harness']['correct']) for q in p['sample_ids']]
    check(p['paired_native_vs_harness'] == dict(zip(('both_correct','harness_only_correct','native_only_correct','neither_correct'),
          (pairs.count(x) for x in ((True,True),(False,True),(True,False),(False,False))))), 'Paired result mismatch')
    for field, declared in p['judge_usage'].items():
        check(declared == sum(r['judge_usage'][field] for r in rows), 'Judge accounting mismatch')
    totals=a['totals']
    for field, usage_key in [('calls','physical_generations'),('known_input','known_input_tokens'),('known_output','known_output_tokens'),
                             ('charged_or_reserved_tokens','charged_or_reserved_tokens'),('unknown_usage_requests','unknown_usage_requests')]:
        check(totals[field] == sum(r['usage'][usage_key] for r in rows), 'Replay aggregate differs')
    check(totals['image_occurrences_verified'] == sum(r['image_occurrences'] for r in rows), 'Image total differs')
    print('PASS: 60 outcomes; all request/task reservations; 49 source hashes; audit binding; paired/subset metrics.')
    print('Scope: public arithmetic and integrity only; unpublished answers are not semantically regraded.')


if __name__ == '__main__':
    main()
