"""Verify public compact-holdout receipts using only the Python standard library.

Checks arithmetic, budgets, chronology, frozen code and replay commitments.
Does not open private labels, call a model, or independently regrade semantics.
"""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

ROOT = Path(__file__).resolve().parents[1]
ID = 'longmemeval_v2_compact_holdout_20261103'
ARMS = ['direct_full', 'direct_headtail_native', 'harness',
        'harness_completion', 'harness_compact']


def check(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def hash_ok(value):
    return isinstance(value, str) and re.fullmatch('[0-9a-f]{64}', value) is not None


def scan(value):
    if isinstance(value, dict):
        check(not {'question', 'answer', 'gold', 'labels', 'content', 'trace',
                   'reasoning', 'api_key', 'headers', 'authorization'} & set(value),
              'Raw/private content key')
        for item in value.values():
            scan(item)
    elif isinstance(value, list):
        for item in value:
            scan(item)
    elif isinstance(value, str):
        check(not re.search(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}|/(?:Users|home|root|uniparser)/|data:image/', value),
              'Private material')


def verify_ledger(entries, budget):
    total = 0
    check(len(entries) <= budget['max_calls'], 'Call ceiling')
    for number, entry in enumerate(entries, 1):
        prompt, output = entry['admission_prompt_tokens'], entry['requested_output_tokens']
        check(entry['physical_request'] == number, 'Physical request ordering')
        check(type(prompt) is int and prompt >= 0 and type(output) is int
              and 0 < output <= budget['max_output_tokens'], 'Token admission')
        check(prompt + output <= budget['context_tokens']
              and total + prompt + output <= budget['total_tokens'], 'Budget reservation')
        check(hash_ok(entry['prompt_sha256']), 'Prompt fingerprint')
        if entry['usage_known']:
            check(type(entry['input_tokens']) is int and 0 <= entry['input_tokens'] <= prompt,
                  'Input usage')
            check(type(entry['output_tokens']) is int and 0 <= entry['output_tokens'] <= output,
                  'Output usage')
            check(entry['charged_tokens'] == entry['input_tokens'] + entry['output_tokens'],
                  'Usage conservation')
        else:
            check(entry['charged_tokens'] == prompt + output, 'Unknown reservation')
        total += entry['charged_tokens']
    check(total <= budget['total_tokens'], 'Task ceiling')
    return total


def compaction_stats(rows):
    events = [event for row in rows for event in row['compaction_events']]
    raw = sum(event['baseline_prompt_tokens'] for event in events)
    display = sum(event['display_prompt_tokens'] for event in events)
    enabled = 0
    for row in rows:
        for event in row['compaction_events']:
            number = event['physical_request']
            entry = row['ledger'][number - 1]
            available = min(row['budget']['context_tokens'], row['budget']['total_tokens']
                            - sum(x['charged_tokens'] for x in row['ledger'][:number - 1]))
            enabled += (event['baseline_prompt_tokens'] + entry['requested_output_tokens'] > available
                        >= event['display_prompt_tokens'] + entry['requested_output_tokens'])
    return {
        'compacted_cases': sum(bool(row['compaction_events']) for row in rows),
        'matched_raw_prompt_tokens': raw,
        'matched_display_prompt_tokens': display,
        'prompt_tokens_saved': raw - display,
        'matched_prompt_reduction': (raw - display) / raw if raw else None,
        'duplicate_source_chars_removed': sum(w['duplicate_source_chars_removed']
                                              for event in events for w in event['windows']),
        'requests_admitted_due_to_display_savings': enabled,
        'native_box_overrides': sum(row['native_box_preserved'] is False for row in rows),
        'output_reservations': dict(Counter(str(x['allocated_output'])
                                          for row in rows for x in row['output_allocations'])),
    }


def main():
    p = json.loads((ROOT / 'benchmarks' / f'{ID}.json').read_text())
    audit_path = ROOT / 'benchmarks' / f'{ID}_audit.json'
    audit = json.loads(audit_path.read_text())
    scan(p)
    scan(audit)
    check(p['arms'] == ARMS and p['n_questions'] == 20 and p['outcomes'] == 100
          and len(set(p['sample_ids'])) == 20, 'Evaluation unit')
    check(p['question_id_overlap_with_prior40'] == p['question_text_hash_overlap_with_prior40'] == 0,
          'Question holdout overlap')
    check(p['n_shared_histories'] == 2 and p['source_histories_reused'] is True
          and p['n_trajectories'] == 200 and p['n_history_images'] == 5095
          and p['n_query_images'] == 4, 'Source reuse disclosure')
    check(p['model'] == 'Qwen3.8-Flash-Next-FP8', 'Exact model')
    check(p['wire'] == {'temperature': 0, 'seed': 20261103,
                         'enable_thinking': False, 'preserve_thinking': False}, 'Generation controls')
    check(p['data_revision'] == 'f152293e235517d504809563c833d7190b8c713b'
          and p['upstream_code_commit'] == '2cc8c540bdb87fe6761629b585e727e1c4704520',
          'Pinned upstream')
    check(p['official_leaderboard_submission'] is False and p['official_LAFS_score'] is None
          and p['native_and_harness_equal_compute'] is False, 'Evaluation scope')
    check(p['tests_passed'] == 258 and p['remote_files_modified'] is False
          and p['github_push'] is False, 'Local execution receipt')
    for arm in ARMS:
        budget = p['budget'][arm]
        expected = (262144, 262144, 1) if arm.startswith('direct_') else (16384, 65536, 8)
        check((budget['context_tokens'], budget['total_tokens'], budget['max_calls']) == expected
              and budget['max_output_tokens'] == 4096, 'Shared hard ceilings')
    check(digest(audit_path) == p['offline_audit_sha256']
          and audit['freeze_sha256'] == p['freeze_sha256'], 'Audit commitment')
    check(digest(ROOT / 'scripts/audits/audit_longmemeval_v2_compact.py') == p['auditor_sha256']
          == audit['auditor_sha256'], 'Auditor fingerprint')
    for path, expected in p['code_sha256'].items():
        check(digest(ROOT / path) == expected, 'Frozen source changed: ' + path)
    check(audit['status'] == 'pass' and not audit['failures'], 'Replay failures')
    check(audit['terminal_records'] == audit['ledgers_passed'] == audit['scores_recomputed']
          == audit['replay_passed'] == 100, 'Audit cardinality')
    check(audit['reviews_identical_and_source_union_preserved'] is True
          and audit['provider_or_tokenizer_calls_made'] == 0
          and audit['independent_semantic_entailment_validation'] is False, 'Audit scope')
    rows = p['results']
    by = {(r['id'], r['arm']): r for r in rows}
    receipts = {(r['id'], r['arm']): r for r in audit['receipts']}
    check(len(rows) == len(by) == len(receipts) == 100
          and set(by) == set(receipts) == {(q, arm) for q in p['sample_ids'] for arm in ARMS},
          'Missing/duplicate outcomes')
    check(len({r['source_sha256'] for r in rows}) == 2, 'Source count')
    prior = json.loads((ROOT / 'benchmarks/longmemeval_v2_qwen_20261103.json').read_text())
    prior_visual = json.loads((ROOT / 'benchmarks/longmemeval_v2_visual_holdout_20261103.json').read_text())
    prior_ids = set(prior['sample_ids']) | set(prior_visual['sample_ids'])
    check(len(prior_ids) == 40 and not prior_ids & set(p['sample_ids']), 'Actual prior-ID exclusion')
    check({r['source_sha256'] for r in rows} == {r['source_sha256'] for r in prior['results']},
          'Reused source fingerprint disclosure')
    check({r['source_text_tokens'] for r in rows} == {31476162, 34235595}, 'Exact source lengths')
    for row in rows:
        arm = row['arm']
        check(type(row['correct']) is bool and row['judge_error'] is None, 'Unevaluable score')
        check(row['model'] == p['model'] and all(x == p['model'] for x in row['actual_models']),
              'Actual model identity')
        check(p['frozen_at_unix'] < row['completed_at_unix'] < p['first_scoring_access_unix']
              and p['all100_fixed_before_scoring_label_access'] is True, 'Scoring chronology')
        check(row['within_budget'] and row['budget'] == p['budget'][arm], 'Budget declaration')
        check(verify_ledger(row['ledger'], row['budget']) == row['tokens']
              and len(row['ledger']) == row['calls'], 'Row usage')
        verify_ledger(row['judge_ledger'], p['judge_budget'])
        receipt = receipts[row['id'], arm]
        check(row['record_sha256'] == receipt['record_sha256'] and receipt['ledger_passed']
              and receipt['score_recomputed'] and receipt['replay_passed'], 'Private record binding')
        for key in ['record_sha256', 'source_sha256', 'answer_sha256', 'parsed_answer_sha256']:
            check(hash_ok(row[key]), 'Fingerprint: ' + key)
        if arm == 'direct_full':
            check(row['status'] == 'native_capacity_denied' and not row['calls']
                  and row['source_text_tokens'] > 262144 + 4096, 'Full Direct capacity')
        else:
            check(row['status'] == 'returned' and row['error'] is None, 'Delivery failure')
            check(row['actual_models'] == [p['model']] and row['calls'] > 0,
                  'Returned model receipt')
        check(row['recovery_calls'] in [0, 1], 'Recovery limit')
        if arm == 'harness':
            check(all(e['requested_output_tokens'] == 1024 for e in row['ledger']),
                  'Legacy actual output reservation')
        if arm == 'direct_headtail_native':
            check(len(row['ledger']) == 1 and row['ledger'][0]['requested_output_tokens'] == 4096,
                  'Direct actual output reservation')
        if arm in ['harness_completion', 'harness_compact']:
            check(hash_ok(row['native_answer_sha256']), 'Native answer commitment')
            if row['native_box_preserved'] is not None:
                check(row['native_box_preserved'] is True
                      and row['native_answer_sha256'] == row['answer_sha256'], 'Native box override')
            for allocation in row['output_allocations']:
                check(allocation['requested_by_controller'] == 1024
                      and allocation['allocated_output'] in [1024, 2048, 4096]
                      and allocation['baseline_evidence_unchanged'] is True, 'Output allocation')
                check(row['ledger'][allocation['physical_request'] - 1]['requested_output_tokens']
                      == allocation['allocated_output'], 'Allocation ledger binding')
            if row['recovery_status'] == 'refusal_rescued_with_observed_quote':
                check(row['quote_valid'] is True and row['recovery_calls'] == 1,
                      'Quote-gated refusal rescue')
        if arm != 'harness_compact':
            check(not row['compaction_events'], 'Unexpected display transform')
        for event in row['compaction_events']:
            entry = row['ledger'][event['physical_request'] - 1]
            check(event['display_prompt_tokens'] == entry['admission_prompt_tokens']
                  and event['baseline_prompt_tokens'] - event['display_prompt_tokens']
                  == event['saved_prompt_tokens'] > 0, 'Matched prompt counts')
            for window in event['windows']:
                lo, hi = window['source_span']
                aliases = sorted(window['original_source_spans'])
                check(0 <= lo < hi and bool(aliases), 'Source span')
                start, end = aliases[0]
                for a, b in aliases[1:]:
                    check(start <= a <= end and a < b, 'Disconnected source alias')
                    end = max(end, b)
                check([start, end] == [lo, hi] and window['reversible'] is True
                      and window['duplicate_source_chars_removed']
                      == sum(b - a for a, b in aliases) - (hi - lo), 'Source union')
                check(hash_ok(window['raw_sha256']) and hash_ok(window['display_sha256'])
                      and type(window['edit_count']) is int and window['edit_count'] >= 0,
                      'Reversible display receipt')
    for arm in ARMS:
        group = [r for r in rows if r['arm'] == arm]
        summary = p['summary'][arm]
        correct = sum(r['correct'] for r in group)
        answered = [r for r in group if r['non_unknown']]
        returned = sum(r['status'] == 'returned' for r in group)
        check(summary['n'] == 20 and summary['correct'] == correct
              and summary['task_success_rate_all_planned'] == correct / 20
              and summary['returned'] == returned
              and summary['accuracy'] == (correct / returned if returned else None), 'Accuracy arithmetic')
        check(summary['unknown'] == sum(r['unknown'] for r in group)
              and summary['answered'] == len(answered) and summary['coverage'] == len(answered) / 20,
              'Coverage arithmetic')
        check(summary['selective_accuracy'] == (correct / len(answered) if answered else None)
              and summary['wrong_answered'] == sum(not r['correct'] for r in answered),
              'Selective precision')
        check(summary['tokens'] == sum(r['tokens'] for r in group)
              and summary['calls'] == sum(r['calls'] for r in group), 'Usage summary')
        for prefix, predicate in [('deterministic', lambda r: not r['scoring_function'].startswith('llm_')),
                                  ('judge', lambda r: r['scoring_function'].startswith('llm_')),
                                  ('image', lambda r: r['question_image'])]:
            subset = [r for r in group if predicate(r)]
            check(summary[prefix + '_n'] == len(subset)
                  and summary[prefix + '_correct'] == sum(r['correct'] for r in subset), 'Subset arithmetic')
        for key, expected in compaction_stats(group).items():
            check(summary[key] == expected, 'Compaction arithmetic: ' + key)
        check(summary['repairs'] == sum(r['recovery_calls'] for r in group), 'Recovery summary')
        check(summary['truncated_generations'] == sum(r['truncated_generations'] for r in group)
              and summary['unboxed_returned_answers'] == sum(r['status'] == 'returned'
                  and not r['final_has_complete_box'] for r in group)
              and summary['recovery_statuses'] == dict(Counter(r['recovery_status']
                  for r in group if r['recovery_status'])), 'Recovery/truncation arithmetic')
    for arm, paired in p['paired'].items():
        pairs = [(by[q, 'harness']['correct'], by[q, arm]['correct']) for q in p['sample_ids']]
        for key, value in zip(['both_correct', 'new_only_correct', 'old_only_correct', 'neither_correct'],
                              [(True, True), (False, True), (True, False), (False, False)]):
            check(paired[key] == pairs.count(value), 'Paired accuracy')
        check(paired['same_first_review_prompt'] == sum(
            by[q, 'harness']['ledger'][0]['prompt_sha256'] == by[q, arm]['ledger'][0]['prompt_sha256']
            for q in p['sample_ids']), 'First review prompt')
    check(audit['totals']['tokens'] == sum(r['tokens'] for r in rows)
          and audit['totals']['generations'] == sum(r['calls'] for r in rows), 'Audit usage')
    check(p['judge_usage']['calls'] == sum(len(r['judge_ledger']) for r in rows)
          and p['judge_usage']['tokens'] == sum(e['charged_tokens']
              for r in rows for e in r['judge_ledger']), 'Separate scoring usage')
    print(json.dumps({'status': 'PASS', 'outcomes': 100, 'source_fingerprints': len(p['code_sha256']),
                      'replays_passed': audit['replay_passed'],
                      'scope': 'Public arithmetic/budget/source integrity; not semantic regrading'}))


if __name__ == '__main__':
    main()
