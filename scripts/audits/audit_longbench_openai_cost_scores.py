"""Offline final cost/identity/score audit, gated after all control outcomes.

No phase runs on import. This file was prepared without opening actual model
predictions, labels or scored artifacts. Run only after root authorizes final
inspection and score_control has produced the complete audit chain.
"""
from __future__ import annotations

from collections import Counter
from decimal import Decimal
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import benchmark_longbench_headtail_control as control
from scripts.audit_context_holdout_execution_v3 import audit_ledger

base, recovery = control.base, control.recovery


def require(value, message):
    if not value:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text())


def strict_choice(content):
    value = str(content or '').replace('*', '').strip()
    for pattern in (r'The correct answer is \(([A-D])\)', r'The correct answer is ([A-D])'):
        match = re.search(pattern, value)
        if match:
            return match.group(1)
    return None


def local_choice(content):
    value = str(content or '')
    values = re.findall(r'(?:correct\s+answer\s+is|answer\s+is|answer\s*:|choice)\s*\(?([ABCD])\)?',
                        value, flags=re.IGNORECASE)
    if values:
        return values[-1].upper()
    values = re.findall(r'\(([ABCD])\)', value, flags=re.IGNORECASE)
    return values[-1].upper() if values else None


def phase_audit(name, out, frozen, records, prices, cap):
    money = load(out/'cost_ledger.json')
    require(money['freeze_sha256'] == base.digest(out/'freeze.json'), name+': monetary freeze identity')
    entries = [entry for entry in money['entries'] if entry.get('status') != 'not_sent']
    require(len(entries) == sum(len(record['ledger']) for record in records), name+': reservation count')
    totals = Counter()
    statuses, actual_models, tiers, errors, safe_provider_events = Counter(), Counter(), Counter(), Counter(), Counter()
    for record in records:
        report = audit_ledger(record['ledger'], record['budget'])
        require(not report['violations'], name+': sequential token budget')
        for field in ('physical_requests', 'prompt_tokens', 'completion_tokens', 'charged_tokens'):
            require(report[field] == record['usage'][field], name+': stored usage conservation')
            totals[field] += report[field]
        totals['unknown_usage_requests'] += report['unknown_usage_requests']
        selected = [entry for entry in entries if entry['model'] == record['model']
                    and entry['id'] == f"{record['arm']}:{record['index']}"]
        require(len(selected) == len(record['ledger']), name+': task money identity')
        for entry, reserved in zip(record['ledger'], selected):
            admitted, output = entry['admission_prompt_tokens'], entry['requested_output_tokens']
            require(reserved['reserved_prompt_tokens'] == admitted and reserved['reserved_output_tokens'] == output,
                    name+': pre-admission conservation')
            price = prices[record['model']]
            rin = max(Decimal(str(price[key])) for key in ('input', 'cached_input', 'cache_write_30m'))
            rout = Decimal(str(price['output']))
            if admitted > 272000:
                rin, rout = rin*2, rout*Decimal('1.5')
            maximum = (admitted*rin+output*rout)/Decimal(1_000_000)
            require(maximum == Decimal(reserved['reserved_upper_usd']), name+': admitted price envelope')
            if entry.get('usage_known'):
                actual = entry['usage']
                expected = (actual['prompt_tokens']*rin+actual['completion_tokens']*rout)/Decimal(1_000_000)
                meta = entry.get('provider_metadata') or {}
                require(meta.get('actual_model') == record['model'], name+': actual model')
                require(meta.get('returned_service_tier') == 'default', name+': actual tier')
                actual_models[meta['actual_model']] += 1
                tiers[meta['returned_service_tier']] += 1
                statuses[meta.get('response_status', 'not_reported')] += 1
                totals['known_usage_identity_checks'] += 1
            else:
                expected = maximum
                totals['unknown_usage_identity_unavailable'] += 1
            require(Decimal(reserved['estimated_upper_usd']) == expected, name+': settlement')
            if entry.get('error_type'):
                errors[entry['error_type']] += 1
        for event in record.get('provider_error_diagnostics', []):
            safe_provider_events[(event.get('stage'), event.get('category'), event.get('http_status'), event.get('code'))] += 1
    cost = sum((Decimal(entry['estimated_upper_usd']) for entry in money['entries']), Decimal(0))
    require(cost == Decimal(money['estimated_upper_usd']) and cost <= Decimal(str(cap)), name+': total money envelope')
    return {'records': len(records), **dict(totals),
            'estimated_generation_cost_upper_usd': str(cost),
            'actual_model_counts_on_known_usage': dict(actual_models), 'tier_counts_on_known_usage': dict(tiers),
            'provider_status_counts_on_known_usage': dict(statuses), 'ledger_error_types': dict(errors),
            'safe_provider_event_counts': [{'stage': key[0], 'category': key[1], 'http_status': key[2],
                                           'code': key[3], 'count': count} for key, count in safe_provider_events.items()],
            'all_known_identities_match': True, 'all_token_and_monetary_checks_passed': True}


def recompute_main(records, labels, rows):
    scored = []
    for record in records:
        if record['arm'] == 'capacity_probe':
            continue
        if record['arm'] == 'direct':
            require(record['prediction'] == local_choice(record.get('content')), 'Direct actual-text primary parser differs')
        elif record.get('result'):
            require(record['prediction'] == record['result'].get('prediction'), 'Harness controller choice differs')
        denied = record['status'] == 'native_admission_denied'
        available = record['prediction'] in ('A', 'B', 'C', 'D') and not denied
        correct = available and record['prediction'] == labels[str(record['index'])]
        scored.append({**record, 'correct': None if denied else bool(correct),
                       'semantic_answer_available': available, 'task_success': bool(correct)})
    return recovery._summarize(scored, rows), scored


def audit_final(config, *, output=None):
    started = time.perf_counter()
    config = control.public_config(config)
    out = Path(config['output'])
    frozen = load(out/'freeze.json')
    run = load(out/'run_summary.json')
    require(run.get('phase') == 'headtail_control_complete'
            and run.get('planned_control_records') == frozen['expected_control_records']
            and run.get('terminal_control_records') == frozen['expected_control_records'],
            'Control completion gate is not satisfied')
    access = load(out/'scoring_access.json')
    require(access.get('all_114_and_control_terminal_before_labels') is True and (out/'audit.json').exists(),
            'Final unified control/114 scoring gate is not satisfied')
    # Only now may prediction objects, scored artifacts and labels be opened.
    frozen, context = control.check_control(config)
    rec_out, parent_out = context['recovery_out'], Path(context['parent_config']['output'])
    require(load(rec_out/'scoring_access.json').get('all_75_parent_and_39_recovery_terminal_before_labels') is True,
            'Recovery first-label-access gate differs')
    require(load(parent_out/'scoring_access.json').get('all_75_terminal_before_label_access') is True,
            'Original first-label-access gate differs')
    originals = [load(path) for path in sorted((parent_out/'predictions').glob('*.json'))]
    recovered = [load(path) for path in sorted((rec_out/'predictions').glob('*.json'))]
    controls = [load(path) for path in sorted((out/'predictions').glob('*.json'))]
    require(len(originals) == 75 and len(recovered) == 39
            and len(controls) == frozen['expected_control_records'], 'Physical record matrix sizes differ')
    labels_path = Path(context['parent_config']['labels'])
    require(base.digest(labels_path) == context['recovery_frozen']['labels_sha256'], 'Label digest differs')
    labels = load(labels_path)
    prices = context['child_config']['prices']
    phase_results = {
        'original': phase_audit('original', parent_out, load(parent_out/'freeze.json'), originals, prices, context['global_cap_usd']),
        'recovery': phase_audit('recovery', rec_out, context['recovery_frozen'], recovered, prices,
                                context['recovery_frozen']['recovery']['new_generation_cost_cap_usd']),
        'headtail': phase_audit('headtail', out, frozen, controls, prices, frozen['control_cap_usd'])}
    cost = sum((Decimal(value['estimated_generation_cost_upper_usd']) for value in phase_results.values()), Decimal(0))
    total = cost+Decimal(context['preflight_cost_upper_usd'])
    require(total <= Decimal(context['global_cap_usd']), 'Three-phase plus preflight cost cap exceeded')
    row_map = {row['index']: row for row in context['rows']}
    original_summary, original_scored = recompute_main(originals, labels, context['rows'])
    original_audit = load(parent_out/'audit.json')
    require(original_summary == original_audit['summary'], 'Raw first-trial score/denominator recomputation differs')
    overlays = {(record['model'], record['index'], record['arm']): record for record in recovered}
    composite_records = []
    for original in originals:
        key = original['model'], original['index'], original['arm']
        chosen = overlays.get(key, original)
        if key in overlays:
            require(recovery.eligible(original), 'Recovery overlaid a prior generation or non-network failure')
            require(original['usage']['charged_tokens'] == 0 and original['usage']['physical_requests'] == 0,
                    'Recovery reset a previously used budget')
        composite_records.append(chosen)
    composite_summary, composite_scored = recompute_main(composite_records, labels, context['rows'])
    composite_audit = load(rec_out/'audit.json')
    require(composite_summary == composite_audit['summary'], 'Composite score/denominator recomputation differs')
    require(composite_audit['first_trial_summary'] == original_summary, 'Raw trial hidden or rewritten')
    headtail_audit = load(out/'audit.json')
    strict_counts, primary_counts = Counter(), Counter()
    for record in controls:
        gold = labels[str(record['index'])]
        primary, strict = local_choice(record.get('content')), strict_choice(record.get('content'))
        require(primary == record['prediction'] and strict == record['strict_official_prediction'],
                'Head/tail actual-text parsing differs')
        primary_counts[record['model']] += primary == gold
        strict_counts[record['model']] += strict == gold
    for model in control.MODELS:
        require(headtail_audit['summary'][model]['primary_correct'] == primary_counts[model],
                'Head/tail primary score differs')
        require(headtail_audit['summary'][model]['strict_official_correct'] == strict_counts[model],
                'Head/tail strict raw-answer score differs')
    native = [record for record in composite_scored if record['arm'] == 'direct']
    native_attempted = [record for record in native if record['ledger']]
    native_context_errors = [record for record in native_attempted if record.get('error')
                            and 'context_length_exceeded' in record['error'].get('message', '')]
    native_unknown_network = [record for record in native_attempted if record.get('error')
                              and record['error'].get('message') == recovery.NETWORK_MESSAGE]
    snapshot_counts = {}
    for name, phase_out in (('original', parent_out), ('recovery', rec_out), ('headtail', out)):
        phase_freeze = load(phase_out/'freeze.json')
        mapping = phase_freeze['code_sha256']
        for code, expected in mapping.items():
            require(base.digest(ROOT/code) == expected and base.digest(phase_out/'frozen_source'/code) == expected,
                    name+': live or frozen code differs')
        snapshot_counts[name] = len(mapping)
    control_mtimes = [path.stat().st_mtime for path in (out/'predictions').glob('*.json')]
    require(not control_mtimes or max(control_mtimes) <= access['first_scored_at_unix'],
            'Control outcomes were written after first-label-access metadata')
    original_mtimes = [path.stat().st_mtime for path in (parent_out/'predictions').glob('*.json')]
    recovery_mtimes = [path.stat().st_mtime for path in (rec_out/'predictions').glob('*.json')]
    original_access = load(parent_out/'scoring_access.json')['first_scored_at_unix']
    recovery_access = load(rec_out/'scoring_access.json')['first_scored_at_unix']
    main_frozen = load(parent_out/'freeze.json')['frozen_at_unix']
    recovery_frozen = context['recovery_frozen']['frozen_at_unix']
    control_registered, control_frozen = frozen['registered_at_unix'], frozen['frozen_at_unix']
    require(max(original_mtimes) <= recovery_frozen, 'Recovery registered before original outcomes finished')
    require(max(recovery_mtimes) <= control_registered, 'Control registered before recovery outcomes finished')
    require(not control_mtimes or min(control_mtimes) >= control_frozen, 'Control outcomes predate their freeze')
    latest_all = max(original_mtimes+recovery_mtimes+control_mtimes)
    earliest_score = min(original_access, recovery_access, access['first_scored_at_unix'])
    require(latest_all <= earliest_score, 'Some physical outcomes postdate a label-access marker')
    def utc(value):
        return datetime.fromtimestamp(value, timezone.utc).isoformat()
    timing = {
        'main_frozen_at_utc': utc(main_frozen), 'latest_original_outcome_utc': utc(max(original_mtimes)),
        'recovery_frozen_at_utc': utc(recovery_frozen), 'latest_recovery_outcome_utc': utc(max(recovery_mtimes)),
        'control_registered_at_utc': utc(control_registered), 'control_frozen_at_utc': utc(control_frozen),
        'latest_control_outcome_utc': utc(max(control_mtimes)) if control_mtimes else None,
        'first_control_score_gate_utc': utc(access['first_scored_at_unix']),
        'first_recovery_score_gate_utc': utc(recovery_access), 'first_original_score_access_utc': utc(original_access),
        'all_physical_outcomes_precede_all_score_markers': True,
    }
    result = {'schema_version': 1, 'status': 'pass', 'scope': 'Independent budget, cost, identity and primary/strict-control score audit',
        'main_freeze_sha256': recovery.PARENT_FREEZE,
        'recovery_freeze_sha256': context['recovery_freeze_sha256'], 'control_freeze_sha256': base.digest(out/'freeze.json'),
        'physical_record_counts': {'original': 75, 'recovery': 39, 'headtail': len(controls)},
        'physical_generations_all_phases': sum(value['physical_requests'] for value in phase_results.values()),
        'known_input_tokens_all_phases': sum(value['prompt_tokens'] for value in phase_results.values()),
        'known_output_tokens_all_phases': sum(value['completion_tokens'] for value in phase_results.values()),
        'charged_or_reserved_tokens_all_phases': sum(value['charged_tokens'] for value in phase_results.values()),
        'unknown_usage_requests_all_phases': sum(value['unknown_usage_requests'] for value in phase_results.values()),
        'logical_main_records': 72, 'logical_capacity_probes': 3, 'distinct_source_contexts': 12,
        'phases': phase_results, 'generation_cost_upper_usd': str(cost),
        'preflight_cost_upper_usd': context['preflight_cost_upper_usd'],
        'combined_cost_with_preflight_upper_usd': str(total), 'global_cap_usd': context['global_cap_usd'],
        'raw_first_trial_summary': original_summary, 'composite_summary': composite_summary,
        'headtail_primary_correct_counts': dict(primary_counts), 'headtail_strict_official_correct_counts': dict(strict_counts),
        'native_direct_observations': {'advertised_admissible': sum(record['measurements'].get('fits_native_direct') is True for record in native),
            'actual_generation_attempts': len(native_attempted),
            'actual_returned': sum(record['status'] == 'returned' for record in native_attempted),
            'known_context_error_responses': len(native_context_errors),
            'unknown_network_delivery': len(native_unknown_network),
            'admission_denials_are_not_semantic_wrong_answers': True},
        'source_snapshot_counts': snapshot_counts, 'original_and_composite_scores_recomputed': True,
        'recovery_wrapper_snapshot_separately_verified_by_check_recovery': True,
        'registration_and_scoring_timing': timing,
        'headtail_actual_raw_content_strict_scores_recomputed': True,
        'harness_strict_actual_candidate_audit': 'Separate trajectory audit; compatibility letter is not treated as raw official format',
        'recovery_eligibility_was_metadata_only': True, 'prior_generated_outcomes_never_retried': True,
        'counting_or_model_api_calls_made_by_this_audit': 0, 'raw_labels_answers_prompts_or_facts_written': False,
        'limitations': ['Cost is an envelope estimate, not invoice; unknown usage retains full reservation.',
            'Shared production scope and aggregation helpers are used for denominator comparison.',
            'File metadata and frozen code order are not complete OS or network access forensics.',
            'Twelve reused sources and two optional lossy-control sources do not establish general accuracy superiority.'],
        'violations': [], 'seconds': round(time.perf_counter()-started, 3)}
    destination = Path(output) if output else out.parent/'final_cost_score_audit.json'
    base.atomic_json(destination, result)
    return result


if __name__ == '__main__':
    config_path = ROOT/'work/longbench_openai_limits_20261102/headtail_config.json'
    report = audit_final(load(config_path))
    print(json.dumps({'status': report['status'], 'physical_record_counts': report['physical_record_counts'],
                      'combined_cost_with_preflight_upper_usd': report['combined_cost_with_preflight_upper_usd']}))
