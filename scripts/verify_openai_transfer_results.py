"""Recompute the public OpenAI transfer arithmetic without APIs or raw data.

Reads only published JSON and hashes declared repository-relative .py files.
It does not read labels, prediction archives, facts or request logs, and does
not re-run the official scorer or establish semantic correctness.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('gpt-6-sol', 'gpt-6-luna', 'gpt-6.1-terra')
SUM_FIELDS = ('physical_generation_calls', 'input_tokens_known', 'output_tokens_known',
              'charged_or_reserved_tokens', 'unknown_usage_requests', 'cached_input_tokens',
              'cache_write_tokens', 'cache_write_30m_tokens', 'cache_write_12h_tokens',
              'reasoning_tokens', 'count_queries_including_cache_hits', 'count_api_requests',
              'count_api_errors', 'unknown_usage_cost_reservations')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(left, right):
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def counter_sum(rows, key):
    return dict(sum((Counter(row[key]) for row in rows), Counter()))


def wilson(k, n):
    z = 1.959963984540054
    p, d = k/n, 1+z*z/n
    middle = (p+z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0, middle-half), min(1, middle+half)]


def verify(payload, *, root=ROOT, integrity=None, reference=None):
    require(payload['schema_version'] == 1 and payload['experiment_id'] == 'openai_transfer_20261102', 'Unsupported experiment')
    require(payload['requested_models'] == list(MODELS) and set(payload['summary']) == set(MODELS), 'Model IDs changed')
    require(payload['n_distinct_source_contexts'] == 12 and payload['n_model_sample_pairs'] == 36, 'Source/pair count changed')
    require(payload['reused_cohort'] is True and payload['new_blind_holdout'] is False, 'Transfer scope changed')
    require(payload['capacity_definition']['direct_inference_performed'] is False
            and payload['capacity_definition']['model_attention_window_changed'] is False, 'Capacity scope changed')
    budget = payload['budget']
    require(budget == {'context_tokens': 16384, 'total_tokens': 65536, 'max_calls': 32,
                       'max_output_tokens': 2048, 'token_safety_margin': 16}, 'Budget changed')
    require(payload['wire_parameters'] == {'api': 'responses', 'reasoning_effort': 'none',
        'store': False, 'service_tier': 'default', 'truncation': 'disabled',
        'temperature_sent': False, 'seed_sent': False, 'implicit_retries': 0}, 'Wire settings changed')
    sources = {row['sample_id']: row for row in payload['sources']}
    require(len(payload['sources']) == len(sources) == 12, 'Duplicate or missing source contexts')
    hashes = [row['context_sha256'] for row in sources.values()]
    require(len(set(hashes)) == 12 and all(re.fullmatch('[0-9a-f]{64}', value) for value in hashes), 'Invalid/distinct source hashes')
    pairs = {(model, source) for model in MODELS for source in sources}
    rows = payload['results']
    require(len(rows) == 36 and {(r['requested_model'], r['sample_id']) for r in rows} == pairs, 'Result matrix mismatch')
    measurements = payload['measurements']
    require(len(measurements) == 36 and {(r['requested_model'], r['sample_id']) for r in measurements} == pairs, 'Measurement matrix mismatch')
    by_pair = {(r['requested_model'], r['sample_id']): r for r in rows}
    prices = payload['cost']['prices_usd_per_million_tokens']
    require(set(prices) == set(MODELS), 'Price/model matrix mismatch')
    for row in rows:
        prefix = f'{row["requested_model"]}/{row["sample_id"]}: '
        score = row['official_score']
        require(type(score) in (int, float) and math.isfinite(score) and 0 <= score <= 1, prefix+'invalid score')
        require(row['full_credit'] is (score == 1) and type(row['parsed_exact_match']) is bool, prefix+'metric flag mismatch')
        require(row['terminal'] is True and type(row['nonempty_output']) is bool, prefix+'nonterminal result')
        require(all(type(row[key]) is int and row[key] >= 0 for key in SUM_FIELDS), prefix+'invalid usage count')
        calls, unknown = row['physical_generation_calls'], row['unknown_usage_requests']
        inp, out, charged = row['input_tokens_known'], row['output_tokens_known'], row['charged_or_reserved_tokens']
        require(calls <= 32 and charged <= 65536 and unknown <= calls, prefix+'budget exceeded')
        require(row['all_usage_known'] is (unknown == 0) and row['within_budget'] is True, prefix+'usage flags mismatch')
        require((charged == inp+out if unknown == 0 else charged >= inp+out), prefix+'token conservation failed')
        require(row['remaining_cumulative_tokens_after_task'] == 65536-charged, prefix+'remaining tokens mismatch')
        require(row['cached_input_tokens']+row['cache_write_tokens'] <= inp
                and row['cache_write_30m_tokens']+row['cache_write_12h_tokens'] <= row['cache_write_tokens']
                and row['reasoning_tokens'] <= out, prefix+'usage details exceed their parent totals')
        require(row['count_api_requests'] <= row['count_queries_including_cache_hits']
                and row['count_api_errors'] <= row['count_api_requests'], prefix+'counting totals mismatch')
        require(sum(row['response_status_counts'].values()) == calls
                and sum(row['incomplete_reason_counts'].values()) <= row['response_status_counts'].get('incomplete', 0), prefix+'response counts mismatch')
        require(row['actual_model_ids'] in ([], [row['requested_model']])
                and (unknown == calls or row['actual_model_ids'] == [row['requested_model']]), prefix+'actual model identity mismatch')
        require(set(row['returned_service_tiers']) <= {'default', 'not_reported'}, prefix+'nondefault/unknown service tier')
        if calls:
            reserved = row['last_request_output_reservation_tokens']
            require(type(reserved) is int and 0 < reserved <= 2048, prefix+'invalid last output reservation')
            if row['last_request_known_output_tokens'] is not None:
                require(row['last_request_known_output_tokens'] <= reserved, prefix+'last output exceeds reservation')
        price = prices[row['requested_model']]
        require(Decimal(str(price['r_max'])) >= max(Decimal(str(value)) for value in price.values()), prefix+'invalid price envelope')
        upper = Decimal(row['estimated_generation_cost_upper_usd'])
        calculated = Decimal(charged)*Decimal(str(price['r_max']))/1_000_000
        require(upper >= calculated and (row['unknown_usage_cost_reservations'] != 0 or upper == calculated), prefix+'upper estimate mismatch')
        base = (Decimal(inp)*Decimal(str(price['input']))+Decimal(out)*Decimal(str(price['output'])))/1_000_000
        require(Decimal(row['estimated_base_io_cost_on_known_usage_usd']) == base, prefix+'base I/O estimate mismatch')
    computed = {}
    for model in MODELS:
        group = [row for row in rows if row['requested_model'] == model]
        summary = payload['summary'][model]
        require(summary['status'] == 'scored' and summary['n_planned'] == summary['n_terminal'] == 12, model+': incomplete model arm')
        full = sum(row['full_credit'] for row in group)
        mean = statistics.mean(row['official_score'] for row in group)
        require(summary['full_credit_count'] == full and close(summary['full_credit_rate'], full/12)
                and close(summary['mean_official_score'], mean)
                and summary['parsed_exact_match_count'] == sum(row['parsed_exact_match'] for row in group)
                and summary['n_nonempty'] == sum(row['nonempty_output'] for row in group), model+': score summary mismatch')
        require(all(close(a, b) for a, b in zip(summary['full_credit_wilson_95_interval'], wilson(full, 12))), model+': descriptive Wilson interval mismatch')
        for key in SUM_FIELDS:
            require(summary[key] == sum(row[key] for row in group), model+': '+key+' mismatch')
        require(summary['recorded_errors'] == sum(row['error'] is not None for row in group)
                and summary['error_type_counts'] == dict(Counter(row['error']['type'] for row in group if row['error']))
                and summary['error_category_counts'] == dict(Counter(row['error']['category'] for row in group if row['error'])), model+': error totals mismatch')
        for key in ('response_status_counts', 'incomplete_reason_counts'):
            require(summary[key] == counter_sum(group, key), model+': '+key+' mismatch')
        for key in ('actual_model_ids', 'returned_service_tiers'):
            require(summary[key] == sorted({value for row in group for value in row[key]}), model+': identity set mismatch')
        for key in ('estimated_generation_cost_upper_usd', 'estimated_base_io_cost_on_known_usage_usd'):
            require(Decimal(summary[key]) == sum((Decimal(row[key]) for row in group), Decimal(0)), model+': '+key+' mismatch')
        require(summary['all_usage_known'] is all(row['all_usage_known'] for row in group)
                and summary['all_within_budget'] is True, model+': aggregate flags mismatch')
        mechanisms = [row['mechanism'] for row in group]
        m = summary['mechanisms']
        for target, key in [('mode_counts', 'mode'), ('stop_reason_counts', 'stop_reason')]:
            require(m[target] == dict(Counter(row[key] or 'unavailable' for row in mechanisms)), model+': '+target+' mismatch')
        require(m['operation_counts'] == dict(Counter(row['plan']['operation'] or 'unavailable' for row in mechanisms)), model+': operation counts mismatch')
        for target, value in {
            'plan_validation_errors': sum(r['plan_validation_error_count'] for r in mechanisms),
            'plan_repairs': sum(r['plan_repair_attempted'] is True for r in mechanisms),
            'single_unresolved_record_at_stop': sum(r['single_unresolved_record_at_stop'] for r in mechanisms),
            'no_fact_gain_stops': sum(r['stopped_without_new_fact_gain'] for r in mechanisms),
            'recorded_final_fact_loss_total': sum(r['final_fact_loss_count'] or 0 for r in mechanisms),
            'final_fact_loss_metric_present': sum(r['final_fact_loss_count'] is not None for r in mechanisms),
        }.items():
            require(m[target] == value, model+': mechanism summary mismatch: '+target)
        computed[model] = {'n': 12, 'full_credit': full, 'mean_official_score': mean,
                           'em': summary['parsed_exact_match_count'], 'calls': summary['physical_generation_calls'],
                           'charged_tokens': summary['charged_or_reserved_tokens']}
    native = payload['capacity_definition']['published_native_context_tokens_per_model']
    require(native == 1050000 and payload['capacity_definition']['frozen_direct_reserved_output_tokens'] == 1024, 'Native/direct limits changed')
    for row in measurements:
        admitted, direct, source = (row[key] for key in ('direct_admission_input_tokens_including_safety_margin',
                                                         'direct_prompt_input_tokens', 'source_envelope_input_tokens'))
        require(all(type(v) is int and v > 0 for v in (admitted, direct, source)), 'Missing/invalid exact input measurement')
        require(row['direct_reserved_output_tokens'] == 1024 and row['native_context_window_tokens'] == native
                and admitted == direct+budget['token_safety_margin'] and row['direct_inference_performed'] is False, 'Direct measurement definition mismatch')
        total = admitted+1024
        fits = {'fits_native_window_including_output_and_margin': total <= native,
                'fits_experimental_request': total <= budget['context_tokens'],
                'fits_experimental_cumulative_budget': total <= budget['total_tokens']}
        require(all(row[key] is value for key, value in fits.items()), 'Capacity fit boolean mismatch')
        reasons = [reason for key, reason in [
            ('fits_experimental_request', 'exceeds_experimental_16384_input_plus_output'),
            ('fits_experimental_cumulative_budget', 'exceeds_experimental_65536_cumulative_budget'),
            ('fits_native_window_including_output_and_margin', 'exceeds_published_native_1050000_input_plus_output')]
            if not fits[key]]
        require(row['capacity_reasons'] == reasons, 'Capacity rejection reasons mismatch')
    for model in MODELS:
        group = [row for row in measurements if row['requested_model'] == model]
        summary = payload['capacity_summary'][model]
        for key, field in [('source_envelope_input_token_range', 'source_envelope_input_tokens'),
                           ('direct_admission_input_token_range_including_safety_margin', 'direct_admission_input_tokens_including_safety_margin')]:
            values = [row[field] for row in group]
            require(summary[key] == [min(values), max(values)], model+': capacity range mismatch')
        for key, field in [('native_window_fits', 'fits_native_window_including_output_and_margin'),
                           ('experimental_request_fits', 'fits_experimental_request'),
                           ('experimental_cumulative_budget_fits', 'fits_experimental_cumulative_budget')]:
            require(summary[key] == sum(row[field] for row in group), model+': capacity summary mismatch')
        require(summary['complete_direct_measurements'] == 12 and summary['direct_reserved_output_token_values'] == [1024], model+': measurement coverage mismatch')
        above = [row for row in group if not row['fits_native_window_including_output_and_margin']]
        results = [by_pair[model, row['sample_id']] for row in above]
        reported = payload['above_native_descriptive_subgroup'][model]
        require(reported['n'] == len(above) and reported['sample_ids'] == sorted(row['sample_id'] for row in above)
                and reported['full_credit_count'] == sum(row['full_credit'] for row in results)
                and close(reported['mean_official_score'], statistics.mean(row['official_score'] for row in results)), model+': above-native metrics mismatch')
        for condition, token_key, id_key in [
            (lambda row: True, 'maximum_source_envelope_input_tokens', 'maximum_source_sample_id'),
            (lambda row: by_pair[model, row['sample_id']]['full_credit'], 'longest_full_credit_source_envelope_input_tokens', 'longest_full_credit_sample_id')]:
            chosen = max((row for row in above if condition(row)), key=lambda r: r['source_envelope_input_tokens'], default=None)
            require(reported[token_key] == (chosen['source_envelope_input_tokens'] if chosen else None)
                    and reported[id_key] == (chosen['sample_id'] if chosen else None), model+': subgroup source extremum mismatch')
        computed[model]['above_native_n'] = len(above)
        computed[model]['above_native_full_credit'] = reported['full_credit_count']
    upper = sum((Decimal(row['estimated_generation_cost_upper_usd']) for row in rows), Decimal(0))
    require(upper == Decimal(payload['cost']['estimated_generation_cost_upper_usd'])
            and upper <= Decimal(payload['cost']['cap_usd']) <= 15, 'Global upper-estimate arithmetic mismatch')
    code = payload['provenance']['formal_source_sha256']
    require(len(code) == 29, 'Frozen source-file count changed')
    for name, value in code.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and path.suffix == '.py'
                and re.fullmatch('[0-9a-f]{64}', value), 'Invalid source fingerprint')
        require(hashlib.sha256((root/path).read_bytes()).hexdigest() == value, 'Current source differs from freeze: '+name)
    require(re.fullmatch('[0-9a-f]{64}', payload['freeze_sha256']), 'Invalid freeze hash')
    if reference is not None:
        original = {row['id']: row['context_sha256'] for row in reference['samples']}
        require(original == {key: row['context_sha256'] for key, row in sources.items()}, 'Original source IDs/hashes differ')
        for name, value in payload['provenance']['original_algorithm_source_sha256'].items():
            require(value == reference['versions']['source_sha256'][name] == code[name], 'Original algorithm identity differs')
        require(payload['provenance']['inputs_sha256'] == reference['data']['inputs_sha256']
                and payload['provenance']['labels_sha256'] == reference['data']['labels_sha256']
                and payload['provenance']['scorer_source_sha256'] == reference['versions']['official_scorer_source_sha256'], 'Dataset/scorer identity differs')
    available = sum(row['mechanism']['mode'] is not None for row in rows)
    result = {'verified': True, 'sources': 12, 'model_sample_pairs': 36, 'models': computed,
              'calls': sum(row['physical_generation_calls'] for row in rows),
              'input_tokens': sum(row['input_tokens_known'] for row in rows),
              'output_tokens': sum(row['output_tokens_known'] for row in rows),
              'charged_tokens': sum(row['charged_or_reserved_tokens'] for row in rows),
              'estimated_upper_usd': str(upper), 'source_hash_files_checked': len(code),
              'available_mechanisms': available, 'unavailable_mechanisms': 36-available}
    if integrity is not None:
        require(integrity['freeze_sha256'] == payload['freeze_sha256']
                and integrity['source_inputs_sha256'] == payload['provenance']['inputs_sha256']
                and integrity['labels_read_by_auditor'] is False and integrity['inference_models_called_by_auditor'] == 0
                and integrity['violations'] == [] and integrity['passed_available_audits'] is True, 'Independent integrity identity/scope mismatch')
        totals = integrity['totals']
        require(integrity['audited_model_sample_pairs'] == totals['terminal_pairs'] == totals['ledger_audits_passed'] == 36
                and totals['physical_requests'] == result['calls'] and totals['prompt_tokens'] == result['input_tokens']
                and totals['completion_tokens'] == result['output_tokens'] and totals['charged_tokens'] == result['charged_tokens'], 'Independent ledger totals mismatch')
        require(totals['mechanism_available'] == totals['mechanism_passed'] == available
                and totals['mechanism_failed'] == 0 and totals['mechanism_unavailable_due_to_terminal_error'] == 36-available,
                'Unavailable mechanisms counted as passes or coverage mismatch')
        require(integrity['mechanism_coverage_denominator'] == {'available': available,
                'all_terminal_pairs': 36, 'unavailable_terminal_errors': 36-available}, 'Mechanism denominator mismatch')
        audit_pairs = {(r['model'], r['id']) for r in integrity['samples']}
        require(len(integrity['samples']) == 36 and audit_pairs == pairs, 'Independent audit pair matrix mismatch')
        frozen = integrity['frozen_code_audit']
        require(frozen['declared_files'] == len(code) and not frozen['missing_snapshot_files']
                and not frozen['snapshot_hash_mismatches'] and not frozen['current_declared_file_differences'], 'Independent source audit mismatch')
        for model in MODELS:
            s, a = payload['summary'][model], integrity['summaries_by_model'][model]
            require(a['ledger_audits_passed'] == 12 and a['physical_requests'] == s['physical_generation_calls']
                    and a['prompt_tokens'] == s['input_tokens_known'] and a['completion_tokens'] == s['output_tokens_known']
                    and a['charged_tokens'] == s['charged_or_reserved_tokens'], model+': independent totals mismatch')
        result['independent_integrity_crosscheck'] = True
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='?', type=Path, default=ROOT/'benchmarks/openai_transfer_20261102.json')
    parser.add_argument('--integrity', type=Path, default=ROOT/'benchmarks/openai_transfer_integrity_20261102.json')
    args = parser.parse_args()
    payload = json.loads(args.results.read_text(encoding='utf-8'))
    reference = json.loads((ROOT/'benchmarks/oolong_v32_20261101.json').read_text(encoding='utf-8'))
    integrity = json.loads(args.integrity.read_text(encoding='utf-8')) if args.integrity.is_file() else None
    print(json.dumps(verify(payload, root=ROOT, integrity=integrity, reference=reference), indent=2))


if __name__ == '__main__':
    main()
