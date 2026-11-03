"""Export a strictly whitelisted report AFTER the frozen 36-task run is scored.

This archival exporter is outside the frozen inference source. It never opens labels or
prediction files, calls a model, imports the runner, or changes frozen inputs.
It reads the already-scored audit only after completion/access gates pass.
It refuses incomplete runs and requires the recorded formal freeze.

Later, from the repository root:
    python scripts/export_openai_transfer_report.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from urllib.parse import urlsplit, urlunsplit


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('gpt-6-sol', 'gpt-6-luna', 'gpt-6.1-terra')
MODEL_DOCUMENTATION_URLS = {
    model: f'https://developers.openai.com/api/docs/models/{model}' for model in MODELS
}
NATIVE_WINDOW = 1_050_000
EXPECTED_FREEZE = '7be98f287df0e854f6753a5811071009518e1c250b8d56cf43c33464f2e112c9'
ERROR_CODES = {'unclassified', 'invalid_api_key', 'model_not_found', 'insufficient_quota',
               'rate_limit_exceeded', 'context_length_exceeded', 'invalid_request_error',
               'unsupported_parameter', 'unsupported_value', 'server_error'}
ERROR_TYPES = {'OpenAIResponsesTransportError', 'OpenAIResponseNotCompletedError',
               'BudgetExhausted', 'BudgetIntegrityError', 'CostCapExceeded',
               'InterruptedTask', 'ModelBlocked', 'RuntimeError', 'ValueError',
               'TypeError', 'KeyError', 'TimeoutError', 'ConnectionError', 'OSError'}
TIERS = {'default', 'auto', 'flex', 'priority', 'scale', 'unknown', 'not_reported'}
PLAN_VALUES = {'records', 'facts', 'count', 'group_counts', 'mode', 'least_common',
               'compare_counts', 'earliest', 'latest', 'frequency', 'sum', 'mean',
               'min', 'max', 'conditional_group', 'label', 'date', 'date_iso',
               'user', 'count', 'argmax', 'argmin', 'none'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def is_count(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def optional_count(value):
    return value if is_count(value) else None


def optional_bool(value):
    return value if type(value) is bool else None


def safe_enum(value):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', value) else None


def safe_model(value):
    return value if isinstance(value, str) and re.fullmatch(r'gpt-[A-Za-z0-9_.-]{1,90}', value) else None


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value), 'Invalid SHA-256 identity')
    return value


def relative_code_hashes(mapping):
    output = {}
    for name, value in mapping.items():
        path = Path(name)
        require(not path.is_absolute() and '..' not in path.parts and path.suffix == '.py',
                'A source fingerprint is not a repository-relative Python path')
        output[path.as_posix()] = sha(value)
    return output


def utc(value):
    require(number(value), 'Missing timestamp')
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def wilson(successes, n):
    z = 1.959963984540054
    p = successes/n
    denominator = 1+z*z/n
    centre = (p+z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/denominator
    return [max(0.0, centre-half), min(1.0, centre+half)]


def official_urls(text):
    """Keep URL provenance, never copy an arbitrary local pricing note."""
    found = []
    for value in re.findall(r'https://[^\s<>\]\)]+', str(text)):
        parsed = urlsplit(value.rstrip('.,;'))
        if (parsed.hostname in {'developers.openai.com', 'platform.openai.com', 'openai.com'}
                and not parsed.username and not parsed.password):
            found.append(urlunsplit((parsed.scheme, parsed.netloc, parsed.path, '', '')))
    return sorted(set(found))


def error_public(error):
    if not error:
        return None
    kind = error.get('type')
    kind = kind if kind in ERROR_TYPES else 'OtherError'
    message = str(error.get('message') or '')
    match = re.search(r'status (\d{3}); code ([A-Za-z_]+)', message)
    code = match.group(2) if match and match.group(2) in ERROR_CODES else None
    status = int(match.group(1)) if match else None
    if kind == 'ModelBlocked':
        code = message if message in {'unauthorized', 'forbidden', 'model_not_found'} else None
    if status in {401, 403, 404}:
        category = 'authentication_or_model_access'
    elif status == 429:
        category = 'quota' if code == 'insufficient_quota' else 'rate_limit'
    elif status is not None and status >= 500:
        category = 'server_or_uncertain_delivery'
    elif code == 'context_length_exceeded':
        category = 'input_count_or_context_capacity'
    elif kind == 'OpenAIResponseNotCompletedError':
        category = 'received_noncompleted_or_identity_tier_failure'
    elif kind in {'BudgetExhausted', 'CostCapExceeded'}:
        category = 'resource_admission'
    elif kind == 'BudgetIntegrityError':
        category = 'usage_integrity'
    elif kind == 'InterruptedTask':
        category = 'interrupted_without_replay'
    elif kind == 'ModelBlocked':
        category = 'model_blocked'
    else:
        category = 'other_recorded_error'
    return {'type': kind, 'http_status': status, 'safe_code': code, 'category': category}


def ledger_public(record, budget):
    """Recompute conservation without retaining text or per-call raw usage."""
    ledger = record.get('ledger', [])
    require(isinstance(ledger, list) and len(ledger) <= budget['max_calls'], 'Invalid request ledger')
    totals = Counter(input_tokens_known=0, output_tokens_known=0, charged_or_reserved_tokens=0,
                     unknown_usage_requests=0, cached_input_tokens=0, cache_write_tokens=0,
                     cache_write_30m_tokens=0, cache_write_12h_tokens=0, reasoning_tokens=0)
    models, tiers, statuses, incomplete_reasons = set(), set(), Counter(), Counter()
    identity_missing = 0
    for index, entry in enumerate(ledger, 1):
        prompt, output, charged = (entry.get(key) for key in
            ('admission_prompt_tokens', 'requested_output_tokens', 'charged_tokens'))
        require(entry.get('physical_request') == index and all(is_count(v) for v in (prompt, output, charged)),
                'Nonsequential or invalid request ledger')
        require(0 < output <= budget['max_output_tokens'] and prompt+output <= budget['context_tokens'],
                'Per-call budget integrity failed')
        require(totals['charged_or_reserved_tokens']+prompt+output <= budget['total_tokens'],
                'Cumulative admission integrity failed')
        usage = entry.get('usage') or {}
        known = all(is_count(usage.get(key)) for key in ('prompt_tokens', 'completion_tokens'))
        require(entry.get('usage_known', False) is known, 'Known-usage flag mismatch')
        if known:
            inp, out = usage['prompt_tokens'], usage['completion_tokens']
            require(inp <= prompt and out <= output and charged == inp+out, 'Provider usage exceeds admission')
            totals['input_tokens_known'] += inp
            totals['output_tokens_known'] += out
        else:
            require(charged == prompt+output, 'Unknown delivery did not retain its reservation')
            totals['unknown_usage_requests'] += 1
        totals['charged_or_reserved_tokens'] += charged
        metadata = entry.get('provider_metadata') or usage.get('openai_responses_metadata') or {}
        model = safe_model(metadata.get('actual_model'))
        if model:
            models.add(model)
        elif known:
            identity_missing += 1
        tier = metadata.get('returned_service_tier')
        tiers.add(tier if tier in TIERS else ('not_reported' if tier is None else 'unknown'))
        status = metadata.get('response_status')
        statuses[status if status in {'completed', 'incomplete', 'failed', 'cancelled', 'unknown'} else 'not_reported'] += 1
        reason = metadata.get('incomplete_reason')
        if reason is not None:
            incomplete_reasons[reason if reason in {'max_output_tokens', 'content_filter'} else 'other'] += 1
        provider = metadata.get('provider_usage') or {}
        details = provider.get('input_tokens_details') or {}
        output_details = provider.get('output_tokens_details') or {}
        for source, target in [('cached_tokens', 'cached_input_tokens'), ('cache_write_tokens', 'cache_write_tokens')]:
            value = details.get(source, 0)
            require(is_count(value), 'Invalid numeric provider detail')
            totals[target] += value
        writes = details.get('cache_write_tokens_details') or {}
        for source, target in [('ttl_30m_tokens', 'cache_write_30m_tokens'), ('ttl_12h_tokens', 'cache_write_12h_tokens')]:
            value = writes.get(source, 0)
            require(is_count(value), 'Invalid numeric cache detail')
            totals[target] += value
        reasoning = output_details.get('reasoning_tokens', 0)
        require(is_count(reasoning), 'Invalid reasoning-token detail')
        totals['reasoning_tokens'] += reasoning
    require(totals['charged_or_reserved_tokens'] <= budget['total_tokens'], 'Cumulative charging integrity failed')
    supplied = record.get('usage') or {}
    require(supplied.get('physical_requests') == len(ledger)
            and supplied.get('charged_tokens') == totals['charged_or_reserved_tokens']
            and supplied.get('prompt_tokens') == totals['input_tokens_known']
            and supplied.get('completion_tokens') == totals['output_tokens_known']
            and supplied.get('within_budget') is True, 'Stored usage summary differs from ledger')
    last = ledger[-1] if ledger else {}
    last_usage = last.get('usage') or {}
    return {**dict(totals), 'physical_generation_calls': len(ledger),
            'all_usage_known': totals['unknown_usage_requests'] == 0,
            'last_request_output_reservation_tokens': optional_count(last.get('requested_output_tokens')),
            'last_request_known_input_tokens': optional_count(last_usage.get('prompt_tokens')),
            'last_request_known_output_tokens': optional_count(last_usage.get('completion_tokens')),
            'remaining_cumulative_tokens_after_task': budget['total_tokens']-totals['charged_or_reserved_tokens'],
            'actual_model_ids': sorted(models), 'returned_service_tiers': sorted(tiers),
            'response_status_counts': dict(statuses), 'incomplete_reason_counts': dict(incomplete_reasons),
            'known_usage_missing_model_identity': identity_missing,
            'within_budget': True}


def mechanism_public(result):
    """Only state/plan counters; no labels, plan values, facts, or trace text."""
    aggregate = result.get('aggregate') or {}
    index = result.get('record_index') or {}
    plan = result.get('plan') or {}
    diagnostics = result.get('planning_diagnostics') or {}
    trace = result.get('trace') or []
    actions = Counter(item.get('action') for item in trace if isinstance(item, dict))
    known, unknown = optional_count(aggregate.get('known_count')), optional_count(aggregate.get('unknown_count'))
    width = None
    for low_key, high_key in [('lower_bound', 'upper_bound'), ('lower', 'upper')]:
        low, high = aggregate.get(low_key), aggregate.get(high_key)
        if number(low) and number(high) and high >= low:
            width = high-low
            break
    return {
        'mode': safe_enum(result.get('mode')), 'stop_reason': safe_enum(result.get('stop_reason')),
        'evidence_complete': optional_bool(result.get('evidence_complete')),
        'answer_is_estimate': optional_bool(result.get('answer_is_estimate')),
        'coverage_complete': optional_bool(aggregate.get('coverage_complete')),
        'semantic_labels_independently_verified': optional_bool(result.get('semantic_labels_independently_verified')),
        'final_fact_loss_count': optional_count(result.get('final_fact_loss_count')),
        'known_records': known, 'unknown_records': unknown,
        'scope_records': optional_count(index.get('scope_records')),
        'parser_complete': optional_bool(index.get('parser_complete')),
        'aggregate_interval_width': width,
        'single_unresolved_record_at_stop': unknown == 1 and result.get('evidence_complete') is False,
        'stopped_without_new_fact_gain': result.get('stop_reason') == 'no_fact_gain',
        'plan': {key: plan.get(key) if isinstance(plan.get(key), str) and plan.get(key) in PLAN_VALUES else None
                 for key in ('kind', 'operation', 'field', 'group_by', 'reduction')},
        'semantic_filter_present': plan.get('semantic_filter') is not None,
        'planner_calls': optional_count(diagnostics.get('planner_calls')),
        'plan_repair_attempted': optional_bool(diagnostics.get('repair_attempted')),
        'plan_validation_error_count': len(diagnostics.get('validation_errors') or []),
        'classification_rounds': actions['classify_records'], 'fact_review_rounds': actions['fact_review'],
        'deterministic_finalization': actions['deterministic_finalize'] > 0,
    }


def measurement_public(record, budget):
    value = record.get('measurements') or {}
    source = optional_count(value.get('source_responses_text_input_tokens'))
    direct = optional_count(value.get('full_direct_admission_input_tokens'))
    reserved = optional_count(value.get('reserved_direct_output_tokens'))
    complete = direct is not None and reserved is not None
    require(reserved is None or reserved == 1024, 'Direct output reserve differs from the frozen control')
    total = direct+reserved if complete else None
    single = total <= budget['context_tokens'] if complete else None
    cumulative = total <= budget['total_tokens'] if complete else None
    native = total <= NATIVE_WINDOW if complete else None
    if complete:
        require(value.get('fits_registered_direct_request') is single
                and value.get('fits_registered_cumulative_budget') is cumulative,
                'Direct capacity booleans differ from their measured values')
    reasons = []
    if complete:
        if not single: reasons.append('exceeds_experimental_16384_input_plus_output')
        if not cumulative: reasons.append('exceeds_experimental_65536_cumulative_budget')
        if not native: reasons.append('exceeds_published_native_1050000_input_plus_output')
    else:
        reasons.append('measurement_unavailable')
    return {'sample_id': record['id'], 'requested_model': record['model'],
            'source_envelope_input_tokens': source,
            'direct_prompt_input_tokens': direct-budget['token_safety_margin'] if direct is not None else None,
            'direct_admission_input_tokens_including_safety_margin': direct,
            'direct_reserved_output_tokens': reserved, 'native_context_window_tokens': NATIVE_WINDOW,
            'fits_native_window_including_output_and_margin': native,
            'fits_experimental_request': single, 'fits_experimental_cumulative_budget': cumulative,
            'capacity_reasons': reasons, 'direct_inference_performed': False}


def require_publishable_strings(payload):
    forbidden_keys = {'prediction', 'gold', 'context', 'question', 'record_facts', 'facts',
                      'reasoning', 'trace', 'headers', 'api_key', 'authorization', 'sample_paths'}
    bad = re.compile(r'\bsk-[A-Za-z0-9_-]{12,}|(?i:Bearer\s+\S+)|/(?:Users|home|root|private|mnt|uniparser)/|\b(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b')
    def walk(value):
        if isinstance(value, dict):
            require(not (forbidden_keys & set(value)), 'Forbidden raw field in public output')
            for item in value.values(): walk(item)
        elif isinstance(value, list):
            for item in value: walk(item)
        elif isinstance(value, str):
            require(not bad.search(value), 'Private path or credential-like text in public output')
    walk(payload)


def build_payload(formal, root):
    # These two small completion artifacts are read before the scored audit.
    run = load_json(formal/'run_summary.json')
    require(run.get('phase') == 'inference_complete' and run.get('planned_predictions') == 36
            and run.get('terminal_predictions') == 36 and run.get('labels_opened') is False,
            'Refusing export before all 36 terminal predictions')
    access = load_json(formal/'scoring_access.json')
    require(access.get('all_36_terminal_before_label_access') is True
            and number(access.get('first_scored_at_unix')), 'Missing all-36 pre-label scoring proof')
    require((formal/'audit.json').is_file(), 'Unified scored audit does not exist')
    frozen = load_json(formal/'freeze.json')
    freeze_sha = digest(formal/'freeze.json')
    require(freeze_sha == EXPECTED_FREEZE, 'This exporter targets a different formal freeze')
    require(access.get('labels_sha256') == frozen.get('labels_sha256'), 'Scoring labels identity mismatch')
    require(access['first_scored_at_unix'] >= frozen['frozen_at_unix'], 'Scoring timestamp precedes freeze')
    config = frozen['config']
    require(config.get('models') == list(MODELS) and config.get('reasoning') == 'none', 'Registered models or effort differ')
    budget = {key: config['budget'][key] for key in ('context_tokens', 'total_tokens', 'max_calls',
                                                   'max_output_tokens', 'token_safety_margin')}
    require(budget == {'context_tokens': 16384, 'total_tokens': 65536, 'max_calls': 32,
                       'max_output_tokens': 2048, 'token_safety_margin': 16}, 'Frozen budget differs')
    code_hashes = relative_code_hashes(frozen['code_sha256'])
    require(all((root/name).is_file() and digest(root/name) == value for name, value in code_hashes.items()),
            'Frozen source changed before publication')
    reference_path = root/'benchmarks/oolong_v32_20261101.json'
    reference = load_json(reference_path)
    old_data, old_versions = reference['data'], reference['versions']
    require(frozen['inputs_sha256'] == old_data['inputs_sha256']
            and frozen['labels_sha256'] == old_data['labels_sha256']
            and frozen['scorer_sha256'] == old_versions['official_scorer_source_sha256']
            and frozen['reference_freeze_sha256'] == reference['internal_source_provenance']['freeze_json_sha256'],
            'Transfer source or original algorithm-reference identities differ')
    core_hashes = {name: value for name, value in old_versions['source_sha256'].items()
                   if name.startswith('long_context_agent/')}
    require(all(code_hashes.get(name) == value for name, value in core_hashes.items()), 'Original V3.2 core changed')
    guard = load_json(formal/'guard_verification.json')
    require(guard.get('verified') is True and guard.get('probe_read_denied') is True
            and guard.get('labels_opened') is False
            and guard.get('profile_sha256') == frozen['guard_profile_sha256'], 'OS label guard proof missing')
    audit = load_json(formal/'audit.json')
    require(audit.get('freeze_sha256') == freeze_sha, 'Scored audit freeze mismatch')
    records = audit.get('samples', [])
    ids = frozen['sample_ids']
    expected = {(model, sample_id) for model in MODELS for sample_id in ids}
    actual = {(r.get('model'), r.get('id')) for r in records}
    require(len(ids) == len(set(ids)) == 12 and len(records) == 36 and actual == expected,
            'Scored model/sample matrix is incomplete or duplicated')
    reference_samples = {sample['id']: sample for sample in reference['samples']}
    require(set(reference_samples) == set(ids), 'Original cohort IDs differ')
    money = load_json(formal/'cost_ledger.json')
    require(money.get('freeze_sha256') == freeze_sha, 'Cost ledger freeze mismatch')
    cap = Decimal(str(config['cost_cap_usd']))
    require(0 < cap <= 15, 'Frozen monetary cap differs')
    cost_groups = defaultdict(list)
    for entry in money['entries']:
        require((entry.get('model'), entry.get('id')) in expected, 'Cost reservation identity mismatch')
        amount = Decimal(entry['estimated_upper_usd'])
        require(amount.is_finite() and amount >= 0, 'Invalid estimated monetary value')
        cost_groups[entry['model'], entry['id']].append(entry)
    total_upper = sum((Decimal(entry['estimated_upper_usd']) for entry in money['entries']), Decimal(0))
    require(total_upper <= cap and total_upper == Decimal(money['estimated_upper_usd']), 'Monetary cap/conservation failure')
    results, measurements = [], []
    for model in MODELS:
        for sample_id in ids:
            record = next(r for r in records if r['model'] == model and r['id'] == sample_id)
            require(record.get('terminal') is True and record.get('method') == 'harness_v32'
                    and record.get('freeze_sha256') == freeze_sha, 'Nonterminal or inconsistent scored record')
            require(record.get('context_sha256') == frozen['source_hashes'][sample_id]
                    == reference_samples[sample_id]['context_sha256'], 'Source hash mismatch')
            score = record.get('official_score')
            require(number(score) and 0 <= score <= 1 and record.get('full_credit') is (score == 1)
                    and type(record.get('parsed_exact_match')) is bool, 'Invalid official metrics')
            tokens = ledger_public(record, budget)
            price = config['prices_per_million'][model]
            entries = cost_groups[model, sample_id]
            upper = sum((Decimal(e['estimated_upper_usd']) for e in entries), Decimal(0))
            require(upper >= Decimal(tokens['charged_or_reserved_tokens'])*Decimal(str(price['r_max']))/Decimal(1_000_000),
                    'Monetary reservations fail to cover token charges')
            base = (Decimal(tokens['input_tokens_known'])*Decimal(str(price['input']))
                    + Decimal(tokens['output_tokens_known'])*Decimal(str(price['output'])))/Decimal(1_000_000)
            counts = record.get('count_audit', [])
            error = error_public(record.get('error'))
            if (error and error['type'] == 'OpenAIResponseNotCompletedError'
                    and tokens['response_status_counts'].get('incomplete', 0)
                    and tokens['incomplete_reason_counts'].get('max_output_tokens', 0)):
                error['category'] = 'received_output_limit_incomplete'
            result = {'sample_id': sample_id, 'requested_model': model,
                'official_score': float(score), 'full_credit': score == 1,
                'parsed_exact_match': record['parsed_exact_match'],
                'terminal': True, 'nonempty_output': bool(str(record.get('prediction') or '').strip()),
                **tokens, 'error': error, 'mechanism': mechanism_public(record.get('result') or {}),
                'count_queries_including_cache_hits': len(counts),
                'count_api_requests': sum(entry.get('cache_hit') is False for entry in counts),
                'count_api_errors': sum(entry.get('status') == 'error' for entry in counts),
                'estimated_generation_cost_upper_usd': str(upper),
                'estimated_base_io_cost_on_known_usage_usd': str(base),
                'unknown_usage_cost_reservations': sum(e.get('usage_known') is not True
                    and e.get('status') != 'not_sent' for e in entries)}
            results.append(result)
            measurements.append(measurement_public(record, budget))
    summaries = {}
    for model in MODELS:
        rows = [row for row in results if row['requested_model'] == model]
        original = audit['summary'][model]
        blocked = original.get('status') == 'model_blocked_not_an_accuracy_estimate'
        correct = sum(row['full_credit'] for row in rows)
        average = statistics.mean(row['official_score'] for row in rows)
        require(blocked or (original['full_credit'] == correct
                and math.isclose(original['mean_official_score'], average, abs_tol=1e-12)
                and original['parsed_exact_match'] == sum(row['parsed_exact_match'] for row in rows)),
                'Published metric recomputation differs from scored audit')
        totals = {key: sum(row[key] for row in rows) for key in (
            'physical_generation_calls', 'input_tokens_known', 'output_tokens_known',
            'charged_or_reserved_tokens', 'unknown_usage_requests', 'cached_input_tokens',
            'cache_write_tokens', 'cache_write_30m_tokens', 'cache_write_12h_tokens', 'reasoning_tokens',
            'count_queries_including_cache_hits', 'count_api_requests', 'count_api_errors',
            'unknown_usage_cost_reservations')}
        mechanisms = [row['mechanism'] for row in rows]
        summary = {'status': 'model_blocked_not_an_accuracy_estimate' if blocked else 'scored',
            'n_planned': 12, 'n_terminal': 12, 'n_nonempty': sum(row['nonempty_output'] for row in rows),
            'full_credit_count': None if blocked else correct, 'full_credit_rate': None if blocked else correct/12,
            'full_credit_wilson_95_interval': None if blocked else wilson(correct, 12),
            'mean_official_score': None if blocked else average,
            'parsed_exact_match_count': None if blocked else sum(row['parsed_exact_match'] for row in rows),
            **totals, 'recorded_errors': sum(row['error'] is not None for row in rows),
            'error_type_counts': dict(Counter(row['error']['type'] for row in rows if row['error'])),
            'error_category_counts': dict(Counter(row['error']['category'] for row in rows if row['error'])),
            'response_status_counts': dict(sum((Counter(row['response_status_counts']) for row in rows), Counter())),
            'incomplete_reason_counts': dict(sum((Counter(row['incomplete_reason_counts']) for row in rows), Counter())),
            'actual_model_ids': sorted({name for row in rows for name in row['actual_model_ids']}),
            'returned_service_tiers': sorted({name for row in rows for name in row['returned_service_tiers']}),
            'all_usage_known': all(row['all_usage_known'] for row in rows),
            'all_within_budget': True,
            'estimated_generation_cost_upper_usd': str(sum((Decimal(row['estimated_generation_cost_upper_usd']) for row in rows), Decimal(0))),
            'estimated_base_io_cost_on_known_usage_usd': str(sum((Decimal(row['estimated_base_io_cost_on_known_usage_usd']) for row in rows), Decimal(0))),
            'mechanisms': {
                'mode_counts': dict(Counter(item['mode'] or 'unavailable' for item in mechanisms)),
                'stop_reason_counts': dict(Counter(item['stop_reason'] or 'unavailable' for item in mechanisms)),
                'operation_counts': dict(Counter(item['plan']['operation'] or 'unavailable' for item in mechanisms)),
                'plan_validation_errors': sum(item['plan_validation_error_count'] for item in mechanisms),
                'plan_repairs': sum(item['plan_repair_attempted'] is True for item in mechanisms),
                'single_unresolved_record_at_stop': sum(item['single_unresolved_record_at_stop'] for item in mechanisms),
                'no_fact_gain_stops': sum(item['stopped_without_new_fact_gain'] for item in mechanisms),
                'recorded_final_fact_loss_total': sum(item['final_fact_loss_count'] or 0 for item in mechanisms),
                'final_fact_loss_metric_present': sum(item['final_fact_loss_count'] is not None for item in mechanisms)}}
        summaries[model] = summary
    capacity, above_native = {}, {}
    for model in MODELS:
        rows = [row for row in measurements if row['requested_model'] == model]
        def extent(key):
            values = [row[key] for row in rows if row[key] is not None]
            return [min(values), max(values)] if values else None
        capacity[model] = {'source_envelope_input_token_range': extent('source_envelope_input_tokens'),
            'direct_admission_input_token_range_including_safety_margin': extent('direct_admission_input_tokens_including_safety_margin'),
            'native_window_fits': sum(row['fits_native_window_including_output_and_margin'] is True for row in rows),
            'experimental_request_fits': sum(row['fits_experimental_request'] is True for row in rows),
            'experimental_cumulative_budget_fits': sum(row['fits_experimental_cumulative_budget'] is True for row in rows),
            'complete_direct_measurements': sum(row['fits_native_window_including_output_and_margin'] is not None for row in rows),
            'direct_reserved_output_token_values': sorted({row['direct_reserved_output_tokens'] for row in rows if row['direct_reserved_output_tokens'] is not None})}
        selected_ids = {row['sample_id'] for row in rows
                        if row['fits_native_window_including_output_and_margin'] is False}
        selected_results = [row for row in results if row['requested_model'] == model
                            and row['sample_id'] in selected_ids]
        selected_measurements = [row for row in rows if row['sample_id'] in selected_ids]
        full_ids = {row['sample_id'] for row in selected_results if row['full_credit']}
        longest_full = max((row for row in selected_measurements if row['sample_id'] in full_ids
                            and row['source_envelope_input_tokens'] is not None),
                           key=lambda row: row['source_envelope_input_tokens'], default=None)
        maximum = max((row for row in selected_measurements if row['source_envelope_input_tokens'] is not None),
                      key=lambda row: row['source_envelope_input_tokens'], default=None)
        above_native[model] = {
            'definition': 'Direct admission input including safety margin plus recorded output reserve exceeds 1050000',
            'interpretation': 'Descriptive token-capacity-defined subgroup, separate from all 12; no direct-answer comparison',
            'n': len(selected_results), 'sample_ids': sorted(selected_ids),
            'full_credit_count': sum(row['full_credit'] for row in selected_results),
            'mean_official_score': statistics.mean(row['official_score'] for row in selected_results) if selected_results else None,
            'longest_full_credit_source_envelope_input_tokens': longest_full['source_envelope_input_tokens'] if longest_full else None,
            'longest_full_credit_sample_id': longest_full['sample_id'] if longest_full else None,
            'maximum_source_envelope_input_tokens': maximum['source_envelope_input_tokens'] if maximum else None,
            'maximum_source_sample_id': maximum['sample_id'] if maximum else None,
        }
    payload = {
        'schema_version': 1, 'experiment_id': 'openai_transfer_20261102', 'experiment_date': '2026-11-02',
        'method_name': 'CRU-Harness V3.2', 'method_id': 'harness_v32',
        'status': 'all_36_terminal_and_unified_scored', 'freeze_sha256': freeze_sha,
        'frozen_at_utc': utc(frozen['frozen_at_unix']), 'scored_at_utc': utc(access['first_scored_at_unix']),
        'requested_models': list(MODELS), 'n_distinct_source_contexts': 12, 'n_model_sample_pairs': 36,
        'reused_cohort': True, 'new_blind_holdout': False,
        'budget': budget, 'wire_parameters': {'api': 'responses', 'reasoning_effort': 'none',
            'store': False, 'service_tier': 'default', 'truncation': 'disabled',
            'temperature_sent': False, 'seed_sent': False, 'implicit_retries': 0},
        'provenance': {'benchmark': 'OOLONG-synth', 'upstream_repository': 'https://github.com/abertsch72/oolong',
            'dataset_repository': old_data['dataset'], 'source_split': old_data['source_split'],
            'dataset_revision': old_data['dataset_revision'], 'parquet_revision': old_data['parquet_revision'],
            'scorer_commit': old_versions['official_scorer_commit'], 'scorer_source_sha256': frozen['scorer_sha256'],
            'inputs_sha256': frozen['inputs_sha256'], 'labels_sha256': frozen['labels_sha256'],
            'original_qwen_experiment_id': reference['experiment_id'],
            'original_qwen_public_summary_sha256': digest(reference_path),
            'reference_algorithm_freeze_sha256': frozen['reference_freeze_sha256'],
            'original_algorithm_source_sha256': core_hashes, 'formal_source_sha256': code_hashes,
            'local_audit_sha256': digest(formal/'audit.json'), 'local_cost_ledger_sha256': digest(formal/'cost_ledger.json'),
            'local_scoring_access_sha256': digest(formal/'scoring_access.json'),
            'exporter_sha256': digest(Path(__file__)), 'raw_artifacts_distributed': False},
        'metric_definitions': {'official_score': 'Upstream OOLONG score on [0,1]; numeric score is 0.75**abs(error)',
            'full_credit': 'official_score == 1', 'parsed_exact_match': 'Official answer-parser strict equality, separate from full credit',
            'denominator': '12 per requested model, including failed, empty and unknown outcomes',
            'source_count': 'Exact complete Responses source-text input envelope, including formatting; not Qwen raw-text tokenizer tokens',
            'direct_count': 'Exact full direct prompt with budget notice; admission adds 16 safety tokens and the separately reported output reserve',
            'sampling_units': 'The same 12 distinct source contexts are reused across models; distinct IDs and hashes do not establish statistical independence, and 36 model/sample pairs are not 36 independent observations',
            'wilson_interval': 'Descriptive nominal 95% Wilson intervals using 12 context units per model; underlying record overlap and dependence are not excluded'},
        'capacity_definition': {'published_native_context_tokens_per_model': NATIVE_WINDOW,
            'frozen_direct_reserved_output_tokens': 1024,
            'native_source': 'Official model metadata verified before run; distinct from the actively imposed 16k window',
            'native_model_catalog': 'https://developers.openai.com/api/docs/models',
            'official_model_pages': dict(MODEL_DOCUMENTATION_URLS),
            'direct_inference_performed': False, 'model_attention_window_changed': False},
        'summary': summaries, 'capacity_summary': capacity, 'above_native_descriptive_subgroup': above_native,
        'sources': [{'sample_id': sample_id, 'context_sha256': frozen['source_hashes'][sample_id],
                     'nominal_dataset_tokens': reference_samples[sample_id]['nominal_dataset_tokens'],
                     'task_group': reference_samples[sample_id]['task_group'],
                     'historical_qwen_raw_source_tokens': reference_samples[sample_id]['qwen_source_tokens']}
                    for sample_id in ids],
        'measurements': measurements, 'results': results,
        'cost': {'currency': 'USD', 'definition': 'Conservative model-generation estimate, NOT an invoice',
            'cap_usd': str(cap), 'estimated_generation_cost_upper_usd': str(total_upper),
            'prices_usd_per_million_tokens': {model: {key: value for key, value in config['prices_per_million'][model].items()
                if key in {'input', 'output', 'r_max', 'cached_input', 'cache_write_30m', 'cache_write_12h'}} for model in MODELS},
            'official_pricing_urls': official_urls(config.get('pricing_source')),
            'admission': '(admitted_input + reserved_output) * r_max / 1e6; persisted before generation',
            'known_usage': '(actual_input + actual_output) * r_max / 1e6; base I/O estimate shown separately',
            'unknown_delivery': 'Entire monetary and token reservation retained',
            'cache_and_reasoning_details': 'Subsets of input/output, never added twice',
            'count_requests': 'Audited separately; not generation calls or generation-token usage'},
        'separate_preflight': {'minimal_api_availability_probe': {'input_tokens': 33, 'output_tokens': 15,
            'total_tokens': 48, 'reasoning_tokens': 0, 'provenance': 'Parent-task reported three minimal API probes'},
            'included_in_formal_scores_or_generation_cost': False,
            'other_integration_probes_included': False},
        'integrity': {'complete_36_pair_set': True, 'all_36_terminal_before_label_access': True,
            'official_scoring_and_independent_parser_agree': True, 'os_label_guard_probe_denied': True,
            'os_guard_limit': 'Probe/profile verification, not a complete OS file-access trace',
            'all_token_ledgers_conserve': True, 'all_estimated_cost_reservations_cover_token_charges': True,
            'current_formal_source_hashes_match_freeze': True, 'original_v32_algorithm_hashes_match': True,
            'model_identity_matches': all(not row['actual_model_ids'] or row['actual_model_ids'] == [row['requested_model']] for row in results),
            'known_usage_missing_model_identity': sum(row['known_usage_missing_model_identity'] for row in results),
            'nondefault_returned_service_tiers': sorted({tier for row in results for tier in row['returned_service_tiers']
                if tier not in {'default', 'not_reported'}})},
        'limitations': ['Previously consumed cohort 5 transfer replay, not a new blind holdout',
            'One run per model/source and 12 distinct source contexts; statistical independence is not established and model/sample pairs are not pooled as n=36 independent observations',
            'Harness-only generation: no causal uplift estimate against direct, RAG, RLM or MemGPT',
            'Direct checks measure capacity/admission only; rejection is not a wrong semantic answer',
            'Native weights and attention windows are unchanged; source envelopes differ from Qwen raw token counts',
            'No new RLM/MemGPT comparison or official leaderboard submission',
            'Public-data pretraining contamination and underlying record overlap are not excluded',
            'Plan, unknown-count and stop counters do not independently verify semantic labels or prove why a record remained unknown']}
    require_publishable_strings(payload)
    return payload


def score_text(value):
    return 'N/A' if value is None else f'{100*value:.2f}'


def range_text(value):
    return 'N/A' if value is None else f'{value[0]:,}–{value[1]:,}'


def report_markdown(payload, *, chinese=False):
    p = payload
    model_sources = ' · '.join(
        f'[{model}]({MODEL_DOCUMENTATION_URLS[model]})' for model in MODELS)
    if chinese:
        lines = ['# CRU-Harness V3.2 三模型迁移评测', '', '**简体中文** | [English](openai_transfer_20261102.md)', '',
            '在同一冻结 V3.2 算法下，对既有 OOLONG-synth cohort 5 的 12 个来源分别运行三个指定 OpenAI 模型。全部 36 个模型—样本对完成后统一评分。这里报告迁移重放成绩，不是新的盲测，也没有 direct 生成比较臂。', '',
            '## 逐模型成绩', '', '| 模型 | 满分题/12 | 满分率 | 官方均分/100 | 严格EM/12 | 非空 | 错误 | 描述性Wilson95%来源单位区间 |',
            '|---|---:|---:|---:|---:|---:|---:|---|']
    else:
        lines = ['# CRU-Harness V3.2 OpenAI transfer evaluation', '', '**English** | [简体中文](openai_transfer_20261102.zh-CN.md)', '',
            'The same frozen V3.2 algorithm was replayed on the 12 previously used OOLONG-synth cohort-5 sources with three specified OpenAI models. All 36 model/sample pairs became terminal before unified scoring. This is a transfer replay, not a new blind holdout; there is no direct-generation comparison arm.', '',
            '## Results by model', '', '| Model | Full /12 | Full-credit rate | Official mean /100 | Strict EM /12 | Nonempty | Errors | Descriptive Wilson 95% context-unit interval |',
            '|---|---:|---:|---:|---:|---:|---:|---|']
    for model in MODELS:
        s = p['summary'][model]
        ci = s['full_credit_wilson_95_interval']
        interval = 'N/A' if ci is None else f'{100*ci[0]:.1f}%–{100*ci[1]:.1f}%'
        rate = 'N/A' if s['full_credit_rate'] is None else f'{100*s["full_credit_rate"]:.2f}%'
        lines.append(f'| {model} | {s["full_credit_count"] if s["full_credit_count"] is not None else "N/A"} | {rate} | {score_text(s["mean_official_score"])} | {s["parsed_exact_match_count"] if s["parsed_exact_match_count"] is not None else "N/A"} | {s["n_nonempty"]}/12 | {s["recorded_errors"]} | {interval} |')
    lines += ['', ('官方均分包含数字题 `0.75 ** abs(error)` 的部分分，不等同于准确率。全部失败、空输出和未知留在每模型 12 题分母中。本轮有12个不同来源context，跨模型重用形成36对；不同ID及全文哈希不证明统计独立，底层记录重用未排除。Wilson区间仅按12个来源单位作描述性展示，不能据此确认稳定领先。' if chinese else
                  'The official mean includes numeric partial credit `0.75 ** abs(error)` and is not an accuracy percentage. Failed, empty and unknown outcomes remain in each model’s denominator of 12. The 12 distinct source contexts are reused across models to form 36 pairs. Distinct IDs and hashes do not establish statistical independence; underlying record overlap is not excluded. Wilson intervals are descriptive context-unit summaries and do not confirm stable superiority.'), '',
              '## 用量与费用' if chinese else '## Usage and estimated cost', '',
              ('| 模型 | 生成调用 | 已知输入tokens | 已知输出tokens | 计费或预留tokens | 未知usage请求 | 费用估计上界USD | 已知usage基础I/O估计USD |' if chinese else
               '| Model | Generation calls | Known input tokens | Known output tokens | Charged/reserved tokens | Unknown-usage requests | Estimated upper USD | Known-usage base I/O estimate USD |'),
              '|---|---:|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        s = p['summary'][model]
        lines.append(f'| {model} | {s["physical_generation_calls"]} | {s["input_tokens_known"]:,} | {s["output_tokens_known"]:,} | {s["charged_or_reserved_tokens"]:,} | {s["unknown_usage_requests"]} | {float(s["estimated_generation_cost_upper_usd"]):.6f} | {float(s["estimated_base_io_cost_on_known_usage_usd"]):.6f} |')
    lines += ['', (f'正式生成的保守费用估计总上界为 **${float(p["cost"]["estimated_generation_cost_upper_usd"]):.6f}**，冻结上限 $15。它不是实际账单。每次先按 `(准入输入+预留输出)×r_max` 持久化预留，已知usage按实际输入输出结算同一包络；未知交付保留整额。基础I/O估计只覆盖已知usage，缓存、cache-write和reasoning细分不重复加到总量。单次16k低于272k长输入档阈值；非default服务档等偏离需单列。' if chinese else
                  f'The formal generation-cost conservative estimate totals **${float(p["cost"]["estimated_generation_cost_upper_usd"]):.6f}**, under the frozen $15 cap. It is not an invoice. Each call reserves `(admitted input + reserved output) × r_max` before generation; known usage settles actual input/output in the same envelope, while uncertain delivery retains the full reservation. Base I/O estimates cover known usage only. Cache, cache-write and reasoning details are not added twice. The 16k request limit is below the 272k long-input pricing threshold; nondefault tiers or other pricing deviations must be reported separately.'), '',
              (('Standard价格、长输入阈值及冻结单价包络的官方来源：' if chinese else
                'Official sources for Standard prices, the long-input threshold and the frozen per-token price envelope: ')
               + model_sources), '',
              ('独立的最小API可用性预检为33输入+15输出=48 tokens，不计入正式成绩或上述费用。计数请求另行统计，不充当生成调用。' if chinese else
               'The separate minimal API-availability preflight used 33 input + 15 output = 48 tokens and is excluded from the formal scores and costs above. Token-count requests are audited separately and are not generation calls.'), '',
              '## 身份与计数' if chinese else '## Model identity and counting', '',
              ('| 请求模型 | 实际返回模型集合 | 返回服务档集合 | 计数API请求 | 计数缓存命中 |' if chinese else
               '| Requested model | Actual model set | Returned tier set | Counting API requests | Counting cache hits |'), '|---|---|---|---:|---:|']
    for model in MODELS:
        s = p['summary'][model]
        lines.append(f'| {model} | {", ".join(s["actual_model_ids"]) or "N/A"} | {", ".join(s["returned_service_tiers"]) or "N/A"} | {s["count_api_requests"]} | {s["count_queries_including_cache_hits"]-s["count_api_requests"]} |')
    lines += ['', ('| 模型 | Provider response_status计数 | Provider incomplete_reason计数 |' if chinese else
                   '| Model | Provider response_status counts | Provider incomplete_reason counts |'),
              '|---|---|---|']
    for model in MODELS:
        s = p['summary'][model]
        statuses = ', '.join(f'{key}: {value}' for key, value in sorted(s['response_status_counts'].items())) or 'none'
        reasons = ', '.join(f'{key}: {value}' for key, value in sorted(s['incomplete_reason_counts'].items())) or 'none'
        lines.append(f'| {model} | {statuses} | {reasons} |')
    lines += ['', ('`incomplete/max_output_tokens`表示服务已返回但未完成，可能发生于预算约束下最后一次较小的输出预留；若usage已知，其真实输入输出仍计入账本。它与HTTP失败、未知usage或一个语义错误答案分开报告，不能据此推断模型理解能力。' if chinese else
                   '`incomplete/max_output_tokens` is a received but unfinished response and can occur when the last output reservation is constrained by the remaining budget. Known actual input/output still enters the ledger. It is distinguished from HTTP failures, unknown usage and an incorrect semantic answer; it does not by itself diagnose model understanding.')]
    lines += ['', '## 完整来源与direct容量控制' if chinese else '## Full-source and direct-capacity controls', '',
              ('| 模型 | 来源输入封装tokens范围 | direct准入输入tokens范围 | 原生窗可容纳 | 16k可容纳 | 65k累计可容纳 | 完整计量 |' if chinese else
               '| Model | Source-envelope token range | Direct admission-input range | Native-window fits | 16k fits | 65k cumulative fits | Complete measurements |'),
              '|---|---:|---:|---:|---:|---:|---:|']
    for model in MODELS:
        c = p['capacity_summary'][model]
        lines.append(f'| {model} | {range_text(c["source_envelope_input_token_range"])} | {range_text(c["direct_admission_input_token_range_including_safety_margin"])} | {c["native_window_fits"]}/12 | {c["experimental_request_fits"]}/12 | {c["experimental_cumulative_budget_fits"]}/12 | {c["complete_direct_measurements"]}/12 |')
    reserves = sorted({m['direct_reserved_output_tokens'] for m in p['measurements'] if m['direct_reserved_output_tokens'] is not None})
    reserve_text = ', '.join(f'{value:,}' for value in reserves) or 'N/A'
    lines += ['', (f'这里区分原生 **1,050,000 tokens**、主动施加的单次16,384和累计65,536。direct可容纳性使用完整提示准入输入（包含16-token安全余量）加实际预留输出 **{reserve_text}**；预留不是自动按2,048输出上限填写。API测量是包含格式的Responses输入封装，不等于历史Qwen无特殊token原文计数。没有执行direct生成，准入拒绝不等于答错，也不能填写direct准确率0%。模型权重及单次注意力窗口没有改变。' if chinese else
                  f'The published native window of **1,050,000 tokens** is separate from the actively imposed 16,384 per-call and 65,536 cumulative limits. Direct admission uses the full prompt (including the 16-token safety margin) plus its actual reserved output **{reserve_text}**, not an assumed 2,048-token output cap. Counts describe complete Responses input envelopes, unlike historical Qwen raw-text counts. No direct generation was run: admission rejection is not a wrong answer and does not imply 0% direct accuracy. Model weights and attention windows are unchanged.'), '',
              (('三个精确模型的原生窗口规格来源：' if chinese else
                'Native-window specifications for the three exact model IDs: ')
               + model_sources), '',
              '### 超原生窗口描述性子组' if chinese else '### Descriptive above-native subgroup', '',
              ('按每模型 `direct准入输入+实际预留输出1,024 > 1,050,000` 定义，属于容量计量子组，与全部12题分开。这里不沿用Qwen扩容倍数，也不代表direct答错。' if chinese else
               'Defined per model by `direct admission input + recorded 1,024 output reserve > 1,050,000`, this is a capacity-defined subgroup, separate from all 12 tasks. Qwen expansion ratios are not transferred, and no direct wrong-answer claim is made.'), '',
              ('| 模型 | 子组n | 满分题 | 官方均分/100 | 最长满分来源封装tokens | 子组最大来源封装tokens |' if chinese else
               '| Model | Subgroup n | Full credit | Official mean /100 | Longest full-credit source envelope | Maximum subgroup source envelope |'),
              '|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        group = p['above_native_descriptive_subgroup'][model]
        longest = group['longest_full_credit_source_envelope_input_tokens']
        maximum = group['maximum_source_envelope_input_tokens']
        lines.append(f'| {model} | {group["n"]} | {group["full_credit_count"]} | {score_text(group["mean_official_score"])} | {longest if longest is not None else "N/A"} | {maximum if maximum is not None else "N/A"} |')
    lines += ['',
              '## 逐来源成绩 调用 tokens' if chinese else '## Per-source score calls and tokens', '',
              ('每格为官方分/100 · 生成调用 · 计费或预留tokens。极小正分可能显示为0.00，JSON保留精确值。' if chinese else
               'Each cell is official score/100 · generation calls · charged/reserved tokens. Very small positive scores may display as 0.00; the JSON retains their exact values.'), '',
              '| Sample ID | gpt-6-sol | gpt-6-luna | gpt-6.1-terra |', '|---|---:|---:|---:|']
    lookup = {(row['requested_model'], row['sample_id']): row for row in p['results']}
    for source in p['sources']:
        cells = []
        for model in MODELS:
            r = lookup[model, source['sample_id']]
            cells.append(f'{score_text(r["official_score"])} · {r["physical_generation_calls"]} · {r["charged_or_reserved_tokens"]:,}')
        lines.append('| '+source['sample_id']+' | '+' | '.join(cells)+' |')
    lines += ['', '## 计划与停止诊断' if chinese else '## Plan and stopping diagnostics', '',
              ('| 模型 | plan验证错误 | plan修复 | 无新增事实停止 | 停止时仅1条未知 | 记录的最终事实丢失 |' if chinese else
               '| Model | Plan-validation errors | Plan repairs | No-fact-gain stops | One unresolved record at stop | Recorded final fact losses |'),
              '|---|---:|---:|---:|---:|---:|']
    for model in MODELS:
        m = p['summary'][model]['mechanisms']
        lines.append(f'| {model} | {m["plan_validation_errors"]} | {m["plan_repairs"]} | {m["no_fact_gain_stops"]} | {m["single_unresolved_record_at_stop"]} | {m["recorded_final_fact_loss_total"]} ({m["final_fact_loss_metric_present"]}/12 measured) |')
    lines += ['', ('这些诊断只来自计划类型、聚合未知量及停止计数，不包含语义标签、候选值或记录正文。单条未知尚存不证明已经定位语义错误；证据计算闭合也不提供独立语义真值保证。逐题错误类型、计数失败及安全HTTP code见JSON。' if chinese else
                  'These diagnostics use plan types, aggregate unknown counts and stop counters only, without labels, candidate values or record text. One unresolved record is not proof of a localized semantic error; calculation closure does not independently verify semantic truth. Per-task error types, counting failures and safe HTTP codes are in the JSON.'), '',
              '## 冻结协议与解释边界' if chinese else '## Frozen protocol and interpretation', '',
              ('Responses使用reasoning=none、store=false、truncation=disabled，请求default服务档，不发送temperature或seed；不宣称确定性。每模型—样本独立共享16k/65k/32次/2k输出账本。12个来源已用于此前V3.2实验，本轮不据gold调优；只有新增传输与美元预算桥。' if chinese else
               'Responses uses reasoning=none, store=false, truncation=disabled and requested default service tier. Temperature and seed are not sent; determinism is not claimed. Each model/sample has its own shared 16k/65k/32-call/2k-output ledger. These sources were already used in the V3.2 experiment; the transfer does not tune against gold. Only the transport and monetary bridge were added.'), '',
              ('本轮没有新跑direct、RLM、MemGPT或RAG，不能估计Harness对各底座的因果增益、宣称跨模型普遍领先或无限准确上下文；不是官方排行榜提交。旧Qwen成绩不并入本轮分母。公开数据预训练接触和底层记录重复未排除。' if chinese else
               'No new direct, RLM, MemGPT or RAG arm was run. This design cannot estimate causal Harness uplift, establish general cross-model superiority or unlimited accurate context, and is not an official leaderboard submission. Historical Qwen scores are not pooled into this denominator. Pretraining exposure and overlap among underlying records remain unexcluded.'), '',
              f'Freeze SHA-256: `{p["freeze_sha256"]}`', '',
              ('来源ID/哈希、数据revision、官方scorer commit、原算法和新桥源码哈希、36条白名单结果、完整计数与费用定义见[公开JSON](../benchmarks/openai_transfer_20261102.json)。原始问题、答案、事实账本、响应轨迹和本机路径不随报告分发。' if chinese else
               'Source IDs/hashes, dataset revisions, official scorer commit, original algorithm and bridge source hashes, all 36 whitelisted results, measurements and cost definitions are in the [public JSON](../benchmarks/openai_transfer_20261102.json). Questions, answers, fact ledgers, raw responses and private paths are not distributed.'), '',
              '[OOLONG](https://github.com/abertsch72/oolong) · [Dataset](https://huggingface.co/datasets/oolongbench/oolong-synth) · [Token counting](https://developers.openai.com/api/docs/guides/token-counting)', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--formal', type=Path, default=ROOT/'work/openai_transfer_20261102/formal')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    root, formal = args.root.resolve(), args.formal.resolve()
    payload = build_payload(formal, root)
    english = report_markdown(payload)
    chinese = report_markdown(payload, chinese=True)
    require_publishable_strings([english, chinese])
    outputs = {
        root/'benchmarks/openai_transfer_20261102.json': json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False)+'\n',
        root/'docs/openai_transfer_20261102.md': english,
        root/'docs/openai_transfer_20261102.zh-CN.md': chinese,
    }
    require(all(not path.exists() for path in outputs), 'Refusing to replace an existing published report')
    # All validation/rendering precedes the first output write.
    for path, text in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix+'.tmp')
        temp.write_text(text, encoding='utf-8')
        temp.replace(path)
    print(json.dumps({'exported': [str(path.relative_to(root)) for path in outputs],
                      'model_sample_pairs': 36, 'distinct_source_contexts': 12}))


if __name__ == '__main__':
    main()
