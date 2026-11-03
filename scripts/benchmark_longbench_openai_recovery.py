"""A separately frozen recovery of zero-generation infrastructure failures.

The original 75 records are immutable. Only the fixed 39 generic network
failures with no generation ledger or token charges can be overlaid. Counting
may retry twice with recorded 5/15-second delays; generations are never
implicitly retried. Labels remain closed until original 75 + recovery 39
terminal records pass integrity checks.
"""
from __future__ import annotations

import argparse
from collections import Counter
from decimal import Decimal
import fcntl
import getpass
import io
import json
import os
from pathlib import Path
import random
import re
import shutil
import statistics
import subprocess
import sys
import time
from urllib.error import HTTPError

from long_context_agent.experiment_budget import ExperimentBudget
from long_context_agent.openai_longbench_bridge import LongBenchResponsesTransport, run_longbench_harness
from long_context_agent.openai_responses_transport import OpenAIResponsesBudgetedClient, OpenAIResponsesTransportError
from scripts import benchmark_longbench_openai_limits as base


ROOT = base.ROOT
PARENT_FREEZE = '941c86120e66020439571fbe8888999c643ebcf83fc315d7ec7e1e0e1045208d'
EXPECTED_ELIGIBLE = 39
NETWORK_MESSAGE = 'OpenAI request failed; response or usage may be unknown'
RETRY_DELAYS = [5, 15]
SAFE_HTTP_CODES = {'context_length_exceeded', 'invalid_request_error', 'invalid_api_key',
    'model_not_found', 'insufficient_quota', 'rate_limit_exceeded', 'unsupported_parameter',
    'unsupported_value', 'server_error', 'string_above_max_length', 'invalid_value', 'invalid_type'}
NUMERIC_ERROR_PATTERNS = {
    'maximum_context_length_tokens': (
        r'(?:maximum|max)\s+context\s+(?:length|window)\s*(?:is|of|:|=)?\s*([\d,]+)\s*tokens',
        r'max_model_len\s*(?:is|of|:|=)?\s*([\d,]+)',
        r'(?:supports?|allows?)\s+(?:at\s+most|up\s+to)\s*([\d,]+)\s*tokens'),
    'requested_total_tokens': (
        r'(?:you\s+)?requested\s+(?:a\s+total\s+of\s+)?([\d,]+)\s*tokens',
        r'(?:requested|request)\s+(?:total\s+)?tokens\s*[:=]\s*([\d,]+)'),
    'input_tokens': (
        r'([\d,]+)\s*(?:tokens\s+)?in\s+(?:your\s+|the\s+)?(?:messages|prompt|input)',
        r'(?:messages|prompt|input)\s+(?:resulted\s+in|results\s+in|contains?|has)\s*([\d,]+)\s*tokens',
        r'input_tokens\s*[:=]\s*([\d,]+)'),
    'completion_tokens': (
        r'([\d,]+)\s*(?:tokens\s+)?(?:in|for)\s+(?:your\s+|the\s+)?(?:completion|output)',
        r'(?:completion|output)_tokens\s*[:=]\s*([\d,]+)'),
}


def safe_http_error(payload, status):
    error = payload.get('error') if isinstance(payload, dict) else {}
    error = error if isinstance(error, dict) else {}
    message = error.get('message') if isinstance(error.get('message'), str) else ''
    fields = {}
    for key, patterns in NUMERIC_ERROR_PATTERNS.items():
        values = set()
        for pattern in patterns:
            for match in re.finditer(pattern, message, flags=re.IGNORECASE):
                try:
                    number = int(match.group(1).replace(',', ''))
                except ValueError:
                    continue
                if 0 <= number <= 2_147_483_647:
                    values.add(number)
        if values:
            fields[key] = sorted(values)
    parameter = error.get('param') if isinstance(error.get('param'), str) else None
    if parameter not in {'input', 'messages', 'model', 'max_tokens', 'max_output_tokens', 'reasoning', 'reasoning.effort', 'truncation'}:
        parameter = parameter if parameter and re.fullmatch(
            r'(?:input|messages)(?:\[\d{1,4}\])?(?:\.(?:content|text|role))(?:\[\d{1,4}\])?(?:\.text)?',
            parameter) else None
    code = error.get('code') if isinstance(error.get('code'), str) else None
    return {'http_status': int(status), 'code': code if code in SAFE_HTTP_CODES else None,
            'param': parameter, 'numeric_candidates': fields,
            'unique_numeric_fields': {key: values[0] for key, values in fields.items() if len(values) == 1}}


def _safe_exception_class(value):
    name = type(value).__name__
    return name if re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]{0,79}', name) else 'OtherError'


class ErrorObservingOpener:
    """Transparent observer: replays exactly the prefix the base parser reads."""

    def __init__(self, delegate, events):
        self.delegate, self.events = delegate, events

    def open(self, request, *args, **kwargs):
        url = request.full_url
        stage = 'input_token_count' if url.endswith('/responses/input_tokens') else 'generation'
        try:
            return self.delegate.open(request, *args, **kwargs)
        except HTTPError as exc:
            try:
                raw = exc.read(65536)
            except Exception:
                raw = b''
            try:
                fields = safe_http_error(json.loads(raw), exc.code)
            except Exception:
                fields = {'http_status': int(exc.code), 'code': None, 'param': None,
                          'numeric_candidates': {}, 'unique_numeric_fields': {},
                          'error_body_unparseable': True}
            self.events.append({'stage': stage, 'category': 'http', **fields})
            # The frozen base _post reads the same 64KiB once and preserves its
            # original sanitized exception and conservative usage accounting.
            raise HTTPError(exc.url, exc.code, exc.msg, exc.headers, io.BytesIO(raw)) from None
        except Exception as exc:
            event = {'stage': stage, 'category': 'non_http', 'exception_class': _safe_exception_class(exc)}
            if type(getattr(exc, 'errno', None)) is int:
                event['errno'] = exc.errno
            reason = getattr(exc, 'reason', None)
            if isinstance(reason, BaseException):
                event['reason_exception_class'] = _safe_exception_class(reason)
                if type(getattr(reason, 'errno', None)) is int:
                    event['reason_errno'] = reason.errno
            tunnel = re.search(r'Tunnel connection failed:\s*(\d{3})\b', str(reason or ''), flags=re.IGNORECASE)
            if tunnel:
                event['tunnel_http_status'] = int(tunnel.group(1))
            self.events.append(event)
            raise


def public_config(value):
    allowed = {'parent_config', 'output', 'preflight_cost_upper_usd', 'count_retry_delays'}
    if not isinstance(value, dict) or set(value)-allowed:
        raise ValueError('Recovery configuration permits only public protocol fields')
    config = json.loads(base.canonical(value))
    config.setdefault('preflight_cost_upper_usd', '0.001364')
    config.setdefault('count_retry_delays', RETRY_DELAYS)
    if config['count_retry_delays'] != RETRY_DELAYS:
        raise ValueError('Only the registered count-only 5/15-second recovery delays are allowed')
    preflight = Decimal(str(config['preflight_cost_upper_usd']))
    if not preflight.is_finite() or not 0 <= preflight <= 1:
        raise ValueError('Invalid separately reserved preflight estimate')
    config['preflight_cost_upper_usd'] = str(preflight)
    for key in ('parent_config', 'output'):
        if not isinstance(config.get(key), str) or not config[key]:
            raise ValueError('Explicit parent configuration and new output paths are required')
        config[key] = str(Path(config[key]).resolve())
    return config


def eligible(record):
    """Metadata-only eligibility; never consult prediction, content or gold."""
    usage, error = record.get('usage') or {}, record.get('error') or {}
    return (record.get('status') == 'error' and record.get('terminal') is True
            and record.get('ledger') == [] and usage.get('physical_requests') == 0
            and usage.get('charged_tokens') == 0
            and error.get('type') == 'OpenAIResponsesTransportError'
            and error.get('message') == NETWORK_MESSAGE)


def _parent_context(config):
    parent_config = base.public_config(json.loads(Path(config['parent_config']).read_text()))
    parent_out = Path(parent_config['output'])
    if Path(config['output']) == parent_out or Path(config['output']).is_relative_to(parent_out):
        raise ValueError('Recovery output must be separate from the original experiment')
    frozen, rows, manifest = base.check_freeze(parent_config)
    if base.digest(parent_out/'freeze.json') != PARENT_FREEZE:
        raise ValueError('Recovery targets a different original freeze')
    summary = json.loads((parent_out/'run_summary.json').read_text())
    if summary.get('phase') != 'inference_complete' or summary.get('terminal_records') != 75:
        raise ValueError('All original 75 records must be terminal first')
    records, hashes = {}, {}
    for model, value, arm in base.expected_jobs(rows, manifest['probe_index']):
        path = base.prediction_path(parent_out, model, value['index'], arm)
        record = json.loads(path.read_text())
        if (record.get('terminal') is not True or record.get('index') != value['index']
                or record.get('model') != model or record.get('arm') != arm
                or record.get('id') != value['_id'] or record.get('freeze_sha256') != PARENT_FREEZE
                or record.get('context_sha256') != frozen['source_hashes'][str(value['index'])]):
            raise ValueError('Original terminal identity differs')
        records[path.name], hashes[path.name] = record, base.digest(path)
    money = json.loads((parent_out/'cost_ledger.json').read_text())
    if money.get('freeze_sha256') != PARENT_FREEZE:
        raise ValueError('Original monetary ledger identity differs')
    parent_cost = sum((Decimal(e['estimated_upper_usd']) for e in money['entries']), Decimal(0))
    if parent_cost != Decimal(money['estimated_upper_usd']):
        raise ValueError('Original monetary ledger does not conserve')
    selected = [record for record in records.values() if eligible(record)]
    if len(selected) != EXPECTED_ELIGIBLE:
        raise ValueError(f'Expected exactly {EXPECTED_ELIGIBLE} zero-generation network failures')
    for record in selected:
        charges = [entry for entry in money['entries'] if entry['model'] == record['model']
                   and entry['id'] == f"{record['arm']}:{record['index']}"]
        if any(Decimal(entry['estimated_upper_usd']) != 0 for entry in charges):
            raise ValueError('An eligible record has an outstanding original generation reservation')
    remaining = (Decimal(str(parent_config['cost_cap_usd']))-parent_cost
                 - Decimal(config['preflight_cost_upper_usd']))
    if remaining <= 0:
        raise ValueError('No generation-cost envelope remains for recovery')
    child_config = {**parent_config, 'output': config['output'], 'cost_cap_usd': str(remaining)}
    return parent_config, frozen, rows, manifest, records, hashes, parent_cost, base.public_config(child_config)


def freeze_recovery(config):
    config = public_config(config)
    parent, parent_freeze, rows, manifest, records, hashes, parent_cost, child = _parent_context(config)
    if any((Path(parent['output'])/name).exists() for name in ('scoring_access.json', 'audit.json')):
        raise ValueError('Recovery registration must precede any original scoring or label-access record')
    eligible_records = [record for record in records.values() if eligible(record)]
    main = [record for record in eligible_records if record['arm'] != 'capacity_probe']
    probes = [record for record in eligible_records if record['arm'] == 'capacity_probe']
    random.Random(20261104).shuffle(main)
    order = main+probes
    base.freeze_limits(child)  # Offline and label-blind; creates the OS guard and code snapshots.
    out = Path(config['output'])
    frozen = json.loads((out/'freeze.json').read_text())
    frozen.update(expected_main_records=len(main), expected_probe_records=len(probes),
                  expected_terminal_records=len(order), parent_terminal_records=75)
    frozen['recovery'] = {
        'protocol': config, 'parent_freeze_sha256': PARENT_FREEZE,
        'parent_config_sha256': base.digest(config['parent_config']),
        'parent_cost_ledger_sha256': base.digest(Path(parent['output'])/'cost_ledger.json'),
        'parent_record_sha256': hashes, 'parent_code_sha256': parent_freeze['code_sha256'],
        'parent_generation_cost_upper_usd': str(parent_cost),
        'preflight_cost_upper_usd': config['preflight_cost_upper_usd'],
        'global_cost_cap_usd': str(parent['cost_cap_usd']),
        'new_generation_cost_cap_usd': child['cost_cap_usd'],
        'expected_recovery_records': len(order), 'expected_parent_records': 75,
        'eligibility': 'terminal generic-network error; ledger empty; physical requests=0; charged tokens=0; no outstanding reservation',
        'eligibility_uses_answers_or_correctness': False,
        'execution_order': [{'model': record['model'], 'index': record['index'], 'arm': record['arm'],
            'parent_record_sha256': hashes[base.prediction_path('.', record['model'], record['index'], record['arm']).name],
            'context_sha256': record['context_sha256']} for record in order],
        'count_retry_delays': RETRY_DELAYS, 'max_count_attempts_per_network_call': 3,
        'implicit_generation_retries': 0, 'parent_generation_budgets_reset': False,
        'wrapper_sha256': base.digest(Path(__file__)),
    }
    target = out/'frozen_source/scripts/benchmark_longbench_openai_recovery.py'
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__), target)
    base.atomic_json(out/'freeze.json', frozen)
    timing = Path(parent['output'])/'long_direct_timing.json'
    prior = json.loads(timing.read_text())['last_start'] if timing.exists() else {}
    base.atomic_json(out/'long_direct_timing.json', {
        'freeze_sha256': base.digest(out/'freeze.json'), 'last_start': prior,
        'seeded_from_original_rate_limit_timing': True})
    return {'phase': 'recovery_frozen', 'expected_recovery_records': len(order),
            'parent_records_preserved': 75, 'labels_read': False,
            'parent_generation_cost_upper_usd': str(parent_cost),
            'new_generation_cost_cap_usd': child['cost_cap_usd'],
            'freeze_sha256': base.digest(out/'freeze.json')}


def check_recovery(config):
    config = public_config(config)
    out = Path(config['output'])
    frozen = json.loads((out/'freeze.json').read_text())
    recovery = frozen['recovery']
    if base.canonical(config) != base.canonical(recovery['protocol']):
        raise ValueError('Recovery protocol changed')
    if (base.digest(Path(__file__)) != recovery['wrapper_sha256']
            or base.digest(out/'frozen_source/scripts/benchmark_longbench_openai_recovery.py') != recovery['wrapper_sha256']):
        raise ValueError('Recovery wrapper changed after freeze')
    parent, old, rows, manifest, records, hashes, parent_cost, child = _parent_context(config)
    if (hashes != recovery['parent_record_sha256']
            or base.digest(config['parent_config']) != recovery['parent_config_sha256']
            or base.digest(Path(parent['output'])/'cost_ledger.json') != recovery['parent_cost_ledger_sha256']
            or old['code_sha256'] != recovery['parent_code_sha256']
            or str(parent_cost) != recovery['parent_generation_cost_upper_usd']):
        raise ValueError('Original experiment changed after recovery freeze')
    checked, _, _ = base.check_freeze(child)
    if checked['code_sha256'] != old['code_sha256']:
        raise ValueError('Recovery altered the original algorithm/controller source set')
    keys = {(record['model'], record['index'], record['arm']) for record in records.values() if eligible(record)}
    order = recovery['execution_order']
    if len(order) != EXPECTED_ELIGIBLE or {(entry['model'], entry['index'], entry['arm']) for entry in order} != keys:
        raise ValueError('Recovery eligibility matrix changed')
    return frozen, child, parent, rows, manifest, records


class CountRetryTransport(LongBenchResponsesTransport):
    """Retry only generic input-count network failures; never generation HTTP."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.count_attempt_audit = []
        self.on_count_attempt_update = None
        self.provider_error_diagnostics = []
        self.opener = ErrorObservingOpener(self.opener, self.provider_error_diagnostics)

    def _notify(self):
        if self.on_count_attempt_update:
            self.on_count_attempt_update(self.count_attempt_audit)

    def _post(self, path, body):
        if path != '/responses/input_tokens':
            return super()._post(path, body)
        request_sha = base.text_digest(base.canonical(body))
        for attempt in range(3):
            entry = {'physical_count_request': len(self.count_attempt_audit)+1,
                     'request_sha256': request_sha, 'attempt': attempt+1,
                     'status': 'inflight', 'model': self.model}
            self.count_attempt_audit.append(entry)
            self._notify()
            try:
                result = super()._post(path, body)
            except Exception as exc:
                retryable = isinstance(exc, OpenAIResponsesTransportError) and str(exc) == NETWORK_MESSAGE
                entry.update(status='error', generic_network_failure=retryable,
                             error_type=type(exc).__name__)
                if retryable and attempt < 2:
                    entry['retry_wait_seconds'] = RETRY_DELAYS[attempt]
                    self._notify()
                    time.sleep(RETRY_DELAYS[attempt])
                    continue
                self._notify()
                raise
            entry.update(status='ok', input_tokens=result.get('input_tokens')
                         if type(result.get('input_tokens')) is int else None)
            self._notify()
            return result
        raise RuntimeError('Unreachable count-attempt state')


def _run_worker(*, api_key, config, transport_factory=CountRetryTransport,
                client_factory=OpenAIResponsesBudgetedClient, harness_runner=run_longbench_harness):
    config = public_config(config)
    frozen, child, parent, rows, manifest, original = check_recovery(config)
    base.verify_guard(child)
    out, freeze_sha = Path(config['output']), base.digest(Path(config['output'])/'freeze.json')
    values = {value['index']: value for value in rows}
    with (out/'.recovery.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (out/'cost_ledger.json').exists() and any(
                json.loads(path.read_text()).get('ledger') for path in (out/'journals').glob('*.json')):
            raise ValueError('Recovery journal has generation usage but its monetary ledger is missing')
        costs = base.MonetaryLedger(out/'cost_ledger.json', cap=child['cost_cap_usd'],
                                   prices=child['prices'], freeze_sha=freeze_sha)
        transports = {model: transport_factory(api_key=api_key, model=model,
            base_url=child['base_url'], reasoning_effort='none', timeout=child['timeout'])
            for model in base.MODELS}
        for job in frozen['recovery']['execution_order']:
            model, index, arm = job['model'], job['index'], job['arm']
            value, transport = values[index], transports[model]
            row = base.make_row(value)
            prompt = base.official_prompt(row)
            target = base.prediction_path(out, model, index, arm)
            if target.exists():
                saved = json.loads(target.read_text())
                if (saved.get('freeze_sha256') != freeze_sha or saved.get('terminal') is not True
                        or saved.get('model') != model or saved.get('index') != index or saved.get('arm') != arm
                        or saved.get('context_sha256') != job['context_sha256']):
                    raise ValueError('An existing recovery terminal identity differs')
                continue
            journal_path = out/'journals'/target.name
            budget_dict = dict(child['budget_harness'] if arm == 'harness' else child['budget_direct'])
            prior = json.loads(journal_path.read_text()) if journal_path.exists() else None
            if prior:
                if prior['freeze_sha256'] != freeze_sha:
                    raise ValueError('Interrupted recovery journal identity differs')
                budget_dict = prior['budget']
            start_count, start_response = len(transport.count_audit), len(transport.response_audit)
            start_attempt = len(getattr(transport, 'count_attempt_audit', []))
            start_diagnostic = len(getattr(transport, 'provider_error_diagnostics', []))
            previous_counts = prior.get('count_audit', []) if prior else []
            previous_responses = prior.get('response_audit', []) if prior else []
            previous_attempts = prior.get('count_attempt_audit', []) if prior else []
            previous_diagnostics = prior.get('provider_error_diagnostics', []) if prior else []
            def journal(ledger):
                base.atomic_json(journal_path, base.scrub_secret({
                    'index': index, 'model': model, 'arm': arm, 'freeze_sha256': freeze_sha,
                    'parent_freeze_sha256': PARENT_FREEZE, 'budget': budget_dict, 'ledger': ledger,
                    'count_audit': previous_counts+transport.count_audit[start_count:],
                    'response_audit': previous_responses+transport.response_audit[start_response:],
                    'count_attempt_audit': previous_attempts+getattr(transport, 'count_attempt_audit', [])[start_attempt:],
                    'provider_error_diagnostics': previous_diagnostics+getattr(transport, 'provider_error_diagnostics', [])[start_diagnostic:],
                }, api_key))
            client = client_factory(transport, ExperimentBudget(**budget_dict), on_ledger_update=journal)
            transport.on_count_attempt_update = lambda _entries: journal(client.ledger)
            capped = base.CostCappedClient(client, costs, f'{arm}:{index}')
            result, error, measurements, content, status = {}, None, {}, '', 'not_started'
            started = time.perf_counter()
            block = out/'model_blocks'/f'{model}.json'
            if prior:
                client.ledger = prior['ledger']
                error = {'type': 'InterruptedTask', 'message': 'Interrupted recovery is terminal; no replay or quota reset'}
                status = 'interrupted'
            elif block.exists():
                error = {'type': 'ModelBlocked', 'message': 'The exact recovery model has an access block'}
                status = 'model_blocked'
            else:
                journal([])
                try:
                    if arm == 'capacity_probe':
                        raw_count = transport.count(messages=[{'role': 'user', 'content': prompt}])
                        budget_dict.update(context_tokens=raw_count+5120, total_tokens=raw_count+5120)
                        client = client_factory(transport, ExperimentBudget(**budget_dict), on_ledger_update=journal)
                        capped = base.CostCappedClient(client, costs, f'{arm}:{index}')
                    source = transport.count(text=row.context)
                    admission = client.count_messages([{'role': 'user', 'content': prompt}])
                    measurements = {'source_envelope_input_tokens': source,
                        'full_prompt_admission_input_tokens': admission,
                        'output_reserved_tokens': 2048 if arm == 'harness' else 1024,
                        'native_context_tokens': base.NATIVE, 'fits_native_direct': admission+1024 <= base.NATIVE,
                        'count_semantics': 'Exact Responses envelope including budget notice and 16 safety tokens'}
                    if arm == 'direct' and admission+1024 > base.NATIVE:
                        status = 'native_admission_denied'
                    elif arm == 'harness':
                        result = harness_runner(row, capped)
                        if result.get('mode') != 'raw_source':
                            raise RuntimeError('Recovery Harness did not use its registered raw-source route')
                        content = str(result.get('prediction') or '')
                        status = 'returned' if content in ('A', 'B', 'C', 'D') else 'no_parseable_answer'
                    else:
                        if arm == 'capacity_probe' and admission+1024 <= base.NATIVE:
                            raise ValueError('The registered recovery probe is not above native admission')
                        if admission > 272000:
                            base._wait_long_direct(out, model, child, freeze_sha)
                        content = capped.complete([{'role': 'user', 'content': prompt}], max_tokens=1024)
                        status = 'probe_unexpectedly_returned' if arm == 'capacity_probe' else 'returned'
                    for entry in client.ledger:
                        metadata = entry.get('provider_metadata') or {}
                        if entry.get('usage_known') and (metadata.get('actual_model') != model
                                or metadata.get('returned_service_tier') != 'default'):
                            raise RuntimeError('Known recovery response identity or tier differs')
                        details = (metadata.get('provider_usage') or {}).get('input_tokens_details') or {}
                        if (details.get('cache_write_tokens_details') or {}).get('ttl_12h_tokens', 0):
                            raise RuntimeError('Unrequested recovery cache TTL changes the cost envelope')
                except Exception as exc:
                    content, result = '', {}
                    error = base.safe_error(exc, api_key)
                    status = ('provider_context_rejected' if arm == 'capacity_probe'
                              and 'status 400; code context_length_exceeded' in error['message'] else 'error')
            messages = [entry.get('error', '') for entry in client.ledger]
            if error:
                messages.append(error['message'])
            if any(any(f'status {code};' in message for code in (401, 403, 404)) for message in messages):
                base.atomic_json(block, {'model': model, 'freeze_sha256': freeze_sha,
                    'reason': 'explicit_authentication_or_model_access_failure', 'time': time.time()})
            prediction = content if arm == 'harness' and content in ('A', 'B', 'C', 'D') else base.extract_choice(content)
            record = {'index': index, 'id': row.item_id, 'model': model, 'arm': arm,
                'terminal': True, 'status': status, 'freeze_sha256': freeze_sha, 'parent_freeze_sha256': PARENT_FREEZE,
                'parent_record_sha256': job['parent_record_sha256'], 'context_sha256': job['context_sha256'],
                'context_chars': len(row.context), 'domain': row.domain, 'budget': budget_dict,
                'prediction': prediction, 'content': content, 'result': result, 'error': error,
                'measurements': measurements, 'usage': client.summary(), 'ledger': client.ledger,
                'ledger_error_count': sum(entry.get('status') == 'error' for entry in client.ledger),
                'count_audit': previous_counts+transport.count_audit[start_count:],
                'response_audit': previous_responses+transport.response_audit[start_response:],
                'count_attempt_audit': previous_attempts+getattr(transport, 'count_attempt_audit', [])[start_attempt:],
                'provider_error_diagnostics': previous_diagnostics+getattr(transport, 'provider_error_diagnostics', [])[start_diagnostic:],
                'logical_parent_physical_requests': 0, 'logical_parent_charged_tokens': 0,
                'seconds': round(time.perf_counter()-started, 4)}
            journal(client.ledger)
            base.atomic_json(target, base.scrub_secret(record, api_key))
            print(json.dumps({'phase': 'recovery', 'index': index, 'model': model, 'arm': arm,
                'status': status, 'calls': len(client.ledger), 'tokens': record['usage']['charged_tokens'],
                'count_attempts': len(record['count_attempt_audit']),
                'error': error['type'] if error else None}), flush=True)
        combined = (Decimal(frozen['recovery']['parent_generation_cost_upper_usd'])
                    + costs.total()+Decimal(config['preflight_cost_upper_usd']))
        if combined > Decimal(frozen['recovery']['global_cost_cap_usd']):
            raise ValueError('Recovery exceeded the combined generation estimate cap')
        summary = {'phase': 'recovery_inference_complete', 'planned_recovery_records': EXPECTED_ELIGIBLE,
            'terminal_recovery_records': sum(base.prediction_path(out, j['model'], j['index'], j['arm']).is_file()
                for j in frozen['recovery']['execution_order']),
            'parent_terminal_records': 75, 'labels_read': False,
            'new_generation_cost_upper_usd': str(costs.total()), 'combined_cost_with_preflight_upper_usd': str(combined)}
        base.atomic_json(out/'run_summary.json', summary)
        return summary


def run_recovery(*, api_key, config):
    config = public_config(config)
    frozen, child, *_ = check_recovery(config)
    if not isinstance(api_key, str) or not api_key:
        raise ValueError('A caller-held credential is required')
    if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').is_file():
        raise RuntimeError('Recovery requires the local macOS label-denial sandbox')
    environment = {key: value for key, value in os.environ.items()
                   if not any(marker in key.upper() for marker in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL'))}
    command = ['/usr/bin/sandbox-exec', '-f', str(Path(config['output'])/'labels_guard.sb'),
               sys.executable, '-m', 'scripts.benchmark_longbench_openai_recovery', '--guarded-worker']
    result = subprocess.run(command, input=json.dumps({'api_key': api_key, 'config': config}),
                            text=True, cwd=ROOT, env=environment, check=False)
    if result.returncode:
        raise RuntimeError('Recovery worker stopped; preserve and inspect its journals before resume')
    return json.loads((Path(config['output'])/'run_summary.json').read_text())


def _validate_recovery_records(frozen, child, originals):
    """All budget/cost/lineage gates precede any label read."""
    from scripts.audit_context_holdout_execution_v3 import audit_ledger
    out, freeze_sha = Path(child['output']), base.digest(Path(child['output'])/'freeze.json')
    records = {}
    costs = base.MonetaryLedger(out/'cost_ledger.json', cap=child['cost_cap_usd'],
                               prices=child['prices'], freeze_sha=freeze_sha)
    for job in frozen['recovery']['execution_order']:
        path = base.prediction_path(out, job['model'], job['index'], job['arm'])
        if not path.exists():
            raise ValueError('All 75 original and 39 recovery terminal records are required before labels')
        record = json.loads(path.read_text())
        parent = originals[path.name]
        if (record.get('terminal') is not True or record.get('freeze_sha256') != freeze_sha
                or record.get('parent_freeze_sha256') != PARENT_FREEZE
                or any(record.get(key) != job[key] for key in ('model', 'index', 'arm', 'context_sha256'))
                or record.get('id') != parent['id'] or record.get('parent_record_sha256') != job['parent_record_sha256']
                or not eligible(parent)):
            raise ValueError('Recovery terminal identity or zero-generation lineage differs')
        budget = record['budget']
        if record['arm'] != 'capacity_probe':
            expected = child['budget_harness'] if record['arm'] == 'harness' else child['budget_direct']
            if budget != expected or parent['budget'] != expected:
                raise ValueError('Recovery changed the original main-arm logical budget')
        elif (budget['max_calls'] != 1 or budget['max_output_tokens'] != 1024
              or budget['context_tokens'] != budget['total_tokens']):
            raise ValueError('Recovery changed the separately registered probe budget')
        token_audit = audit_ledger(record['ledger'], budget)
        if token_audit['violations']:
            raise ValueError('Recovery token ledger failed before labels')
        for stored, audited in (('physical_requests', 'physical_requests'), ('prompt_tokens', 'prompt_tokens'),
                                ('completion_tokens', 'completion_tokens'), ('charged_tokens', 'charged_tokens')):
            if record['usage'].get(stored) != token_audit[audited]:
                raise ValueError('Recovery usage summary differs from its physical ledger')
        if record['usage'].get('within_budget') is not True:
            raise ValueError('Recovery usage summary does not certify its budget')
        if (parent['usage']['physical_requests']+len(record['ledger']) > budget['max_calls']
                or parent['usage']['charged_tokens']+record['usage']['charged_tokens'] > budget['total_tokens']):
            raise ValueError('Shared parent plus recovery logical budget exceeded')
        entries = [entry for entry in costs.entries if entry['model'] == job['model']
                   and entry['id'] == f"{job['arm']}:{job['index']}" and entry.get('status') != 'not_sent']
        if len(entries) != len(record['ledger']):
            raise ValueError('Recovery generation/monetary reservations differ')
        for money, entry in zip(entries, record['ledger']):
            if (money['reserved_prompt_tokens'] != entry['admission_prompt_tokens']
                    or money['reserved_output_tokens'] != entry['requested_output_tokens']):
                raise ValueError('Recovery monetary pre-admission differs')
            rin, rout = costs.rates(job['model'], entry['admission_prompt_tokens'])
            inp, outp = ((entry['usage']['prompt_tokens'], entry['usage']['completion_tokens'])
                         if entry.get('usage_known') else
                         (entry['admission_prompt_tokens'], entry['requested_output_tokens']))
            if Decimal(money['estimated_upper_usd']) != (inp*rin+outp*rout)/Decimal(1_000_000):
                raise ValueError('Recovery monetary settlement differs')
            metadata = entry.get('provider_metadata') or {}
            if entry.get('usage_known') and (metadata.get('actual_model') != job['model']
                                           or metadata.get('returned_service_tier') != 'default'):
                raise ValueError('Known recovery model identity or tier differs')
            details = (metadata.get('provider_usage') or {}).get('input_tokens_details') or {}
            if (details.get('cache_write_tokens_details') or {}).get('ttl_12h_tokens', 0):
                raise ValueError('Recovery response reports an unregistered cache-price category')
        records[path.name] = record
    sent = [entry for entry in costs.entries if entry.get('status') != 'not_sent']
    if len(sent) != sum(len(record['ledger']) for record in records.values()) or costs.total() > Decimal(child['cost_cap_usd']):
        raise ValueError('Recovery global token/cost correspondence failed')
    return records, costs


def _summarize(main, rows):
    summary = {}
    for model in base.MODELS:
        output = {}
        for arm in ('direct', 'harness'):
            group = [record for record in main if record['model'] == model and record['arm'] == arm]
            answers = [record for record in group if record['semantic_answer_available']]
            item = {'n_planned': 12, 'n_answer_available': len(answers),
                'correct_answers': sum(record['task_success'] for record in group),
                'task_success_rate_all_planned': sum(record['task_success'] for record in group)/12,
                'accuracy_among_available_answers': statistics.mean(record['correct'] for record in answers) if answers else None,
                'available_answer_accuracy_is_secondary_conditional': True,
                'native_admission_denied': sum(record['status'] == 'native_admission_denied' for record in group),
                'status_counts': dict(Counter(record['status'] for record in group)),
                'physical_generations': sum(record['usage']['physical_requests'] for record in group),
                'charged_or_reserved_tokens': sum(record['usage']['charged_tokens'] for record in group)}
            if arm == 'direct':
                admissible = [record for record in group if record['measurements'].get('fits_native_direct') is True]
                item.update(n_native_admissible=len(admissible),
                    native_eligibility_unknown=sum('fits_native_direct' not in record['measurements'] for record in group),
                    correct_native_admissible=sum(record['task_success'] for record in admissible),
                    primary_task_success_rate_native_admissible=(
                        sum(record['task_success'] for record in admissible)/len(admissible) if admissible else None))
            else:
                item['primary_task_success_rate_all_planned'] = item['task_success_rate_all_planned']
            output[arm] = item
        pairs, available = [], []
        for value in rows:
            arms = {record['arm']: record for record in main if record['model'] == model and record['index'] == value['index']}
            if arms['direct']['measurements'].get('fits_native_direct') is True:
                pairs.append((arms['direct']['task_success'], arms['harness']['task_success']))
            if all(arms[arm]['semantic_answer_available'] for arm in ('direct', 'harness')):
                available.append((arms['direct']['correct'], arms['harness']['correct']))
        def pair_counts(values):
            return {'n': len(values), 'both_correct': sum(a and b for a, b in values),
                'direct_only_correct': sum(a and not b for a, b in values),
                'harness_only_correct': sum(b and not a for a, b in values),
                'both_wrong': sum(not a and not b for a, b in values)}
        output['paired_native_admissible'] = {**pair_counts(pairs), 'failed_or_missing_answers_retained': True}
        output['paired_available'] = {**pair_counts(available), 'secondary_conditioned_on_both_answers_available': True}
        summary[model] = output
    return summary


def score_recovery(config):
    config = public_config(config)
    frozen, child, parent, rows, manifest, originals = check_recovery(config)
    recovered, costs = _validate_recovery_records(frozen, child, originals)
    out, freeze_sha = Path(config['output']), base.digest(Path(config['output'])/'freeze.json')
    combined = (Decimal(frozen['recovery']['parent_generation_cost_upper_usd'])
                + costs.total()+Decimal(config['preflight_cost_upper_usd']))
    if combined > Decimal(frozen['recovery']['global_cost_cap_usd']):
        raise ValueError('Combined parent/recovery/preflight estimate exceeds the original global cap')
    report_path = out/'audit.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report.get('freeze_sha256') != freeze_sha:
            raise ValueError('Existing recovery score identity differs')
        return report['summary']
    access = out/'scoring_access.json'
    if not access.exists():
        base.atomic_json(access, {'first_scored_at_unix': time.time(),
            'all_75_parent_and_39_recovery_terminal_before_labels': True,
            'labels_sha256': frozen['labels_sha256']})
    parent_audit = Path(parent['output'])/'audit.json'
    if not parent_audit.exists():
        if (Path(parent['output'])/'scoring_access.json').exists():
            raise ValueError('Original scoring was interrupted; preserve its first-access record for review')
        base.score_limits(parent)  # First-ever raw-trial scoring, after the combined 114-record gate.
    original_report = json.loads(parent_audit.read_text())
    if original_report.get('freeze_sha256') != PARENT_FREEZE:
        raise ValueError('Original score audit identity differs')
    if base.digest(child['labels']) != frozen['labels_sha256']:
        raise ValueError('Recovery labels differ from their original declaration')
    labels = json.loads(Path(child['labels']).read_text())
    if set(labels) != {str(row['index']) for row in rows}:
        raise ValueError('Recovery label identities differ')
    main, probes, recovery_only = [], [], []
    for name, original in originals.items():
        record = recovered.get(name, original)
        logical = {**record, 'logical_main_freeze_sha256': PARENT_FREEZE,
                   'recovery_freeze_sha256': freeze_sha if name in recovered else None,
                   'outcome_origin': 'zero_generation_network_recovery' if name in recovered else 'original_trial',
                   'original_record_sha256': frozen['recovery']['parent_record_sha256'][name],
                   'original_status': original['status'],
                   'original_physical_generations': original['usage']['physical_requests'],
                   'logical_parent_plus_recovery_generations': original['usage']['physical_requests']
                        +(record['usage']['physical_requests'] if name in recovered else 0)}
        if logical['arm'] == 'capacity_probe':
            probes.append(logical)
        else:
            denied = logical['status'] == 'native_admission_denied'
            present = logical['prediction'] in ('A', 'B', 'C', 'D') and not denied
            correct = present and logical['prediction'] == labels[str(logical['index'])]
            logical.update(gold=labels[str(logical['index'])], correct=None if denied else bool(correct),
                           semantic_answer_available=present, task_success=bool(correct))
            main.append(logical)
        if name in recovered:
            recovery_only.append(logical)
    report = {'experiment': frozen['experiment']+' — zero-generation infrastructure recovery overlay',
        'freeze_sha256': freeze_sha, 'parent_freeze_sha256': PARENT_FREEZE,
        'sample_status': frozen['sample_status'], 'summary': _summarize(main, rows),
        'samples': main, 'capacity_probes': probes,
        'first_trial_summary': original_report['summary'],
        'first_trial_status_counts': dict(Counter(record['status'] for record in originals.values())),
        'first_trial_audit_sha256': base.digest(parent_audit),
        'recovery_status_counts': dict(Counter(record['status'] for record in recovered.values())),
        'recovery_samples': recovery_only, 'original_records_preserved': 75, 'new_recovery_records': len(recovered),
        'parent_generation_cost_upper_usd': frozen['recovery']['parent_generation_cost_upper_usd'],
        'recovery_generation_cost_upper_usd': str(costs.total()),
        'preflight_cost_upper_usd': config['preflight_cost_upper_usd'],
        'estimated_generation_cost_upper_usd': str(Decimal(frozen['recovery']['parent_generation_cost_upper_usd'])+costs.total()),
        'combined_cost_with_preflight_upper_usd': str(combined),
        'not_same_compute_comparison': True, 'official_leaderboard_submission': False,
        'recovery_eligibility_used_answers_or_correctness': False}
    base.atomic_json(out/'recovery_audit.json', {'freeze_sha256': freeze_sha,
                      'samples': recovery_only, 'summary': report['recovery_status_counts']})
    base.atomic_json(report_path, report)
    return report['summary']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('freeze', 'run', 'score'), nargs='?')
    parser.add_argument('--config')
    parser.add_argument('--execute-frozen-run', action='store_true')
    parser.add_argument('--guarded-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.guarded_worker:
        payload = json.loads(sys.stdin.read())
        try:
            _run_worker(api_key=payload['api_key'], config=payload['config'])
        except Exception as exc:
            print(json.dumps({'recovery_worker_error_type': type(exc).__name__}), flush=True)
            return 1
        return 0
    if not args.phase or not args.config:
        parser.error('Specify phase and public recovery configuration')
    config = json.loads(Path(args.config).read_text())
    if args.phase == 'freeze':
        result = freeze_recovery(config)
    elif args.phase == 'score':
        result = score_recovery(config)
    else:
        if not args.execute_frozen_run:
            parser.error('Execution requires --execute-frozen-run')
        result = run_recovery(api_key=getpass.getpass('OpenAI API key (RAM only): '), config=config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
