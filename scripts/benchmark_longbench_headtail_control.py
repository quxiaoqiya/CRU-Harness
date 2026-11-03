"""Conditional, pre-label head/tail quality control under the same USD50 cap.

This is a character-span head/tail approximation with exact whole-prompt
admission, not a bit-identical reproduction of upstream token-half truncation.
Selection depends only on remaining money and the two longest fixed sources.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import fcntl
import getpass
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from long_context_agent.experiment_budget import ExperimentBudget
from long_context_agent.openai_responses_transport import OpenAIResponsesBudgetedClient
from scripts import benchmark_longbench_openai_recovery as recovery


base, ROOT = recovery.base, recovery.ROOT
MODELS = base.MODELS
BUDGET = {'context_tokens': 800_000, 'total_tokens': 800_000,
          'max_calls': 1, 'max_output_tokens': 1024, 'token_safety_margin': 16}
UPSTREAM_COMMIT = 'c5ea10bcd06285223c58dfed76bbc92d22273709'
UPSTREAM_URL = f'https://raw.githubusercontent.com/THUDM/LongBench/{UPSTREAM_COMMIT}/pred.py'


def public_config(value):
    if not isinstance(value, dict) or set(value) != {'recovery_config', 'output'}:
        raise ValueError('Provide only the recovery configuration path and a separate output path')
    if any(not isinstance(path, str) or not path for path in value.values()):
        raise ValueError('Both public path fields must be nonempty strings')
    return {key: str(Path(path).resolve()) for key, path in value.items()}


def money_plan(remaining):
    remaining = Decimal(str(remaining))
    if not remaining.is_finite() or remaining < 0:
        raise ValueError('Invalid combined remaining estimate')
    if remaining >= 13:
        return {'planned_indices': [143, 260], 'control_cap_usd': '13', 'expected_control_records': 6}
    if remaining >= Decimal('6.5'):
        return {'planned_indices': [143], 'control_cap_usd': '6.5', 'expected_control_records': 3}
    return {'planned_indices': [], 'control_cap_usd': '0', 'expected_control_records': 0,
            'budget_skip_reason': 'Less than USD6.50 remains under the combined USD50 estimate cap'}


def _context(config, *, require_unscored=False):
    config = public_config(config)
    if set(config) != {'recovery_config', 'output'}:
        raise ValueError('Both public path fields must be nonempty strings')
    rec_config = recovery.public_config(json.loads(Path(config['recovery_config']).read_text()))
    rec_out = Path(rec_config['output'])
    run = json.loads((rec_out/'run_summary.json').read_text())
    if (run.get('phase') != 'recovery_inference_complete' or run.get('terminal_recovery_records') != 39
            or len(list((rec_out/'predictions').glob('*.json'))) != 39):
        raise ValueError('All original75 and recovery39 terminals are required before control preparation')
    frozen, child, parent, rows, manifest, originals = recovery.check_recovery(rec_config)
    _, costs = recovery._validate_recovery_records(frozen, child, originals)
    if Path(config['output']) in (Path(parent['output']), rec_out):
        raise ValueError('Control output must be separate from both preceding trials')
    if require_unscored:
        forbidden = [Path(parent['output'])/name for name in ('audit.json', 'scoring_access.json')]
        forbidden += [rec_out/name for name in ('audit.json', 'recovery_audit.json', 'scoring_access.json')]
        if any(path.exists() for path in forbidden):
            raise ValueError('Quality-control registration must precede any scoring or label access')
    parent_cost = Decimal(frozen['recovery']['parent_generation_cost_upper_usd'])
    preflight = Decimal(rec_config['preflight_cost_upper_usd'])
    global_cap = Decimal(frozen['recovery']['global_cost_cap_usd'])
    remaining = global_cap-parent_cost-costs.total()-preflight
    return {'config': config, 'recovery_config': rec_config, 'recovery_out': rec_out,
            'recovery_frozen': frozen, 'child_config': child, 'parent_config': parent,
            'rows': rows, 'manifest': manifest,
            'recovery_freeze_sha256': base.digest(rec_out/'freeze.json'),
            'parent_cost_upper_usd': str(parent_cost), 'recovery_cost_upper_usd': str(costs.total()),
            'preflight_cost_upper_usd': str(preflight), 'global_cap_usd': str(global_cap),
            'remaining_before_control_usd': str(remaining),
            'recovery_record_sha256': {path.name: base.digest(path) for path in (rec_out/'predictions').glob('*.json')},
            'parent_cost_ledger_sha256': base.digest(Path(parent['output'])/'cost_ledger.json'),
            'recovery_cost_ledger_sha256': base.digest(rec_out/'cost_ledger.json')}


def character_headtail(context, retained_chars):
    retained = max(2, min(len(context), int(retained_chars)))
    if retained >= len(context):
        return context, [[0, len(context)]], None
    head, tail = (retained+1)//2, retained//2
    spans = [[0, head], [len(context)-tail, len(context)]]
    return context[:head]+context[len(context)-tail:], spans, [head, len(context)-tail]


def strict_official_choice(content):
    """Official-style capitalized answer patterns; actual text, no inferred vote."""
    clean = str(content or '').replace('*', '').strip()
    for pattern in (r'The correct answer is \(([A-D])\)', r'The correct answer is ([A-D])'):
        match = re.search(pattern, clean)
        if match:
            return match.group(1)
    return None


def prepare_source(value, preflight_admission, clients, *, on_round=None):
    context = value['context']
    retained = min(len(context), max(2, int(len(context)*790_000/int(preflight_admission))))
    history = []
    for round_index in range(8):
        trimmed, spans, dropped = character_headtail(context, retained)
        candidate = {**value, 'context': trimmed}
        prompt = base.official_prompt(base.make_row(candidate))
        counts = {model: clients[model].count_messages([{'role': 'user', 'content': prompt}]) for model in MODELS}
        step = {'round': round_index+1, 'retained_chars': len(trimmed),
                'direct_admission_tokens_by_model': counts,
                'all_models_fit': all(count+1024 <= BUDGET['context_tokens'] for count in counts.values())}
        history.append(step)
        if on_round:
            on_round(history)
        if step['all_models_fit']:
            metadata = {'index': value['index'], 'id': value['_id'],
                'original_context_sha256': base.text_digest(context),
                'trimmed_context_sha256': base.text_digest(trimmed),
                'original_context_chars': len(context), 'retained_chars': len(trimmed),
                'source_spans': spans, 'dropped_middle_span': dropped,
                'direct_admission_tokens_by_model': counts,
                'trimmed_prompt_sha256': base.text_digest(prompt),
                'preparation_rounds': history, 'same_trimmed_source_for_all_models': True,
                'inserted_characters': 0, 'full_source_retained': dropped is None}
            return candidate, metadata
        retained = max(2, min(retained-1, int(retained*(790_000/max(counts.values()))*.99)))
    raise ValueError('Shared head/tail did not fit all models within eight exact-count rounds')


def code_fingerprints():
    return {**base.source_fingerprints(),
            'scripts/benchmark_longbench_openai_recovery.py': base.digest(Path(recovery.__file__)),
            'scripts/benchmark_longbench_headtail_control.py': base.digest(Path(__file__))}


def prepare_control(*, api_key, config, transport_factory=recovery.CountRetryTransport,
                    client_factory=OpenAIResponsesBudgetedClient):
    config = public_config(config)
    out = Path(config['output'])
    if (out/'freeze.json').exists():
        frozen, _ = check_control(config)
        return {'phase': 'control_already_frozen', 'expected_control_records': frozen['expected_control_records'],
                'freeze_sha256': base.digest(out/'freeze.json')}
    context = _context(config, require_unscored=True)
    plan = money_plan(context['remaining_before_control_usd'])
    out.mkdir(parents=True, exist_ok=True)
    with (out/'.prepare.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (out/'preparation_registration.json').exists():
            raise ValueError('An interrupted preparation is registered; inspect it instead of silently retrying')
        registration = {'config': config, **plan, 'selection_uses_answers_or_model_outputs': False,
            'selection_rule': 'Money thresholds13/6.5 only; fixed longest natural source indices143 then260',
            'parent_freeze_sha256': recovery.PARENT_FREEZE,
            'recovery_freeze_sha256': context['recovery_freeze_sha256'],
            'parent_cost_upper_usd': context['parent_cost_upper_usd'],
            'recovery_cost_upper_usd': context['recovery_cost_upper_usd'],
            'preflight_cost_upper_usd': context['preflight_cost_upper_usd'],
            'global_cap_usd': context['global_cap_usd'],
            'remaining_before_control_usd': context['remaining_before_control_usd'],
            'registered_at_unix': time.time()}
        base.atomic_json(out/'preparation_registration.json', registration)
        selected, metadata, transports = [], [], {}
        if plan['expected_control_records']:
            if not isinstance(api_key, str) or not api_key:
                raise ValueError('Count-only preparation requires a caller-held RAM credential')
            transports = {model: transport_factory(api_key=api_key, model=model,
                base_url=context['child_config']['base_url'], reasoning_effort='none',
                timeout=context['child_config']['timeout']) for model in MODELS}
            clients = {model: client_factory(transport, ExperimentBudget(**BUDGET))
                       for model, transport in transports.items()}
            by_index = {value['index']: value for value in context['rows']}
            preflight = {value['index']: value for value in context['manifest']['selected']}
            for index in plan['planned_indices']:
                value = by_index[index]
                estimate = preflight[index]['preflight_native_direct']['direct_admission_input_tokens']
                def progress(history):
                    base.atomic_json(out/f'preparation_{index}.json', {
                        'index': index, 'rounds': history, 'generation_requested': False})
                candidate, item = prepare_source(value, estimate, clients, on_round=progress)
                selected.append(candidate)
                metadata.append(item)
        (out/'inputs.jsonl').write_text(''.join(json.dumps(value, ensure_ascii=False)+'\n' for value in selected))
        manifest = {'schema_version': 1, 'control_kind': 'balanced_character_headtail_with_exact_whole_prompt_admission',
            'upstream_pred_url': UPSTREAM_URL, 'upstream_pred_commit': UPSTREAM_COMMIT,
            'not_bit_identical_to_upstream_token_half': True, 'selected': metadata,
            'selection_uses_answers_or_model_outputs': False, 'generation_requested_during_preparation': False}
        base.atomic_json(out/'manifest.json', manifest)
        base.atomic_json(out/'preparation_count_audit.json', {model: {
            'count_audit': transport.count_audit,
            'physical_count_attempts': getattr(transport, 'count_attempt_audit', []),
            'safe_provider_error_diagnostics': getattr(transport, 'provider_error_diagnostics', [])}
            for model, transport in transports.items()})
        probe = out/'label_guard_probe.txt'
        probe.write_text('No-label control isolation probe.\n')
        parent = context['parent_config']
        profile = '(version 1)\n(allow default)\n'
        profile += ''.join('(deny file-read* (subpath '+json.dumps(str(path))+'))\n'
            for path in (Path(parent['labels']).parent, ROOT/'docs', ROOT/'benchmarks'))
        profile += ''.join('(deny file-read* (literal '+json.dumps(str(path))+'))\n'
            for path in (Path(parent['raw_cache']), probe))
        (out/'labels_guard.sb').write_text(profile)
        frozen = {**registration, 'schema_version': 1, 'experiment': 'LongBench conditional head/tail quality control',
            'budget': BUDGET, 'models': list(MODELS), 'frozen_at_unix': time.time(),
            'prepared_inputs_sha256': base.digest(out/'inputs.jsonl'),
            'manifest_sha256': base.digest(out/'manifest.json'),
            'guard_profile_sha256': base.digest(out/'labels_guard.sb'),
            'preparation_count_audit_sha256': base.digest(out/'preparation_count_audit.json'),
            'code_sha256': code_fingerprints(), 'runtime': base.runtime_identity(),
            'recovery_record_sha256': context['recovery_record_sha256'],
            'parent_cost_ledger_sha256': context['parent_cost_ledger_sha256'],
            'recovery_cost_ledger_sha256': context['recovery_cost_ledger_sha256'],
            'upstream_pred_url': UPSTREAM_URL, 'upstream_pred_commit': UPSTREAM_COMMIT,
            'control_definition': 'Balanced character source prefix+suffix with no inserted characters; exact whole-prompt admission for all models; not upstream token-half bitwise reproduction',
            'labels_read': False, 'all_114_terminal_before_control_registration': True,
            'comparison_has_unequal_compute': True}
        for name in frozen['code_sha256']:
            target = out/'frozen_source'/name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT/name, target)
        base.atomic_json(out/'freeze.json', frozen)
        if not plan['expected_control_records']:
            base.atomic_json(out/'budget_skip.json', {**plan, 'remaining_before_control_usd': context['remaining_before_control_usd'],
                'parent_freeze_sha256': recovery.PARENT_FREEZE,
                'recovery_freeze_sha256': context['recovery_freeze_sha256'], 'generation_requested': False})
        costs = base.MonetaryLedger(out/'cost_ledger.json', cap=plan['control_cap_usd'],
            prices=context['child_config']['prices'], freeze_sha=base.digest(out/'freeze.json'))
        costs.save()
        prior_timing = context['recovery_out']/'long_direct_timing.json'
        timing = json.loads(prior_timing.read_text()).get('last_start', {}) if prior_timing.exists() else {}
        base.atomic_json(out/'long_direct_timing.json', {
            'freeze_sha256': base.digest(out/'freeze.json'), 'last_start': timing})
    return {'phase': 'headtail_control_frozen', 'expected_control_records': plan['expected_control_records'],
            'planned_indices': plan['planned_indices'], 'control_cap_usd': plan['control_cap_usd'],
            'freeze_sha256': base.digest(out/'freeze.json'), 'labels_read': False}


def check_control(config):
    config = public_config(config)
    out = Path(config['output'])
    frozen = json.loads((out/'freeze.json').read_text())
    context = _context(config)
    if (base.canonical(config) != base.canonical(frozen['config'])
            or code_fingerprints() != frozen['code_sha256']
            or base.runtime_identity() != frozen['runtime']
            or context['recovery_freeze_sha256'] != frozen['recovery_freeze_sha256']
            or context['recovery_record_sha256'] != frozen['recovery_record_sha256']
            or context['parent_cost_ledger_sha256'] != frozen['parent_cost_ledger_sha256']
            or context['recovery_cost_ledger_sha256'] != frozen['recovery_cost_ledger_sha256']):
        raise ValueError('Control code, configuration or parent/recovery snapshot changed')
    for name, field in (('inputs.jsonl', 'prepared_inputs_sha256'), ('manifest.json', 'manifest_sha256'),
                        ('labels_guard.sb', 'guard_profile_sha256'), ('preparation_count_audit.json', 'preparation_count_audit_sha256')):
        if base.digest(out/name) != frozen[field]:
            raise ValueError('Prepared control input or metadata changed')
    if money_plan(frozen['remaining_before_control_usd'])['planned_indices'] != frozen['planned_indices']:
        raise ValueError('Control selection differs from the pre-label money-only rule')
    plan = money_plan(frozen['remaining_before_control_usd'])
    if any(frozen[key] != plan[key] for key in ('control_cap_usd', 'expected_control_records')):
        raise ValueError('Control cap or planned denominator differs from the money-only rule')
    if any(base.digest(out/'frozen_source'/name) != value for name, value in frozen['code_sha256'].items()):
        raise ValueError('Control source snapshot changed')
    return frozen, context


def prediction_path(out, model, index):
    return Path(out)/'predictions'/f'{index}__{model}__headtail_control.json'


def _run_worker(*, api_key, config, transport_factory=recovery.CountRetryTransport,
                client_factory=OpenAIResponsesBudgetedClient):
    frozen, context = check_control(config)
    out, freeze_sha = Path(config['output']), base.digest(Path(config['output'])/'freeze.json')
    with (out/'.run.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        expected = [(model, index) for index in frozen['planned_indices'] for model in MODELS]
        scoring_paths = [Path(context['parent_config']['output'])/'scoring_access.json',
                         context['recovery_out']/'scoring_access.json', out/'scoring_access.json']
        if any(path.exists() for path in scoring_paths) and any(
                not prediction_path(out, model, index).exists() for model, index in expected):
            raise ValueError('Missing control outcomes cannot be regenerated after labels have opened')
        if expected:
            base.verify_guard({'output': str(out)})
        if not (out/'cost_ledger.json').exists() and any(
                json.loads(path.read_text()).get('ledger') for path in (out/'journals').glob('*.json')):
            raise ValueError('Existing control generations lack their monetary ledger')
        costs = base.MonetaryLedger(out/'cost_ledger.json', cap=frozen['control_cap_usd'],
            prices=context['child_config']['prices'], freeze_sha=freeze_sha)
        values = {value['index']: value for value in [
            json.loads(line) for line in (out/'inputs.jsonl').read_text().splitlines() if line.strip()]}
        metadata = {item['index']: item for item in json.loads((out/'manifest.json').read_text())['selected']}
        transports = {model: transport_factory(api_key=api_key, model=model,
            base_url=context['child_config']['base_url'], reasoning_effort='none',
            timeout=context['child_config']['timeout']) for model in MODELS} if expected else {}
        for model, index in expected:
            target = prediction_path(out, model, index)
            if target.exists():
                saved = json.loads(target.read_text())
                if saved.get('freeze_sha256') != freeze_sha or saved.get('model') != model or saved.get('index') != index:
                    raise ValueError('Existing control terminal identity differs')
                continue
            transport = transports[model]
            path = out/'journals'/target.name
            previous = json.loads(path.read_text()) if path.exists() else None
            counts_start, responses_start = len(transport.count_audit), len(transport.response_audit)
            attempts_start = len(getattr(transport, 'count_attempt_audit', []))
            diagnostics_start = len(getattr(transport, 'provider_error_diagnostics', []))
            def journal(ledger):
                base.atomic_json(path, base.scrub_secret({'model': model, 'index': index, 'freeze_sha256': freeze_sha,
                    'ledger': ledger, 'count_audit': (previous or {}).get('count_audit', [])+transport.count_audit[counts_start:],
                    'response_audit': (previous or {}).get('response_audit', [])+transport.response_audit[responses_start:],
                    'count_attempt_audit': (previous or {}).get('count_attempt_audit', [])+getattr(transport, 'count_attempt_audit', [])[attempts_start:],
                    'provider_error_diagnostics': (previous or {}).get('provider_error_diagnostics', [])+getattr(transport, 'provider_error_diagnostics', [])[diagnostics_start:]}, api_key))
            client = client_factory(transport, ExperimentBudget(**BUDGET), on_ledger_update=journal)
            transport.on_count_attempt_update = lambda _entries: journal(client.ledger)
            content, error, measurements, status = '', None, {}, 'not_started'
            started = time.perf_counter()
            if previous:
                if previous.get('freeze_sha256') != freeze_sha:
                    raise ValueError('Interrupted control journal identity differs')
                client.ledger = previous['ledger']
                status, error = 'interrupted', {'type': 'InterruptedTask', 'message': 'No control generation replay'}
            else:
                journal([])
                try:
                    prompt = base.official_prompt(base.make_row(values[index]))
                    admission = client.count_messages([{'role': 'user', 'content': prompt}])
                    measurements = {'full_prompt_admission_input_tokens': admission, 'output_reserved_tokens': 1024,
                        'prepared_admission_input_tokens': metadata[index]['direct_admission_tokens_by_model'][model],
                        'whole_prompt_fits_control_budget': admission+1024 <= BUDGET['context_tokens']}
                    if not measurements['whole_prompt_fits_control_budget']:
                        raise ValueError('Frozen shared head/tail no longer fits exact control admission')
                    if admission > 272000:
                        timing_config = {**context['child_config'], 'output': str(out)}
                        base._wait_long_direct(out, model, timing_config, freeze_sha)
                    content = base.CostCappedClient(client, costs, f'headtail_control:{index}').complete(
                        [{'role': 'user', 'content': prompt}], max_tokens=1024)
                    for entry in client.ledger:
                        meta = entry.get('provider_metadata') or {}
                        if entry.get('usage_known') and (meta.get('actual_model') != model
                                or meta.get('returned_service_tier') != 'default'):
                            raise ValueError('Actual control model identity or tier differs')
                        details = (meta.get('provider_usage') or {}).get('input_tokens_details') or {}
                        if (details.get('cache_write_tokens_details') or {}).get('ttl_12h_tokens', 0):
                            raise ValueError('Control reports unregistered cache-price category')
                    status = 'returned'
                except Exception as exc:
                    content, status, error = '', 'error', base.safe_error(exc, api_key)
            record = {'index': index, 'id': values[index]['id'], 'model': model, 'arm': 'headtail_control',
                'terminal': True, 'status': status, 'freeze_sha256': freeze_sha,
                'parent_freeze_sha256': recovery.PARENT_FREEZE, 'recovery_freeze_sha256': frozen['recovery_freeze_sha256'],
                'context_sha256': metadata[index]['trimmed_context_sha256'],
                'original_context_sha256': metadata[index]['original_context_sha256'],
                'source_spans': metadata[index]['source_spans'], 'budget': BUDGET,
                'prediction': base.extract_choice(content), 'strict_official_prediction': strict_official_choice(content),
                'strict_format_accepted': strict_official_choice(content) is not None,
                'content': content, 'error': error, 'measurements': measurements, 'usage': client.summary(),
                'ledger': client.ledger,
                'count_audit': (previous or {}).get('count_audit', [])+transport.count_audit[counts_start:],
                'response_audit': (previous or {}).get('response_audit', [])+transport.response_audit[responses_start:],
                'count_attempt_audit': (previous or {}).get('count_attempt_audit', [])+getattr(transport, 'count_attempt_audit', [])[attempts_start:],
                'provider_error_diagnostics': (previous or {}).get('provider_error_diagnostics', [])+getattr(transport, 'provider_error_diagnostics', [])[diagnostics_start:],
                'seconds': round(time.perf_counter()-started, 4)}
            journal(client.ledger)
            base.atomic_json(target, base.scrub_secret(record, api_key))
            print(json.dumps({'phase': 'headtail_control', 'index': index, 'model': model,
                'status': status, 'calls': len(client.ledger), 'tokens': record['usage']['charged_tokens'],
                'error': error['type'] if error else None}), flush=True)
        combined = (Decimal(context['parent_cost_upper_usd'])+Decimal(context['recovery_cost_upper_usd'])
                    +Decimal(context['preflight_cost_upper_usd'])+costs.total())
        if combined > Decimal(context['global_cap_usd']):
            raise ValueError('Combined original/recovery/control/preflight estimate exceeds global cap')
        summary = {'phase': 'headtail_control_complete', 'planned_control_records': len(expected),
            'terminal_control_records': sum(prediction_path(out, model, index).is_file() for model, index in expected),
            'labels_read': False, 'budget_skipped': not expected,
            'estimated_control_cost_upper_usd': str(costs.total()),
            'combined_cost_with_preflight_upper_usd': str(combined)}
        base.atomic_json(out/'run_summary.json', summary)
        return summary


def run_control(*, api_key, config):
    config = public_config(config)
    frozen, _ = check_control(config)
    if frozen['expected_control_records'] == 0:
        return _run_worker(api_key='', config=config)
    if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').is_file():
        raise RuntimeError('Strict control execution requires local macOS sandbox-exec')
    environment = {key: value for key, value in os.environ.items()
                   if not any(marker in key.upper() for marker in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL'))}
    command = ['/usr/bin/sandbox-exec', '-f', str(Path(config['output'])/'labels_guard.sb'),
               sys.executable, '-m', 'scripts.benchmark_longbench_headtail_control', '--guarded-worker']
    result = subprocess.run(command, input=json.dumps({'api_key': api_key, 'config': config}),
                            text=True, cwd=ROOT, env=environment, check=False)
    if result.returncode:
        raise RuntimeError('Head/tail worker stopped; preserve its journals before any resume')
    return json.loads((Path(config['output'])/'run_summary.json').read_text())


def execute_control(*, api_key, config):
    prepare_control(api_key=api_key, config=config)
    return run_control(api_key=api_key, config=config)


def score_control(config):
    from scripts.audit_context_holdout_execution_v3 import audit_ledger
    config = public_config(config)
    frozen, context = check_control(config)
    out, freeze_sha = Path(config['output']), base.digest(Path(config['output'])/'freeze.json')
    records = []
    costs = base.MonetaryLedger(out/'cost_ledger.json', cap=frozen['control_cap_usd'],
        prices=context['child_config']['prices'], freeze_sha=freeze_sha)
    metadata_by_index = {item['index']: item for item in json.loads((out/'manifest.json').read_text())['selected']}
    for index in frozen['planned_indices']:
        for model in MODELS:
            path = prediction_path(out, model, index)
            if not path.exists():
                raise ValueError('All 114 preceding and all registered control terminals are required before labels')
            record = json.loads(path.read_text())
            if (record.get('terminal') is not True or record.get('index') != index or record.get('model') != model
                    or record.get('freeze_sha256') != freeze_sha or record.get('budget') != BUDGET):
                raise ValueError('Control terminal identity/budget differs')
            item = metadata_by_index[index]
            if (record.get('id') != item['id'] or record.get('context_sha256') != item['trimmed_context_sha256']
                    or record.get('original_context_sha256') != item['original_context_sha256']
                    or record.get('source_spans') != item['source_spans']):
                raise ValueError('Control source projection identity differs')
            audit = audit_ledger(record['ledger'], BUDGET)
            if audit['violations'] or record['usage']['charged_tokens'] != audit['charged_tokens']:
                raise ValueError('Control token ledger failed before labels')
            if (record['usage']['physical_requests'] != audit['physical_requests']
                    or record['usage']['prompt_tokens'] != audit['prompt_tokens']
                    or record['usage']['completion_tokens'] != audit['completion_tokens']):
                raise ValueError('Control usage summary differs from its physical ledger')
            if (record.get('prediction') != base.extract_choice(record.get('content'))
                    or record.get('strict_official_prediction') != strict_official_choice(record.get('content'))):
                raise ValueError('Control parsers differ from actual response text')
            entries = [entry for entry in costs.entries if entry['model'] == model
                       and entry['id'] == f'headtail_control:{index}' and entry.get('status') != 'not_sent']
            if len(entries) != len(record['ledger']):
                raise ValueError('Control monetary/physical correspondence failed')
            for money, entry in zip(entries, record['ledger']):
                if (money['reserved_prompt_tokens'] != entry['admission_prompt_tokens']
                        or money['reserved_output_tokens'] != entry['requested_output_tokens']):
                    raise ValueError('Control monetary pre-admission differs')
                rin, rout = costs.rates(model, entry['admission_prompt_tokens'])
                inp, output = ((entry['usage']['prompt_tokens'], entry['usage']['completion_tokens'])
                               if entry.get('usage_known') else
                               (entry['admission_prompt_tokens'], entry['requested_output_tokens']))
                if Decimal(money['estimated_upper_usd']) != (inp*rin+output*rout)/Decimal(1_000_000):
                    raise ValueError('Control monetary settlement differs')
                metadata = entry.get('provider_metadata') or {}
                if entry.get('usage_known') and (metadata.get('actual_model') != model
                                               or metadata.get('returned_service_tier') != 'default'):
                    raise ValueError('Known control response model/tier differs')
            records.append(record)
    if len(records) != frozen['expected_control_records']:
        raise ValueError('Incomplete registered control set')
    if costs.total() > Decimal(frozen['control_cap_usd']):
        raise ValueError('Control exceeded its own estimate cap')
    if len([entry for entry in costs.entries if entry.get('status') != 'not_sent']) != sum(len(record['ledger']) for record in records):
        raise ValueError('Control global monetary/physical correspondence differs')
    combined = (Decimal(context['parent_cost_upper_usd'])+Decimal(context['recovery_cost_upper_usd'])
                +Decimal(context['preflight_cost_upper_usd'])+costs.total())
    if combined > Decimal(context['global_cap_usd']):
        raise ValueError('Control exceeded the combined estimate cap')
    if (out/'audit.json').exists():
        return json.loads((out/'audit.json').read_text())['summary']
    base.atomic_json(out/'scoring_access.json', {'first_scored_at_unix': time.time(),
        'all_114_and_control_terminal_before_labels': True,
        'control_terminal_count': len(records), 'control_budget_skip': not records})
    recovery.score_recovery(context['recovery_config'])
    composite = json.loads((context['recovery_out']/'audit.json').read_text())
    labels = json.loads(Path(context['parent_config']['labels']).read_text())
    scored = []
    for record in records:
        gold = labels[str(record['index'])]
        scored.append({**record, 'gold': gold, 'correct': record['prediction'] == gold,
                       'strict_official_correct': record['strict_official_prediction'] == gold})
    summary = {}
    for model in MODELS:
        group = [record for record in scored if record['model'] == model]
        harness = [record for record in composite['samples'] if record['model'] == model
                   and record['arm'] == 'harness' and record['index'] in frozen['planned_indices']]
        summary[model] = {'n_planned': len(group), 'primary_correct': sum(record['correct'] for record in group),
            'primary_accuracy_all_planned': sum(record['correct'] for record in group)/len(group) if group else None,
            'strict_official_correct': sum(record['strict_official_correct'] for record in group),
            'strict_format_accepted': sum(record['strict_format_accepted'] for record in group),
            'harness_same_sources_correct': sum(record['task_success'] for record in harness),
            'physical_generations': sum(record['usage']['physical_requests'] for record in group),
            'charged_or_reserved_tokens': sum(record['usage']['charged_tokens'] for record in group),
            'status_counts': dict(Counter(record['status'] for record in group))}
    report = {'experiment': frozen['experiment'], 'freeze_sha256': freeze_sha,
        'parent_freeze_sha256': recovery.PARENT_FREEZE, 'recovery_freeze_sha256': frozen['recovery_freeze_sha256'],
        'control_kind': 'balanced_character_headtail_with_exact_whole_prompt_admission',
        'not_bit_identical_to_upstream_token_half': True, 'comparison_has_unequal_compute': True,
        'budget_skipped': not records, 'planned_indices': frozen['planned_indices'],
        'summary': summary, 'samples': scored,
        'estimated_control_cost_upper_usd': str(costs.total()),
        'combined_cost_with_preflight_upper_usd': str(combined),
        'first_trial_audit_sha256': base.digest(Path(context['parent_config']['output'])/'audit.json'),
        'recovery_composite_audit_sha256': base.digest(context['recovery_out']/'audit.json')}
    base.atomic_json(out/'audit.json', report)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config')
    parser.add_argument('--execute-control', action='store_true')
    parser.add_argument('--score', action='store_true')
    parser.add_argument('--guarded-worker', action='store_true', help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.guarded_worker:
        value = json.loads(sys.stdin.read())
        try:
            _run_worker(api_key=value['api_key'], config=value['config'])
        except Exception as exc:
            print(json.dumps({'headtail_worker_error_type': type(exc).__name__}), flush=True)
            return 1
        return 0
    if not args.config:
        parser.error('--config is required')
    config = json.loads(Path(args.config).read_text())
    if args.score:
        result = score_control(config)
    elif args.execute_control:
        result = execute_control(api_key=getpass.getpass('OpenAI API key (RAM only): '), config=config)
    else:
        parser.error('Choose --execute-control or --score')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
