"""Frozen, label-isolated LongBench native-direct / bounded-Harness evaluation.

Nothing runs on import. Labels are first opened by score_limits after all 72
main records and three separately registered capacity probes are terminal.
Credentials are caller-held RAM values passed through an anonymous stdin pipe.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from decimal import Decimal
import fcntl
import getpass
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import statistics
import subprocess
import sys
import time

from long_context_agent.experiment_budget import ExperimentBudget
from long_context_agent.openai_longbench_bridge import LongBenchResponsesTransport, run_longbench_harness
from long_context_agent.openai_responses_transport import OpenAIResponsesBudgetedClient
from scripts.benchmark_openai_transfer import (
    CostCapExceeded, CostCappedClient, atomic_json, canonical, digest, safe_error,
    scrub_secret, text_digest,
)
from scripts.longbench_v2_protocol import LongBenchRow, extract_choice, official_prompt


ROOT = Path(__file__).resolve().parents[1]
MODELS = ('gpt-6-sol', 'gpt-6-luna', 'gpt-6.1-terra')
NATIVE = 1_050_000
HARNESS_LIMITS = {'context_tokens': 16_384, 'total_tokens': 65_536,
                  'max_calls': 32, 'max_output_tokens': 2_048, 'token_safety_margin': 16}
DIRECT_LIMITS = {'context_tokens': NATIVE, 'total_tokens': NATIVE,
                 'max_calls': 1, 'max_output_tokens': 1_024, 'token_safety_margin': 16}
EXPECTED_PRICES = {
    'gpt-6-sol': {'input': 2, 'output': 10, 'cache_write_30m': 2.5, 'cached_input': .2},
    'gpt-6-luna': {'input': .1, 'output': .5, 'cache_write_30m': .125, 'cached_input': .01},
    'gpt-6.1-terra': {'input': 1, 'output': 5, 'cache_write_30m': 1.25, 'cached_input': .05},
}


def public_config(value):
    allowed = {'models', 'inputs', 'labels', 'manifest', 'raw_cache', 'output',
               'budget_harness', 'budget_direct', 'reasoning', 'timeout',
               'cost_cap_usd', 'prices', 'probes_per_model', 'long_direct_spacing_seconds',
               'base_url'}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError('Unsupported configuration fields; credentials are not configuration')
    config = json.loads(canonical(value))
    if (config.get('models') != list(MODELS) or config.get('reasoning') != 'none'
            or config.get('budget_harness') != HARNESS_LIMITS
            or config.get('budget_direct') != DIRECT_LIMITS
            or config.get('probes_per_model') != 1):
        raise ValueError('Exact models, registered budgets, reasoning=none and one probe per model are required')
    config.setdefault('base_url', 'https://api.openai.com/v1')
    config.setdefault('timeout', 900)
    config.setdefault('long_direct_spacing_seconds', 65)
    if config['base_url'] != 'https://api.openai.com/v1':
        raise ValueError('Only the official OpenAI API is registered')
    for name in ('timeout', 'long_direct_spacing_seconds'):
        if type(config[name]) not in (int, float) or not 0 <= config[name] <= 3600:
            raise ValueError('Invalid timeout or spacing')
    if config['timeout'] == 0:
        raise ValueError('A positive timeout is required')
    cap = Decimal(str(config.get('cost_cap_usd', 0)))
    if not cap.is_finite() or not 0 < cap <= 50:
        raise ValueError('The explicit generation-cost estimate cap must be positive and no higher than USD 50')
    if config.get('prices') != EXPECTED_PRICES:
        raise ValueError('Prices differ from the registered Standard / default-30m envelope')
    for name in ('inputs', 'labels', 'manifest', 'raw_cache', 'output'):
        if not isinstance(config.get(name), str) or not config[name]:
            raise ValueError('All local input and output paths must be explicit')
        config[name] = str(Path(config[name]).resolve())
    # A dedicated label directory lets the OS also deny candidate-label copies.
    label_dir = Path(config['labels']).parent
    if any(Path(config[name]).is_relative_to(label_dir) for name in ('inputs', 'manifest', 'output')):
        raise ValueError('Labels require a separate directory from inference artifacts')
    return config


def read_inputs(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    forbidden = {'answer', 'answers', 'gold', 'label', 'labels', 'correct'}
    needed = {'_id', 'id', 'index', 'context', 'question', 'choice_A', 'choice_B', 'choice_C', 'choice_D'}
    if len(rows) != 12 or len({r.get('index') for r in rows}) != 12 or len({r.get('_id') for r in rows}) != 12:
        raise ValueError('Exactly 12 distinct source cases are required')
    for row in rows:
        if forbidden & set(row) or not needed <= set(row):
            raise ValueError('Inference rows contain labels or lack required fields')
        if type(row['index']) is not int or row['index'] < 0 or row['id'] != row['_id']:
            raise ValueError('Invalid source identity')
        if any(not isinstance(row[key], str) for key in needed - {'index'}):
            raise ValueError('Source, question and choices must be text')
    return rows


def make_row(value):
    row = LongBenchRow.from_mapping(value, offset=value['index'])
    if row.answer:
        raise ValueError('A label reached the inference row')
    return row


def verify_inputs(config):
    manifest = json.loads(Path(config['manifest']).read_text())
    if digest(config['inputs']) != manifest.get('inputs_sha256'):
        raise ValueError('Inputs differ from the selection manifest')
    for key in ('labels_sha256', 'cache_sha256'):
        if not isinstance(manifest.get(key), str) or not re.fullmatch('[0-9a-f]{64}', manifest[key]):
            raise ValueError('Manifest needs declared label and raw-cache hashes')
    rows = read_inputs(config['inputs'])
    selected = {item['index']: item for item in manifest.get('selected', [])}
    if len(selected) != 12 or set(selected) != {row['index'] for row in rows}:
        raise ValueError('Manifest case matrix differs')
    hashes = []
    for value in rows:
        item = selected[value['index']]
        source_sha = text_digest(value['context'])
        if (item.get('id') != value['_id'] or item.get('context_sha256') != source_sha
                or item.get('prompt_sha256') != text_digest(official_prompt(make_row(value)))
                or item.get('context_chars') != len(value['context'])):
            raise ValueError('Source or question/prompt changed after selection')
        hashes.append(source_sha)
    if len(set(hashes)) != 12 or manifest.get('probe_index') not in selected:
        raise ValueError('Distinct sources and a registered probe source are required')
    return rows, manifest


def source_fingerprints():
    files = []
    for folder in ('long_context_agent', 'reading_memory', 'independent_ingestion'):
        files.extend((ROOT / folder).rglob('*.py'))
    files.extend(ROOT / 'scripts' / name for name in (
        'benchmark_longbench_openai_limits.py', 'benchmark_openai_transfer.py',
        'benchmark_longbench_v2_full.py', 'benchmark_longbench_v2_compare.py',
        'longbench_v2_protocol.py', 'audit_context_holdout_execution_v3.py'))
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(set(files))}


def runtime_identity():
    return {'python': sys.version, 'platform': platform.platform()}


def freeze_limits(config):
    config = public_config(config)
    out = Path(config['output'])
    if out.exists() and any(out.iterdir()):
        raise ValueError('Use a new empty output directory')
    rows, manifest = verify_inputs(config)
    out.mkdir(parents=True, exist_ok=True)
    probe = out / 'label_guard_probe.txt'
    probe.write_text('No-label isolation probe.\n')
    directories = [Path(config['labels']).parent, ROOT/'docs', ROOT/'benchmarks']
    profile = '(version 1)\n(allow default)\n'
    profile += ''.join('(deny file-read* (subpath '+json.dumps(str(p))+'))\n' for p in directories)
    profile += ''.join('(deny file-read* (literal '+json.dumps(str(p))+'))\n'
                       for p in (Path(config['raw_cache']), probe))
    (out/'labels_guard.sb').write_text(profile)
    frozen = {'schema_version': 1, 'experiment': 'LongBench v2 OpenAI native-direct / bounded Harness limits',
              'frozen_at_unix': time.time(), 'config': config, 'expected_main_records': 72,
              'expected_probe_records': 3, 'expected_terminal_records': 75,
              'sample_indices': [row['index'] for row in rows],
              'sample_ids': {str(row['index']): row['_id'] for row in rows},
              'source_hashes': {str(row['index']): text_digest(row['context']) for row in rows},
              'inputs_sha256': manifest['inputs_sha256'], 'labels_sha256': manifest['labels_sha256'],
              'raw_cache_sha256_declared': manifest['cache_sha256'],
              'manifest_sha256': digest(config['manifest']), 'probe_index': manifest['probe_index'],
              'code_sha256': source_fingerprints(), 'runtime': runtime_identity(),
              'guard_profile_sha256': digest(out/'labels_guard.sb'),
              'wire_parameters': {'reasoning_effort': 'none', 'store': False, 'service_tier': 'default',
                                  'truncation': 'disabled', 'temperature_sent': False, 'seed_sent': False},
              'transport_encoding': {'oversized_text_encoding': 'ordered_input_text_parts',
                  'text_part_character_limit': LongBenchResponsesTransport.TEXT_PART_CHARS,
                  'text_concatenation': 'lossless_no_inserted_delimiters',
                  'short_input_wire_unchanged': True, 'prompt_cache_ttl_default': '30m',
                  'prompt_cache_ttl_overridden': False,
                  'interpretation': 'Lossless wire encoding only; not model-context expansion'},
              'sample_status': 'Previously used public LongBench sources; regression evaluation, not a fresh blind holdout',
              'cost_definition': 'Conservative generation estimate, not invoice. Separate input max ordinary/read/default-30m write price and output price; input >272000 applies whole-request 2x input / 1.5x output. Unknown delivery retains the reservation.',
              'comparison': 'Native-direct and bounded Harness have different ceilings; not a same-compute accuracy claim',
              'labels_read': False}
    for name in frozen['code_sha256']:
        target = out/'frozen_source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/name, target)
    atomic_json(out/'freeze.json', frozen)
    return {'phase': 'frozen', 'expected_terminal_records': 75, 'labels_read': False,
            'freeze_sha256': digest(out/'freeze.json')}


def check_freeze(config):
    config = public_config(config)
    out = Path(config['output'])
    frozen = json.loads((out/'freeze.json').read_text())
    if (canonical(config) != canonical(frozen['config'])
            or source_fingerprints() != frozen['code_sha256']
            or runtime_identity() != frozen['runtime']
            or digest(config['manifest']) != frozen['manifest_sha256']
            or digest(out/'labels_guard.sb') != frozen['guard_profile_sha256']):
        raise ValueError('Frozen config, source, runtime or guard changed')
    rows, manifest = verify_inputs(config)
    return frozen, rows, manifest


class MonetaryLedger:
    """Default-30m input envelope and output envelope, persisted before sending."""

    def __init__(self, path, *, cap, prices, freeze_sha):
        self.path, self.cap = Path(path), Decimal(str(cap))
        self.prices, self.freeze_sha, self.entries = prices, freeze_sha, []
        if self.path.exists():
            previous = json.loads(self.path.read_text())
            if previous['freeze_sha256'] != freeze_sha or Decimal(previous['cap_usd']) != self.cap:
                raise ValueError('Existing monetary ledger belongs to another freeze')
            self.entries = previous['entries']
            if any(not Decimal(e['estimated_upper_usd']).is_finite() or Decimal(e['estimated_upper_usd']) < 0
                   for e in self.entries):
                raise ValueError('Invalid monetary ledger')

    def rates(self, model, prompt):
        price = self.prices[model]
        long = prompt > 272_000
        return (max(Decimal(str(price[k])) for k in ('input', 'cache_write_30m', 'cached_input'))
                * (2 if long else 1),
                Decimal(str(price['output'])) * (Decimal('1.5') if long else 1))

    def total(self):
        return sum((Decimal(e['estimated_upper_usd']) for e in self.entries), Decimal(0))

    def save(self):
        atomic_json(self.path, {'freeze_sha256': self.freeze_sha, 'cap_usd': str(self.cap),
                               'estimated_upper_usd': str(self.total()), 'entries': self.entries})

    def reserve(self, *, model, sample_id, prompt, completion):
        rin, rout = self.rates(model, prompt)
        amount = (prompt*rin+completion*rout)/Decimal(1_000_000)
        if self.total()+amount > self.cap:
            raise CostCapExceeded('Generation-cost estimate cap denies this request before sending')
        entry = {'reservation': len(self.entries)+1, 'model': model, 'id': sample_id,
                 'reserved_prompt_tokens': prompt, 'reserved_output_tokens': completion,
                 'input_envelope_per_million': str(rin), 'output_per_million': str(rout),
                 'long_input_admission': prompt > 272_000, 'default_cache_ttl': '30m',
                 'reserved_upper_usd': str(amount), 'estimated_upper_usd': str(amount),
                 'status': 'inflight', 'usage_known': False}
        self.entries.append(entry)
        self.save()
        return entry

    def settle(self, entry, ledger):
        if ledger.get('usage_known'):
            usage = ledger['usage']
            amount = (usage['prompt_tokens']*Decimal(entry['input_envelope_per_million'])
                      + usage['completion_tokens']*Decimal(entry['output_per_million']))/Decimal(1_000_000)
            entry.update(usage_known=True, actual_prompt_tokens=usage['prompt_tokens'],
                         actual_output_tokens=usage['completion_tokens'], estimated_upper_usd=str(amount),
                         status='actual_tokens_at_admitted_price_envelope')
            self.save()
            if amount > Decimal(entry['reserved_upper_usd']) or self.total() > self.cap:
                raise RuntimeError('Actual usage exceeded its monetary admission')
        else:
            entry['status'] = 'uncertain_delivery_retains_reservation'
            self.save()


def prediction_path(out, model, index, arm):
    return Path(out)/'predictions'/f'{index}__{model}__{arm}.json'


def expected_jobs(rows, probe_index):
    main = [(model, value, arm) for model in MODELS for value in rows for arm in ('direct', 'harness')]
    random.Random(20261103).shuffle(main)
    probe = next(value for value in rows if value['index'] == probe_index)
    return main+[(model, probe, 'capacity_probe') for model in MODELS]


def verify_guard(config):
    out = Path(config['output'])
    try:
        (out/'label_guard_probe.txt').read_text()
    except PermissionError:
        pass
    else:
        raise RuntimeError('OS label guard probe is readable; refusing inference')
    atomic_json(out/'guard_verification.json', {'verified': True, 'probe_read_denied': True,
                'labels_read': False, 'profile_sha256': digest(out/'labels_guard.sb'), 'time': time.time()})


def _wait_long_direct(out, model, config, freeze_sha):
    path = out/'long_direct_timing.json'
    timing = json.loads(path.read_text()) if path.exists() else {'freeze_sha256': freeze_sha, 'last_start': {}}
    if timing['freeze_sha256'] != freeze_sha:
        raise ValueError('Timing metadata belongs to another freeze')
    delay = timing['last_start'].get(model, 0)+config['long_direct_spacing_seconds']-time.time()
    if delay > 0:
        print(json.dumps({'phase': 'long_direct_spacing', 'model': model,
                          'wait_seconds': round(delay, 1)}), flush=True)
    while delay > 0:
        time.sleep(min(30, delay))
        delay = timing['last_start'].get(model, 0)+config['long_direct_spacing_seconds']-time.time()
    timing['last_start'][model] = time.time()
    atomic_json(path, timing)


def _run_worker(*, api_key, config, transport_factory=LongBenchResponsesTransport,
                client_factory=OpenAIResponsesBudgetedClient, harness_runner=run_longbench_harness):
    config = public_config(config)
    frozen, rows, manifest = check_freeze(config)
    verify_guard(config)
    out, freeze_sha = Path(config['output']), digest(Path(config['output'])/'freeze.json')
    with (out/'.run.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (out/'cost_ledger.json').exists() and any(
                json.loads(p.read_text()).get('ledger') for p in (out/'journals').glob('*.json')):
            raise ValueError('Existing generation journals lack their cost ledger; refusing quota reset')
        costs = MonetaryLedger(out/'cost_ledger.json', cap=config['cost_cap_usd'],
                               prices=config['prices'], freeze_sha=freeze_sha)
        transports = {model: transport_factory(api_key=api_key, model=model,
                      base_url=config['base_url'], reasoning_effort='none', timeout=config['timeout'])
                      for model in MODELS}
        jobs = expected_jobs(rows, manifest['probe_index'])
        for model, value, arm in jobs:
            index, row = value['index'], make_row(value)
            target = prediction_path(out, model, index, arm)
            if target.exists():
                saved = json.loads(target.read_text())
                if (saved.get('freeze_sha256') != freeze_sha or saved.get('model') != model
                        or saved.get('index') != index or saved.get('arm') != arm
                        or saved.get('context_sha256') != text_digest(row.context) or saved.get('terminal') is not True):
                    raise ValueError('Existing terminal identity differs from the freeze')
                continue
            transport, prompt = transports[model], official_prompt(row)
            start_counts, start_responses = len(transport.count_audit), len(transport.response_audit)
            previous_counts, previous_responses = [], []
            journal_path = out/'journals'/target.name
            budget_dict = dict(config['budget_harness'] if arm == 'harness' else config['budget_direct'])
            if journal_path.exists():
                previous = json.loads(journal_path.read_text())
                if previous.get('freeze_sha256') != freeze_sha:
                    raise ValueError('Interrupted journal belongs to another freeze')
                budget_dict = previous['budget']
            budget = ExperimentBudget(**budget_dict)
            def journal(ledger):
                atomic_json(journal_path, scrub_secret({
                    'index': index, 'model': model, 'arm': arm, 'freeze_sha256': freeze_sha,
                    'budget': budget_dict, 'ledger': ledger,
                    'count_audit': previous_counts+transport.count_audit[start_counts:],
                    'response_audit': previous_responses+transport.response_audit[start_responses:]}, api_key))
            client = client_factory(transport, budget, on_ledger_update=journal)
            capped = CostCappedClient(client, costs, f'{arm}:{index}')
            result, error, measurements, content, status = {}, None, {}, '', 'not_started'
            started = time.perf_counter()
            block = out/'model_blocks'/f'{model}.json'
            if journal_path.exists():
                previous_counts = previous.get('count_audit', [])
                previous_responses = previous.get('response_audit', [])
                client.ledger = previous['ledger']
                error = {'type': 'InterruptedTask', 'message': 'Interrupted task is terminal; no replay or quota reset'}
                status = 'interrupted'
            elif block.exists():
                error = {'type': 'ModelBlocked', 'message': 'The exact registered model has an access block'}
                status = 'model_blocked'
            else:
                journal([])
                try:
                    if arm == 'capacity_probe':
                        # A probe has its own registered submission ceiling. It
                        # does not redefine the provider's native window. Count
                        # failures remain terminal errors, not capacity results.
                        raw_prompt = transport.count(messages=[{'role': 'user', 'content': prompt}])
                        budget_dict.update(context_tokens=raw_prompt+1024+4096, total_tokens=raw_prompt+1024+4096)
                        budget = ExperimentBudget(**budget_dict)
                        client = client_factory(transport, budget, on_ledger_update=journal)
                        capped = CostCappedClient(client, costs, f'{arm}:{index}')
                    source_tokens = transport.count(text=row.context)
                    admission = client.count_messages([{'role': 'user', 'content': prompt}])
                    measurements = {'source_envelope_input_tokens': source_tokens,
                                    'full_prompt_admission_input_tokens': admission,
                                    'output_reserved_tokens': 1024 if arm != 'harness' else 2048,
                                    'native_context_tokens': NATIVE,
                                    'fits_native_direct': admission+1024 <= NATIVE,
                                    'count_semantics': 'Exact Responses input envelope; admission includes budget notice and 16 safety tokens'}
                    if arm == 'direct' and admission+1024 > NATIVE:
                        status = 'native_admission_denied'
                    elif arm == 'harness':
                        result = harness_runner(row, capped)
                        if result.get('mode') != 'raw_source':
                            raise RuntimeError('Registered Harness extension did not use the raw-source route')
                        content = str(result.get('prediction') or '')
                        status = 'returned' if content in 'ABCD' and len(content) == 1 else 'no_parseable_answer'
                    else:
                        if arm == 'capacity_probe' and admission+1024 <= NATIVE:
                            raise ValueError('Registered capacity probe does not exceed native admission')
                        if admission > 272_000:
                            _wait_long_direct(out, model, config, freeze_sha)
                        content = capped.complete([{'role': 'user', 'content': prompt}], max_tokens=1024)
                        status = 'probe_unexpectedly_returned' if arm == 'capacity_probe' else 'returned'
                    # A controller can catch a failed verifier: verify every known response,
                    # including incomplete responses, rather than only its final candidate.
                    for entry in client.ledger:
                        meta = entry.get('provider_metadata') or {}
                        if entry.get('usage_known') and meta.get('actual_model') != model:
                            raise RuntimeError('Known response model differs from its registered exact ID')
                        if entry.get('usage_known') and meta.get('returned_service_tier') != 'default':
                            raise RuntimeError('Known response service tier differs from default')
                        details = (meta.get('provider_usage') or {}).get('input_tokens_details') or {}
                        writes = details.get('cache_write_tokens_details') or {}
                        if writes.get('ttl_12h_tokens', 0):
                            raise RuntimeError('Unrequested 12h cache writes differ from the registered price envelope')
                except Exception as exc:
                    content, result = '', {}
                    error = safe_error(exc, api_key)
                    if arm == 'capacity_probe' and 'status 400; code context_length_exceeded' in error['message']:
                        status = 'provider_context_rejected'
                    else:
                        status = 'error'
            access_errors = [entry.get('error', '') for entry in client.ledger]
            if error:
                access_errors.append(error['message'])
            if any(re.search(r'status (401|403|404)\b', message) for message in access_errors):
                atomic_json(block, {'model': model, 'freeze_sha256': freeze_sha,
                                   'reason': 'explicit_authentication_or_model_access_failure', 'time': time.time()})
            prediction = content if arm == 'harness' and len(content) == 1 and content in 'ABCD' else extract_choice(content)
            record = {'index': index, 'id': row.item_id, 'model': model, 'arm': arm,
                      'terminal': True, 'status': status, 'freeze_sha256': freeze_sha,
                      'context_sha256': text_digest(row.context), 'context_chars': len(row.context),
                      'domain': row.domain, 'budget': budget_dict, 'prediction': prediction,
                      'content': content, 'result': result, 'error': error, 'measurements': measurements,
                      'usage': client.summary(), 'ledger': client.ledger,
                      'ledger_error_count': sum(entry.get('status') == 'error' for entry in client.ledger),
                      'count_audit': previous_counts+transport.count_audit[start_counts:],
                      'response_audit': previous_responses+transport.response_audit[start_responses:],
                      'seconds': round(time.perf_counter()-started, 4)}
            journal(client.ledger)
            atomic_json(target, scrub_secret(record, api_key))
            print(json.dumps({'index': index, 'model': model, 'arm': arm, 'status': status,
                              'calls': len(client.ledger), 'tokens': record['usage']['charged_tokens'],
                              'error': error['type'] if error else None}), flush=True)
        summary = {'phase': 'inference_complete', 'planned_main_records': 72, 'planned_probe_records': 3,
                   'terminal_records': sum(prediction_path(out, model, row['index'], arm).is_file()
                                           for model, row, arm in jobs),
                   'labels_read': False, 'estimated_generation_cost_upper_usd': str(costs.total())}
        atomic_json(out/'run_summary.json', summary)
        return summary


def run_limits(*, api_key, config):
    config = public_config(config)
    check_freeze(config)
    if not isinstance(api_key, str) or not api_key:
        raise ValueError('A caller-held credential is required')
    if sys.platform != 'darwin' or not Path('/usr/bin/sandbox-exec').is_file():
        raise RuntimeError('This strict protocol requires local macOS sandbox-exec')
    environment = {name: value for name, value in os.environ.items()
                   if not any(token in name.upper() for token in ('KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL'))}
    command = ['/usr/bin/sandbox-exec', '-f', str(Path(config['output'])/'labels_guard.sb'),
               sys.executable, '-m', 'scripts.benchmark_longbench_openai_limits', '--guarded-worker']
    result = subprocess.run(command, input=json.dumps({'api_key': api_key, 'config': config}),
                            text=True, cwd=ROOT, env=environment, check=False)
    if result.returncode:
        raise RuntimeError('Guarded LongBench worker stopped; inspect sanitized journals before resume')
    return json.loads((Path(config['output'])/'run_summary.json').read_text())


def score_limits(config):
    config = public_config(config)
    frozen, rows, manifest = check_freeze(config)
    out, freeze_sha = Path(config['output']), digest(Path(config['output'])/'freeze.json')
    records = []
    for model, value, arm in expected_jobs(rows, manifest['probe_index']):
        path = prediction_path(out, model, value['index'], arm)
        if not path.exists():
            raise ValueError('All 75 terminal records must exist before labels are opened')
        record = json.loads(path.read_text())
        if (record.get('terminal') is not True or record.get('index') != value['index']
                or record.get('id') != value['_id'] or record.get('model') != model or record.get('arm') != arm
                or record.get('freeze_sha256') != freeze_sha
                or record.get('context_sha256') != frozen['source_hashes'][str(value['index'])]):
            raise ValueError('Terminal identity differs from the freeze')
        records.append(record)
    from scripts.audit_context_holdout_execution_v3 import audit_ledger
    costs = MonetaryLedger(out/'cost_ledger.json', cap=config['cost_cap_usd'],
                           prices=config['prices'], freeze_sha=freeze_sha)
    sent_reservations = [entry for entry in costs.entries if entry.get('status') != 'not_sent']
    if len(sent_reservations) != sum(len(record['ledger']) for record in records):
        raise ValueError('Monetary reservations do not correspond to physical requests')
    if costs.total() > Decimal(str(config['cost_cap_usd'])):
        raise ValueError('Monetary cap exceeded before label access')
    for record in records:
        if audit_ledger(record['ledger'], record['budget'])['violations']:
            raise ValueError('Token ledger failed integrity before label access')
        entries = [entry for entry in costs.entries if entry['model'] == record['model']
                   and entry['id'] == f"{record['arm']}:{record['index']}" and entry.get('status') != 'not_sent']
        if len(entries) != len(record['ledger']):
            raise ValueError('Per-task monetary reservation mismatch')
        for money, usage in zip(entries, record['ledger']):
            if (money['reserved_prompt_tokens'] != usage['admission_prompt_tokens']
                    or money['reserved_output_tokens'] != usage['requested_output_tokens']):
                raise ValueError('Monetary pre-admission mismatch')
            rin, rout = costs.rates(record['model'], usage['admission_prompt_tokens'])
            charged = ((usage['usage']['prompt_tokens']*rin+usage['usage']['completion_tokens']*rout)
                       if usage.get('usage_known') else
                       (usage['admission_prompt_tokens']*rin+usage['requested_output_tokens']*rout))/Decimal(1_000_000)
            if Decimal(money['estimated_upper_usd']) != charged:
                raise ValueError('Monetary settlement mismatch before label access')
            metadata = usage.get('provider_metadata') or {}
            if usage.get('usage_known') and (metadata.get('actual_model') != record['model']
                                            or metadata.get('returned_service_tier') != 'default'):
                raise ValueError('Known response identity or service tier failed before label access')
    # Label reading begins only after the whole matrix passes the gates above.
    if digest(config['labels']) != frozen['labels_sha256']:
        raise ValueError('Labels differ from their frozen declared digest')
    labels = json.loads(Path(config['labels']).read_text())
    if set(labels) != {str(row['index']) for row in rows} or any(value not in 'ABCD' or len(value) != 1 for value in labels.values()):
        raise ValueError('Label identities or choices differ from the frozen cohort')
    atomic_json(out/'scoring_access.json', {'first_scored_at_unix': time.time(),
                'all_75_terminal_before_label_access': True, 'labels_sha256': frozen['labels_sha256']})
    main = []
    for record in records:
        if record['arm'] == 'capacity_probe':
            continue
        denied = record['status'] == 'native_admission_denied'
        available = record['prediction'] in ('A', 'B', 'C', 'D')
        main.append({**record, 'gold': labels[str(record['index'])],
                     'correct': None if denied else bool(available and record['prediction'] == labels[str(record['index'])]),
                     'semantic_answer_available': available and not denied,
                     'task_success': bool(available and record['prediction'] == labels[str(record['index'])]) and not denied})
    summary = {}
    for model in MODELS:
        model_summary = {}
        for arm in ('direct', 'harness'):
            group = [record for record in main if record['model'] == model and record['arm'] == arm]
            answers = [record for record in group if record['semantic_answer_available']]
            model_summary[arm] = {'n_planned': 12, 'n_answer_available': len(answers),
                'correct_answers': sum(record['task_success'] for record in group),
                'task_success_rate_all_planned': sum(record['task_success'] for record in group)/12,
                'accuracy_among_available_answers': statistics.mean(record['correct'] for record in answers) if answers else None,
                'available_answer_accuracy_is_secondary_conditional': True,
                'native_admission_denied': sum(record['status'] == 'native_admission_denied' for record in group),
                'status_counts': dict(Counter(record['status'] for record in group)),
                'physical_generations': sum(record['usage']['physical_requests'] for record in group),
                'charged_or_reserved_tokens': sum(record['usage']['charged_tokens'] for record in group)}
            if arm == 'direct':
                eligible = [record for record in group if record['measurements'].get('fits_native_direct') is True]
                model_summary[arm].update(
                    n_native_admissible=len(eligible),
                    native_eligibility_unknown=sum('fits_native_direct' not in record['measurements'] for record in group),
                    correct_native_admissible=sum(record['task_success'] for record in eligible),
                    primary_task_success_rate_native_admissible=(
                        sum(record['task_success'] for record in eligible)/len(eligible) if eligible else None))
            else:
                model_summary[arm]['primary_task_success_rate_all_planned'] = model_summary[arm]['task_success_rate_all_planned']
        paired, native_pairs = [], []
        for value in rows:
            arms = {record['arm']: record for record in main if record['model'] == model and record['index'] == value['index']}
            if all(arms[arm]['semantic_answer_available'] for arm in ('direct', 'harness')):
                paired.append((arms['direct']['correct'], arms['harness']['correct']))
            if arms['direct']['measurements'].get('fits_native_direct') is True:
                native_pairs.append((arms['direct']['task_success'], arms['harness']['task_success']))
        model_summary['paired_available'] = {'n': len(paired),
            'secondary_conditioned_on_both_answers_available': True,
            'both_correct': sum(a and b for a, b in paired), 'direct_only_correct': sum(a and not b for a, b in paired),
            'harness_only_correct': sum(b and not a for a, b in paired), 'both_wrong': sum(not a and not b for a, b in paired)}
        model_summary['paired_native_admissible'] = {'n': len(native_pairs),
            'failed_or_missing_answers_retained': True,
            'both_correct': sum(a and b for a, b in native_pairs),
            'direct_only_correct': sum(a and not b for a, b in native_pairs),
            'harness_only_correct': sum(b and not a for a, b in native_pairs),
            'both_wrong': sum(not a and not b for a, b in native_pairs)}
        summary[model] = model_summary
    report = {'experiment': frozen['experiment'], 'freeze_sha256': freeze_sha,
              'sample_status': frozen['sample_status'], 'summary': summary, 'samples': main,
              'capacity_probes': [record for record in records if record['arm'] == 'capacity_probe'],
              'estimated_generation_cost_upper_usd': str(costs.total()),
              'not_same_compute_comparison': True, 'official_leaderboard_submission': False}
    atomic_json(out/'audit.json', report)
    return summary


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
            print(json.dumps({'worker_error_type': type(exc).__name__}), flush=True)
            return 1
        return 0
    if not args.phase or not args.config:
        parser.error('Specify a phase and public JSON config')
    config = json.loads(Path(args.config).read_text())
    if args.phase == 'freeze':
        result = freeze_limits(config)
    elif args.phase == 'score':
        result = score_limits(config)
    else:
        if not args.execute_frozen_run:
            parser.error('Model execution requires --execute-frozen-run')
        result = run_limits(api_key=getpass.getpass('OpenAI API key (RAM only): '), config=config)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
