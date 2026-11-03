#!/usr/bin/env python3
"""Frozen, label-blind context comparison with shared physical call budgets.

Run and scoring are separate phases. The run phase never opens labels.
All methods share the same model, sampling settings, context ceiling,
cumulative input+output token ceiling, and physical generation request cap.
"""
from __future__ import annotations

import argparse
import fcntl
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
from importlib.metadata import version, PackageNotFoundError
import json
from pathlib import Path
import platform
import random
import shutil
import statistics
import sys
import time
import traceback

from long_context_agent.experiment_budget import BudgetedClient, ExperimentBudget, ProviderTransport, load_provider_env


ROOT = Path(__file__).resolve().parents[1]
CODE_FILES = [
    'long_context_agent/harness.py',
    'long_context_agent/__init__.py',
    'long_context_agent/adaptive_evidence.py',
    'long_context_agent/experiment_budget.py',
    'scripts/context_baselines.py',
    'scripts/memgpt_official_adapter.py',
    'scripts/prepare_context_holdout.py',
    'scripts/benchmark_context_holdout.py',
    'scripts/analyze_context_holdout.py',
    'scripts/run_guarded_context_holdout.py',
]


def digest_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    tmp.replace(path)


def read_inputs(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample identities')
    forbidden = {'answer', 'answers', 'gold', 'label', 'labels', 'context_window_text_with_labels'}
    if any(forbidden.intersection(row) for row in rows):
        raise ValueError('Inference inputs contain a label field')
    for row in rows:
        if not {'id', 'context', 'question'}.issubset(row):
            raise ValueError('Missing inference fields')
    return rows


def source_fingerprints():
    hashes = {name: digest_file(ROOT / name) for name in CODE_FILES}
    # Include every local harness module, including the isolated immutable v3.1
    # comparator. No comparator may accidentally import the upgraded ledger.
    for path in sorted((ROOT / 'long_context_agent').rglob('*.py')):
        hashes[str(path.relative_to(ROOT))] = digest_file(path)
    legacy_manifest = ROOT / 'long_context_agent/legacy_v31/manifest.json'
    if legacy_manifest.exists():
        hashes[str(legacy_manifest.relative_to(ROOT))] = digest_file(legacy_manifest)
    round2 = ROOT / 'scripts/prepare_context_holdout_round2.py'
    if round2.exists():
        hashes[str(round2.relative_to(ROOT))] = digest_file(round2)
    for name in ('long_context_agent/adaptive_evidence_v3.py', 'long_context_agent/fact_ledger.py',
                 'scripts/prepare_context_holdout_round3.py', 'tests/test_fact_ledger.py',
                 'tests/test_adaptive_evidence_v3.py', 'tests/test_context_holdout_round3.py',
                 'scripts/audit_context_holdout_execution_v3.py', 'tests/test_context_holdout_execution_v3.py',
                 'work/context_harness_v3_protocol_20261101.json',
                 'scripts/prepare_context_holdout_round4.py', 'tests/test_context_holdout_round4.py',
                 'tests/test_record_parser_v31.py', 'tests/test_adaptive_evidence_v31.py',
                 'long_context_agent/record_protocol.py', 'tests/test_record_protocol.py',
                 'work/context_harness_v31_protocol_20261101.json',
                 'scripts/prepare_context_holdout_round5.py', 'tests/test_context_holdout_round5.py',
                 'tests/test_adaptive_evidence_v32.py', 'tests/test_semantic_coverage_v32.py',
                 'tests/test_legacy_v31.py', 'work/context_harness_v32_protocol_20261101.json',
                 'work/materialize_cohort5_curl.py'):
        path = ROOT / name
        if path.exists():
            hashes[name] = digest_file(path)
    # Pin the actual vendored code, not just a potentially stale commit label.
    vendor = ROOT / 'work/context_baselines_vendor'
    if vendor.exists():
        for path in sorted(vendor.rglob('*.py')):
            if any(p in {'site-packages', '.venv', 'venv', '__pycache__'} for p in path.parts):
                continue
            hashes[str(path.relative_to(ROOT))] = digest_file(path)
        # System prompts and packaged schemas affect inference too.
        for package in (vendor / 'rlm/rlm', vendor / 'memgpt/memgpt'):
            if package.exists():
                for path in sorted(package.rglob('*')):
                    if path.is_file() and path.suffix in {'.txt', '.json', '.yaml', '.yml', '.toml'}:
                        hashes[str(path.relative_to(ROOT))] = digest_file(path)
        for name in ('manifest.json', 'requirements.lock.txt', 'memgpt_import_probe.json'):
            if (vendor / name).exists():
                hashes[str((vendor / name).relative_to(ROOT))] = digest_file(vendor / name)
    for name in ('manifest.json', 'requirements.lock.txt'):
        path = ROOT / 'work/memgpt_official_probe' / name
        if path.exists():
            hashes[str(path.relative_to(ROOT))] = digest_file(path)
    for name in ('context_adapter_compatibility_20261101.json',
                 'context_holdout_round2_protocol_registration.json',
                 'context_native_budget_verification_20261101.json'):
        path = ROOT / 'work' / name
        if path.exists():
            hashes[str(path.relative_to(ROOT))] = digest_file(path)
    return hashes


def runtime_identity():
    packages = {}
    for name in ('rich', 'python-dotenv', 'python-dateutil', 'numpy', 'openai', 'tiktoken'):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = None
    return {'python': sys.version, 'platform': platform.platform(), 'packages': packages}


def freeze(args):
    out = Path(args.output)
    if (out / 'freeze.json').exists():
        raise ValueError('A freeze manifest already exists; do not replace an experiment')
    rows = read_inputs(args.inputs)
    manifest_path = Path(args.inputs).parent / 'manifest.json'
    data_manifest = json.loads(manifest_path.read_text())
    if data_manifest['inputs_sha256'] != digest_file(args.inputs):
        raise ValueError('Inputs do not match the sealed data manifest')
    budget = ExperimentBudget(context_tokens=args.context_tokens, total_tokens=args.total_tokens,
                              max_calls=args.max_calls, max_output_tokens=args.max_output_tokens)
    record = {'frozen_at_unix': time.time(), 'inputs': str(Path(args.inputs).resolve()),
              'inputs_sha256': digest_file(args.inputs), 'sample_ids': [r['id'] for r in rows],
              'methods': args.methods.split(','), 'budget': asdict(budget),
              'model': args.model, 'code_sha256': source_fingerprints(),
              'runtime': runtime_identity(),
              'data_manifest': str(manifest_path.resolve()),
              'data_manifest_sha256': digest_file(manifest_path),
              'labels_sha256': data_manifest['labels_sha256'],
              'official_scorer_sha256': data_manifest['scorer_sha256'],
              'labels_opened_by_runner': False,
              'selection': 'Preselected by independent data preparation before model inference',
              'limitations': ['Public dataset; model pretraining contamination cannot be ruled out',
                              'Budget ceilings are equal; actual consumption may differ',
                              'This is a local controlled evaluation, not an official leaderboard submission']}
    out.mkdir(parents=True, exist_ok=True)
    guard_probe = (out / 'label_guard_probe.txt').resolve()
    guard_probe.write_text('Isolation probe only; contains no answers.\n')
    label_path = (Path(args.inputs).parent / 'labels.json').resolve()
    profile = '(version 1)\n(allow default)\n' + ''.join(
        '(deny file-read* (literal ' + json.dumps(str(path)) + '))\n'
        for path in (label_path, guard_probe))
    (out / 'labels_guard.sb').write_text(profile)
    record['label_guard_profile_sha256'] = digest_file(out / 'labels_guard.sb')
    record['label_guard_probe'] = str(guard_probe)
    for name in record['code_sha256']:
        destination = out / 'frozen_source' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, destination)
    atomic_json(out / 'freeze.json', record)
    print(json.dumps({'phase': 'frozen', 'samples': len(rows), 'methods': record['methods'], 'budget': record['budget']}), flush=True)


def check_freeze(args):
    frozen = json.loads((Path(args.output) / 'freeze.json').read_text())
    if digest_file(args.inputs) != frozen['inputs_sha256']:
        raise ValueError('Frozen inputs changed')
    if digest_file(frozen['data_manifest']) != frozen['data_manifest_sha256']:
        raise ValueError('Sealed data manifest changed')
    if digest_file(Path(args.output) / 'labels_guard.sb') != frozen['label_guard_profile_sha256']:
        raise ValueError('Label isolation profile changed')
    current = source_fingerprints()
    if current != frozen['code_sha256']:
        changed = sorted(k for k in set(current) | set(frozen['code_sha256']) if current.get(k) != frozen['code_sha256'].get(k))
        raise ValueError('Code changed after freeze: ' + ', '.join(changed[:10]))
    if runtime_identity() != frozen['runtime']:
        raise ValueError('Runtime dependencies changed after freeze')
    return frozen


def run_one(sample, method, transport, budget, output, freeze_sha):
    target = Path(output) / 'predictions' / f'{sample["id"]}__{method}.json'
    if target.exists():
        existing = json.loads(target.read_text())
        if existing.get('freeze_sha256') != freeze_sha:
            raise ValueError('Cannot reuse predictions from another freeze')
        return existing
    journal_path = Path(output) / 'journals' / f'{sample["id"]}__{method}.json'
    def journal(ledger):
        atomic_json(journal_path, {'id': sample['id'], 'method': method,
                                  'freeze_sha256': freeze_sha, 'ledger': ledger})
    client = BudgetedClient(transport, budget, on_ledger_update=journal)
    if journal_path.exists():
        previous = json.loads(journal_path.read_text())
        if previous.get('freeze_sha256') != freeze_sha:
            raise ValueError('An interrupted attempt belongs to another freeze')
        client.ledger = previous.get('ledger', [])
        # The controller's Python/REPL state is not checkpointed. Fail closed
        # instead of restarting with a fresh quota or fabricating a final.
        record = {'id': sample['id'], 'method': method, 'freeze_sha256': freeze_sha,
                  'context_sha256': hashlib.sha256(sample['context'].encode()).hexdigest(),
                  'context_chars': len(sample['context']),
                  'context_len': sample.get('context_len', sample.get('metadata', {}).get('context_len')),
                  'task_type': sample.get('task_type'), 'prediction': '', 'result': {},
                  'error': {'type': 'InterruptedTask', 'message': 'Prior attempt was interrupted; quota was not reset and requests were not replayed'},
                  'usage': client.summary(), 'ledger': client.ledger, 'seconds': 0}
        atomic_json(target, record)
        return record
    journal([])
    started = time.perf_counter()
    # Deliberately pass only these fields across the inference boundary.
    payload = {k: sample[k] for k in ('id', 'context', 'question', 'task_type') if k in sample}
    result, error = {}, None
    try:
        if method == 'adaptive_v3':
            from long_context_agent.harness import LongContextHarness
            harness = LongContextHarness(None, max_context_tokens=budget.context_tokens,
                                         reserve_tokens=min(budget.max_output_tokens, budget.context_tokens - 1))
            result = harness.answer_with_structured_facts(payload, client)
            result['baseline_identity'] = {'implementation': 'local_structured_facts_v3',
                                            'revision': '3.2', 'method': method, 'deterministic_aggregation': True}
        elif method == 'adaptive_v31':
            from long_context_agent.legacy_v31 import run_adaptive_v3
            manifest_path = ROOT / 'long_context_agent/legacy_v31/manifest.json'
            manifest = json.loads(manifest_path.read_text())
            for name, expected in manifest['files'].items():
                if digest_file(manifest_path.parent / name) != expected:
                    raise ValueError('Frozen v3.1 comparator changed: ' + name)
            result = run_adaptive_v3(payload, client)
            result['baseline_identity'] = {
                'implementation': 'isolated_frozen_structured_facts_v31', 'revision': '3.1',
                'method': method, 'deterministic_aggregation': True,
                'source_experiment': manifest['source_experiment'],
                'source_freeze_sha256': manifest['source_freeze_sha256'],
                'original_source_sha256': manifest['files'],
                'differences': ['Only package namespace and minimal __init__ change; all four controller/ledger/protocol source files are byte-identical to the original snapshot.']}
        elif method in {'adaptive', 'static'}:
            from long_context_agent.adaptive_evidence import AdaptiveConfig
            from long_context_agent.harness import LongContextHarness
            config = AdaptiveConfig(max_rounds=1) if method == 'static' else AdaptiveConfig()
            harness = LongContextHarness(None, max_context_tokens=budget.context_tokens,
                                         reserve_tokens=min(budget.max_output_tokens, budget.context_tokens - 1))
            result = harness.answer_with_evidence_gaps(payload, client, config=config)
            result['baseline_identity'] = {'implementation': 'local', 'method': method,
                                            'adaptive_followup_enabled': method == 'adaptive'}
        elif method == 'rlm':
            from scripts.context_baselines import run_rlm
            result = run_rlm(payload, client)
        elif method == 'memgpt':
            from scripts.memgpt_official_adapter import run_memgpt_official
            result = run_memgpt_official(payload, client)
        else:
            raise ValueError('Unknown method: ' + method)
        if result.get('error'):
            error = result['error'] if isinstance(result['error'], dict) else {'type': 'MethodError', 'message': str(result['error'])}
    except Exception as exc:
        error = {'type': type(exc).__name__, 'message': str(exc)[:1000],
                 'traceback': traceback.format_exc(limit=5)}
    record = {'id': sample['id'], 'method': method, 'freeze_sha256': freeze_sha,
              'context_sha256': hashlib.sha256(sample['context'].encode()).hexdigest(),
              'context_chars': len(sample['context']),
              'context_len': sample.get('context_len', sample.get('metadata', {}).get('context_len')), 'task_type': sample.get('task_type'),
              'prediction': str(result.get('answer') or ''), 'result': result,
              'error': error, 'usage': client.summary(), 'ledger': client.ledger,
              'seconds': round(time.perf_counter() - started, 4)}
    atomic_json(target, record)
    print(json.dumps({'id': sample['id'], 'method': method, 'answered': bool(record['prediction'].strip()),
                      'calls': record['usage']['physical_requests'],
                      'tokens': record['usage']['charged_tokens'],
                      'error': error['type'] if error else None, 'seconds': record['seconds']}), flush=True)
    return record


def _run(args):
    frozen = check_freeze(args)
    provider = load_provider_env(args.provider_env)
    provider['model'] = frozen['model']
    transport = ProviderTransport(**provider, timeout=args.timeout)
    rows = read_inputs(args.inputs)
    jobs = [(row, method) for row in rows for method in frozen['methods']]
    random.Random(frozen['budget']['seed']).shuffle(jobs)
    freeze_sha = digest_file(Path(args.output) / 'freeze.json')
    budget = ExperimentBudget(**frozen['budget'])
    # This is tokenizer measurement, not a model generation. It is recorded
    # separately rather than silently counted as an inference request.
    measured = Path(args.output) / 'context_measurements.json'
    if not measured.exists():
        model_info = transport.request('/models')
        native = next((x.get('max_model_len') for x in model_info.get('data', []) if x.get('id') == frozen['model']), None)
        measures = []
        for row in rows:
            actual = transport.count(text=row['context'])
            measures.append({'id': row['id'], 'context_chars': len(row['context']),
                             'nominal_dataset_tokens': row.get('metadata', {}).get('context_len'),
                             'qwen_source_tokens': actual, 'provider_reported_native_limit': native,
                             'exceeds_native_limit': actual > native if native else None,
                             'exceeds_experimental_limit': actual > budget.context_tokens})
        atomic_json(measured, {'count_method': 'provider /tokenize raw source add_special_tokens=false',
                             'model': frozen['model'], 'samples': measures})
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, sample, method, transport, budget, args.output, freeze_sha)
                   for sample, method in jobs]
        for f in as_completed(futures):
            f.result()
    print(json.dumps({'phase': 'inference_complete', 'method_sample_pairs': len(jobs), 'labels_opened': False}), flush=True)


def run(args):
    lock_path = Path(args.output) / '.run.lock'
    with lock_path.open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another process already owns this experiment output') from exc
        _run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['freeze', 'run'])
    parser.add_argument('--inputs', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--provider-env')
    parser.add_argument('--model', default='Qwen3.8-Flash-Next-FP8')
    parser.add_argument('--methods', default='adaptive,static,rlm,memgpt')
    parser.add_argument('--context-tokens', type=int, default=16384)
    parser.add_argument('--total-tokens', type=int, default=65536)
    parser.add_argument('--max-calls', type=int, default=8)
    parser.add_argument('--max-output-tokens', type=int, default=2048)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=180)
    args = parser.parse_args()
    (freeze if args.phase == 'freeze' else run)(args)


if __name__ == '__main__':
    main()
