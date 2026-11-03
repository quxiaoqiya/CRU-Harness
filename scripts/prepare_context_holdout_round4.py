#!/usr/bin/env python3
"""Seal stage-two metadata before parser/plan repair; materialize only explicitly.

No model requests. Default execution reads only the existing metadata catalogue
and prior manifests. --materialize is a later, separate preparation phase: the
process exports gold solely into private labels.json and never prints semantics.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
import shutil
import sys
from urllib.parse import urlencode

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.prepare_context_holdout import (
    DATASET, REVISION, SCORER_COMMIT, SCORER_URL, choose_holdout, digest,
    fetch, inference_record, json_bytes, label_record,
)
from scripts.prepare_context_holdout_round3 import atomic_bytes, prior_exclusions, seal_plan

LENGTHS = (262144, 1048576)
SEED, PER_LENGTH, N = 20261104, 4, 8
PRIOR_DIRS = tuple(ROOT / 'data' / name for name in (
    'context_holdout_20261101', 'context_holdout_20261101_round2', 'context_holdout_20261101_round3'))
DEFAULT_OUT = ROOT / 'data/context_holdout_20261101_round4'
META_KEYS = ('id', 'context_len', 'dataset', 'task_group', 'task', 'answer_type', 'context_window_id', '_row_idx')


def select_from_catalog(catalog: list[dict], manifests: list[dict]) -> tuple[list[dict], set, set]:
    contexts, hashes = prior_exclusions(manifests)
    available = [row for row in catalog if (str(row['dataset']), int(row['context_window_id'])) not in contexts]
    chosen = choose_holdout(available, lengths=LENGTHS, per_length=PER_LENGTH, seed=SEED)
    identities = {(str(r['dataset']), int(r['context_window_id'])) for r in chosen}
    if len(chosen) != N or len(identities) != N or identities & contexts:
        raise ValueError('Fresh stage-two selection violates context identity exclusions')
    return [{key: row[key] for key in META_KEYS} for row in chosen], contexts, hashes


def selection_plan() -> tuple[dict, list[dict]]:
    manifests = [json.loads((folder / 'manifest.json').read_text()) for folder in PRIOR_DIRS]
    if [m['n'] for m in manifests] != [18, 18, 12]:
        raise ValueError('Expected consumed cohort sizes 18,18,12')
    if len({m['parquet_revision'] for m in manifests}) != 1:
        raise ValueError('Prior source revisions differ')
    catalog_path = PRIOR_DIRS[0] / 'source/metadata_catalog.json'
    raw = catalog_path.read_bytes(); catalog = json.loads(raw)
    chosen, contexts, hashes = select_from_catalog(catalog, manifests)
    if len(contexts) != 48 or len(hashes) != 48:
        raise ValueError('Exactly 48 consumed source identities and hashes must be excluded')
    availability = []
    for length in LENGTHS:
        ids = {(str(r['dataset']), int(r['context_window_id'])) for r in catalog if int(r['context_len']) == length}
        availability.append({'context_len': length, 'distinct_contexts': len(ids),
                             'consumed_contexts': len(ids & contexts), 'remaining_contexts': len(ids - contexts)})
    plan = {
        'cohort': 4, 'stage': 2, 'seed': SEED, 'lengths': list(LENGTHS), 'per_length': PER_LENGTH, 'n': N,
        'source_dataset': DATASET, 'source_split': 'test', 'dataset_revision': REVISION,
        'parquet_revision': manifests[0]['parquet_revision'], 'catalog_source': str(catalog_path.relative_to(ROOT)),
        'catalog_sha256': digest(raw), 'preparer_sha256': digest(Path(__file__).read_bytes()),
        'consumed_cohorts': [{'cohort': i + 1, 'n': m['n'],
                              'manifest': str((folder / 'manifest.json').relative_to(ROOT)),
                              'manifest_sha256': digest((folder / 'manifest.json').read_bytes()),
                              'inputs_sha256': m['inputs_sha256']} for i, (folder, m) in enumerate(zip(PRIOR_DIRS, manifests))],
        'excluded_context_count': 48,
        'prior_exclusion_contexts': [{'dataset': dataset, 'context_window_id': identity} for dataset, identity in sorted(contexts)],
        'prior_exclusion_context_hashes': sorted(hashes), 'availability': availability,
        'metadata_rows': len(catalog), 'selected_metadata': chosen,
        'selection_rule': 'Exclude all cohort1/2/3 context identities, use unchanged seeded metadata-only task-group/dataset round-robin, one question per distinct context.',
        'selection_uses_question_text': False, 'selection_uses_answers': False,
        'materialization_hash_policy': 'Fail closed on any selected/consumed context-hash overlap; no semantic or outcome-driven replacement.',
        'stage_registration': {
            'motivation': 'A separate parser/plan robustness repair is being considered after consumed-cohort diagnostics. Keep frozen cohort3 running unchanged; do not reuse it as untouched validation.',
            'selection_precedes_new_stage_inference': True, 'cohort3_results_remain_a_separate_stage': True,
            'must_not_merge_stages_or_select_best_run': True,
            'new_stage_requires_a_new_source_freeze_before_inference': True,
        },
        'planned_primary_budget': {'context_tokens': 16384, 'total_tokens': 65536, 'max_calls': 32, 'max_output_tokens': 2048},
        'length_interpretation': 'These are OOLONG nominal source-token buckets. Measure actual Qwen tokens separately; exceeding experimental 16k does not establish exceeding native Qwen capacity.',
    }
    return plan, chosen


def materialize(out: Path, plan: dict, chosen: list[dict], workers=8) -> dict:
    if (out / 'manifest.json').exists():
        raise ValueError('Cohort4 already sealed; do not replace')
    plan_raw = seal_plan(out, plan)
    if json.loads(fetch(f'https://huggingface.co/api/datasets/{DATASET}'))['sha'] != REVISION:
        raise ValueError('Public upstream source revision changed')
    source = out / 'source'; source.mkdir(exist_ok=True)
    cache = source / 'unlabeled_input_cache'; cache.mkdir(exist_ok=True)
    label_path = out / 'labels.json'
    labels = json.loads(label_path.read_text()) if label_path.exists() else {}
    records = {}

    def load(meta):
        identity = f'oolong-synth-test-{meta["id"]}'
        cached = cache / f'{identity}.json'
        if cached.exists() and identity in labels:
            record = json.loads(cached.read_text())
            if record['id'] != identity or record['context_sha256'] != digest(record['context']) or record['question_sha256'] != digest(record['question']):
                raise ValueError('Resumed inference cache identity/hash mismatch')
            return record, labels[identity]
        url = 'https://datasets-server.huggingface.co/rows?' + urlencode({
            'dataset': DATASET, 'config': 'default', 'split': 'test', 'offset': meta['_row_idx'], 'length': 1})
        entry = json.loads(fetch(url, timeout=90, retries=3))['rows'][0]
        if entry.get('truncated_cells'):
            raise ValueError('Selected public row was truncated')
        row = {key: value for key, value in entry['row'].items() if key != 'context_window_text_with_labels'}
        if any(str(row[key]) != str(meta[key]) for key in META_KEYS if key != '_row_idx'):
            raise ValueError('Selected metadata/viewer identity differs')
        return inference_record(row, split='test'), label_record(row, split='test')

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for future in as_completed([pool.submit(load, row) for row in chosen]):
            record, label = future.result()
            records[record['id']] = record; labels[record['id']] = label
            atomic_bytes(cache / f'{record["id"]}.json', json_bytes(record) + b'\n')
            atomic_bytes(label_path, json_bytes(labels) + b'\n', private=True)
            print(json.dumps({'materialized_id': record['id'], 'context_chars': record['context_chars']}), flush=True)
    inputs = [records[f'oolong-synth-test-{row["id"]}'] for row in chosen]
    identities = {(r['metadata']['dataset'], int(r['metadata']['context_window_id'])) for r in inputs}
    hashes = {r['context_sha256'] for r in inputs}
    if len(identities) != N or len(hashes) != N or hashes & set(plan['prior_exclusion_context_hashes']):
        raise ValueError('Repeated source/hash; refuse content-based replacement')
    forbidden = {'answer', 'answers', 'gold', 'labels', 'label', 'context_window_text_with_labels'}
    if any(forbidden & set(row) for row in inputs) or set(labels) != {row['id'] for row in inputs}:
        raise ValueError('Input/label separation or identity mismatch')
    if json.loads(fetch(f'https://huggingface.co/api/datasets/{DATASET}'))['sha'] != REVISION:
        raise ValueError('Public upstream source changed during preparation')
    input_raw = b''.join(json_bytes(row) + b'\n' for row in inputs)
    labels_raw = json_bytes(labels) + b'\n'
    atomic_bytes(out / 'inputs.jsonl', input_raw)
    atomic_bytes(label_path, labels_raw, private=True)
    for name in ('eval_helpers_original.py', 'OOLONG_LICENSE'):
        shutil.copyfile(PRIOR_DIRS[0] / 'source' / name, source / name)
    first = json.loads((PRIOR_DIRS[0] / 'manifest.json').read_text())
    if digest((source / 'eval_helpers_original.py').read_bytes()) != first['scorer_sha256']:
        raise ValueError('Copied official scorer differs')
    manifest = {
        'benchmark': 'OOLONG-synth fresh cohort4, separate second-stage local pilot', 'cohort': 4, 'stage': 2,
        'official_leaderboard_submission': False, 'dataset': DATASET, 'source_split': 'test',
        'dataset_revision': REVISION, 'parquet_revision': plan['parquet_revision'],
        'seed': SEED, 'lengths': list(LENGTHS), 'per_length': PER_LENGTH, 'n': N,
        'selection_plan_sha256': digest(plan_raw), 'selection': plan['selection_rule'],
        'selection_uses_answers': False, 'selection_uses_question_text': False,
        'inputs_sha256': digest(input_raw), 'labels_sha256': digest(labels_raw),
        'scorer_sha256': first['scorer_sha256'], 'scorer_commit': SCORER_COMMIT, 'scorer_source_url': SCORER_URL,
        'metrics': first['metrics'], 'input_field': 'context_window_text', 'forbidden_field': 'context_window_text_with_labels',
        'selected': [{key: row[key] for key in ('id', 'task_type', 'metadata', 'context_sha256', 'question_sha256', 'context_chars')} for row in inputs],
        'exclusions': {'consumed_cohorts': [1, 2, 3], 'consumed_context_count': 48,
                       'context_ids': plan['prior_exclusion_contexts'], 'context_hashes': plan['prior_exclusion_context_hashes'],
                       'context_id_overlap': 0, 'context_hash_overlap': 0},
        'availability': plan['availability'], 'stage_registration': plan['stage_registration'],
        'label_access': 'Preparation/scoring only; labels.json 0600, inference must be OS-denied label access.',
        'length_interpretation': plan['length_interpretation'], 'planned_primary_budget': plan['planned_primary_budget'],
        'limitations': ['8-context pilot, not an official leaderboard result.',
                        'Public data may overlap model pretraining; distinct context IDs/hashes do not prove disjoint underlying records.',
                        'Report this follow-up separately from unchanged cohort3; never merge stages or pick the better run.',
                        'Once scored this cohort is consumed, not untouched validation for further tuning.'],
    }
    atomic_bytes(out / 'manifest.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--materialize', action='store_true', help='Explicit later preparation phase; default is metadata-only plan sealing')
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plan, chosen = selection_plan(); raw = seal_plan(args.out, plan)
    result = {'phase': 'metadata_only_plan', 'n': N, 'seed': SEED,
              'selection_plan_sha256': digest(raw), 'availability': plan['availability'],
              'length_counts': dict(Counter(row['context_len'] for row in chosen))}
    if args.materialize:
        manifest = materialize(args.out, plan, chosen, args.workers)
        result.update(phase='materialized', inputs_sha256=manifest['inputs_sha256'], labels_sha256=manifest['labels_sha256'])
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
