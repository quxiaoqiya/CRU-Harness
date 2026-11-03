#!/usr/bin/env python3
"""Prepare a fresh OOLONG cohort after baseline-compatibility repair.

Uses only the previously downloaded metadata catalogue for selection. Never
modifies the first cohort, its frozen preparer/scorer, or any remote file.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
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

PRIOR = ROOT / 'data/context_holdout_20261101'
DEFAULT_OUT = ROOT / 'data/context_holdout_20261101_round2'
LENGTHS = (65536, 131072, 524288)
SEED = 20261102
PER_LENGTH = 6


def selection_plan() -> tuple[dict, list[dict]]:
    prior_manifest = json.loads((PRIOR / 'manifest.json').read_text())
    catalog_path = PRIOR / 'source/metadata_catalog.json'
    catalog_bytes = catalog_path.read_bytes()
    catalog = json.loads(catalog_bytes)
    excluded_contexts = {
        (str(row['metadata']['dataset']), int(row['metadata']['context_window_id']))
        for row in prior_manifest['selected']
    }
    excluded_hashes = {row['context_sha256'] for row in prior_manifest['selected']}
    available = [row for row in catalog if (str(row['dataset']), int(row['context_window_id'])) not in excluded_contexts]
    chosen = choose_holdout(available, lengths=LENGTHS, per_length=PER_LENGTH, seed=SEED)
    assert len(chosen) == 18
    assert len({(str(row['dataset']), int(row['context_window_id'])) for row in chosen}) == 18
    chosen_metadata = [{key: row[key] for key in ['id', 'context_len', 'dataset', 'task_group', 'task', 'answer_type', 'context_window_id', '_row_idx']} for row in chosen]
    plan = {
        'cohort': 2,
        'seed': SEED,
        'lengths': list(LENGTHS),
        'per_length': PER_LENGTH,
        'n': len(chosen),
        'source_dataset': DATASET,
        'source_split': 'test',
        'dataset_revision': REVISION,
        'parquet_revision': prior_manifest['parquet_revision'],
        'catalog_source': str(catalog_path.relative_to(ROOT)),
        'catalog_sha256': digest(catalog_bytes),
        'prior_manifest_sha256': digest((PRIOR / 'manifest.json').read_bytes()),
        'prior_cohort_consumed': True,
        'prior_exclusion_contexts': [{'dataset': name, 'context_window_id': identity} for name, identity in sorted(excluded_contexts)],
        'prior_exclusion_context_hashes': sorted(excluded_hashes),
        'metadata_rows_before_exclusion': len(catalog),
        'metadata_rows_after_exclusion': len(available),
        'selection_rule': 'Filter out every cohort1 (dataset,context_window_id), then use unchanged seeded metadata-only task-group/dataset round-robin; one question per unique context globally.',
        'selection_uses_question_text': False,
        'selection_uses_answers': False,
        'hash_identity_policy': 'After materialization, fail closed if a selected context hash repeats cohort1 or another cohort2 record; do not select using answer content or observed outcomes.',
        'selected_metadata': chosen_metadata,
        'reason_for_new_cohort': 'The RLM adapter incorrectly blocked legitimate _strptime/hasattr operations. Baseline compatibility will be repaired while the Harness algorithm remains frozen. Cohort1 was already scored and is consumed; use this fresh cohort for the repaired comparison.',
        'planned_primary_budget': {'max_calls': 32, 'total_tokens': 65536, 'context_tokens': 16384},
    }
    return plan, chosen


def prepare(out: Path, workers: int = 8) -> dict:
    if (out / 'manifest.json').exists():
        raise RuntimeError('cohort2 is already sealed; refusing to replace it')
    out.mkdir(parents=True, exist_ok=True)
    source = out / 'source'
    source.mkdir(exist_ok=True)
    plan, chosen = selection_plan()
    plan_raw = json_bytes(plan) + b'\n'
    (out / 'selection_plan.json').write_bytes(plan_raw)
    print(json.dumps({'selection_plan_sha256': digest(plan_raw), 'n': len(chosen), 'lengths': dict(Counter(row['context_len'] for row in chosen)), 'task_groups': dict(Counter(row['task_group'] for row in chosen))}), flush=True)
    observed_revision = json.loads(fetch(f'https://huggingface.co/api/datasets/{DATASET}'))['sha']
    if observed_revision != REVISION:
        raise RuntimeError('upstream dataset revision changed; refusing an unreviewed dataset update')

    def materialize(meta: dict) -> dict:
        url = 'https://datasets-server.huggingface.co/rows?' + urlencode({'dataset': DATASET, 'config': 'default', 'split': 'test', 'offset': meta['_row_idx'], 'length': 1})
        payload = json.loads(fetch(url, timeout=90, retries=3))
        entry = payload['rows'][0]
        if entry.get('truncated_cells'):
            raise RuntimeError('upstream viewer truncated a selected example')
        # The preparation process may handle gold only to isolate labels.
        # Gold per-record classification contexts are never persisted or used.
        row = {key: value for key, value in entry['row'].items() if key != 'context_window_text_with_labels'}
        if int(row['id']) != int(meta['id']):
            raise RuntimeError('metadata/viewer row identity mismatch')
        if int(row['context_len']) != int(meta['context_len']):
            raise RuntimeError('source nominal length changed')
        print(f'materialized cohort2 id={meta["id"]} nominal_length={meta["context_len"]}', flush=True)
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(materialize, chosen))
    inputs = [inference_record(row, split='test') for row in rows]
    prior_hashes = set(plan['prior_exclusion_context_hashes'])
    context_hashes = [row['context_sha256'] for row in inputs]
    if set(context_hashes) & prior_hashes:
        raise RuntimeError('context hash overlaps consumed cohort1; fail closed without silently reselecting')
    if len(set(context_hashes)) != len(context_hashes):
        raise RuntimeError('duplicate cohort2 context hash; fail closed without silently reselecting')
    if len({row['id'] for row in inputs}) != len(inputs):
        raise RuntimeError('duplicate cohort2 item identity')
    final_revision = json.loads(fetch(f'https://huggingface.co/api/datasets/{DATASET}'))['sha']
    if final_revision != REVISION:
        raise RuntimeError('upstream revision changed during materialization')

    labels = {row['id']: row for row in (label_record(row, split='test') for row in rows)}
    inputs_bytes = b''.join(json_bytes(row) + b'\n' for row in inputs)
    labels_bytes = json_bytes(labels) + b'\n'
    (out / 'inputs.jsonl').write_bytes(inputs_bytes)
    (out / 'labels.json').write_bytes(labels_bytes)
    (out / 'labels.json').chmod(0o600)
    for name in ['eval_helpers_original.py', 'OOLONG_LICENSE']:
        shutil.copyfile(PRIOR / 'source' / name, source / name)
    scorer_bytes = (source / 'eval_helpers_original.py').read_bytes()
    prior_manifest = json.loads((PRIOR / 'manifest.json').read_text())
    if digest(scorer_bytes) != prior_manifest['scorer_sha256']:
        raise RuntimeError('upstream scorer copy changed')

    manifest = {
        'benchmark': 'OOLONG-synth independent local test slice, fresh cohort2',
        'cohort': 2,
        'official_leaderboard_submission': False,
        'source_url': f'https://huggingface.co/datasets/{DATASET}',
        'dataset': DATASET,
        'dataset_revision': REVISION,
        'parquet_revision': plan['parquet_revision'],
        'source_split': 'test',
        'seed': SEED,
        'lengths': list(LENGTHS),
        'per_length': PER_LENGTH,
        'n': len(inputs),
        'selection': plan['selection_rule'],
        'selection_plan_sha256': digest(plan_raw),
        'selection_uses_answers': False,
        'selection_uses_question_text': False,
        'input_field': 'context_window_text',
        'forbidden_field': 'context_window_text_with_labels',
        'inputs_sha256': digest(inputs_bytes),
        'labels_sha256': digest(labels_bytes),
        'label_access': 'Data preparation/scoring only; labels must be denied to all method processes and opened only after predictions are sealed. Root/development must not inspect fresh questions or labels before freeze.',
        'scorer_source_url': SCORER_URL,
        'scorer_commit': SCORER_COMMIT,
        'scorer_sha256': digest(scorer_bytes),
        'metrics': prior_manifest['metrics'],
        'selected': [{key: row.get(key) for key in ['id', 'task_type', 'metadata', 'context_sha256', 'question_sha256', 'context_chars']} for row in inputs],
        'exclusions': {'prior_cohort': 1, 'prior_cohort_consumed': True, 'context_ids': plan['prior_exclusion_contexts'], 'context_hashes': plan['prior_exclusion_context_hashes'], 'context_id_overlap': 0, 'context_hash_overlap': 0},
        'reason_for_new_cohort': plan['reason_for_new_cohort'],
        'planned_primary_budget': plan['planned_primary_budget'],
        'preparation_dependencies': {'python': '3.12', 'metadata': 'Reuses pinned full cohort1 metadata catalog; no PyArrow import or Parquet redownload.'},
        'limitations': ['18-question pilot, not an official leaderboard score or reproduction of paper numbers.', 'Public test data unused for this project tuning; model pretraining contamination cannot be excluded.', 'Distinct context hashes/IDs do not prove disjoint underlying source records.', 'Nominal source lengths and measured Qwen lengths must be reported separately.', 'Cohort1 outcomes may motivate the integration repair but cohort2 content/answers cannot be used to tune methods.', 'After scoring, cohort2 is consumed and cannot be reused as untouched validation after further tuning.'],
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.plan_only:
        plan, _ = selection_plan()
        args.out.mkdir(parents=True, exist_ok=True)
        raw = json_bytes(plan) + b'\n'
        (args.out / 'selection_plan.json').write_bytes(raw)
        print(json.dumps({'n': plan['n'], 'selection_plan_sha256': digest(raw), 'lengths': plan['lengths'], 'seed': plan['seed']}, indent=2))
        return
    manifest = prepare(args.out, workers=args.workers)
    print(json.dumps({key: manifest[key] for key in ['cohort', 'n', 'seed', 'lengths', 'inputs_sha256', 'labels_sha256', 'selection_plan_sha256']}, indent=2))


if __name__ == '__main__':
    main()
