#!/usr/bin/env python3
"""Seal an untouched OOLONG cohort for the fact-ledger Harness comparison.

Selection uses the existing metadata catalogue only. Both earlier cohorts
are consumed and excluded. Gold is handled by this preparation process only,
and persisted solely in labels.json, never in cached inference inputs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
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

PRIOR_DIRS = (ROOT / 'data/context_holdout_20261101', ROOT / 'data/context_holdout_20261101_round2')
DEFAULT_OUT = ROOT / 'data/context_holdout_20261101_round3'
LENGTHS = (65536, 131072, 524288)
SEED = 20261103
PER_LENGTH = 4
N = len(LENGTHS) * PER_LENGTH
META_KEYS = ('id', 'context_len', 'dataset', 'task_group', 'task', 'answer_type', 'context_window_id', '_row_idx')


def atomic_bytes(path: Path, value: bytes, *, private: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    if private:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(value)
    else:
        temporary.write_bytes(value)
    temporary.replace(path)
    if private:
        path.chmod(0o600)


def prior_exclusions(manifests: list[dict]) -> tuple[set[tuple[str, int]], set[str]]:
    contexts: set[tuple[str, int]] = set()
    hashes: set[str] = set()
    expected = 0
    for manifest in manifests:
        if manifest['dataset_revision'] != REVISION:
            raise ValueError('Earlier cohort dataset revision differs from pinned source')
        expected += len(manifest['selected'])
        for row in manifest['selected']:
            contexts.add((str(row['metadata']['dataset']), int(row['metadata']['context_window_id'])))
            hashes.add(str(row['context_sha256']))
    if len(contexts) != expected or len(hashes) != expected:
        raise ValueError('Consumed cohort identity/hash sets are not disjoint')
    return contexts, hashes


def select_from_catalog(catalog: list[dict], manifests: list[dict]) -> tuple[list[dict], set[tuple[str, int]], set[str]]:
    contexts, hashes = prior_exclusions(manifests)
    available = [row for row in catalog if (str(row['dataset']), int(row['context_window_id'])) not in contexts]
    chosen = choose_holdout(available, lengths=LENGTHS, per_length=PER_LENGTH, seed=SEED)
    identities = {(str(row['dataset']), int(row['context_window_id'])) for row in chosen}
    if len(chosen) != N or len(identities) != N or identities & contexts:
        raise ValueError('Fresh metadata selection failed uniqueness/exclusion checks')
    return chosen, contexts, hashes


def selection_plan() -> tuple[dict, list[dict]]:
    manifests = [json.loads((directory / 'manifest.json').read_text()) for directory in PRIOR_DIRS]
    if any(int(m['n']) != 18 for m in manifests):
        raise ValueError('Expected the two consumed 18-context cohorts')
    if len({m['parquet_revision'] for m in manifests}) != 1:
        raise ValueError('Earlier parquet revisions differ')
    catalog_path = PRIOR_DIRS[0] / 'source/metadata_catalog.json'
    catalog_raw = catalog_path.read_bytes()
    catalog = json.loads(catalog_raw)
    chosen, contexts, hashes = select_from_catalog(catalog, manifests)
    if len(contexts) != 36 or len(hashes) != 36:
        raise ValueError('Expected exactly 36 distinct consumed source contexts')
    availability = []
    for length in LENGTHS:
        source_ids = {(str(r['dataset']), int(r['context_window_id']))
                      for r in catalog if int(r['context_len']) == length}
        availability.append({'context_len': length, 'distinct_contexts': len(source_ids),
                             'consumed_contexts': len(source_ids & contexts),
                             'remaining_contexts': len(source_ids - contexts)})
    plan = {
        'cohort': 3, 'seed': SEED, 'lengths': list(LENGTHS), 'per_length': PER_LENGTH, 'n': N,
        'source_dataset': DATASET, 'source_split': 'test', 'dataset_revision': REVISION,
        'parquet_revision': manifests[0]['parquet_revision'],
        'catalog_source': str(catalog_path.relative_to(ROOT)), 'catalog_sha256': digest(catalog_raw),
        'preparer_sha256': digest(Path(__file__).read_bytes()),
        'consumed_cohorts': [{'cohort': i + 1, 'manifest': str((d / 'manifest.json').relative_to(ROOT)),
                              'manifest_sha256': digest((d / 'manifest.json').read_bytes()),
                              'inputs_sha256': m['inputs_sha256'], 'n': m['n']}
                             for i, (d, m) in enumerate(zip(PRIOR_DIRS, manifests))],
        'prior_cohorts_consumed': True,
        'prior_exclusion_contexts': [{'dataset': name, 'context_window_id': identity} for name, identity in sorted(contexts)],
        'prior_exclusion_context_hashes': sorted(hashes),
        'metadata_rows_before_exclusion': len(catalog),
        'metadata_rows_after_exclusion': sum((str(r['dataset']), int(r['context_window_id'])) not in contexts for r in catalog),
        'sample_count_adjustment': {'original_target_per_length': 6, 'actual_per_length': PER_LENGTH,
                                    'original_target_n': 18, 'actual_n': N,
                                    'reason': 'Metadata scarcity only: each requested length has 16 distinct contexts, 12 already consumed and 4 left. Preserve the original length gradient and strict exclusions rather than reusing contexts or selecting on question/answer content.',
                                    'availability': availability},
        'selection_rule': 'Exclude every consumed cohort1/cohort2 (dataset,context_window_id), then use the unchanged seeded metadata-only task-group/dataset round-robin; one question per distinct context globally.',
        'selection_uses_question_text': False, 'selection_uses_answers': False,
        'hash_identity_policy': 'Fail closed on any repeated context hash after materialization; never silently reselect using content or outcomes.',
        'selected_metadata': [{key: row[key] for key in META_KEYS} for row in chosen],
        'reason_for_new_cohort': 'Cohorts1/2 are already scored and consumed. A structured fact ledger, convergence control and deterministic aggregation are being developed from consumed/dev material; this cohort is reserved for the frozen new comparison.',
        'planned_primary_budget': {'context_tokens': 16384, 'total_tokens': 65536, 'max_calls': 32, 'max_output_tokens': 2048},
    }
    return plan, chosen


def validate_materialized(inputs: list[dict], excluded_hashes: set[str]) -> None:
    if len(inputs) != N or len({row['id'] for row in inputs}) != N:
        raise ValueError(f'Expected {N} unique fresh question identities')
    hashes = [row['context_sha256'] for row in inputs]
    if len(set(hashes)) != N or set(hashes) & excluded_hashes:
        raise ValueError('A source hash repeats another selected or consumed context; refusing reselection')
    forbidden = {'answer', 'answers', 'gold', 'label', 'labels', 'context_window_text_with_labels'}
    if any(forbidden & set(row) for row in inputs):
        raise ValueError('Inference record contains a forbidden gold field')


def seal_plan(out: Path, plan: dict) -> bytes:
    raw = json_bytes(plan) + b'\n'
    path = out / 'selection_plan.json'
    if path.exists() and path.read_bytes() != raw:
        raise RuntimeError('Existing selection plan differs; refusing to silently replace it')
    if not path.exists():
        atomic_bytes(path, raw)
    return raw


def prepare(out: Path, workers: int = 8) -> dict:
    if (out / 'manifest.json').exists():
        raise RuntimeError('cohort3 is sealed; refusing to replace it')
    out.mkdir(parents=True, exist_ok=True)
    source = out / 'source'
    source.mkdir(exist_ok=True)
    plan, chosen = selection_plan()
    plan_raw = seal_plan(out, plan)
    print(json.dumps({'selection_plan_sha256': digest(plan_raw), 'n': N,
                      'nominal_lengths': dict(Counter(row['context_len'] for row in chosen)),
                      'task_groups': dict(Counter(row['task_group'] for row in chosen))}), flush=True)
    observed = json.loads(fetch(f'https://huggingface.co/api/datasets/{DATASET}'))['sha']
    if observed != REVISION:
        raise RuntimeError('Upstream dataset revision changed')
    label_path = out / 'labels.json'
    labels = json.loads(label_path.read_text()) if label_path.exists() else {}
    input_cache = source / 'unlabeled_input_cache'
    input_cache.mkdir(exist_ok=True)

    def materialize(meta: dict) -> tuple[dict, dict]:
        item_id = f'oolong-synth-test-{meta["id"]}'
        cached = input_cache / f'{item_id}.json'
        if cached.exists() and item_id in labels:
            record = json.loads(cached.read_text())
            if (record['id'] != item_id or record['metadata']['context_len'] != meta['context_len']
                    or record['context_sha256'] != digest(record['context'])
                    or record['question_sha256'] != digest(record['question'])):
                raise RuntimeError('Cached inference identity/hash mismatch')
            return record, labels[item_id]
        url = 'https://datasets-server.huggingface.co/rows?' + urlencode({
            'dataset': DATASET, 'config': 'default', 'split': 'test', 'offset': meta['_row_idx'], 'length': 1})
        payload = json.loads(fetch(url, timeout=90, retries=3))
        entry = payload['rows'][0]
        if entry.get('truncated_cells'):
            raise RuntimeError('Viewer truncated a selected example')
        row = {key: value for key, value in entry['row'].items() if key != 'context_window_text_with_labels'}
        if int(row['id']) != int(meta['id']) or int(row['context_len']) != int(meta['context_len']):
            raise RuntimeError('Metadata/viewer identity mismatch')
        record, gold = inference_record(row, split='test'), label_record(row, split='test')
        print(f'materialized cohort3 id={meta["id"]} nominal_length={meta["context_len"]}', flush=True)
        return record, gold

    records: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(materialize, meta) for meta in chosen]
        for future in as_completed(futures):
            record, gold = future.result()
            records[record['id']] = record
            labels[record['id']] = gold
            atomic_bytes(input_cache / f'{record["id"]}.json', json_bytes(record) + b'\n')
            # labels.json is the only persisted file containing gold. It is
            # also the file denied by the later OS inference guard.
            atomic_bytes(label_path, json_bytes(labels) + b'\n', private=True)
    inputs = [records[f'oolong-synth-test-{meta["id"]}'] for meta in chosen]
    validate_materialized(inputs, set(plan['prior_exclusion_context_hashes']))
    expected_ids = {row['id'] for row in inputs}
    if set(labels) != expected_ids:
        raise ValueError('Label identities do not equal the sealed inference identities')
    if json.loads(fetch(f'https://huggingface.co/api/datasets/{DATASET}'))['sha'] != REVISION:
        raise RuntimeError('Upstream revision changed during preparation')
    inputs_raw = b''.join(json_bytes(row) + b'\n' for row in inputs)
    labels_raw = json_bytes(labels) + b'\n'
    atomic_bytes(out / 'inputs.jsonl', inputs_raw)
    atomic_bytes(label_path, labels_raw, private=True)
    for name in ('eval_helpers_original.py', 'OOLONG_LICENSE'):
        shutil.copyfile(PRIOR_DIRS[0] / 'source' / name, source / name)
    prior_manifest = json.loads((PRIOR_DIRS[0] / 'manifest.json').read_text())
    scorer_hash = digest((source / 'eval_helpers_original.py').read_bytes())
    if scorer_hash != prior_manifest['scorer_sha256']:
        raise ValueError('Official scorer copy changed')
    manifest = {
        'benchmark': 'OOLONG-synth independent local test slice, fresh cohort3', 'cohort': 3,
        'official_leaderboard_submission': False, 'source_url': f'https://huggingface.co/datasets/{DATASET}',
        'dataset': DATASET, 'dataset_revision': REVISION, 'parquet_revision': plan['parquet_revision'],
        'source_split': 'test', 'seed': SEED, 'lengths': list(LENGTHS), 'per_length': PER_LENGTH, 'n': N,
        'selection': plan['selection_rule'], 'selection_plan_sha256': digest(plan_raw),
        'sample_count_adjustment': plan['sample_count_adjustment'],
        'selection_uses_answers': False, 'selection_uses_question_text': False,
        'input_field': 'context_window_text', 'forbidden_field': 'context_window_text_with_labels',
        'inputs_sha256': digest(inputs_raw), 'labels_sha256': digest(labels_raw),
        'label_access': 'Preparation/scoring only. Method processes must be OS-denied access to labels.json. Developers must not inspect fresh questions or labels before frozen inference.',
        'scorer_source_url': SCORER_URL, 'scorer_commit': SCORER_COMMIT, 'scorer_sha256': scorer_hash,
        'metrics': prior_manifest['metrics'],
        'selected': [{key: row.get(key) for key in ('id', 'task_type', 'metadata', 'context_sha256', 'question_sha256', 'context_chars')} for row in inputs],
        'exclusions': {'consumed_cohorts': [1, 2], 'consumed_context_count': 36,
                       'context_ids': plan['prior_exclusion_contexts'], 'context_hashes': plan['prior_exclusion_context_hashes'],
                       'context_id_overlap': 0, 'context_hash_overlap': 0},
        'reason_for_new_cohort': plan['reason_for_new_cohort'], 'planned_primary_budget': plan['planned_primary_budget'],
        'preparation_dependencies': {'metadata': 'Existing pinned 5200-row catalogue; no new Parquet download.'},
        'limitations': ['12-context pilot, not an official leaderboard score.',
                        'Unused by this project tuning does not exclude model pretraining contamination.',
                        'Distinct context IDs/hashes do not prove disjoint underlying source records.',
                        'Nominal dataset lengths and measured Qwen lengths must be reported separately.',
                        'After scoring this cohort is consumed and cannot be reused as untouched validation after further tuning.'],
    }
    atomic_bytes(out / 'manifest.json', (json.dumps(manifest, ensure_ascii=False, indent=2) + '\n').encode())
    verification = {
        'cohort': 3, 'n': N, 'unique_context_ids': len({(r['metadata']['dataset'], r['metadata']['context_window_id']) for r in inputs}),
        'unique_context_hashes': len({r['context_sha256'] for r in inputs}), 'excluded_prior_contexts': 36,
        'context_id_overlap': 0, 'context_hash_overlap': 0,
        'inputs_sha256': digest(inputs_raw), 'labels_sha256': digest(labels_raw),
        'manifest_sha256': digest((out / 'manifest.json').read_bytes()), 'selection_plan_sha256': digest(plan_raw),
        'label_mode': oct(label_path.stat().st_mode & 0o777), 'inference_inputs_contain_gold_fields': False,
        'prior_manifests_unchanged': all(digest((ROOT / x['manifest']).read_bytes()) == x['manifest_sha256'] for x in plan['consumed_cohorts']),
    }
    atomic_bytes(out / 'preparation_verification.json', (json.dumps(verification, ensure_ascii=False, indent=2) + '\n').encode())
    (out / 'README.md').write_text(
        '# Fresh OOLONG cohort3\n\n'
        '12 contexts selected using metadata only with seed 20261103; prior 36 contexts excluded. '
        'The preparation process handles gold only to separate it into labels.json. '
        'Only inputs.jsonl may enter method processes; do not inspect fresh questions or labels during development.\n\n'
        'Source revisions, selection/exclusion hashes and scoring rules are in manifest.json. '
        'Nominal source lengths are not measured Qwen token counts. '
        'The official numeric score includes partial credit and is not binary accuracy. '
        'This is a local subset evaluation, not an official leaderboard submission.\n')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=DEFAULT_OUT)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if args.plan_only:
        if (args.out / 'manifest.json').exists():
            raise RuntimeError('cohort3 already sealed')
        plan, _ = selection_plan()
        raw = seal_plan(args.out, plan)
        print(json.dumps({'n': N, 'seed': SEED, 'selection_plan_sha256': digest(raw),
                          'nominal_lengths': plan['lengths']}, indent=2))
        return
    manifest = prepare(args.out, args.workers)
    print(json.dumps({key: manifest[key] for key in ('cohort', 'n', 'seed', 'lengths', 'inputs_sha256', 'labels_sha256', 'selection_plan_sha256')}, indent=2))


if __name__ == '__main__':
    main()
