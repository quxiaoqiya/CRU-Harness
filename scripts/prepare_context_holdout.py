#!/usr/bin/env python3
"""Prepare an untouched OOLONG test slice without exposing labels to inference.

Only public reads are performed. Selected inputs, sealed labels, provenance,
and official scorer source are stored on the local Mac. No model is called.
"""
from __future__ import annotations

import argparse
import ast
import concurrent.futures
from datetime import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import sys
import time
from typing import Any
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

DATASET = 'oolongbench/oolong-synth'
REVISION = '81d2f8d9349f6ff329d695f9be494051e9f05cb9'
SCORER_COMMIT = '0bb7eabe839218fee7fe8d007f41cfc2fd3ae24c'
SCORER_URL = f'https://raw.githubusercontent.com/abertsch72/oolong/{SCORER_COMMIT}/src/eval/eval_helpers.py'
DEFAULT_OUTPUT = Path(__file__).resolve().parents[1] / 'data/context_holdout_20261101'
META_COLUMNS = ['id', 'context_len', 'dataset', 'task_group', 'task', 'answer_type', 'input_subset', 'num_labels', 'context_window_id']
INPUT_COLUMNS = ['id', 'context_len', 'dataset', 'context_window_text', 'question', 'task_group', 'task', 'answer_type', 'context_window_id']


def digest(value: bytes | str) -> str:
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def json_bytes(value: Any) -> bytes:
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'))
    # Keep JSONL compatible with readers using str.splitlines(), since a few
    # public source texts contain Unicode line-separator characters.
    return text.replace('\u0085', r'\u0085').replace('\u2028', r'\u2028').replace('\u2029', r'\u2029').encode()


def fetch(url: str, *, headers: dict[str, str] | None = None, timeout: float = 30, retries: int = 3, max_bytes: int = 32 * 1024 * 1024) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(url, headers={'User-Agent': 'local-context-holdout-preparation/1', **(headers or {})})
            with urlopen(request, timeout=timeout) as response:
                advertised = int(response.headers.get('Content-Length') or 0)
                if advertised > max_bytes:
                    raise ValueError(f'response exceeds local read bound: {advertised} > {max_bytes}')
                data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise ValueError('response exceeds local read bound')
                return data
        except Exception as exc:
            last = exc
            if attempt + 1 < retries:
                time.sleep(min(4, 2 ** attempt))
    raise RuntimeError(f'public read failed: {url}: {last}') from last


class HTTPRangeFile(io.RawIOBase):
    """Minimal seekable read-only HTTP file used by pyarrow Parquet readers."""
    def __init__(self, url: str, size: int):
        self.url = url
        self.size = size
        self.position = 0
        self.cache: dict[tuple[int, int], bytes] = {}
        self.bytes_read = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = 0) -> int:
        if whence == 0:
            value = offset
        elif whence == 1:
            value = self.position + offset
        elif whence == 2:
            value = self.size + offset
        else:
            raise ValueError('invalid seek')
        if value < 0:
            raise ValueError('negative seek')
        self.position = value
        return value

    def read(self, length: int = -1) -> bytes:
        if length < 0:
            length = self.size - self.position
        length = max(0, min(length, self.size - self.position))
        if not length:
            return b''
        if length > 128 * 1024 * 1024:
            raise ValueError('refusing a >128MiB single range read')
        start_position = self.position
        block_size = 65536
        first = (start_position // block_size) * block_size
        stop = start_position + length
        pieces = []
        for block_start in range(first, stop, block_size):
            block_length = min(block_size, self.size - block_start)
            key = (block_start, block_length)
            if key not in self.cache:
                raw = fetch(self.url, headers={'Range': f'bytes={block_start}-{block_start + block_length - 1}'}, timeout=30, max_bytes=max(block_length, self.size if self.size <= 2 * 1024 * 1024 else block_length))
                if len(raw) == self.size and len(raw) != block_length:
                    raw = raw[block_start:block_start + block_length]
                if len(raw) != block_length:
                    raise IOError(f'bad range length {len(raw)} != {block_length}')
                self.cache[key] = raw
                self.bytes_read += len(raw)
            block = self.cache[key]
            left = max(start_position, block_start) - block_start
            right = min(stop, block_start + block_length) - block_start
            pieces.append(block[left:right])
        raw = b''.join(pieces)
        self.position += len(raw)
        return raw



def longbench_audit(root: Path) -> dict[str, Any]:
    index_path = root / 'docs/longbench_v2_index.json'
    index = json.loads(index_path.read_text())
    coverage = {}
    for name in ['direct.jsonl', 'harness_262144.jsonl']:
        path = root / 'docs/longbench_v2_full' / name
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        ids = sorted({int(row['index']) for row in rows})
        coverage[name] = {'records': len(rows), 'unique_indices': len(ids), 'indices': ids, 'sha256': digest(path.read_bytes())}
    return {'dataset': 'zai-org/LongBench-v2', 'local_index_rows': len(index), 'coverage': coverage,
            'all_local_rows_previously_evaluated': all(v['unique_indices'] == len(index) for v in coverage.values()),
            'conclusion': 'No local LongBench-v2 row is an untouched holdout; use a separate benchmark.'}


def choose_holdout(metadata: list[dict[str, Any]], lengths=(65536, 131072, 524288), per_length=6, seed=20261101) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    seen_contexts: set[tuple[str, int]] = set()
    for length in lengths:
        candidates = [row for row in metadata if int(row['context_len']) == length]
        rng.shuffle(candidates)
        # Round-robin task group and dataset to avoid an all-counting or
        # one-domain convenience slice. No question text or answer is used.
        candidates.sort(key=lambda row: (str(row['task_group']), str(row['dataset'])))
        queues: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in candidates:
            queues.setdefault((str(row['task_group']), str(row['dataset'])), []).append(row)
        keys = list(queues)
        rng.shuffle(keys)
        chosen: list[dict[str, Any]] = []
        while len(chosen) < per_length and keys:
            next_keys = []
            for key in keys:
                queue = queues[key]
                while queue:
                    row = queue.pop()
                    context = (str(row['dataset']), int(row['context_window_id']))
                    if context not in seen_contexts:
                        chosen.append(row)
                        seen_contexts.add(context)
                        break
                if queue:
                    next_keys.append(key)
                if len(chosen) >= per_length:
                    break
            keys = next_keys
        if len(chosen) != per_length:
            raise ValueError(f'need {per_length} independent contexts at {length}, got {len(chosen)}')
        selected.extend(chosen)
    return selected


def inference_record(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    context = str(row['context_window_text'])
    question = str(row['question'])
    return {
        'id': f'oolong-synth-{split}-{row["id"]}',
        'context': context,
        'question': question,
        'task_type': f'oolong/{row["dataset"]}/{row["task_group"]}/{row["task"]}',
        'metadata': {key: row[key] for key in META_COLUMNS if key in row},
        'source_split': split,
        'context_sha256': digest(context),
        'question_sha256': digest(question),
        'context_chars': len(context),
    }


def label_record(row: dict[str, Any], *, split: str) -> dict[str, Any]:
    return {'id': f'oolong-synth-{split}-{row["id"]}', 'answer': row['answer'], 'answer_type': row['answer_type'],
            'context_window_id': row['context_window_id'], 'dataset': row['dataset']}


def synth_attempt_answer_parse(answer: str) -> tuple[str, str]:
    """Exact lightweight copy of the upstream synth answer parser."""
    confidence = 'low'
    if ':' not in answer:
        return (answer if len(answer) < 20 else answer.split()[-1] if answer.split() else ''), confidence
    candidate = answer.split(':')[-1].strip().replace('*', '').replace('[', '').replace(']', '')
    confidence = 'med'
    if any(marker in answer for marker in ['User:', 'Answer:', 'Date:', 'Label']):
        confidence = 'high'
    if len(candidate) < 20:
        confidence = 'vhigh'
    elif 'more common' in candidate:
        candidate = 'more common'
    elif 'less common' in candidate:
        candidate = 'less common'
    elif 'same frequency' in candidate:
        candidate = 'same frequency'
    return candidate, confidence


def score_prediction(prediction: str, label: dict[str, Any]) -> dict[str, Any]:
    """Upstream OOLONG synth score plus separately named strict exact match.

    Numeric error receives 0.75**abs(error), not binary accuracy. Parsing and
    comparison-phrase/date behavior follow the pinned upstream scorer.
    """
    answer = str(label['answer'])
    gold = ast.literal_eval(answer)[0] if 'datetime' not in answer else datetime.strptime(answer, '[datetime.date(%Y, %m, %d)]')
    parsed, confidence = synth_attempt_answer_parse(str(prediction or ''))
    exact = str(parsed) == str(gold)
    score: float = 0.0
    if exact:
        score = 1.0
    elif parsed in ['more common', 'less common', 'same frequency']:
        score = float(parsed in str(gold))
    elif label['answer_type'] == 'ANSWER_TYPE.NUMERIC':
        try:
            score = 0.75 ** abs(int(gold) - int(parsed))
        except (ValueError, TypeError, OverflowError):
            confidence = 'low'
    elif label['answer_type'] == 'ANSWER_TYPE.DATE':
        try:
            from dateutil.parser import parse
            score = float(parse(parsed) == gold)
        except (ValueError, TypeError, OverflowError):
            confidence = 'low'
    return {'official_score': float(score), 'exact_match': exact, 'full_credit': bool(score == 1.0), 'parsed_answer': str(parsed), 'parse_confidence': confidence}


def _parquet_runtime(out: Path):
    try:
        import pyarrow.parquet as pq
    except ImportError:
        sys.path.insert(0, str(out / 'source/pyarrow_runtime'))
        import pyarrow.parquet as pq
    return pq


def prepare(out: Path, *, seed: int, per_length: int, workers: int) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    source = out / 'source'
    source.mkdir(exist_ok=True)
    if (out / 'manifest.json').exists():
        raise RuntimeError('holdout already sealed; refusing to silently replace it')
    pq = _parquet_runtime(out)
    original_revision = json.loads(fetch(f'https://huggingface.co/api/datasets/{DATASET}'))['sha']
    if original_revision != REVISION:
        raise RuntimeError('official dataset revision changed; review and pin a new revision explicitly')
    converted_revision = json.loads(fetch(f'https://huggingface.co/api/datasets/{DATASET}/revision/refs%2Fconvert%2Fparquet'))['sha']
    files = json.loads(fetch(f'https://datasets-server.huggingface.co/parquet?dataset={quote(DATASET, safe="")}'))['parquet_files']
    test_files = [item for item in files if item['split'] == 'test']
    def catalog_one(item):
        cache_path = source / ('catalog_' + item['filename'] + '.json')
        if cache_path.exists():
            cached = json.loads(cache_path.read_text())
            if cached and converted_revision in cached[0].get('_url', ''):
                return cached
        url = item['url'].replace('refs%2Fconvert%2Fparquet', converted_revision)
        stream = HTTPRangeFile(url, int(item['size']))
        parquet = pq.ParquetFile(stream)
        rows = []
        for group in range(parquet.num_row_groups):
            table = parquet.read_row_group(group, columns=META_COLUMNS)
            for local, row in enumerate(table.to_pylist()):
                rows.append({**row, '_file': item['filename'], '_row_group': group, '_local_row': local, '_url': url, '_file_size': item['size']})
        cache_path.write_text(json.dumps(rows, ensure_ascii=False))
        print(f'catalogued {item["filename"]}: {len(rows)} metadata rows', flush=True)
        return rows
    metadata = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for rows in pool.map(catalog_one, test_files):
            metadata.extend(rows)
    for row_index, row in enumerate(metadata):
        row['_row_idx'] = row_index
    (source / 'metadata_catalog.json').write_text(json.dumps(metadata, ensure_ascii=False, indent=2))
    chosen = choose_holdout(metadata, per_length=per_length, seed=seed)
    def read_one(meta):
        # Read the selected row through the viewer to avoid decompressing a
        # complete large row group. Fields with gold classifications are
        # immediately dropped and never written to inference files.
        url = 'https://datasets-server.huggingface.co/rows?' + urlencode({'dataset': DATASET, 'config': 'default', 'split': 'test', 'offset': meta['_row_idx'], 'length': 1})
        payload = json.loads(fetch(url, timeout=90, retries=2))
        entry = payload['rows'][0]
        if entry.get('truncated_cells'):
            raise RuntimeError('dataset viewer truncated a selected row')
        row = {key: value for key, value in entry['row'].items() if key != 'context_window_text_with_labels'}
        if int(row['id']) != int(meta['id']):
            raise RuntimeError('metadata/input identity mismatch')
        print(f'materialized id={meta["id"]} length={meta["context_len"]}', flush=True)
        return row
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        rows = list(pool.map(read_one, chosen))
    inputs = [inference_record(row, split='test') for row in rows]
    labels = {record['id']: record for record in [label_record(row, split='test') for row in rows]}
    inputs_bytes = b''.join(json_bytes(record) + b'\n' for record in inputs)
    labels_bytes = json_bytes(labels) + b'\n'
    (out / 'inputs.jsonl').write_bytes(inputs_bytes)
    (out / 'labels.json').write_bytes(labels_bytes)
    (out / 'labels.json').chmod(0o600)
    (source / 'eval_helpers_original.py').write_bytes(fetch(SCORER_URL))
    audit = longbench_audit(Path(__file__).resolve().parents[1])
    (out / 'prior_exposure_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    manifest = {
        'benchmark': 'OOLONG-synth independent local test slice', 'official_leaderboard_submission': False,
        'source_url': f'https://huggingface.co/datasets/{DATASET}', 'dataset': DATASET, 'dataset_revision': REVISION,
        'preparation_dependencies': {'python': '3.12', 'pyarrow': '21.0.0'},
        'parquet_revision': converted_revision, 'source_split': 'test', 'seed': seed, 'lengths': [65536, 131072, 524288],
        'per_length': per_length, 'n': len(inputs), 'selection': 'metadata-only seeded task-group/dataset round-robin; one question per unique dataset/context_window_id globally',
        'selection_uses_answers': False, 'selection_uses_question_text': False,
        'input_field': 'context_window_text', 'forbidden_field': 'context_window_text_with_labels',
        'inputs_sha256': digest(inputs_bytes), 'labels_sha256': digest(labels_bytes),
        'label_access': 'Data preparation/scoring only. Inputs contain no gold answer or per-record gold classification labels. Do not pass labels.json to method processes.',
        'scorer_source_url': SCORER_URL, 'scorer_commit': SCORER_COMMIT,
        'scorer_sha256': digest((source / 'eval_helpers_original.py').read_bytes()),
        'metrics': {'official_score': 'Mean upstream OOLONG score; numeric partial credit is 0.75**abs(error)', 'exact_match': 'Strict equality after the upstream answer parser, separate from official score', 'full_credit': 'Fraction with official_score==1'},
        'selected': [{key: record.get(key) for key in ['id', 'task_type', 'metadata', 'context_sha256', 'question_sha256', 'context_chars']} for record in inputs],
        'limitations': ['Small pilot, not an official leaderboard result.', 'Public test set; absence from this project tuning does not establish absence from model pretraining.', 'Nominal source lengths use the dataset tokenizer; actual Qwen lengths must be measured separately.', 'OOD benchmark measures aggregation and semantic classification; not a new LongBench score.', 'Preparation code handles labels only for separation and scoring; freeze implementation before first test run and do not tune on test outcomes.'],
    }
    (out / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    return manifest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--seed', type=int, default=20261101)
    parser.add_argument('--per-length', type=int, default=6)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(longbench_audit(Path(__file__).resolve().parents[1]), ensure_ascii=False, indent=2))
    else:
        manifest = prepare(args.out, seed=args.seed, per_length=args.per_length, workers=args.workers)
        print(json.dumps({key: manifest[key] for key in ['benchmark', 'n', 'lengths', 'inputs_sha256', 'labels_sha256']}, indent=2))


if __name__ == '__main__':
    main()
