"""Recompute published historical LongBench result arithmetic (no API calls).

This checks the derived results, not the original model requests or upstream
ground truth. Raw historical artifacts are identified by SHA-256 in the JSON.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import re


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def stats(rows: list[dict], method: str) -> dict:
    n = len(rows)
    answered = sum(row[method]['prediction'] is not None for row in rows)
    correct = sum(row[method]['correct'] for row in rows)
    return {
        'n': n, 'answered': answered, 'missing': n - answered, 'correct': correct,
        'accuracy_all': correct / n,
        'accuracy_answered': correct / answered if answered else None,
        'context_limit_rejections': sum(row[method]['error_kind'] == 'context_limit' for row in rows),
    }


def verify(payload: dict) -> dict:
    require(payload['schema_version'] == 1, 'Unsupported schema')
    require(payload['scope']['official_leaderboard_submission'] is False, 'Scope changed')
    require(payload['dataset']['download_revision'] is None, 'Historical revision was not recorded')
    output = {}
    for key, n, methods in [
        ('pilot_10', 10, ['direct', 'reader', 'harness']),
        ('full_503', 503, ['direct', 'harness']),
        ('long_subset_82', 82, ['direct', 'harness']),
    ]:
        run = payload[key]
        rows = run['samples']
        require(len(rows) == n == len({r['index'] for r in rows}) == len({r['id'] for r in rows}), f'{key}: sample identity mismatch')
        for row in rows:
            gold = row['expected_choice']
            require(gold in 'ABCD' and len(gold) == 1, f'{key}: invalid expected choice')
            for method in methods:
                result = row[method]
                require(result['prediction'] in (None, 'A', 'B', 'C', 'D'), f'{key}: invalid prediction')
                require(result['correct'] is (result['prediction'] == gold), f'{key}: inconsistent correctness')
            if 'source' in row:
                source = row['source']
                require(re.fullmatch('[0-9a-f]{64}', source['context_sha256']) is not None, f'{key}: invalid source hash')
                require(source['legacy_context_tokens_source'] in ['char_estimate', 'provider_prompt_tokens'], f'{key}: invalid token provenance')
                require(row['harness']['reused_direct'] == (row['harness']['mode'] == 'direct_pass'), f'{key}: inconsistent direct reuse')
        output[key] = {}
        for method in methods:
            calculated = stats(rows, method)
            require(calculated == run['summary'][method], f'{key}/{method}: summary mismatch')
            output[key][method] = calculated
        if 'by_domain' in run:
            require(set(run['by_domain']) == {r['domain'] for r in rows}, f'{key}: domain set mismatch')
            for domain, summary in run['by_domain'].items():
                for method in methods:
                    require(summary[method] == stats([r for r in rows if r['domain'] == domain], method), f'{key}/{domain}: summary mismatch')
            require(dict(Counter(r['harness']['mode'] for r in rows)) == run['summary']['harness_modes'], f'{key}: mode mismatch')
            require(dict(Counter(r['source']['legacy_context_tokens_source'] for r in rows)) == run['summary']['token_value_sources'], f'{key}: length source mismatch')
            for field, row_field in [('option_verification_views', 'option_verification_views'), ('option_verification_errors', 'option_verification_error_count'), ('option_verification_retries', 'option_verification_retries')]:
                require(sum(r['harness'][row_field] for r in rows) == run['summary'][field], f'{key}: verification total mismatch')
            rejected = [r for r in rows if r['direct']['error_kind'] == 'context_limit']
            output[key]['harness_on_direct_context_rejections'] = {'n': len(rejected), 'correct': sum(r['harness']['correct'] for r in rejected)}
    full = {r['index']: r for r in payload['full_503']['samples']}
    long = {r['index']: r for r in payload['long_subset_82']['samples']}
    threshold = payload['long_subset_82']['selection']['threshold']
    selected = {i for i, r in full.items() if r['source']['index_estimated_prompt_tokens'] > threshold or r['source']['legacy_context_tokens_value'] > threshold}
    require(selected == set(long), 'Subset selection mismatch')
    for i, row in long.items():
        for field in ['id', 'expected_choice', 'source', 'direct']:
            require(row[field] == full[i][field], f'Subset {i}: cached {field} mismatch')
    for name, value in payload['source_artifacts_sha256'].items():
        require(not Path(name).is_absolute() and re.fullmatch('[0-9a-f]{64}', value) is not None, 'Invalid artifact reference')
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', nargs='?', type=Path, default=Path(__file__).resolve().parents[1] / 'benchmarks/longbench_v2_historical.json')
    args = parser.parse_args()
    output = verify(json.loads(args.results.read_text(encoding='utf-8')))
    print(json.dumps({'verified': True, 'results': output}, indent=2))


if __name__ == '__main__':
    main()
