import copy
import json
from pathlib import Path
import tempfile
import unittest

from scripts.prepare_context_holdout import REVISION, json_bytes
from scripts.prepare_context_holdout_round3 import (
    LENGTHS, N, PER_LENGTH, atomic_bytes, prior_exclusions, seal_plan, select_from_catalog, validate_materialized,
)


def fabricated_catalog():
    rows = []
    for length in LENGTHS:
        for index in range(36):
            rows.append({'id': length + index, 'context_len': length,
                         'dataset': f'd-{index % 3}', 'context_window_id': length + index,
                         'task_group': ['counting', 'timeline', 'user'][index % 3],
                         'task': 'fabricated', 'answer_type': 'ANSWER_TYPE.NUMERIC', '_row_idx': len(rows)})
    return rows


def consumed_manifests(catalog):
    manifests = []
    for cohort in range(2):
        selected = []
        for length in LENGTHS:
            candidates = [r for r in catalog if r['context_len'] == length]
            for row in candidates[cohort * 6:(cohort + 1) * 6]:
                selected.append({'metadata': {k: row[k] for k in ('dataset', 'context_window_id')},
                                 'context_sha256': f'consumed-{row["id"]}'})
        manifests.append({'dataset_revision': REVISION, 'selected': selected})
    return manifests


class Cohort3Tests(unittest.TestCase):
    def test_selection_is_deterministic_excludes_both_consumed_cohorts(self):
        catalog = fabricated_catalog()
        manifests = consumed_manifests(catalog)
        selected, contexts, hashes = select_from_catalog(catalog, manifests)
        selected_again, _, _ = select_from_catalog(catalog, manifests)
        self.assertEqual(selected, selected_again)
        self.assertEqual(len(contexts), 36)
        self.assertEqual(len(hashes), 36)
        self.assertEqual(len(selected), N)
        self.assertEqual([sum(r['context_len'] == length for r in selected) for length in LENGTHS], [PER_LENGTH] * 3)
        ids = {(r['dataset'], r['context_window_id']) for r in selected}
        self.assertEqual(len(ids), N)
        self.assertFalse(ids & contexts)

    def test_selection_does_not_consult_question_or_gold_fields(self):
        catalog = fabricated_catalog()
        manifests = consumed_manifests(catalog)
        original, _, _ = select_from_catalog(catalog, manifests)
        altered = copy.deepcopy(catalog)
        for row in altered:
            row.update(question='Never choose using this question', answer='NOT_A_SELECTOR')
        selected, _, _ = select_from_catalog(altered, manifests)
        self.assertEqual([r['id'] for r in original], [r['id'] for r in selected])

    def test_prior_context_or_hash_overlap_fails_closed(self):
        catalog = fabricated_catalog()
        manifests = consumed_manifests(catalog)
        manifests[1]['selected'][0] = manifests[0]['selected'][0]
        with self.assertRaises(ValueError):
            prior_exclusions(manifests)

    def test_prior_revision_change_is_rejected(self):
        manifests = consumed_manifests(fabricated_catalog())
        manifests[1]['dataset_revision'] = 'unreviewed'
        with self.assertRaises(ValueError):
            prior_exclusions(manifests)

    def test_exhausted_length_bucket_is_not_refilled_with_consumed_contexts(self):
        catalog = [r for r in fabricated_catalog() if r['id'] - r['context_len'] < 15]
        manifests = consumed_manifests(catalog)
        with self.assertRaises(ValueError):
            select_from_catalog(catalog, manifests)

    def test_source_hash_overlap_is_rejected_after_materialization(self):
        rows = [{'id': str(i), 'context_sha256': f'hash-{i}'} for i in range(N)]
        validate_materialized(rows, {'consumed'})
        with self.assertRaises(ValueError):
            validate_materialized(rows, {'hash-9'})
        rows[-1]['context_sha256'] = rows[0]['context_sha256']
        with self.assertRaises(ValueError):
            validate_materialized(rows, set())

    def test_inference_gold_field_is_rejected(self):
        rows = [{'id': str(i), 'context_sha256': f'hash-{i}'} for i in range(N)]
        rows[0]['answer'] = 'forbidden'
        with self.assertRaises(ValueError):
            validate_materialized(rows, set())

    def test_sealed_selection_cannot_be_silently_replaced(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temporary:
            out = Path(temporary)
            first = seal_plan(out, {'seed': 20261103})
            self.assertEqual(first, seal_plan(out, {'seed': 20261103}))
            with self.assertRaises(RuntimeError):
                seal_plan(out, {'seed': 0})

    def test_gold_is_only_written_with_private_mode_and_atomic_replace(self):
        with tempfile.TemporaryDirectory(dir='/private/tmp') as temporary:
            path = Path(temporary) / 'labels.json'
            atomic_bytes(path, b'{"fabricated": [17]}\n', private=True)
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(json.loads(path.read_text()), {'fabricated': [17]})
            self.assertFalse(path.with_name('labels.json.tmp').exists())

    def test_cached_input_json_preserves_unicode_without_extra_lines(self):
        raw = json_bytes({'context': 'a\u2028b\u2029c', 'question': 'fabricated'}) + b'\n'
        self.assertEqual(len(raw.decode().splitlines()), 1)
        self.assertEqual(json.loads(raw)['context'], 'a\u2028b\u2029c')


if __name__ == '__main__':
    unittest.main()
