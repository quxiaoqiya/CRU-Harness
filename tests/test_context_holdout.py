import io
import json
import unittest
from unittest.mock import patch

from scripts.prepare_context_holdout import HTTPRangeFile, choose_holdout, inference_record, json_bytes, score_prediction


class HoldoutTests(unittest.TestCase):
    def test_selection_is_seeded_and_contexts_are_disjoint(self):
        rows = []
        for length in [65536, 131072, 524288]:
            for index in range(30):
                rows.append({'id': length + index, 'context_len': length, 'dataset': f'd{index % 8}',
                             'context_window_id': index, 'task_group': ['counting', 'temporal', 'user'][index % 3]})
        selected = choose_holdout(rows)
        self.assertEqual(selected, choose_holdout(rows))
        self.assertEqual(len(selected), 18)
        self.assertEqual(len({(r['dataset'], r['context_window_id']) for r in selected}), 18)
        self.assertEqual([sum(r['context_len'] == n for r in selected) for n in [65536, 131072, 524288]], [6, 6, 6])

    def test_insufficient_independent_contexts_fails(self):
        rows = [{'id': i, 'context_len': 32768, 'dataset': 'd', 'context_window_id': 1, 'task_group': 'c'} for i in range(8)]
        with self.assertRaises(ValueError):
            choose_holdout(rows, lengths=[32768], per_length=2)

    def test_inference_input_drops_gold_and_labeled_context(self):
        row = {'id': 4, 'context_len': 32768, 'dataset': 'd', 'context_window_id': 7,
               'task': 'count', 'task_group': 'c', 'answer_type': 'ANSWER_TYPE.NUMERIC',
               'context_window_text': 'unlabeled public input', 'context_window_text_with_labels': 'GOLD_CLASS',
               'question': 'count entries', 'answer': '[123456789]'}
        result = inference_record(row, split='test')
        self.assertNotIn('answer', result)
        self.assertNotIn('123456789', json.dumps(result))
        self.assertNotIn('GOLD_CLASS', json.dumps(result))
        self.assertEqual(result['context'], row['context_window_text'])

    def test_unicode_source_line_separators_cannot_break_jsonl(self):
        text = 'source\u0085second\u2028third\u2029fourth'
        raw = json_bytes({'context': text}) + b'\n'
        self.assertEqual(len(raw.decode().splitlines()), 1)
        self.assertEqual(json.loads(raw)['context'], text)

    def test_official_numeric_partial_credit_is_not_exact_match(self):
        label = {'answer': '[10]', 'answer_type': 'ANSWER_TYPE.NUMERIC'}
        result = score_prediction('Answer: 12', label)
        self.assertAlmostEqual(result['official_score'], 0.75 ** 2)
        self.assertFalse(result['exact_match'])
        self.assertFalse(result['full_credit'])
        self.assertEqual(score_prediction('Answer: 10', label)['official_score'], 1.0)

    def test_official_comparison_phrase_and_date(self):
        result = score_prediction('Answer: more common', {'answer': "['more common than the other label']", 'answer_type': 'ANSWER_TYPE.COMPARISON'})
        self.assertEqual(result['official_score'], 1.0)
        self.assertFalse(result['exact_match'])
        self.assertEqual(score_prediction('Date: 2020-01-02', {'answer': '[datetime.date(2020, 1, 2)]', 'answer_type': 'ANSWER_TYPE.DATE'})['official_score'], 1.0)

    def test_range_reader_seeks_and_handles_small_full_body(self):
        raw = b'abcdefghij'
        with patch('scripts.prepare_context_holdout.fetch', return_value=raw):
            reader = HTTPRangeFile('https://example.invalid/read-only', len(raw))
            self.assertEqual(reader.seek(-4, 2), 6)
            self.assertEqual(reader.read(3), b'ghi')
            self.assertEqual(reader.tell(), 9)
            self.assertEqual(reader.read(), b'j')


if __name__ == '__main__':
    unittest.main()
