import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from long_context_agent.multimodal_page_harness import PageIndex
from long_context_agent.multimodal_page_harness_v2 import CountFacts, CoverageRecall, recover_answer, run_stream_counts, VisualMemoryConfig, count_records


class CountsClient:
    def __init__(self, outputs):
        self.outputs = iter(outputs)
        self.ledger = []
        self.remaining_calls = len(outputs)
        self.requests = []

    def can_call(self, *args, **kwargs):
        return self.remaining_calls > 0

    def complete(self, messages, max_tokens):
        self.requests.append(messages)
        self.ledger.append({})
        self.remaining_calls -= 1
        value = next(self.outputs)
        if isinstance(value, Exception):
            raise value
        return value


class VisualMemoryV2Tests(unittest.TestCase):
    def test_duplicate_fact_is_not_double_counted(self):
        memory = CountFacts(['a'])
        for request in [1, 2]:
            memory.observe('a', {'count': 3}, exposed={'a'}, physical_request=request)
        self.assertEqual(memory.summary()['totals'], {'count': 3})

    def test_absent_source_makes_total_unknown_not_zero(self):
        memory = CountFacts(['a', 'b'])
        memory.observe('a', {'count': 3}, exposed={'a'}, physical_request=1)
        self.assertFalse(memory.summary()['complete'])
        self.assertIsNone(memory.summary()['totals'])

    def test_conflicting_observations_require_fresh_review(self):
        memory = CountFacts(['a'])
        memory.observe('a', {'count': 3}, exposed={'a'}, physical_request=1)
        memory.observe('a', {'count': 4}, exposed={'a'}, physical_request=2)
        self.assertFalse(memory.summary()['complete'])
        memory.observe('a', {'count': 4}, exposed={'a'}, physical_request=3, reviewed=True)
        self.assertEqual(memory.summary()['totals'], {'count': 4})

    def test_fact_cannot_claim_unseen_pixels(self):
        with self.assertRaises(ValueError):
            CountFacts(['a']).observe('a', {'count': 7}, exposed=set(), physical_request=1)

    def test_compact_count_array_preserves_source_order_and_missing_is_rejected(self):
        self.assertEqual(count_records('{"counts":[[2,1],[0,3]]}', ['b','a'], ['figures','tables']),
                         [{'id':'b','figures':2,'tables':1},{'id':'a','figures':0,'tables':3}])
        self.assertEqual(count_records('{"counts":[[2,1]]}', ['b','a'], ['figures','tables']), [])
        self.assertEqual(count_records('[[2,1],[0,3]]', ['b','a'], ['figures','tables']),
                         [{'id':'b','figures':2,'tables':1},{'id':'a','figures':0,'tables':3}])
        self.assertEqual(count_records('[{"page_id":"a","count":2}]', ['a'], ['count']),
                         [{'page_id':'a','id':'a','count':2}])

    def test_negative_null_and_boolean_are_not_counts(self):
        for value in [-1, None, True]:
            memory = CountFacts(['a'])
            self.assertFalse(memory.observe('a', {'count': value}, exposed={'a'}, physical_request=1))
            self.assertFalse(memory.summary()['complete'])

    def test_prose_wrapped_json_and_false_tool_final_recovery(self):
        self.assertEqual(recover_answer('Observed pixels.\n{"answer":"yellow"}')[0], 'yellow')
        self.assertEqual(recover_answer('', [{'function': {'name': 'final_answer', 'arguments': '{"answer":"10"}'}}])[0], '10')
        self.assertEqual(recover_answer('', [{'function': {'name': 'search_pages', 'arguments': '{"answer":"10"}'}}])[0], '')

    def make_index(self, directory, n=2):
        data = b'original source image'
        (Path(directory) / 'page.png').write_bytes(data)
        return PageIndex([{'id': 'p' + str(i), 'doc_id': 'source.pdf', 'page': i, 'text': '',
            'image': 'page.png', 'image_sha256': hashlib.sha256(data).hexdigest()} for i in range(1, n + 1)], directory)

    def test_image_only_repeated_search_advances(self):
        with tempfile.TemporaryDirectory() as directory:
            recall = CoverageRecall(self.make_index(directory, 32))
            first = {p['id'] for p in recall.search('injector', 'source.pdf')}
            second = {p['id'] for p in recall.search('injector', 'source.pdf')}
            self.assertFalse(first & second)
            self.assertTrue(any(int(p[1:]) > 8 for p in first))

    def test_count_recovery_and_aggregation_use_real_returned_pixels(self):
        with tempfile.TemporaryDirectory() as directory:
            client = CountsClient(['', '{"pages":[{"id":"p1","count":2},{"id":"p2","count":3}]}'])
            result = run_stream_counts('Count faces', self.make_index(directory), ['p1', 'p2'], client)
            self.assertEqual(result['answer'], '5')
            self.assertEqual(result['recovery_calls'], 1)
            self.assertEqual(result['pixel_page_ids'], ['p1', 'p2'])
            self.assertEqual(len(client.ledger), 2)

    def test_incomplete_recovery_returns_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_stream_counts('Count faces', self.make_index(directory), ['p1', 'p2'], CountsClient(['']))
            self.assertEqual(result['answer'], 'UNKNOWN')
            self.assertEqual(result['pixel_page_ids'], ['p1', 'p2'])

    def test_uncertain_transport_is_not_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            client = CountsClient([TimeoutError('delivery uncertain'), '{}'])
            with self.assertRaises(TimeoutError):
                run_stream_counts('Count faces', self.make_index(directory), ['p1', 'p2'], client)
            self.assertEqual(len(client.ledger), 1)


if __name__ == '__main__':
    unittest.main()
