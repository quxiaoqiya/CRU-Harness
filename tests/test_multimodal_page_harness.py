import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from long_context_agent.multimodal_page_harness import PageIndex, head_tail, run_memory


def call(name, args, key):
    return {'role': 'assistant', 'content': None, 'tool_calls': [
        {'id': key, 'type': 'function', 'function': {'name': name, 'arguments': json.dumps(args)}}]}


class FakeClient:
    def __init__(self, stop_after=99):
        self.ledger = []
        self.remaining_calls = 8
        self.remaining_tokens = 262144
        self.stop_after = stop_after
        self.messages = []
        self.outputs = [call('search_pages', {'query': 'diagram'}, 'search'),
                        call('read_pages', {'page_ids': ['p1']}, 'read'),
                        {'role': 'assistant', 'content': '{"answer":"UNKNOWN","evidence_ids":["p1"]}'}]

    def can_call(self, *args, **kwargs):
        return len(self.ledger) < self.stop_after

    def complete_native(self, messages, *args, **kwargs):
        self.messages.append(json.loads(json.dumps(messages)))
        self.ledger.append({})
        self.remaining_calls -= 1
        return self.outputs[len(self.ledger) - 1]


class PageMemoryIntegrityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.image = b'original image bytes'
        (self.root / 'page.png').write_bytes(self.image)
        self.pages = [{'id': 'p1', 'doc_id': 'source.pdf', 'page': 1, 'text': 'diagram',
                       'image': 'page.png', 'image_sha256': hashlib.sha256(self.image).hexdigest()}]

    def tearDown(self):
        self.tmp.cleanup()

    def test_text_memory_never_reads_or_sends_pixels(self):
        index = PageIndex(self.pages, self.root)
        (self.root / 'page.png').unlink()
        client = FakeClient()
        result = run_memory('What color?', index, client, multimodal=False)
        self.assertEqual(result['pixel_page_ids'], [])
        self.assertNotIn('image_url', json.dumps(client.messages))
        self.assertEqual(result['read_page_ids'], ['p1'])

    def test_pixels_count_only_after_physical_model_exposure(self):
        result = run_memory('What color?', PageIndex(self.pages, self.root), FakeClient(stop_after=2), multimodal=True)
        self.assertEqual(result['read_page_ids'], ['p1'])
        self.assertEqual(result['pixel_page_ids'], [])

    def test_exposed_original_images_are_bound_to_hash(self):
        client = FakeClient()
        result = run_memory('What color?', PageIndex(self.pages, self.root), client, multimodal=True)
        self.assertEqual(result['pixel_page_ids'], ['p1'])
        self.assertEqual(result['final_pixel_page_ids'], ['p1'])
        self.assertIn('image_url', json.dumps(client.messages[-1]))
        self.assertNotIn('image_url', json.dumps(client.messages[0]))

    def test_corrupted_source_is_rejected_before_exposure(self):
        (self.root / 'page.png').write_bytes(b'altered')
        with self.assertRaisesRegex(ValueError, 'changed'):
            run_memory('What color?', PageIndex(self.pages, self.root), FakeClient(), multimodal=True)

    def test_gold_cannot_enter_memory(self):
        with self.assertRaisesRegex(ValueError, 'Scoring labels'):
            PageIndex([{**self.pages[0], 'answer': 'red'}], self.root)

    def test_unreturned_page_cannot_be_read(self):
        client = FakeClient()
        client.outputs[1] = call('read_pages', {'page_ids': ['unobserved']}, 'read')
        result = run_memory('What color?', PageIndex(self.pages, self.root), client, multimodal=True)
        self.assertEqual(result['pixel_page_ids'], [])

    def test_repeat_search_can_page_to_new_sources(self):
        pages = [{**self.pages[0], 'id': 'p' + str(i), 'page': i} for i in range(20)]
        index = PageIndex(pages, self.root)
        first = {p['id'] for p in index.search('diagram', limit=8)}
        second = {p['id'] for p in index.search('diagram', limit=8, offset=8)}
        self.assertFalse(first & second)

    def test_direct_head_tail_keeps_page_order_and_no_duplicates(self):
        ids = list(range(12))
        self.assertEqual(head_tail(ids, 5), [0, 1, 2, 10, 11])
        self.assertEqual(head_tail(ids, 1), [0])
        self.assertEqual(head_tail(ids, 12), ids)


if __name__ == '__main__':
    unittest.main()
