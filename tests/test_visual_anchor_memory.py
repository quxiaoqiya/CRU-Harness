import json
from types import SimpleNamespace

from long_context_agent.experiment_budget import ExperimentBudget
from long_context_agent.trajectory_memory import TrajectoryStore
from long_context_agent.trajectory_memory_completion import PerCallBudgetedClient
from long_context_agent.visual_anchor_memory import (
    AnchorConfig, VisualAnchorIndex, action_target, normalize_perception,
    run_visual_anchor,
)


def make_store(tmp_path):
    # Synthetic PNG bytes are never decoded by these transport-level tests.
    (tmp_path/'current.png').write_bytes(b'current screenshot')
    (tmp_path/'before.png').write_bytes(b'before screenshot')
    (tmp_path/'after.png').write_bytes(b'after screenshot')
    store = TrajectoryStore(tmp_path)
    store.insert({'id': 't', 'states': [
        {'url': '/review/edit', 'action': None, 'screenshot': 'before.png',
         'accessibility_tree': "RootWebArea 'Edit Review'\n[42] button 'Back'"},
        {'url': '/review/all', 'action': "click('42')", 'screenshot': 'after.png',
         'accessibility_tree': "RootWebArea 'All Reviews'\nheading 'All Reviews'"},
    ]})
    store.finalize()
    return store


def test_incoming_action_uses_previous_control_and_current_destination(tmp_path):
    index = VisualAnchorIndex(make_store(tmp_path))
    assert len(index.transitions) == 1
    transition = index.transitions[0]
    assert (transition['before'], transition['after']) == (0, 1)
    assert transition['control'] == "[42] button 'Back'"


def test_action_parser_rejects_agent_message_and_executable_expressions():
    assert action_target("send_msg_to_user('success')", '') is None
    assert action_target("click(__import__('os').system('bad'))", '') is None
    assert action_target("click('x')", "[x] button 'Exact control'") == ('click', "[x] button 'Exact control'")


def test_perception_drops_answer_field_and_bounds_generated_hints():
    data = normalize_perception(json.dumps({'answer': 'do this', 'visible_labels': ['A']*30,
        'visible_state_facts': ['State: blocked'], 'retrieval_queries': ['q']*20,
        'transition_needed': 'true'}))
    assert 'answer' not in data and len(data['visible_labels']) == 16
    assert len(data['retrieval_queries']) == 4 and data['transition_needed'] is False


def test_transition_retrieval_preserves_both_observations(tmp_path):
    index = VisualAnchorIndex(make_store(tmp_path))
    perception = normalize_perception(json.dumps({'visible_labels': ['Edit Review', 'Back'], 'transition_needed': True}))
    cards, links = index.retrieve_anchors(['Back Edit Review'], perception)
    assert [n for n, _ in cards][:2] == [0, 1] and links
    assert cards[0][1].field_sha256 != cards[1][1].field_sha256


def test_distinct_screenshots_with_identical_tree_are_not_discarded(tmp_path):
    store = make_store(tmp_path)
    store._rows['t']['states'][1]['accessibility_tree'] = store._rows['t']['states'][0]['accessibility_tree']
    store.index = None
    store.finalize()
    index = VisualAnchorIndex(store)
    cards, _ = index.retrieve_anchors(['Edit Review Back'], normalize_perception('{}'))
    assert {n for n, _ in cards} == {0, 1}
    assert cards[0][1].field_sha256 == cards[1][1].field_sha256
    assert cards[0][1].image != cards[1][1].image


class Transport:
    model = 'test'
    def __init__(self, empty_final=False):
        self.bodies = []
        self.empty_final = empty_final
    def count(self, **kwargs):
        return 600
    def request(self, path, body):
        self.bodies.append(body)
        n = len(self.bodies)
        if n == 1:
            content = json.dumps({'visible_labels': ['Edit Review', 'Back'], 'visible_state_facts': ['Status: pending'],
                                  'retrieval_queries': ['Edit Review Back destination'], 'transition_needed': True})
        elif 'Do not answer yet' in json.dumps(body['messages']):
            content = json.dumps({'relevant_ids': [], 'missing': [], 'complete': True, 'conflicts': []})
        else:
            content = '' if self.empty_final else '\\boxed{All Reviews}'
        return {'usage': {'prompt_tokens': 600, 'completion_tokens': 50},
                'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}


def test_actual_counted_original_images_and_complete_transition_are_in_final(tmp_path):
    store = make_store(tmp_path)
    transport = Transport()
    client = PerCallBudgetedClient(transport, ExperimentBudget(16384,65536,8,4096))
    result = run_visual_anchor(VisualAnchorIndex(store), 'Where does Back go?', client,
                               domain_system='policy', query_image='current.png')
    assert result['answer'] == '\\boxed{All Reviews}' and len(transport.bodies) == 3
    final = result['image_events'][-1]
    assert len(final['images']) == 3 and final['images'][-1]['query_image']
    assert len(final['transition_receipts']) == 1 and not result['pinned_omitted']
    assert 'All Reviews' in json.dumps(transport.bodies[-1]['messages'])
    assert client.summary()['within_budget']


def test_query_only_control_never_supplies_historical_evidence(tmp_path):
    store = make_store(tmp_path)
    transport = Transport()
    client = PerCallBudgetedClient(transport, ExperimentBudget(16384,65536,8,4096))
    result = run_visual_anchor(VisualAnchorIndex(store), 'Question', client,
        domain_system='policy', query_image='current.png', query_only=True)
    assert len(transport.bodies) == 2 and not result['final_evidence_ids']
    assert all(len(e['images']) == 1 and e['images'][0]['query_image'] for e in result['image_events'])


def test_empty_final_is_not_silently_retried(tmp_path):
    store = make_store(tmp_path)
    transport = Transport(empty_final=True)
    client = PerCallBudgetedClient(transport, ExperimentBudget(16384,65536,8,4096))
    result = run_visual_anchor(VisualAnchorIndex(store), 'Question', client,
        domain_system='policy', query_image='current.png', query_only=True)
    assert result['answer'] == '' and len(transport.bodies) == 2


def test_no_query_image_delegates_to_existing_completion_route(tmp_path, monkeypatch):
    import long_context_agent.visual_anchor_memory as module
    calls = []
    monkeypatch.setattr(module, 'run_completion_harness', lambda *args: calls.append(args) or {'answer': 'original'})
    result = run_visual_anchor(VisualAnchorIndex(make_store(tmp_path)), 'text question', SimpleNamespace(),
                               domain_system='policy', query_image=None)
    assert result == {'answer': 'original'} and len(calls) == 1
