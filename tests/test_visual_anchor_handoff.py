import json
from types import SimpleNamespace

from long_context_agent.experiment_budget import ExperimentBudget
from long_context_agent.trajectory_memory import TrajectoryStore
from long_context_agent.trajectory_memory_completion import PerCallBudgetedClient
from long_context_agent.visual_anchor_memory_v3 import (
    AnchorConfig, VisualAnchorIndex, normalize_actions, normalize_review,
    prioritize_evidence, run_visual_anchor, needs_transition,
)


def test_ui_dialects_and_order_are_literal_only():
    tree = "[42] textbox 'Assigned to'\n[43] button 'Update'"
    events = normalize_actions("clear('42')\npress('42', 'Tab')\nclick('43')", tree)
    assert [e['name'] for e in events] == ['clear', 'press', 'click']
    assert events[1]['control'] == "[42] textbox 'Assigned to'"
    assert normalize_actions("tab_focus(2)", tree)[0]['control'] == ''
    assert normalize_actions("focus('42')", tree)[0]['control']
    assert normalize_actions("dblclick('43')", tree)[0]['control']
    assert normalize_actions("noop()", tree)[0]['name'] == 'noop'
    assert not normalize_actions("click(__import__('os').system('bad'))", tree)
    assert not normalize_actions("unknown(); click('43')", tree)
    assert not normalize_actions("send_msg_to_user('saved')", tree)


def test_tagged_thought_is_never_rendered_as_event():
    events = normalize_actions("<think>The answer is saved</think><action>click('43')</action>", "[43] button 'Update'")
    assert len(events) == 1 and 'answer' not in json.dumps(events)
    assert not normalize_actions("<think>click('43')</think>", '')


def test_latest_selection_replaces_full_old_pin_set_and_preserves_pair():
    result = prioritize_evidence(['new2', 'new3'], ['old1', 'old2', 'old3', 'old4'],
        ['new1', 'new2', 'new3'], [{'before': 'new1', 'after': 'new2'}], 6)
    assert result[:3] == ['new2', 'new1', 'new3']
    assert len(result) == 6


def test_complete_cannot_hide_required_gap_but_optional_search_is_not_gap():
    result = normalize_review(json.dumps({'complete': True, 'required_gaps': ['saved state'],
        'followup_queries': ['saved value'], 'relevant_ids': ['E1', 'invented'],
        'visual_ids': ['E1', 'invented']}), ['E1'])
    assert not result['complete'] and result['reported_complete']
    assert result['relevant_ids'] == ['E1'] and result['visual_ids'] == ['E1']
    result = normalize_review(json.dumps({'complete': True, 'followup_queries': ['optional']}), [])
    assert result['complete'] and not result['required_gaps']
    result = normalize_review(json.dumps({'complete': True, 'requirements': [
        {'fact': 'saved', 'verdict': 'support', 'evidence_ids': ['invented']}]}), ['E1'])
    assert not result['complete'] and result['requirements'][0]['verdict'] == 'missing'


def make_store(tmp_path):
    (tmp_path/'current.png').write_bytes(b'current')
    store = TrajectoryStore(tmp_path)
    states = []
    for i in range(12):
        (tmp_path/f'{i}.png').write_bytes(str(i).encode())
        states.append({'accessibility_tree': f"RootWebArea 'Device {i}'\n[{i}] button 'Save'",
                       'screenshot': f'{i}.png', 'action': f"click('{i-1}')" if i else None})
    store.insert({'id': 't', 'states': states});store.finalize()
    return store


class Transport:
    model = 'test'
    def __init__(self):
        self.bodies = []
    def count(self, **kwargs):
        return 600
    def request(self, path, body):
        self.bodies.append(body)
        n = len(self.bodies)
        if n == 1:
            content = '{}'
        elif n in (2, 3):
            text = body['messages'][1]['content'][0]['text']
            rows = [json.loads(s.split('\n', 1)[0]) for s in text.split('Observed history fields:\n')[1].split('\n\n')]
            ids = [r['evidence_id'] for r in rows]
            content = json.dumps({'relevant_ids': ids[:4], 'visual_ids': ids[:1],
                'complete': True, 'required_gaps': ['confirm saved identity'],
                'followup_queries': ['Device saved identity'], 'conflicts': ['historical entity differs']})
        else:
            content = '\\boxed{I do not know}'
        return {'usage': {'prompt_tokens': 600, 'completion_tokens': 100},
                'choices': [{'message': {'content': content}, 'finish_reason': 'stop'}]}


def test_multiround_new_evidence_gaps_identity_and_pixels_reach_final(tmp_path):
    index = VisualAnchorIndex(make_store(tmp_path))
    # Deterministic disjoint batches reproduce two reviews with a full old pin set.
    index.retrieve_anchors = lambda queries, perception, seen: (
        [(n, index.card(n, 'Device')) for n in (range(6) if not seen else range(6, 12))], [])
    transport = Transport()
    client = PerCallBudgetedClient(transport, ExperimentBudget(16384, 65536, 8, 4096))
    result = run_visual_anchor(index, 'Which device was saved?', client,
                               domain_system='policy', query_image='current.png')
    assert len(transport.bodies) == 4 and len(result['trace']) == 2
    assert set(result['trace'][-1]['selected_ids']) <= set(result['final_evidence_ids'])
    assert not result['pinned_omitted'] and not result['final_visual_omitted']
    final = json.dumps(transport.bodies[-1]['messages'])
    assert 'confirm saved identity' in final and 'historical entity differs' in final
    assert 'historical example' in final and 'text only in THIS packet' in final
    assert all('action_leading_into_this_state' not in p for p in [final])
    assert client.summary()['within_budget']


def test_query_only_has_no_history_and_no_implicit_retry(tmp_path):
    transport = Transport()
    transport.request = lambda path, body: {'usage': {'prompt_tokens': 600, 'completion_tokens': 0},
        'choices': [{'message': {'content': ''}, 'finish_reason': 'stop'}]}
    client = PerCallBudgetedClient(transport, ExperimentBudget(16384, 65536, 8, 4096))
    result = run_visual_anchor(VisualAnchorIndex(make_store(tmp_path)), 'Question', client,
        domain_system='policy', query_image='current.png', query_only=True)
    assert len(client.ledger) == 2 and result['answer'] == ''
    assert not result['final_evidence_ids']
    assert all(len(e['images']) == 1 for e in result['image_events'])


def test_text_route_remains_original_completion(tmp_path, monkeypatch):
    import long_context_agent.visual_anchor_memory_v3 as module
    monkeypatch.setattr(module, 'run_completion_harness', lambda *a: {'answer': 'original'})
    assert run_visual_anchor(VisualAnchorIndex(make_store(tmp_path)), 'text', SimpleNamespace(),
        domain_system='policy', query_image=None) == {'answer': 'original'}


def test_transition_fallback_is_narrow():
    assert needs_transition('Why does clicking Update do nothing?')
    assert needs_transition('Does Change Billing overwrite Shipping?')
    assert not needs_transition('How many users are listed?')
    assert not needs_transition('How do I locate an item?')
