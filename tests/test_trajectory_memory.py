from pathlib import Path
import base64
import json
import pytest

from long_context_agent.trajectory_memory import (TrajectoryConfig, TrajectoryStore,
    TrajectoryEvidenceClient, retrieve_trajectory_evidence)
from long_context_agent.adaptive_evidence import EvidenceWindow


PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aB1kAAAAASUVORK5CYII=')


def trajectory(*, image=None, text='The current order status is approved.'):
    return {'id': 't1', 'domain': 'web', 'goal': 'Review an order', 'outcome': 'success',
            'environment': 'shop', 'start_url': 'https://example.test/orders',
            'states': [{'state_index': 0, 'step': 0, 'action': 'click(order)',
                        'thought': 'Inspect current order', 'url': 'https://example.test/order/1',
                        'accessibility_tree': text, 'screenshot': image}]}


class FakeClient:
    context_tokens = 16384
    def __init__(self, answers=()):
        self.ledger, self.sent, self.counted = [], [], []
        self.answers = iter(answers)
    @property
    def remaining_tokens(self):return 65536-len(self.ledger)*500
    @property
    def remaining_calls(self):return 8-len(self.ledger)
    def count_text(self, text):return max(1, len(text)//4)
    def count_messages(self, messages):
        self.counted.append(messages)
        return max(1, len(json.dumps(messages))//4)
    def can_call(self, messages, max_tokens=1024):
        return self.count_messages(messages)+max_tokens <= self.context_tokens
    def complete(self, messages, max_tokens=1024):
        assert self.can_call(messages,max_tokens)
        self.sent.append(messages);self.ledger.append({'status':'ok'})
        return next(self.answers, '\\boxed{approved}')


def test_lossless_state_fields_and_original_source_identity(tmp_path):
    row = trajectory()
    store = TrajectoryStore(tmp_path);store.insert(row);store.finalize()
    assert json.dumps(row['states'][0], ensure_ascii=False,sort_keys=True) in store.context
    state = store.bindings[0]
    assert '[state 0]' in store.context[state.start:state.end]
    assert store.metadata()['state_count'] == 1
    with pytest.raises(ValueError,match='silently change'):
        store.insert(trajectory(text='changed'))


def test_labels_and_scoring_metadata_rejected(tmp_path):
    store = TrajectoryStore(tmp_path)
    for name in ('answer','gold','question_id','question_type','eval_function'):
        with pytest.raises(ValueError,match='labels or query metadata'):
            store.insert({**trajectory(),name:'secret'})


def test_image_admission_and_generation_have_identical_bytes(tmp_path):
    (tmp_path/'state.png').write_bytes(PNG)
    (tmp_path/'query.png').write_bytes(PNG)
    store = TrajectoryStore(tmp_path);store.insert(trajectory(image='state.png'));store.finalize()
    raw = FakeClient();client = TrajectoryEvidenceClient(raw,store,domain_system='Never guess.',query_image='query.png')
    window = EvidenceWindow(0,len(store.context),store.context,'q',1)
    messages=[{'role':'system','content':'Answer from evidence.'},{'role':'user','content':window.render()}]
    client.count_messages(messages)
    counted=raw.counted[-1]
    client.complete(messages)
    assert counted==raw.sent[-1]
    images=[p for p in raw.sent[-1][-1]['content'] if p['type']=='image_url']
    assert len(images)==2
    assert client.image_events[0]['images'][0]['trajectory_id']=='t1'
    assert client.image_events[0]['images'][1]['query_image'] is True


def test_missing_or_external_assets_are_never_silently_dropped(tmp_path):
    store=TrajectoryStore(tmp_path);store.insert(trajectory(image='missing.png'));store.finalize()
    client=TrajectoryEvidenceClient(FakeClient(),store,domain_system='policy')
    w=EvidenceWindow(0,len(store.context),store.context,'q',1)
    with pytest.raises(ValueError,match='Missing screenshot'):
        client.prepare([{'role':'system','content':'policy'},{'role':'user','content':w.render()}])
    with pytest.raises(ValueError):store.resolve_image('../outside.png')


def test_retrieved_images_are_capped_and_bound_to_exposed_states(tmp_path):
    for i in range(5):(tmp_path/f'{i}.png').write_bytes(PNG)
    row=trajectory();row['states']=[{**row['states'][0],'state_index':i,'screenshot':f'{i}.png'} for i in range(5)]
    store=TrajectoryStore(tmp_path);store.insert(row);store.finalize()
    first=store.bindings[0]
    selected=store.images_for_spans([(first.start,first.end)],limit=3)
    assert len(selected)==1 and selected[0].state_index==0
    assert len(store.images_for_spans([(0,len(store.context))],limit=3))==3
    broad=store.images_for_spans([(0,len(store.context))],limit=2,selection='uniform')
    assert [s.state_index for s in broad]==[0,4]


def test_text_only_inputs_keep_text_and_add_no_history_images(tmp_path):
    store=TrajectoryStore(tmp_path);store.insert(trajectory());store.finalize()
    raw=FakeClient();client=TrajectoryEvidenceClient(raw,store,domain_system='policy')
    w=EvidenceWindow(0,len(store.context),store.context,'q',1)
    prepared,images=client.prepare([{'role':'system','content':'Answer.'},{'role':'user','content':w.render()}])
    assert not images and prepared[-1]['content']==[{'type':'text','text':w.render()}]


def test_benchmark_final_reader_keeps_official_memory_question_image_order(tmp_path):
    (tmp_path/'query.png').write_bytes(PNG)
    store=TrajectoryStore(tmp_path);store.insert(trajectory());store.finalize()
    client=TrajectoryEvidenceClient(FakeClient(),store,domain_system='official domain policy',
        query_image='query.png',question_text='Which status?')
    w=EvidenceWindow(0,len(store.context),store.context,'q',1)
    messages=__import__('long_context_agent.adaptive_evidence',fromlist=['_messages'])._messages(
        'Which status?',[w],tentative='UNVERIFIED GUESS',final=True)
    prepared,_=client.prepare(messages)
    assert prepared[0]['content']=='official domain policy'
    parts=prepared[1]['content']
    assert parts[0]=={'type':'text','text':'### Memory context:\n'}
    assert parts[1]['text']==w.render()
    assert parts[-2]=={'type':'text','text':'\n\n### Question to answer:\nWhich status?'}
    assert parts[-1]['type']=='image_url'
    assert 'UNVERIFIED GUESS' not in str(parts)


def test_evidence_gap_controller_pins_supported_raw_windows(tmp_path):
    store=TrajectoryStore(tmp_path,config=TrajectoryConfig(block_chars=400,halo_chars=0,
        max_rounds=2,max_windows_per_round=1))
    store.insert(trajectory(text=('approved order status '*2000)));store.finalize()
    first=store.index.rank(__import__('long_context_agent.adaptive_evidence',fromlist=['EvidenceGap']).EvidenceGap('q','approved'))[0]
    replies=[json.dumps({'answer':'approved','needs':[{'id':'more','query':'status','reason':'cross-check'}],
        'supported_spans':[first.span],'complete':False}),
        json.dumps({'answer':'approved','needs':[],'supported_spans':[],'complete':False}),
        '\\boxed{approved}']
    raw=FakeClient(replies);client=TrajectoryEvidenceClient(raw,store,domain_system='Do not guess.')
    result=retrieve_trajectory_evidence(store,'approved',client)
    assert result['answer']=='\\boxed{approved}'
    assert first.span in result['final_spans']
    assert not result['supported_spans_not_in_final']
    assert len(raw.ledger)<=3
