import base64,json
from types import SimpleNamespace
from long_context_agent.trajectory_memory import TrajectoryStore,TrajectoryEvidenceClient
from long_context_agent.trajectory_memory_assist import AssistEvidenceClient,AssistConfig,boxed_answer,run_assisted_trajectory
from long_context_agent.adaptive_evidence import EvidenceWindow,_messages

PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aB1kAAAAASUVORK5CYII=')


class Client:
    context_tokens=16384;remaining_tokens=65536;remaining_calls=8
    def __init__(self):self.ledger=[];self.requests=[]
    def count_messages(self,m):
        return sum(len(p.get('text',''))//4+100 for x in m for p in (x['content'] if isinstance(x['content'],list) else [{'text':x['content']}]))
    def can_call(self,m,max_tokens=1024):return self.count_messages(m)+max_tokens<=min(self.context_tokens,self.remaining_tokens)
    def complete(self,m,max_tokens=1024):
        self.requests.append(m);self.ledger.append({});self.remaining_calls-=1
        return '\\boxed{native answer}'


def setup(tmp_path):
    store=TrajectoryStore(tmp_path)
    states=[]
    for i in range(5):
        # Different bytes make images distinguishable without any model decoding.
        (tmp_path/f'{i}.png').write_bytes(PNG+bytes([i]))
        states.append({'accessibility_tree':f'observed {i}','url':'https://example.test','screenshot':f'{i}.png','action':f'click({i})'})
    store.insert({'id':'t','states':states});store.finalize()
    return store


def test_balanced_box_parser_keeps_explicit_native_answer():
    assert boxed_answer('explanation \\boxed{a{b}} trailing')=='a{b}'
    assert boxed_answer('\\boxed{partial') is None
    assert boxed_answer('\\boxed{UNKNOWN}')=='UNKNOWN'


def test_review_payload_is_identical_to_baseline(tmp_path):
    store=setup(tmp_path);raw=Client();a=AssistEvidenceClient(raw,store,domain_system='official',question_text='after click?')
    base=TrajectoryEvidenceClient(raw,store,domain_system='official',question_text='after click?')
    w=EvidenceWindow(0,len(store.context),store.context,'q',1);m=_messages('question',[w])
    assert a.prepare(m)==base.prepare(m)


def test_final_supplements_preserve_original_text_images_and_question_order(tmp_path):
    store=setup(tmp_path);raw=Client();a=AssistEvidenceClient(raw,store,domain_system='official',question_text='after click?')
    a.supported_pairs=[(0,len(store.context))]
    w=EvidenceWindow(0,len(store.context),store.context,'q',1);m=_messages('question',[w],final=True)
    base,base_images=TrajectoryEvidenceClient(raw,store,domain_system='official',question_text='after click?').prepare(m)
    actual,images=a.prepare(m)
    assert images[:len(base_images)]==base_images and len(images)>len(base_images)
    original_parts=base[-1]['content']
    assert all(p in actual[-1]['content'] for p in original_parts)
    assert actual[-1]['content'][-1]==original_parts[-1]


def test_no_budget_for_augmentation_keeps_fitting_baseline(tmp_path):
    store=setup(tmp_path);raw=Client();a=AssistEvidenceClient(raw,store,domain_system='official',question_text='after click?')
    a.supported_pairs=[(0,len(store.context))]
    w=EvidenceWindow(0,len(store.context),store.context,'q',1);m=_messages('question',[w],final=True)
    base,imgs=TrajectoryEvidenceClient(raw,store,domain_system='official',question_text='after click?').prepare(m)
    raw.context_tokens=raw.count_messages(base)+1024;a.client.context_tokens=raw.context_tokens
    actual,actual_imgs=a.prepare(m)
    assert actual==base and actual_imgs==imgs


def test_valid_native_box_is_not_replaced_by_a_confidence_gate(tmp_path,monkeypatch):
    import long_context_agent.trajectory_memory_assist as mod
    store=setup(tmp_path);raw=Client();client=AssistEvidenceClient(raw,store,domain_system='official')
    monkeypatch.setattr(mod,'retrieve_trajectory_evidence',lambda *args:{'answer':'\\boxed{valid answer}'})
    result=run_assisted_trajectory(store,'q',client)
    assert result['answer']=='\\boxed{valid answer}' and result['repair_calls']==0 and not raw.requests


def test_invalid_repair_keeps_original_and_does_not_force_unknown(tmp_path,monkeypatch):
    import long_context_agent.trajectory_memory_assist as mod
    store=setup(tmp_path);raw=Client();client=AssistEvidenceClient(raw,store,domain_system='official',question_text='q')
    original='An unfinished native explanation '*30
    client.final_messages=_messages('q',[EvidenceWindow(0,len(store.context),store.context,'q',1)],final=True)
    monkeypatch.setattr(mod,'retrieve_trajectory_evidence',lambda *args:{'answer':original})
    monkeypatch.setattr(raw,'complete',lambda *args,**kwargs:'unusable unboxed repair')
    result=run_assisted_trajectory(store,'q',client)
    assert result['answer']==original and result['format_recovery']=='invalid_repair_kept_original'
    assert result['repair_calls']==1 and client._repairing is False
