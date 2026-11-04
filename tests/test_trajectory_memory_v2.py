import base64
from dataclasses import replace
import json
from pathlib import Path

from long_context_agent.trajectory_memory import TrajectoryStore
from long_context_agent.trajectory_memory_v2 import (VisualStateIndex, VisualConfig, assemble,
    evidence_gate, run_visual_harness)

PNG=base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+aB1kAAAAASUVORK5CYII=')


def fixture(tmp_path):
    (tmp_path/'a.png').write_bytes(PNG)
    store=TrajectoryStore(tmp_path)
    store.insert({'id':'t','goal':'Inspect panel','outcome':'success','states':[
        {'accessibility_tree':"button 'Open panel'",'action':None,'thought':'SECRET_FAKE_LABEL', 'screenshot':'a.png'},
        {'accessibility_tree':"heading 'Toolbox'\nlink 'Manage'\nlink 'Settings'",'action':"click('Open panel')",'screenshot':'a.png'},
        {'accessibility_tree':"heading 'Done'",'action':"click('Manage')",'screenshot':'a.png'}]})
    return VisualStateIndex(store)


def review(card):
    return {'answer':'two','decision':'supported','confidence':.9,'all_requirements_covered':True,
            'claims':[{'requirement_index':0,'fact':'labels','verdict':'support','citations':[card.id],
                       'quote':"link 'Settings'",'visual':False}], 'missing':[],'conflicts':[]}


def test_observation_index_excludes_agent_thoughts_and_binds_raw_field(tmp_path):
    idx=fixture(tmp_path)
    assert 'secret_fake_label' not in idx.postings
    card=idx.card(1,'Toolbox')
    raw=idx.states[1][3]
    assert all(raw[a:b]==x for (a,b),x in zip(card.ranges,card.excerpts))
    assert card.action=="click('Open panel')"
    assert idx.rank('Toolbox Settings')[0]==1


def test_transition_retrieval_pairs_before_and_after_without_wrong_trajectory(tmp_path):
    idx=fixture(tmp_path)
    selected=idx.retrieve(['Toolbox'],set(),temporal=True)
    assert {c.state_index for _,c in selected}=={0,1,2}
    assert len({i for i,_ in selected})==len(selected)
    assert not idx.retrieve(['Toolbox'],{0,1,2},temporal=True)


def test_gate_abstains_on_unobserved_missing_quote_or_uncovered_requirement(tmp_path):
    card=fixture(tmp_path).card(1,'Toolbox');r=review(card)
    assert evidence_gate(r,[card],set(),required_count=1)[0]
    r['missing']=['Inspect another page']
    assert not evidence_gate(r,[card],set())[0]
    r=review(card);r['claims'][0]['quote']='Imaginary UI'
    assert evidence_gate(r,[card],set())[1]=='quote_not_in_observation'
    r=review(card)
    assert evidence_gate(r,[card],set(),required_count=2)[1]=='missing_requirement_claim'
    r['claims'][0]['verdict']='unobserved'
    assert not evidence_gate(r,[card],set())[0]


def test_visual_citation_requires_original_image_in_this_call_and_false_premise_requires_opposition(tmp_path):
    card=fixture(tmp_path).card(1,'Toolbox');r=review(card)
    r['claims'][0].update(quote='',visual=True)
    assert evidence_gate(r,[card],set())[1]=='unseen_image'
    assert evidence_gate(r,[card],{card.id})[0]
    r['decision']='contradicted'
    assert evidence_gate(r,[card],{card.id})[1]=='premise_not_contradicted'
    r['claims'][0]['verdict']='oppose'
    assert evidence_gate(r,[card],{card.id})[0]


def test_image_count_and_generation_payload_has_original_bytes_and_state_receipts(tmp_path):
    idx=fixture(tmp_path);card=idx.card(1,'Toolbox')
    messages,images=assemble(idx.store,'How many?', 'Policy','Review',[card],query_image='a.png')
    parts=messages[-1]['content'];payload=[p['image_url']['url'] for p in parts if p['type']=='image_url']
    assert len(payload)==2 and all(base64.b64decode(p.split(',')[1])==PNG for p in payload)
    assert images[0]['id']==card.id and images[-1]['query_image']


def test_malformed_and_overconfident_unbound_answers_end_in_unknown(tmp_path):
    idx=fixture(tmp_path)
    class Client:
        context_tokens=16384
        remaining_tokens=65536
        remaining_calls=8
        def __init__(self):self.ledger=[]
        def can_call(self,*args):return True
        def complete(self,messages,max_tokens):
            self.ledger.append({});self.remaining_calls-=1;self.remaining_tokens-=500
            if len(self.ledger)==1:return json.dumps({'queries':['Toolbox'],'requirements':['Count links']})
            return json.dumps({'answer':'ninety','confidence':1,'decision':'supported','all_requirements_covered':True,
                               'claims':[{'verdict':'support','citations':['MADE_UP']}], 'missing':[]})
    result=run_visual_harness(idx,'Count links',Client(),domain_system='Never guess')
    assert result['answer']=='\\boxed{UNKNOWN}' and result['decision']=='abstain'
    assert result['support_ids_not_in_final']==[]


def test_partial_support_survives_missing_other_requirement(tmp_path):
    from long_context_agent.trajectory_memory_v2 import partial_support
    card=fixture(tmp_path).card(1,'Toolbox');r=review(card)
    r['missing']=['Find another page'];r['all_requirements_covered']=False
    assert not evidence_gate(r,[card],set())[0]
    assert partial_support(r,[card],set())==[card.id]


def test_premise_contradiction_can_cancel_dependent_requirement_but_missing_is_not_absence(tmp_path):
    card=fixture(tmp_path).card(1,'Toolbox');r=review(card)
    r.update(answer='The premise is false',decision='contradicted')
    r['claims'][0]['verdict']='oppose'
    r['claims'].append({**r['claims'][0],'requirement_index':1,'verdict':'inapplicable'})
    assert evidence_gate(r,[card],set(),required_count=2)[0]
    r['decision']='supported'
    assert not evidence_gate(r,[card],set(),required_count=2)[0]
    r['decision']='insufficient'
    assert not evidence_gate(r,[card],set(),required_count=2)[0]


def test_observed_url_is_valid_support_but_goal_is_not(tmp_path):
    idx=fixture(tmp_path);card=replace(idx.card(1,'Toolbox'),url='https://example.test/panel')
    r=review(card);r['claims'][0]['quote']=card.url
    assert evidence_gate(r,[card],set())[0]
    r['claims'][0]['quote']=card.goal
    assert not evidence_gate(r,[card],set())[0]
