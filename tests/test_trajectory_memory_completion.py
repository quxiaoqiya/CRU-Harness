import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from long_context_agent.experiment_budget import ExperimentBudget
from long_context_agent.trajectory_memory_completion import PerCallBudgetedClient,CompletionEvidenceClient,CompletionConfig,run_completion_harness
from long_context_agent.trajectory_memory import TrajectoryStore,TrajectoryEvidenceClient
from long_context_agent.adaptive_evidence import EvidenceWindow,_messages


class Transport:
 model='test'
 def __init__(self):self.counted=[];self.sent=[]
 def count(self,**kwargs):self.counted.append(kwargs);return 100
 def request(self,path,body):
  self.sent.append(body)
  return {'usage':{'prompt_tokens':100,'completion_tokens':5},'choices':[{'message':{'content':'\\boxed{native}'},'finish_reason':'stop'}]}


def store(tmp_path):
 s=TrajectoryStore(tmp_path);s.insert({'id':'t','states':[{'accessibility_tree':"link 'Observed label'"}]});s.finalize();return s


def test_request_notice_tracks_actual_output_and_matches_counted_generation():
 t=Transport();c=PerCallBudgetedClient(t,ExperimentBudget(16384,65536,8,4096))
 m=[{'role':'system','content':'policy'},{'role':'user','content':'question'}]
 assert c.can_call(m,1024)
 assert 'maximum output per request: 1024 tokens.' in t.counted[-1]['messages'][0]['content']
 c.complete(m,4096)
 assert 'maximum output per request: 4096 tokens.' in t.sent[-1]['messages'][0]['content']
 assert t.counted[-1]['messages']==t.sent[-1]['messages'] and t.sent[-1]['max_tokens']==4096


def test_native_final_keeps_baseline_evidence_and_reasoning_policy(tmp_path):
 s=store(tmp_path);raw=PerCallBudgetedClient(Transport(),ExperimentBudget(16384,65536,8,4096))
 c=CompletionEvidenceClient(raw,s,domain_system='original policy',question_text='q')
 w=EvidenceWindow(0,len(s.context),s.context,'q',1);messages=_messages('q',[w],final=True)
 assert c.prepare(messages)==TrajectoryEvidenceClient(raw,s,domain_system='original policy',question_text='q').prepare(messages)
 c.complete(messages,1024)
 assert raw.ledger[-1]['requested_output_tokens']==4096
 assert c.output_allocations[-1]['baseline_evidence_unchanged']


def test_output_expansion_never_removes_context_to_fit(tmp_path,monkeypatch):
 s=store(tmp_path);raw=PerCallBudgetedClient(Transport(),ExperimentBudget(16384,65536,8,4096))
 c=CompletionEvidenceClient(raw,s,domain_system='policy',question_text='q')
 monkeypatch.setattr(raw,'can_call',lambda m,max_tokens=1024: max_tokens<=1024)
 w=EvidenceWindow(0,len(s.context),s.context,'q',1);c.complete(_messages('q',[w],final=True))
 assert raw.ledger[-1]['requested_output_tokens']==1024


def test_valid_native_answer_is_preserved_without_a_second_judgment(tmp_path,monkeypatch):
 import long_context_agent.trajectory_memory_completion as mod
 s=store(tmp_path);raw=PerCallBudgetedClient(Transport(),ExperimentBudget(16384,65536,8,4096));c=CompletionEvidenceClient(raw,s,domain_system='policy')
 monkeypatch.setattr(mod,'retrieve_trajectory_evidence',lambda *a:{'answer':'\\boxed{correct native answer}'})
 result=run_completion_harness(s,'q',c)
 assert result['answer']=='\\boxed{correct native answer}' and result['recovery_calls']==0
 assert result['native_non_unknown_box_preserved'] is True and not raw.ledger


def test_refusal_rescue_quote_cannot_come_from_question_or_agent_thought(tmp_path):
 s=store(tmp_path);c=CompletionEvidenceClient(SimpleNamespace(),s,domain_system='policy')
 images=[{'trajectory_id':'t','state_index':0}]
 assert c.observed_quote('\\boxed{x}\nEvidence: "Observed label"',images)
 assert not c.observed_quote('\\boxed{x}\nEvidence: "Invented label"',images)
 assert not c.observed_quote('\\boxed{x}\nEvidence: "Observed label"',[{'query_image':True}])
