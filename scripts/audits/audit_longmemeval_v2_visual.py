"""Replay frozen local visual holdout records without provider/tokenizer I/O."""
from pathlib import Path
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts import benchmark_longmemeval_v2_visual as b
from scripts.audits.audit_longmemeval_v2 import RecordedTransport,ledger_check
from long_context_agent.experiment_budget import BudgetedClient
from long_context_agent.trajectory_memory import TrajectoryEvidenceClient,retrieve_trajectory_evidence
from long_context_agent.trajectory_memory_v2 import VisualStateIndex,run_visual_harness
from long_context_agent.visual_recall import PerceptualVisualIndex,route_for_query
WORK=ROOT/'work/longmemeval_v2_visual_holdout_20261103'


def main():
 start=time.monotonic();w=WORK;o=w/'formal';f=b.check(w)
 paths=[b.record_path(o,q,a) for q in f['question_ids'] for a in f['arms']]
 assert len(paths)==f['expected_terminal_records'] and all(p.exists() for p in paths)
 access=json.loads((o/'scoring_access.json').read_text());a=json.loads((o/'audit.json').read_text())
 labels=json.loads((w/'private_labels/labels.json').read_text());qmap={q['id']:q for q in b.read_rows(w/'inputs.jsonl')}
 stores,_=b.load_stores(w);indexes={};policy=b.policies(w);m=b.official_scorer(w)
 totals={'generations':0,'tokens':0,'images':0,'counts_replayed':0};receipts=[];failures=[]
 scored={(r['id'],r['arm']):r for r in a['scored']}
 for path in paths:
  r=json.loads(path.read_text());qid,arm=r['id'],r['arm'];q=qmap[qid];store=stores[qid]
  assert r['terminal'] and r['freeze_sha256']==b.digest(o/'freeze.json')
  assert access['terminal_record_sha256'][path.name]==b.digest(path) and r['completed_at_unix']<=access['opened_at_unix']
  stats=ledger_check(r['ledger'],b.BUDGETS[arm]);assert r['usage']['within_budget']
  assert r['usage']['charged_tokens']==stats['charged_or_reserved_tokens'] and r['source_metadata']==store.metadata()
  assert all(x['actual_model']==b.MODEL for x in r['response_events'])
  for event in r['result'].get('image_events',[]):
   for im in event['images']:
    if im.get('query_image'):assert b.digest(w/'data'/q['image'])==im['image_sha256']
    else:
     state=next(x for x in store.bindings if x.trajectory_id==im['trajectory_id'] and x.state_index==im['state_index'])
     assert list((state.start,state.end))==im['source_state_span']
     assert b.digest(store.resolve_image(state.image))==im['image_sha256']
    totals['images']+=1
   for card in event.get('evidence_receipts',[]):
    state=store._rows[card['trajectory_id']]['states'][card['state_index']]
    field=str(state.get(card['field']) or '')
    assert b.text_digest(field)==card['field_sha256']
    assert [b.text_digest(field[x:y]) for x,y in card['field_ranges']]==card['excerpt_sha256']
  replay=True;reason=None
  try:
   transport=RecordedTransport(r);client=BudgetedClient(transport,b.BUDGETS[arm])
   if arm=='harness_visual_route' and q['image']:
    features=json.loads((w/'visual_features.json').read_text())
    assert b.digest(w/'visual_features.json')==f['visual_features_sha256']
    index=PerceptualVisualIndex(store,features,q['image'])
    result=run_visual_harness(index,q['question'],client,domain_system=policy[q['domain']],query_image=q['image'])
    result['visual_match_events']=index.visual_trace
    result['visual_feature_cache_sha256']=f['visual_features_sha256']
    assert r['route']==route_for_query(q['image']) and result==r['result'],'Perceptual route mismatch'
   elif arm=='harness_v2':
    key=id(store)
    if key not in indexes:indexes[key]=VisualStateIndex(store)
    result=run_visual_harness(indexes[key],q['question'],client,domain_system=policy[q['domain']],query_image=q['image'])
    assert result==r['result'],'Visual controller mismatch'
   else:
    pol=policy[q['domain']]
    if arm=='direct_headtail_native_concise':pol+=' Return only the shortest final answer in \\boxed{...}. No analysis or commentary. If evidence is insufficient use \\boxed{UNKNOWN}; do not guess.'
    adapted=TrajectoryEvidenceClient(client,store,domain_system=pol,query_image=q['image'],
      history_image_limit=(len(store.bindings) if arm=='direct_full' else 16-int(bool(q['image'])) if arm.startswith('direct_headtail_native') else None),
      image_selection='uniform' if arm.startswith('direct_headtail_native') else 'overlap',question_text=q['question'])
    if arm in ('harness','harness_visual_route'):assert retrieve_trajectory_evidence(store,q['question'],adapted)==r['result']
    elif arm=='direct_full' and r['status']=='native_capacity_denied':assert not r['ledger'] and r['source_text_tokens_exact']>262144+4096
    else:
     retained=min(len(store.context),int(len(store.context)*240000/r['source_text_tokens_exact']))
     for _ in range(16):
      windows=b.headtail_windows(store,retained);messages=b._messages(q['question'],windows,final=True)
      adapted.count_messages(messages)
      if adapted.can_call(messages,1024):break
      retained=max(2,int(retained*.9))
     assert adapted.complete(messages,1024)==r['answer']
     assert adapted.image_events==r['result']['image_events']
   assert transport.count_pos==len(r['count_events']) and transport.request_pos==len(r['ledger'])
   totals['counts_replayed']+=transport.count_pos
  except (AssertionError,ValueError,LookupError,IndexError) as exc:
   replay=False;reason=type(exc).__name__+': '+str(exc)[:80];failures.append({'id':qid,'arm':arm,'reason':reason})
  row=scored[qid,arm];label=labels[qid];parsed=m.extract_boxed_answer(r['answer']);assert parsed==row['parsed_answer']
  expected=False
  if r['status']=='returned' and r['answer'].strip() and not m.is_unknown(parsed):
   name=m.eval_name(label['eval_function'])
   if name.startswith('llm_'):
    ledger_check(row['judge_ledger'],b.JUDGE_BUDGET)
    if row['judge_error'] is not None:expected=None
    else:expected=m._parse_llm_binary_judgement(row['judge_ledger'][-1]['content'])[0]==1
   else:expected=bool(m.eval_from_spec(label['eval_function'],parsed,label['answer']))
  assert row['correct'] is expected
  totals['generations']+=stats['calls'];totals['tokens']+=stats['charged_or_reserved_tokens']
  receipts.append({'id':qid,'arm':arm,'record_sha256':b.digest(path),'ledger_passed':True,'score_recomputed':True,'replay_passed':replay})
 report={'status':'pass' if not failures else 'partial_replay','terminal_records':len(paths),
   'freeze_sha256':b.digest(o/'freeze.json'),'auditor_sha256':b.digest(__file__),
   'ledgers_passed':len(receipts),'scores_recomputed':len(receipts),'replay_passed':sum(r['replay_passed'] for r in receipts),
   'failures':failures,'totals':totals,'receipts':receipts,'provider_or_tokenizer_calls_made':0,
   'independent_semantic_entailment_validation':False,'seconds':time.monotonic()-start}
 b.atomic(w/'offline_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['receipts']},indent=2))


if __name__=='__main__':main()
