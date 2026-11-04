"""Predeclared conservative route: image-content recall only for query images.

No-query-image questions execute the unchanged legacy Harness, with fresh calls.
This extension is frozen before reading held-out labels; original arms persist.
"""
from dataclasses import asdict
from pathlib import Path
import fcntl,json,os,random,shutil,subprocess,sys,time
from scripts import benchmark_longmemeval_v2_visual as b
from long_context_agent.experiment_budget import BudgetedClient,load_provider_env
from long_context_agent.trajectory_memory import TrajectoryEvidenceClient,retrieve_trajectory_evidence
from long_context_agent.trajectory_memory_v2 import run_visual_harness
from long_context_agent.visual_recall import PerceptualVisualIndex,route_for_query
ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'work/longmemeval_v2_visual_holdout_20261103'
WORK=ROOT/'work/longmemeval_v2_visual_route_20261103'
ARM='harness_visual_route'
b.ARMS=(*b.ARMS,ARM)
b.BUDGETS[ARM]=b.BUDGETS['harness_v2']
_original_hashes=b.code_hashes

def hashes():
 return {**_original_hashes(),**{p:b.digest(ROOT/p) for p in ['scripts/benchmark_longmemeval_v2_visual_route.py','scripts/prepare_visual_features.py','tests/test_visual_recall.py']}}
b.code_hashes=hashes


def prepare():
 if (BASE/'formal/scoring_access.json').exists():raise ValueError('Must register before held-out gold access')
 if WORK.exists():raise ValueError('An existing extension is immutable')
 WORK.mkdir()
 for name in ['data','upstream','selected_trajectories.jsonl','source_preflight_counts.json','visual_features.json']:
  (WORK/name).symlink_to((BASE/name).resolve(),target_is_directory=(BASE/name).is_dir())
 for name in ['inputs.jsonl','haystacks.json','assets_manifest.json']:shutil.copyfile(BASE/name,WORK/name)
 (WORK/'private_labels').mkdir();shutil.copyfile(BASE/'private_labels/labels.json',WORK/'private_labels/labels.json');(WORK/'private_labels/labels.json').chmod(0o600)
 selection=json.loads((BASE/'selection_manifest.json').read_text());selection['arms']=[ARM]
 selection['predeclared_route']='query image -> perceptual state retrieval + evidence gate; no query image -> unchanged legacy controller; all20 fresh generations'
 (WORK/'selection_manifest.json').write_text(json.dumps(selection,indent=2)+'\n')
 b.freeze(WORK)
 freeze_path=WORK/'formal/freeze.json';f=json.loads(freeze_path.read_text())
 f.update(visual_features_sha256=b.digest(WORK/'visual_features.json'),
   registered_before_any_holdout_gold_access=True,base_freeze_sha256=b.digest(BASE/'formal/freeze.json'),
   routing_rule=selection['predeclared_route'],fresh_calls_for_all20=True,
   isolated_feature_environment='Pillow11.3.0; base inference environment unchanged',
   feature_algorithm='dHash horizontal64+vertical64+RGB8x8; layout heuristic, not semantic embedding')
 b.atomic(freeze_path,f)
 profile=WORK/'formal/labels_guard.sb'
 profile.write_text(profile.read_text()+'(deny file-read* (subpath '+json.dumps(str((BASE/'formal').resolve()))+'))\n')
 f['guard_profile_sha256']=b.digest(profile);b.atomic(freeze_path,f)
 print(json.dumps({'registered':True,'new_arm':ARM,'fresh_questions':20,'image_branch':4,'legacy_branch':16,'freeze_sha256':b.digest(freeze_path)}),flush=True)


def check():
 f=b.check(WORK)
 if b.digest(WORK/'visual_features.json')!=f['visual_features_sha256']:raise ValueError('Feature cache changed')
 return f


def worker(provider):
 f=check();b.check(WORK,worker=True);out=WORK/'formal';sha=b.digest(out/'freeze.json')
 with (out/'.run.lock').open('a+') as lock:
  fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
  stores,insertion=b.load_stores(WORK);b.atomic(out/'insertion_audit.json',insertion)
  features=json.loads((WORK/'visual_features.json').read_text());assets=json.loads((WORK/'assets_manifest.json').read_text())
  expected={**assets['history_images'],**assets['query_images']}
  if {r['path']:r['image_sha256'] for r in features['features']}!=expected:raise ValueError('Feature/source binding changed')
  for name,expected_sha in expected.items():
   if b.digest(WORK/'data'/name)!=expected_sha:raise ValueError('Original image changed')
  counts={sha:r['canonical_text_tokens'] for sha,r in json.loads((WORK/'source_preflight_counts.json').read_text())['counts'].items()}
  policies=b.policies(WORK);transport=b.AuditedTransport(**provider,timeout=360)
  jobs=b.read_rows(WORK/'inputs.jsonl');random.Random(b.SEED).shuffle(jobs)
  for q in jobs:
   target=b.record_path(out,q['id'],ARM)
   if target.exists():continue
   if (out/'scoring_access.json').exists():raise ValueError('No generation after extension gold access')
   store=stores[q['id']];journal=out/'journals'/target.name
   c0,r0,e0=len(transport.count_events),len(transport.response_events),len(transport.error_events)
   def save(ledger):b.atomic(journal,{'ledger':ledger,'count_events':transport.count_events[c0:],
      'response_events':transport.response_events[r0:],'transport_errors':transport.error_events[e0:]})
   client=BudgetedClient(transport,b.BUDGETS[ARM],on_ledger_update=save)
   started=time.perf_counter();result={};answer='';error=None
   if journal.exists():
    client.ledger=json.loads(journal.read_text())['ledger'];status='interrupted';error={'type':'InterruptedTaskNotReplayed'}
   else:
    save([])
    try:
     if q['image']:
      index=PerceptualVisualIndex(store,features,q['image'])
      result=run_visual_harness(index,q['question'],client,domain_system=policies[q['domain']],query_image=q['image'])
      result['visual_match_events']=index.visual_trace
      result['visual_feature_cache_sha256']=f['visual_features_sha256']
     else:
      adapted=TrajectoryEvidenceClient(client,store,domain_system=policies[q['domain']],question_text=q['question'])
      result=retrieve_trajectory_evidence(store,q['question'],adapted)
     answer=result['answer'];status='returned' if answer.strip() else 'empty'
    except Exception as exc:status='error';error={'type':type(exc).__name__}
   record={'id':q['id'],'arm':ARM,'domain':q['domain'],'question_image':bool(q['image']),
    'route':route_for_query(q['image']),'model':b.MODEL,'freeze_sha256':sha,'terminal':True,
    'status':status,'answer':answer,'result':result,'error':error,'source_metadata':store.metadata(),
    'source_text_tokens_exact':counts[store.metadata()['canonical_source_sha256']],
    'ledger':client.ledger,'usage':client.summary(),'count_events':transport.count_events[c0:],
    'response_events':transport.response_events[r0:],'transport_errors':transport.error_events[e0:],
    'seconds':time.perf_counter()-started,'completed_at_unix':time.time()}
   b.atomic(target,record)
   print(json.dumps({'extension':q['id'],'route':record['route'],'status':status,'tokens':record['usage']['charged_tokens']}),flush=True)
  return {'terminal_records':len(jobs),'all_fresh':True}


def main():
 if '--worker' in sys.argv:
  result=worker(json.loads(sys.stdin.read())['provider'])
 else:
  phase=sys.argv[1]
  if phase=='prepare':prepare();return
  provider=load_provider_env(sys.argv[2] if len(sys.argv)>2 else None)
  if phase=='run':
   check();env={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['KEY','TOKEN','PASSWORD','SECRET'])}
   result=subprocess.run(['/usr/bin/sandbox-exec','-f',str(WORK/'formal/labels_guard.sb'),sys.executable,
     '-m','scripts.benchmark_longmemeval_v2_visual_route','--worker'],input=json.dumps({'provider':provider}),
     text=True,cwd=ROOT,env=env)
   if result.returncode:raise RuntimeError('Extension worker failed; do not resend uncertain generations')
   return
  if phase=='score':check();result=b.score(WORK,provider)
  else:raise ValueError('Use prepare, run, score')
 print(json.dumps(result,indent=2))


if __name__=='__main__':main()
