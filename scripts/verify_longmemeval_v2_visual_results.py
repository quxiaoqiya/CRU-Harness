"""Offline public integrity/metric checks. Does not semantically grade answers."""
from collections import Counter
import hashlib,json,math
from pathlib import Path
import re
ROOT=Path(__file__).resolve().parents[1]
ID='longmemeval_v2_visual_holdout_20261103'


def check(ok,message):
 if not ok:raise ValueError(message)

def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def scan(v):
 if isinstance(v,dict):
  check(not {'question','answer','gold','labels','content','trace','reasoning','api_key','headers','authorization'}&set(v),'Raw/private fields')
  for x in v.values():scan(x)
 elif isinstance(v,list):
  for x in v:scan(x)
 elif isinstance(v,str):
  check(not re.search(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}|/(?:Users|home|root|uniparser)/|data:image/',v),'Private material')

def ledger(items,budget):
 total=0
 check(len(items)<=budget['max_calls'],'Call ceiling')
 for number,e in enumerate(items,1):
  check(e['physical_request']==number,'Ledger order')
  prompt,out=e['admission_prompt_tokens'],e['requested_output_tokens']
  check(type(prompt) is int and prompt>=0 and type(out) is int and 0<out<=budget['max_output_tokens'],'Invalid admission')
  check(prompt+out<=budget['context_tokens'] and total+prompt+out<=budget['total_tokens'],'Reservation exceeded')
  if e['usage_known']:
   check(type(e['input_tokens']) is int and 0<=e['input_tokens']<=prompt,'Input usage')
   check(type(e['output_tokens']) is int and 0<=e['output_tokens']<=out,'Output usage')
   check(e['charged_tokens']==e['input_tokens']+e['output_tokens'],'Usage conservation')
  else:check(e['charged_tokens']==prompt+out,'Unknown usage reservation')
  total+=e['charged_tokens']
 check(total<=budget['total_tokens'],'Total ceiling')
 return total


def main():
 p=json.loads((ROOT/'benchmarks'/f'{ID}.json').read_text());a=json.loads((ROOT/'benchmarks'/f'{ID}_audit.json').read_text())
 exta=json.loads((ROOT/'benchmarks'/f'{ID}_route_audit.json').read_text())
 scan(p);scan(a);scan(exta)
 check(digest(ROOT/'benchmarks'/f'{ID}_route_audit.json')==p['route_offline_audit_sha256'],'Route audit commitment')
 check(exta['freeze_sha256']==p['route_freeze_sha256'],'Route freeze binding')
 check(p['route_registered_before_any_gold'] and p['route_registered_at_unix']<min(p['first_scoring_access_unix'],p['route_first_scoring_access_unix']),'Registration chronology')
 check(p['n_questions']==20 and len(set(p['sample_ids']))==20 and len(p['arms'])==6,'Scope')
 check(p['question_id_overlap_with_pilot']==p['question_text_hash_overlap_with_pilot']==0,'Question overlap')
 check(p['source_histories_reused_from_development'] is True and p['n_shared_histories']==2,'History disclosure')
 check(p['visual_features_integrity']['status']=='pass' and p['visual_features_integrity']['image_features_rederived']==5099,'Feature rederivation')
 check(p['visual_features_integrity']['features_sha256']==p['visual_features_sha256'],'Feature cache binding')
 check(digest(ROOT/'scripts/prepare_visual_features.py')==p['visual_features_integrity']['featurizer_sha256'],'Featurizer changed')
 check(p['official_leaderboard_submission'] is False and p['independent_semantic_validation'] is False,'Evaluation scope')
 check(p['model']=='Qwen3.8-Flash-Next-FP8' and p['data_revision']=='f152293e235517d504809563c833d7190b8c713b','Model/data pin')
 check(p['upstream_code_commit']=='2cc8c540bdb87fe6761629b585e727e1c4704520','Code pin')
 check(digest(ROOT/'benchmarks'/f'{ID}_audit.json')==p['offline_audit_sha256'],'Audit commitment')
 check(digest(ROOT/'scripts/audits/audit_longmemeval_v2_visual.py')==p['auditor_sha256'],'Auditor fingerprint')
 for path,sha in p['code_sha256'].items():check(digest(ROOT/path)==sha,'Frozen source changed: '+path)
 check(a['freeze_sha256']==p['freeze_sha256'] and a['terminal_records']==a['ledgers_passed']==a['scores_recomputed']==100,'Audit cardinality')
 check(a['provider_or_tokenizer_calls_made']==0 and a['independent_semantic_entailment_validation'] is False,'Replay scope')
 check(a['replay_passed']==sum(r['replay_passed'] for r in a['receipts']),'Replay denominator')
 check(exta['terminal_records']==exta['ledgers_passed']==exta['scores_recomputed']==20,'Route audit cardinality')
 check(exta['provider_or_tokenizer_calls_made']==0,'Route replay performed I/O')
 rows=p['results'];by={(r['id'],r['arm']):r for r in rows}
 check(len(rows)==len(by)==120 and set(by)=={(q,arm) for q in p['sample_ids'] for arm in p['arms']},'Missing/duplicate outcomes')
 receipts={(r['id'],r['arm']):r for r in a['receipts']+exta['receipts']}
 for r in rows:
  check(r['model']==p['model'] and all(x==p['model'] for x in r['actual_models']),'Actual model')
  check(type(r['correct']) is bool and r['judge_error'] is None,'Unevaluable result')
  check(r['completed_at_unix']<=(p['route_first_scoring_access_unix'] if r['arm']=='harness_visual_route' else p['first_scoring_access_unix']) and p['frozen_at_unix']<p['first_scoring_access_unix'],'Gold access order')
  check(r['budget']==p['budgets'][r['arm']],'Budget differs')
  check(ledger(r['ledger'],r['budget'])==r['tokens'] and len(r['ledger'])==r['calls'],'Row usage')
  ledger(r['judge_ledger'],p['judge_budget'])
  check(receipts[r['id'],r['arm']]['record_sha256']==r['record_sha256'],'Private record binding')
  if r['arm']=='direct_full':check(r['status']=='native_capacity_denied' and not r['calls'] and r['source_text_tokens']>262144+4096,'Full direct admission')
  else:check(r['status']=='returned','Generation delivery failure')
  if r['arm']=='harness_v2':check(r['final_support_not_in_final']==0,'Final support omitted')
 for arm in p['arms']:
  rr=[r for r in rows if r['arm']==arm];s=p['summary'][arm];answered=[r for r in rr if r['non_unknown_answer']]
  correct=sum(r['correct'] for r in rr)
  check(s['n']==20 and s['correct']==correct and s['task_accuracy']==correct/20,'Overall accuracy')
  check(s['answered']==len(answered) and s['coverage']==len(answered)/20,'Coverage')
  check(s['selective_accuracy']==(correct/len(answered) if answered else None),'Selective accuracy')
  check(s['wrong_answered']==sum(not r['correct'] for r in answered),'Wrong-answer count')
  check(s['tokens']==sum(r['tokens'] for r in rr) and s['calls']==sum(r['calls'] for r in rr),'Arm usage')
  check(s['unknown']==sum(r['unknown'] for r in rr),'UNKNOWN count')
  for prefix,cond in [('deterministic',lambda r:not r['scoring_function'].startswith('llm_')),
                      ('judge',lambda r:r['scoring_function'].startswith('llm_')),
                      ('image',lambda r:r['question_image'])]:
   sub=[r for r in rr if cond(r)]
   check(s[prefix+'_n']==len(sub) and s[prefix+'_correct']==sum(r['correct'] for r in sub),'Subset arithmetic')
  for ability,sub in s['ability'].items():
   group=[r for r in rr if r['ability']==ability]
   check(sub=={'n':len(group),'correct':sum(r['correct'] for r in group)},'Ability arithmetic')
  if arm!='harness_v2':
   pairs=[(by[q,arm]['correct'],by[q,'harness_v2']['correct']) for q in p['sample_ids']]
   check(p['paired'][arm]==dict(zip(('both_correct','new_only_correct','control_only_correct','neither_correct'),
    (pairs.count(x) for x in [(True,True),(False,True),(True,False),(False,False)]))),'Paired arithmetic')
 legacy=[(by[q,'harness'],by[q,'harness_visual_route']) for q in p['sample_ids'] if not by[q,'harness']['question_image']]
 check(len(legacy)==p['legacy_branch_comparison']['n']==16,'Legacy route cardinality')
 for field,label in [('answer_sha256','answer_hash_matches'),('parsed_answer_sha256','parsed_answer_hash_matches')]:
  check(p['legacy_branch_comparison'][label]==sum(a[field]==b[field] for a,b in legacy),'Legacy answer hash comparison')
 check(p['legacy_branch_comparison']['first_prompt_hash_matches']==sum(a['ledger'][0]['prompt_sha256']==b['ledger'][0]['prompt_sha256'] for a,b in legacy),'Legacy first prompt')
 visual=[(by[q,'harness'],by[q,'harness_visual_route']) for q in p['sample_ids'] if by[q,'harness']['question_image']]
 check(p['visual_route_tradeoff']=={'n':4,'old_correct_to_unknown':sum(a['correct'] is True and b['unknown'] for a,b in visual),
  'old_incorrect_to_correct':sum(a['correct'] is False and b['correct'] is True for a,b in visual),
  'old_incorrect_to_unknown':sum(a['correct'] is False and b['unknown'] for a,b in visual)},'Visual tradeoff')
 check(p['all120_fixed_before_any_scoring_access']==all(r['completed_at_unix']<=min(p['first_scoring_access_unix'],p['route_first_scoring_access_unix']) for r in rows),'Global scoring chronology')
 check(a['totals']['tokens']+exta['totals']['tokens']==sum(r['tokens'] for r in rows) and a['totals']['generations']+exta['totals']['generations']==sum(r['calls'] for r in rows),'Audit totals')
 print(json.dumps({'status':'PASS','outcomes':120,'source_fingerprints':len(p['code_sha256']),
                   'replays_passed':a['replay_passed']+exta['replay_passed'],'scope':'Public arithmetic, budget reservations and integrity; not semantic regrading'}))


if __name__=='__main__':main()
