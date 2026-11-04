"""Stdlib-only public arithmetic, budget and frozen-source verification."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import re

ROOT=Path(__file__).resolve().parents[1]
ID='longmemeval_v2_visual_anchor_20261103'


def check(ok,message):
    if not ok:raise ValueError(message)


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scan(value):
    if isinstance(value,dict):
        check(not {'question','answer','gold','labels','content','trace','reasoning','api_key','authorization','headers'}&set(value),'Private content key')
        for v in value.values():scan(v)
    elif isinstance(value,list):
        for v in value:scan(v)
    elif isinstance(value,str):
        check(not re.search(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}|/(?:Users|home|root|uniparser)/|data:image/',value),'Private content')


def ledger(entries,budget):
    total=0
    check(len(entries)<=budget['max_calls'],'Call ceiling')
    for n,e in enumerate(entries,1):
        prompt,out=e['admission_prompt_tokens'],e['requested_output_tokens']
        check(e['physical_request']==n and type(prompt)is int and prompt>=0,'Request ordering')
        check(type(out)is int and 0<out<=budget['max_output_tokens'],'Output ceiling')
        check(prompt+out<=budget['context_tokens'] and total+prompt+out<=budget['total_tokens'],'Reservation')
        if e['usage_known']:
            check(0<=e['input_tokens']<=prompt and 0<=e['output_tokens']<=out,'Usage')
            check(e['charged_tokens']==e['input_tokens']+e['output_tokens'],'Usage conservation')
        else:check(e['charged_tokens']==prompt+out,'Uncertain reservation')
        total+=e['charged_tokens']
    check(total<=budget['total_tokens'],'Task ceiling')
    return total


def main():
    p=json.loads((ROOT/'benchmarks'/f'{ID}.json').read_text());scan(p)
    check(p['model']=='Qwen3.8-Flash-Next-FP8' and p['tests_passed']==56,'Model/tests')
    check(p['data_revision']=='f152293e235517d504809563c833d7190b8c713b'
          and p['upstream_code_commit']=='2cc8c540bdb87fe6761629b585e727e1c4704520','Upstream pins')
    check(p['source_histories_reused'] is True and p['n_shared_histories']==2
          and p['n_trajectories']==200 and p['n_history_images']==5095,'Source scope')
    check(p['official_leaderboard_submission'] is False and p['official_LAFS_score'] is None
          and p['independent_semantic_validation'] is False,'Scoring scope')
    check(p['remote_inference_files_modified'] is False and p['github_push_during_benchmark'] is False,'Local evaluation scope')
    prior=set()
    for cohort in ['qwen','visual_holdout','compact_holdout']:
        prior.update(json.loads((ROOT/'benchmarks'/f'longmemeval_v2_{cohort}_20261103.json').read_text())['sample_ids'])
    check(len(prior)==60,'Prior cohort count')
    check(p['selected_holdout_candidate_version']==1,'Development-selected candidate')
    for stage,n,expected in [('preliminary_development',12,36),('development',12,36),('holdout',8,32)]:
        e=p['experiments'][stage];rows=e['results'];by={(r['id'],r['arm']):r for r in rows}
        check(e['n_questions']==n and len(set(e['sample_ids']))==n
              and e['outcomes']==len(rows)==len(by)==expected,'Evaluation units')
        check(set(by)=={(q,a) for q in e['sample_ids'] for a in e['arms']},'Outcome coverage')
        if stage=='holdout':
            check(not set(e['sample_ids'])&prior
                  and e['question_id_overlap_with_prior60']==e['question_text_hash_overlap_with_prior60']==0,'Holdout exclusion')
            check(e['visual_contracts_version']==1,'Preselected holdout candidate')
        else:check(set(e['sample_ids'])<=prior,'Development disclosure')
        adapter=e['posthoc_scoring_metadata_adapter']
        if stage=='preliminary_development':
            check(adapter['adapter_sha256']==digest(ROOT/'scripts/audits/score_longmemeval_v2_visual_anchor_dev.py')
                  and adapter['inference_records_changed'] is False
                  and adapter['reference_answers_or_judge_prompts_changed'] is False
                  and adapter['completed_judge_requests_reused'] is True,'Metadata-only scorer recovery')
        else:check(adapter is None,'Stage scored without posthoc metadata adaptation')
        audit_path=ROOT/'benchmarks'/f'{ID}_{stage}_audit.json';audit=json.loads(audit_path.read_text());scan(audit)
        check(digest(audit_path)==e['offline_audit_sha256'] and audit['freeze_sha256']==e['freeze_sha256'],'Audit binding')
        check(audit['auditor_sha256']==e['auditor_sha256']==digest(ROOT/'scripts/audits/audit_longmemeval_v2_visual_anchor.py'),'Auditor fingerprint')
        check(audit['status']=='pass' and not audit['failures'] and audit['terminal_records']==audit['ledgers_passed']
              ==audit['scores_recomputed']==audit['replay_passed']==e['replay_passed']==expected,'Audit cardinality')
        check(audit['provider_or_tokenizer_calls_made']==0
              and audit['independent_semantic_entailment_validation'] is False,'Replay scope')
        for path,sha in e['code_sha256'].items():check(digest(ROOT/path)==sha,'Source changed: '+path)
        receipts={(r['id'],r['arm']):r for r in audit['receipts']}
        check(set(receipts)==set(by),'Receipt units')
        check(len({r['source_sha256'] for r in rows})==2
              and {r['source_text_tokens'] for r in rows}=={31476162,34235595},'Source identity/counts')
        for r in rows:
            check(r['question_image'] is True and type(r['correct'])is bool
                  and r['judge_error'] is None and r['status']=='returned','Visual scored outcome')
            check(r['model']==p['model'] and r['actual_models']==[p['model']],'Actual model')
            check(bool(re.fullmatch('[0-9a-f]{64}',r['query_image_sha256'])),'Query-image commitment')
            check(e['frozen_at_unix']<r['completed_at_unix']<e['first_scoring_access_unix']
                  and e['all_outcomes_fixed_before_scoring'] is True,'Scoring chronology')
            check(r['budget']==e['budgets'][r['arm']] and r['within_budget'],'Budget identity')
            expected_budget=(262144,262144,1) if r['arm']=='direct_headtail_native' else (16384,65536,8)
            check(tuple(r['budget'][k] for k in ['context_tokens','total_tokens','max_calls'])==expected_budget
                  and r['budget']['max_output_tokens']==4096,'Budget protocol')
            check(ledger(r['ledger'],r['budget'])==r['tokens'] and len(r['ledger'])==r['calls'],'Usage binding')
            ledger(r['judge_ledger'],e['judge_budget'])
            check(receipts[r['id'],r['arm']]['record_sha256']==r['record_sha256']
                  and receipts[r['id'],r['arm']]['replay_passed'],'Record binding')
            check(not r['pinned_omitted'] and set(r['pinned_evidence_ids'])<=set(r['final_evidence_ids']),'Pinned retention')
            if r['arm']=='visual_query_only':
                check(r['calls']==2 and not r['history_image_occurrences'] and not r['transition_pairs']
                      and not r['final_evidence_ids'],'No-memory control')
            if r['arm'] in ['visual_query_only','harness_visual_anchor']:
                check(r['no_confidence_gate'] is True and r['query_image_occurrences']==r['calls'],'Image-route invariants')
        for arm,s in e['summary'].items():
            rr=[r for r in rows if r['arm']==arm]
            check(s['n']==n and s['correct']==sum(r['correct'] for r in rr)
                  and s['accuracy']==s['correct']/n,'Accuracy')
            for key,calc in [('unknown',sum(r['unknown'] for r in rr)),('tokens',sum(r['tokens'] for r in rr)),
                             ('calls',sum(r['calls'] for r in rr)),('transition_cases',sum(r['transition_pairs']>0 for r in rr)),
                             ('perception_errors',sum(r['perception_parse_error'] is True for r in rr))]:
                check(s[key]==calc,'Summary '+key)
            for d,v in s['domain'].items():
                check(v=={'n':sum(r['domain']==d for r in rr),'correct':sum(r['correct'] and r['domain']==d for r in rr)},'Domain arithmetic')
        for control,paired in e['paired'].items():
            pairs=[(by[q,control]['correct'],by[q,'harness_visual_anchor']['correct']) for q in e['sample_ids']]
            check(paired==dict(zip(['both_correct','anchor_only','control_only','neither'],[pairs.count(x) for x in [(True,True),(False,True),(True,False),(False,False)]])),'Paired arithmetic')
        for q in e['sample_ids']:
            check(len({by[q,a]['query_image_sha256'] for a in e['arms']})==1,'Same original query image across methods')
        check(audit['totals']['generations']==sum(r['calls'] for r in rows)
              and audit['totals']['tokens']==sum(r['tokens'] for r in rows),'Audit totals')
        check(e['judge_calls']==sum(len(r['judge_ledger']) for r in rows)
              and e['judge_tokens']==sum(x['charged_tokens'] for r in rows for x in r['judge_ledger']),'Separate judge usage')
    dev_images={r['query_image_sha256'] for r in p['experiments']['preliminary_development']['results']}
    held_images={r['query_image_sha256'] for r in p['experiments']['holdout']['results']}
    check(p['holdout_query_image_hash_overlap_with_development']==len(dev_images&held_images),'Image overlap disclosure')
    print(json.dumps({'status':'PASS','preliminary_development_outcomes':36,'development_outcomes':36,'holdout_outcomes':32,
                      'scope':'Public arithmetic/budgets/fingerprints; not independent semantic regrading'}))


if __name__=='__main__':main()
