"""Six-lane, checkpointed full LME-V2 Small evaluation on the local Mac.

The unchanged V1/V3 text fallback is evaluated once, explicitly shared. Gold
access begins only after all 1,353 logical / 931 physical jobs are terminal.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import Counter
import copy
import fcntl
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import threading
import time

from scripts import benchmark_longmemeval_v2_visual_handoff as b
from long_context_agent.trajectory_memory_completion import CompletionEvidenceClient, run_completion_harness

ROOT = b.ROOT
WORK = ROOT/'work/longmemeval_v2_full_qwen_20261104'
ARMS = ('direct_headtail_native','harness_visual_anchor_v1','harness_visual_anchor')
CHANNELS = {'direct':3,'harness':3,'judge':6}


def code_hashes():
    result = b.code_hashes()
    for name in ['scripts/benchmark_longmemeval_v2_full.py','scripts/prepare_longmemeval_v2_full.py',
                 'tests/test_longmemeval_v2_full.py']:
        result[name] = b.digest(ROOT/name)
    return result


def freeze(work):
    # Extend base registration before any inference can begin. The final hash,
    # guard and source snapshot returned here are the registered experiment.
    b.freeze(work)
    out = work/'formal'
    value = json.loads((out/'freeze.json').read_text())
    selection = json.loads((work/'selection_manifest.json').read_text())
    inputs = b.read_rows(work/'inputs.jsonl')
    assert len(inputs)==451 and sum(bool(q['image']) for q in inputs)==29
    assert selection['logical_records']==1353 and selection['physical_inference_jobs']==931
    guard = (out/'labels_guard.sb').read_text()
    guard += ''.join('(deny file-read* (subpath '+json.dumps(str(p.resolve()))+'))\n'
                     for p in (ROOT/'work').glob('longmemeval_v2_*/private_labels'))
    (out/'labels_guard.sb').write_text(guard)
    value.update(experiment='LongMemEval-V2 Small full451 Qwen V1/V3 comparison',
        haystack_tier='small',expected_physical_records=931,shared_text_path_reuse=True,
        shared_text_records=422,channels=CHANNELS,transport_consecutive_error_stop=3,
        code_sha256=code_hashes(),guard_profile_sha256=b.digest(out/'labels_guard.sb'),
        cohorts_sha256=selection['cohorts_sha256'],official_leaderboard_submission=False,
        question_disjoint_from_development=False,previously_consumed_question_count=selection['n_previously_consumed'],
        implicit_generation_retries=0,
        recovery_policy='Existing completion-controller bounded recovery is retained and metered; uncertain transport deliveries are never replayed')
    for name,expected in value['code_sha256'].items():
        target=out/'frozen_source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes());assert b.digest(target)==expected
    b.atomic(out/'freeze.json',value)
    return {'freeze_sha256':b.digest(out/'freeze.json'),'questions':451,'logical_records':1353,
            'physical_jobs':931,'channels':CHANNELS}


def check(work,worker=False):
    value=b.check(work,worker=worker)
    assert value['expected_terminal_records']==1353 and value['expected_physical_records']==931
    assert b.digest(work/'formal/labels_guard.sb')==value['guard_profile_sha256']
    return value


def physical_jobs(inputs):
    result=[]
    for row in inputs:
        result += [(row,'direct_headtail_native'),(row,'harness_visual_anchor')]
        if row['image']:result.append((row,'harness_visual_anchor_v1'))
    random.Random(b.SEED).shuffle(result)
    return result


def alias_record(record,source_path):
    if record['question_image'] or record['arm']!='harness_visual_anchor':
        raise ValueError('Only the identical no-query-image fallback may be shared')
    result=copy.deepcopy(record)
    result.update(arm='harness_visual_anchor_v1',shared_physical_record={
        'file':source_path.name,'sha256':b.digest(source_path),
        'reason':'V1 and V3 both delegate to identical CompletionEvidenceClient/run_completion_harness with the same arguments',
        'additional_generation_calls':0,'additional_tokens':0})
    return result


def worker(work,provider):
    frozen=check(work,worker=True);out=work/'formal';freeze_sha=b.digest(out/'freeze.json')
    with (out/'.run.lock').open('a+') as runlock:
        fcntl.flock(runlock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (out/'run_summary.json').exists():
            previous=json.loads((out/'run_summary.json').read_text())
            if previous['phase']=='inference_complete':return previous
        if (out/'scoring_access.json').exists():raise ValueError('Cannot run after gold access')
        inputs=b.read_rows(work/'inputs.jsonl');stores,insertion=b.load_stores(work);systems=b.policies(work)
        b.atomic(out/'insertion_audit.json',insertion)
        assets=json.loads((work/'assets_manifest.json').read_text())
        for relative,expected in {**assets['history_images'],**assets['query_images']}.items():
            if b.digest(work/'data'/relative)!=expected:raise ValueError('Screenshot changed')
        counts=json.loads((work/'source_preflight_counts.json').read_text())
        native_counts={k:v['canonical_text_tokens'] for k,v in counts['counts'].items()}
        assert len(native_counts)==2
        indexes={}
        # Build once, before threads: source/index objects are read-only during
        # retrieval; each job gets a separate transport, budget and event log.
        for row in inputs:
            if not row['image']:continue
            store=stores[row['id']]
            for legacy in (False,True):
                key=(id(store),legacy)
                if key not in indexes:indexes[key]=(b.LegacyVisualIndex if legacy else b.VisualAnchorIndex)(store)
        lock=threading.Lock();stop=threading.Event();consecutive_errors=0
        done=0;started=time.time()

        def execute(job):
            nonlocal done,consecutive_errors
            row,arm=job;qid=row['id'];target=b.record_path(out,qid,arm)
            if stop.is_set():return
            if target.exists():
                if json.loads(target.read_text())['freeze_sha256']!=freeze_sha:raise ValueError('Foreign checkpoint')
                return
            journal=out/'journals'/target.name
            store=stores[qid];source_sha=store.metadata()['canonical_source_sha256']
            transport=b.AuditedTransport(**provider,timeout=360)
            def save(ledger):
                b.atomic(journal,{'id':qid,'arm':arm,'freeze_sha256':freeze_sha,'ledger':ledger,
                    'count_events':transport.count_events,'response_events':transport.response_events,
                    'transport_errors':transport.error_events})
            client=b.BudgetedClient(transport,b.BUDGETS[arm],on_ledger_update=save)
            status,answer,result,error='not_started','',{},None
            begin=time.perf_counter()
            if journal.exists():
                previous=json.loads(journal.read_text());client.ledger=previous['ledger']
                transport.count_events=previous['count_events'];transport.response_events=previous['response_events']
                transport.error_events=previous['transport_errors']
                status,error='interrupted',{'type':'InterruptedTask','message':'Uncertain delivery is not replayed'}
            else:
                save([])
                try:
                    if arm=='direct_headtail_native':
                        adapted=b.TrajectoryEvidenceClient(client,store,domain_system=systems[row['domain']],
                            query_image=row['image'],history_image_limit=16-int(bool(row['image'])),
                            image_selection='uniform',question_text=row['question'])
                        retained=min(len(store.context),int(len(store.context)*240000/native_counts[source_sha]))
                        preparation=[]
                        for _ in range(16):
                            windows=b.headtail_windows(store,retained);messages=b._messages(row['question'],windows,final=True)
                            count=adapted.count_messages(messages)
                            preparation.append({'retained_chars':sum(w.end-w.start for w in windows),'admission_tokens':count})
                            if adapted.can_call(messages,4096):break
                            retained=max(2,int(retained*min(.99,(262144-4096)/max(1,count))*.99))
                        else:raise ValueError('Direct admission failed after sixteen deterministic shrinks')
                        answer=adapted.complete(messages,4096)
                        result={'full_canonical_text_tokens':native_counts[source_sha],
                            'retained_source_spans':[[w.start,w.end] for w in windows],
                            'retained_source_chars':sum(w.end-w.start for w in windows),'preparation_rounds':preparation,
                            'history_image_limit':adapted.history_image_limit,'full_multimodal_source_preserved':False,
                            'image_events':adapted.image_events}
                    elif not row['image']:
                        # Exactly the no-image fallback used by both frozen APIs.
                        result=run_completion_harness(store,row['question'],CompletionEvidenceClient(
                            client,store,domain_system=systems[row['domain']],question_text=row['question']))
                        answer=result['answer']
                    else:
                        legacy=arm=='harness_visual_anchor_v1'
                        result=(b.run_legacy_visual if legacy else b.run_visual_anchor)(indexes[id(store),legacy],
                            row['question'],client,domain_system=systems[row['domain']],query_image=row['image'])
                        answer=result['answer']
                    status='returned' if answer.strip() else 'empty'
                except Exception as exc:
                    status,error='error',{'type':type(exc).__name__}
            usage=client.summary()
            record={'id':qid,'domain':row['domain'],'question_image':bool(row['image']),
                'arm':arm,'model':b.MODEL,'freeze_sha256':freeze_sha,'terminal':True,'status':status,
                'answer':answer,'result':result,'error':error,'source_metadata':store.metadata(),
                'source_text_tokens_exact':native_counts[source_sha],'usage':usage,'ledger':client.ledger,
                'count_events':transport.count_events,'response_events':transport.response_events,
                'transport_errors':transport.error_events,'seconds':time.perf_counter()-begin,
                'completed_at_unix':time.time()}
            b.atomic(target,record)
            with lock:
                done+=1
                consecutive_errors=consecutive_errors+1 if transport.error_events else 0
                if consecutive_errors>=frozen['transport_consecutive_error_stop']:stop.set()
                if done%10==0 or status!='returned':
                    print(json.dumps({'physical_jobs_finished_this_run':done,'planned':931,
                        'latest_id':qid,'arm':arm,'status':status,'stop':stop.is_set()}),flush=True)

        jobs=physical_jobs(inputs)
        with ThreadPoolExecutor(max_workers=CHANNELS['direct']) as dp, ThreadPoolExecutor(max_workers=CHANNELS['harness']) as hp:
            futures=[(dp if arm=='direct_headtail_native' else hp).submit(execute,(row,arm)) for row,arm in jobs]
            for future in as_completed(futures):future.result()
        physical=[b.record_path(out,r['id'],a) for r,a in jobs]
        missing=[p for p in physical if not p.exists()]
        if missing:
            summary={'phase':'inference_incomplete','physical_records':len(physical)-len(missing),
                'missing_physical_records':len(missing),'labels_opened':False,'circuit_breaker':stop.is_set()}
            b.atomic(out/'run_summary.json',summary);return summary
        for row in inputs:
            if row['image']:continue
            source=b.record_path(out,row['id'],'harness_visual_anchor')
            target=b.record_path(out,row['id'],'harness_visual_anchor_v1')
            value=alias_record(json.loads(source.read_text()),source)
            if target.exists():
                if json.loads(target.read_text())!=value:raise ValueError('Shared result changed')
            else:b.atomic(target,value)
        records=[json.loads(p.read_text()) for p in physical]
        summary={'phase':'inference_complete','terminal_records':1353,'physical_records':931,
            'shared_records':422,'labels_opened':False,'seconds_this_run':time.time()-started,
            'physical_generations':sum(r['usage']['physical_requests'] for r in records),
            'physical_charged_tokens':sum(r['usage']['charged_tokens'] for r in records),
            'status_counts_physical':dict(Counter(r['status'] for r in records))}
        b.atomic(out/'run_summary.json',summary);print(json.dumps(summary),flush=True);return summary


def run(work,provider):
    check(work)
    if sys.platform!='darwin':raise RuntimeError('Local macOS sandbox required')
    env={k:v for k,v in os.environ.items() if not any(s in k.upper() for s in ('KEY','TOKEN','PASSWORD','SECRET'))}
    result=subprocess.run(['/usr/bin/sandbox-exec','-f',str(work/'formal/labels_guard.sb'),sys.executable,
        '-B','-m','scripts.benchmark_longmemeval_v2_full','--worker'],
        input=json.dumps({'work':str(work),'provider':provider}),text=True,cwd=ROOT,env=env)
    if result.returncode:raise RuntimeError('Worker stopped; preserve checkpoints')
    return json.loads((work/'formal/run_summary.json').read_text())


def summarize(rows):
    n=len(rows)
    return {'n':n,'correct':sum(r['correct'] is True for r in rows),
        'accuracy_all_planned':sum(r['correct'] is True for r in rows)/n if n else None,
        'unknown':sum(r['unknown'] for r in rows),'judge_failures':sum(bool(r['judge_error']) for r in rows),
        'status_counts':dict(Counter(r['status'] for r in rows)),
        'method_assigned_tokens':sum(r['usage']['charged_tokens'] for r in rows),
        'method_assigned_generations':sum(r['usage']['physical_requests'] for r in rows)}


def score(work,provider):
    frozen=check(work);out=work/'formal';inputs=b.read_rows(work/'inputs.jsonl')
    paths=[b.record_path(out,r['id'],a) for r in inputs for a in ARMS]
    if not all(p.is_file() for p in paths):raise ValueError('All 1353 logical records required before gold access')
    records=[json.loads(p.read_text()) for p in paths]
    assert all(r['terminal'] and r['usage']['within_budget'] for r in records)
    access={'opened_at_unix':time.time(),'all_registered_terminal_before_gold':True,
        'terminal_record_sha256':{p.name:b.digest(p) for p in paths}}
    if (out/'scoring_access.json').exists():
        assert json.loads((out/'scoring_access.json').read_text())['terminal_record_sha256']==access['terminal_record_sha256']
    else:b.atomic(out/'scoring_access.json',access)
    if (out/'audit.json').exists():return json.loads((out/'audit.json').read_text())['summary']
    labels=json.loads((work/'private_labels/labels.json').read_text())
    cohorts=json.loads((work/'private_labels/cohorts.json').read_text())
    selection=json.loads((work/'selection_manifest.json').read_text())
    assert b.digest(work/'private_labels/labels.json')==selection['labels_sha256']
    assert b.digest(work/'private_labels/cohorts.json')==frozen['cohorts_sha256']
    questions={q['id']:q for q in inputs};metrics=b.official_scorer(work)
    progress_lock=threading.Lock();completed=0
    def one(record):
        nonlocal completed
        qid,arm=record['id'],record['arm'];target=out/'scored_records'/f'{qid}__{arm}.json'
        if target.exists():return json.loads(target.read_text())
        label=labels[qid];parsed=metrics.extract_boxed_answer(record['answer']);unknown=metrics.is_unknown(parsed)
        name=metrics.eval_name(label['eval_function']);correct=False;ledger=[];error=None
        if record['status']=='returned' and record['answer'].strip() and not unknown:
            if name in ('llm_abstention_checker','llm_gotchas_checker'):
                jp=out/'judge_journals'/target.name
                if jp.exists():ledger=json.loads(jp.read_text())['ledger']
                else:
                    constructor=metrics._build_abstention_judge_messages if name=='llm_abstention_checker' else metrics._build_gotchas_judge_messages
                    messages=constructor(question_text=questions[qid]['question'],reference_answer=label['answer'],
                        model_full_response=record['answer'],model_final_answer=parsed)
                    def save(entries):b.atomic(jp,{'ledger':entries,'inference_record_sha256':b.digest(b.record_path(out,qid,arm))})
                    client=b.BudgetedClient(b.AuditedTransport(**provider,timeout=180),b.JUDGE_BUDGET,on_ledger_update=save)
                    try:client.complete(messages,512)
                    except Exception as exc:error=type(exc).__name__
                    ledger=client.ledger
                if error is None:
                    if len(ledger)==1 and ledger[0].get('status')=='ok':
                        try:correct=metrics._parse_llm_binary_judgement(ledger[0]['content'])[0]==1
                        except Exception as exc:error=type(exc).__name__
                    else:error='InterruptedJudgeNotReplayed'
                if error:correct=None
            else:correct=bool(metrics.eval_from_spec(label['eval_function'],parsed,label['answer']))
        row={**record,'parsed_answer':parsed,'unknown':unknown,'correct':correct,'scoring_function':name,
            'question_type':label['question_type'],'cohort':cohorts[qid],
            'judge_model':b.MODEL if name.startswith('llm_') else None,'judge_ledger':ledger,'judge_error':error}
        b.atomic(target,row)
        with progress_lock:
            completed+=1
            if completed%25==0:print(json.dumps({'phase':'scoring','physical_scored_this_run':completed,'planned':931}),flush=True)
        return row
    physical=[r for r in records if 'shared_physical_record' not in r]
    with ThreadPoolExecutor(max_workers=CHANNELS['judge']) as pool:
        scored=list(pool.map(one,physical))
    by={(r['id'],r['arm']):r for r in scored}
    for record in records:
        if 'shared_physical_record' not in record:continue
        source=by[record['id'],'harness_visual_anchor']
        row={**source,**record,'judge_compute_shared':True}
        b.atomic(out/'scored_records'/f"{row['id']}__{row['arm']}.json",row);scored.append(row)
    summary={}
    for arm in ARMS:
        group=[r for r in scored if r['arm']==arm]
        summary[arm]={'overall':summarize(group),
            'by_question_type':{k:summarize([r for r in group if r['question_type']==k]) for k in sorted({r['question_type'] for r in group})},
            'by_domain':{k:summarize([r for r in group if r['domain']==k]) for k in ('enterprise','web')},
            'by_cohort':{k:summarize([r for r in group if r['cohort']==k]) for k in ('previously_consumed','unused_question')},
            'visual':summarize([r for r in group if r['question_image']]),
            'no_query_image':summarize([r for r in group if not r['question_image']]),
            'flawed_premise':summarize([r for r in group if r['scoring_function']=='llm_abstention_checker'])}
    report={'freeze_sha256':b.digest(out/'freeze.json'),'summary':summary,'scored':scored,
        'logical_records':1353,'physical_records':931,'shared_records':422,
        'all_registered_fixed_before_labels':True,'official_leaderboard_submission':False,
        'physical_judge_calls':sum(len(r['judge_ledger']) for r in physical if 'judge_ledger' in r)}
    # physical is the prediction list, not scored rows: count real judge ledgers.
    report['physical_judge_calls']=sum(len(r['judge_ledger']) for r in scored if not r.get('judge_compute_shared'))
    b.atomic(out/'audit.json',report);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',nargs='?',choices=('freeze','run','score'))
    parser.add_argument('--work',type=Path,default=WORK);parser.add_argument('--provider-env',type=Path)
    parser.add_argument('--worker',action='store_true')
    args=parser.parse_args();work=args.work.resolve()
    if args.worker:
        value=json.loads(sys.stdin.read());result=worker(Path(value['work']),value['provider'])
    elif args.phase=='freeze':result=freeze(work)
    else:
        provider=b.load_provider_env(args.provider_env)
        if provider['model']!=b.MODEL:raise ValueError('Registered Qwen model required')
        result=(run if args.phase=='run' else score)(work,provider)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
