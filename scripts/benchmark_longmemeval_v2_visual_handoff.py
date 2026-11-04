"""LongMemEval-V2 screenshot perception and observed-transition recall.

All orchestration is local. Gold/rubrics open only after all inference records
are terminal. The Qwen judge substitution is explicit, not an official LAFS run.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
from dataclasses import asdict
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from urllib.error import HTTPError

from long_context_agent.adaptive_evidence import EvidenceWindow, _messages
from long_context_agent.experiment_budget import (BudgetedClient, ExperimentBudget,
    ProviderTransport, load_provider_env)
from long_context_agent.trajectory_memory_completion import (CompletionEvidenceClient,CompletionConfig,run_completion_harness,PerCallBudgetedClient)
from long_context_agent.trajectory_memory_compact import CompactCompletionClient,run_compact_completion
from long_context_agent.visual_anchor_memory_v3 import VisualAnchorIndex,run_visual_anchor,AnchorConfig
from long_context_agent.visual_anchor_memory import VisualAnchorIndex as LegacyVisualIndex, run_visual_anchor as run_legacy_visual
from long_context_agent.trajectory_memory import (TrajectoryConfig, TrajectoryStore,
    TrajectoryEvidenceClient, retrieve_trajectory_evidence)


ROOT=Path(__file__).resolve().parents[1]
DEFAULT_WORK=ROOT/'work/longmemeval_v2_visual_handoff_dev_20261104'
ARMS=('direct_headtail_native','harness_completion','visual_query_only','harness_visual_anchor','harness_visual_anchor_v1')
MODEL='Qwen3.8-Flash-Next-FP8'
SEED=20261103
BUDGETS={
    'direct_full':ExperimentBudget(262144,262144,1,4096,seed=SEED),
    'direct_headtail_native':ExperimentBudget(262144,262144,1,4096,seed=SEED),
    'direct_headtail_native_concise':ExperimentBudget(262144,262144,1,4096,seed=SEED),
    'harness':ExperimentBudget(16384,65536,8,4096,seed=SEED),
    'harness_compact':ExperimentBudget(16384,65536,8,4096,seed=SEED),
    'harness_completion':ExperimentBudget(16384,65536,8,4096,seed=SEED),
    'visual_query_only':ExperimentBudget(16384,65536,8,4096,seed=SEED),
    'harness_visual_anchor_v1':ExperimentBudget(16384,65536,8,4096,seed=SEED),
    'harness_visual_anchor':ExperimentBudget(16384,65536,8,4096,seed=SEED),
}
BudgetedClient=PerCallBudgetedClient
JUDGE_BUDGET=ExperimentBudget(16384,16384,1,512,seed=SEED)


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while block:=f.read(4*1024*1024):h.update(block)
    return h.hexdigest()


def text_digest(text):return hashlib.sha256(text.encode()).hexdigest()


def atomic(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');tmp.replace(path)


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def policies(work):
    tree=ast.parse((work/'upstream/evaluation/harness.py').read_text())
    for node in tree.body:
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='DOMAIN_SYSTEM_PROMPTS' for t in node.targets):
            return ast.literal_eval(node.value)
    raise ValueError('Pinned upstream domain prompts not found')


def code_hashes():
    # The old public verifier separately establishes these historical hashes.
    paths=[]
    for directory in ('long_context_agent','reading_memory','independent_ingestion'):
        paths.extend(p for p in (ROOT/directory).rglob('*.py') if 'legacy_v31' not in p.parts)
    paths.extend([Path(__file__),ROOT/'tests/test_trajectory_memory.py',ROOT/'tests/test_longmemeval_v2_benchmark.py',ROOT/'tests/test_trajectory_memory_v2.py',ROOT/'tests/test_trajectory_memory_assist.py',ROOT/'tests/test_trajectory_memory_completion.py',ROOT/'tests/test_trajectory_memory_compact.py',ROOT/'tests/test_visual_anchor_memory.py',ROOT/'tests/test_visual_anchor_contracts.py',ROOT/'tests/test_visual_anchor_handoff.py',ROOT/'scripts/prepare_longmemeval_v2_visual_handoff.py'])
    return {str(p.relative_to(ROOT)):digest(p) for p in sorted(set(paths))}


def freeze(work):
    out=work/'formal'
    if (out/'freeze.json').exists():raise ValueError('An existing experiment is immutable')
    selection=json.loads((work/'selection_manifest.json').read_text())
    assets=json.loads((work/'assets_manifest.json').read_text())
    inputs=read_rows(work/'inputs.jsonl')
    if len(inputs)!=selection['n_questions'] or any(set(r)!={'id','domain','question','image'} for r in inputs):
        raise ValueError('Registered label-free inputs are required')
    if digest(work/'inputs.jsonl')!=selection['inputs_sha256']:
        raise ValueError('Selected inputs changed')
    if digest(work/'selected_trajectories.jsonl')!=assets['selected_trajectories_sha256']:
        raise ValueError('Selected sources changed')
    upstream_sha=subprocess.check_output(['git','rev-parse','HEAD'],cwd=work/'upstream',text=True).strip()
    if upstream_sha!=selection['upstream_code_commit']:raise ValueError('Upstream code revision changed')
    arms=selection['arms']
    if not arms or any(a not in ARMS for a in arms):raise ValueError('Unregistered arm')
    out.mkdir(parents=True,exist_ok=True)
    probe=out/'label_guard_probe.txt';probe.write_text('must not be readable by the inference worker\n')
    profile='(version 1)\n(allow default)\n'
    profile+=''.join('(deny file-read* (subpath '+json.dumps(str(p.resolve()))+'))\n'
        for p in [work/'private_labels',ROOT/'benchmarks',ROOT/'docs'])
    profile+=''.join('(deny file-read* (literal '+json.dumps(str(p.resolve()))+'))\n'
        for p in [work/'data/questions.jsonl',probe])
    previous=ROOT/'work/longmemeval_v2_qwen_20261103'
    profile+=''.join('(deny file-read* (subpath '+json.dumps(str(p.resolve()))+'))\n' for p in [previous/'private_labels',previous/'formal'])
    if selection['stage']=='holdout':
        profile+=''.join('(deny file-read* (subpath '+json.dumps(str(p.resolve()))+'))\n' for p in (ROOT/'work').glob('longmemeval_v2_*/private_labels'))
    profile+=''.join('(deny file-read* (subpath '+json.dumps(str(p.resolve()))+'))\n' for p in (ROOT/'work').glob('longmemeval_v2_*/formal') if p.parent.resolve()!=work.resolve())
    (out/'labels_guard.sb').write_text(profile)
    frozen={'schema_version':1,'experiment':'LongMemEval-V2 Qwen multimodal pilot',
        'frozen_at_unix':time.time(),'work_directory':str(work.resolve()),'model':MODEL,
        'question_ids':[r['id'] for r in inputs],'arms':arms,'expected_terminal_records':len(inputs)*len(arms),'stage':selection['stage'],
        'selection_manifest_sha256':digest(work/'selection_manifest.json'),
        'inputs_sha256':digest(work/'inputs.jsonl'),'haystacks_sha256':digest(work/'haystacks.json'),
        'source_sha256':digest(work/'selected_trajectories.jsonl'),
        'assets_manifest_sha256':digest(work/'assets_manifest.json'),
        'source_token_count_cache_sha256':digest(work/'source_preflight_counts.json'),
        'source_token_count_cache_is_exact_qwen_text_only':True,
        'guard_profile_sha256':digest(out/'labels_guard.sb'),
        'upstream_code_commit':upstream_sha,'data_revision':selection['data_revision'],
        'upstream_scoring_sha256':digest(work/'upstream/evaluation/qa_eval_metrics.py'),
        'upstream_prompt_source_sha256':digest(work/'upstream/evaluation/harness.py'),
        'runtime':{'python':sys.version,'platform':sys.platform,'executable':sys.executable},
        'code_sha256':code_hashes(),'budgets':{k:asdict(BUDGETS[k]) for k in arms},
        'trajectory_config':asdict(TrajectoryConfig()),'completion_config':asdict(CompletionConfig()),'anchor_config':asdict(AnchorConfig()),'visual_contracts_version':3,'seed':SEED,'temperature':0,
        'thinking_enabled':False,'implicit_generation_retries':0,
        'labels_opened_for_scoring':False,'official_leaderboard_submission':False,
        'same_compute_comparison':False,'judge_model':MODEL,'question_disjoint_from_development':selection['stage']=='holdout',
        'source_histories_reused_from_development':True,
        'judge_budget':asdict(JUDGE_BUDGET),
        'judge_substitution_from_official_default':True,
        'direct_headtail_definition':'Two original character spans, prefix and suffix; target240k text input tokens before exact multimodal admission; up to16 total original screenshots, uniformly sampled from retained state spans',
        'full_direct_definition':'Full canonical text lower-bound admission first; if it alone exceeds native capacity, deny without silently dropping screenshots',
        'all_image_asset_hashes_verified_before_generation':True}
    atomic(out/'freeze.json',frozen)
    for name,expected in frozen['code_sha256'].items():
        target=out/'frozen_source'/name;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes());assert digest(target)==expected
    return {'frozen':True,'sha256':digest(out/'freeze.json'),'records':len(inputs)*len(arms)}


def check(work, *, worker=False):
    out=work/'formal';frozen=json.loads((out/'freeze.json').read_text())
    for file,key in [('inputs.jsonl','inputs_sha256'),('haystacks.json','haystacks_sha256'),
        ('selected_trajectories.jsonl','source_sha256'),('assets_manifest.json','assets_manifest_sha256'),
        ('selection_manifest.json','selection_manifest_sha256')]:
        if digest(work/file)!=frozen[key]:raise ValueError('Frozen input/metadata changed: '+file)
    if digest(work/'source_preflight_counts.json')!=frozen['source_token_count_cache_sha256']:
        raise ValueError('Exact source token-count cache changed')
    for name,expected in frozen['code_sha256'].items():
        if digest(ROOT/name)!=expected or digest(out/'frozen_source'/name)!=expected:
            raise ValueError('Frozen source changed: '+name)
    if digest(work/'upstream/evaluation/qa_eval_metrics.py')!=frozen['upstream_scoring_sha256']:
        raise ValueError('Official scoring source changed')
    if digest(work/'upstream/evaluation/harness.py')!=frozen['upstream_prompt_source_sha256']:
        raise ValueError('Official prompt source changed')
    if worker:
        try:(out/'label_guard_probe.txt').read_text()
        except PermissionError:pass
        else:raise ValueError('Label isolation is not enforced')
    return frozen


class AuditedTransport(ProviderTransport):
    def __init__(self,**kwargs):
        super().__init__(**kwargs);self.count_events=[];self.response_events=[];self.error_events=[]
    def count(self,**kwargs):
        before=len(self._counts);value=super().count(**kwargs)
        self.count_events.append({'payload_sha256':text_digest(json.dumps(kwargs,sort_keys=True,ensure_ascii=False)),
            'tokens':value,'cache_hit':len(self._counts)==before,
            'image_parts':sum(1 for m in kwargs.get('messages',[]) for p in (m.get('content') if isinstance(m.get('content'),list) else []) if p.get('type')=='image_url')})
        return value
    def request(self,path,body=None,**kwargs):
        try:payload=super().request(path,body,**kwargs)
        except HTTPError as exc:
            raw=exc.read(65536)
            try:
                obj=json.loads(raw);error=obj.get('error',{})
                code=error.get('code') if isinstance(error,dict) else None
            except (ValueError,TypeError):code=None
            self.error_events.append({'stage':'generation' if path=='/chat/completions' else 'count',
                'http_status':exc.code,'code':code if code in {'context_length_exceeded','invalid_request_error','rate_limit_exceeded','server_error'} else None})
            raise HTTPError(exc.url,exc.code,exc.msg,exc.headers,io.BytesIO(raw)) from None
        except Exception as exc:
            self.error_events.append({'stage':'generation' if path=='/chat/completions' else 'count',
                'error_type':type(exc).__name__})
            raise
        if path=='/chat/completions':
            event={'actual_model':payload.get('model'),'usage':payload.get('usage'),
                'response_id':payload.get('id'),'finish_reason':(payload.get('choices') or [{}])[0].get('finish_reason')}
            self.response_events.append(event)
            if payload.get('model')!=self.model:raise ValueError('Actual model identity differs')
        return payload


def load_stores(work):
    rows={r['id']:r for r in read_rows(work/'selected_trajectories.jsonl')}
    haystacks=json.loads((work/'haystacks.json').read_text())
    stores={};by_question={};timing=[]
    for qid,ids in haystacks.items():
        key=tuple(ids)
        if key not in stores:
            started=time.perf_counter();store=TrajectoryStore(work/'data')
            for tid in ids:store.insert(rows[tid])
            store.finalize();stores[key]=store
            timing.append({'source_sha256':store.metadata()['canonical_source_sha256'],
                'insertion_and_index_seconds':time.perf_counter()-started,**store.metadata()})
        by_question[qid]=stores[key]
    return by_question,timing


def record_path(out,qid,arm):return out/'predictions'/f'{qid}__{arm}.json'


def headtail_windows(store,retained):
    retained=max(2,min(len(store.context),int(retained)))
    if retained>=len(store.context):return [EvidenceWindow(0,len(store.context),store.context,'direct_full',0)]
    half=(retained+1)//2;tail=retained//2
    return [EvidenceWindow(0,half,store.context[:half],'direct_head',0),
            EvidenceWindow(len(store.context)-tail,len(store.context),store.context[-tail:],'direct_tail',0)]


def worker(work, provider):
    frozen=check(work,worker=True);out=work/'formal';freeze_sha=digest(out/'freeze.json')
    with (out/'.run.lock').open('a+') as lock:
        fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        inputs=read_rows(work/'inputs.jsonl');stores,insertion=load_stores(work);system=policies(work)
        visual_indexes={}
        assets=json.loads((work/'assets_manifest.json').read_text())
        for relative,expected in {**assets['history_images'],**assets['query_images']}.items():
            if digest(work/'data'/relative)!=expected:raise ValueError('Source screenshot bytes changed')
        atomic(out/'insertion_audit.json',insertion)
        transport=AuditedTransport(**provider,timeout=360)
        count_cache=json.loads((work/'source_preflight_counts.json').read_text())
        native_text_counts={sha:row['canonical_text_tokens'] for sha,row in count_cache['counts'].items()}
        if len(native_text_counts)!=2 or any(type(n) is not int or n<=0 for n in native_text_counts.values()):
            raise ValueError('Two exact source-text tokenization cache entries are required')
        jobs=[(row,arm) for row in inputs for arm in frozen['arms']]
        random.Random(SEED).shuffle(jobs)
        for row,arm in jobs:
            qid=row['id'];target=record_path(out,qid,arm)
            if target.exists():
                if json.loads(target.read_text()).get('freeze_sha256')!=freeze_sha:raise ValueError('Old terminal from another freeze')
                continue
            if (out/'scoring_access.json').exists():raise ValueError('Never fill missing predictions after gold access')
            store=stores[qid];journal=out/'journals'/target.name
            counts_start,responses_start,errors_start=len(transport.count_events),len(transport.response_events),len(transport.error_events)
            def save(ledger):atomic(journal,{'id':qid,'arm':arm,'freeze_sha256':freeze_sha,'ledger':ledger,
                'count_events':transport.count_events[counts_start:], 'response_events':transport.response_events[responses_start:],
                'transport_errors':transport.error_events[errors_start:]})
            client=BudgetedClient(transport,BUDGETS[arm],on_ledger_update=save)
            policy=system[row['domain']]
            if arm=='direct_headtail_native_concise':policy+=' Return only the shortest final answer in \\boxed{...}. No analysis or commentary. If evidence is insufficient use \\boxed{UNKNOWN}; do not guess.'
            adapted=TrajectoryEvidenceClient(client,store,domain_system=policy,query_image=row['image'],
                history_image_limit=(len(store.bindings) if arm=='direct_full' else
                    16-int(bool(row['image'])) if arm.startswith('direct_headtail_native') else None),
                image_selection='uniform' if arm.startswith('direct_headtail_native') else 'overlap',
                question_text=row['question'])
            status,answer,result,error='not_started','',{},None
            started=time.perf_counter()
            if journal.exists():
                previous=json.loads(journal.read_text());client.ledger=previous['ledger']
                status,error='interrupted',{'type':'InterruptedTask','message':'No uncertain generation replay'}
            else:
                save([])
                try:
                    source_sha=store.metadata()['canonical_source_sha256']
                    if source_sha not in native_text_counts:
                        # Count the full original text projection, not a
                        # truncated proxy. This is a lower bound on full MM input.
                        native_text_counts[source_sha]=transport.count(text=store.context)
                    source_tokens=native_text_counts[source_sha]
                    if arm in ('harness_visual_anchor','visual_query_only','harness_visual_anchor_v1'):
                        key=(id(store),arm=='harness_visual_anchor_v1')
                        if key not in visual_indexes:visual_indexes[key]=(LegacyVisualIndex if arm=='harness_visual_anchor_v1' else VisualAnchorIndex)(store)
                        result=(run_legacy_visual if arm=='harness_visual_anchor_v1' else run_visual_anchor)(visual_indexes[key],row['question'],client,
                            domain_system=system[row['domain']],query_image=row['image'],query_only=arm=='visual_query_only')
                        answer=result['answer'];status='returned' if answer.strip() else 'empty'
                    elif arm in ('harness_compact','harness_completion'):
                        cls=CompactCompletionClient if arm=='harness_compact' else CompletionEvidenceClient
                        adapted=cls(client,store,domain_system=system[row['domain']],query_image=row['image'],question_text=row['question'],completion_config=CompletionConfig())
                        result=(run_compact_completion if arm=='harness_compact' else run_completion_harness)(store,row['question'],adapted)
                        answer=result['answer'];status='returned' if answer.strip() else 'empty'
                    elif arm=='harness':
                        result=retrieve_trajectory_evidence(store,row['question'],adapted)
                        answer=result['answer'];status='returned' if answer.strip() else 'empty'
                    elif arm=='direct_full' and source_tokens>262144+4096:
                        status='native_capacity_denied'
                        result={'full_canonical_text_tokens':source_tokens,'full_multimodal_count_is_lower_bounded_by_text':True,
                            'omitted_history_images_for_generation':False,'generation_sent':False}
                    else:
                        retained=len(store.context) if arm=='direct_full' else min(len(store.context),int(len(store.context)*240000/max(1,source_tokens)))
                        rounds=[]
                        for attempt in range(16):
                            windows=headtail_windows(store,retained)
                            messages=_messages(row['question'],windows,final=True)
                            count=adapted.count_messages(messages)
                            rounds.append({'retained_chars':sum(w.end-w.start for w in windows),'admission_tokens':count})
                            if adapted.can_call(messages,4096):break
                            retained=max(2,int(retained*min(.99,(262144-4096)/max(1,count))*.99))
                        else:raise ValueError('Native head/tail could not fit after sixteen deterministic shrinks')
                        if arm=='direct_full' and retained<len(store.context):raise ValueError('Full Direct cannot silently truncate')
                        answer=adapted.complete(messages,4096);status='returned' if answer.strip() else 'empty'
                        result={'full_canonical_text_tokens':source_tokens,'retained_source_spans':[[w.start,w.end] for w in windows],
                            'retained_source_chars':sum(w.end-w.start for w in windows),'preparation_rounds':rounds,
                            'history_image_limit':adapted.history_image_limit,
                            'full_multimodal_source_preserved':arm=='direct_full','image_events':adapted.image_events}
                except Exception as exc:
                    status='error';error={'type':type(exc).__name__}
            usage=client.summary()
            record={'id':qid,'domain':row['domain'],'question_image':bool(row['image']),
                'arm':arm,'model':MODEL,'freeze_sha256':freeze_sha,'terminal':True,'status':status,
                'answer':answer,'result':result,'error':error,'source_metadata':store.metadata(),
                'source_text_tokens_exact':native_text_counts.get(store.metadata()['canonical_source_sha256']),
                'usage':usage,'ledger':client.ledger,'count_events':transport.count_events[counts_start:],
                'response_events':transport.response_events[responses_start:],
                'transport_errors':transport.error_events[errors_start:],
                'seconds':time.perf_counter()-started,'completed_at_unix':time.time()}
            atomic(target,record)
            print(json.dumps({'id':qid,'arm':arm,'status':status,'calls':usage['physical_requests'],
                'tokens':usage['charged_tokens'],'seconds':round(record['seconds'],2)}),flush=True)
        expected=[record_path(out,r['id'],a) for r in inputs for a in frozen['arms']]
        if not all(p.exists() for p in expected):raise ValueError('Missing registered terminal records')
        summary={'phase':'inference_complete','terminal_records':len(expected),'labels_opened':False,
            'physical_generations':sum(json.loads(p.read_text())['usage']['physical_requests'] for p in expected)}
        atomic(out/'source_text_token_counts.json',native_text_counts)
        atomic(out/'run_summary.json',summary);return summary


def run(work,provider):
    check(work)
    if sys.platform!='darwin':raise RuntimeError('This run requires local macOS label-isolation support')
    environment={k:v for k,v in os.environ.items() if not any(x in k.upper() for x in ['KEY','TOKEN','PASSWORD','SECRET'])}
    result=subprocess.run(['/usr/bin/sandbox-exec','-f',str(work/'formal/labels_guard.sb'),
        sys.executable,'-m','scripts.benchmark_longmemeval_v2_visual_handoff','--worker'],
        input=json.dumps({'work':str(work),'provider':provider}),text=True,cwd=ROOT,env=environment)
    if result.returncode:raise RuntimeError('Worker stopped; preserve journals and never blindly replay generations')
    return json.loads((work/'formal/run_summary.json').read_text())


def official_scorer(work):
    path=work/'upstream/evaluation/qa_eval_metrics.py'
    spec=importlib.util.spec_from_file_location('pinned_lme_v2_scorer',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def score(work,provider):
    frozen=check(work);out=work/'formal';inputs=read_rows(work/'inputs.jsonl')
    paths=[record_path(out,r['id'],a) for r in inputs for a in frozen['arms']]
    if len(paths)!=frozen['expected_terminal_records'] or not all(p.exists() for p in paths):raise ValueError('All registered terminals required before gold access')
    records=[json.loads(p.read_text()) for p in paths]
    if not all(r.get('terminal') is True and r['usage']['within_budget'] for r in records):raise ValueError('Terminal or budget violation')
    if (out/'audit.json').exists():return json.loads((out/'audit.json').read_text())['summary']
    access={'opened_at_unix':time.time(),'all_registered_terminal_before_gold':True,
        'terminal_record_sha256':{p.name:digest(p) for p in paths}}
    if (out/'scoring_access.json').exists():
        old=json.loads((out/'scoring_access.json').read_text())
        if old['terminal_record_sha256']!=access['terminal_record_sha256']:
            raise ValueError('Inference records changed after gold access')
    else:atomic(out/'scoring_access.json',access)
    labels=json.loads((work/'private_labels/labels.json').read_text())
    if digest(work/'private_labels/labels.json')!=json.loads((work/'selection_manifest.json').read_text())['labels_sha256']:
        raise ValueError('Scoring labels changed')
    questions={r['id']:r for r in inputs};metrics=official_scorer(work)
    transport=AuditedTransport(**provider,timeout=180);scored=[]
    for record in records:
        qid,arm=record['id'],record['arm'];label=labels[qid]
        parsed=metrics.extract_boxed_answer(record['answer']);unknown=metrics.is_unknown(parsed)
        name=metrics.eval_name(label['eval_function']);judge_ledger=[];judge_error=None
        cached=out/'scored_records'/f'{qid}__{arm}.json'
        if cached.exists():scored.append(json.loads(cached.read_text()));continue
        correct=False
        if record['status']=='returned' and record['answer'].strip() and not unknown:
            if name.startswith('llm_'):
                constructor=(metrics._build_abstention_judge_messages if name=='llm_abstention_checker' else metrics._build_gotchas_judge_messages)
                messages=constructor(question_text=questions[qid]['question'],reference_answer=label['answer'],
                    model_full_response=record['answer'],model_final_answer=parsed)
                jp=out/'judge_journals'/f'{qid}__{arm}.json'
                if jp.exists():
                    judge_ledger=json.loads(jp.read_text())['ledger']
                    if len(judge_ledger)==1 and judge_ledger[0].get('status')=='ok' and judge_ledger[0].get('content'):
                        try:
                            value,_=metrics._parse_llm_binary_judgement(judge_ledger[0]['content']);correct=value==1
                        except Exception as exc:judge_error=type(exc).__name__;correct=None
                    else:judge_error='InterruptedJudgeNotReplayed';correct=None
                else:
                    def judge_save(ledger):atomic(jp,{'ledger':ledger,'inference_record_sha256':digest(record_path(out,qid,arm))})
                    client=BudgetedClient(transport,JUDGE_BUDGET,on_ledger_update=judge_save)
                    try:
                        response=client.complete(messages,512);value,_=metrics._parse_llm_binary_judgement(response);correct=value==1
                    except Exception as exc:judge_error=type(exc).__name__;correct=None
                    judge_ledger=client.ledger
            else:
                if name not in {'norm_phrase_set_match','norm_phrase_set_match_ordered','mc_choice_match','mc_choice_set_match'}:
                    raise ValueError('Unregistered scoring function')
                correct=bool(metrics.eval_from_spec(label['eval_function'],parsed,label['answer']))
        row={**record,'parsed_answer':parsed,'unknown':unknown,'correct':correct,
            'scoring_function':name,'question_type':label['question_type'],
            'judge_model':MODEL if name.startswith('llm_') else None,
            'judge_ledger':judge_ledger,'judge_error':judge_error}
        atomic(cached,row);scored.append(row)
    summary={}
    for arm in frozen['arms']:
        group=[r for r in scored if r['arm']==arm]
        returned_evaluable=[r for r in group if r['status']=='returned' and r['correct'] is not None]
        summary[arm]={'n':len(group),'correct':sum(r['correct'] is True for r in group),
            'task_success_rate_all_planned':sum(r['correct'] is True for r in group)/len(group),
            'semantic_accuracy_among_returned_evaluable':(sum(r['correct'] is True for r in returned_evaluable)/len(returned_evaluable)
                if returned_evaluable else None),
            'returned':sum(r['status']=='returned' for r in group),
            'unknown':sum(r['unknown'] for r in group),'judge_failures':sum(r['judge_error'] is not None for r in group),
            'status_counts':dict(Counter(r['status'] for r in group)),
            'generation_calls':sum(r['usage']['physical_requests'] for r in group),
            'charged_or_reserved_tokens':sum(r['usage']['charged_tokens'] for r in group),
            'question_image_correct':sum(r['correct'] is True for r in group if r['question_image']),
            'question_image_n':sum(r['question_image'] for r in group)}
    report={'experiment':frozen['experiment'],'freeze_sha256':digest(out/'freeze.json'),
        'summary':summary,'scored':scored,'all_registered_fixed_before_labels':True,
        'judge_substitution_from_official_default':True,'official_leaderboard_submission':False,
        'judge_calls':sum(len(r['judge_ledger']) for r in scored)}
    atomic(out/'audit.json',report);return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',nargs='?',choices=['freeze','run','score'])
    parser.add_argument('--work',type=Path,default=DEFAULT_WORK)
    parser.add_argument('--provider-env',type=Path)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.worker:
        value=json.loads(sys.stdin.read());result=worker(Path(value['work']),value['provider'])
    elif args.phase=='freeze':result=freeze(args.work.resolve())
    elif args.phase in ('run','score'):
        provider=load_provider_env(args.provider_env)
        if provider['model']!=MODEL:raise ValueError('Configured Qwen model differs from the registered model')
        result=(run if args.phase=='run' else score)(args.work.resolve(),provider)
    else:parser.error('Select a phase')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':main()
