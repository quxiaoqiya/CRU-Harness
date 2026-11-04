"""Post-inference metadata adapter: default omitted report category to gotchas.

Only question_type report metadata differs from the frozen scorer. Reference
answers, official prompts, judgments, inference records and cached judge calls
are unchanged. This adapter is used only for the already completed dev stage.
"""
from pathlib import Path
from collections import Counter
import json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.benchmark_longmemeval_v2_visual_anchor import (MODEL,BudgetedClient,JUDGE_BUDGET,check,record_path,read_rows,digest,atomic,official_scorer,AuditedTransport)
from long_context_agent.experiment_budget import load_provider_env

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
            'scoring_function':name,'question_type':label.get('question_type','gotchas'),
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


if __name__=='__main__':
    w=Path(sys.argv[1]).resolve()
    provider=load_provider_env(sys.argv[2])
    print(json.dumps(score(w,provider),indent=2),flush=True)
