"""Offline post-score ledger/source/image/score audit; no provider calls."""
from pathlib import Path
from collections import Counter
import json
import re
import time

ROOT=Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0,str(ROOT))
from scripts import benchmark_longmemeval_v2 as bench
from long_context_agent.experiment_budget import BudgetedClient
from long_context_agent.trajectory_memory import TrajectoryEvidenceClient,retrieve_trajectory_evidence

WORK=ROOT/'work/longmemeval_v2_qwen_20261103'


class RecordedTransport:
    def __init__(self,record):
        self.model=bench.MODEL;self.counts=record['count_events'];self.entries=record['ledger']
        self.count_pos=self.request_pos=0
    def count(self,**kwargs):
        if self.count_pos>=len(self.counts):raise ValueError('Missing recorded tokenizer event')
        item=self.counts[self.count_pos]
        actual=bench.text_digest(json.dumps(kwargs,sort_keys=True,ensure_ascii=False))
        if item['payload_sha256']!=actual:raise ValueError('Tokenizer input hash differs from original')
        self.count_pos+=1;return item['tokens']
    def request(self,path,body=None):
        if path!='/chat/completions':raise ValueError('Offline replay attempted provider I/O')
        item=self.entries[self.request_pos]
        if item.get('status')!='ok' or not item.get('usage_known'):
            raise LookupError('An uncertain delivery has no complete response to replay')
        actual=bench.text_digest(json.dumps(body['messages'],sort_keys=True,ensure_ascii=False))
        if actual!=item['prompt_sha256']:raise ValueError('Generation prompt/image payload differs')
        if body['model']!=bench.MODEL or body['seed']!=bench.SEED or body['temperature']!=0:
            raise ValueError('Model generation settings changed')
        self.request_pos+=1
        message={'content':item['content'],'tool_calls':item.get('tool_calls')}
        if item.get('reasoning_present'):message['reasoning_content']='[presence flag only]'
        return {'model':bench.MODEL,'usage':item['usage'],
            'choices':[{'message':message,'finish_reason':item.get('finish_reason')}]}


def ledger_check(entries,budget):
    total=known_input=known_output=unknown=0
    assert len(entries)<=budget.max_calls
    for number,item in enumerate(entries,1):
        assert item['physical_request']==number
        admission=item['admission_prompt_tokens'];output=item['requested_output_tokens']
        assert admission>=0 and 0<output<=budget.max_output_tokens
        assert admission+output<=budget.context_tokens
        assert total+admission+output<=budget.total_tokens
        if item.get('usage_known'):
            used=item['usage'];inp,comp=used['prompt_tokens'],used['completion_tokens']
            assert 0<=inp<=admission and 0<=comp<=output
            assert item['charged_tokens']==inp+comp
            known_input+=inp;known_output+=comp
        else:
            unknown+=1;assert item['charged_tokens']==admission+output
        total+=item['charged_tokens']
    assert total<=budget.total_tokens
    return {'calls':len(entries),'known_input':known_input,'known_output':known_output,
        'unknown_usage_requests':unknown,'charged_or_reserved_tokens':total}


def main():
    started=time.monotonic();out=WORK/'formal'
    frozen=json.loads((out/'freeze.json').read_text())
    paths=[bench.record_path(out,q,a) for q in frozen['question_ids'] for a in bench.ARMS]
    assert len(paths)==60 and all(p.exists() for p in paths)
    assert (out/'audit.json').exists() and (out/'scoring_access.json').exists()
    bench.check(WORK)
    access=json.loads((out/'scoring_access.json').read_text())
    records=[json.loads(p.read_text()) for p in paths]
    scored=json.loads((out/'audit.json').read_text())
    labels=json.loads((WORK/'private_labels/labels.json').read_text())
    questions={r['id']:r for r in bench.read_rows(WORK/'inputs.jsonl')}
    stores,insertion=bench.load_stores(WORK);policy=bench.policies(WORK)
    metrics=bench.official_scorer(WORK)
    totals=Counter();receipts=[];violations=[]
    for record,path in zip(records,paths):
        qid,arm=record['id'],record['arm'];store=stores[qid];question=questions[qid]
        assert record['terminal'] and record['freeze_sha256']==bench.digest(out/'freeze.json')
        assert access['terminal_record_sha256'][path.name]==bench.digest(path)
        assert record['completed_at_unix']<=access['opened_at_unix']
        assert record['model']==bench.MODEL
        stats=ledger_check(record['ledger'],bench.BUDGETS[arm]);totals.update(stats)
        assert record['usage']['charged_tokens']==stats['charged_or_reserved_tokens']
        assert record['usage']['within_budget']
        assert record['source_metadata']==store.metadata()
        assert all(e['actual_model']==bench.MODEL for e in record['response_events'])
        image_count=0
        for event in record['result'].get('image_events',[]):
            assert 1<=event['physical_request']<=len(record['ledger'])
            actual_images=event['images'];image_count+=len(actual_images)
            if arm=='harness':assert len(actual_images)<=3+int(bool(question['image']))
            if arm=='direct_headtail_native':assert len(actual_images)<=16
            if question['image']:assert sum(e.get('query_image') is True for e in actual_images)==1
            for image in actual_images:
                if image.get('query_image'):
                    assert bench.digest(WORK/'data'/question['image'])==image['image_sha256']
                else:
                    state=next(s for s in store.bindings if s.trajectory_id==image['trajectory_id'] and s.state_index==image['state_index'])
                    assert [state.start,state.end]==image['source_state_span']
                    assert bench.digest(store.resolve_image(state.image))==image['image_sha256']
        totals['image_occurrences_verified']+=image_count
        replay='passed';reason=None
        try:
            transport=RecordedTransport(record)
            client=BudgetedClient(transport,bench.BUDGETS[arm])
            adapted=TrajectoryEvidenceClient(client,store,domain_system=policy[question['domain']],
                query_image=question['image'],question_text=question['question'],
                history_image_limit=(len(store.bindings) if arm=='direct_full' else
                    16-int(bool(question['image'])) if arm=='direct_headtail_native' else None),
                image_selection='uniform' if arm=='direct_headtail_native' else 'overlap')
            if arm=='direct_full' and record['status']=='native_capacity_denied':
                assert record['source_text_tokens_exact']>262144+4096
                assert not record['ledger'] and not record['answer']
            elif arm=='harness':
                result=retrieve_trajectory_evidence(store,question['question'],adapted)
                assert result==record['result'],'Controller result differs'
                assert result['answer']==record['answer']
            elif arm=='direct_headtail_native':
                retained=min(len(store.context),int(len(store.context)*240000/record['source_text_tokens_exact']))
                for _ in range(16):
                    windows=bench.headtail_windows(store,retained)
                    messages=bench._messages(question['question'],windows,final=True)
                    adapted.count_messages(messages)
                    if adapted.can_call(messages,1024):break
                    retained=max(2,int(retained*.9))
                assert adapted.complete(messages,1024)==record['answer']
                assert adapted.image_events==record['result']['image_events']
            else:raise LookupError('No complete outcome available for deterministic replay')
            assert transport.count_pos==len(record['count_events'])
            assert transport.request_pos==len(record['ledger'])
            totals['generation_entries_replayed']+=transport.request_pos
            totals['tokenizer_events_replayed']+=transport.count_pos
        except (LookupError,ValueError,AssertionError,IndexError) as exc:
            replay='unavailable_or_mismatch';reason=type(exc).__name__+': '+str(exc)[:100]
            # Preserve the inability to reproduce; never silently count it as pass.
            violations.append({'id':qid,'arm':arm,'stage':'trajectory_replay','reason':reason})
        row=next(r for r in scored['scored'] if r['id']==qid and r['arm']==arm)
        parsed=metrics.extract_boxed_answer(record['answer']);assert row['parsed_answer']==parsed
        expected=False
        if record['status']=='returned' and record['answer'].strip() and not metrics.is_unknown(parsed):
            name=metrics.eval_name(labels[qid]['eval_function'])
            if name.startswith('llm_'):
                ledger_check(row['judge_ledger'],bench.JUDGE_BUDGET)
                if row['judge_error'] is not None:expected=None
                else:
                    value,_=metrics._parse_llm_binary_judgement(row['judge_ledger'][-1]['content']);expected=value==1
            else:expected=bool(metrics.eval_from_spec(labels[qid]['eval_function'],parsed,labels[qid]['answer']))
        assert row['correct'] is expected
        receipts.append({'id':qid,'arm':arm,'record_sha256':bench.digest(path),
            'ledger_passed':True,'source_binding_passed':True,'image_occurrences_verified':image_count,
            'score_recomputed':True,'trajectory_replay':replay,'replay_reason':reason,
            'answer_sha256':bench.text_digest(record['answer'])})
        print(json.dumps({'audited':len(receipts),'expected':60,'replay_unavailable':len(violations)}),flush=True)
    assert sum(r['ledger_passed'] for r in receipts)==60
    report={'schema_version':1,'status':'pass' if not violations else 'partial_replay',
        'freeze_sha256':bench.digest(out/'freeze.json'),'auditor_sha256':bench.digest(__file__),
        'terminal_records':60,'ledgers_passed':60,'scores_recomputed':60,
        'trajectory_passed':sum(r['trajectory_replay']=='passed' for r in receipts),
        'trajectory_unavailable_or_mismatched':len(violations),'totals':dict(totals),
        'receipts':receipts,'violations':violations,'provider_or_tokenizer_calls_made':0,
        'replay_uses_recorded_exact_token_counts_and_frozen_production_controller':True,
        'semantic_entailment_or_gold_truth_independently_verified':False,
        'seconds':time.monotonic()-started}
    bench.atomic(WORK/'offline_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k not in ['receipts','violations']},indent=2))


if __name__=='__main__':main()
