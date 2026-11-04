"""Offline full-run source, billing, scoring and recorded-controller audit."""
from collections import Counter
import json
from pathlib import Path
import sys
import time

from scripts import benchmark_longmemeval_v2_full as f
from scripts.audits.audit_longmemeval_v2 import ledger_check
from scripts.audits.audit_longmemeval_v2_visual_handoff import ExactRecordedTransport
from long_context_agent.visual_anchor_memory import action_target
from long_context_agent.visual_anchor_memory_v3 import normalize_actions

b=f.b


def replay(record,row,store,systems,indexes):
    arm=record['arm'];transport=ExactRecordedTransport(record)
    client=b.BudgetedClient(transport,b.BUDGETS[arm])
    if arm=='direct_headtail_native':
        adapted=b.TrajectoryEvidenceClient(client,store,domain_system=systems[row['domain']],
            question_text=row['question'],query_image=row['image'],history_image_limit=16-int(bool(row['image'])),image_selection='uniform')
        retained=min(len(store.context),int(len(store.context)*240000/record['source_text_tokens_exact']))
        for _ in range(16):
            windows=b.headtail_windows(store,retained);messages=b._messages(row['question'],windows,final=True)
            count=adapted.count_messages(messages)
            if adapted.can_call(messages,4096):break
            retained=max(2,int(retained*min(.99,(262144-4096)/max(1,count))*.99))
        assert adapted.complete(messages,4096)==record['answer']
        assert adapted.image_events==record['result']['image_events']
    elif not row['image']:
        result=f.run_completion_harness(store,row['question'],f.CompletionEvidenceClient(
            client,store,domain_system=systems[row['domain']],question_text=row['question']))
        assert json.loads(json.dumps(result))==record['result']
    else:
        legacy=arm=='harness_visual_anchor_v1';key=(id(store),legacy)
        if key not in indexes:indexes[key]=(b.LegacyVisualIndex if legacy else b.VisualAnchorIndex)(store)
        result=(b.run_legacy_visual if legacy else b.run_visual_anchor)(indexes[key],row['question'],client,
            domain_system=systems[row['domain']],query_image=row['image'])
        assert json.loads(json.dumps(result))==record['result']
    assert transport.request_pos==len(record['ledger'])
    assert transport.count_pos==len(record['count_events'])
    return transport.count_pos


def main(work):
    started=time.time();work=Path(work).resolve();frozen=f.check(work);out=work/'formal'
    audit=json.loads((out/'audit.json').read_text());access=json.loads((out/'scoring_access.json').read_text())
    labels=json.loads((work/'private_labels/labels.json').read_text());scorer=b.official_scorer(work)
    inputs={r['id']:r for r in b.read_rows(work/'inputs.jsonl')}
    stores,_=b.load_stores(work);systems=b.policies(work);indexes={};receipts=[];violations=[];totals=Counter()
    scored={(r['id'],r['arm']):r for r in audit['scored']}
    assert len(scored)==1353
    image_hash_cache={}
    def image_hash(path):
        key=str(path)
        if key not in image_hash_cache:image_hash_cache[key]=b.digest(path)
        return image_hash_cache[key]
    for qid,row in inputs.items():
        store=stores[qid]
        for arm in f.ARMS:
            path=b.record_path(out,qid,arm);record=json.loads(path.read_text());score=scored[qid,arm]
            assert b.digest(path)==access['terminal_record_sha256'][path.name]
            assert record['freeze_sha256']==b.digest(out/'freeze.json') and record['terminal']
            assert record['completed_at_unix']<access['opened_at_unix']
            assert record['source_metadata']==store.metadata()
            assert record['usage']['within_budget']
            stats=ledger_check(record['ledger'],b.BUDGETS[arm])
            assert stats['charged_or_reserved_tokens']==record['usage']['charged_tokens']
            shared=record.get('shared_physical_record')
            if shared:
                assert arm=='harness_visual_anchor_v1' and not row['image']
                source_path=b.record_path(out,qid,'harness_visual_anchor')
                assert shared['file']==source_path.name and shared['sha256']==b.digest(source_path)
                assert record==f.alias_record(json.loads(source_path.read_text()),source_path)
                canonical=scored[qid,'harness_visual_anchor']
                assert score['correct'] is canonical['correct'] and score['judge_ledger']==canonical['judge_ledger']
                assert score['judge_compute_shared'] is True
                totals['shared_records']+=1
                replay_state='shared_identical_path'
            else:
                totals['physical_records']+=1;totals['physical_generations']+=stats['calls']
                totals['physical_tokens']+=stats['charged_or_reserved_tokens']
                assert all(x['actual_model']==b.MODEL for x in record['response_events'])
                for event in record['result'].get('image_events',[]):
                    assert sum(x.get('query_image') is True for x in event['images'])==int(bool(row['image']))
                    cards={c['id']:c for c in event.get('evidence_receipts',[])}
                    for card in cards.values():
                        state=store._rows[card['trajectory_id']]['states'][card['state_index']]
                        field=str(state.get(card['field']) or '')
                        assert b.text_digest(field)==card['field_sha256']
                        assert [b.text_digest(field[a:z]) for a,z in card['field_ranges']]==card['excerpt_sha256']
                        totals['field_receipts']+=1
                    for pair in event.get('transition_receipts',[]):
                        a,z=cards[pair['before']],cards[pair['after']]
                        assert a['trajectory_id']==z['trajectory_id'] and a['state_index']+1==z['state_index']
                        states=store._rows[a['trajectory_id']]['states']
                        raw=states[z['state_index']].get('action');tree=str(states[a['state_index']].get('accessibility_tree') or '')
                        if arm=='harness_visual_anchor':
                            normalized=normalize_actions(raw,tree)
                            assert normalized and json.dumps(normalized,ensure_ascii=False)==pair['incoming_action']
                            assert '\n'.join(e['control'] for e in normalized if e['control'])==pair['observed_control']
                        else:
                            assert str(raw)==pair['incoming_action']
                            assert action_target(raw,tree)[1]==pair['observed_control']
                        totals['transition_receipts']+=1
                    for image in event['images']:
                        if image.get('query_image'):expected_path=store.resolve_image(row['image'])
                        else:
                            state=store._rows[image['trajectory_id']]['states'][image['state_index']]
                            expected_path=store.resolve_image(state['screenshot'])
                        assert image_hash(expected_path)==image['image_sha256'];totals['image_occurrences']+=1
                if record['status'] in ('returned','empty') and all(e.get('status')=='ok' for e in record['ledger']):
                    try:
                        totals['tokenizer_events_replayed']+=replay(record,row,store,systems,indexes)
                        replay_state='passed';totals['physical_replays_passed']+=1
                    except Exception as exc:
                        replay_state='mismatch';violations.append({'id':qid,'arm':arm,'stage':'replay','error_type':type(exc).__name__,'reason':str(exc)[:120]})
                else:
                    replay_state='unavailable_uncertain_or_error';totals['physical_replays_unavailable']+=1
            parsed=scorer.extract_boxed_answer(record['answer']);assert parsed==score['parsed_answer']
            expected=False
            if record['status']=='returned' and record['answer'].strip() and not scorer.is_unknown(parsed):
                if scorer.eval_name(labels[qid]['eval_function']).startswith('llm_'):
                    ledger_check(score['judge_ledger'],b.JUDGE_BUDGET)
                    expected=None if score['judge_error'] else scorer._parse_llm_binary_judgement(score['judge_ledger'][0]['content'])[0]==1
                else:expected=bool(scorer.eval_from_spec(labels[qid]['eval_function'],parsed,labels[qid]['answer']))
            assert score['correct'] is expected
            receipts.append({'id':qid,'arm':arm,'record_sha256':b.digest(path),
                'ledger_passed':True,'score_recomputed':True,'replay':replay_state,'shared':bool(shared)})
        if len(receipts)%75==0:print(json.dumps({'audited_logical_records':len(receipts),'of':1353}),flush=True)
    assert totals['physical_records']==931 and totals['shared_records']==422
    result={'status':'integrity_mismatch' if violations else ('pass_with_unreplayable_failures' if totals['physical_replays_unavailable'] else 'pass'),
        'freeze_sha256':b.digest(out/'freeze.json'),'auditor_sha256':b.digest(__file__),
        'terminal_records':len(receipts),'totals':dict(totals),'receipts':receipts,'violations':violations,
        'provider_or_tokenizer_calls_made':0,'independent_semantic_entailment_validation':False,
        'seconds':time.time()-started}
    b.atomic(work/'offline_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if k not in ('receipts','violations')},indent=2))
    if violations:raise RuntimeError('Replay mismatch; preserve evidence and do not publish as passed')


if __name__=='__main__':main(sys.argv[1] if len(sys.argv)>1 else f.WORK)
