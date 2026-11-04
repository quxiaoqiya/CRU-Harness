"""Independent source/transition checks and zero-I/O recorded controller replay."""
from pathlib import Path
import json
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts import benchmark_longmemeval_v2_visual_anchor as b
from scripts.audits.audit_longmemeval_v2 import RecordedTransport, ledger_check
from long_context_agent.trajectory_memory_completion import (
    PerCallBudgetedClient, CompletionEvidenceClient, run_completion_harness,
)
from long_context_agent.trajectory_memory import TrajectoryEvidenceClient
from long_context_agent.visual_anchor_memory import (
    VisualAnchorIndex, run_visual_anchor, action_target,
)


class ExactRecordedTransport(RecordedTransport):
    def request(self, path, body=None):
        assert body['max_tokens'] == self.entries[self.request_pos]['requested_output_tokens']
        assert body['chat_template_kwargs'] == {'enable_thinking': False, 'preserve_thinking': False}
        return super().request(path, body)


def main(work):
    start = time.monotonic()
    work = Path(work).resolve()
    frozen = b.check(work)
    if frozen.get('visual_contracts_version') == 2:
        from long_context_agent.visual_anchor_memory_v2 import (
            VisualAnchorIndex as IndexClass, run_visual_anchor as run_anchor,
            observed_effect,
        )
    else:
        IndexClass, run_anchor = VisualAnchorIndex, run_visual_anchor
    out = work/'formal'
    scored = json.loads((out/'audit.json').read_text())
    by = {(r['id'], r['arm']): r for r in scored['scored']}
    access = json.loads((out/'scoring_access.json').read_text())
    labels = json.loads((work/'private_labels/labels.json').read_text())
    questions = {r['id']: r for r in b.read_rows(work/'inputs.jsonl')}
    stores, _ = b.load_stores(work)
    policies = b.policies(work)
    scorer = b.official_scorer(work)
    indexes = {}
    receipts, failures = [], []
    totals = {'generations': 0, 'tokens': 0, 'image_occurrences': 0,
              'field_receipts': 0, 'transition_receipts': 0, 'counts_replayed': 0}
    for qid in frozen['question_ids']:
        question, store = questions[qid], stores[qid]
        for arm in frozen['arms']:
            path = b.record_path(out, qid, arm)
            record = json.loads(path.read_text())
            assert record['terminal'] and record['freeze_sha256'] == b.digest(out/'freeze.json')
            assert record['completed_at_unix'] < access['opened_at_unix']
            assert b.digest(path) == access['terminal_record_sha256'][path.name]
            stats = ledger_check(record['ledger'], b.BUDGETS[arm])
            assert record['usage']['charged_tokens'] == stats['charged_or_reserved_tokens']
            assert record['usage']['within_budget'] and record['source_metadata'] == store.metadata()
            assert all(x['actual_model'] == b.MODEL for x in record['response_events'])
            for event in record['result'].get('image_events', []):
                assert sum(im.get('query_image') is True for im in event['images']) == 1
                if arm in ('harness_visual_anchor', 'visual_query_only'):
                    assert len(event['images']) <= 4
                if arm == 'visual_query_only':
                    assert len(event['images']) == 1 and not event['evidence_receipts'] and not event['transition_receipts']
                cards = {c['id']: c for c in event.get('evidence_receipts', [])}
                for card in cards.values():
                    state = store._rows[card['trajectory_id']]['states'][card['state_index']]
                    field = str(state.get(card['field']) or '')
                    assert b.text_digest(field) == card['field_sha256']
                    assert [b.text_digest(field[a:z]) for a, z in card['field_ranges']] == card['excerpt_sha256']
                    totals['field_receipts'] += 1
                for pair in event.get('transition_receipts', []):
                    before, after = cards[pair['before']], cards[pair['after']]
                    assert before['trajectory_id'] == after['trajectory_id']
                    assert before['state_index'] + 1 == after['state_index']
                    states = store._rows[before['trajectory_id']]['states']
                    assert pair['incoming_action'] == str(states[after['state_index']].get('action', ''))
                    target = action_target(pair['incoming_action'], str(states[before['state_index']].get('accessibility_tree') or ''))
                    assert target and target[1] == pair['observed_control']
                    if frozen.get('visual_contracts_version') == 2:
                        assert pair['observed_effect'] == observed_effect(
                            states[before['state_index']], states[after['state_index']], pair['incoming_action'])
                    totals['transition_receipts'] += 1
                for image in event['images']:
                    if image.get('query_image'):
                        assert b.digest(work/'data'/question['image']) == image['image_sha256']
                    else:
                        state = next(x for x in store.bindings if x.trajectory_id == image['trajectory_id'] and x.state_index == image['state_index'])
                        assert list((state.start, state.end)) == image['source_state_span']
                        assert b.digest(store.resolve_image(state.image)) == image['image_sha256']
                    totals['image_occurrences'] += 1
            replay = True
            try:
                transport = ExactRecordedTransport(record)
                client = PerCallBudgetedClient(transport, b.BUDGETS[arm])
                if arm in ('harness_visual_anchor', 'visual_query_only'):
                    if id(store) not in indexes:
                        indexes[id(store)] = IndexClass(store)
                    result = run_anchor(indexes[id(store)], question['question'], client,
                        domain_system=policies[question['domain']], query_image=question['image'], query_only=arm=='visual_query_only')
                    assert not result['pinned_omitted']
                    assert json.loads(json.dumps(result)) == record['result'], 'Visual controller replay differs'
                elif arm == 'harness_completion':
                    adapted = CompletionEvidenceClient(client, store, domain_system=policies[question['domain']],
                        question_text=question['question'], query_image=question['image'])
                    assert run_completion_harness(store, question['question'], adapted) == record['result']
                else:
                    assert arm == 'direct_headtail_native'
                    adapted = TrajectoryEvidenceClient(client, store, domain_system=policies[question['domain']],
                        question_text=question['question'], query_image=question['image'],
                        history_image_limit=16-int(bool(question['image'])), image_selection='uniform')
                    retained = min(len(store.context), int(len(store.context)*240000/record['source_text_tokens_exact']))
                    for _ in range(16):
                        windows = b.headtail_windows(store, retained)
                        messages = b._messages(question['question'], windows, final=True)
                        count = adapted.count_messages(messages)
                        if adapted.can_call(messages, 4096):
                            break
                        retained = max(2, int(retained*min(.99, (262144-4096)/max(1,count))*.99))
                    assert adapted.complete(messages, 4096) == record['answer']
                    assert adapted.image_events == record['result']['image_events']
                assert transport.count_pos == len(record['count_events'])
                assert transport.request_pos == len(record['ledger'])
                totals['counts_replayed'] += transport.count_pos
            except (AssertionError, ValueError, IndexError, LookupError) as exc:
                replay = False
                failures.append({'id': qid, 'arm': arm, 'error_type': type(exc).__name__, 'reason': str(exc)[:100]})
            row, label = by[qid, arm], labels[qid]
            parsed = scorer.extract_boxed_answer(record['answer'])
            assert parsed == row['parsed_answer']
            expected = False
            if record['status'] == 'returned' and record['answer'].strip() and not scorer.is_unknown(parsed):
                if scorer.eval_name(label['eval_function']).startswith('llm_'):
                    ledger_check(row['judge_ledger'], b.JUDGE_BUDGET)
                    expected = None if row['judge_error'] else scorer._parse_llm_binary_judgement(row['judge_ledger'][-1]['content'])[0] == 1
                else:
                    expected = bool(scorer.eval_from_spec(label['eval_function'], parsed, label['answer']))
            assert row['correct'] is expected
            totals['generations'] += stats['calls']
            totals['tokens'] += stats['charged_or_reserved_tokens']
            receipts.append({'id': qid, 'arm': arm, 'record_sha256': b.digest(path),
                'ledger_passed': True, 'score_recomputed': True, 'replay_passed': replay})
    report = {'status': 'pass' if not failures else 'partial_replay',
        'freeze_sha256': b.digest(out/'freeze.json'), 'auditor_sha256': b.digest(__file__),
        'terminal_records': len(receipts), 'ledgers_passed': len(receipts),
        'scores_recomputed': len(receipts), 'replay_passed': sum(r['replay_passed'] for r in receipts),
        'totals': totals, 'receipts': receipts, 'failures': failures,
        'provider_or_tokenizer_calls_made': 0, 'independent_semantic_entailment_validation': False,
        'seconds': time.monotonic()-start}
    b.atomic(work/'offline_audit.json', report)
    print(json.dumps({k:v for k,v in report.items() if k != 'receipts'}, indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
