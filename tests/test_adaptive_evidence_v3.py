from __future__ import annotations

from dataclasses import replace
from datetime import date, timedelta
import json

from long_context_agent.adaptive_evidence_v3 import AdaptiveV3Config, run_adaptive_v3


class CappedClient:
    """A shared input+output ledger, including planning and failed reviews."""

    def __init__(self, responder, *, calls=24, tokens=60000, context=2600):
        self.responder = responder
        self.remaining_calls = calls
        self.remaining_tokens = tokens
        self.context_tokens = context
        self.messages = []
        self.usage = []

    def count_text(self, value):
        return max(1, (len(value) + 3) // 4)

    def count_messages(self, messages):
        return 5 + sum(4 + self.count_text(m['content']) for m in messages)

    def can_call(self, messages, max_tokens):
        return self.remaining_calls > 0 and self.count_messages(messages) + max_tokens <= min(self.context_tokens, self.remaining_tokens)

    def complete(self, messages, max_tokens=1024):
        assert self.can_call(messages, max_tokens)
        before = self.remaining_tokens
        reply = self.responder(messages, len(self.messages) + 1)
        assert self.count_text(reply) <= max_tokens
        self.messages.append(messages)
        self.remaining_calls -= 1
        self.remaining_tokens -= self.count_messages(messages) + self.count_text(reply)
        self.usage.append({'before': before, 'after': self.remaining_tokens, 'reserved_output': max_tokens})
        assert self.remaining_tokens >= 0
        return reply


CONFIG = AdaptiveV3Config(planner_output_tokens=512, classification_output_tokens=1024,
                          max_records_per_batch=8, fact_review_output_tokens=512,
                          final_output_tokens=256, final_reserve_tokens=2000,
                          raw_windows_per_round=1)


def logs(values=None, *, count=120, users=None):
    values = values or ['positive' if i % 3 == 0 else 'negative' for i in range(count)]
    users = users or ['7'] * count
    return f'The following lines contain {count} records, one per line.\n' + ''.join(
        f'Date: {date(2024, 1, 1) + timedelta(days=i)} || User: {users[i]} || '
        f'Instance: semantic={values[i]}; source content ' + 'filler ' * 18 + '\n'
        for i in range(count))


def plan(**updates):
    result = {'kind': 'records', 'operation': 'count', 'filters': {}, 'field': 'label',
              'labels': ['positive', 'negative'], 'target': 'positive',
              'answer_template': 'Answer: {value}'}
    result.update(updates)
    return result


def responder_for(task_plan, transform=None):
    def respond(messages, number):
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            return json.dumps(task_plan)
        assert messages[0]['content'].startswith('Classify or extract')
        records = json.loads(messages[-1]['content'].split('\nRECORDS:\n', 1)[1])
        # Compact production input exposes aliases/bodies only. Legacy output
        # may omit source_span; the decoder uses its immutable local binding.
        items = [{'id': r['id'], **({'source_span': r['source_span']} if 'source_span' in r else {}),
                  'value': r['text'].split('semantic=', 1)[1].split(';', 1)[0]} for r in records]
        if transform:
            items = transform(items)
        return json.dumps({'records': items})
    return respond


def test_scoped_count_classifies_only_target_records_and_aggregates_locally():
    users = ['7' if i < 5 else '99' for i in range(120)]
    client = CappedClient(responder_for(plan(filters={'user': '7'})))
    result = run_adaptive_v3({'question': 'How many positive items for user 7? Answer: number',
                              'context': logs(users=users)}, client, CONFIG)
    assert result['answer'] == 'Answer: 2'
    assert result['evidence_complete'] and not result['answer_is_estimate']
    assert result['aggregate']['known_count'] == 5
    assert result['aggregate']['unknown_count'] == 0
    assert result['record_index']['scope_records'] == 5
    assert len(client.messages) == 2
    assert 'User: 99' not in client.messages[1][-1]['content']
    assert result['trace'][-1]['model_final_call'] is False
    assert result['final_fact_loss_count'] == 0


def test_multiple_user_scope_includes_comma_and_conjunction_members():
    users = ['7'] * 3 + ['8'] * 3 + ['9'] * 3 + ['99'] * 111
    for clause, expected, scope in [('user IDs 7 and 8', 2, 6),
                                     ('user IDs 7, 8, and 9', 3, 9),
                                     ('users 7 & 8', 2, 6)]:
        client = CappedClient(responder_for(plan()))
        result = run_adaptive_v3({'question': f'For {clause}, count positive records.',
                                  'context': logs(users=users)}, client, CONFIG)
        assert result['answer'] == f'Answer: {expected}' and result['evidence_complete']
        assert result['record_index']['scope_records'] == scope


def test_all_record_facts_survive_many_batches_and_count_is_not_model_arithmetic():
    client = CappedClient(responder_for(plan()))
    result = run_adaptive_v3({'question': 'How many positive records?', 'context': logs()}, client, CONFIG)
    assert result['answer'] == 'Answer: 40'
    assert result['aggregate']['known_count'] == 120
    assert len(result['record_facts']) == 120
    assert result['fact_state']['known_count'] == 120
    assert result['trace'][-1]['retained_facts'] == 120
    assert result['trace'][-1]['fact_digest'] == result['fact_state']['fact_digest']
    assert result['evidence_complete']
    assert all('Do not count' in m[0]['content'] for m in client.messages[1:])


def test_physical_record_count_uses_index_without_semantic_calls():
    client = CappedClient(responder_for(plan(target=None, labels=[], count_all_records=True)))
    result = run_adaptive_v3({'question': 'How many records?', 'context': logs()}, client, CONFIG)
    assert result['answer'] == 'Answer: 120' and result['evidence_complete']
    assert len(client.messages) == 1
    assert result['aggregate']['known_count'] == 120
    assert len(result['scope_record_refs']) == len(result['evidence_spans']) == 120
    assert len(result['deterministic_field_spans']) == 120
    assert not result['model_exposed_spans']
    assert result['model_source_coverage']['fraction'] == 0
    assert all(ref['source_fields']['user'] == '7' and ref['source_fields']['date_iso']
               for ref in result['scope_record_refs'])


def test_explicit_source_labels_have_complete_refs_even_without_model_facts():
    context = '\n'.join(json.dumps({'user': '7', 'date': str(date(2024, 1, 1) + timedelta(days=i)),
                                    'label': 'positive' if i % 2 else 'negative',
                                    'text': 'original body ' * 20}) for i in range(120))
    client = CappedClient(responder_for(plan()))
    result = run_adaptive_v3({'question': 'Count positive records.', 'context': context}, client, CONFIG)
    assert result['answer'] == 'Answer: 60' and result['evidence_complete']
    assert len(client.messages) == 1 and result['record_facts'] == {}
    assert len(result['scope_record_refs']) == 120
    assert len(result['evidence_spans']) == 120
    assert {ref['source_fields']['label'] for ref in result['scope_record_refs']} == {'positive', 'negative'}
    assert all('original body' not in json.dumps(ref) for ref in result['scope_record_refs'])
    for ref in result['scope_record_refs']:
        start, end = map(int, ref['source_span'].split()[-1].split(':'))
        source_row = json.loads(context[start:end])
        assert source_row['label'] == ref['source_fields']['label']


def test_invalid_spans_and_duplicates_cannot_inflate_counts_or_continue_forever():
    def malformed(items):
        return [dict(item, source_span='source_chars 0:1') for item in items] * 2

    client = CappedClient(responder_for(plan(), malformed))
    result = run_adaptive_v3({'question': 'Count positive records.', 'context': logs()}, client, CONFIG)
    reviews = [t for t in result['trace'] if t['action'] == 'classify_records']
    assert len(reviews) == 2 and result['stop_reason'] == 'no_fact_gain'
    assert not result['evidence_complete']
    assert result['aggregate']['known_count'] == 0
    assert result['aggregate']['unknown_count'] == 120
    assert result['aggregate']['estimate'] is None and result['answer'] == 'Answer: unknown'
    assert all(e['status'] == 'rejected' for r in reviews for e in r['fact_events'])


def test_duplicate_alias_in_same_batch_remains_unknown_and_never_double_counts():
    client = CappedClient(responder_for(plan(filters={'user': '7'}), lambda rows: rows * 2))
    result = run_adaptive_v3({'question': 'Count positives for user 7.',
                              'context': logs(users=['7'] * 3 + ['9'] * 117)}, client, CONFIG)
    assert result['answer'] == 'Answer: unknown' and not result['evidence_complete']
    events = result['trace'][1]['fact_events']
    assert sum(e['status'] == 'accepted' for e in events) == 0
    assert sum(e.get('reason') == 'duplicate_alias' for e in events) == 6
    assert result['aggregate']['known_count'] == 0 and result['aggregate']['unknown_count'] == 3


def test_earliest_and_latest_complete_from_known_chronological_prefix():
    values = ['negative'] * 120
    values[2], values[117] = 'positive', 'positive'
    for operation, expected in [('earliest', '2024-01-03'), ('latest', '2024-04-27')]:
        client = CappedClient(responder_for(plan(operation=operation, answer_template='Date: {value}')))
        result = run_adaptive_v3({'question': f'{operation} positive record date?', 'context': logs(values)},
                                 client, replace(CONFIG, max_records_per_batch=2))
        assert result['answer'] == 'Date: ' + expected
        assert result['evidence_complete'] and not result['aggregate']['coverage_complete']
        assert result['aggregate']['known_count'] == 4
        assert result['aggregate']['unknown_count'] == 116
        assert len(client.messages) == 3


def test_partial_budget_result_preserves_unknown_mass_and_estimate_flag():
    client = CappedClient(responder_for(plan()), calls=2)
    result = run_adaptive_v3({'question': 'How many positive records?', 'context': logs()}, client, CONFIG)
    assert not result['evidence_complete'] and result['answer_is_estimate']
    assert result['aggregate']['value'] is None
    assert result['aggregate']['known_count'] == 8
    assert result['aggregate']['unknown_count'] == 112
    assert result['aggregate']['lower_bound'] <= result['aggregate']['estimate'] <= result['aggregate']['upper_bound']
    assert result['final_fact_loss_count'] == 0


def test_group_frequency_compares_deterministic_filtered_counts():
    values = ['positive'] * 60 + ['negative'] * 60
    task_plan = plan(operation='compare_counts', metric='frequency', empty_scope_is_expected=True, groups=[
        {'name': 'early', 'filters': {'date_before': '2024-03-01'}},
        {'name': 'late', 'filters': {'date_from': '2024-03-01'}}],
        answer_template='Relation: {relation}')
    client = CappedClient(responder_for(task_plan))
    result = run_adaptive_v3({'question': 'Are positives more frequent before March?', 'context': logs(values)}, client, CONFIG)
    assert result['answer'] == 'Relation: more common than'
    assert result['evidence_complete']
    assert result['aggregate']['metric'] == 'frequency'
    assert len(result['aggregate']['groups']) == 2


def test_empty_group_frequency_is_undefined_instead_of_zero_or_crash():
    task_plan = plan(operation='compare_counts', metric='frequency', empty_scope_is_expected=True, groups=[
        {'name': 'empty', 'filters': {'date_before': '2020-01-01'}},
        {'name': 'all', 'filters': {'date_from': '2020-01-01'}}])
    client = CappedClient(responder_for(task_plan))
    result = run_adaptive_v3({'question': 'Compare positive frequency before 2020-01-01 and on or after 2020-01-01.', 'context': logs()}, client, CONFIG)
    assert not result['evidence_complete']
    assert result['aggregate']['value'] is None and result['aggregate']['estimate'] is None
    assert result['answer'] == 'Answer: unknown'


def narrative():
    return ''.join(f'key record {i}: information token {i}. ' + 'padding ' * 175 + '\n' for i in range(90))


def narrative_responder(messages, number):
    if messages[0]['content'].startswith('Plan evidence acquisition'):
        return json.dumps({'kind': 'facts', 'queries': ['key record information']})
    payload = json.loads(messages[-1]['content'])
    if 'raw_evidence' not in payload:
        return 'best answer'
    return json.dumps({'claims': [], 'needs': [{'id': 'same-gap', 'query': 'key record information'}], 'complete': False})


def test_repeated_gap_without_information_gain_stops_after_two_reviews():
    client = CappedClient(narrative_responder)
    result = run_adaptive_v3({'question': 'Find the key record information.', 'context': narrative()}, client, CONFIG)
    reviews = [t for t in result['trace'] if t['action'] == 'fact_review']
    assert len(reviews) == 2
    assert result['stop_reason'] == 'no_fact_gain'
    assert result['answer'] == 'best answer'
    assert client.remaining_calls == 20
    assert result['trace'][-1]['action'] == 'final_answer'


def test_final_contains_early_source_fact_and_token_reserve_after_later_reviews():
    review_count = 0

    def respond(messages, number):
        nonlocal review_count
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            return json.dumps({'kind': 'facts', 'queries': ['key record information']})
        payload = json.loads(messages[-1]['content'])
        if 'raw_evidence' not in payload:
            assert payload['retained_fact_ledger'][0]['candidates'][0]['value'] == 'durable early fact'
            assert len(payload['retained_fact_ledger']) == 1
            assert client.remaining_tokens >= client.count_messages(messages) + CONFIG.final_output_tokens
            return 'durable early fact'
        review_count += 1
        evidence = payload['raw_evidence'][0]
        claim = {'subject': 'entity', 'predicate': 'resolution', 'value': 'durable early fact',
                 'quote': evidence['text'][:60], 'source_span': evidence['source_span']}
        return json.dumps({'claims': [claim] if review_count == 1 else [],
                           'needs': [{'id': 'remaining', 'query': 'key record information'}], 'complete': False})

    client = CappedClient(respond, tokens=7000)
    result = run_adaptive_v3({'question': 'Find the key record information.', 'context': narrative()}, client, CONFIG)
    assert result['answer'] == 'durable early fact'
    assert len(result['retained_facts']) == 1 and result['final_fact_loss_count'] == 0
    assert result['trace'][-1]['retained_fact_count'] == 1
    assert review_count >= 1


def test_narrative_invalid_quote_is_rejected_and_small_budget_keeps_final():
    def respond(messages, number):
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            return json.dumps({'kind': 'facts'})
        payload = json.loads(messages[-1]['content'])
        if 'raw_evidence' not in payload:
            return 'unknown'
        return json.dumps({'claims': [{'subject': 'x', 'predicate': 'p', 'value': 12,
                                      'quote': 'invented outside source', 'source_span': payload['raw_evidence'][0]['source_span']}],
                           'complete': True})

    client = CappedClient(respond, tokens=3000, calls=3)
    result = run_adaptive_v3({'question': 'Find exact fact.', 'context': narrative()}, client, CONFIG)
    assert not result['evidence_complete'] and not result['retained_facts']
    assert result['trace'][-1]['action'] == 'final_answer'
    assert client.remaining_tokens >= 0


def test_invalid_claim_cannot_close_query_even_when_another_claim_is_valid():
    def respond(messages, number):
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            return json.dumps({'kind': 'facts'})
        payload = json.loads(messages[-1]['content'])
        if 'raw_evidence' not in payload:
            return 'tentative'
        raw = payload['raw_evidence'][0]
        valid = {'subject': 'x', 'predicate': 'p', 'value': 'found',
                 'quote': raw['text'][:30], 'source_span': raw['source_span']}
        invalid = dict(valid, subject='y', quote='invented outside source')
        return json.dumps({'claims': [valid, invalid], 'complete': True})

    client = CappedClient(respond, calls=3)
    result = run_adaptive_v3({'question': 'Find exact fact.', 'context': narrative()}, client, CONFIG)
    assert len(result['retained_facts']) == 1
    assert result['rejected_facts'] and not result['evidence_complete']


def test_independent_sources_are_retained_as_corroboration_without_duplicate_fact():
    def respond(messages, number):
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            return json.dumps({'kind': 'facts'})
        payload = json.loads(messages[-1]['content'])
        if 'raw_evidence' not in payload:
            fact = payload['retained_fact_ledger'][0]['candidates'][0]
            assert len(fact['supporting_sources']) == 1
            return 'confirmed'
        raw = payload['raw_evidence'][0]
        return json.dumps({'claims': [{'subject': 'x', 'predicate': 'p', 'value': 'found',
                                      'quote': raw['text'][:30], 'source_span': raw['source_span']}],
                           'needs': [{'id': 'more', 'query': 'key record information'}], 'complete': False})

    client = CappedClient(respond, calls=4)
    result = run_adaptive_v3({'question': 'Find exact fact.', 'context': narrative()}, client, CONFIG)
    assert len(result['retained_facts']) == 1 and result['answer'] == 'confirmed'
    reviews = [t for t in result['trace'] if t['action'] == 'fact_review']
    assert reviews[1]['fact_events'][0]['status'] == 'corroborated'


def test_oversize_atomic_identity_and_values_are_rejected_without_tail_collapse():
    for field, common_length in [('value', 400), ('subject', 120), ('predicate', 120)]:
        def respond(messages, number):
            if messages[0]['content'].startswith('Plan evidence acquisition'):
                return json.dumps({'kind': 'facts'})
            payload = json.loads(messages[-1]['content'])
            if 'raw_evidence' not in payload:
                assert payload['retained_fact_ledger'] == []
                return 'unknown'
            raw = payload['raw_evidence'][0]
            claim = {'subject': 'x', 'predicate': 'p', 'value': 'found',
                     'quote': raw['text'][:30], 'source_span': raw['source_span']}
            # These claims differ only after the former truncation boundary.
            first, second = dict(claim), dict(claim)
            first[field], second[field] = 'a' * common_length + 'X', 'a' * common_length + 'Y'
            return json.dumps({'claims': [first, second], 'complete': True})

        client = CappedClient(respond, calls=3)
        result = run_adaptive_v3({'question': 'Find the exact fact.', 'context': narrative()}, client, CONFIG)
        assert not result['retained_facts'] and not result['evidence_complete']
        assert len(result['rejected_facts']) == 2
        assert all(event['reason'] == 'oversize_atom' for event in result['rejected_facts'])


def test_short_source_direct_pass_uses_one_budgeted_call():
    client = CappedClient(lambda messages, number: '42')
    result = run_adaptive_v3({'context': 'The answer is 42.', 'question': 'What is the answer?'}, client, CONFIG)
    assert result['mode'] == 'direct_pass_v3' and result['answer'] == '42'
    assert len(client.messages) == 1
