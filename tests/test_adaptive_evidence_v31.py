"""Planning and integrated structured-source regression tests; synthetic fixtures only."""
from dataclasses import asdict, replace
import importlib.util
import hashlib
import json
from pathlib import Path
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


from long_context_agent import adaptive_evidence_v3 as candidate
regression = load('_v31_synthetic_regression_fixtures', ROOT / 'tests' / 'test_adaptive_evidence_v3.py')
regression.run_adaptive_v3 = candidate.run_adaptive_v3
regression.AdaptiveV3Config = candidate.AdaptiveV3Config
regression.CONFIG = candidate.AdaptiveV3Config(**asdict(regression.CONFIG))
CappedClient, logs, plan, CONFIG = regression.CappedClient, regression.logs, regression.plan, regression.CONFIG


@pytest.mark.parametrize('name', [name for name in vars(regression) if name.startswith('test_')
                                 and name != 'test_empty_group_frequency_is_undefined_instead_of_zero_or_crash'])
def test_existing_synthetic_safety_regressions(name):
    getattr(regression, name)()


def model_with_plans(plans):
    sent_plans = []
    classify = regression.responder_for(plan())

    def respond(messages, number):
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            index = len(sent_plans)
            sent_plans.append(messages)
            return json.dumps(plans[min(index, len(plans) - 1)])
        if messages[0]['content'].startswith('Classify or extract'):
            return classify(messages, number)
        payload = json.loads(messages[-1]['content'])
        return json.dumps({'claims': [], 'needs': [], 'complete': False}) if 'raw_evidence' in payload else 'unknown'
    return respond, sent_plans


def run_plans(plans, *, question='Count positive records.', context=None, calls=24, config=CONFIG):
    responder, sent = model_with_plans(plans)
    client = CappedClient(responder, calls=calls)
    result = candidate.run_adaptive_v3({'context': context or logs(), 'question': question}, client, config)
    return result, client, sent


@pytest.mark.parametrize('bad_filter,code', [
    ({'label': 'positive'}, 'semantic_filter_unsupported'),
    ({'category': 'positive'}, 'semantic_filter_unsupported'),
    ({'body': 'positive'}, 'text_filter_unsupported'),
    ({'unavailable_field': 'positive'}, 'source_filter_unsupported'),
    ({'user': {'contains': '7'}}, 'filter_operator_unsupported'),
])
def test_bad_filter_gets_one_directed_budgeted_repair(bad_filter, code):
    result, client, sent = run_plans([plan(filters=bad_filter), plan()])
    assert result['answer'] == 'Answer: 40' and result['evidence_complete']
    diagnostic = result['planning_diagnostics']
    assert diagnostic['planner_calls'] == 2 and diagnostic['repair_attempted']
    assert diagnostic['validation_errors'][0]['code'] == code
    assert code in sent[1][-1]['content']
    assert client.remaining_calls == 24 - len(client.messages)
    assert any(t['action'] == 'plan_repair' for t in result['trace'])
    assert result['final_fact_loss_count'] == 0


def test_invalid_repair_falls_back_without_a_second_repair_or_zero_answer():
    bad = plan(filters={'label': 'positive'})
    result, client, sent = run_plans([bad, bad, plan()], config=replace(CONFIG, max_plan_repairs=99))
    assert len(sent) == 2
    assert result['mode'] == 'adaptive_fact_memory_v3'
    assert result['answer'] == 'unknown' and not result['evidence_complete']
    assert result['planning_diagnostics']['fallback_reason'] == 'invalid_plan_after_one_repair'
    assert len(result['planning_diagnostics']['validation_errors']) == 2
    assert client.remaining_tokens >= 0


def test_facts_kind_does_not_inherit_irrelevant_record_schema_errors():
    raw = {'kind': 'facts', 'operation': 'not-an-operator', 'filters': 'irrelevant',
           'groups': {'irrelevant': True}, 'queries': ['narrative evidence']}
    result, client, sent = run_plans([raw], context=regression.narrative())
    assert len(sent) == 1 and result['plan']['kind'] == 'facts'
    assert not result['planning_diagnostics']['validation_errors']
    assert not result['planning_diagnostics']['repair_attempted']


@pytest.mark.parametrize('bad,code', [
    ({**plan(), 'kind': 'record_aggregation'}, 'kind_invalid'),
    ({**plan(), 'operation': 'semantic_counting'}, 'operation_unsupported'),
    ({**plan(), 'filters': ['user=7']}, 'filters_not_object'),
])
def test_record_kind_operation_and_filter_grammar_are_not_silently_misclassified(bad, code):
    result, client, sent = run_plans([bad, plan()])
    assert len(sent) == 2 and result['evidence_complete']
    assert result['planning_diagnostics']['validation_errors'][0]['code'] == code
    assert result['mode'] == 'adaptive_fact_aggregation_v3'


def test_empty_scope_requires_repair_and_is_not_silently_zero():
    bad = plan(filters={'user': '999'})
    result, client, sent = run_plans([bad, bad])
    assert len(sent) == 2 and result['answer'] == 'unknown'
    assert not result['evidence_complete']
    assert result['planning_diagnostics']['validation_errors'][0]['code'] == 'scope_empty_unconfirmed'


def test_expected_empty_scope_needs_literal_question_grounding_and_complete_index():
    first = plan(filters={'user': '999'})
    confirmed = {**first, 'empty_scope_is_expected': True}
    result, client, sent = run_plans([first, confirmed], question='For user 999, count positive records.')
    assert result['answer'] == 'Answer: 0' and result['evidence_complete']
    assert len(sent) == 2
    assert result['planning_diagnostics']['scope_validation']['scopes'][0]['expected_empty_confirmed']
    invalid, _, _ = run_plans([confirmed, confirmed], question='Count positive records.')
    assert invalid['mode'] == 'adaptive_fact_memory_v3' and not invalid['evidence_complete']


def test_all_uncertain_scope_cannot_be_treated_as_no_data():
    context = '\n'.join(json.dumps({'date': 'invalid-date', 'user': '7', 'text': 'padding ' * 40}) for _ in range(120))
    bad = plan(filters={'date_before': '2024-01-01'})
    result, client, sent = run_plans([bad, bad], question='Count positives before 2024-01-01.', context=context)
    assert len(sent) == 2 and not result['evidence_complete']
    assert result['planning_diagnostics']['validation_errors'][0]['code'] == 'scope_all_uncertain'
    assert result['answer'] == 'unknown'


def test_existing_source_fields_still_aggregate_without_semantic_requests():
    context = '\n'.join(json.dumps({'user': '7', 'site': 'west', 'score': 2,
                                    'date': '2024-01-01', 'text': 'padding ' * 40}) for _ in range(120))
    numeric = plan(operation='sum', field='score', target=None, labels=[], filters={'site': 'west'})
    result, client, sent = run_plans([numeric], context=context)
    assert result['answer'] == 'Answer: 240' and result['evidence_complete']
    assert len(client.messages) == 1
    assert all(r['source_fields']['score'] == 2 for r in result['scope_record_refs'])


def test_repair_never_spends_the_last_final_call():
    result, client, sent = run_plans([plan(filters={'label': 'positive'})], calls=2)
    assert len(sent) == 1 and len(client.messages) == 2
    assert result['trace'][-1]['action'] == 'final_answer'
    assert result['planning_diagnostics']['fallback_reason'] == 'invalid_plan_repair_not_admitted'


@pytest.mark.parametrize('operation', ['mode', 'least_common', 'group_counts'])
@pytest.mark.parametrize('invalid_args,code', [
    ({'target': 'label'}, 'closed_taxonomy_arguments'),
    ({'compare': ['amber', 'teal']}, 'closed_taxonomy_arguments'),
    ({'groups': [{'name': 'extra', 'filters': {'user': '7'}}]}, 'closed_taxonomy_arguments'),
    ({'labels': ['amber', 'teal', 'violet', '__other__']}, 'closed_taxonomy_reserved_class'),
    ({'labels': ['amber', 'teal', 'violet', 'label']}, 'taxonomy_placeholder_class'),
])
def test_closed_taxonomy_contract_rejects_pollution_instead_of_adding_zero_classes(operation, invalid_args, code):
    task = plan(operation=operation, target=None, labels=['amber', 'teal', 'violet'])
    with pytest.raises(candidate.PlanValidationError) as error:
        candidate._normal_plan({**task, **invalid_args}, 'Use the three actual source categories.')
    assert error.value.code == code


def test_bad_target_and_complement_get_one_repair_then_true_least_common_label():
    labels = ['amber', 'teal', 'violet']
    values = ['amber'] * 6 + ['teal'] * 5 + ['violet'] * 4 + ['amber'] * 105
    users = ['7'] * 15 + ['99'] * 105
    good = plan(operation='least_common', filters={'user': '7'}, target=None,
                labels=labels, answer_template='Label: {label}')
    bad = {**good, 'target': 'label', 'labels': [*labels, '__other__']}
    result, client, sent = run_plans([bad, good], context=logs(values, users=users),
                                     question='For user 7, which of amber, teal, and violet is least common?')
    assert len(sent) == 2 and result['planning_diagnostics']['repair_attempted']
    assert result['planning_diagnostics']['validation_errors'][0]['code'] == 'closed_taxonomy_arguments'
    assert result['plan']['labels'] == labels and result['plan']['target'] is None
    assert result['aggregate']['counts'] == {'amber': 6, 'teal': 5, 'violet': 4}
    assert result['answer'] == 'Label: violet' and result['evidence_complete']
    assert '__other__' not in result['aggregate']['counts'] and 'label' not in result['aggregate']['counts']
    assert result['trace'][-1]['model_final_call'] is False
    assert len(result['record_facts']) == 15 and result['final_fact_loss_count'] == 0


def test_unrepaired_closed_taxonomy_pollution_cannot_report_complete_zero_class():
    bad = plan(operation='least_common', target='label', labels=['amber', 'teal', 'violet', '__other__'])
    result, client, sent = run_plans([bad, bad])
    assert len(sent) == 2 and result['mode'] == 'adaptive_fact_memory_v3'
    assert result['answer'] == 'unknown' and not result['evidence_complete']
    assert result['planning_diagnostics']['fallback_reason'] == 'invalid_plan_after_one_repair'


def test_explicit_user_union_cannot_be_silently_narrowed_to_first_user():
    context = logs(users=['7'] * 3 + ['8'] * 3 + ['99'] * 114)
    bad, good = plan(filters={'user': '7'}), plan(filters={'user_ids': ['7', '8']})
    result, client, sent = run_plans([bad, good], context=context,
                                     question='For the subset associated with user IDs 7 and 8, count positive records.')
    assert len(sent) == 2 and result['answer'] == 'Answer: 2'
    assert result['evidence_complete'] and result['record_index']['scope_records'] == 6
    assert result['planning_diagnostics']['validation_errors'][0]['code'] == 'user_scope_union_mismatch'


def test_per_user_groups_may_partition_the_complete_requested_union():
    context = logs(users=['7'] * 3 + ['8'] * 3 + ['99'] * 114)
    groups = [{'name': 'left', 'filters': {'user_ids': ['7']}},
              {'name': 'right', 'filters': {'user_ids': ['8']}}]
    result, client, sent = run_plans([plan(operation='compare_counts', groups=groups)], context=context,
                                     question='For user IDs 7 and 8, compare positive record counts across these users.')
    assert len(sent) == 1 and result['evidence_complete']
    assert result['aggregate']['value'] == 'same frequency as'
    assert result['record_index']['scope_records'] == 6


@pytest.mark.parametrize('projection,expected', [('larger_group', '8'), ('smaller_group', '7')])
def test_comparison_can_return_source_user_key_instead_of_relation(projection, expected):
    context = logs(users=['7'] * 3 + ['8'] * 6 + ['99'] * 111)
    groups = [{'name': 'left', 'filters': {'user': '7'}}, {'name': 'right', 'filters': {'user': '8'}}]
    task = plan(operation='compare_counts', groups=groups, comparison_return=projection,
                answer_template='User: {value}')
    result, client, sent = run_plans([task], context=context,
                                     question='Among user IDs 7 and 8, which user has the requested positive record count?')
    assert result['evidence_complete'] and result['answer'] == 'User: ' + expected
    assert result['aggregate']['relation'] == 'less common than'
    assert result['aggregate']['ties'] == []
    assert [g['key'] for g in result['aggregate']['groups']] == ['7', '8']


def test_semantic_target_cannot_be_aggregated_as_numeric_user_id_dimension():
    bad = plan(operation='compare_counts', field='user', groups=[
        {'name': 'left', 'filters': {'user': '7'}}, {'name': 'right', 'filters': {'user': '8'}}])
    with pytest.raises(candidate.PlanValidationError) as error:
        candidate._normal_plan(bad, 'Compare positive counts for user IDs 7 and 8.',
                               candidate.parse_records(logs(users=['7'] * 60 + ['8'] * 60)))
    assert error.value.code == 'target_dimension_mismatch'


def grouped_dates(*, invalid_date=False):
    dates = ['2024-01-01'] * 6 + ['2024-01-02'] * 5 + ['2024-01-03'] * 4 + ['2024-02-01'] * 105
    if invalid_date:
        dates[14] = 'invalid-date'
    return '\n'.join(json.dumps({'user': '7' if i < 15 else '99', 'date': day,
                                 'text': 'raw source body ' * 20}) for i, day in enumerate(dates))


@pytest.mark.parametrize('comparison,expected', [('eq', 1), ('ne', 2), ('gt', 1), ('gte', 2), ('lt', 1), ('lte', 2)])
def test_group_counts_having_then_count_groups_is_deterministic(comparison, expected):
    task = plan(operation='group_counts', field='date_iso', filters={'user': '7'}, target=None, labels=[],
                postprocess={'operation': 'count_groups', 'having': {comparison: 5}})
    result, client, sent = run_plans([task], context=grouped_dates(),
                                     question=f'For user 7, how many dates have instance count {comparison} 5?')
    assert result['answer'] == f'Answer: {expected}' and result['evidence_complete']
    assert result['aggregate']['base_operation'] == 'group_counts'
    assert result['aggregate']['counts'] == {'2024-01-01': 6, '2024-01-02': 5, '2024-01-03': 4}
    assert len(client.messages) == 1 and result['trace'][-1]['model_final_call'] is False


def test_group_keys_postprocess_returns_only_satisfying_keys():
    task = plan(operation='group_counts', field='date_iso', filters={'user': '7'}, target=None, labels=[],
                postprocess={'operation': 'group_keys', 'having': {'eq': 5}})
    result, client, sent = run_plans([task], context=grouped_dates(), question='For user 7, which dates have exactly 5 instances?')
    assert result['aggregate']['value'] == ['2024-01-02']
    assert result['answer'] == 'Answer: 2024-01-02' and result['evidence_complete']


def test_unknown_group_metadata_does_not_close_second_order_aggregate():
    task = plan(operation='group_counts', field='date_iso', filters={'user': '7'}, target=None, labels=[],
                postprocess={'operation': 'count_groups', 'having': {'eq': 5}})
    result, client, sent = run_plans([task], context=grouped_dates(invalid_date=True),
                                     question='For user 7, how many dates have exactly 5 instances?')
    assert not result['evidence_complete'] and result['answer_is_estimate']
    assert result['aggregate']['unknown_count'] == 1
    assert result['aggregate']['value'] is None and result['aggregate']['estimate'] is None
    assert result['answer'] == 'Answer: unknown' and result['stop_reason'] == 'unresolved_source_metadata'
    assert len(client.messages) == 1


def test_partial_semantic_group_counts_cannot_be_reused_as_exact_group_count():
    task = plan(operation='group_counts', labels=['positive', 'negative'], target=None,
                postprocess={'operation': 'count_groups', 'having': {'gt': 3}})
    result, client, sent = run_plans([task], calls=2, question='How many labels have more than 3 records?')
    assert result['aggregate']['unknown_count'] > 0 and not result['evidence_complete']
    assert result['aggregate']['value'] is None and result['aggregate']['estimate'] is None


def test_full_source_taxonomy_is_kept_when_only_one_class_is_targeted():
    task = plan(target='wanted', labels=['wanted', 'native_second', 'native_third'], taxonomy_complete=True,
                filters={'user': '7'})
    values = ['wanted', 'native_second', 'native_third'] + ['native_second'] * 117
    result, client, sent = run_plans([task], context=logs(values, users=['7'] * 3 + ['99'] * 117),
                                     question='Source classes are wanted, native_second, native_third. Count wanted for user 7.')
    assert result['plan']['labels'] == task['labels'] and '__other__' not in result['plan']['labels']
    assert result['aggregate']['known_count'] == 3 and result['answer'] == 'Answer: 1'
    assert result['evidence_complete'] and len(client.messages) == 2


def test_out_of_enum_retry_contains_prior_error_and_unlisted_class_stays_unknown():
    classes = []
    task = plan(target='wanted', labels=['wanted'], filters={'user': '7'})

    def respond(messages, number):
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            return json.dumps(task)
        records = json.loads(messages[-1]['content'].split('\nRECORDS:\n', 1)[1])
        classes.append(records)
        if len(classes) == 2:
            retry = messages[-1]['content'].split('\nRETRY_FEEDBACK:\n', 1)[1].split('\nRECORDS:\n', 1)[0]
            prior = json.loads(retry)['r0']
            assert prior['reason'] == 'value_outside_taxonomy'
            assert prior['previous_value'] == 'genuinely_unlisted_class'
            assert prior['allowed_labels'] == ['wanted', '__other__']
        return json.dumps({'records': [{'id': r['id'], 'value': 'genuinely_unlisted_class'} for r in records]})

    client = CappedClient(respond)
    result = candidate.run_adaptive_v3({'question': 'Count wanted for user 7.',
                                       'context': logs(users=['7'] + ['99'] * 119)}, client, CONFIG)
    assert len(classes) == 2  # Identical source rows, different explicit retry feedback.
    assert not result['evidence_complete'] and result['aggregate']['known_count'] == 0
    assert result['aggregate']['unknown_count'] == 1 and result['answer'] == 'Answer: unknown'
    assert result['stop_reason'] == 'no_fact_gain'


def compact_responder(messages, number):
    if messages[0]['content'].startswith('Plan evidence acquisition'):
        return json.dumps(plan())
    user = messages[-1]['content']
    rows = json.loads(user.split('\nRECORDS:\n', 1)[1])
    task = json.loads(user.split('\nTASK PLAN:\n', 1)[1].split('\nSOURCE DEFINITIONS:\n', 1)[0])
    assert all(set(row) == {'id', 'text'} for row in rows)
    values = [task['labels'].index(row['text'].split('semantic=', 1)[1].split(';', 1)[0]) for row in rows]
    return json.dumps({'values': values})


def test_compact_vector_binds_full_source_and_reports_body_only_exposure_in_wire_order():
    context = logs()
    client = CappedClient(compact_responder, context=4000)
    result = candidate.run_adaptive_v3({'context': context, 'question': 'Count positive records.'}, client,
                                       replace(CONFIG, max_records_per_batch=16))
    assert result['answer'] == 'Answer: 40' and result['evidence_complete']
    batches = [item for item in result['trace'] if item['action'] == 'classify_records']
    assert batches[0]['selected_aliases'] == [f'r{i}' for i in range(16)]
    for batch, messages in zip(batches, client.messages[1:]):
        wire = messages[-1]['content'].split('\nRECORDS:\n', 1)[1]
        assert batch['record_protocol']['wire_sha256'] == hashlib.sha256(wire.encode()).hexdigest()
        assert 'source_chars' not in wire and 'User:' not in wire and 'Date:' not in wire
        bindings = batch['record_protocol']['bindings']
        assert [bindings[alias]['record_id'] for alias in batch['selected_aliases']] == batch['selected_record_ids']
        assert [bindings[alias]['body_source_span'] for alias in batch['selected_aliases']] == batch['body_exposed_spans']
    for fields in result['record_facts'].values():
        ref = fields['label']['evidence'][0]['provenance']['projection']
        assert ref['protocol'] == 'compact_values_v1' and ref['source_sha256'] == result['source_sha256']
        assert ref['body_source_span'] != ref['source_span']
    assert result['model_exposed_spans'] == result['body_exposed_spans']
    assert result['model_source_coverage']['covered_chars'] < sum(int(s.split(':')[-1]) - int(s.split()[-1].split(':')[0])
                                                               for s in result['full_record_source_spans'])


def test_projection_failure_preserves_all_previously_accepted_facts(monkeypatch):
    original = candidate.project_records
    projection_calls = 0

    def project_with_one_failure(records, **kwargs):
        nonlocal projection_calls
        projection_calls += 1
        if projection_calls == 2:
            raise candidate.ProjectionError('body_not_verbatim_source', record_id=records[0].record_id)
        return original(records, **kwargs)

    monkeypatch.setattr(candidate, 'project_records', project_with_one_failure)
    client = CappedClient(compact_responder, context=4000)
    result = candidate.run_adaptive_v3({'context': logs(), 'question': 'Count positive records.'}, client, CONFIG)
    assert result['mode'] == 'adaptive_fact_aggregation_v3'
    assert not result['evidence_complete'] and result['aggregate']['unknown_count'] == 1
    assert result['aggregate']['known_count'] == len(result['record_facts']) == 119
    assert result['trace'][-1]['retained_facts'] == 119 and result['final_fact_loss_count'] == 0
    assert any(item['action'] == 'record_projection_rejected' for item in result['trace'])


def test_preferred_alias_pairs_bind_out_of_order_without_positional_shift():
    def paired(messages, number):
        reply = compact_responder(messages, number)
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            return reply
        rows = json.loads(messages[-1]['content'].split('\nRECORDS:\n', 1)[1])
        values = json.loads(reply)['values']
        return json.dumps({'records': [[row['id'], value] for row, value in reversed(list(zip(rows, values)))]})

    client = CappedClient(paired, context=4000)
    result = candidate.run_adaptive_v3({'context': logs(), 'question': 'Count positive records.'}, client, CONFIG)
    assert result['answer'] == 'Answer: 40' and result['evidence_complete']
    batches = [item for item in result['trace'] if item['action'] == 'classify_records']
    assert all(item['record_protocol']['format'] == 'compact_alias_pairs_v1' for item in batches)
    assert len(result['record_facts']) == 120 and result['final_fact_loss_count'] == 0


def test_missing_pair_keeps_its_record_unknown_without_discarding_other_facts():
    def incomplete(messages, number):
        reply = compact_responder(messages, number)
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            return reply
        rows = json.loads(messages[-1]['content'].split('\nRECORDS:\n', 1)[1])
        values = json.loads(reply)['values']
        # Keep one record missing in every retry without shifting its peers.
        return json.dumps({'records': [[row['id'], value] for row, value in zip(rows[1:], values[1:])]})

    client = CappedClient(incomplete, context=4000)
    result = candidate.run_adaptive_v3({'context': logs(), 'question': 'Count positive records.'}, client, CONFIG)
    assert not result['evidence_complete'] and 0 < result['aggregate']['known_count'] < 120
    assert result['aggregate']['unknown_count'] > 0 and result['final_fact_loss_count'] == 0
    assert any(event.get('reason') == 'missing_alias' for batch in result['trace'] if batch['action'] == 'classify_records'
               for event in batch['fact_events'])


def test_repeated_user_clauses_form_complete_positive_union_and_keep_exclusions():
    assert candidate._requested_user_ids('For User 7 or User 8 and user 9, count positives.') == ['7', '8', '9']
    assert candidate._requested_user_ids('For user IDs 7, 8 or user 9 and User 7, count positives.') == ['7', '8', '9']
    assert candidate._requested_user_ids('Exclude User 7 or User 8 and user 9 from this subset.') == []
    assert candidate._requested_user_ids('Excluding User 7 or User 8. For User 9, count positives.') == ['9']
    with pytest.raises(candidate.PlanValidationError) as error:
        candidate._normal_plan(plan(filters={'user': '7'}), 'For User 7 or User 8 and user 9, count positives.')
    assert error.value.code == 'user_scope_union_mismatch'


def test_repeated_user_groups_repair_incomplete_base_once_with_machine_union():
    question = 'Between User 7 or User 8, compare positive counts in the source interval from 2024-01-01 to 2024-01-10.'
    groups = [{'name': '7', 'filters': {'user': '7', 'date_from': '2024-01-01', 'date_to': '2024-01-10'}},
              {'name': '8', 'filters': {'user': '8', 'date_from': '2024-01-01', 'date_to': '2024-01-10'}}]
    bad = plan(operation='compare_counts', filters={'user': '7'}, groups=groups)
    good = plan(operation='compare_counts', filters={}, groups=groups)
    result, client, sent = run_plans([bad, good], question=question,
                                     context=logs(users=['7'] * 3 + ['8'] * 6 + ['99'] * 111))
    assert len(sent) == 2 and result['evidence_complete']
    assert result['planning_diagnostics']['validation_errors'][0]['code'] == 'user_scope_union_mismatch'
    assert '"requested_user_ids":["7","8"]' in sent[1][-1]['content']
    assert result['plan']['filters'] == {} and result['record_index']['scope_records'] == 9
    assert [g['known_count'] for g in result['aggregate']['groups']] == [3, 6]



def test_combined_candidates_repair_scope_and_preserve_count_checked_boundary_proof(monkeypatch):
    from long_context_agent import fact_ledger as ledger
    monkeypatch.setattr(candidate, 'parse_records', ledger.parse_records)
    monkeypatch.setattr(candidate, 'FactLedger', ledger.FactLedger)
    monkeypatch.setattr(candidate, 'normalize_date', ledger.normalize_date)
    prefix = 'The following lines contain 120 records, one per line.\n\nCheck every source record.\n\n'
    context = prefix + logs().split('\n', 1)[1] + prefix.replace('The following lines', 'Recall: the preceding lines', 1)
    result, client, sent = run_plans([plan(filters={'label': 'positive'}), plan()], context=context)
    assert result['answer'] == 'Answer: 40' and result['evidence_complete']
    assert result['record_index']['parser_complete']
    assert not result['record_index']['unparsed_ranges']
    assert result['record_index']['ignored_wrapper_ranges']
    proof = result['record_index']['boundary_evidence']
    assert any(item['kind'] == 'count_checked_footer' and item['parsed_records'] == 120 for item in proof)
    assert len(result['scope_record_refs']) == 120
