"""Joint semantic selection/grouping regressions from synthetic source rows."""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from long_context_agent import adaptive_evidence_v3 as controller
from long_context_agent.fact_ledger import FactLedger, parse_records


def source(rows):
    return '\n'.join(json.dumps({'user': user, 'label': label, 'text': 'source evidence ' * 20})
                     for user, label in rows)


def task(**changes):
    value = {'kind': 'records', 'operation': 'conditional_group', 'field': 'label',
             'filters': {}, 'labels': ['correct', 'incorrect'], 'taxonomy_complete': True,
             'semantic_filter': {'field': 'label', 'operator': 'eq', 'value': 'incorrect'},
             'group_by': 'user', 'reduction': 'argmax', 'target': None,
             'answer_template': 'User: {value}'}
    return {**value, **changes}


def grouped(rows, **changes):
    collection = parse_records(source(rows))
    plan = controller._normal_plan(task(**changes), 'Which user has the most incorrect instances?', collection)
    return controller._aggregate(FactLedger(collection), plan)


def test_semantic_selection_then_metadata_grouping_uses_matching_counts_not_user_mode():
    result = grouped([('7', 'incorrect')] * 3 + [('8', 'correct')] * 8 + [('8', 'incorrect')])
    assert result['complete'] and result['value'] == '7'
    assert result['counts'] == {'7': 3, '8': 1}
    assert result['group_populations'] == {'7': 3, '8': 9}
    assert result['known_count'] == 12 and result['unknown_count'] == 0


def test_argmin_keeps_real_metadata_groups_with_zero_matching_records():
    result = grouped([('7', 'incorrect'), ('8', 'correct')], reduction='argmin')
    assert result['value'] == '8' and result['counts']['8'] == 0
    assert result['complete'] and result['coverage_complete']


def test_argmax_tie_is_an_explicit_set_instead_of_first_user_guess():
    result = grouped([('7', 'incorrect'), ('8', 'incorrect')])
    assert result['value'] == ['7', '8'] and result['ties'] == ['7', '8']
    assert result['complete']


@pytest.mark.parametrize('operator,value,counts', [
    ('in', ['correct', 'incorrect'], {'7': 2, '8': 1}),
    ('not_in', ['incorrect'], {'7': 1, '8': 0}),
    ('ne', 'correct', {'7': 1, '8': 1}),
])
def test_conditional_predicate_operators(operator, value, counts):
    result = grouped([('7', 'correct'), ('7', 'incorrect'), ('8', 'incorrect')],
                     reduction='counts', semantic_filter={'field': 'label', 'operator': operator, 'value': value})
    assert result['value'] == counts and result['complete']


def test_partial_unknown_label_mass_is_assigned_to_its_actual_metadata_group():
    collection = parse_records(source([('7', None)] * 3 + [('8', None)] * 2))
    ledger = FactLedger(collection)
    for row in collection.records[:3]:
        ledger.add(row.record_id, 'incorrect')
    result = controller._aggregate(ledger, controller._normal_plan(task(), 'Which user has the most incorrect instances?', collection))
    assert result['value'] == '7' and result['complete']
    assert not result['coverage_complete'] and result['unknown_count'] == 2
    assert result['bounds'] == {'7': {'lower': 3, 'upper': 3}, '8': {'lower': 0, 'upper': 2}}


def test_ten_thousand_groups_can_certify_without_quadratic_competitor_materialization():
    text = '\n'.join(json.dumps({'user': str(i), 'label': None, 'text': 'unseen'})
                     for i in range(10000))
    text += '\n' + '\n'.join(json.dumps({'user': 'winner', 'label': 'incorrect', 'text': 'known'})
                             for _ in range(3))
    collection = parse_records(text)
    plan = controller._normal_plan(task(), 'Which user has the most incorrect instances?', collection)
    result = controller._aggregate(FactLedger(collection), plan)
    assert result['value'] == 'winner' and result['complete']
    assert not result['coverage_complete'] and result['unknown_count'] == 10000
    assert len(result['bounds']) == 10001 and result['bounds']['0']['upper'] == 1


@pytest.mark.parametrize('reduction,labels,expected', [
    ('argmax', [('7', 'incorrect'), ('7', 'incorrect'), ('8', 'incorrect'), ('8', None), ('9', 'correct')], None),
    ('argmin', [('7', 'correct'), ('8', 'incorrect'), ('8', None), ('9', 'incorrect')], '7'),
    ('argmin', [('7', 'correct'), ('7', None), ('8', 'incorrect')], None),
])
def test_strongest_competing_bound_preserves_strict_inequality_and_tie_blockers(reduction, labels, expected):
    result = grouped(labels, reduction=reduction)
    assert result['value'] == expected
    assert result['complete'] is (expected is not None)


def test_partial_empirical_lead_cannot_certify_argmax_if_unseen_labels_can_flip_it():
    collection = parse_records(source([('7', None)] * 3 + [('8', None)] * 5))
    ledger = FactLedger(collection)
    ledger.add(collection.records[0].record_id, 'incorrect')
    ledger.add(collection.records[3].record_id, 'correct')
    result = controller._aggregate(ledger, controller._normal_plan(task(), 'Which user has the most incorrect instances?', collection))
    assert result['value'] is None and not result['complete']
    assert result['estimate'] == '7' and result['bounds']['8']['upper'] == 4


def test_missing_group_identity_and_conflicting_labels_are_not_dropped():
    collection = parse_records(source([('7', None), ('8', 'incorrect'), (None, 'incorrect')]))
    ledger = FactLedger(collection)
    ledger.add(collection.records[0].record_id, 'incorrect')
    ledger.add(collection.records[0].record_id, 'correct')
    result = controller._aggregate(ledger, controller._normal_plan(task(), 'Which user has the most incorrect instances?', collection))
    assert result['missing_group_count'] == 1 and result['conflict_count'] == 1
    assert result['unknown_count'] == 1 and result['total_count'] == 3
    assert not result['complete'] and result['value'] is None


@pytest.mark.parametrize('changes,code', [
    ({'semantic_filter': {'label': 'incorrect'}}, 'semantic_predicate_invalid'),
    ({'semantic_filter': {'field': 'user', 'operator': 'eq', 'value': '7'}}, 'semantic_predicate_field_invalid'),
    ({'semantic_filter': {'field': 'label', 'operator': 'contains', 'value': 'incorrect'}}, 'semantic_predicate_operator_invalid'),
    ({'semantic_filter': {'field': 'label', 'operator': 'eq', 'value': ['incorrect']}}, 'semantic_predicate_value_invalid'),
    ({'semantic_filter': {'field': 'label', 'operator': 'eq', 'value': 'invented'}}, 'semantic_predicate_outside_taxonomy'),
    ({'group_by': 'label'}, 'conditional_group_dimension_invalid'),
    ({'group_by': 'missing'}, 'conditional_group_source_dimension_missing'),
    ({'field': 'user'}, 'conditional_group_field_invalid'),
    ({'reduction': 'guess'}, 'conditional_group_reduction_invalid'),
    ({'target': 'incorrect'}, 'conditional_group_arguments'),
])
def test_joint_operator_contract_rejects_dimension_and_predicate_errors(changes, code):
    collection = parse_records(source([('7', 'incorrect')]))
    with pytest.raises(controller.PlanValidationError) as caught:
        controller._normal_plan(task(**changes), 'Which user has most incorrect?', collection)
    assert caught.value.code == code


def test_semantic_filter_cannot_be_silently_ignored_by_legacy_operator():
    with pytest.raises(controller.PlanValidationError) as caught:
        controller._normal_plan(task(operation='count', target='incorrect'), 'Count incorrect.')
    assert caught.value.code == 'conditional_group_operator_required'


def test_incomplete_taxonomy_uses_explicit_complement_without_conflating_group_ids():
    plan = controller._normal_plan(task(labels=[], taxonomy_complete=False), 'Which user has most incorrect?')
    assert plan['labels'] == ['incorrect', '__other__'] and plan['field'] == 'label'
    assert plan['target'] is None and plan['group_by'] == 'user'


def test_base_user_union_and_per_user_subgroup_intersect_deterministically():
    assert controller._merge_filters({'user_ids': ['7', '8']}, {'user_ids': ['7']}) == {'user_ids': ['7']}
    assert controller._merge_filters({'user': '7'}, {'users': ['7', '8']}) == {'user_ids': ['7']}
    with pytest.raises(ValueError):
        controller._merge_filters({'user_ids': ['7']}, {'user': '8'})


def test_joint_operator_uses_real_budgeted_classification_then_local_final():
    path = Path(__file__).with_name('test_adaptive_evidence_v3.py')
    spec = importlib.util.spec_from_file_location('_v32_test_client', path)
    fixture = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = fixture
    spec.loader.exec_module(fixture)
    rows = ['incorrect'] * 12 + ['correct'] * 28 + ['incorrect'] * 3 + ['correct'] * 77
    users = ['7'] * 40 + ['8'] * 80
    client = fixture.CappedClient(fixture.responder_for(task()), context=3000)
    result = controller.run_adaptive_v3(
        {'question': 'Which user has most incorrect instances?', 'context': fixture.logs(rows, users=users)},
        client, replace(fixture.CONFIG, max_records_per_batch=16))
    assert result['answer'] == 'User: 7' and result['evidence_complete']
    assert result['aggregate']['counts']['7'] > result['aggregate']['bounds']['8']['upper']
    assert result['trace'][-1]['model_final_call'] is False
    assert result['final_fact_loss_count'] == 0 and result['aggregate']['group_by'] == 'user'
