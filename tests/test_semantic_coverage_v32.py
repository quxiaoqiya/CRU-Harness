"""Exact source-body reuse: multiplicity, source identity, and unknown safety."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from long_context_agent.adaptive_evidence_v3 import AdaptiveV3Config, run_adaptive_v3
from long_context_agent.fact_ledger import parse_records
from long_context_agent.record_protocol import body_projection
from long_context_agent.semantic_coverage import ExactBodyIndex


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('_coverage_v32_fixtures', ROOT / 'tests/test_adaptive_evidence_v3.py')
fixtures = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = fixtures
spec.loader.exec_module(fixtures)


def source(bodies, users=None):
    users = users or ['7'] * len(bodies)
    return f'The following lines contain {len(bodies)} records, one per line.\n' + ''.join(
        f'Date: 2024-01-01 || User: {users[i]} || Instance: {body}\n'
        for i, body in enumerate(bodies))


def task(**changes):
    return fixtures.plan(**changes)


def execute(context, *, task_plan=None, unknown=False, calls=24, config=None):
    task_plan = task_plan or task()
    def respond(messages, number):
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            return json.dumps(task_plan)
        rows = json.loads(messages[-1]['content'].split('\nRECORDS:\n', 1)[1])
        labels = task_plan['labels']
        return json.dumps({'records': [[r['id'], None if unknown else labels.index(
            r['text'].split('semantic=', 1)[1].split(';', 1)[0])] for r in rows]})
    client = fixtures.CappedClient(respond, calls=calls, context=4000)
    config = config or AdaptiveV3Config(max_records_per_batch=8,
                                      planner_output_tokens=512,
                                      classification_output_tokens=1024)
    result = run_adaptive_v3({'question': 'Count positive records.', 'context': context}, client, config)
    return result, client


def test_exact_body_groups_ignore_user_metadata_but_keep_distinct_source_records():
    bodies = ['semantic=positive; shared payload'] * 200 + ['semantic=negative; other payload'] * 100
    context = source(bodies, ['7'] * 150 + ['8'] * 150)
    result, client = execute(context)
    assert result['answer'] == 'Answer: 200' and result['evidence_complete']
    assert result['aggregate']['known_count'] == 300
    assert len(result['record_facts']) == 300
    assert len(client.messages) == 2
    assert result['semantic_coverage']['unique_projectable_bodies'] == 2
    assert result['semantic_coverage']['coalesced_duplicates'] == 298
    assert result['semantic_coverage']['transferred_records'] == 298
    assert len(result['model_exposed_spans']) == 2
    assert len(result['inferred_body_spans']) == 298
    assert not set(result['model_exposed_spans']) & set(result['inferred_body_spans'])
    assert result['model_source_coverage']['fraction'] < result['coverage']['fraction']
    assert 'exact_body_label_transfer' in result['coverage_basis']
    selected = result['trace'][1]['selected_record_ids']
    by_id = parse_records(context).by_id
    for rid, facts in result['record_facts'].items():
        cell = facts['label']
        assert len(cell['evidence']) == 1
        ref = cell['evidence'][0]
        assert ref['source_sha256'] == hashlib.sha256(context.encode()).hexdigest()
        assert ref['source_span'] == by_id[rid].source_span
        provenance = ref['provenance']
        if rid in selected:
            assert provenance['stage'] == 'record_classification'
            continue
        assert provenance['stage'] == 'exact_body_fanout'
        representative = by_id[provenance['representative_record_id']]
        body, (start, end) = body_projection(by_id[rid])
        assert body == body_projection(representative)[0]
        assert provenance['body_source_span'] == f'source_chars {start}:{end}'
        assert provenance['body_sha256'] == hashlib.sha256(body.encode()).hexdigest()
    review = result['trace'][1]
    assert review['direct_information_gain'] == 2
    assert review['fanout_information_gain'] == 298
    assert review['information_gain'] == 300


def test_full_body_not_normalized_for_unicode_case_whitespace_or_substring():
    context = source(['semantic=positive; Alpha  Beta', 'semantic=positive; Alpha Beta',
                      'semantic=positive; alpha  Beta', 'semantic=positive; Alpha  Beta extra',
                      'semantic=positive; caf\u00e9', 'semantic=positive; cafe\u0301'] * 50)
    index = ExactBodyIndex(parse_records(context).records)
    assert len(index.groups) == 6
    assert sorted(len(group) for group in index.groups.values()) == [50] * 6
    assert index.summary()['normalization'] == 'none'


def test_unknown_representative_cannot_repeatedly_retry_other_copies():
    result, client = execute(source(['semantic=positive; identical'] * 300), unknown=True)
    reviews = [t for t in result['trace'] if t['action'] == 'classify_records']
    assert len(reviews) == 2 and len(client.messages) == 3
    assert all(len(r['selected_record_ids']) == 1 for r in reviews)
    assert result['stop_reason'] == 'no_fact_gain'
    assert not result['record_facts'] and not result['evidence_complete']
    assert result['aggregate']['known_count'] == 0
    assert result['aggregate']['unknown_count'] == 300
    assert all(not r['fanout_events'] and r['information_gain'] == 0 for r in reviews)
    assert len(result['model_exposed_spans']) == 1
    assert result['inferred_body_spans'] == []


def test_scope_filters_precede_body_coalescing_and_never_import_other_user_rows():
    bodies = ['semantic=positive; shared'] * 600
    result, _ = execute(source(bodies, ['7'] * 300 + ['8'] * 300),
                        task_plan=task(filters={'user': '7'}))
    assert result['answer'] == 'Answer: 300' and result['evidence_complete']
    assert result['semantic_coverage']['source_records'] == 300
    assert len(result['record_facts']) == 300
    assert {r['source_fields']['user'] for r in result['scope_record_refs']} == {'7'}


def test_explicit_source_labels_are_not_overwritten_from_an_identical_body():
    # The parser can contain source metadata labels. They remain independent
    # records: transfer runs only for unresolved cells.
    context = '\n'.join(json.dumps({'date': '2024-01-01', 'user': '7',
                                   'body': 'semantic=positive; shared',
                                   **({'label': 'negative'} if i == 0 else {})}) for i in range(300))
    result, _ = execute(context)
    assert result['answer'] == 'Answer: 299' and result['evidence_complete']
    assert len(result['record_facts']) == 299
    first = parse_records(context).records[0]
    assert first.record_id not in result['record_facts']


def test_unprojectable_decoded_json_body_does_not_join_verbatim_copies():
    context = '\n'.join([json.dumps({'body': 'same\nbody'})] * 3)
    records = parse_records(context).records
    index = ExactBodyIndex(records)
    assert not index.groups
    assert len(index.unprojectable_ids) == 3
    assert len(index.representatives(records, {r.record_id for r in records})) == 3


def test_small_scope_default_retains_legacy_direct_protocol_and_can_explicitly_enable():
    context = source(['semantic=positive; repeated ' + 'padding ' * 12] * 120)
    small, client = execute(context)
    assert not small['semantic_coverage']['enabled']
    assert len(client.messages) == 16
    compact, client = execute(context, config=AdaptiveV3Config(
        max_records_per_batch=8, coalescing_min_scope_records=1))
    assert compact['semantic_coverage']['enabled'] and compact['answer'] == 'Answer: 120'
    assert len(client.messages) == 2


def test_disable_coalescing_preserves_all_physical_classification_costs():
    result, client = execute(source(['semantic=positive; repeated'] * 300), calls=50,
                              config=AdaptiveV3Config(max_records_per_batch=8,
                                                      exact_body_coalescing=False))
    assert result['answer'] == 'Answer: 300' and result['evidence_complete']
    assert not result['semantic_coverage']['enabled']
    assert len(client.messages) == 39
    assert result['semantic_coverage']['transferred_records'] == 0
