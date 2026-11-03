"""Only synthetic inputs/labels; never open the fresh cohort."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from scripts import audit_context_holdout_execution_v3 as audit


BUDGET = {'context_tokens': 16384, 'total_tokens': 65536, 'max_calls': 32,
          'max_output_tokens': 2048, 'temperature': 0., 'seed': 20261101, 'token_safety_margin': 16}


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def entry():
    return {'physical_request': 1, 'prompt_sha256': 'f' * 64, 'native_tools': True,
            'tools_sha256': 'e' * 64, 'admission_prompt_tokens': 116,
            'requested_output_tokens': 200, 'charged_tokens': 110,
            'status': 'ok', 'usage_known': True,
            'usage': {'prompt_tokens': 100, 'completion_tokens': 10, 'total_tokens': 110}}


def fixture(tmp_path):
    root, out, data = tmp_path / 'root', tmp_path / 'out', tmp_path / 'data'
    source = root / 'a.py'
    source.parent.mkdir(); source.write_text('x = 1\n')
    snapshot = out / 'frozen_source/a.py'
    snapshot.parent.mkdir(parents=True); snapshot.write_bytes(source.read_bytes())
    data.mkdir(); (data / 'inputs.jsonl').write_text('Synthetic input bytes, not parsed by auditor\n')
    labels = {'synthetic': {'id': 'synthetic', 'answer': '[10]', 'answer_type': 'ANSWER_TYPE.NUMERIC',
                            'dataset': 'fake', 'context_window_id': 1}}
    write(data / 'labels.json', labels)
    scorer = data / 'source/eval_helpers_original.py'
    scorer.parent.mkdir()
    original = audit.ROOT / 'data/context_holdout_20261101/source/eval_helpers_original.py'
    if not original.is_file():
        original = audit.ROOT / 'third_party/oolong/eval_helpers_original.py'
    scorer.write_bytes(original.read_bytes())
    manifest = {'inputs_sha256': audit.digest_file(data / 'inputs.jsonl'),
                'labels_sha256': audit.digest_file(data / 'labels.json'),
                'scorer_sha256': audit.digest_file(scorer), 'scorer_commit': audit.OFFICIAL_COMMIT,
                'selected': [{'id': 'synthetic', 'context_sha256': 'a' * 64}]}
    write(data / 'manifest.json', manifest)
    probe = out / 'label_guard_probe.txt'; probe.write_text('probe only')
    profile = out / 'labels_guard.sb'
    profile.write_text('(version 1)\n(allow default)\n' + ''.join(
        '(deny file-read* (literal ' + json.dumps(str(p.resolve())) + '))\n'
        for p in (data / 'labels.json', probe)))
    frozen = {'sample_ids': ['synthetic'], 'methods': ['adaptive_v3', 'static'],
              'inputs': str(data / 'inputs.jsonl'), 'inputs_sha256': manifest['inputs_sha256'],
              'data_manifest': str(data / 'manifest.json'), 'data_manifest_sha256': audit.digest_file(data / 'manifest.json'),
              'labels_sha256': manifest['labels_sha256'], 'official_scorer_sha256': manifest['scorer_sha256'],
              'budget': BUDGET, 'code_sha256': {'a.py': audit.digest_file(source)},
              'runtime': audit.runtime_identity(), 'model': 'synthetic-model', 'labels_opened_by_runner': False,
              'label_guard_probe': str(probe), 'label_guard_profile_sha256': audit.digest_file(profile)}
    write(out / 'freeze.json', frozen)
    write(out / 'guard_verification.json', {'verified': True, 'probe_read_denied': True, 'labels_not_opened': True,
          'profile_sha256': audit.digest_file(profile), 'time': 1})
    for method in frozen['methods']:
        record = {'id': 'synthetic', 'method': method, 'freeze_sha256': audit.digest_file(out / 'freeze.json'),
                  'context_sha256': 'a' * 64, 'prediction': 'Answer: 10', 'ledger': [entry()],
                  'usage': {'physical_requests': 1, 'prompt_tokens': 100, 'completion_tokens': 10,
                            'charged_tokens': 110, 'all_usage_known': True, 'within_budget': True, 'budget': BUDGET}}
        write(out / f'predictions/synthetic__{method}.json', record)
        write(out / f'journals/synthetic__{method}.json', {k: record[k] for k in ('id', 'method', 'freeze_sha256', 'ledger')})
    return root, out, data


def test_known_usage_conservation_and_native_schema():
    result = audit.audit_ledger([entry()], BUDGET)
    assert result['violations'] == []
    assert result['charged_tokens'] == 110
    assert result['native_tool_requests'] == 1
    assert result['admission_minus_reported_prompt'] == {16: 1}


@pytest.mark.parametrize(('key', 'value', 'violation'), [
    ('physical_request', 2, 'nonsequential_identity'),
    ('requested_output_tokens', 3000, 'output_limit'),
    ('admission_prompt_tokens', 17000, 'context_limit'),
    ('charged_tokens', 109, 'reported_usage_charge_mismatch'),
    ('tools_sha256', None, 'native_schema_hash_missing'),
])
def test_ledger_detects_independent_failures(key, value, violation):
    row = entry(); row[key] = value
    assert any(violation in x for x in audit.audit_ledger([row], BUDGET)['violations'])


def test_unknown_usage_must_charge_whole_reservation():
    row = entry(); row.pop('usage'); row['usage_known'] = False
    assert any('unknown_usage_not_fully_reserved' in x for x in audit.audit_ledger([row], BUDGET)['violations'])
    row['charged_tokens'] = 316
    result = audit.audit_ledger([row], BUDGET)
    assert not result['violations'] and result['unknown_usage_requests'] == 1


def test_default_audit_never_opens_or_hashes_labels(tmp_path, monkeypatch):
    root, out, data = fixture(tmp_path)
    original = Path.read_bytes
    def forbidden_labels(path):
        if path == data / 'labels.json':
            raise AssertionError('Default audit touched labels')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', forbidden_labels)
    result = audit.audit_execution(out, root=root)
    assert result['passed'] and result['complete']
    assert not result['labels_opened_by_this_audit']
    assert result['physical_generation_requests'] == 2


def test_pending_journal_is_charged_and_scoring_gate_stays_closed(tmp_path, monkeypatch):
    root, out, data = fixture(tmp_path)
    (out / 'predictions/synthetic__static.json').unlink()
    original = Path.read_bytes
    def forbidden_labels(path):
        if path == data / 'labels.json':
            raise AssertionError('Incomplete execution touched labels')
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', forbidden_labels)
    result = audit.audit_execution(out, root=root, allow_incomplete=True)
    assert result['passed'] and not result['complete']
    assert result['physical_generation_requests'] == 2
    assert len(result['pending_journal_accounting']) == 1
    with pytest.raises(ValueError, match='Cannot open labels'):
        audit.audit_execution(out, root=root, allow_incomplete=True, verify_scores=True)


def test_modified_snapshot_fails_before_labels(tmp_path, monkeypatch):
    root, out, data = fixture(tmp_path)
    (out / 'frozen_source/a.py').write_text('x = 2\n')
    result = audit.audit_execution(out, root=root)
    assert not result['passed']
    assert result['source_and_snapshot_checks']['snapshot_hash_mismatches'] == ['a.py']
    with pytest.raises(ValueError, match='Cannot open labels'):
        audit.audit_execution(out, root=root, verify_scores=True)


def test_postcompletion_ast_score_independently_matches_published_audit(tmp_path):
    root, out, data = fixture(tmp_path)
    write(out / 'audit.json', {'samples': [{'id': 'synthetic', 'method': method, 'official_score': 1., 'full_credit': True}
           for method in ('adaptive_v3', 'static')],
          'summary': {method: {'mean_official_score': 1., 'full_credit': 1} for method in ('adaptive_v3', 'static')}})
    result = audit.audit_execution(out, root=root, verify_scores=True)
    assert result['passed'] and result['labels_opened_by_this_audit']
    assert result['official_scorer_verification']['mismatches'] == []
    assert (out / 'independent_scoring_access.json').exists()


def test_postcompletion_score_mismatch_is_detected(tmp_path):
    root, out, data = fixture(tmp_path)
    write(out / 'audit.json', {'samples': [{'id': 'synthetic', 'method': method, 'official_score': 0., 'full_credit': False}
           for method in ('adaptive_v3', 'static')],
          'summary': {method: {'mean_official_score': 0., 'full_credit': 0} for method in ('adaptive_v3', 'static')}})
    result = audit.audit_execution(out, root=root, verify_scores=True)
    assert not result['passed']
    assert len(result['official_scorer_verification']['mismatches']) == 4


def record_mechanism_fixture():
    from long_context_agent.adaptive_evidence_v3 import _aggregate, _compact_aggregate
    from long_context_agent.fact_ledger import FactLedger, parse_records
    sample = {'context': 'ID: 1 || User: 7 || Instance: alpha\nID: 2 || User: 7 || Instance: beta\n', 'question': 'Count alpha.'}
    collection = parse_records(sample['context']); ledger = FactLedger(collection)
    plan = {'operation': 'count', 'filters': {}, 'field': 'label', 'labels': ['alpha', 'beta'],
            'target': 'alpha', 'compare': [], 'groups': [], 'count_all_records': False}
    events = [ledger.add(record.record_id, value, source_span=record.source_span, source_sha256=collection.source_sha256)
              for record, value in zip(collection.records, ['alpha', 'beta'])]
    aggregate = _compact_aggregate(_aggregate(ledger, plan)); state = ledger.state(compact=True)
    result = {'mode': 'adaptive_fact_aggregation_v3', 'source_sha256': collection.source_sha256,
              'plan': plan, 'record_facts': ledger.facts, 'fact_state': state, 'aggregate': aggregate,
              'trace': [{'action': 'classify_records', 'fact_events': events, 'information_gain': 2,
                         'stagnant_rounds': 0, 'selected_record_ids': [r.record_id for r in collection.records]},
                        {'action': 'deterministic_finalize', 'fact_digest': state['fact_digest'],
                         'retained_facts': 2, 'aggregate': aggregate, 'model_final_call': False}],
              'final_fact_loss_count': 0}
    return {'result': result, 'context_sha256': collection.source_sha256}, sample


def test_mechanism_reconstructs_aggregate_digest_and_source_provenance():
    record, sample = record_mechanism_fixture()
    result = audit.audit_v3_mechanisms(record, sample)
    assert result['passed'], result['violations']
    assert result['provenance_references_checked'] == 2
    fact = next(iter(record['result']['record_facts'].values()))['label']
    fact['evidence'][0]['source_span'] = 'source_chars 0:1'
    assert 'record_fact_provenance_mismatch' in audit.audit_v3_mechanisms(record, sample)['violations']


def test_mechanism_does_not_trust_zero_final_loss_claim():
    record, sample = record_mechanism_fixture()
    record['result']['record_facts'].pop(next(iter(record['result']['record_facts'])))
    violations = audit.audit_v3_mechanisms(record, sample)['violations']
    assert 'record_event_cells_not_conserved' in violations
    assert 'reconstructed_deterministic_aggregate_mismatch' in violations
    assert record['result']['final_fact_loss_count'] == 0


def narrative_mechanism_fixture():
    import hashlib
    sample = {'context': 'A has value 1. B has value 2.', 'question': 'What is the value?'}
    span = f'source_chars 0:{len(sample["context"])}'
    fact = {'subject': 'A', 'predicate': 'value', 'status': 'conflict',
            'candidates': [{'value': 1, 'quote': 'A has value 1.', 'source_span': span},
                           {'value': 2, 'quote': 'B has value 2.', 'source_span': span}]}
    facts = {'a\x00value': fact}
    digest = hashlib.sha256(json.dumps(facts, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    sha = hashlib.sha256(sample['context'].encode()).hexdigest()
    result = {'mode': 'adaptive_fact_memory_v3', 'source_sha256': sha, 'retained_facts': [fact],
              'trace': [{'action': 'fact_review', 'fact_events': [{'key': 'a\x00value', 'status': 'accepted'}, {'key': 'a\x00value', 'status': 'conflict'}],
                         'new_spans': [span], 'information_gain': 2, 'stagnant_rounds': 0, 'retained_facts': 1},
                        {'action': 'final_answer', 'retained_fact_count': 1, 'fact_digest': digest}],
              'final_fact_loss_count': 0}
    return {'result': result, 'context_sha256': sha}, sample


def test_narrative_conflict_keys_candidates_and_digest_are_conserved():
    record, sample = narrative_mechanism_fixture()
    assert audit.audit_v3_mechanisms(record, sample)['passed']
    record['result']['retained_facts'][0]['candidates'].pop()
    violations = audit.audit_v3_mechanisms(record, sample)['violations']
    assert 'narrative_conflict_candidates_lost' in violations
    assert 'narrative_final_digest_mismatch' in violations


def test_mechanism_stops_after_two_zero_gain_reviews():
    record, sample = narrative_mechanism_fixture()
    record['result']['trace'][1:1] = [{'action': 'fact_review', 'fact_events': [], 'new_spans': [],
                                     'information_gain': 0, 'stagnant_rounds': n, 'retained_facts': 1}
                                    for n in (1, 2, 3)]
    violations = audit.audit_v3_mechanisms(record, sample)['violations']
    assert any('zero_gain_run_exceeds_two' in x for x in violations)
    assert any('continued_after_stagnation_limit' in x for x in violations)


def projected_protocol_fixture(reply=None, *, count=3, context=None):
    from long_context_agent.adaptive_evidence import parse_review, source_coverage
    from long_context_agent.adaptive_evidence_v3 import _aggregate, _compact_aggregate
    from long_context_agent.fact_ledger import FactLedger, parse_records
    from long_context_agent.record_protocol import project_records, decode_classifications
    sample = {'context': context or ''.join(f'ID: {i} || User: {i} || Instance: body {i} alpha\n' for i in range(count)),
              'question': 'Count alpha records.'}
    collection = parse_records(sample['context']); ledger = FactLedger(collection)
    plan = {'operation': 'count', 'filters': {}, 'field': 'label', 'labels': ['alpha', 'beta', '__other__'],
            'target': 'alpha', 'compare': [], 'groups': [], 'count_all_records': False}
    projection = project_records(collection.records, field='label', labels=plan['labels'])
    aliases = [row['id'] for row in projection.rows]
    reply = {'values': [0] * count} if reply is None else reply
    raw = json.dumps(reply)
    decoded = decode_classifications(parse_review(raw), projection, field='label', labels=plan['labels'])
    events = [{'status': 'unknown' if e['reason'] == 'unknown_value' else 'rejected', **e} for e in decoded['rejected']]
    for item in decoded['items']:
        events.append(ledger.add(item['record_id'], item['value'], source_span=item['source_span'],
                                 source_sha256=collection.source_sha256,
                                 provenance={'stage': 'record_classification', 'batch': 0, 'projection': item['provenance']}))
    aggregate = _compact_aggregate(_aggregate(ledger, plan)); state = ledger.state(compact=True)
    bindings = projection.bindings
    bodies = [bindings[alias]['body_source_span'] for alias in aliases]
    coverage = source_coverage([audit._span_bounds(span) for span in bodies], len(sample['context']))
    gain = sum(e.get('status') == 'accepted' for e in events)
    result = {'mode': 'adaptive_fact_aggregation_v3', 'source_sha256': collection.source_sha256,
              'plan': plan, 'record_facts': ledger.facts, 'fact_state': state, 'aggregate': aggregate,
              'model_exposed_spans': bodies, 'model_source_coverage': coverage,
              'trace': [{'action': 'classify_records', 'selected_record_ids': [r.record_id for r in collection.records],
                         'source_spans': [r.source_span for r in collection.records], 'selected_aliases': aliases,
                         'body_exposed_spans': bodies, 'fact_events': events, 'information_gain': gain,
                         'stagnant_rounds': 0 if gain else 1, 'budget_before': {'calls': BUDGET['max_calls']},
                         'record_protocol': {'format': decoded['format'], 'wire_sha256': projection.wire_sha256,
                                             'bindings_sha256': projection.bindings_sha256, 'bindings': bindings,
                                             'batch_rejected': decoded['batch_rejected']}},
                        {'action': 'deterministic_finalize', 'fact_digest': state['fact_digest'],
                         'retained_facts': len(ledger.facts), 'aggregate': aggregate, 'model_final_call': False}]}
    return {'result': result, 'context_sha256': collection.source_sha256,
            'ledger': [{'physical_request': 1, 'content': raw}]}, sample, collection


def test_compact_protocol_proves_body_identity_and_vector_source_mapping():
    record, sample, collection = projected_protocol_fixture({'values': [0, 1, 0]})
    checks = audit.audit_v3_mechanisms(record, sample, BUDGET)
    assert checks['passed'], checks['violations']
    protocol = checks['record_protocol_audit']
    assert protocol['raw_responses_replayed'] == 1 and protocol['projection_bindings_checked'] == 3
    assert protocol['projected_body_ranges'] == 3


def test_alias_order_uses_wire_rows_not_lexically_sorted_bindings():
    record, sample, collection = projected_protocol_fixture(count=12)
    assert audit.audit_record_protocol(record, sample['context'], collection, BUDGET)['passed']
    review = record['result']['trace'][0]
    review['selected_aliases'] = list(review['record_protocol']['bindings'])
    assert review['selected_aliases'][2] == 'r10'
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert any('selected_alias_order_mismatch' in x for x in checks['violations'])


def test_body_projection_rejects_same_text_in_header_as_body_source():
    record, sample, collection = projected_protocol_fixture(count=1, context='ID: 1 || User: alpha || Instance: alpha\n')
    binding = record['result']['trace'][0]['record_protocol']['bindings']['r0']
    decoy = sample['context'].index('alpha')
    binding.update(body_source_span=f'source_chars {decoy}:{decoy + 5}', body_start=decoy, body_end=decoy + 5)
    assert 'projection_body_source_range_mismatch' in audit.audit_body_projection(binding, collection.records[0], sample['context'])


def test_projection_body_and_wire_hashes_are_checked_without_exposing_headers():
    record, sample, collection = projected_protocol_fixture()
    binding = record['result']['trace'][0]['record_protocol']['bindings']['r0']
    binding['body_sha256'] = '0' * 64
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert any('projection_body_hash_mismatch' in x for x in checks['violations'])
    record, sample, collection = projected_protocol_fixture()
    record['result']['trace'][0]['record_protocol']['wire_sha256'] = '0' * 64
    assert any('wire_body_projection_hash_mismatch' in x for x in audit.audit_record_protocol(record, sample['context'], collection, BUDGET)['violations'])


def test_model_coverage_cannot_claim_full_record_headers_were_exposed():
    record, sample, collection = projected_protocol_fixture()
    record['result']['model_exposed_spans'] = [r.source_span for r in collection.records]
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert 'model_exposure_claims_headers_or_omits_body' in checks['violations']


@pytest.mark.parametrize('reply', [{'values': [0, 1]}, {'values': [0, 1, 0, 1]}, [0, 1, 0]])
def test_vector_length_or_nonobject_failure_is_whole_batch_and_never_shifts(reply):
    record, sample, collection = projected_protocol_fixture(reply)
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert checks['passed'], checks['violations']
    assert checks['strict_batch_rejections_verified'] == 1
    assert not record['result']['record_facts']
    record['result']['trace'][0]['fact_events'].append({'status': 'accepted', 'field': 'label', 'record_id': collection.records[0].record_id})
    assert any('strict_vector_alias_mapping_or_no_shift_mismatch' in x for x in audit.audit_record_protocol(record, sample['context'], collection, BUDGET)['violations'])


@pytest.mark.parametrize('invalid', [99, -1, True, 'alpha'])
def test_invalid_index_keeps_later_valid_position_bound_to_its_original_record(invalid):
    record, sample, collection = projected_protocol_fixture({'values': [0, invalid, 1]})
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert checks['passed'], checks['violations']
    assert set(record['result']['record_facts']) == {collection.records[0].record_id, collection.records[2].record_id}
    last_event = record['result']['trace'][0]['fact_events'][-1]
    last_event['record_id'] = collection.records[1].record_id
    assert any('strict_vector_alias_mapping_or_no_shift_mismatch' in x for x in audit.audit_record_protocol(record, sample['context'], collection, BUDGET)['violations'])


def test_null_position_stays_unknown_without_double_counting_decoder_and_ledger_events():
    record, sample, collection = projected_protocol_fixture({'values': [0, None, 1]})
    checks = audit.audit_v3_mechanisms(record, sample, BUDGET)
    assert checks['passed'], checks['violations']
    assert len(record['result']['record_facts']) == 2


def test_legacy_records_and_duplicate_alias_rejection_remain_compatible():
    record, sample, collection = projected_protocol_fixture({'records': [
        {'id': 'r0', 'value': 'alpha'}, {'id': 'r0', 'value': 'beta'}, {'id': 'r2', 'value': 'beta'}]})
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert checks['passed'], checks['violations']
    assert set(record['result']['record_facts']) == {collection.records[2].record_id}


def test_evidence_projection_is_checked_even_when_full_record_ref_is_correct():
    record, sample, collection = projected_protocol_fixture()
    cell = next(iter(record['result']['record_facts'].values()))['label']
    ref = cell['evidence'][0]
    assert ref['source_span'] == collection.records[0].source_span
    ref['provenance']['projection']['body_sha256'] = '0' * 64
    checks = audit.audit_v3_mechanisms(record, sample, BUDGET)
    assert 'projection_body_hash_mismatch' in checks['violations']
    assert 'evidence_projection_not_bound_to_selected_batch' in checks['violations']


@pytest.mark.parametrize(('kind', 'batch_size', 'context_limit'), [('values', 8, 2600), ('pairs', 64, 5000)])
def test_auditor_accepts_actual_controller_synthetic_compact_trace(kind, batch_size, context_limit):
    import runpy
    from dataclasses import replace
    from long_context_agent.adaptive_evidence_v3 import run_adaptive_v3
    fixtures = runpy.run_path(str(audit.ROOT / 'tests/test_adaptive_evidence_v3.py'))
    responses = []
    def respond(messages, number):
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            reply = json.dumps(fixtures['plan']())
        else:
            bodies = json.loads(messages[-1]['content'].split('\nRECORDS:\n', 1)[1])
            values = [0 if 'semantic=positive;' in row['text'] else 1 for row in bodies]
            reply = json.dumps({'values': values} if kind == 'values' else
                               {'records': [[row['id'], value] for row, value in zip(bodies, values)]})
        responses.append(reply)
        return reply
    client = fixtures['CappedClient'](respond, context=context_limit)
    sample = {'context': fixtures['logs'](), 'question': 'Count positive records.'}
    result = run_adaptive_v3(sample, client, replace(fixtures['CONFIG'], max_records_per_batch=batch_size))
    record = {'result': result, 'context_sha256': result['source_sha256'],
              'ledger': [{'physical_request': i + 1, 'content': raw} for i, raw in enumerate(responses)]}
    checks = audit.audit_v3_mechanisms(record, sample, {**BUDGET, 'max_calls': 24})
    assert checks['passed'], checks['violations']
    assert checks['record_protocol_audit']['projection_bindings_checked'] == 120


def test_new_protocol_cannot_drop_projection_ref_or_mutate_unread_coverage():
    record, sample, collection = projected_protocol_fixture()
    cell = next(iter(record['result']['record_facts'].values()))['label']
    cell['evidence'][0]['provenance'].pop('projection')
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert 'protocol_fact_missing_projection_provenance' in checks['violations']
    record, sample, collection = projected_protocol_fixture()
    record['result']['model_source_coverage']['unread_ranges'] = []
    assert 'model_body_coverage_mismatch' in audit.audit_record_protocol(record, sample['context'], collection, BUDGET)['violations']


def test_alias_pairs_allow_reordered_response_without_source_shift():
    record, sample, collection = projected_protocol_fixture({'records': [['r2', 1], ['r0', 0], ['r1', None]]})
    checks = audit.audit_v3_mechanisms(record, sample, BUDGET)
    assert checks['passed'], checks['violations']
    assert record['result']['record_facts'][collection.records[0].record_id]['label']['value'] == 'alpha'
    assert record['result']['record_facts'][collection.records[2].record_id]['label']['value'] == 'beta'
    assert collection.records[1].record_id not in record['result']['record_facts']


@pytest.mark.parametrize('reply', [
    {'records': [['r2', 1]]},
    {'records': [['r0', 0], ['r0', 1], ['r2', 1]]},
    {'records': [['r0', 0], ['r1', 999], ['r2', 1]]},
    {'records': [['r0', 0], [1, 1], ['r2', 1]]},
    {'records': [['r0', 0], ['r1'], ['r2', 1]]},
    {'records': [['r0', 0], 'malformed', ['r2', 1]]},
    {'records': []},
])
def test_alias_pair_omission_duplicate_bad_value_or_alias_never_shifts_later_source(reply):
    record, sample, collection = projected_protocol_fixture(reply)
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert checks['passed'], checks['violations']
    assert checks['strict_batch_rejections_verified'] == 0
    if reply['records']:
        assert record['result']['record_facts'][collection.records[2].record_id]['label']['value'] == 'beta'
    else:
        assert not record['result']['record_facts']


def test_mixed_alias_pairs_and_legacy_dicts_reject_entire_batch():
    record, sample, collection = projected_protocol_fixture({'records': [['r0', 0], {'id': 'r1', 'value': 'beta'}]})
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert checks['passed'], checks['violations']
    assert checks['strict_batch_rejections_verified'] == 1
    assert not record['result']['record_facts']


def fanout_protocol_fixture():
    import hashlib
    from long_context_agent.adaptive_evidence import parse_review
    from long_context_agent.adaptive_evidence_v3 import _aggregate, _compact_aggregate
    from long_context_agent.fact_ledger import FactLedger, parse_records
    from long_context_agent.record_protocol import project_records, decode_classifications
    from long_context_agent.semantic_coverage import ExactBodyIndex
    context = 'ID: 1 || User: 7 || Instance: same body\nID: 2 || User: 8 || Instance: same body\n'
    # Start with one representative exposed; second source row is initially not
    # a fact. The independent auditor must prove the whole-body transfer.
    record, sample, _ = projected_protocol_fixture(count=1, context=context.splitlines(keepends=True)[0])
    sample['context'] = context
    collection = parse_records(context); source, target = collection.records
    projection = project_records([source], field='label', labels=record['result']['plan']['labels'])
    decoded = decode_classifications(parse_review(record['ledger'][0]['content']), projection,
                                     field='label', labels=record['result']['plan']['labels'])
    ledger = FactLedger(collection); item = decoded['items'][0]
    direct = ledger.add(source.record_id, item['value'], source_span=source.source_span,
                        source_sha256=collection.source_sha256,
                        provenance={'stage': 'record_classification', 'batch': 0, 'projection': item['provenance']})
    index = ExactBodyIndex(collection.records)
    derived = ledger.add(target.record_id, item['value'], source_span=target.source_span,
                         source_sha256=collection.source_sha256,
                         provenance=index.fanout_provenance(source.record_id, target.record_id, item['provenance']))
    result = record['result']; review = result['trace'][0]
    review.update(selected_record_ids=[source.record_id], source_spans=[source.source_span], fact_events=[direct], fanout_events=[{**derived, 'representative_record_id': source.record_id,
                   'body_sha256': hashlib.sha256(b'same body').hexdigest()}], information_gain=2,
                  record_protocol={'format': decoded['format'], 'wire_sha256': projection.wire_sha256,
                                   'bindings_sha256': projection.bindings_sha256, 'bindings': projection.bindings,
                                   'batch_rejected': False})
    state = ledger.state(compact=True); aggregate = _compact_aggregate(_aggregate(ledger, result['plan']))
    inferred = [index.binding_by_id[target.record_id].body_source_span]
    result.update(source_sha256=collection.source_sha256, record_facts=ledger.facts, fact_state=state,
                  aggregate=aggregate, inferred_body_spans=inferred,
                  semantic_coverage={**index.summary(), 'transferred_records': 1, 'direct_model_classified_records': 1, 'derived_facts_are_model_exposure': False},
                  coverage_basis='complete_model_body_exposure_or_exact_body_label_transfer_or_consumed_deterministic_source_fields')
    from long_context_agent.adaptive_evidence import source_coverage
    result['model_source_coverage'] = source_coverage([audit._span_bounds(x) for x in result['model_exposed_spans']], len(context))
    result['trace'][-1].update(fact_digest=state['fact_digest'], retained_facts=2, aggregate=aggregate,
                              exact_body_transferred_records=1)
    record['context_sha256'] = collection.source_sha256
    return record, sample, collection


def test_exact_body_fanout_independently_proves_source_transfer_and_retention():
    record, sample, collection = fanout_protocol_fixture()
    checks = audit.audit_v3_mechanisms(record, sample, BUDGET)
    assert checks['passed'], checks['violations']
    assert checks['provenance_references_checked'] == 2
    assert checks['record_protocol_audit']['fanout_records_checked'] == 1
    assert checks['record_protocol_audit']['projected_body_ranges'] == 1


def test_fanout_cannot_copy_a_hash_or_label_between_different_full_bodies():
    record, sample, collection = fanout_protocol_fixture()
    # Hold identities fixed to exercise the auditor's full-string predicate,
    # rather than failing earlier at the immutable source-identity gate.
    collection.records[1].fields['body'] = 'SAME body'
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert any('fanout_body_not_exactly_equal' in e for e in checks['violations'])


def test_fanout_representative_must_be_valid_same_round_raw_response():
    record, sample, collection = fanout_protocol_fixture()
    record['ledger'][0]['content'] = '{"values":[null]}'
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert any('fanout_event_not_bound_to_valid_representative' in e for e in checks['violations'])


def test_fanout_ref_must_keep_target_span_and_exact_representative_binding():
    record, sample, collection = fanout_protocol_fixture()
    target = collection.records[1]
    ref = record['result']['record_facts'][target.record_id]['label']['evidence'][0]['provenance']
    ref['representative_projection']['body_sha256'] = '0' * 64
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert 'fanout_provenance_not_bound_to_raw_response' in checks['violations']


def test_fanout_inferred_spans_never_count_as_actual_model_exposure():
    record, sample, collection = fanout_protocol_fixture()
    record['result']['model_exposed_spans'] += record['result']['inferred_body_spans']
    checks = audit.audit_record_protocol(record, sample['context'], collection, BUDGET)
    assert 'model_exposure_claims_headers_or_omits_body' in checks['violations']


def test_fanout_accepted_fact_cannot_disappear_from_final_ledger():
    record, sample, collection = fanout_protocol_fixture()
    record['result']['record_facts'].pop(collection.records[1].record_id)
    checks = audit.audit_v3_mechanisms(record, sample, BUDGET)
    assert 'record_event_cells_not_conserved' in checks['violations']
    assert 'raw_response_accepted_cells_not_retained' in checks['violations']


def test_legacy_v31_mechanism_reconstruction_uses_frozen_original_namespace(monkeypatch):
    record, sample = record_mechanism_fixture()
    record['method'] = 'adaptive_v31'
    import long_context_agent.adaptive_evidence_v3 as current
    monkeypatch.setattr(current, '_aggregate', lambda *args: (_ for _ in ()).throw(AssertionError('mutable implementation used')))
    assert audit.audit_v3_mechanisms(record, sample)['passed']


@pytest.mark.parametrize('unknown', [False, True])
def test_auditor_replays_actual_coalescing_controller_and_group_retry_cap(unknown):
    import runpy
    fixtures = runpy.run_path(str(audit.ROOT / 'tests/test_semantic_coverage_v32.py'))
    sample = {'context': fixtures['source'](['semantic=positive; shared'] * 200 +
                                            ['semantic=negative; other'] * 100),
              'question': 'Count positive records.'}
    result, client = fixtures['execute'](sample['context'], unknown=unknown)
    raw = []
    task = fixtures['task']()
    for messages in client.messages:
        if messages[0]['content'].startswith('Plan evidence acquisition'):
            raw.append(json.dumps(task))
        else:
            rows = json.loads(messages[-1]['content'].split('\nRECORDS:\n', 1)[1])
            raw.append(json.dumps({'records': [[r['id'], None if unknown else task['labels'].index(
                r['text'].split('semantic=', 1)[1].split(';', 1)[0])] for r in rows]}))
    record = {'result': result, 'context_sha256': result['source_sha256'],
              'ledger': [{'physical_request': i + 1, 'content': content} for i, content in enumerate(raw)]}
    checks = audit.audit_v3_mechanisms(record, sample, {**BUDGET, 'max_calls': 24})
    assert checks['passed'], checks['violations']
    assert checks['record_protocol_audit']['fanout_records_checked'] == (0 if unknown else 298)


@pytest.mark.parametrize(('values', 'reduction'), [
    (['alpha', 'alpha', 'beta'], 'argmax'),
    (['alpha', 'alpha', None], 'argmax'),
    (['alpha', 'beta', None], 'argmax'),
    (['alpha', 'beta', 'alpha'], 'counts'),
    (['beta', 'beta', None], 'argmin'),
])
def test_conditional_group_reduction_has_independent_count_and_bound_audit(values, reduction):
    from long_context_agent.fact_ledger import FactLedger, parse_records
    collection = parse_records('ID: 1 || User: 7 || Instance: a\nID: 2 || User: 7 || Instance: b\nID: 3 || User: 8 || Instance: c\n')
    ledger = FactLedger(collection)
    for source_record, value in zip(collection.records, values):
        if value is not None:
            ledger.add(source_record.record_id, value)
    plan = {'operation': 'conditional_group', 'field': 'label', 'filters': {}, 'labels': ['alpha', 'beta'],
            'semantic_filter': {'field': 'label', 'operator': 'eq', 'value': 'alpha'}, 'group_by': 'user', 'reduction': reduction}
    actual = ledger.conditional_group(filters={}, field='label', labels=plan['labels'], predicate=plan['semantic_filter'],
                                      group_by='user', reduction=reduction)
    result = {'plan': plan, 'aggregate': actual, 'record_facts': ledger.facts}
    checks = audit.audit_conditional_group(result, collection)
    assert checks['passed'], checks['violations']
    assert checks['production_reducer_called'] is False
    actual['counts']['7'] += 1
    assert 'independent_conditional_group_counts_mismatch' in audit.audit_conditional_group(result, collection)['violations']
