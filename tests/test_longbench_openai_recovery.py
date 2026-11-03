"""Offline checks for zero-generation-only recovery and transparent diagnostics."""
from dataclasses import asdict
from decimal import Decimal
import errno
import io
import json
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from long_context_agent.experiment_budget import ExperimentBudget
from long_context_agent.openai_responses_transport import OpenAIResponsesBudgetedClient, OpenAIResponsesTransportError
from scripts import benchmark_longbench_openai_recovery as recovery


@pytest.fixture
def trial(tmp_path, monkeypatch):
    base = recovery.base
    source, private = tmp_path/'source', tmp_path/'private_labels'
    source.mkdir()
    private.mkdir()
    rows = [{'_id': f'id-{i}', 'id': f'id-{i}', 'index': i,
             'domain': 'Multi-Document QA', 'sub_domain': 'mock', 'length': 'long', 'difficulty': 'easy',
             'context': ('LOW' if i < 6 else 'HIGH')+f' mock source {i}.',
             'question': 'Choose the mock option.', 'choice_A': 'one', 'choice_B': 'two',
             'choice_C': 'three', 'choice_D': 'four'} for i in range(12)]
    inputs, labels, manifest, cache = source/'inputs.jsonl', private/'labels.json', source/'manifest.json', source/'raw.jsonl'
    inputs.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    labels.write_text(json.dumps({str(i): 'A' for i in range(12)}))
    cache.write_text('Mock denied source-bearing cache.')
    manifest.write_text(json.dumps({'inputs_sha256': base.digest(inputs), 'labels_sha256': base.digest(labels),
        'cache_sha256': base.digest(cache), 'probe_index': 6, 'selected': [
            {'index': row['index'], 'id': row['id'], 'context_sha256': base.text_digest(row['context']),
             'prompt_sha256': base.text_digest(base.official_prompt(base.make_row(row))),
             'context_chars': len(row['context'])} for row in rows]}))
    parent = {'models': list(base.MODELS), 'inputs': str(inputs), 'labels': str(labels),
        'manifest': str(manifest), 'raw_cache': str(cache), 'output': str(tmp_path/'original'),
        'budget_harness': base.HARNESS_LIMITS, 'budget_direct': base.DIRECT_LIMITS,
        'reasoning': 'none', 'timeout': 900, 'cost_cap_usd': 50, 'prices': base.EXPECTED_PRICES,
        'probes_per_model': 1, 'long_direct_spacing_seconds': 0}
    parent_path = tmp_path/'parent_config.json'
    parent_path.write_text(json.dumps(parent))
    base.freeze_limits(parent)
    parent_sha = base.digest(Path(parent['output'])/'freeze.json')
    monkeypatch.setattr(recovery, 'PARENT_FREEZE', parent_sha)
    jobs = base.expected_jobs(rows, 6)
    selected = {(model, value['index'], arm) for model, value, arm in jobs[:36]+jobs[-3:]}
    assert len(selected) == 39
    out = Path(parent['output'])
    costs = base.MonetaryLedger(out/'cost_ledger.json', cap=50, prices=parent['prices'], freeze_sha=parent_sha)
    originals = {}
    nongeneric = [job for job in jobs if (job[0], job[1]['index'], job[2]) not in selected]
    generated_keys = [(model, value['index'], arm) for model, value, arm in nongeneric[:2]]
    for model, row, arm in jobs:
        key = model, row['index'], arm
        budget = dict(base.HARNESS_LIMITS if arm == 'harness' else base.DIRECT_LIMITS)
        ledger, error, prediction, status = [], {'type': 'FixtureError', 'message': 'A preserved non-network failure'}, None, 'error'
        if key in selected:
            error = {'type': 'OpenAIResponsesTransportError', 'message': recovery.NETWORK_MESSAGE}
        elif key in generated_keys:
            known = key == generated_keys[0]
            metadata = {'actual_model': model, 'returned_service_tier': 'default'} if known else {}
            ledger = [{'physical_request': 1, 'prompt_sha256': '0'*64, 'tools_sha256': None,
                'native_tools': False, 'admission_prompt_tokens': 96, 'requested_output_tokens': 16,
                'charged_tokens': 92 if known else 112, 'status': 'ok' if known else 'error',
                'usage_known': known, 'usage': {'prompt_tokens': 80, 'completion_tokens': 12, 'total_tokens': 92} if known else {},
                'provider_metadata': metadata}]
            money = costs.reserve(model=model, sample_id=f'{arm}:{row["index"]}', prompt=96, completion=16)
            costs.settle(money, ledger[0])
            if known:
                prediction, status, error = 'A', 'returned', None
            else:
                error = {'type': 'OpenAIResponsesTransportError', 'message': recovery.NETWORK_MESSAGE}
        elif arm == 'direct':
            error, status = None, 'native_admission_denied'
        usage = {'budget': asdict(ExperimentBudget(**budget)), 'physical_requests': len(ledger),
            'prompt_tokens': sum(e['usage'].get('prompt_tokens', 0) for e in ledger),
            'completion_tokens': sum(e['usage'].get('completion_tokens', 0) for e in ledger),
            'charged_tokens': sum(e['charged_tokens'] for e in ledger),
            'all_usage_known': all(e['usage_known'] for e in ledger), 'within_budget': True}
        record = {'index': row['index'], 'id': row['id'], 'model': model, 'arm': arm,
            'terminal': True, 'status': status, 'freeze_sha256': parent_sha,
            'context_sha256': base.text_digest(row['context']), 'context_chars': len(row['context']),
            'domain': row['domain'], 'budget': budget, 'prediction': prediction,
            'content': 'The correct answer is (A)' if prediction else '', 'result': {},
            'error': error, 'measurements': {'fits_native_direct': status != 'native_admission_denied'},
            'usage': usage, 'ledger': ledger}
        path = base.prediction_path(out, model, row['index'], arm)
        base.atomic_json(path, record)
        originals[path.name] = record
    costs.save()
    base.atomic_json(out/'run_summary.json', {'phase': 'inference_complete', 'terminal_records': 75})
    config = {'parent_config': str(parent_path), 'output': str(tmp_path/'recovery'),
              'preflight_cost_upper_usd': '.001364', 'count_retry_delays': [5, 15]}
    return config, parent, rows, originals, selected


class FakeTransport:
    requests = []

    def __init__(self, *, api_key, model, **kwargs):
        self.model = model
        self.count_audit, self.response_audit, self.count_attempt_audit = [], [], []
        self.provider_error_diagnostics = []

    @staticmethod
    def tokens(messages=None, text=None, **kwargs):
        value = text if text is not None else '\n'.join(item['content'] for item in messages)
        return 1_060_000 if 'HIGH' in value else 500_000 if 'LOW' in value else 80

    def count(self, **kwargs):
        count = self.tokens(**kwargs)
        self.count_audit.append({'cache_hit': False, 'status': 'ok', 'input_tokens': count})
        return count

    def request(self, path, body):
        type(self).requests.append((self.model, path))
        text = '\n'.join(item['content'] for item in body['messages'])
        if 'HIGH' in text:
            raise OpenAIResponsesTransportError('OpenAI HTTP request failed (status 400; code context_length_exceeded)')
        count = self.tokens(messages=body['messages'])
        metadata = {'actual_model': self.model, 'returned_service_tier': 'default', 'response_status': 'completed'}
        return {'choices': [{'message': {'content': 'The correct answer is (A)', 'openai_response': metadata}}],
                'usage': {'prompt_tokens': count, 'completion_tokens': 12, 'total_tokens': count+12}}


def harness(row, client):
    assert row.answer == ''
    text = client.complete([{'role': 'user', 'content': 'bounded mock evidence'}], max_tokens=2048)
    return {'mode': 'raw_source', 'prediction': recovery.base.extract_choice(text)}


def test_metadata_eligibility_never_reads_answers_and_excludes_prior_generation():
    class PoisonChoice(dict):
        def get(self, key, *args):
            if key in {'prediction', 'content', 'gold', 'correct', 'result'}:
                raise AssertionError('Selection read a semantic field')
            return super().get(key, *args)
    record = PoisonChoice(terminal=True, status='error', ledger=[],
        usage={'physical_requests': 0, 'charged_tokens': 0},
        error={'type': 'OpenAIResponsesTransportError', 'message': recovery.NETWORK_MESSAGE})
    assert recovery.eligible(record)
    for change in ({'ledger': [{}]}, {'usage': {'physical_requests': 1, 'charged_tokens': 0}},
                   {'usage': {'physical_requests': 0, 'charged_tokens': 1}},
                   {'error': {'type': 'OpenAIResponsesTransportError',
                              'message': 'OpenAI HTTP request failed (status 400; code context_length_exceeded)'}}):
        assert not recovery.eligible({**record, **change})


def test_freeze_subtracts_original_and_preflight_cost_and_preserves_parent(trial, monkeypatch):
    config, parent, rows, originals, selected = trial
    before = {p.name: recovery.base.digest(p) for p in (Path(parent['output'])/'predictions').glob('*.json')}
    real_digest = recovery.base.digest
    def protected(path):
        if Path(path) in (Path(parent['labels']), Path(parent['raw_cache'])):
            raise AssertionError('Freeze opened label-bearing bytes')
        return real_digest(path)
    monkeypatch.setattr(recovery.base, 'digest', protected)
    result = recovery.freeze_recovery(config)
    assert result['expected_recovery_records'] == 39
    frozen, child, *_ = recovery.check_recovery(config)
    assert (frozen['expected_main_records'], frozen['expected_probe_records'],
            frozen['expected_terminal_records']) == (36, 3, 39)
    original_cost = Decimal(frozen['recovery']['parent_generation_cost_upper_usd'])
    assert Decimal(child['cost_cap_usd']) == 50-original_cost-Decimal('.001364')
    assert frozen['recovery']['eligibility_uses_answers_or_correctness'] is False
    assert len(frozen['recovery']['execution_order']) == 39
    assert before == {p.name: recovery.base.digest(p) for p in (Path(parent['output'])/'predictions').glob('*.json')}


def test_recovery_cannot_be_registered_after_labels_or_scoring(trial):
    config, parent, *_ = trial
    marker = Path(parent['output'])/'scoring_access.json'
    marker.write_text('{"fixture":true}')
    with pytest.raises(ValueError, match='precede any original scoring'):
        recovery.freeze_recovery(config)


def test_recovery_39_gate_shared_budgets_and_composite_keep_raw_trial(trial, monkeypatch):
    config, parent, rows, originals, selected = trial
    recovery.freeze_recovery(config)
    with pytest.raises(ValueError, match='39 recovery'):
        recovery.score_recovery(config)
    monkeypatch.setattr(recovery.base, 'verify_guard', lambda config: None)
    original_read = Path.read_text
    def protected(path, *args, **kwargs):
        if path in (Path(parent['labels']), Path(parent['raw_cache'])):
            raise AssertionError('Recovery inference read labels')
        return original_read(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', protected)
    FakeTransport.requests = []
    result = recovery._run_worker(api_key='isolated-unit-test-sentinel', config=config,
        transport_factory=FakeTransport, harness_runner=harness)
    assert result['terminal_recovery_records'] == 39
    request_count = len(FakeTransport.requests)
    recovery._run_worker(api_key='isolated-unit-test-sentinel', config=config,
        transport_factory=FakeTransport, harness_runner=harness)
    assert len(FakeTransport.requests) == request_count
    monkeypatch.setattr(Path, 'read_text', original_read)
    summary = recovery.score_recovery(config)
    assert set(summary) == set(recovery.base.MODELS)
    out = Path(config['output'])
    report = json.loads((out/'audit.json').read_text())
    assert len(report['samples']) == 72 and len(report['capacity_probes']) == 3
    assert report['original_records_preserved'] == 75 and report['new_recovery_records'] == 39
    assert report['first_trial_status_counts']['error'] >= 39
    assert Decimal(report['combined_cost_with_preflight_upper_usd']) <= 50
    raw_audit = Path(parent['output'])/'audit.json'
    before = recovery.base.digest(raw_audit)
    recovery.score_recovery(config)
    assert recovery.base.digest(raw_audit) == before
    assert all(json.loads((Path(parent['output'])/'predictions'/name).read_text()) == value
               for name, value in originals.items())
    # Tampering with a child ceiling is rejected even if a recorded outcome succeeded.
    item = next(item for item in report['recovery_samples'] if item['arm'] != 'capacity_probe')
    path = recovery.base.prediction_path(out, item['model'], item['index'], item['arm'])
    saved = json.loads(path.read_text())
    saved['budget']['max_calls'] += 1
    recovery.base.atomic_json(path, saved)
    with pytest.raises(ValueError, match='logical budget'):
        recovery.score_recovery(config)


class Reply(io.BytesIO):
    def __enter__(self):
        return self
    def __exit__(self, *args):
        self.close()


def test_count_network_retries_are_explicit_but_generation_never_retries(monkeypatch):
    waits = []
    monkeypatch.setattr(recovery.time, 'sleep', waits.append)
    class MockOpener:
        def __init__(self):
            self.counts, self.generations = 0, 0
        def open(self, request, **kwargs):
            if request.full_url.endswith('/responses/input_tokens'):
                self.counts += 1
                if self.counts <= 2:
                    raise URLError(PermissionError(errno.EPERM, 'private-placeholder'))
                return Reply(b'{"object":"response.input_tokens","input_tokens":80}')
            self.generations += 1
            raise URLError(OSError('Tunnel connection failed: 407 private-placeholder'))
    opener = MockOpener()
    transport = recovery.CountRetryTransport(api_key='test-value', model='gpt-6-luna', opener=opener)
    assert transport.count(text='mock') == 80
    assert waits == [5, 15] and opener.counts == 3
    assert [entry['status'] for entry in transport.count_attempt_audit] == ['error', 'error', 'ok']
    with pytest.raises(OpenAIResponsesTransportError):
        transport._post('/responses', {'input': 'mock'})
    assert opener.generations == 1 and waits == [5, 15]
    events = transport.provider_error_diagnostics
    assert events[0]['reason_errno'] == errno.EPERM
    assert events[-1]['tunnel_http_status'] == 407
    assert 'private-placeholder' not in json.dumps(events)


def test_http_observer_rewinds_error_preserves_accounting_and_never_retries(tmp_path):
    secret = 'private-source-and-key-marker'
    body = json.dumps({'error': {'code': 'context_length_exceeded', 'param': 'input',
        'message': "Maximum context length is 262144 tokens. You requested 998674 tokens "
                   "(997650 in the messages, 1024 in the completion). "+secret}}).encode()
    class MockOpener:
        generations = 0
        def open(self, request, **kwargs):
            if request.full_url.endswith('/responses/input_tokens'):
                return Reply(b'{"object":"response.input_tokens","input_tokens":80}')
            self.generations += 1
            raise HTTPError(request.full_url, 400, 'Bad Request', {}, io.BytesIO(body))
    opener = MockOpener()
    transport = recovery.CountRetryTransport(api_key=secret, model='gpt-6-sol', opener=opener)
    client = OpenAIResponsesBudgetedClient(transport, ExperimentBudget())
    costs = recovery.base.MonetaryLedger(tmp_path/'cost.json', cap=1,
        prices=recovery.base.EXPECTED_PRICES, freeze_sha='test')
    capped = recovery.base.CostCappedClient(client, costs, 'harness:mock')
    with pytest.raises(OpenAIResponsesTransportError, match='code context_length_exceeded'):
        capped.complete([{'role': 'user', 'content': 'mock'}], max_tokens=16)
    assert opener.generations == 1 and len(client.ledger) == 1
    assert client.ledger[0]['charged_tokens'] == 112
    assert costs.total() == Decimal('.000400')
    event = transport.provider_error_diagnostics[0]
    assert event['stage'] == 'generation'
    assert event['unique_numeric_fields'] == {
        'maximum_context_length_tokens': 262144, 'requested_total_tokens': 998674,
        'input_tokens': 997650, 'completion_tokens': 1024}
    assert secret not in json.dumps(transport.provider_error_diagnostics)
    unsafe = recovery.safe_http_error({'error': {'code': secret, 'param': secret, 'message': secret}}, 400)
    assert unsafe['code'] is None and unsafe['param'] is None and secret not in json.dumps(unsafe)


def test_parent_key_stays_in_anonymous_pipe(trial, monkeypatch):
    config, *_ = trial
    recovery.freeze_recovery(config)
    monkeypatch.setattr(recovery.sys, 'platform', 'darwin')
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-be-inherited')
    seen = {}
    def subprocess_run(command, **kwargs):
        seen.update(command=command, **kwargs)
        assert json.loads(kwargs['input'])['api_key'] == 'test-pipe-sentinel'
        assert 'OPENAI_API_KEY' not in kwargs['env']
        assert 'test-pipe-sentinel' not in json.dumps(command)
        recovery.base.atomic_json(Path(config['output'])/'run_summary.json', {'phase': 'mock-no-network'})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(recovery.subprocess, 'run', subprocess_run)
    assert recovery.run_recovery(api_key='test-pipe-sentinel', config=config)['phase'] == 'mock-no-network'
    assert '--guarded-worker' in seen['command']
