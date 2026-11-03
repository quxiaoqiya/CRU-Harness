"""No-network checks for native-direct / bounded-Harness registered execution."""
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from long_context_agent.openai_responses_transport import OpenAIResponseNotCompletedError, OpenAIResponsesTransportError
from scripts import benchmark_longbench_openai_limits as runner


@pytest.fixture
def experiment(tmp_path):
    source = tmp_path/'source'
    source.mkdir()
    private = tmp_path/'private_labels'
    private.mkdir()
    rows = [{'_id': f'fixture-{i}', 'id': f'fixture-{i}', 'index': i,
             'domain': 'Multi-Document QA', 'sub_domain': 'fixture',
             'difficulty': 'easy', 'length': 'long',
             'context': ('LOW' if i < 6 else 'HIGH')+f' complete fixture source {i}.',
             'question': 'Choose the fixture option.', 'choice_A': 'one',
             'choice_B': 'two', 'choice_C': 'three', 'choice_D': 'four'} for i in range(12)]
    inputs, labels, manifest, cache = source/'inputs.jsonl', private/'labels.json', source/'manifest.json', source/'raw_cache.jsonl'
    inputs.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    labels.write_text(json.dumps({str(row['index']): 'A' for row in rows}))
    cache.write_text('Fixture source-bearing cache is denied to the worker.\n')
    manifest.write_text(json.dumps({
        'inputs_sha256': runner.digest(inputs), 'labels_sha256': runner.digest(labels),
        'cache_sha256': runner.digest(cache), 'probe_index': 6,
        'selected': [{'index': row['index'], 'id': row['_id'],
                      'context_sha256': runner.text_digest(row['context']),
                      'prompt_sha256': runner.text_digest(runner.official_prompt(runner.make_row(row))),
                      'context_chars': len(row['context'])} for row in rows]}))
    config = {'models': list(runner.MODELS), 'inputs': str(inputs), 'labels': str(labels),
              'manifest': str(manifest), 'raw_cache': str(cache), 'output': str(tmp_path/'output'),
              'budget_harness': dict(runner.HARNESS_LIMITS), 'budget_direct': dict(runner.DIRECT_LIMITS),
              'reasoning': 'none', 'timeout': 900, 'cost_cap_usd': 50,
              'prices': runner.EXPECTED_PRICES, 'probes_per_model': 1,
              'long_direct_spacing_seconds': 0}
    return config, rows


class FakeTransport:
    requests = []
    wrong_bounded_model = False

    def __init__(self, *, api_key, model, **kwargs):
        self.model = model
        self.count_audit, self.response_audit = [], []

    @staticmethod
    def tokens(*, messages=None, text=None, **kwargs):
        content = text if text is not None else '\n'.join(x['content'] for x in messages)
        if 'HIGH' in content:
            return 1_060_000
        if 'LOW' in content:
            return 500_000
        return 80

    def count(self, **kwargs):
        value = self.tokens(**kwargs)
        self.count_audit.append({'cache_hit': False, 'status': 'ok', 'input_tokens': value})
        return value

    def request(self, path, body):
        type(self).requests.append((self.model, path))
        text = '\n'.join(x['content'] for x in body['messages'])
        if 'HIGH' in text:
            raise OpenAIResponsesTransportError(
                'OpenAI HTTP request failed (status 400; code context_length_exceeded)')
        count = self.tokens(messages=body['messages'])
        actual = 'gpt-other' if self.wrong_bounded_model and 'bounded fixture' in text else self.model
        metadata = {'actual_model': actual, 'returned_service_tier': 'default',
                    'response_status': 'completed', 'response_id': f'fixture-{len(self.requests)}',
                    'provider_usage': {'input_tokens': count, 'output_tokens': 12, 'total_tokens': count+12}}
        self.response_audit.append(metadata)
        return {'choices': [{'message': {'content': 'The correct answer is (A)',
                                         'openai_response': metadata}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': count, 'completion_tokens': 12, 'total_tokens': count+12,
                          'openai_responses_metadata': metadata}}


def bounded_harness(row, client):
    assert row.answer == ''
    answer = client.complete([{'role': 'user', 'content': 'bounded fixture evidence'}], max_tokens=2048)
    return {'mode': 'raw_source', 'prediction': runner.extract_choice(answer)}


def execute(config, monkeypatch, transport=FakeTransport):
    monkeypatch.setattr(runner, 'verify_guard', lambda value: None)
    return runner._run_worker(api_key='no-network-test-sentinel', config=config,
                              transport_factory=transport, harness_runner=bounded_harness)


def test_exact_config_rejects_substitution_secrets_and_fake_native_budget(experiment):
    config, _ = experiment
    for change in ({'models': ['gpt-6.1-sol', 'gpt-6-luna', 'gpt-6.1-terra']},
                   {'api_key': 'sentinel'}, {'reasoning': 'high'},
                   {'cost_cap_usd': 51}, {'budget_direct': dict(runner.HARNESS_LIMITS)}):
        with pytest.raises(ValueError):
            runner.public_config({**config, **change})


def test_freeze_never_reads_labels_or_raw_cache_and_covers_label_directory(experiment, monkeypatch):
    config, _ = experiment
    original = runner.digest
    def digest(path):
        if Path(path) in (Path(config['labels']), Path(config['raw_cache'])):
            raise AssertionError('Label-bearing bytes were accessed during freeze')
        return original(path)
    monkeypatch.setattr(runner, 'digest', digest)
    result = runner.freeze_limits(config)
    assert result['expected_terminal_records'] == 75
    profile = (Path(config['output'])/'labels_guard.sb').read_text()
    assert str(Path(config['labels']).parent) in profile
    assert str(runner.ROOT/'docs') in profile and str(runner.ROOT/'benchmarks') in profile
    assert config['raw_cache'] in profile
    with pytest.raises(ValueError, match='new empty'):
        runner.freeze_limits(config)
    current = runner.source_fingerprints()
    monkeypatch.setattr(runner, 'source_fingerprints', lambda: {**current, 'changed.py': '0'*64})
    with pytest.raises(ValueError, match='Frozen'):
        runner.check_freeze(config)


def test_input_envelope_long_tier_reservation_and_unknown_resume(tmp_path):
    costs = runner.MonetaryLedger(tmp_path/'cost.json', cap=50,
        prices=runner.EXPECTED_PRICES, freeze_sha='fixture')
    assert costs.rates('gpt-6-sol', 272_000) == (Decimal('2.5'), Decimal('10'))
    assert costs.rates('gpt-6-sol', 272_001) == (Decimal('5.0'), Decimal('15.0'))
    entry = costs.reserve(model='gpt-6-sol', sample_id='direct:1', prompt=1_000_000, completion=1024)
    assert Decimal(entry['estimated_upper_usd']) == Decimal('5.01536')
    costs.settle(entry, {'usage_known': False})
    restored = runner.MonetaryLedger(tmp_path/'cost.json', cap=50,
        prices=runner.EXPECTED_PRICES, freeze_sha='fixture')
    assert restored.total() == Decimal('5.01536')
    restored.settle(restored.entries[0], {'usage_known': True,
        'usage': {'prompt_tokens': 999_000, 'completion_tokens': 10}})
    assert restored.total() == Decimal('4.99515')
    tiny = runner.MonetaryLedger(tmp_path/'tiny.json', cap='.00001',
        prices=runner.EXPECTED_PRICES, freeze_sha='fixture')
    with pytest.raises(runner.CostCapExceeded):
        tiny.reserve(model='gpt-6-sol', sample_id='harness:1', prompt=100, completion=16)
    assert tiny.entries == []


def test_no_scoring_before_all_75_and_native_denial_is_not_wrong_answer(experiment, monkeypatch, capsys):
    config, _ = experiment
    runner.freeze_limits(config)
    with pytest.raises(ValueError, match='75 terminal'):
        runner.score_limits(config)
    original = Path.read_text
    def protected(path, *args, **kwargs):
        if path == Path(config['labels']) or path == Path(config['raw_cache']):
            raise AssertionError('Inference opened label-bearing data')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', protected)
    FakeTransport.requests = []
    assert execute(config, monkeypatch)['terminal_records'] == 75
    assert len(FakeTransport.requests) == 57  # 18 direct, 36 Harness, 3 failed probes.
    execute(config, monkeypatch)
    assert len(FakeTransport.requests) == 57
    for path in Path(config['output']).rglob('*.json'):
        assert 'no-network-test-sentinel' not in path.read_text()
    assert 'no-network-test-sentinel' not in capsys.readouterr().out
    monkeypatch.setattr(Path, 'read_text', original)
    summary = runner.score_limits(config)
    for model in runner.MODELS:
        assert summary[model]['direct']['native_admission_denied'] == 6
        assert summary[model]['direct']['accuracy_among_available_answers'] == 1
        assert summary[model]['harness']['correct_answers'] == 12
        assert summary[model]['paired_available']['n'] == 6
    report = json.loads((Path(config['output'])/'audit.json').read_text())
    denied = [item for item in report['samples'] if item['status'] == 'native_admission_denied']
    assert len(denied) == 18 and all(item['correct'] is None for item in denied)
    assert all(item['status'] == 'provider_context_rejected' for item in report['capacity_probes'])


def test_interrupted_attempt_retains_ledger_and_never_replays(experiment, monkeypatch):
    config, rows = experiment
    runner.freeze_limits(config)
    FakeTransport.requests = []
    execute(config, monkeypatch)
    path = runner.prediction_path(config['output'], runner.MODELS[0], rows[0]['index'], 'harness')
    saved = json.loads(path.read_text())
    path.unlink()
    execute(config, monkeypatch)
    restored = json.loads(path.read_text())
    assert len(FakeTransport.requests) == 57
    assert restored['status'] == 'interrupted'
    assert restored['error']['type'] == 'InterruptedTask'
    assert restored['ledger'] == saved['ledger']
    assert restored['count_audit'] == saved['count_audit']
    assert restored['prediction'] is None


def test_counter_failure_is_terminal_error_including_probe(experiment, monkeypatch):
    config, _ = experiment
    runner.freeze_limits(config)
    class FailedCounter(FakeTransport):
        def count(self, **kwargs):
            raise OpenAIResponsesTransportError('Fixture counter unavailable')
    FakeTransport.requests = []
    execute(config, monkeypatch, FailedCounter)
    records = [json.loads(path.read_text()) for path in (Path(config['output'])/'predictions').glob('*.json')]
    assert len(records) == 75 and all(record['status'] == 'error' for record in records)
    assert all(record['usage']['physical_requests'] == 0 for record in records)
    assert not FakeTransport.requests


def test_primary_native_pairs_retain_incomplete_direct_and_failed_harness(experiment, monkeypatch):
    config, _ = experiment
    runner.freeze_limits(config)
    class IncompleteDirect(FakeTransport):
        def request(self, path, body):
            text = '\n'.join(item['content'] for item in body['messages'])
            if self.model == runner.MODELS[0] and 'LOW complete fixture source 0.' in text:
                type(self).requests.append((self.model, path))
                count = self.tokens(messages=body['messages'])
                metadata = {'actual_model': self.model, 'returned_service_tier': 'default',
                            'response_status': 'incomplete', 'incomplete_reason': 'max_output_tokens'}
                raise OpenAIResponseNotCompletedError(
                    usage={'prompt_tokens': count, 'completion_tokens': 1024, 'total_tokens': count+1024},
                    metadata=metadata)
            return super().request(path, body)
    def partly_failed_harness(row, client):
        if client.model == runner.MODELS[0] and row.offset == 1:
            raise RuntimeError('Fixture explicit Harness failure')
        return bounded_harness(row, client)
    monkeypatch.setattr(runner, 'verify_guard', lambda value: None)
    runner._run_worker(api_key='no-network-test-sentinel', config=config,
                      transport_factory=IncompleteDirect, harness_runner=partly_failed_harness)
    summary = runner.score_limits(config)[runner.MODELS[0]]
    assert summary['direct']['n_native_admissible'] == 6
    assert summary['direct']['primary_task_success_rate_native_admissible'] == 5/6
    assert summary['direct']['n_answer_available'] == 5
    assert summary['harness']['primary_task_success_rate_all_planned'] == 11/12
    assert summary['paired_native_admissible']['n'] == 6
    assert summary['paired_native_admissible']['harness_only_correct'] == 1
    assert summary['paired_native_admissible']['direct_only_correct'] == 1
    assert summary['paired_available']['n'] == 4


def test_caught_controller_identity_failure_cannot_be_scored(experiment, monkeypatch):
    config, _ = experiment
    runner.freeze_limits(config)
    monkeypatch.setattr(FakeTransport, 'wrong_bounded_model', True)
    execute(config, monkeypatch)
    path = runner.prediction_path(config['output'], runner.MODELS[0], 0, 'harness')
    result = json.loads(path.read_text())
    assert result['status'] == 'error' and result['prediction'] is None
    with pytest.raises(ValueError, match='identity or service tier'):
        runner.score_limits(config)


def test_parent_passes_secret_only_in_stdin_and_strips_environment(experiment, monkeypatch):
    config, _ = experiment
    runner.freeze_limits(config)
    monkeypatch.setattr(runner.sys, 'platform', 'darwin')
    monkeypatch.setenv('OPENAI_API_KEY', 'do-not-inherit-test-value')
    captured = {}
    def child(command, **kwargs):
        captured.update(command=command, **kwargs)
        assert json.loads(kwargs['input'])['api_key'] == 'private-unit-test-value'
        assert 'OPENAI_API_KEY' not in kwargs['env']
        assert 'private-unit-test-value' not in json.dumps(command)
        runner.atomic_json(Path(config['output'])/'run_summary.json', {'phase': 'fixture_no_network'})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', child)
    assert runner.run_limits(api_key='private-unit-test-value', config=config)['phase'] == 'fixture_no_network'
    assert '--guarded-worker' in captured['command']


def test_os_guard_probe_denies_unguarded_worker_before_inference(experiment):
    config, _ = experiment
    runner.freeze_limits(config)
    with pytest.raises(RuntimeError, match='probe is readable'):
        runner._run_worker(api_key='fixture', config=config, transport_factory=FakeTransport,
                           harness_runner=bounded_harness)
