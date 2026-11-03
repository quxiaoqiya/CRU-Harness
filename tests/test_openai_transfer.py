"""No-network checks for the frozen transfer runner and monetary gate."""
from decimal import Decimal
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from long_context_agent.experiment_budget import BudgetedClient, ExperimentBudget
from scripts import benchmark_openai_transfer as runner


@pytest.fixture
def transfer(tmp_path, monkeypatch):
    source = tmp_path / 'source'
    source.mkdir()
    rows = [{'id': f'sample-{i}', 'context': 'A local fixture.', 'question': 'Return one.'}
            for i in range(12)]
    inputs = source / 'inputs.jsonl'
    inputs.write_text(''.join(json.dumps(row)+'\n' for row in rows))
    labels = source / 'labels.json'
    labels.write_text(json.dumps({row['id']: {'id': row['id'], 'answer': '[1]',
        'answer_type': 'ANSWER_TYPE.NUMERIC', 'dataset': 'fixture', 'context_window_id': i}
        for i, row in enumerate(rows)}))
    scorer = runner.ROOT / 'third_party/oolong/eval_helpers_original.py'
    monkeypatch.setattr(runner, 'EXPECTED_INPUTS_SHA256', runner.digest(inputs))
    monkeypatch.setattr(runner, 'EXPECTED_LABELS_SHA256', runner.digest(labels))
    manifest = source / 'manifest.json'
    manifest.write_text(json.dumps({'inputs_sha256': runner.digest(inputs),
        'labels_sha256': runner.digest(labels), 'scorer_sha256': runner.digest(scorer),
        'selected': [{'id': row['id'], 'context_sha256': runner.text_digest(row['context']),
                      'question_sha256': runner.text_digest(row['question'])} for row in rows]}))
    reference = source / 'reference.json'
    reference.write_text(json.dumps({'code_sha256': {
        'long_context_agent/harness.py': runner.digest(runner.ROOT/'long_context_agent/harness.py')}}))
    config = {'models': list(runner.MODELS), 'confirmed_models': list(runner.MODELS),
        'sample_paths': {'inputs': str(inputs), 'manifest': str(manifest), 'labels': str(labels),
                         'scorer': str(scorer), 'reference_freeze': str(reference)},
        'budget': dict(runner.LIMITS), 'reasoning': 'none', 'output': str(tmp_path/'output'),
        'cost_cap_usd': 15, 'pricing_source': 'Fixture prices for no-network testing only',
        'prices_per_million': {model: {'input': 0.1, 'output': float(runner.R_MAX[model]),
                                     'r_max': float(runner.R_MAX[model])}
                               for model in runner.MODELS}}
    return config, rows


class FakeTransport:
    requests = []

    def __init__(self, *, api_key, model, **_kwargs):
        self.model = model
        self.count_audit, self.response_audit = [], []

    def count(self, **_kwargs):
        self.count_audit.append({'cache_hit': False, 'status': 'ok', 'input_tokens': 50})
        return 50

    def request(self, path, body):
        type(self).requests.append((self.model, path))
        metadata = {'actual_model': self.model, 'response_status': 'completed',
                    'response_id': f'fixture-{len(type(self).requests)}',
                    'provider_usage': {'input_tokens': 50, 'output_tokens': 3, 'total_tokens': 53}}
        self.response_audit.append(metadata)
        return {'choices': [{'message': {'content': 'Answer: 1', 'openai_response': metadata}}],
                'usage': {'prompt_tokens': 50, 'completion_tokens': 3,
                          'openai_responses_metadata': metadata}}


def test_config_rejects_substitution_credentials_and_reasoning_change(transfer):
    config, _ = transfer
    for change in ({'models': ['gpt-6-sol', 'gpt-6-luna', 'gpt-6-terra']},
                   {'api_key': 'a test sentinel'}, {'reasoning': 'low'}, {'cost_cap_usd': 16}):
        with pytest.raises(ValueError):
            runner.public_config({**config, **change})


def test_freeze_is_offline_label_blind_and_rejects_code_change(transfer, monkeypatch):
    config, _ = transfer
    original = runner.digest
    def protected(path):
        if Path(path) == Path(config['sample_paths']['labels']):
            raise AssertionError('Labels were accessed during freeze')
        return original(path)
    monkeypatch.setattr(runner, 'digest', protected)
    assert runner.freeze_transfer(config)['expected_predictions'] == 36
    with pytest.raises(ValueError, match='new empty'):
        runner.freeze_transfer(config)
    current = runner.source_fingerprints()
    monkeypatch.setattr(runner, 'source_fingerprints', lambda: {**current, 'changed.py': '0'*64})
    with pytest.raises(ValueError, match='Frozen code'):
        runner.check_freeze(config)


def test_monetary_gate_blocks_before_generation_and_unknown_cost_survives_resume(tmp_path):
    prices = {'m': {'input': 2, 'output': 10, 'r_max': 10}}
    costs = runner.MonetaryLedger(tmp_path/'costs.json', cap='0.001', prices=prices, freeze_sha='f')
    costs.reserve(model='m', sample_id='s', prompt=50, completion=50)
    with pytest.raises(runner.CostCapExceeded):
        costs.reserve(model='m', sample_id='s', prompt=1, completion=1)
    restored = runner.MonetaryLedger(tmp_path/'costs.json', cap='0.001', prices=prices, freeze_sha='f')
    assert restored.total() == Decimal('0.001')
    restored.settle(restored.entries[0], {'usage_known': True,
        'usage': {'prompt_tokens': 10, 'completion_tokens': 5}})
    assert restored.total() == Decimal('0.00015')
    assert restored.entries[0]['estimated_base_input_output_usd'] == '0.00007'


def test_cost_denial_adds_no_physical_generation(tmp_path):
    costs = runner.MonetaryLedger(tmp_path/'costs.json', cap='0.000001',
        prices={'m': {'input': 1, 'output': 10, 'r_max': 10}}, freeze_sha='f')
    transport = FakeTransport(api_key='test sentinel', model='m')
    client = BudgetedClient(transport, ExperimentBudget())
    capped = runner.CostCappedClient(client, costs, 's')
    before = len(FakeTransport.requests)
    with pytest.raises(runner.CostCapExceeded):
        capped.complete([{'role': 'user', 'content': 'hello'}])
    assert client.ledger == [] and len(FakeTransport.requests) == before


def test_worker_is_label_blind_all_36_before_score_and_no_replay(transfer, monkeypatch, capsys):
    config, _ = transfer
    runner.freeze_transfer(config)
    with pytest.raises(ValueError, match='36 terminal'):
        runner.score_transfer(config)
    # The real public entry must verify sandbox denial; only this isolated
    # unit test replaces that OS check and uses an in-memory fake transport.
    monkeypatch.setattr(runner, 'verify_guard', lambda _config: None)
    read_text = Path.read_text
    def protected(path, *args, **kwargs):
        if path == Path(config['sample_paths']['labels']):
            raise AssertionError('The inference worker opened labels')
        return read_text(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', protected)
    FakeTransport.requests = []
    public = runner.public_config(config)
    assert runner._run_worker(api_key='unit-test-sentinel', config=public,
                              transport_factory=FakeTransport)['terminal_predictions'] == 36
    assert len(FakeTransport.requests) == 36
    runner._run_worker(api_key='unit-test-sentinel', config=public, transport_factory=FakeTransport)
    assert len(FakeTransport.requests) == 36
    for file in Path(config['output']).rglob('*.json'):
        assert 'unit-test-sentinel' not in file.read_text()
    assert 'unit-test-sentinel' not in capsys.readouterr().out
    monkeypatch.setattr(Path, 'read_text', read_text)
    summary = runner.score_transfer(config)
    assert set(summary) == set(runner.MODELS)
    assert all(item['n'] == 12 and item['full_credit'] == 12 for item in summary.values())


def test_guard_probe_refuses_execution_before_any_generation(transfer):
    config, _ = transfer
    runner.freeze_transfer(config)
    before = len(FakeTransport.requests)
    with pytest.raises(RuntimeError, match='guard probe is readable'):
        runner._run_worker(api_key='unit-test-sentinel', config=runner.public_config(config),
                           transport_factory=FakeTransport)
    assert len(FakeTransport.requests) == before


def test_parent_key_uses_only_anonymous_stdin_pipe(transfer, monkeypatch):
    config, _ = transfer
    runner.freeze_transfer(config)
    monkeypatch.setattr(runner.sys, 'platform', 'darwin')
    monkeypatch.setenv('OPENAI_API_KEY', 'must-not-be-inherited')
    seen = {}
    def child(command, **kwargs):
        seen.update(command=command, **kwargs)
        payload = json.loads(kwargs['input'])
        assert payload['api_key'] == 'unit-test-sentinel'
        runner.atomic_json(Path(config['output'])/'run_summary.json', {'phase': 'fake-no-network'})
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', child)
    assert runner.run_transfer(api_key='unit-test-sentinel', config=config)['phase'] == 'fake-no-network'
    assert 'unit-test-sentinel' not in json.dumps(seen['command'])
    assert 'OPENAI_API_KEY' not in seen['env']


def test_explicit_model_block_never_substitutes_or_calls_that_model(transfer, monkeypatch, capsys):
    config, _ = transfer
    runner.freeze_transfer(config)
    runner.record_model_block(config, runner.MODELS[0], 'forbidden')
    monkeypatch.setattr(runner, 'verify_guard', lambda _config: None)
    FakeTransport.requests = []
    runner._run_worker(api_key='unit-test-sentinel', config=runner.public_config(config),
                       transport_factory=FakeTransport)
    assert len(FakeTransport.requests) == 24
    assert all(model != runner.MODELS[0] for model, _ in FakeTransport.requests)
    report = runner.score_transfer(config)
    assert report[runner.MODELS[0]]['status'] == 'model_blocked_not_an_accuracy_estimate'
    assert 'mean_official_score' not in report[runner.MODELS[0]]


def test_interrupted_task_preserves_previous_usage_and_count_audit_without_replay(transfer, monkeypatch, capsys):
    config, rows = transfer
    runner.freeze_transfer(config)
    monkeypatch.setattr(runner, 'verify_guard', lambda _config: None)
    FakeTransport.requests = []
    public = runner.public_config(config)
    runner._run_worker(api_key='unit-test-sentinel', config=public, transport_factory=FakeTransport)
    out = Path(config['output'])
    target = runner.prediction_path(out, runner.MODELS[0], rows[0]['id'])
    previous = json.loads(target.read_text())
    target.unlink()  # simulate an interrupted final artifact write
    runner._run_worker(api_key='unit-test-sentinel', config=public, transport_factory=FakeTransport)
    restored = json.loads(target.read_text())
    assert len(FakeTransport.requests) == 36
    assert restored['error']['type'] == 'InterruptedTask'
    assert restored['ledger'] == previous['ledger']
    assert restored['count_audit'] == previous['count_audit']
    assert restored['usage']['charged_tokens'] == previous['usage']['charged_tokens']
    assert runner.score_transfer(config)[runner.MODELS[0]]['full_credit'] == 11
