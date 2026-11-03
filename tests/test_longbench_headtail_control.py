"""Synthetic no-network checks for the conditional head/tail quality control."""
from decimal import Decimal
import json
from pathlib import Path

import pytest

from scripts import benchmark_longbench_headtail_control as control


def test_money_only_selection_and_balanced_source_projection():
    assert control.money_plan(13)['planned_indices'] == [143, 260]
    assert control.money_plan('12.99')['planned_indices'] == [143]
    assert control.money_plan('6.5')['expected_control_records'] == 3
    assert control.money_plan('6.49')['expected_control_records'] == 0
    text, spans, dropped = control.character_headtail('0123456789abcdefghij', 7)
    assert text == '0123hij'
    assert spans == [[0, 4], [17, 20]] and dropped == [4, 17]
    source = '0123456789abcdefghij'
    assert ''.join(source[a:b] for a, b in spans) == text


def test_prepare_is_finite_exact_for_all_models_and_does_not_generate():
    value = {'id': 'mock', '_id': 'mock', 'index': 143,
        'context': 'abcdefghij'*5000, 'question': 'Choose.', 'choice_A': 'one',
        'choice_B': 'two', 'choice_C': 'three', 'choice_D': 'four'}
    class Counter:
        def __init__(self, factor):
            self.factor, self.prompts = factor, []
        def count_messages(self, messages):
            self.prompts.append(messages[0]['content'])
            return len(messages[0]['content'])*self.factor
    clients = {model: Counter(100+index) for index, model in enumerate(control.MODELS)}
    prepared, item = control.prepare_source(value, 1_000_000, clients)
    assert item['inserted_characters'] == 0 and item['dropped_middle_span'] is not None
    assert len(item['preparation_rounds']) <= 8
    assert all(number+1024 <= 800000 for number in item['direct_admission_tokens_by_model'].values())
    assert len({client.prompts[-1] for client in clients.values()}) == 1
    assert prepared['context'] == ''.join(value['context'][a:b] for a, b in item['source_spans'])
    calls = []
    class AlwaysOver:
        def count_messages(self, messages):
            calls.append(1)
            return 900000
    with pytest.raises(ValueError, match='eight exact-count'):
        control.prepare_source(value, 1_000_000, {model: AlwaysOver() for model in control.MODELS})
    assert len(calls) == 8*3


def test_strict_parser_is_raw_official_style_not_a_compatibility_letter():
    assert control.strict_official_choice('A') is None
    assert control.strict_official_choice('(A)') is None
    assert control.strict_official_choice('**The correct answer is (A)**') == 'A'
    assert control.strict_official_choice('The correct answer is B') == 'B'
    assert control.strict_official_choice('the correct answer is (A)') is None
    assert control.base.extract_choice('the correct answer is (A)') == 'A'


@pytest.fixture
def synthetic_control(tmp_path, monkeypatch):
    parent_out, rec_out, private = tmp_path/'original', tmp_path/'recovery', tmp_path/'private'
    for path in (parent_out, rec_out, private):
        path.mkdir()
    labels = private/'labels.json'
    labels.write_text('{"143":"A","260":"A"}')
    rows = [{'id': f'mock-{index}', '_id': f'mock-{index}', 'index': index,
        'context': ('abcdefghij'*5000)+str(index), 'question': 'Choose.',
        'choice_A': 'one', 'choice_B': 'two', 'choice_C': 'three', 'choice_D': 'four',
        'domain': 'Code Repository Understanding', 'sub_domain': 'mock', 'length': 'long',
        'difficulty': 'hard'} for index in (143, 260)]
    state = {'scored': False, 'remaining': '20'}
    config = {'recovery_config': str(tmp_path/'recovery_config.json'), 'output': str(tmp_path/'headtail')}
    child = {'base_url': 'https://api.openai.com/v1', 'timeout': 900,
             'long_direct_spacing_seconds': 0, 'prices': control.base.EXPECTED_PRICES}
    def context(value, require_unscored=False):
        if require_unscored and state['scored']:
            raise ValueError('Already scored')
        return {'config': control.public_config(value), 'recovery_config': {},
            'recovery_out': rec_out, 'recovery_frozen': {}, 'child_config': child,
            'parent_config': {'output': str(parent_out), 'labels': str(labels), 'raw_cache': str(tmp_path/'raw.jsonl')},
            'rows': rows, 'manifest': {'selected': [{'index': row['index'],
                'preflight_native_direct': {'direct_admission_input_tokens': 4_000_000}} for row in rows]},
            'recovery_freeze_sha256': '1'*64, 'parent_cost_upper_usd': '10',
            'recovery_cost_upper_usd': str(Decimal('39.998636')-Decimal(state['remaining'])),
            'preflight_cost_upper_usd': '.001364', 'global_cap_usd': '50',
            'remaining_before_control_usd': state['remaining'], 'recovery_record_sha256': {},
            'parent_cost_ledger_sha256': '2'*64, 'recovery_cost_ledger_sha256': '3'*64}
    def score_recovery(value):
        state['scored'] = True
        control.base.atomic_json(parent_out/'audit.json', {'mock': True})
        control.base.atomic_json(rec_out/'scoring_access.json', {'mock': True})
        control.base.atomic_json(rec_out/'audit.json', {'samples': [
            {'index': row['index'], 'model': model, 'arm': 'harness', 'task_success': True}
            for row in rows for model in control.MODELS]})
        return {}
    monkeypatch.setattr(control, '_context', context)
    monkeypatch.setattr(control.recovery, 'score_recovery', score_recovery)
    monkeypatch.setattr(control.base, 'verify_guard', lambda value: None)
    return config, state, labels


class FakeTransport:
    generations = 0
    wrong_model = False

    def __init__(self, *, api_key, model, **kwargs):
        self.model = model
        self.count_audit, self.response_audit, self.count_attempt_audit = [], [], []
        self.provider_error_diagnostics = []

    def count(self, *, messages=None, text=None, **kwargs):
        value = text if text is not None else '\n'.join(item['content'] for item in messages)
        number = len(value)*2
        self.count_audit.append({'input_tokens': number, 'cache_hit': False, 'status': 'ok'})
        return number

    def request(self, path, body):
        type(self).generations += 1
        number = self.count(messages=body['messages'])
        meta = {'actual_model': 'gpt-other' if self.wrong_model else self.model,
                'returned_service_tier': 'default', 'response_status': 'completed'}
        self.response_audit.append(meta)
        return {'choices': [{'message': {'content': '**The correct answer is (A)**', 'openai_response': meta}}],
                'usage': {'prompt_tokens': number, 'completion_tokens': 12, 'total_tokens': number+12}}


def test_all_control_gates_before_labels_and_no_repeat_generation(synthetic_control):
    config, state, labels = synthetic_control
    FakeTransport.generations = 0
    result = control.prepare_control(api_key='unit-test-private-value', config=config, transport_factory=FakeTransport)
    assert result['expected_control_records'] == 6
    assert FakeTransport.generations == 0
    with pytest.raises(ValueError, match='all registered control'):
        control.score_control(config)
    assert not state['scored']
    control._run_worker(api_key='unit-test-private-value', config=config, transport_factory=FakeTransport)
    assert FakeTransport.generations == 6
    control._run_worker(api_key='unit-test-private-value', config=config, transport_factory=FakeTransport)
    assert FakeTransport.generations == 6
    summary = control.score_control(config)
    assert state['scored']
    assert all(value['n_planned'] == 2 and value['primary_correct'] == 2
               and value['strict_official_correct'] == 2 for value in summary.values())
    assert all('unit-test-private-value' not in path.read_text()
               for path in Path(config['output']).rglob('*.json'))


def test_budget_skip_still_provides_explicit_zero_control_gate(synthetic_control):
    config, state, _ = synthetic_control
    state['remaining'] = '6'
    FakeTransport.generations = 0
    prepared = control.prepare_control(api_key='', config=config, transport_factory=FakeTransport)
    assert prepared['expected_control_records'] == 0
    assert (Path(config['output'])/'budget_skip.json').exists()
    run = control._run_worker(api_key='', config=config, transport_factory=FakeTransport)
    assert run['budget_skipped'] and run['terminal_control_records'] == 0
    summary = control.score_control(config)
    assert FakeTransport.generations == 0
    assert all(value['n_planned'] == 0 and value['primary_accuracy_all_planned'] is None for value in summary.values())


def test_actual_control_model_guard_fails_before_label_scoring(synthetic_control, monkeypatch):
    config, state, _ = synthetic_control
    control.prepare_control(api_key='unit-test-private-value', config=config, transport_factory=FakeTransport)
    monkeypatch.setattr(FakeTransport, 'wrong_model', True)
    control._run_worker(api_key='unit-test-private-value', config=config, transport_factory=FakeTransport)
    with pytest.raises(ValueError, match='model/tier'):
        control.score_control(config)
    assert not state['scored']
