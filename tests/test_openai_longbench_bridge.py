"""Offline Responses/controller integration; counts and outputs are scripted."""
from __future__ import annotations

from collections import deque
from dataclasses import replace
import json

import pytest

from long_context_agent.experiment_budget import BudgetExhausted, BudgetIntegrityError, ExperimentBudget
from long_context_agent.openai_longbench_bridge import (
    ExactBudgetEstimator, LongBenchBudgetBridge, LongBenchEvidenceConfig, LongBenchResponsesTransport,
    run_longbench_harness,
)
from long_context_agent.openai_responses_transport import (
    OpenAIResponseNotCompletedError, OpenAIResponsesBudgetedClient,
    OpenAIResponsesTransport, OpenAIResponsesTransportError,
)
from scripts.longbench_v2_protocol import LongBenchRow


UNKNOWN_OPTIONS = json.dumps({'options': {
    letter: {'status': 'unknown', 'evidence_spans': [], 'reason': ''}
    for letter in 'ABCD'}, 'recommended': ''})


class ScriptedResponses(LongBenchResponsesTransport):
    """No socket/credential access; simulated counts are not benchmark data."""

    def __init__(self, replies=(), *, incomplete=False, delivery_error=False,
                 actual_model='gpt-6.1-terra'):
        super().__init__(api_key='test-local-only', model='gpt-6.1-terra')
        self.replies = deque(replies)
        self.incomplete = incomplete
        self.delivery_error = delivery_error
        self.actual_model = actual_model
        self.generation_bodies = []
        self.count_bodies = []

    @staticmethod
    def scripted_count(value):
        if isinstance(value, str):
            return 4 + len(value)//4
        def text_length(content):
            if isinstance(content, str):
                return len(content)
            assert all(part.get('type') == 'input_text' and isinstance(part.get('text'), str) for part in content)
            return sum(len(part['text']) for part in content)
        return 4 + sum(4+text_length(item['content'])//4 for item in value)

    def _post(self, path, body):
        if path == '/responses/input_tokens':
            self.count_bodies.append(body)
            return {'object': 'response.input_tokens',
                    'input_tokens': self.scripted_count(body['input'])}
        assert path == '/responses'
        self.generation_bodies.append(body)
        if self.delivery_error:
            raise OpenAIResponsesTransportError('Scripted uncertain delivery')
        reply = self.replies.popleft() if self.replies else 'The correct answer is (A)'
        inp = self.scripted_count(body['input'])
        out = body['max_output_tokens'] if self.incomplete else min(body['max_output_tokens'], max(1, len(reply)//8))
        return {'id': 'resp_fixture_'+str(len(self.generation_bodies)),
                'model': self.actual_model, 'status': 'incomplete' if self.incomplete else 'completed',
                'service_tier': 'default',
                'incomplete_details': {'reason': 'max_output_tokens'} if self.incomplete else None,
                'usage': {'input_tokens': inp, 'output_tokens': out, 'total_tokens': inp+out,
                          'input_tokens_details': {'cached_tokens': 0},
                          'output_tokens_details': {'reasoning_tokens': 0}},
                'output': [{'type': 'message', 'role': 'assistant', 'status': 'completed',
                            'content': [{'type': 'output_text', 'text': reply}]}]}


def budgeted(transport=None, **changes):
    budget = ExperimentBudget(**({'context_tokens': 16_384, 'total_tokens': 65_536,
                                 'max_calls': 32, 'max_output_tokens': 2_048} | changes))
    return OpenAIResponsesBudgetedClient(transport or ScriptedResponses(), budget)


def long_row(**changes):
    row = LongBenchRow(item_id='fixture-longbench', domain='Single-Document QA',
        sub_domain='fixture', difficulty='hard', length='long',
        question='What is the color of the flag?', choice_a='red', choice_b='blue',
        choice_c='green', choice_d='yellow', answer='',
        context=('The flag is red. The city archive records the flag color.\n'*4_000), offset=0)
    return replace(row, **changes)


@pytest.mark.parametrize('messages,text', [
    (None, None), (None, ''), (None, 'Short plain text.'),
    ([{'role': 'system', 'content': 'Instructions'}, {'role': 'user', 'content': '中文🙂\nQuestion'}], None),
])
def test_short_input_wire_stays_identical_to_published_transport(messages, text):
    assert LongBenchResponsesTransport._input(messages=messages, text=text) == OpenAIResponsesTransport._input(messages=messages, text=text)


def test_unicode_content_parts_preserve_every_character_role_and_order():
    class SmallParts(LongBenchResponsesTransport):
        TEXT_PART_CHARS = 7
    text = '前缀🙂e\u0301\r\n' + '\x00雪山\u200d🚀\t中间' + '尾端\n'
    encoded = SmallParts._input(messages=[{'role': 'developer', 'content': 'Rule'},
                                         {'role': 'user', 'content': text}])
    assert encoded[0] == {'role': 'developer', 'content': 'Rule'}
    assert encoded[1]['role'] == 'user'
    parts = encoded[1]['content']
    assert len(parts) > 1 and all(0 < len(part['text']) <= 7 for part in parts)
    assert all(set(part) == {'type', 'text'} and part['type'] == 'input_text' for part in parts)
    assert ''.join(part['text'] for part in parts) == text
    assert ''.join(part['text'] for part in parts).encode('utf-8') == text.encode('utf-8')


def test_bare_long_text_uses_lossless_equivalent_user_message():
    class SmallParts(LongBenchResponsesTransport):
        TEXT_PART_CHARS = 5
    text = 'ABC🙂\n123雪' * 3
    encoded = SmallParts._input(text=text)
    assert len(encoded) == 1 and encoded[0]['role'] == 'user'
    assert ''.join(part['text'] for part in encoded[0]['content']) == text
    assert SmallParts._input(text='ABCDE') == 'ABCDE'
    assert SmallParts._input(messages=[{'role': 'user', 'content': 'ABCDE'}]) == [{'role': 'user', 'content': 'ABCDE'}]


@pytest.mark.parametrize('kwargs', [
    {'messages': [{'role': 'user', 'content': [{'type': 'input_text', 'text': 'externally supplied'}]}]},
    {'messages': [{'role': 'tool', 'content': 'Unsupported role'}]},
    {'messages': [{'role': 'user', 'content': 'Text', 'name': 'unsupported'}]},
    {'text': 'text', 'messages': []},
])
def test_multipart_encoding_preserves_strict_external_input_validation(kwargs):
    with pytest.raises((ValueError, NotImplementedError)):
        LongBenchResponsesTransport._input(**kwargs)


def test_over_ten_million_characters_count_and_generation_share_complete_parts():
    # Deliberately exceeds the observed 10,485,760-character single-field
    # limit. This mock checks encoding, not model/native context capacity.
    text = '前🙂\n' + 'a'*10_485_761 + '\r\n末尾雪'
    messages = [{'role': 'system', 'content': 'Answer briefly'}, {'role': 'user', 'content': text}]
    transport = ScriptedResponses(['The correct answer is (A)'])
    count = transport.count(messages=messages)
    response = transport.request('/chat/completions', {
        'model': transport.model, 'messages': messages, 'max_tokens': 64,
        'seed': 20261102, 'temperature': 0, 'chat_template_kwargs': {'enable_thinking': False}})
    assert len(transport.count_bodies) == len(transport.generation_bodies) == 1
    counted, generated = transport.count_bodies[0], transport.generation_bodies[0]
    assert counted['input'] == generated['input']
    assert counted['reasoning'] == generated['reasoning'] == {'effort': 'none'}
    assert counted['truncation'] == generated['truncation'] == 'disabled'
    assert generated['store'] is False and generated['service_tier'] == 'default'
    assert not {'seed', 'temperature', 'chat_template_kwargs', 'extra_body'} & generated.keys()
    parts = generated['input'][1]['content']
    assert len(parts) == 2 and max(len(part['text']) for part in parts) <= 10_000_000
    assert ''.join(part['text'] for part in parts) == text
    assert response['usage']['prompt_tokens'] == count
    config = transport.configuration()
    assert config['text_part_character_limit'] == 10_000_000
    assert config['text_concatenation'] == 'lossless_no_inserted_delimiters'
    assert config['prompt_cache_ttl_default'] == '30m' and not config['prompt_cache_ttl_overridden']
    assert config['implicit_retries'] == 0 and config['short_input_wire_unchanged'] is True


def test_one_legacy_call_is_one_budgeted_generation_with_wire_parameters():
    transport = ScriptedResponses(['The correct answer is (B)'])
    client = budgeted(transport)
    bridge = LongBenchBudgetBridge(client)
    result = bridge.complete('Question and local evidence', max_tokens=64,
                             enable_thinking=False, preserve_thinking=True, reasoning_effort=None)
    assert result.content == 'The correct answer is (B)'
    assert result.reasoning == '' and result.attempts == 1 and not result.fallback_used
    assert bridge.calls == len(transport.generation_bodies) == 1
    assert result.usage['total_tokens'] == client.ledger[0]['charged_tokens']
    body = transport.generation_bodies[0]
    assert body['reasoning'] == {'effort': 'none'}
    assert body['store'] is False and body['truncation'] == 'disabled'
    assert body['service_tier'] == 'default'
    assert not {'seed', 'temperature', 'chat_template_kwargs', 'extra_body'} & body.keys()


def test_empty_completed_answer_has_no_implicit_recovery():
    transport = ScriptedResponses(['', 'The correct answer is (A)'])
    bridge = LongBenchBudgetBridge(budgeted(transport))
    result = bridge.complete_answer('Question', max_tokens=64, enable_thinking=False)
    assert result.content == '' and result.attempts == 1
    assert bridge.calls == len(transport.generation_bodies) == 1
    assert len(transport.replies) == 1


def test_received_incomplete_keeps_actual_usage_and_never_retries():
    transport = ScriptedResponses(incomplete=True)
    client = budgeted(transport)
    bridge = LongBenchBudgetBridge(client)
    with pytest.raises(OpenAIResponseNotCompletedError):
        bridge.complete_answer('Question', max_tokens=64, enable_thinking=False)
    entry = client.ledger[0]
    assert bridge.calls == len(transport.generation_bodies) == 1
    assert entry['usage_known'] is True and entry['charged_tokens'] == entry['usage']['total_tokens']
    assert entry['provider_metadata']['incomplete_reason'] == 'max_output_tokens'


def test_uncertain_delivery_keeps_full_reservation_and_no_retry():
    transport = ScriptedResponses(delivery_error=True)
    client = budgeted(transport)
    with pytest.raises(OpenAIResponsesTransportError):
        LongBenchBudgetBridge(client).complete_answer('Question', max_tokens=64, enable_thinking=False)
    entry = client.ledger[0]
    assert len(transport.generation_bodies) == 1
    assert not entry.get('usage_known')
    assert entry['charged_tokens'] == entry['admission_prompt_tokens'] + 64


def test_exact_routing_includes_notice_margin_and_propagates_count_failure(monkeypatch):
    transport = ScriptedResponses()
    client = budgeted(transport)
    estimator = ExactBudgetEstimator(client)
    estimate = estimator.estimate('A short prompt', scope='domain')
    assert estimate.tokens == client.count_messages([{'role': 'user', 'content': 'A short prompt'}])
    assert estimate.method == 'openai_responses_exact_admission' and estimate.scope == 'domain'
    assert 'Experiment resource limits:' in transport.count_bodies[0]['input'][0]['content']
    def unavailable(**kwargs):
        raise BudgetIntegrityError('Scripted counting outage')
    monkeypatch.setattr(transport, 'count', unavailable)
    with pytest.raises(BudgetIntegrityError, match='counting outage'):
        estimator.estimate('Cannot use a character fallback')
    assert not transport.generation_bodies


@pytest.mark.parametrize('kwargs', [
    {'enable_thinking': True}, {'enable_thinking': False, 'reasoning_effort': 'high'},
    {'enable_thinking': False, 'max_tokens': 8},
])
def test_incompatible_legacy_requests_fail_before_generation(kwargs):
    transport = ScriptedResponses()
    bridge = LongBenchBudgetBridge(budgeted(transport))
    with pytest.raises(ValueError):
        bridge.complete('Question', **({'max_tokens': 64} | kwargs))
    assert not transport.generation_bodies and bridge.calls == 0


def test_wrong_actual_model_blocks_subsequent_controller_calls():
    transport = ScriptedResponses(actual_model='gpt-6-sol')
    bridge = LongBenchBudgetBridge(budgeted(transport))
    with pytest.raises(BudgetIntegrityError, match='exact model ID'):
        bridge.complete_answer('Question', max_tokens=64, enable_thinking=False)
    with pytest.raises(BudgetIntegrityError, match='previous response'):
        bridge.complete_answer('Another candidate', max_tokens=64, enable_thinking=False)
    assert len(transport.generation_bodies) == bridge.calls == 1


def test_real_existing_controller_runs_raw_source_votes_verifiers_and_cost_gate(tmp_path):
    from scripts.benchmark_openai_transfer import CostCappedClient, MonetaryLedger
    transport = ScriptedResponses(['The correct answer is (A)']*3 + [UNKNOWN_OPTIONS]*3)
    client = budgeted(transport)
    money = MonetaryLedger(tmp_path/'cost.json', cap=15,
        prices={'gpt-6.1-terra': {'input': 1, 'output': 5, 'r_max': 5}}, freeze_sha='0'*64)
    guarded = CostCappedClient(client, money, 'fixture-longbench')
    result = run_longbench_harness(long_row(), guarded)
    assert result['mode'] == 'raw_source' and result['prediction'] == 'A'
    assert not result['reused_direct'] and 'correct' not in result and result['label_free']
    assert result['authoritative_physical_calls'] == result['model_calls'] == 6
    assert len(client.ledger) == len(transport.generation_bodies) == len(money.entries) == 6
    assert result['shared_usage']['within_budget'] and result['shared_usage']['all_usage_known']
    assert all(row['usage_known'] for row in money.entries)
    assert all(entry['admission_prompt_tokens']+entry['requested_output_tokens'] <= 16_384 for entry in client.ledger)
    assert result['option_verification_retries'] == 0


def test_explicit_controller_json_recovery_uses_shared_ledger():
    replies = ['The correct answer is (A)']*3 + ['Not JSON', UNKNOWN_OPTIONS, UNKNOWN_OPTIONS, UNKNOWN_OPTIONS]
    client = budgeted(ScriptedResponses(replies))
    result = run_longbench_harness(long_row(), client)
    assert result['prediction'] == 'A' and result['option_verification_retries'] == 1
    assert result['authoritative_physical_calls'] == result['model_calls'] == 7
    assert len(client.ledger) == 7 and client.summary()['within_budget']


def test_oversized_evidence_never_reaches_generation():
    transport = ScriptedResponses()
    client = budgeted(transport)
    groups = [{'candidate_index': 0, 'query_mode': 'question', 'text': 'X'*100_000}]
    result = run_longbench_harness(long_row(), client, groups=groups)
    assert result['prediction'] is None
    assert result['candidate_errors'] and result['option_verification_errors']
    assert result['authoritative_physical_calls'] == 0 and not transport.generation_bodies


def test_label_or_false_extension_refused_before_generation():
    transport = ScriptedResponses()
    client = budgeted(transport)
    with pytest.raises(ValueError, match='label-free'):
        run_longbench_harness(long_row(answer='A'), client)
    assert not transport.count_bodies
    with pytest.raises(ValueError, match='Full prompt fits'):
        run_longbench_harness(long_row(context='The flag is red.'), client)
    assert not transport.generation_bodies


def test_exact_call_and_cumulative_limits_remain_authoritative():
    transport = ScriptedResponses(['The correct answer is (A)']*3)
    client = budgeted(transport, max_calls=1)
    result = run_longbench_harness(long_row(), client,
        config=LongBenchEvidenceConfig(option_verify=False))
    assert len(client.ledger) == len(transport.generation_bodies) == 1
    assert result['authoritative_physical_calls'] == 1
    assert result['candidate_errors'] and result['shared_usage']['within_budget']
