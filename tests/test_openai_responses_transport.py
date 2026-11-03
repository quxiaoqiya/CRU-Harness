"""Responses integration invariants, using scripted HTTP only."""
from __future__ import annotations

import copy
import io
import json
from urllib.error import HTTPError, URLError

import pytest

from long_context_agent.experiment_budget import BudgetExhausted, BudgetIntegrityError, ExperimentBudget
from long_context_agent.openai_responses_transport import (
    OpenAIResponseNotCompletedError, OpenAIResponsesBudgetedClient,
    OpenAIResponsesTransport, OpenAIResponsesTransportError,
)


class MockHTTP:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.calls = []

    def open(self, request, *, timeout):
        self.calls.append({'url': request.full_url, 'method': request.get_method(),
                           'body': json.loads(request.data), 'timeout': timeout})
        result = self.responses.pop(0)
        if isinstance(result, Exception):
            raise result
        return io.BytesIO(json.dumps(result).encode())


def counted(n=10):
    return {'object': 'response.input_tokens', 'input_tokens': n}


def generated(*, status='completed', text='Answer: 200', input_tokens=10,
              output_tokens=6, reasoning_tokens=2):
    return {'id': 'resp_example', 'model': 'gpt-test-resolved', 'status': status, 'service_tier': 'default',
            'incomplete_details': {'reason': 'max_output_tokens'} if status == 'incomplete' else None,
            'usage': {'input_tokens': input_tokens, 'output_tokens': output_tokens,
                      'total_tokens': input_tokens + output_tokens,
                      'input_tokens_details': {'cached_tokens': 3},
                      'output_tokens_details': {'reasoning_tokens': reasoning_tokens}},
            'output': [{'type': 'reasoning', 'id': 'reasoning_example',
                        'summary': [{'type': 'summary_text', 'text': 'Do not expose this.'}]},
                       {'type': 'message', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': text}]}]}


def transport(http, **kwargs):
    return OpenAIResponsesTransport(api_key='unit-test-credential-canary',
                                    model='gpt-test', opener=http, **kwargs)


def legacy_body(**kwargs):
    return {'model': 'gpt-test', 'messages': [{'role': 'user', 'content': 'Question'}],
            'max_tokens': 32, 'temperature': 0.0, 'seed': 20261101,
            'chat_template_kwargs': {'enable_thinking': False}, **kwargs}


def test_exact_count_cache_and_count_audit_do_not_retain_source_or_credentials():
    http = MockHTTP(counted(17), counted(18))
    t = transport(http)
    assert t.count(text='private source canary') == 17
    assert t.count(text='private source canary') == 17
    assert t.count(text='another source') == 18
    assert len(http.calls) == 2
    assert http.calls[0]['url'] == 'https://api.openai.com/v1/responses/input_tokens'
    assert http.calls[0]['method'] == 'POST'
    assert http.calls[0]['body'] == {'model': 'gpt-test', 'input': 'private source canary',
                                     'reasoning': {'effort': 'none'}, 'truncation': 'disabled'}
    assert [row['cache_hit'] for row in t.count_audit] == [False, True, False]
    audit = json.dumps(t.count_audit)
    assert 'private source canary' not in audit and 'unit-test-credential-canary' not in audit


def test_count_and_generation_use_identical_input_and_reasoning_without_legacy_options():
    http = MockHTTP(counted(), generated())
    t = transport(http, reasoning_effort='low')
    body = legacy_body()
    original = copy.deepcopy(body)
    t.count(messages=body['messages'])
    result = t.request('/chat/completions', body)
    count_body, generation_body = [row['body'] for row in http.calls]
    assert generation_body == {**count_body, 'store': False, 'service_tier': 'default', 'max_output_tokens': 32}
    assert count_body['reasoning'] == {'effort': 'low'}
    assert http.calls[1]['url'].endswith('/v1/responses')
    assert not {'seed', 'temperature', 'chat_template_kwargs'} & generation_body.keys()
    assert result['openai_response']['omitted_legacy_parameters'] == ['chat_template_kwargs', 'seed', 'temperature']
    assert body == original


def test_usage_includes_reasoning_once_and_preserves_actual_response_metadata():
    http = MockHTTP(counted(), generated())
    t = transport(http)
    client = OpenAIResponsesBudgetedClient(t, ExperimentBudget(max_calls=1))
    assert client.complete([{'role': 'user', 'content': 'Question'}], max_tokens=32) == 'Answer: 200'
    row = client.ledger[0]
    assert row['usage_known'] and row['charged_tokens'] == 16
    assert row['usage']['prompt_tokens'] == 10
    assert row['usage']['completion_tokens'] == 6
    meta = row['provider_metadata']
    assert meta['actual_model'] == 'gpt-test-resolved' and meta['response_id'] == 'resp_example'
    assert meta['response_status'] == 'completed'
    assert meta['returned_service_tier'] == 'default'
    assert meta['provider_usage']['output_tokens_details']['reasoning_tokens'] == 2
    assert meta == row['usage']['openai_responses_metadata']
    assert 'Do not expose this.' not in json.dumps(t.response_audit)
    assert json.loads(json.dumps(client.ledger))[0]['charged_tokens'] == 16
    calls = len(http.calls)
    with pytest.raises(BudgetExhausted):
        client.complete([], max_tokens=32)
    assert len(http.calls) == calls


@pytest.mark.parametrize('count', [True, -1, 10.5, '10', None])
def test_nonexact_count_fails_without_fallback(count):
    http = MockHTTP(counted(count))
    t = transport(http)
    with pytest.raises(BudgetIntegrityError):
        t.count(text='source')
    assert len(http.calls) == 1 and t.count_audit[0]['status'] == 'error'


@pytest.mark.parametrize('options', [{'tools': []}, {'tools': [{'type': 'function'}]},
                                     {'tool_choice': 'none'}, {'tool_choice': 'auto'}])
def test_tools_are_rejected_in_count_and_generation_before_http(options):
    http = MockHTTP()
    t = transport(http)
    with pytest.raises(NotImplementedError):
        t.count(messages=[], **options)
    with pytest.raises(NotImplementedError):
        t.request('/chat/completions', legacy_body(**options))
    assert http.calls == []


@pytest.mark.parametrize('messages', [
    [{'role': 'tool', 'content': 'result'}],
    [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': 'private'}}]}],
    [{'role': 'assistant', 'content': '', 'tool_calls': []}],
])
def test_unsupported_message_semantics_are_not_silently_dropped(messages):
    http = MockHTTP()
    t = transport(http)
    with pytest.raises(NotImplementedError):
        t.count(messages=messages)
    assert http.calls == []


@pytest.mark.parametrize('status', ['incomplete', 'failed', 'cancelled', 'in_progress'])
def test_received_noncompleted_response_is_an_error_with_actual_billing(status):
    http = MockHTTP(counted(), generated(status=status, text='Answer: partial'))
    t = transport(http)
    snapshots = []
    client = OpenAIResponsesBudgetedClient(t, ExperimentBudget(max_calls=1),
        on_ledger_update=lambda rows: snapshots.append(copy.deepcopy(rows)))
    with pytest.raises(OpenAIResponseNotCompletedError) as error:
        client.complete([], max_tokens=32)
    assert 'partial' not in str(error.value)
    row = client.ledger[0]
    assert row['status'] == 'error' and row['usage_known'] and row['charged_tokens'] == 16
    assert row['provider_metadata']['response_status'] == status
    assert 'content' not in row
    assert client.summary()['all_usage_known'] and client.summary()['within_budget']
    assert snapshots[-1][0]['charged_tokens'] == 16
    assert len(http.calls) == 2
    assert 'Answer: partial' not in json.dumps(t.response_audit)


def test_incomplete_usage_exceeding_reservation_remains_integrity_failure():
    http = MockHTTP(counted(), generated(status='incomplete', output_tokens=100))
    client = OpenAIResponsesBudgetedClient(transport(http), ExperimentBudget())
    with pytest.raises(BudgetIntegrityError):
        client.complete([], max_tokens=32)
    assert client.ledger[0]['charged_tokens'] == 110
    assert client.ledger[0]['error_type'] == 'BudgetIntegrityError'
    assert not client.summary()['within_budget']


def test_uncertain_generation_failure_keeps_reservation_and_never_retries_or_leaks():
    http = MockHTTP(counted(), URLError('unit-test-credential-canary private request body'))
    t = transport(http)
    client = OpenAIResponsesBudgetedClient(t, ExperimentBudget(max_calls=1))
    with pytest.raises(OpenAIResponsesTransportError):
        client.complete([], max_tokens=32)
    row = client.ledger[0]
    assert row['charged_tokens'] == 10 + 16 + 32  # original safety margin and reservation
    assert row.get('usage_known') is not True
    assert len(http.calls) == 2
    audit = json.dumps([client.ledger, t.count_audit, t.response_audit])
    assert 'unit-test-credential-canary' not in audit and 'private request body' not in audit


def test_http_error_body_and_reason_are_never_exposed():
    secret = 'unit-test-credential-canary'
    error = HTTPError('https://example.invalid/' + secret, 429, secret, {},
                      io.BytesIO(json.dumps({'error': {'message': secret}}).encode()))
    http = MockHTTP(error)
    t = transport(http)
    with pytest.raises(OpenAIResponsesTransportError) as caught:
        t.count(text='source')
    assert str(caught.value) == 'OpenAI HTTP request failed (status 429; code unclassified)'
    assert secret not in str(caught.value)
    assert len(http.calls) == 1


@pytest.mark.parametrize('code, expected', [
    ('rate_limit_exceeded', 'rate_limit_exceeded'),
    ('unit-test-credential-canary', 'unclassified'),
])
def test_http_error_codes_are_allowlisted_without_echoing_provider_messages(code, expected):
    error = HTTPError('https://example.invalid', 429, 'private reason', {},
                      io.BytesIO(json.dumps({'error': {'code': code,
                                                       'message': 'unit-test-credential-canary'}}).encode()))
    http = MockHTTP(error)
    t = transport(http)
    with pytest.raises(OpenAIResponsesTransportError) as caught:
        t.count(text='source')
    assert f'code {expected}' in str(caught.value)
    assert 'unit-test-credential-canary' not in str(caught.value)
    assert len(http.calls) == 1


def test_unknown_usage_fields_are_not_copied_into_audits():
    response = generated()
    response['usage']['unexpected_echo'] = 'private source canary'
    response['usage']['input_tokens_details']['unexpected_echo'] = 'unit-test-credential-canary'
    http = MockHTTP(response)
    t = transport(http)
    result = t.request('/chat/completions', legacy_body())
    audit = json.dumps([result['usage'], t.response_audit])
    assert 'private source canary' not in audit and 'unit-test-credential-canary' not in audit


def test_cache_write_lifetimes_are_preserved_without_double_counting_tokens():
    response = generated()
    response['usage']['input_tokens_details'].update(
        cache_write_tokens=6,
        cache_write_tokens_details={'ttl_30m_tokens': 2, 'ttl_12h_tokens': 4})
    http = MockHTTP(counted(), response)
    client = OpenAIResponsesBudgetedClient(transport(http), ExperimentBudget())
    client.complete([], max_tokens=32)
    row = client.ledger[0]
    assert row['charged_tokens'] == 16
    details = row['provider_metadata']['provider_usage']['input_tokens_details']
    assert details == {'cached_tokens': 3, 'cache_write_tokens': 6,
                       'cache_write_tokens_details': {'ttl_30m_tokens': 2, 'ttl_12h_tokens': 4}}


def test_different_returned_tier_is_explicitly_rejected_with_known_usage():
    response = generated()
    response['service_tier'] = 'priority'
    http = MockHTTP(counted(), response)
    client = OpenAIResponsesBudgetedClient(transport(http), ExperimentBudget())
    with pytest.raises(OpenAIResponseNotCompletedError):
        client.complete([], max_tokens=32)
    assert client.ledger[0]['charged_tokens'] == 16
    assert client.ledger[0]['provider_metadata']['service_tier_mismatch']


@pytest.mark.parametrize('usage', [None, {'input_tokens': True, 'output_tokens': 1},
                                   {'input_tokens': 10, 'output_tokens': 6, 'total_tokens': 999}])
def test_invalid_generation_usage_is_fail_closed(usage):
    response = generated()
    response['usage'] = usage
    http = MockHTTP(counted(), response)
    client = OpenAIResponsesBudgetedClient(transport(http), ExperimentBudget())
    with pytest.raises(BudgetIntegrityError):
        client.complete([], max_tokens=32)
    assert client.ledger[0]['charged_tokens'] == 58
    assert not client.summary()['all_usage_known']


def test_unexpected_tool_output_is_not_accepted_and_keeps_known_usage():
    response = generated()
    response['output'] = [{'type': 'function_call', 'arguments': 'private source canary'}]
    http = MockHTTP(counted(), response)
    client = OpenAIResponsesBudgetedClient(transport(http), ExperimentBudget())
    with pytest.raises(OpenAIResponseNotCompletedError):
        client.complete([], max_tokens=32)
    assert client.ledger[0]['usage_known'] and client.ledger[0]['charged_tokens'] == 16
    assert client.ledger[0]['provider_metadata']['unsupported_or_incomplete_output']


def test_refusal_is_explicit_and_never_replaced_by_reasoning_text():
    response = generated()
    response['output'][1]['content'] = [{'type': 'refusal', 'refusal': 'Cannot answer.'}]
    http = MockHTTP(response)
    t = transport(http)
    result = t.request('/chat/completions', legacy_body())
    assert result['choices'][0]['message']['content'] == ''
    assert result['choices'][0]['message']['refusal'] == 'Cannot answer.'
    assert result['choices'][0]['finish_reason'] == 'refusal'
    assert result['openai_response']['refusal_present']


@pytest.mark.parametrize('body', [legacy_body(model='different'), legacy_body(max_tokens=15),
                                 legacy_body(max_tokens=True), legacy_body(stream=True)])
def test_unsupported_requests_are_rejected_without_generation(body):
    http = MockHTTP()
    t = transport(http)
    with pytest.raises((ValueError, NotImplementedError)):
        t.request('/chat/completions', body)
    assert http.calls == []


def test_real_v32_algorithm_runs_through_scripted_responses_http():
    """Exercise planner -> labels -> provenance -> aggregation, not model quality."""
    from long_context_agent import AdaptiveV3Config, LongContextHarness

    class HarnessHTTP:
        def __init__(self):
            self.generation_calls = 0

        @staticmethod
        def scripted_count(value):
            # Deliberately simulated HTTP tokenizer, never a real token claim.
            if isinstance(value, str):
                return max(1, (len(value) + 3) // 4)
            return 5 + sum(4 + max(1, (len(m['content']) + 3) // 4) for m in value)

        def open(self, request, *, timeout):
            body = json.loads(request.data)
            if request.full_url.endswith('/responses/input_tokens'):
                return io.BytesIO(json.dumps(counted(self.scripted_count(body['input']))).encode())
            assert request.full_url.endswith('/responses')
            assert body['reasoning'] == {'effort': 'none'}
            assert body['service_tier'] == 'default' and body['store'] is False
            messages = body['input']
            if 'Plan evidence acquisition' in messages[0]['content']:
                reply = {'kind': 'records', 'operation': 'count', 'filters': {}, 'field': 'label',
                         'labels': ['positive', 'negative'], 'target': 'positive',
                         'answer_template': 'Answer: {value}'}
            else:
                assert 'Classify or extract' in messages[0]['content']
                rows = json.loads(messages[-1]['content'].split('\nRECORDS:\n', 1)[1])
                reply = {'records': [[row['id'], 0 if 'semantic=positive;' in row['text'] else 1]
                                     for row in rows]}
            self.generation_calls += 1
            content = json.dumps(reply)
            response = generated(text=content, input_tokens=self.scripted_count(messages),
                                 output_tokens=self.scripted_count(content), reasoning_tokens=0)
            response['id'] = f'resp_scripted_{self.generation_calls}'
            return io.BytesIO(json.dumps(response).encode())

    http = HarnessHTTP()
    t = transport(http)
    budget = ExperimentBudget(context_tokens=4096, total_tokens=16384, max_calls=8,
                              max_output_tokens=1024)
    client = OpenAIResponsesBudgetedClient(t, budget)
    harness = LongContextHarness(None, max_context_tokens=4096, reserve_tokens=1024)
    context = 'The following lines contain 300 records, one per line.\n' + ''.join(
        'Date: 2024-01-01 || User: 7 || Instance: '
        f"semantic={'positive' if i < 200 else 'negative'}; shared payload\n" for i in range(300))
    result = harness.answer_with_structured_facts(
        {'id': 'scripted-responses-integration', 'context': context, 'question': 'Count positive records.'},
        client, config=AdaptiveV3Config(planner_output_tokens=512,
                                       classification_output_tokens=1024, final_output_tokens=256))
    assert result['mode'] == 'adaptive_fact_aggregation_v3'
    assert result['answer'] == 'Answer: 200' and result['evidence_complete']
    assert result['semantic_coverage']['known_physical_records'] == 300
    assert result['semantic_coverage']['transferred_records'] == 298
    assert result['final_fact_loss_count'] == 0
    assert http.generation_calls == client.summary()['physical_requests'] == 2
    assert client.summary()['all_usage_known'] and client.summary()['within_budget']
    assert all(row['provider_metadata']['returned_service_tier'] == 'default' for row in client.ledger)
    json.dumps([t.count_audit, t.response_audit, client.ledger])
