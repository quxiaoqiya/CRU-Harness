"""Text-only Responses API bridge for the frozen V3.2 budget client.

Exact admission: https://developers.openai.com/api/docs/guides/token-counting
Wire schema: https://developers.openai.com/api/reference/resources/responses/methods/create

No SDK retries, model substitution, tools, server-side conversation, or implicit
truncation. Credentials are supplied by the caller and never included in audit
records. ``count(text=...)`` counts a Responses text input, including its format;
it is not the earlier Qwen raw-text tokenizer measurement.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.parse import urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from .experiment_budget import BudgetedClient, BudgetIntegrityError


class OpenAIResponsesTransportError(RuntimeError):
    """A sanitized HTTP/transport failure; delivery or usage may be unknown."""


class OpenAIResponseNotCompletedError(RuntimeError):
    """A received response cannot be accepted as a complete text answer."""

    def __init__(self, *, usage: dict[str, Any], metadata: dict[str, Any]):
        self.usage = usage
        self.metadata = metadata
        # No provider error message or partial source-derived text in errors.
        super().__init__('Responses generation was not a completed text answer')


class _NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # A redirect must not forward a bearer credential to another endpoint.
        return None


def _integer(value: Any) -> bool:
    return type(value) is int and value >= 0


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def _identity(value: Any):
    return value if isinstance(value, str) and re.fullmatch(r'[A-Za-z0-9_.:/-]{1,200}', value) else None


class OpenAIResponsesTransport:
    """Synchronous text-only transport with exact input counts and no retries.

    Legacy seed, temperature, and Qwen template settings are not sent. Their
    omission is explicit in response metadata. Configure reasoning effort on
    this transport, consistently for counting and generation.
    """

    def __init__(self, *, api_key: str, model: str,
                 base_url: str = 'https://api.openai.com/v1',
                 reasoning_effort: str = 'none', timeout: float = 180,
                 opener=None):
        parsed = urlsplit(base_url)
        if (parsed.scheme not in {'https', 'http'} or not parsed.netloc
                or parsed.username or parsed.password or parsed.query or parsed.fragment):
            raise ValueError('A plain HTTP(S) API base URL is required')
        if not isinstance(api_key, str) or not api_key:
            raise ValueError('An API credential is required')
        if not isinstance(model, str) or not model.strip():
            raise ValueError('An explicit model ID is required')
        if reasoning_effort not in {'none', 'minimal', 'low', 'medium', 'high', 'xhigh', 'max'}:
            raise ValueError('Unsupported reasoning effort')
        self.base_url = base_url.rstrip('/')
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self._api_key = api_key
        self.opener = opener if opener is not None else build_opener(_NoRedirect())
        self._counts: dict[str, int] = {}
        self._lock = threading.RLock()
        self.count_audit: list[dict[str, Any]] = []
        self.response_audit: list[dict[str, Any]] = []

    def configuration(self) -> dict[str, Any]:
        return {'api': 'responses', 'model': self.model,
                'reasoning_effort': self.reasoning_effort,
                'store': False, 'truncation': 'disabled', 'service_tier': 'default', 'implicit_retries': 0,
                'tools_supported': False,
                'text_count_semantics': 'complete Responses text input, including input formatting'}

    def _post(self, path: str, body: dict[str, Any]) -> dict[str, Any]:
        request = Request(
            self.base_url + path, method='POST',
            data=json.dumps(body, ensure_ascii=False).encode(),
            headers={'Authorization': 'Bearer ' + self._api_key,
                     'Content-Type': 'application/json'})
        try:
            with self.opener.open(request, timeout=self.timeout) as response:
                payload = json.load(response)
        except HTTPError as exc:
            # Only known error codes may leave the response body. Never retain
            # arbitrary provider messages, URLs, reasons, headers or echo data.
            safe_code = 'unclassified'
            try:
                candidate = json.loads(exc.read(65536)).get('error', {}).get('code')
                if candidate in {'invalid_api_key', 'model_not_found', 'insufficient_quota',
                                 'rate_limit_exceeded', 'context_length_exceeded',
                                 'invalid_request_error', 'unsupported_parameter',
                                 'unsupported_value', 'server_error'}:
                    safe_code = candidate
            except Exception:
                pass
            raise OpenAIResponsesTransportError(
                f'OpenAI HTTP request failed (status {exc.code}; code {safe_code})') from None
        except Exception:
            raise OpenAIResponsesTransportError('OpenAI request failed; response or usage may be unknown') from None
        if not isinstance(payload, dict):
            raise BudgetIntegrityError('OpenAI response was not a JSON object')
        return payload

    @staticmethod
    def _no_tools(tools, tool_choice):
        if tools is not None or tool_choice is not None:
            raise NotImplementedError('This Responses adapter supports text-only Harness calls, not tools')

    @staticmethod
    def _input(*, messages=None, text=None):
        if messages is not None and text is not None:
            raise ValueError('Specify messages or text, not both')
        if messages is None:
            if text is not None and not isinstance(text, str):
                raise TypeError('Text input must be a string')
            return text or ''
        if not isinstance(messages, (list, tuple)):
            raise TypeError('Messages must be a sequence of text messages')
        result = []
        for message in messages:
            if (not isinstance(message, dict)
                    or set(message) - {'role', 'content'}
                    or message.get('role') not in {'system', 'developer', 'user', 'assistant'}
                    or not isinstance(message.get('content'), str)):
                raise NotImplementedError('Only role/content text messages are supported')
            result.append({'role': message['role'], 'content': message['content']})
        return result

    def _common(self, *, messages=None, text=None):
        return {'model': self.model, 'input': self._input(messages=messages, text=text),
                'reasoning': {'effort': self.reasoning_effort}, 'truncation': 'disabled'}

    def count(self, *, messages=None, text=None, tools=None, tool_choice=None) -> int:
        self._no_tools(tools, tool_choice)
        body = self._common(messages=messages, text=text)
        digest = _digest(body)
        with self._lock:
            started = time.perf_counter()
            entry = {'input_sha256': digest, 'model': self.model,
                     'cache_hit': digest in self._counts, 'status': 'inflight'}
            self.count_audit.append(entry)
            try:
                if digest not in self._counts:
                    payload = self._post('/responses/input_tokens', body)
                    count = payload.get('input_tokens')
                    if (payload.get('object') != 'response.input_tokens' or not _integer(count)):
                        raise BudgetIntegrityError('OpenAI counting endpoint did not return an exact integer input count')
                    self._counts[digest] = count
                entry.update(status='ok', input_tokens=self._counts[digest])
                return self._counts[digest]
            except Exception as exc:
                entry.update(status='error', error_type=type(exc).__name__)
                raise
            finally:
                entry['seconds'] = round(time.perf_counter() - started, 4)

    def request(self, path: str, body: dict[str, Any] | None = None, *, outside_v1=False):
        if path != '/chat/completions' or outside_v1:
            raise NotImplementedError('Only the budget-client chat interface is supported')
        if not isinstance(body, dict):
            raise TypeError('A chat request body is required')
        self._no_tools(body.get('tools'), body.get('tool_choice'))
        allowed = {'model', 'messages', 'max_tokens', 'temperature', 'seed',
                   'chat_template_kwargs', 'tools', 'tool_choice'}
        if set(body) - allowed:
            raise NotImplementedError('Unsupported legacy chat request parameters')
        if body.get('model', self.model) != self.model:
            raise ValueError('Request model must match the configured transport model')
        output_limit = body.get('max_tokens')
        if not _integer(output_limit) or output_limit < 16:
            raise ValueError('Responses output token limit must be an integer of at least 16')
        if 'messages' not in body:
            raise ValueError('Chat messages are required')
        request_body = self._common(messages=body['messages'])
        request_body.update(store=False, service_tier='default', max_output_tokens=output_limit)
        metadata = {**self.configuration(), 'request_sha256': _digest(request_body),
                    'requested_output_tokens': output_limit,
                    'mapped_legacy_parameters': {'max_tokens': 'max_output_tokens'},
                    'omitted_legacy_parameters': sorted(
                        name for name in ('temperature', 'seed', 'chat_template_kwargs') if name in body)}
        started = time.perf_counter()
        with self._lock:
            self.response_audit.append(metadata)
        try:
            payload = self._post('/responses', request_body)
            return self._normalize(payload, metadata)
        except Exception as exc:
            metadata['error_type'] = type(exc).__name__
            raise
        finally:
            metadata['seconds'] = round(time.perf_counter() - started, 4)

    def _normalize(self, payload: dict[str, Any], metadata: dict[str, Any]):
        raw_usage = payload.get('usage')
        if not isinstance(raw_usage, dict):
            raise BudgetIntegrityError('OpenAI generation did not return exact usage')
        prompt, completion = raw_usage.get('input_tokens'), raw_usage.get('output_tokens')
        if not _integer(prompt) or not _integer(completion):
            raise BudgetIntegrityError('OpenAI generation did not return exact integer usage')
        if ('total_tokens' in raw_usage
                and (not _integer(raw_usage['total_tokens']) or raw_usage['total_tokens'] != prompt + completion)):
            raise BudgetIntegrityError('OpenAI total usage was inconsistent')
        # Only documented numeric usage fields enter the audit; never copy an
        # arbitrary provider JSON object that could echo input or credentials.
        provider_usage = {'input_tokens': prompt, 'output_tokens': completion}
        if 'total_tokens' in raw_usage:
            provider_usage['total_tokens'] = raw_usage['total_tokens']
        for field, counter, ceiling in (('input_tokens_details', 'cached_tokens', prompt),
                                        ('output_tokens_details', 'reasoning_tokens', completion)):
            if field in raw_usage:
                details = raw_usage[field]
                if (not isinstance(details, dict) or counter not in details
                        or not _integer(details[counter]) or details[counter] > ceiling):
                    raise BudgetIntegrityError('OpenAI detailed usage was invalid')
                provider_usage[field] = {counter: details[counter]}
                if field == 'input_tokens_details' and 'cache_write_tokens' in details:
                    written = details['cache_write_tokens']
                    if not _integer(written) or written + details[counter] > ceiling:
                        raise BudgetIntegrityError('OpenAI cache-write usage was invalid')
                    provider_usage[field]['cache_write_tokens'] = written
                    if 'cache_write_tokens_details' in details:
                        writes = details['cache_write_tokens_details']
                        lifetimes = ('ttl_30m_tokens', 'ttl_12h_tokens')
                        if (not isinstance(writes, dict)
                                or any(not _integer(writes.get(name)) for name in lifetimes)
                                or sum(writes[name] for name in lifetimes) != written):
                            raise BudgetIntegrityError('OpenAI cache-write lifetime usage was invalid')
                        provider_usage[field]['cache_write_tokens_details'] = {
                            name: writes[name] for name in lifetimes}
        status = payload.get('status')
        metadata.update(response_id=_identity(payload.get('id')), actual_model=_identity(payload.get('model')),
                        response_status=status if isinstance(status, str) and status in {'completed', 'incomplete', 'failed', 'cancelled', 'queued', 'in_progress'} else 'unknown',
                        provider_usage=provider_usage)
        tier = payload.get('service_tier')
        metadata['returned_service_tier'] = tier if isinstance(tier, str) and tier in {'auto', 'default', 'flex', 'priority', 'scale'} else ('unknown' if tier is not None else None)
        details = payload.get('incomplete_details') or {}
        reason = details.get('reason') if isinstance(details, dict) else None
        metadata['incomplete_reason'] = reason if isinstance(reason, str) and reason in {'max_output_tokens', 'content_filter'} else ('other' if reason else None)
        usage = {'prompt_tokens': prompt, 'completion_tokens': completion,
                 'total_tokens': prompt + completion,
                 'openai_responses_metadata': metadata}
        if payload.get('status') != 'completed':
            raise OpenAIResponseNotCompletedError(usage=usage, metadata=metadata)
        if tier is not None and tier != 'default':
            metadata['service_tier_mismatch'] = True
            raise OpenAIResponseNotCompletedError(usage=usage, metadata=metadata)
        if not metadata['response_id'] or not metadata['actual_model']:
            metadata['invalid_response_identity'] = True
            raise OpenAIResponseNotCompletedError(usage=usage, metadata=metadata)
        output = payload.get('output')
        if not isinstance(output, list):
            metadata['invalid_output_items'] = True
            raise OpenAIResponseNotCompletedError(usage=usage, metadata=metadata)
        texts, refusals = [], []
        for item in output:
            if not isinstance(item, dict):
                metadata['invalid_output_items'] = True
                raise OpenAIResponseNotCompletedError(usage=usage, metadata=metadata)
            if item.get('type') == 'reasoning':
                continue
            if (item.get('type') != 'message' or item.get('role') != 'assistant'
                    or item.get('status') != 'completed' or not isinstance(item.get('content'), list)):
                metadata['unsupported_or_incomplete_output'] = True
                raise OpenAIResponseNotCompletedError(usage=usage, metadata=metadata)
            for part in item['content']:
                if isinstance(part, dict) and part.get('type') == 'output_text' and isinstance(part.get('text'), str):
                    texts.append(part['text'])
                elif isinstance(part, dict) and part.get('type') == 'refusal' and isinstance(part.get('refusal'), str):
                    refusals.append(part['refusal'])
                else:
                    metadata['unsupported_or_incomplete_output'] = True
                    raise OpenAIResponseNotCompletedError(usage=usage, metadata=metadata)
        metadata['refusal_present'] = bool(refusals)
        message = {'role': 'assistant', 'content': ''.join(texts),
                   'openai_response': metadata}
        if refusals:
            message['refusal'] = ''.join(refusals)
        return {'choices': [{'message': message, 'finish_reason': 'refusal' if refusals else 'stop'}],
                'usage': usage, 'openai_response': metadata}


class OpenAIResponsesBudgetedClient(BudgetedClient):
    """Keep received failure usage without changing frozen admission logic.

    A known incomplete response is charged at its actual input/output usage and
    still raises. An uncertain HTTP/delivery failure keeps the full reservation
    established by the original client. No request is retried.
    """

    def _complete(self, messages, *, max_tokens=2048, tools=None, tool_choice=None):
        with self._lock:
            try:
                message = super()._complete(messages, max_tokens=max_tokens,
                                            tools=tools, tool_choice=tool_choice)
            except OpenAIResponseNotCompletedError as exc:
                entry = self.ledger[-1]
                usage = exc.usage
                prompt, completion = usage['prompt_tokens'], usage['completion_tokens']
                entry.update(usage=usage, usage_known=True, charged_tokens=prompt + completion,
                             provider_metadata=exc.metadata)
                if (prompt > entry['admission_prompt_tokens']
                        or completion > entry['requested_output_tokens']
                        or sum(row['charged_tokens'] for row in self.ledger) > self.budget.total_tokens):
                    entry.update(error_type='BudgetIntegrityError', error='Provider usage exceeded the admitted reservation')
                    if self.on_ledger_update:
                        self.on_ledger_update(self.ledger)
                    raise BudgetIntegrityError('Provider usage exceeded the admitted reservation') from None
                if self.on_ledger_update:
                    self.on_ledger_update(self.ledger)
                raise
            self.ledger[-1]['provider_metadata'] = message['openai_response']
            if self.on_ledger_update:
                self.on_ledger_update(self.ledger)
            return message
