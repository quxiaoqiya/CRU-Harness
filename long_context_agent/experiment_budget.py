"""One physical-request/token ledger shared by held-out context methods.

No implicit retries, model fallbacks, or unmetered recursive calls are allowed.
Admission uses the serving model's /tokenize endpoint, including its chat
template, and reserves the requested output before sending a generation.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import time
from typing import Any
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener


class BudgetExhausted(RuntimeError):
    pass


class BudgetIntegrityError(RuntimeError):
    pass


def load_provider_env(path: str | Path | None = None) -> dict[str, str]:
    """Read only the configured provider credential; never serialize it."""
    values: dict[str, str] = {}
    if path:
        for line in Path(path).read_text().splitlines():
            if not line.strip() or line.lstrip().startswith('#') or '=' not in line:
                continue
            key, value = line.removeprefix('export ').split('=', 1)
            values[key.strip()] = value.strip().strip("\"'")
    key = next((os.environ.get(k) or values.get(k) for k in
                ('Q38_API_KEY', 'QWEN38_FLASH_API_KEY')
                if os.environ.get(k) or values.get(k)), '')
    if not key:
        raise RuntimeError('Q38_API_KEY or QWEN38_FLASH_API_KEY is required')
    return {
        'base_url': os.getenv('Q38_BASE_URL', 'http://127.0.0.1:18016/v1'),
        'model': os.getenv('Q38_MODEL', 'Qwen3.8-Flash-Next-FP8'),
        'api_key': key,
    }


@dataclass(frozen=True)
class ExperimentBudget:
    context_tokens: int = 16_384
    total_tokens: int = 65_536
    max_calls: int = 8
    max_output_tokens: int = 2_048
    temperature: float = 0.0
    seed: int = 20261101
    token_safety_margin: int = 16


class ProviderTransport:
    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float = 180):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        # Do not route local SSH forwarding through an unrelated HTTP proxy.
        self.opener = build_opener(ProxyHandler({}))
        self._counts: dict[str, int] = {}
        self._lock = threading.RLock()

    def request(self, path: str, body: dict[str, Any] | None = None, *, outside_v1=False):
        base = self.base_url.removesuffix('/v1') if outside_v1 else self.base_url
        request = Request(base + path,
                          data=json.dumps(body, ensure_ascii=False).encode() if body is not None else None,
                          headers={'Authorization': 'Bearer ' + self.api_key,
                                   'Content-Type': 'application/json'})
        with self.opener.open(request, timeout=self.timeout) as response:
            return json.load(response)

    def count(self, *, messages=None, text=None, tools=None, tool_choice=None) -> int:
        body: dict[str, Any] = {'model': self.model}
        if messages is not None:
            body.update(messages=messages, add_generation_prompt=True,
                        chat_template_kwargs={'enable_thinking': False, 'preserve_thinking': False})
            if tools is not None:
                body['tools'] = tools
            if tool_choice is not None:
                body['tool_choice'] = tool_choice
        else:
            body.update(prompt=text or '', add_special_tokens=False)
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        with self._lock:
            if digest in self._counts:
                return self._counts[digest]
        payload = self.request('/tokenize', body, outside_v1=True)
        count = payload.get('count')
        if not isinstance(count, int) or count < 0:
            raise BudgetIntegrityError('Serving tokenizer did not return an exact count')
        with self._lock:
            self._counts[digest] = count
        return count


class BudgetedClient:
    """A separate instance must be used for each (method, sample) pair."""
    def __init__(self, transport: ProviderTransport, budget: ExperimentBudget, *, on_ledger_update=None):
        self.transport = transport
        self.budget = budget
        self.model = transport.model
        self.context_tokens = budget.context_tokens
        self.ledger: list[dict[str, Any]] = []
        self.last_usage: dict[str, Any] = {}
        self._lock = threading.RLock()
        self.on_ledger_update = on_ledger_update

    @property
    def remaining_calls(self):
        return max(0, self.budget.max_calls - len(self.ledger))

    @property
    def remaining_tokens(self):
        return max(0, self.budget.total_tokens - sum(x['charged_tokens'] for x in self.ledger))

    def count_text(self, text: str):
        return self.transport.count(text=text)

    def _with_budget_notice(self, messages):
        notice = (f'Experiment resource limits: input plus output per call <= {self.context_tokens} tokens; '
                  f'remaining physical generation requests INCLUDING this one: {self.remaining_calls}; '
                  f'remaining cumulative input plus output tokens: {self.remaining_tokens}; '
                  f'maximum output per request: {self.budget.max_output_tokens} tokens. '
                  'Root, recursive and summary requests all share these limits. '
                  'Complete the requested answer within these resources.')
        prepared = [dict(message) for message in messages]
        systems = []
        while prepared and prepared[0].get('role') == 'system':
            systems.append(str(prepared.pop(0).get('content') or ''))
        if systems:
            return [{'role': 'system', 'content': '\n\n'.join([notice, *systems])}, *prepared]
        return [{'role': 'system', 'content': notice}, *prepared]

    def count_messages(self, messages, *, tools=None, tool_choice=None):
        return self.transport.count(messages=self._with_budget_notice(messages), tools=tools, tool_choice=tool_choice) + self.budget.token_safety_margin

    def can_call(self, messages, max_tokens=2048, *, tools=None, tool_choice=None):
        output = min(int(max_tokens), self.budget.max_output_tokens)
        if output <= 0 or not self.remaining_calls:
            return False
        prompt = self.count_messages(messages, tools=tools, tool_choice=tool_choice)
        return (prompt + output <= self.context_tokens
                and prompt + output <= self.remaining_tokens)

    def complete(self, messages, max_tokens=2048) -> str:
        message = self._complete(messages, max_tokens=max_tokens)
        return str(message.get('content') or '')

    def complete_native(self, messages, tools, *, tool_choice='auto', max_tokens=2048) -> dict[str, Any]:
        """Native tool transport shares the exact same ledger and ceilings."""
        return self._complete(messages, max_tokens=max_tokens, tools=tools, tool_choice=tool_choice)

    def _complete(self, messages, *, max_tokens=2048, tools=None, tool_choice=None) -> dict[str, Any]:
        with self._lock:
            output = min(int(max_tokens), self.budget.max_output_tokens)
            # Recursive programs may submit a whole batch after their quota
            # is gone. Deny before tokenizer I/O; no new inference can fit.
            if output <= 0 or not self.remaining_calls or not self.remaining_tokens:
                raise BudgetExhausted(
                    f'Admission denied: output={output}, remaining_tokens={self.remaining_tokens}, '
                    f'remaining_calls={self.remaining_calls}')
            prepared = self._with_budget_notice(messages)
            prompt = self.transport.count(messages=prepared, tools=tools, tool_choice=tool_choice) + self.budget.token_safety_margin
            if output <= 0 or not self.remaining_calls or prompt + output > min(self.context_tokens, self.remaining_tokens):
                raise BudgetExhausted(
                    f'Admission denied: prompt={prompt}, output={output}, '
                    f'remaining_tokens={self.remaining_tokens}, remaining_calls={self.remaining_calls}')
            started = time.perf_counter()
            entry: dict[str, Any] = {
                'physical_request': len(self.ledger) + 1,
                'prompt_sha256': hashlib.sha256(json.dumps(prepared, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
                'tools_sha256': hashlib.sha256(json.dumps(tools, sort_keys=True, ensure_ascii=False).encode()).hexdigest() if tools is not None else None,
                'native_tools': tools is not None,
                'admission_prompt_tokens': prompt,
                'requested_output_tokens': output,
                # On uncertain transport failure charge the entire reservation.
                'charged_tokens': prompt + output,
                'status': 'inflight',
            }
            self.ledger.append(entry)
            # Persist the reservation before sending. A process interruption
            # must never make an already-sent request disappear on resume.
            if self.on_ledger_update:
                self.on_ledger_update(self.ledger)
            body = {'model': self.model, 'messages': prepared, 'max_tokens': output,
                    'temperature': self.budget.temperature, 'seed': self.budget.seed,
                    'chat_template_kwargs': {'enable_thinking': False, 'preserve_thinking': False}}
            if tools is not None:
                body['tools'] = tools
            if tool_choice is not None:
                body['tool_choice'] = tool_choice
            try:
                payload = self.transport.request('/chat/completions', body)
                usage = dict(payload.get('usage') or {})
                self.last_usage = usage
                actual_prompt, completion = usage.get('prompt_tokens'), usage.get('completion_tokens')
                known = all(isinstance(x, int) and x >= 0 for x in (actual_prompt, completion))
                entry.update(usage=usage, usage_known=known)
                if known:
                    entry['charged_tokens'] = actual_prompt + completion
                    if actual_prompt > prompt or completion > output:
                        raise BudgetIntegrityError('Provider usage exceeded the admitted reservation')
                message = (payload.get('choices') or [{}])[0].get('message') or {}
                content = str(message.get('content') or '')
                entry.update(status='ok', finish_reason=(payload.get('choices') or [{}])[0].get('finish_reason'),
                             empty_content=not bool(content.strip()),
                             reasoning_present=bool(message.get('reasoning') or message.get('reasoning_content')),
                             content=content, tool_calls=message.get('tool_calls'))
                if sum(x['charged_tokens'] for x in self.ledger) > self.budget.total_tokens:
                    raise BudgetIntegrityError('Cumulative token budget violated')
                return dict(message)
            except Exception as exc:
                entry.update(status='error', error_type=type(exc).__name__, error=str(exc)[:400])
                raise
            finally:
                entry['seconds'] = round(time.perf_counter() - started, 4)
                if self.on_ledger_update:
                    self.on_ledger_update(self.ledger)

    def summary(self):
        usages = [x.get('usage') or {} for x in self.ledger]
        return {'budget': asdict(self.budget), 'physical_requests': len(self.ledger),
                'prompt_tokens': sum(x.get('prompt_tokens', 0) for x in usages),
                'completion_tokens': sum(x.get('completion_tokens', 0) for x in usages),
                'charged_tokens': sum(x['charged_tokens'] for x in self.ledger),
                'all_usage_known': all(x.get('usage_known', False) for x in self.ledger),
                'within_budget': (len(self.ledger) <= self.budget.max_calls
                                  and sum(x['charged_tokens'] for x in self.ledger) <= self.budget.total_tokens
                                  and all(x['admission_prompt_tokens'] + x['requested_output_tokens'] <= self.context_tokens for x in self.ledger)
                                  and not any(x.get('error_type') == 'BudgetIntegrityError' for x in self.ledger)),
                'model_seconds': round(sum(x.get('seconds', 0) for x in self.ledger), 4)}
