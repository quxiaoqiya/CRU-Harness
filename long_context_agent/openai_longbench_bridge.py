"""Budgeted Responses bridge for the existing LongBench evidence controller.

The controller's source selection, voting, and option verification are reused
unchanged. Every legacy ``complete`` or ``complete_answer`` invokes exactly
one budgeted generation. Explicit controller recovery calls share the same
token, physical-generation, and caller-supplied monetary ledger.

This module never loads dataset labels, starts a model run on import, or
implements an approximate-tokenizer fallback. It is a benchmark adapter,
intended to run from this repository alongside ``scripts``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import time
from types import SimpleNamespace
from typing import Any

from .experiment_budget import BudgetIntegrityError
from .openai_responses_transport import OpenAIResponsesTransport
from .token_budget import TokenEstimate


_DEFAULT_EFFORT = object()


class LongBenchResponsesTransport(OpenAIResponsesTransport):
    """Lossless wire encoding for source strings larger than one API field.

    The Responses schema limits an individual text string to 10,485,760
    characters and permits an ordered list of ``input_text`` content parts:
    https://developers.openai.com/api/reference/resources/responses/subresources/input_tokens/methods/count

    Caller-facing inputs retain the base transport's text-only validation.
    Strings over 10,000,000 characters become consecutive content parts in
    the same message. Concatenating their ``text`` values recovers the exact
    original string; no separator, normalization, or source truncation is
    introduced. Bare long strings use the documented equivalent user role.
    Smaller inputs keep the original wire representation.

    Counting and generation both invoke the inherited ``_common`` method,
    so both receive exactly this encoding, including part boundaries. Native
    context limits and generation budgets remain separate and unchanged.
    """

    TEXT_PART_CHARS = 10_000_000

    @classmethod
    def _text_parts(cls, value: str) -> list[dict[str, str]]:
        return [{'type': 'input_text', 'text': value[start:start+cls.TEXT_PART_CHARS]}
                for start in range(0, len(value), cls.TEXT_PART_CHARS)]

    @classmethod
    def _input(cls, *, messages=None, text=None):
        validated = super()._input(messages=messages, text=text)
        if isinstance(validated, str):
            if len(validated) <= cls.TEXT_PART_CHARS:
                return validated
            return [{'role': 'user', 'content': cls._text_parts(validated)}]
        return [
            {'role': message['role'],
             'content': (cls._text_parts(message['content'])
                         if len(message['content']) > cls.TEXT_PART_CHARS else message['content'])}
            for message in validated
        ]

    def configuration(self) -> dict[str, Any]:
        return {**super().configuration(),
                'oversized_text_encoding': 'ordered_input_text_parts',
                'text_part_character_limit': self.TEXT_PART_CHARS,
                'text_concatenation': 'lossless_no_inserted_delimiters',
                'short_input_wire_unchanged': True,
                'prompt_cache_ttl_default': '30m',
                'prompt_cache_ttl_overridden': False}


@dataclass(frozen=True)
class LongBenchChatResult:
    """The existing controller's result shape, without hidden recovery calls."""

    content: str
    reasoning: str
    usage: dict[str, Any]
    seconds: float
    fallback_used: bool = False
    attempts: int = 1


class LongBenchBudgetBridge:
    """Adapt a budget client, optionally wrapped by a monetary gate.

    ``client`` must expose the existing BudgetedClient protocol, including
    ``complete``, ``count_messages``, ``ledger``, and ``budget``. The Responses
    transport owns wire parameters and supplies exact usage. In particular,
    no legacy temperature, seed, thinking template, or reasoning override is
    forwarded by this bridge.
    """

    def __init__(self, client: Any):
        self.client = client
        self.model = client.model
        transport = client.transport
        if getattr(transport, 'reasoning_effort', None) != 'none':
            raise ValueError('The registered LongBench bridge requires reasoning=none')
        if int(client.budget.max_output_tokens) < 16:
            raise ValueError('Responses requires at least 16 output tokens')
        self._identity_failure = False

    @property
    def calls(self) -> int:
        return len(self.client.ledger)

    def configuration(self) -> dict[str, Any]:
        return {'bridge': 'longbench_budgeted_responses', 'reasoning_effort': 'none',
                'implicit_answer_recovery': False, 'implicit_http_retries': 0,
                'text_only': True, 'actual_ledger_authoritative': True}

    def complete(self, prompt: str, *, max_tokens: int,
                 enable_thinking: bool, preserve_thinking: bool = True,
                 reasoning_effort: Any = _DEFAULT_EFFORT) -> LongBenchChatResult:
        if self._identity_failure:
            raise BudgetIntegrityError('A previous response violated the registered model identity')
        if not isinstance(prompt, str):
            raise TypeError('A text prompt is required')
        if enable_thinking:
            raise ValueError('The registered LongBench bridge requires thinking disabled')
        if reasoning_effort is not _DEFAULT_EFFORT and reasoning_effort not in (None, 'none'):
            raise ValueError('Per-call reasoning overrides are not supported')
        if type(max_tokens) is not int or max_tokens < 16:
            raise ValueError('The output reservation must be an integer of at least 16')
        before = len(self.client.ledger)
        started = time.perf_counter()
        content = self.client.complete([{'role': 'user', 'content': prompt}], max_tokens=max_tokens)
        if len(self.client.ledger) != before + 1:
            self._identity_failure = True
            raise BudgetIntegrityError('One bridge call must correspond to one physical-generation entry')
        entry = self.client.ledger[-1]
        actual_model = (entry.get('provider_metadata') or {}).get('actual_model')
        if actual_model != self.model:
            self._identity_failure = True
            raise BudgetIntegrityError('Response model differs from the registered exact model ID')
        return LongBenchChatResult(
            content=str(content or ''), reasoning='', usage=dict(entry.get('usage') or {}),
            seconds=float(entry.get('seconds', time.perf_counter()-started)))

    def complete_answer(self, prompt: str, *, max_tokens: int,
                        enable_thinking: bool, preserve_thinking: bool = True) -> LongBenchChatResult:
        # Empty or incomplete outputs are not silently retried here. The
        # unchanged controller explicitly budgets its own fallback attempts.
        return self.complete(prompt, max_tokens=max_tokens,
                             enable_thinking=enable_thinking,
                             preserve_thinking=preserve_thinking)


class ExactBudgetEstimator:
    """Exact admission count for the controller's full-prompt route decision.

    Counts include the budget notice and safety margin that will be present
    on the actual budgeted request. Errors propagate; neither calibration nor
    character-ratio estimates are substituted when counting is unavailable.
    """

    chars_per_token = 4.0  # Descriptive interface field; never used for routing.
    observations = 0

    def __init__(self, client: Any):
        self.client = client

    def estimate(self, prompt: str, scope: str | None = None) -> TokenEstimate:
        count = self.client.count_messages([{'role': 'user', 'content': prompt}])
        if type(count) is not int or count < 0:
            raise BudgetIntegrityError('Exact routing count is not a nonnegative integer')
        return TokenEstimate(tokens=count, method='openai_responses_exact_admission',
                             chars_per_token=(len(prompt)/count if count else 0.0),
                             observations=0, scope=scope)


@dataclass(frozen=True)
class LongBenchEvidenceConfig:
    chunk_chars: int = 48_000
    candidate_count: int = 3
    max_windows: int = 8
    window_chars: int = 4_000
    evidence_max_chars: int = 28_000
    candidate_max_tokens: int = 2_048
    option_verify: bool = True
    option_verify_tokens: int = 2_048
    option_verify_retries: int = 1
    reserve_tokens: int = 2_048

    def namespace(self) -> SimpleNamespace:
        if any(type(value) is not int or value <= 0 for name, value in asdict(self).items()
               if name not in {'option_verify', 'option_verify_retries'}):
            raise ValueError('Evidence counts and limits must be positive integers')
        if type(self.option_verify) is not bool or type(self.option_verify_retries) is not int or self.option_verify_retries < 0:
            raise ValueError('Invalid option-verification configuration')
        return SimpleNamespace(
            reserve_tokens=self.reserve_tokens, candidate_max_tokens=self.candidate_max_tokens,
            harness_option_verify=self.option_verify,
            harness_option_verify_tokens=self.option_verify_tokens,
            harness_option_verify_retries=self.option_verify_retries)


def run_longbench_harness(row: Any, client: Any, *,
                         config: LongBenchEvidenceConfig | None = None,
                         groups: list[dict[str, Any]] | None = None,
                         document: Any = None, require_raw_source: bool = True) -> dict[str, Any]:
    """Run the existing evidence controller on a label-free benchmark row.

    Natural long-source experiments should retain ``require_raw_source=True``
    so a full-prompt fit cannot be misreported as context extension. No Direct
    result is reused. ``document`` is optional because the raw-source route
    does not read an IR; avoiding the unused legacy IR also avoids its
    quadratic prefix-word counting on very large stress sources.

    Candidate views use original source spans. A candidate or verifier that
    exceeds exact request/cumulative admission is denied by ``client`` before
    generation; it is never clipped or silently forwarded past the ceiling.
    The controller's diagnostic call counter is not the authoritative ledger.
    """
    from scripts.benchmark_longbench_v2_full import chunk_items, run_harness
    from scripts.longbench_v2_protocol import evidence_window_groups, official_prompt

    if getattr(row, 'answer', None) != '':
        raise ValueError('Inference requires a label-free LongBenchRow with answer empty')
    config = config or LongBenchEvidenceConfig()
    args = config.namespace()
    if config.reserve_tokens >= client.context_tokens:
        raise ValueError('Reserved output must be smaller than the request ceiling')
    if max(config.candidate_max_tokens, config.option_verify_tokens) > client.budget.max_output_tokens:
        raise ValueError('Controller output reservations exceed the registered client ceiling')
    estimator = ExactBudgetEstimator(client)
    estimate = estimator.estimate(official_prompt(row), scope=row.domain)
    if require_raw_source and estimate.tokens + config.reserve_tokens <= client.context_tokens:
        raise ValueError('Full prompt fits the registered Harness window; raw-source extension is not triggered')
    if groups is None:
        groups = evidence_window_groups(
            row, chunk_items(row.context, config.chunk_chars),
            candidate_count=config.candidate_count, max_windows=config.max_windows,
            window_chars=config.window_chars, max_total_chars=config.evidence_max_chars)
    result = run_harness(row, document, groups, None, args=args, estimator=estimator,
                         budget=client.context_tokens, client=LongBenchBudgetBridge(client))
    result.pop('correct', None)
    result.update(bridge='longbench_budgeted_responses', label_free=True,
                  shared_usage=client.summary(), authoritative_physical_calls=len(client.ledger))
    return result


__all__ = ['LongBenchResponsesTransport', 'LongBenchBudgetBridge', 'LongBenchChatResult',
           'ExactBudgetEstimator', 'LongBenchEvidenceConfig', 'run_longbench_harness']
