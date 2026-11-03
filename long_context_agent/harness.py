"""Model-first orchestration for long-context reading."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .token_budget import AdaptiveTokenEstimator


@dataclass(frozen=True)
class HarnessResult:
    mode: str
    estimated_prompt_tokens: int
    context_limit_tokens: int
    reserve_tokens: int
    document: Any
    memory: Any = None
    state: Any = None
    reason: str = ""
    estimate_method: str = ""
    estimator_chars_per_token: float = 0.0
    estimator_observations: int = 0
    estimate_scope: str = ""

    @property
    def used_reader(self) -> bool:
        return self.mode == "chunked_reader"


class LongContextHarness:
    """Route full-context prompts around the reader when they fit."""

    def __init__(
        self,
        reader_agent: Any,
        *,
        max_context_tokens: int = 262_144,
        reserve_tokens: int = 8_192,
        chars_per_token: float = 4.0,
        read_when_over_budget: bool = False,
        token_estimator: AdaptiveTokenEstimator | None = None,
    ):
        if max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be positive")
        if reserve_tokens < 0 or reserve_tokens >= max_context_tokens:
            raise ValueError("reserve_tokens must be non-negative and below max_context_tokens")
        self.reader_agent = reader_agent
        self.max_context_tokens = int(max_context_tokens)
        self.reserve_tokens = int(reserve_tokens)
        self.token_estimator = token_estimator or AdaptiveTokenEstimator(chars_per_token=chars_per_token)
        self.chars_per_token = self.token_estimator.chars_per_token
        self.read_when_over_budget = bool(read_when_over_budget)

    def estimate_prompt_tokens(self, prompt: str, *, scope: str | None = None) -> int:
        return self.token_estimator.estimate(prompt, scope=scope).tokens

    def observe_prompt_tokens(self, prompt: str, actual_prompt_tokens: int, *, scope: str | None = None) -> None:
        """Calibrate future routing decisions from provider usage metadata."""
        self.token_estimator.observe(prompt, actual_prompt_tokens, scope=scope)
        self.chars_per_token = self.token_estimator.chars_per_token

    def answer_with_evidence_gaps(self, sample: dict[str, Any], client: Any, *, config: Any = None) -> dict[str, Any]:
        """Answer a task with adaptive evidence acquisition under a hard ledger.

        Unlike ``prepare`` (which only selects a route), this opt-in execution
        API performs budgeted model calls. The client's context and cumulative
        limits are authoritative, and must not exceed this harness's ceiling.
        Labels/scoring callbacks are deliberately absent from the interface.
        """
        from .adaptive_evidence import run_adaptive
        if int(client.context_tokens) > self.max_context_tokens:
            raise ValueError('Budget client context exceeds the harness ceiling')
        return run_adaptive(sample, client, config=config)

    def answer_with_structured_facts(self, sample: dict[str, Any], client: Any, *, config: Any = None) -> dict[str, Any]:
        """Accumulate sourced facts and compute aggregates before answering.

        The v2 window controller remains available through
        ``answer_with_evidence_gaps`` for reproducible paired comparisons.
        """
        from .adaptive_evidence_v3 import run_adaptive_v3
        if int(client.context_tokens) > self.max_context_tokens:
            raise ValueError('Budget client context exceeds the harness ceiling')
        return run_adaptive_v3(sample, client, config=config)

    def prepare(
        self,
        source: Any,
        *,
        full_prompt: str,
        scope: str | None = None,
        resume: bool = False,
        **read_kwargs: Any,
    ) -> HarnessResult:
        estimate = self.token_estimator.estimate(full_prompt, scope=scope)
        estimated = estimate.tokens
        usable = self.max_context_tokens - self.reserve_tokens
        if estimated <= usable:
            return HarnessResult(
                mode="direct_pass",
                estimated_prompt_tokens=estimated,
                context_limit_tokens=self.max_context_tokens,
                reserve_tokens=self.reserve_tokens,
                document=source,
                reason="full_prompt_fits_model_context",
                estimate_method=estimate.method,
                estimator_chars_per_token=estimate.chars_per_token,
                estimator_observations=estimate.observations,
                estimate_scope=estimate.scope or "",
            )
        if not self.read_when_over_budget:
            return HarnessResult(
                mode="raw_source",
                estimated_prompt_tokens=estimated,
                context_limit_tokens=self.max_context_tokens,
                reserve_tokens=self.reserve_tokens,
                document=source,
                reason="full_prompt_exceeds_model_context_raw_source_route",
                estimate_method=estimate.method,
                estimator_chars_per_token=estimate.chars_per_token,
                estimator_observations=estimate.observations,
                estimate_scope=estimate.scope or "",
            )
        document, memory, state = self.reader_agent.read(source, resume=resume, **read_kwargs)
        return HarnessResult(
            mode="chunked_reader",
            estimated_prompt_tokens=estimated,
            context_limit_tokens=self.max_context_tokens,
            reserve_tokens=self.reserve_tokens,
            document=document,
            memory=memory,
            state=state,
            reason="full_prompt_exceeds_model_context",
            estimate_method=estimate.method,
            estimator_chars_per_token=estimate.chars_per_token,
            estimator_observations=estimate.observations,
            estimate_scope=estimate.scope or "",
        )


__all__ = ["HarnessResult", "LongContextHarness"]
