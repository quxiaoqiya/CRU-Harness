"""Adaptive prompt-token estimation for the long-context Harness.

Estimation has three layers, in priority order:

1. a local tokenizer, when one is available;
2. a per-scope character/token coefficient, seeded from measured data and
   updated from provider-reported ``usage.prompt_tokens``;
3. a global character/token ratio, updated the same way.

The scope is normally the document domain.  Code, tables and dialogue tokenize
very differently, so one global ratio silently routes them to retrieval at the
wrong time; a per-domain coefficient fixes that without requiring a tokenizer
installation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping


TokenCounter = Callable[[str], int]


@dataclass(frozen=True)
class TokenEstimate:
    """A token estimate plus the method and calibration state that produced it."""

    tokens: int
    method: str
    chars_per_token: float
    observations: int
    scope: str | None = None


@dataclass
class ScopeCalibration:
    """Mutable per-scope coefficient and its observation bookkeeping."""

    chars_per_token: float
    observations: int = 0
    last_observed_chars: int = 0
    last_observed_tokens: int = 0


class AdaptiveTokenEstimator:
    """Estimate prompt tokens with a tokenizer, per-scope ratios, and EWMA.

    ``tokenizer`` may be any local callable returning a token count.  When it
    is unavailable, the estimator uses the scope coefficient (for example a
    domain such as ``Code Repository Understanding``) and falls back to the
    global ratio for unknown scopes.  Both ratios are clamped so a single
    outlier response cannot destabilise routing.
    """

    def __init__(
        self,
        *,
        chars_per_token: float = 4.0,
        tokenizer: TokenCounter | None = None,
        safety_margin: float = 1.02,
        ewma_decay: float = 0.75,
        scope_ewma_decay: float = 0.9,
        min_chars_per_token: float = 1.0,
        max_chars_per_token: float = 12.0,
        scope_coefficients: Mapping[str, float] | None = None,
    ) -> None:
        if chars_per_token <= 0:
            raise ValueError("chars_per_token must be positive")
        if safety_margin < 1.0:
            raise ValueError("safety_margin must be at least 1")
        if not 0.0 <= ewma_decay < 1.0:
            raise ValueError("ewma_decay must be in [0, 1)")
        if not 0.0 <= scope_ewma_decay < 1.0:
            raise ValueError("scope_ewma_decay must be in [0, 1)")
        if min_chars_per_token <= 0 or max_chars_per_token < min_chars_per_token:
            raise ValueError("invalid chars_per_token bounds")
        self.min_chars_per_token = float(min_chars_per_token)
        self.max_chars_per_token = float(max_chars_per_token)
        self.chars_per_token = self._clamp(chars_per_token)
        self.tokenizer = tokenizer
        self.safety_margin = float(safety_margin)
        self.ewma_decay = float(ewma_decay)
        self.scope_ewma_decay = float(scope_ewma_decay)
        self.observations = 0
        self.last_observed_chars = 0
        self.last_observed_tokens = 0
        self.scopes: MutableMapping[str, ScopeCalibration] = {
            str(scope): ScopeCalibration(chars_per_token=self._clamp(ratio))
            for scope, ratio in (scope_coefficients or {}).items()
        }

    def _clamp(self, ratio: float) -> float:
        return float(min(max(ratio, self.min_chars_per_token), self.max_chars_per_token))

    @classmethod
    def from_calibration_file(
        cls,
        path: str | Path,
        *,
        tokenizer: TokenCounter | None = None,
        **kwargs: Any,
    ) -> "AdaptiveTokenEstimator":
        """Build an estimator from a measurement file, ignoring a missing file."""
        calibration_path = Path(path)
        if not calibration_path.exists():
            return cls(tokenizer=tokenizer, **kwargs)
        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        global_block = payload.get("global") or {}
        scopes = {
            str(scope): float((block or {}).get("chars_per_token", 4.0))
            for scope, block in (payload.get("scopes") or {}).items()
            if (block or {}).get("chars_per_token")
        }
        estimator = cls(
            chars_per_token=float(global_block.get("chars_per_token", 4.0)),
            tokenizer=tokenizer,
            scope_coefficients=scopes,
            **kwargs,
        )
        if global_block.get("observations"):
            estimator.observations = int(global_block["observations"])
        for scope, block in (payload.get("scopes") or {}).items():
            if scope in estimator.scopes and (block or {}).get("observations"):
                estimator.scopes[scope].observations = int(block["observations"])
        return estimator

    def _tokenizer_count(self, prompt: str) -> int | None:
        if self.tokenizer is None:
            return None
        try:
            value = self.tokenizer(prompt)
            if isinstance(value, dict):
                value = value.get("input_ids", value.get("tokens", 0))
            if hasattr(value, "shape") and len(getattr(value, "shape", ())) >= 1:
                value = value.shape[-1]
            if not isinstance(value, (int, float)):
                value = len(value)
            return max(1, int(value))
        except Exception:
            # A tokenizer is an optimization, not a reason to make routing
            # unavailable.  Fall back to the calibrated ratio.
            return None

    def _resolve(self, scope: str | None) -> tuple[float, str, int]:
        if scope:
            calibration = self.scopes.get(str(scope))
            if calibration is not None:
                method = "scope_calibrated" if calibration.observations else "scope_prior"
                return calibration.chars_per_token, method, calibration.observations
        method = "adaptive_chars_per_token" if self.observations else "fallback_chars_per_token"
        return self.chars_per_token, method, self.observations

    def estimate(self, prompt: str, scope: str | None = None) -> TokenEstimate:
        text = prompt or ""
        tokenized = self._tokenizer_count(text)
        if tokenized is not None:
            ratio, _, observations = self._resolve(scope)
            return TokenEstimate(
                tokens=max(1, math.ceil(tokenized * self.safety_margin)),
                method="tokenizer",
                chars_per_token=ratio,
                observations=observations,
                scope=str(scope) if scope else None,
            )
        ratio, method, observations = self._resolve(scope)
        estimated = max(1, math.ceil(len(text) / ratio * self.safety_margin))
        return TokenEstimate(
            tokens=estimated,
            method=method,
            chars_per_token=ratio,
            observations=observations,
            scope=str(scope) if scope else None,
        )

    def observe(self, prompt: str, actual_prompt_tokens: int, scope: str | None = None) -> None:
        """Update the coefficient for one scope from provider-reported usage.

        When a scope is given only that scope moves: pooling code, tables and
        prose into one ratio is what caused the misroutes this calibration
        fixes.  Without a scope the global ratio is updated for compatibility.
        """
        actual = int(actual_prompt_tokens)
        if actual <= 0 or not prompt:
            return
        observed_ratio = self._clamp(len(prompt) / actual)
        if scope:
            key = str(scope)
            calibration = self.scopes.setdefault(
                key, ScopeCalibration(chars_per_token=self.chars_per_token)
            )
            calibration.chars_per_token = self._clamp(
                self.scope_ewma_decay * calibration.chars_per_token
                + (1.0 - self.scope_ewma_decay) * observed_ratio
            )
            calibration.observations += 1
            calibration.last_observed_chars = len(prompt)
            calibration.last_observed_tokens = actual
            return
        self.chars_per_token = self._clamp(
            self.ewma_decay * self.chars_per_token + (1.0 - self.ewma_decay) * observed_ratio
        )
        self.observations += 1
        self.last_observed_chars = len(prompt)
        self.last_observed_tokens = actual

    def observe_usage(self, prompt: str, usage: Any, scope: str | None = None) -> None:
        """Consume an OpenAI-compatible usage mapping when available."""
        if isinstance(usage, dict):
            value = usage.get("prompt_tokens")
            if isinstance(value, (int, float)):
                self.observe(prompt, int(value), scope=scope)

    def scope_state(self, scope: str) -> dict[str, Any] | None:
        calibration = self.scopes.get(str(scope))
        if calibration is None:
            return None
        return {
            "chars_per_token": round(calibration.chars_per_token, 6),
            "observations": calibration.observations,
            "last_observed_chars": calibration.last_observed_chars,
            "last_observed_tokens": calibration.last_observed_tokens,
        }

    def state(self) -> dict[str, Any]:
        return {
            "chars_per_token": round(self.chars_per_token, 6),
            "observations": self.observations,
            "safety_margin": self.safety_margin,
            "last_observed_chars": self.last_observed_chars,
            "last_observed_tokens": self.last_observed_tokens,
            "method": "tokenizer"
            if self.tokenizer is not None
            else ("adaptive_chars_per_token" if self.observations else "fallback_chars_per_token"),
            "scopes": {
                scope: {
                    "chars_per_token": round(calibration.chars_per_token, 6),
                    "observations": calibration.observations,
                }
                for scope, calibration in sorted(self.scopes.items())
            },
        }


def build_local_tokenizer(model_name: str | None = None) -> TokenCounter | None:
    """Return a local tokenizer callable when a tokenizer library is installed.

    Supports a Hugging Face tokenizer file path (``Q38_TOKENIZER_PATH``) or a
    model name resolved through ``transformers``.  Returning ``None`` is a
    normal outcome on machines without either dependency; callers then fall
    back to the calibrated coefficients.
    """
    import os

    tokenizer_path = os.getenv("Q38_TOKENIZER_PATH")
    try:
        from tokenizers import Tokenizer  # type: ignore

        if tokenizer_path and Path(tokenizer_path).exists():
            tokenizer = Tokenizer.from_file(tokenizer_path)
            return lambda text: len(tokenizer.encode(text).ids)
    except Exception:
        pass
    try:
        from transformers import AutoTokenizer  # type: ignore

        if model_name:
            tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=False)
            return lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    except Exception:
        return None
    return None


__all__ = [
    "AdaptiveTokenEstimator",
    "ScopeCalibration",
    "TokenEstimate",
    "TokenCounter",
    "build_local_tokenizer",
]
