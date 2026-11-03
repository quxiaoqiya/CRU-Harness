from __future__ import annotations

import json
import tempfile
from pathlib import Path

from long_context_agent.harness import LongContextHarness
from long_context_agent.token_budget import AdaptiveTokenEstimator


def _approx(value: float, expected: float, tolerance: float) -> bool:
    return abs(value - expected) <= tolerance


def test_scope_prior_overrides_the_global_ratio() -> None:
    estimator = AdaptiveTokenEstimator(
        chars_per_token=4.0,
        scope_coefficients={"Code Repository Understanding": 2.5},
    )
    prompt = "x" * 800_000
    default = estimator.estimate(prompt)
    scoped = estimator.estimate(prompt, scope="Code Repository Understanding")

    assert default.method == "fallback_chars_per_token"
    assert scoped.method == "scope_prior"
    assert scoped.chars_per_token == 2.5
    assert scoped.tokens > default.tokens
    assert scoped.scope == "Code Repository Understanding"


def test_unknown_scope_falls_back_to_the_global_ratio() -> None:
    estimator = AdaptiveTokenEstimator(chars_per_token=4.0, scope_coefficients={"Known": 2.5})
    result = estimator.estimate("x" * 4000, scope="Unknown Domain")

    assert result.method == "fallback_chars_per_token"
    assert result.chars_per_token == 4.0
    assert result.scope == "Unknown Domain"


def test_scope_observations_do_not_move_the_global_ratio() -> None:
    estimator = AdaptiveTokenEstimator(chars_per_token=4.0, scope_coefficients={"Code": 3.0})
    estimator.observe("x" * 900_000, 250_000, scope="Code")

    scoped = estimator.scopes["Code"]
    assert scoped.observations == 1
    # 900k chars / 250k tokens = 3.6 chars/token; slow EWMA towards it.
    assert 3.0 < scoped.chars_per_token < 3.6
    assert estimator.chars_per_token == 4.0
    assert estimator.observations == 0
    assert _approx(estimator.chars_per_token, 4.0, 0.0001)


def test_unscoped_observations_still_update_the_global_ratio() -> None:
    estimator = AdaptiveTokenEstimator(chars_per_token=4.0)
    estimator.observe("x" * 300_000, 100_000)
    assert estimator.observations == 1
    # EWMA(0.75) from 4.0 towards the observed 3.0 is 3.75.
    assert _approx(estimator.chars_per_token, 3.75, 0.01)


def test_real_tokenizer_takes_precedence_over_coefficients() -> None:
    estimator = AdaptiveTokenEstimator(
        chars_per_token=4.0,
        tokenizer=lambda text: len(text) // 3,
        scope_coefficients={"Code": 2.5},
    )
    result = estimator.estimate("x" * 3000, scope="Code")

    assert result.method == "tokenizer"
    assert _approx(result.tokens, 1020, 1)
    assert result.scope == "Code"


def test_calibration_file_round_trip() -> None:
    with tempfile.TemporaryDirectory() as root:
        path = Path(root) / "calibration.json"
        path.write_text(
            json.dumps(
                {
                    "global": {"chars_per_token": 3.7425, "observations": 401},
                    "scopes": {
                        "Code Repository Understanding": {
                            "chars_per_token": 3.775,
                            "observations": 18,
                        }
                    },
                }
            ),
            encoding="utf-8",
        )
        estimator = AdaptiveTokenEstimator.from_calibration_file(path)
        assert _approx(estimator.chars_per_token, 3.7425, 0.001)
        assert estimator.observations == 401
        scoped = estimator.estimate("x" * 3775, scope="Code Repository Understanding")
        assert scoped.method == "scope_calibrated"
        assert scoped.observations == 18
        assert _approx(scoped.tokens, 1020, 1)


def test_missing_calibration_file_falls_back_to_the_default_ratio() -> None:
    with tempfile.TemporaryDirectory() as root:
        estimator = AdaptiveTokenEstimator.from_calibration_file(Path(root) / "absent.json")
    assert estimator.chars_per_token == 4.0
    assert estimator.estimate("x" * 8000).method == "fallback_chars_per_token"


def test_harness_routes_with_the_scope_coefficient() -> None:
    estimator = AdaptiveTokenEstimator(chars_per_token=4.0, scope_coefficients={"Code": 2.5})
    harness = LongContextHarness(
        reader_agent=None,
        max_context_tokens=262_144,
        reserve_tokens=8_192,
        token_estimator=estimator,
    )
    prompt = "x" * 900_000

    default = harness.prepare(object(), full_prompt=prompt)
    scoped = harness.prepare(object(), full_prompt=prompt, scope="Code")

    assert default.mode == "direct_pass"
    assert scoped.mode == "raw_source"
    assert scoped.estimate_scope == "Code"
    assert scoped.estimator_chars_per_token == 2.5
    assert scoped.estimated_prompt_tokens > default.estimated_prompt_tokens


def test_state_reports_every_scope() -> None:
    estimator = AdaptiveTokenEstimator(
        chars_per_token=4.0,
        scope_coefficients={"Code": 2.5, "Dialogue": 3.6},
    )
    state = estimator.state()
    assert state["scopes"]["Code"]["chars_per_token"] == 2.5
    assert state["scopes"]["Dialogue"]["chars_per_token"] == 3.6
