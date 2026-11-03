"""Export the scored, frozen LongBench paired experiment as public metadata.

This helper does not run on import. It refuses to load scored audits until
all 75 original, 39 recovery, and registered auxiliary records and the unified-scoring proof exist.
It never opens label files, prediction archives, source inputs, or raw cache.
Only the already-scored audit may supply the short A/B/C/D result letters.

Execute only after the parent task announces every registered phase complete:
    python work/longbench_openai_limits_20261102/export_public.py
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo


ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "long_context_agent" / "harness.py").is_file())
MODELS = ("gpt-6-sol", "gpt-6-luna", "gpt-6.1-terra")
NATIVE = 1_050_000
EXPECTED_FREEZE = "941c86120e66020439571fbe8888999c643ebcf83fc315d7ec7e1e0e1045208d"
EXPECTED_RECOVERY_FREEZE = "febefea46171ba7d843104aa8857bff82670b597e6b0bb0de6b640e7c7f312c1"
EXPECTED_HEADTAIL_FREEZE = "dfb25fdb5cd26067a881dca5747b78ded10c7a8e5389091eacb2529a26890c3f"
ZERO_GENERATION_NETWORK_MESSAGE = "OpenAI request failed; response or usage may be unknown"
HARNESS_BUDGET = {"context_tokens": 16_384, "total_tokens": 65_536,
                  "max_calls": 32, "max_output_tokens": 2_048, "token_safety_margin": 16}
DIRECT_BUDGET = {"context_tokens": NATIVE, "total_tokens": NATIVE,
                 "max_calls": 1, "max_output_tokens": 1_024, "token_safety_margin": 16}
HEADTAIL_BUDGET = {"context_tokens": 800_000, "total_tokens": 800_000,
                   "max_calls": 1, "max_output_tokens": 1_024, "token_safety_margin": 16}
MODEL_URLS = {model: "https://developers.openai.com/api/docs/models/" + model for model in MODELS}
SAFE_HTTP_CODES = {
    "unclassified", "invalid_api_key", "model_not_found", "insufficient_quota",
    "rate_limit_exceeded", "context_length_exceeded", "invalid_request_error",
    "unsupported_parameter", "unsupported_value", "server_error", "string_above_max_length",
    "invalid_value", "invalid_type",
}
SAFE_ERROR_TYPES = {
    "OpenAIResponsesTransportError", "OpenAIResponseNotCompletedError",
    "BudgetExhausted", "BudgetIntegrityError", "CostCapExceeded",
    "InterruptedTask", "ModelBlocked", "RuntimeError", "ValueError", "TypeError",
    "KeyError", "TimeoutError", "ConnectionError", "OSError", "ParseError",
}
SUM_FIELDS = (
    "physical_generations", "known_input_tokens", "known_output_tokens",
    "known_usage_tokens", "unknown_usage_requests", "unknown_usage_reserved_tokens",
    "charged_or_reserved_tokens", "ledger_error_requests", "count_queries_including_cache_hits",
    "count_api_requests", "count_api_errors", "cached_input_tokens", "cache_write_tokens",
    "cache_write_30m_tokens", "cache_write_12h_tokens", "reasoning_tokens",
    "count_logical_api_queries", "count_logical_api_errors", "count_cache_hits",
    "original_count_queries", "original_count_api_attempts", "original_count_api_errors",
    "recovery_count_queries", "recovery_count_api_attempts", "recovery_count_api_errors",
    "count_retry_attempts",
    "headtail_count_queries", "headtail_count_api_attempts", "headtail_count_api_errors",
)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def count(value):
    return type(value) is int and value >= 0


def number(value):
    return type(value) in (int, float) and math.isfinite(value)


def choice(value):
    return value if isinstance(value, str) and len(value) == 1 and value in "ABCD" else None


def enum(value):
    return value if isinstance(value, str) and re.fullmatch(r"[A-Za-z0-9_.-]{1,100}", value) else None


def sha(value):
    require(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value), "Invalid SHA-256")
    return value


def utc(value):
    require(number(value), "Missing timestamp")
    return datetime.fromtimestamp(value, timezone.utc).isoformat()


def ratio(k, n):
    return k / n if n else None


def wilson(k, n):
    if not n:
        return None
    z = 1.959963984540054
    p, d = k / n, 1 + z * z / n
    center = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, center - half), min(1.0, center + half)]


def safe_error(error):
    if not error:
        return None
    require(isinstance(error, dict), "Error metadata has the wrong shape")
    kind = error.get("type")
    kind = kind if kind in SAFE_ERROR_TYPES else "OtherError"
    match = re.search(r"status (\d{3}); code ([A-Za-z_]+)", str(error.get("message") or ""))
    status = int(match.group(1)) if match else None
    code = match.group(2) if match and match.group(2) in SAFE_HTTP_CODES else None
    if code == "context_length_exceeded":
        category = "provider_context_capacity_rejection"
    elif code == "string_above_max_length":
        category = "wire_text_field_size_limit"
    elif status in (401, 403, 404) or kind == "ModelBlocked":
        category = "authentication_or_model_access"
    elif status == 429:
        category = "quota" if code == "insufficient_quota" else "rate_limit"
    elif status is not None and status >= 500:
        category = "server_error_or_uncertain_delivery"
    elif kind == "OpenAIResponseNotCompletedError":
        category = "received_noncompleted_or_identity_tier_error"
    elif kind in ("BudgetExhausted", "CostCapExceeded"):
        category = "resource_admission"
    elif kind == "BudgetIntegrityError":
        category = "usage_integrity"
    elif kind == "InterruptedTask":
        category = "interrupted_without_replay"
    elif kind == "OpenAIResponsesTransportError" and status is None:
        category = "transport_or_uncertain_delivery"
    else:
        category = "other_recorded_error"
    return {"type": kind, "http_status": status, "safe_code": code, "category": category}


def safe_controller(result):
    """Counts and fixed controller states only; no evidence or verifier prose."""
    candidates = result.get("candidate_results") or []
    candidate_errors = result.get("candidate_errors") or []
    verifier_errors = result.get("option_verification_errors") or []
    errors = candidate_errors + verifier_errors
    types = Counter(e.get("type") if e.get("type") in SAFE_ERROR_TYPES else "OtherError"
                    for e in errors if isinstance(e, dict))
    return {
        "mode": enum(result.get("mode")),
        "bridge": enum(result.get("bridge")),
        "label_free": result.get("label_free") is True,
        "reused_direct": result.get("reused_direct") is True,
        "candidate_views": len(candidates),
        "query_mode_counts": [{"query_mode": mode, "count": n} for mode, n in sorted(
            Counter(enum(c.get("query_mode")) or "unreported"
                    for c in candidates if isinstance(c, dict)).items())],
        "combined_evidence_chars": result.get("evidence_chars") if count(result.get("evidence_chars")) else None,
        "successful_option_verifier_results": len(result.get("option_verifications") or []),
        "caught_candidate_errors": len(candidate_errors),
        "caught_option_verifier_errors": len(verifier_errors),
        "caught_error_type_counts": dict(types),
        "caught_verifier_parse_errors": sum(e.get("type") == "ParseError" for e in verifier_errors
                                            if isinstance(e, dict)),
        "explicit_verifier_retry_attempts": result.get("option_verification_retries", 0),
        "fallback_answer_attempted": result.get("fallback_answer") is True,
        "option_verification_override": result.get("option_verification_override") is True,
        "controller_model_call_counter_diagnostic_only": result.get("model_calls") if count(result.get("model_calls")) else None,
        "authoritative_physical_calls": result.get("authoritative_physical_calls") if count(result.get("authoritative_physical_calls")) else None,
    }


def rate_envelope(price, admitted_input):
    long = admitted_input > 272_000
    rin = max(Decimal(str(price[k])) for k in ("input", "cached_input", "cache_write_30m"))
    rout = Decimal(str(price["output"]))
    return rin * (2 if long else 1), rout * (Decimal("1.5") if long else 1)


def public_ledger(record, price, money, prior_counts=None, phase=None):
    ledger, budget = record.get("ledger") or [], record["budget"]
    require(isinstance(ledger, list) and len(ledger) <= budget["max_calls"], "Physical-generation limit failed")
    require(len(ledger) == len(money), "Monetary reservations do not match physical generations")
    totals = Counter({key: 0 for key in SUM_FIELDS})
    statuses, reasons, tiers, models, errors = Counter(), Counter(), set(), set(), Counter()
    upper = base = Decimal(0)
    missing_identity = 0
    long_requests = 0
    for position, (entry, reserve) in enumerate(zip(ledger, money), 1):
        prompt, output, charged = (entry.get(k) for k in
                                  ("admission_prompt_tokens", "requested_output_tokens", "charged_tokens"))
        require(all(count(v) for v in (prompt, output, charged)) and output > 0, "Invalid token ledger number")
        require(entry.get("physical_request") == position, "Generation numbering mismatch")
        require(output <= budget["max_output_tokens"] and prompt + output <= budget["context_tokens"],
                "Request token admission exceeded its registered ceiling")
        require(totals["charged_or_reserved_tokens"] + prompt + output <= budget["total_tokens"],
                "Cumulative token admission exceeded its registered ceiling")
        require(reserve["reserved_prompt_tokens"] == prompt and reserve["reserved_output_tokens"] == output,
                "Monetary admission differs from token admission")
        rin, rout = rate_envelope(price, prompt)
        reserved_amount = (Decimal(prompt) * rin + Decimal(output) * rout) / Decimal(1_000_000)
        require(Decimal(reserve["reserved_upper_usd"]) == reserved_amount
                and Decimal(reserve["input_envelope_per_million"]) == rin
                and Decimal(reserve["output_per_million"]) == rout,
                "Frozen monetary envelope mismatch")
        usage = entry.get("usage") or {}
        known = all(count(usage.get(k)) for k in ("prompt_tokens", "completion_tokens"))
        require((entry.get("usage_known") is True) is known and reserve.get("usage_known") is known,
                "Known-usage flags disagree")
        if known:
            inp, out = usage["prompt_tokens"], usage["completion_tokens"]
            require(inp <= prompt and out <= output and charged == inp + out, "Known usage exceeds admission")
            totals["known_input_tokens"] += inp
            totals["known_output_tokens"] += out
            totals["known_usage_tokens"] += inp + out
            amount = (Decimal(inp) * rin + Decimal(out) * rout) / Decimal(1_000_000)
            long_factor = Decimal(2) if prompt > 272_000 else Decimal(1)
            output_factor = Decimal("1.5") if prompt > 272_000 else Decimal(1)
            base += (Decimal(inp) * Decimal(str(price["input"])) * long_factor
                     + Decimal(out) * Decimal(str(price["output"])) * output_factor) / Decimal(1_000_000)
        else:
            require(charged == prompt + output, "Unknown usage did not retain token reservation")
            totals["unknown_usage_requests"] += 1
            totals["unknown_usage_reserved_tokens"] += charged
            amount = reserved_amount
        require(Decimal(reserve["estimated_upper_usd"]) == amount, "Monetary settlement mismatch")
        upper += amount
        totals["charged_or_reserved_tokens"] += charged
        totals["ledger_error_requests"] += entry.get("status") == "error"
        if prompt > 272_000:
            long_requests += 1
        meta = entry.get("provider_metadata") or usage.get("openai_responses_metadata") or {}
        actual = meta.get("actual_model")
        if known:
            require(actual == record["model"], "Known returned model differs from registered model")
            require(meta.get("returned_service_tier") == "default", "Known service tier differs from default")
        if actual in MODELS:
            models.add(actual)
        elif known:
            missing_identity += 1
        tier = meta.get("returned_service_tier")
        tiers.add(tier if tier in {"default", "auto", "flex", "priority", "scale", "unknown"} else "not_reported")
        status = meta.get("response_status")
        statuses[status if status in {"completed", "incomplete", "failed", "cancelled"} else "not_reported"] += 1
        reason = meta.get("incomplete_reason")
        if reason is not None:
            reasons[reason if reason in {"max_output_tokens", "content_filter"} else "other"] += 1
        provider = meta.get("provider_usage") or {}
        details = provider.get("input_tokens_details") or {}
        writes = details.get("cache_write_tokens_details") or {}
        out_details = provider.get("output_tokens_details") or {}
        for key, value in {
            "cached_input_tokens": details.get("cached_tokens", 0),
            "cache_write_tokens": details.get("cache_write_tokens", 0),
            "cache_write_30m_tokens": writes.get("ttl_30m_tokens", 0),
            "cache_write_12h_tokens": writes.get("ttl_12h_tokens", 0),
            "reasoning_tokens": out_details.get("reasoning_tokens", 0),
        }.items():
            require(count(value), "Provider detail is not a nonnegative integer")
            totals[key] += value
        require(writes.get("ttl_12h_tokens", 0) == 0, "Unregistered 12-hour cache-write price category")
        if entry.get("status") == "error":
            public_error = safe_error({"type": meta.get("error_type") or "OpenAIResponsesTransportError",
                                       "message": entry.get("error") or ""})
            errors[public_error["category"]] += 1
    totals["physical_generations"] = len(ledger)
    own_counts = record.get("count_audit") or []
    phase = phase or ("recovery" if record.get("outcome_origin") == "zero_generation_network_recovery"
                      or "count_attempt_audit" in record else "original")
    require(phase in {"original", "recovery", "headtail"}, "Unknown counting phase")
    is_recovery = phase == "recovery"
    is_headtail = phase == "headtail"
    original_counts = list(prior_counts or []) if is_recovery else own_counts
    if is_headtail:
        original_counts = []
    recovery_counts = own_counts if is_recovery else []
    headtail_counts = own_counts if is_headtail else []
    counts = original_counts + recovery_counts + headtail_counts
    attempts = record.get("count_attempt_audit") or []
    if is_recovery or is_headtail:
        require(all(type(a.get("attempt")) is int and 1 <= a["attempt"] <= 3
                    and a.get("status") in {"ok", "error"} for a in attempts),
                "Recovery count-attempt audit is incomplete or outside its registered retry limit")
    else:
        require(not attempts, "An original trial contains recovery count attempts")
    totals["original_count_queries"] = len(original_counts)
    totals["original_count_api_attempts"] = sum(c.get("cache_hit") is False for c in original_counts)
    totals["original_count_api_errors"] = sum(c.get("status") == "error" for c in original_counts)
    totals["recovery_count_queries"] = len(recovery_counts)
    totals["recovery_count_api_attempts"] = len(attempts) if is_recovery else 0
    totals["recovery_count_api_errors"] = sum(a.get("status") == "error" for a in attempts)
    if is_headtail:
        totals["recovery_count_api_errors"] = 0
    totals["headtail_count_queries"] = len(headtail_counts)
    totals["headtail_count_api_attempts"] = len(attempts) if is_headtail else 0
    totals["headtail_count_api_errors"] = sum(a.get("status") == "error" for a in attempts) if is_headtail else 0
    totals["count_retry_attempts"] = sum(a.get("attempt", 1) > 1 for a in attempts)
    totals["count_queries_including_cache_hits"] = len(counts)
    totals["count_logical_api_queries"] = sum(c.get("cache_hit") is False for c in counts)
    totals["count_logical_api_errors"] = sum(c.get("status") == "error" for c in counts)
    totals["count_cache_hits"] = sum(c.get("cache_hit") is True for c in counts)
    totals["count_api_requests"] = (totals["original_count_api_attempts"] + totals["recovery_count_api_attempts"]
                                    + totals["headtail_count_api_attempts"])
    totals["count_api_errors"] = (totals["original_count_api_errors"] + totals["recovery_count_api_errors"]
                                 + totals["headtail_count_api_errors"])
    summary = record["usage"]
    require(summary["physical_requests"] == len(ledger)
            and summary["charged_tokens"] == totals["charged_or_reserved_tokens"]
            and summary["prompt_tokens"] == totals["known_input_tokens"]
            and summary["completion_tokens"] == totals["known_output_tokens"]
            and summary["within_budget"] is True, "Stored usage summary differs from ledger")
    require(totals["known_usage_tokens"] + totals["unknown_usage_reserved_tokens"]
            == totals["charged_or_reserved_tokens"] <= budget["total_tokens"], "Token conservation failed")
    require(record.get("ledger_error_count") == totals["ledger_error_requests"], "Stored ledger-error count differs")
    return {
        **dict(totals), "all_usage_known": totals["unknown_usage_requests"] == 0,
        "actual_model_ids": sorted(models), "returned_service_tiers": sorted(tiers),
        "known_usage_missing_model_identity": missing_identity,
        "response_status_counts": dict(statuses), "incomplete_reason_counts": dict(reasons),
        "ledger_error_category_counts": dict(errors), "long_input_generation_requests": long_requests,
        "estimated_generation_cost_upper_usd": str(upper),
        "estimated_uncached_base_io_cost_on_known_usage_usd": str(base),
        "within_registered_budget": True,
    }


def measurement(record):
    m = record.get("measurements") or {}
    source, full = m.get("source_envelope_input_tokens"), m.get("full_prompt_admission_input_tokens")
    reserve = m.get("output_reserved_tokens")
    require(source is None or count(source), "Invalid source-envelope count")
    require(full is None or count(full), "Invalid full-prompt admission count")
    require(reserve is None or reserve in (1024, 2048), "Invalid recorded output reserve")
    complete = full is not None
    fits = full + 1024 <= NATIVE if complete else None
    if complete:
        require(m.get("fits_native_direct") is fits and m.get("native_context_tokens") == NATIVE,
                "Native capacity metadata mismatch")
    return {
        "source_envelope_input_tokens": source,
        "full_prompt_admission_input_tokens": full,
        "recorded_output_reserve_tokens": reserve,
        "native_direct_comparison_output_reserve_tokens": 1024,
        "fits_native_direct": fits,
        "full_prompt_fits_harness_working_window_with_2048_output":
            full + 2048 <= 16384 if complete else None,
        "count_semantics": "Responses input envelope; admission includes resource notice and 16 safety tokens",
    }


def totals(rows):
    result = {key: sum(r[key] for r in rows) for key in SUM_FIELDS}
    result.update(
        actual_model_ids=sorted({m for r in rows for m in r["actual_model_ids"]}),
        returned_service_tiers=sorted({t for r in rows for t in r["returned_service_tiers"]}),
        known_usage_missing_model_identity=sum(r["known_usage_missing_model_identity"] for r in rows),
        estimated_generation_cost_upper_usd=str(sum(
            (Decimal(r["estimated_generation_cost_upper_usd"]) for r in rows), Decimal(0))),
        estimated_uncached_base_io_cost_on_known_usage_usd=str(sum(
            (Decimal(r["estimated_uncached_base_io_cost_on_known_usage_usd"]) for r in rows), Decimal(0))),
    )
    for key in ("response_status_counts", "incomplete_reason_counts", "ledger_error_category_counts"):
        result[key] = dict(sum((Counter(r[key]) for r in rows), Counter()))
    return result


def pair_table(pairs):
    return {
        "n": len(pairs), "both_correct": sum(a and b for a, b in pairs),
        "direct_only_correct": sum(a and not b for a, b in pairs),
        "harness_only_correct": sum(b and not a for a, b in pairs),
        "both_unsuccessful": sum(not a and not b for a, b in pairs),
    }


def same_float(left, right):
    if left is None or right is None:
        return left is right
    return math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-15)


def size_result(rows, length_by_index):
    present = [r for r in rows if count(length_by_index.get(r["index"]))]
    if not present:
        return None
    r = max(present, key=lambda x: (length_by_index[x["index"]], -x["index"]))
    value = length_by_index[r["index"]]
    return {"index": r["index"], "id": r["id"],
            "full_native_direct_admission_input_tokens": value,
            "relative_to_native_window": value / NATIVE,
            "relative_to_harness_request_ceiling": value / 16384}


def ensure_public(value):
    forbidden = {
        "context", "question", "choice_A", "choice_B", "choice_C", "choice_D",
        "gold", "answer", "answers", "labels", "content", "reasoning", "trace", "facts",
        "record_facts", "evidence_text", "headers", "authorization", "api_key", "sample_paths",
        "raw_cache", "private_labels",
    }
    bad = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}|(?i:Bearer\s+\S+)|"
                     r"/(?:Users|home|root|private|uniparser|mnt)/|"
                     r"\b(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b")
    def walk(item, path="public"):
        if isinstance(item, dict):
            require(not (forbidden & set(item)),
                    "A forbidden raw field reached " + path + ": " + ", ".join(sorted(forbidden & set(item))))
            for key, child in item.items():
                walk(child, path + "." + str(key))
        elif isinstance(item, list):
            for position, child in enumerate(item):
                walk(child, path + "[" + str(position) + "]")
        elif isinstance(item, str):
            require(not bad.search(item), "A private path or credential-like string reached public export")
    walk(value)


def zero_generation_eligible(record):
    """The frozen recovery rule; never inspect outputs or correctness."""
    usage, error = record.get("usage") or {}, record.get("error") or {}
    return (
        record.get("terminal") is True and record.get("status") == "error"
        and record.get("ledger") == [] and usage.get("physical_requests") == 0
        and usage.get("charged_tokens") == 0
        and error.get("type") == "OpenAIResponsesTransportError"
        and error.get("message") == ZERO_GENERATION_NETWORK_MESSAGE
    )


def record_key(record):
    return record["model"], record["index"], record["arm"]


def record_filename(key):
    model, index, arm = key
    return f"{index}__{model}__{arm}.json"


def safe_provider_observations(record):
    """Keep only registered numeric capacity fields and safe error metadata."""
    allowed_numbers = {
        "maximum_context_length_tokens", "requested_total_tokens",
        "input_tokens", "completion_tokens",
    }
    result = []
    for event in record.get("provider_error_diagnostics") or []:
        require(event.get("stage") in {"input_token_count", "generation"}, "Unknown diagnostic stage")
        item = {"stage": event["stage"], "category": event.get("category")}
        require(item["category"] in {"http", "non_http"}, "Unknown diagnostic category")
        if item["category"] == "http":
            require(type(event.get("http_status")) is int and 100 <= event["http_status"] <= 599,
                    "Invalid safe HTTP status")
            item["http_status"] = event["http_status"]
            item["safe_code"] = event.get("code") if event.get("code") in SAFE_HTTP_CODES else None
            parameter = event.get("param")
            item["safe_param"] = parameter if isinstance(parameter, str) and re.fullmatch(
                r"(?:input|messages|model|max_tokens|max_output_tokens|reasoning|truncation)"
                r"(?:\[\d{1,4}\])?(?:\.(?:content|text|role|effort))?(?:\[\d{1,4}\])?(?:\.text)?",
                parameter) else None
            numbers = {}
            for name, values in (event.get("numeric_candidates") or {}).items():
                require(name in allowed_numbers and isinstance(values, list)
                        and all(type(n) is int and 0 <= n <= 2_147_483_647 for n in values),
                        "Unsafe provider numeric diagnostic")
                numbers[name] = sorted(set(values))
            unique = {name: values[0] for name, values in numbers.items() if len(values) == 1}
            require((event.get("unique_numeric_fields") or {}) == unique,
                    "Unique provider numeric diagnostic differs from candidates")
            item.update(numeric_candidates=numbers, unique_numeric_fields=unique)
            item["error_body_unparseable"] = event.get("error_body_unparseable") is True
        else:
            for name in ("exception_class", "reason_exception_class"):
                if event.get(name) is not None:
                    item[name] = enum(event[name]) or "OtherError"
            for name in ("errno", "reason_errno", "tunnel_http_status"):
                if type(event.get(name)) is int:
                    item[name] = event[name]
        result.append(item)
    return result


def recovery_context(formal, recovery, root):
    """Both completion/access gates precede loading either scored audit."""
    run = load(recovery / "run_summary.json")
    require(run.get("phase") == "recovery_inference_complete"
            and run.get("planned_recovery_records") == 39
            and run.get("terminal_recovery_records") == 39
            and run.get("parent_terminal_records") == 75
            and run.get("labels_read") is False, "All 75 original plus 39 recovery terminal records are required")
    access = load(recovery / "scoring_access.json")
    require(access.get("all_75_parent_and_39_recovery_terminal_before_labels") is True
            and number(access.get("first_scored_at_unix")), "The all-114 scoring proof is missing")
    frozen = load(recovery / "freeze.json")
    require(digest(recovery / "freeze.json") == EXPECTED_RECOVERY_FREEZE, "Recovery freeze differs")
    protocol = frozen["recovery"]
    require(protocol["parent_freeze_sha256"] == EXPECTED_FREEZE
            and protocol["expected_parent_records"] == 75
            and protocol["expected_recovery_records"] == 39
            and protocol["eligibility_uses_answers_or_correctness"] is False
            and protocol["parent_generation_budgets_reset"] is False
            and protocol["implicit_generation_retries"] == 0
            and protocol["max_count_attempts_per_network_call"] == 3
            and protocol["count_retry_delays"] == [5, 15], "Recovery registration differs")
    parent_freeze = load(formal / "freeze.json")
    require(frozen["code_sha256"] == parent_freeze["code_sha256"] == protocol["parent_code_sha256"],
            "Recovery changed the original algorithm/controller source set")
    for name, expected in frozen["code_sha256"].items():
        require(digest(root / name) == expected
                and digest(recovery / "frozen_source" / name) == expected,
                "A recovery algorithm source snapshot changed")
    wrapper = "scripts/benchmark_longbench_openai_recovery.py"
    require(digest(root / wrapper) == protocol["wrapper_sha256"]
            == digest(recovery / "frozen_source" / wrapper), "Recovery wrapper changed")
    require(access["labels_sha256"] == frozen["labels_sha256"] == parent_freeze["labels_sha256"],
            "Recovery declared label identity differs")
    require(access["first_scored_at_unix"] >= frozen["frozen_at_unix"],
            "Recovery scoring predates its freeze")
    guard = load(recovery / "guard_verification.json")
    require(guard.get("verified") is True and guard.get("probe_read_denied") is True
            and guard.get("labels_read") is False
            and guard.get("profile_sha256") == frozen["guard_profile_sha256"],
            "Recovery OS guard proof is missing")
    # These audits are the first files here that contain scored result letters.
    original = load(formal / "audit.json")
    composite = load(recovery / "audit.json")
    require(original["freeze_sha256"] == EXPECTED_FREEZE
            and composite["freeze_sha256"] == EXPECTED_RECOVERY_FREEZE
            and composite["parent_freeze_sha256"] == EXPECTED_FREEZE,
            "Scored original/composite identity differs")
    require(composite["original_records_preserved"] == 75 and composite["new_recovery_records"] == 39
            and composite["recovery_eligibility_used_answers_or_correctness"] is False
            and composite["first_trial_audit_sha256"] == digest(formal / "audit.json"),
            "Composite recovery lineage differs")
    originals = {record_key(r): r for r in original["samples"] + original["capacity_probes"]}
    order = protocol["execution_order"]
    eligible_keys = {record_key(r) for r in order}
    require(len(originals) == 75 and len(order) == len(eligible_keys) == 39,
            "Recovery eligibility matrix differs")
    require(eligible_keys == {key for key, r in originals.items() if zero_generation_eligible(r)},
            "Recovery selected a nonzero-generation or non-network outcome")
    for entry in order:
        key = record_key(entry)
        require(entry["parent_record_sha256"] == protocol["parent_record_sha256"][record_filename(key)]
                and entry["context_sha256"] == originals[key]["context_sha256"],
                "Recovery eligibility source/record hash differs")
    require(composite["first_trial_summary"] == original["summary"]
            and composite["first_trial_status_counts"] == dict(Counter(r["status"] for r in originals.values())),
            "Original incomplete-trial results were not retained")
    parent_money = load(formal / "cost_ledger.json")
    new_money = load(recovery / "cost_ledger.json")
    require(digest(formal / "cost_ledger.json") == protocol["parent_cost_ledger_sha256"]
            and parent_money["freeze_sha256"] == EXPECTED_FREEZE
            and new_money["freeze_sha256"] == EXPECTED_RECOVERY_FREEZE, "Cost-ledger phase identity differs")
    parent_upper = sum((Decimal(e["estimated_upper_usd"]) for e in parent_money["entries"]), Decimal(0))
    new_upper = sum((Decimal(e["estimated_upper_usd"]) for e in new_money["entries"]), Decimal(0))
    preflight_upper = Decimal(protocol["preflight_cost_upper_usd"])
    require(parent_upper == Decimal(parent_money["estimated_upper_usd"])
            == Decimal(protocol["parent_generation_cost_upper_usd"])
            == Decimal(composite["parent_generation_cost_upper_usd"]), "Original cost snapshot changed")
    require(new_upper == Decimal(new_money["estimated_upper_usd"])
            == Decimal(composite["recovery_generation_cost_upper_usd"]), "Recovery cost does not conserve")
    remaining = Decimal(protocol["global_cost_cap_usd"]) - parent_upper - preflight_upper
    require(remaining == Decimal(protocol["new_generation_cost_cap_usd"]) == Decimal(new_money["cap_usd"])
            and new_upper <= remaining
            and parent_upper + new_upper + preflight_upper
                == Decimal(composite["combined_cost_with_preflight_upper_usd"]) <= Decimal(50),
            "Original plus recovery plus preflight exceeds the shared global cap")
    require(parent_upper + new_upper == Decimal(composite["estimated_generation_cost_upper_usd"]),
            "Composite formal generation cost differs")
    for entry in parent_money["entries"]:
        arm, index = entry["id"].split(":")
        if (entry["model"], int(index), arm) in eligible_keys:
            require(Decimal(entry["estimated_upper_usd"]) == 0,
                    "A selected original task has an outstanding generation reservation")
    return {
        "directory": recovery, "freeze": frozen, "protocol": protocol, "access": access,
        "original_audit": original, "audit": composite, "originals": originals,
        "eligible_keys": eligible_keys, "parent_money": parent_money, "new_money": new_money,
        "merged_money": {
            "freeze_sha256": EXPECTED_FREEZE, "cap_usd": "50",
            "estimated_upper_usd": str(parent_upper + new_upper),
            "entries": parent_money["entries"] + new_money["entries"],
        },
        "parent_upper": parent_upper, "new_upper": new_upper, "preflight_upper": preflight_upper,
    }


def headtail_context(control, formal, recovery, root):
    """Final control/skip proof must precede reading ANY scored phase."""
    prior = load(recovery / "run_summary.json")
    require(prior.get("phase") == "recovery_inference_complete"
            and prior.get("terminal_recovery_records") == 39
            and prior.get("parent_terminal_records") == 75 and prior.get("labels_read") is False,
            "All 114 original/recovery records must complete before any scored audit is read")
    frozen = load(control / "freeze.json")
    require(digest(control / "freeze.json") == EXPECTED_HEADTAIL_FREEZE, "Head/tail freeze differs")
    expected = frozen["expected_control_records"]
    remaining = Decimal(frozen["remaining_before_control_usd"])
    indices, cap = ([143, 260], Decimal(13)) if remaining >= 13 else (
        ([143], Decimal("6.5")) if remaining >= Decimal("6.5") else ([], Decimal(0)))
    require(expected in (0, 3, 6) and expected == 3 * len(indices)
            and frozen["planned_indices"] == indices and Decimal(frozen["control_cap_usd"]) == cap,
            "Conditional head/tail selection differs from the money-and-length rule")
    require(frozen["parent_freeze_sha256"] == EXPECTED_FREEZE
            and frozen["recovery_freeze_sha256"] == EXPECTED_RECOVERY_FREEZE
            and frozen["models"] == list(MODELS) and frozen["budget"] == HEADTAIL_BUDGET
            and frozen["labels_read"] is False
            and frozen["all_114_terminal_before_control_registration"] is True
            and frozen["selection_uses_answers_or_model_outputs"] is False
            and frozen["comparison_has_unequal_compute"] is True,
            "Head/tail registration identity differs")
    require(remaining == Decimal(frozen["global_cap_usd"])
                - Decimal(frozen["parent_cost_upper_usd"]) - Decimal(frozen["recovery_cost_upper_usd"])
                - Decimal(frozen["preflight_cost_upper_usd"])
            and Decimal(frozen["global_cap_usd"]) == 50, "Control remaining-cap arithmetic differs")
    run = load(control / "run_summary.json")
    require(run.get("phase") == "headtail_control_complete"
            and run.get("planned_control_records") == expected
            and run.get("terminal_control_records") == expected
            and run.get("labels_read") is False and run.get("budget_skipped") is (expected == 0),
            "All registered head/tail controls or the budget skip must complete before export")
    access = load(control / "scoring_access.json")
    require(access.get("all_114_and_control_terminal_before_labels") is True
            and access.get("control_terminal_count") == expected
            and access.get("control_budget_skip") is (expected == 0)
            and number(access.get("first_scored_at_unix"))
            and access["first_scored_at_unix"] >= frozen["frozen_at_unix"],
            "The final all-phases scoring proof is missing")
    registration = load(control / "preparation_registration.json")
    for name in ("planned_indices", "control_cap_usd", "expected_control_records",
                 "parent_freeze_sha256", "recovery_freeze_sha256", "remaining_before_control_usd",
                 "selection_uses_answers_or_model_outputs", "registered_at_unix"):
        require(registration[name] == frozen[name], "Pre-label control registration differs from freeze")
    require(registration["registered_at_unix"] <= frozen["frozen_at_unix"], "Control freeze predates its registration")
    parent = load(formal / "freeze.json")
    child = load(recovery / "freeze.json")
    codes = frozen["code_sha256"]
    require(len(codes) >= 58 and all(codes.get(k) == v for k, v in parent["code_sha256"].items())
            and codes.get("scripts/benchmark_longbench_openai_recovery.py") == child["recovery"]["wrapper_sha256"]
            and "scripts/benchmark_longbench_headtail_control.py" in codes,
            "Control altered the preceding algorithm or recovery code")
    for name, value in codes.items():
        rel = Path(name)
        require(not rel.is_absolute() and ".." not in rel.parts and rel.suffix == ".py", "Unsafe control source path")
        require(digest(root / rel) == sha(value)
                and digest(control / "frozen_source" / rel) == value, "A control source/snapshot changed")
    for filename, name in (
        ("inputs.jsonl", "prepared_inputs_sha256"), ("manifest.json", "manifest_sha256"),
        ("labels_guard.sb", "guard_profile_sha256"),
        ("preparation_count_audit.json", "preparation_count_audit_sha256"),
    ):
        require(digest(control / filename) == frozen[name], "Prepared head/tail artifact changed")
    manifest = load(control / "manifest.json")
    require(manifest["not_bit_identical_to_upstream_token_half"] is True
            and manifest["selection_uses_answers_or_model_outputs"] is False
            and manifest["generation_requested_during_preparation"] is False
            and [r["index"] for r in manifest["selected"]] == indices,
            "Head/tail preparation description differs")
    for source in manifest["selected"]:
        length, retained = source["original_context_chars"], source["retained_chars"]
        spans, dropped = source["source_spans"], source["dropped_middle_span"]
        require(count(length) and count(retained) and 2 <= retained < length
                and len(spans) == 2 and spans[0][0] == 0 and spans[1][1] == length,
                "Invalid lossy source spans")
        h, t = spans[0][1], length - spans[1][0]
        require(all(count(v) for span in spans for v in span)
                and h + t == retained and abs(h - t) <= 1
                and dropped == [h, length - t] and dropped[0] < dropped[1]
                and source["inserted_characters"] == 0
                and source["full_source_retained"] is False
                and source["same_trimmed_source_for_all_models"] is True,
                "The prepared control is not a balanced, lossless-of-retained-spans character head/tail")
        require(source["original_context_sha256"] == parent["source_hashes"][str(source["index"])]
                and source["id"] == parent["sample_ids"][str(source["index"])], "Control original source differs")
        sha(source["trimmed_context_sha256"]); sha(source["trimmed_prompt_sha256"])
        counts = source["direct_admission_tokens_by_model"]
        require(set(counts) == set(MODELS)
                and all(count(v) and v + 1024 <= 800_000 for v in counts.values()),
                "Prepared common head/tail source exceeds a model's exact control ceiling")
    guard_verified = None
    if expected:
        guard = load(control / "guard_verification.json")
        require(guard.get("verified") is True and guard.get("probe_read_denied") is True
                and guard.get("labels_read") is False
                and guard.get("profile_sha256") == frozen["guard_profile_sha256"],
                "Head/tail OS guard proof is missing")
        guard_verified = True
    else:
        skipped = load(control / "budget_skip.json")
        require(skipped["expected_control_records"] == 0 and skipped["planned_indices"] == []
                and skipped["generation_requested"] is False
                and Decimal(skipped["remaining_before_control_usd"]) == remaining,
                "A control budget skip was not recorded explicitly")
    # Only after the final all-phase proof may this audit's short letters be read.
    audit = load(control / "audit.json")
    freeze_hash = digest(control / "freeze.json")
    require(audit["freeze_sha256"] == freeze_hash
            and audit["parent_freeze_sha256"] == EXPECTED_FREEZE
            and audit["recovery_freeze_sha256"] == EXPECTED_RECOVERY_FREEZE
            and audit["budget_skipped"] is (expected == 0)
            and audit["planned_indices"] == indices
            and audit["not_bit_identical_to_upstream_token_half"] is True
            and audit["comparison_has_unequal_compute"] is True,
            "Scored head/tail audit identity differs")
    return {"directory": control, "freeze": frozen, "freeze_sha256": freeze_hash,
            "manifest": manifest, "audit": audit, "access": access,
            "guard_verified": guard_verified, "code_sha256": codes}


def build_public(formal, root, recovery=None, control=None):
    # Only small completion proofs may be opened before the already-scored audit.
    run = load(formal / "run_summary.json")
    require(run.get("phase") == "inference_complete"
            and run.get("planned_main_records") == 72
            and run.get("planned_probe_records") == 3
            and run.get("terminal_records") == 75 and run.get("labels_read") is False,
            "Refusing export before all 75 terminal records")
    recovery = recovery or formal.parent / "recovery"
    control = control or formal.parent / "headtail"
    headtail = headtail_context(control, formal, recovery, root)
    recovered_context = recovery_context(formal, recovery, root)
    recovery_freeze = recovered_context["freeze"]
    recovery_protocol = recovered_context["protocol"]
    original_audit = recovered_context["original_audit"]
    original_records = recovered_context["originals"]
    eligible_recovery_keys = recovered_context["eligible_keys"]
    access = load(formal / "scoring_access.json")
    require(access.get("all_75_terminal_before_label_access") is True
            and number(access.get("first_scored_at_unix")), "Unified-scoring proof is missing")
    require((formal / "audit.json").is_file(), "Scored audit is missing")
    freeze = load(formal / "freeze.json")
    freeze_hash = digest(formal / "freeze.json")
    require(freeze_hash == EXPECTED_FREEZE, "Exporter targets a different formal freeze")
    require(access["labels_sha256"] == freeze["labels_sha256"], "Label identity metadata differs")
    require(access["first_scored_at_unix"] >= freeze["frozen_at_unix"], "Scoring predates the freeze")
    require(access["first_scored_at_unix"] >= recovered_context["access"]["first_scored_at_unix"],
            "Original-trial scoring began before the all-114 recovery gate")
    require(recovered_context["access"]["first_scored_at_unix"] >= headtail["access"]["first_scored_at_unix"],
            "Main/recovery scoring began before the final head/tail completion gate")
    config = freeze["config"]
    require(config["models"] == list(MODELS) and config["reasoning"] == "none", "Registered models or effort changed")
    require(config["budget_harness"] == HARNESS_BUDGET and config["budget_direct"] == DIRECT_BUDGET,
            "Frozen budgets changed")
    require(config["cost_cap_usd"] == 50 and config["probes_per_model"] == 1, "Cost or probe protocol changed")
    source_hashes = {}
    for filename, expected in freeze["code_sha256"].items():
        path = Path(filename)
        require(not path.is_absolute() and ".." not in path.parts and path.suffix == ".py",
                "Code identity is not a repository-relative Python path")
        source_hashes[path.as_posix()] = sha(expected)
        require(digest(root / path) == expected
                and digest(formal / "frozen_source" / path) == expected,
                "Live or snapshot code changed")
    require(len(source_hashes) == 56, "Frozen source-file count changed")
    manifest = load(Path(config["manifest"]))
    require(digest(Path(config["manifest"])) == freeze["manifest_sha256"], "Manifest changed")
    require(manifest["inputs_sha256"] == freeze["inputs_sha256"]
            and manifest["labels_sha256"] == freeze["labels_sha256"], "Declared dataset identities differ")
    require(manifest.get("upstream_revision") is None and manifest.get("labels_used_for_selection") is False,
            "Upstream provenance or selection description changed")
    selected = manifest["selected"]
    indices = freeze["sample_indices"]
    require(len(indices) == len(set(indices)) == len(selected) == 12
            and [r["index"] for r in selected] == indices, "Frozen source matrix differs")
    require(len({r["context_sha256"] for r in selected}) == 12, "Duplicate full source contexts")
    guard = load(formal / "guard_verification.json")
    require(guard.get("verified") is True and guard.get("probe_read_denied") is True
            and guard.get("labels_read") is False
            and guard.get("profile_sha256") == freeze["guard_profile_sha256"], "OS guard proof missing")
    audit = recovered_context["audit"]
    require(audit["freeze_sha256"] == EXPECTED_RECOVERY_FREEZE
            and audit["parent_freeze_sha256"] == freeze_hash, "Scored composite audit identity mismatch")
    require(audit.get("not_same_compute_comparison") is True
            and audit.get("official_leaderboard_submission") is False, "Comparison scope changed")
    main, probes = audit["samples"], audit["capacity_probes"]
    expected_main = {(m, i, a) for m in MODELS for i in indices for a in ("direct", "harness")}
    expected_probes = {(m, freeze["probe_index"], "capacity_probe") for m in MODELS}
    require(len(main) == 72 and {(r["model"], r["index"], r["arm"]) for r in main} == expected_main,
            "Main matrix is incomplete or duplicated")
    require(len(probes) == 3 and {(r["model"], r["index"], r["arm"]) for r in probes} == expected_probes,
            "Capacity-probe matrix is incomplete or duplicated")
    money = recovered_context["merged_money"]
    require(money["freeze_sha256"] == freeze_hash and Decimal(str(money["cap_usd"])) == Decimal(50),
            "Monetary ledger identity changed")
    cost_groups = defaultdict(list)
    for entry in money["entries"]:
        arm, index_text = entry["id"].split(":")
        require((entry["model"], int(index_text), arm) in expected_main | expected_probes,
                "Unknown monetary reservation identity")
        if entry["status"] == "not_sent":
            require(Decimal(entry["estimated_upper_usd"]) == 0, "Unsent reservation was charged")
        else:
            cost_groups[entry["model"], int(index_text), arm].append(entry)
    records, capacity = [], []
    gold_by_index = {}
    for raw in main + probes:
        model, index, arm = raw["model"], raw["index"], raw["arm"]
        key = (model, index, arm)
        original = original_records[key]
        is_recovered = key in eligible_recovery_keys
        execution_freeze = EXPECTED_RECOVERY_FREEZE if is_recovered else freeze_hash
        require(raw["terminal"] is True and raw["freeze_sha256"] == execution_freeze
                and raw["logical_main_freeze_sha256"] == freeze_hash
                and raw["context_sha256"] == freeze["source_hashes"][str(index)]
                and raw["id"] == freeze["sample_ids"][str(index)], "Terminal identity differs")
        require(raw["outcome_origin"] == (
                    "zero_generation_network_recovery" if is_recovered else "original_trial")
                and raw["original_record_sha256"]
                    == recovery_protocol["parent_record_sha256"][record_filename(key)]
                and raw["original_status"] == original["status"]
                and raw["original_physical_generations"] == original["usage"]["physical_requests"],
                "Logical result lineage differs")
        if is_recovered:
            require(zero_generation_eligible(original)
                    and raw.get("logical_parent_physical_requests") == 0
                    and raw.get("logical_parent_charged_tokens") == 0,
                    "A recovery overlaid an original generated or charged task")
        else:
            require(all(raw.get(name) == original.get(name) for name in (
                "status", "prediction", "budget", "measurements", "usage", "ledger",
                "error", "count_audit", "response_audit", "result")),
                "An unrecovered original outcome was changed")
        expected_budget = HARNESS_BUDGET if arm == "harness" else DIRECT_BUDGET
        if arm != "capacity_probe":
            require(raw["budget"] == expected_budget, "Main-arm budget differs")
        else:
            require(raw["budget"]["max_calls"] == 1 and raw["budget"]["max_output_tokens"] == 1024,
                    "Capacity-probe submission protocol differs")
        usage = public_ledger(
            raw, config["prices"][model], cost_groups[model, index, arm],
            prior_counts=original.get("count_audit") if is_recovered else None)
        require(raw["logical_parent_plus_recovery_generations"] == usage["physical_generations"]
                and original["usage"]["charged_tokens"]
                    + (raw["usage"]["charged_tokens"] if is_recovered else 0) <= raw["budget"]["total_tokens"],
                "Parent plus recovery logical budget exceeded")
        public = {
            "index": index, "id": raw["id"], "requested_model": model, "arm": arm,
            "terminal": True, "status": enum(raw["status"]) or "unrecognized_status",
            "budget": {k: raw["budget"][k] for k in DIRECT_BUDGET},
            "measurement": measurement(raw), **usage,
            "terminal_error": safe_error(raw.get("error")),
            "actual_execution_freeze_sha256": execution_freeze,
            "logical_main_freeze_sha256": freeze_hash,
            "outcome_origin": raw["outcome_origin"],
            "original_record_sha256": raw["original_record_sha256"],
            "original_terminal_status": original["status"],
            "original_physical_generations": original["usage"]["physical_requests"],
            "original_charged_or_reserved_tokens": original["usage"]["charged_tokens"],
            "elapsed_seconds": (raw["seconds"] + (original["seconds"] if is_recovered else 0)
                                if number(raw.get("seconds")) and number(original.get("seconds")) else None),
            "provider_error_observations": safe_provider_observations(raw),
        }
        if public["terminal_error"] and public["terminal_error"]["safe_code"] == "context_length_exceeded":
            public["terminal_error"]["category"] = (
                "provider_context_rejected_within_published_admission"
                if arm == "direct" and public["measurement"]["fits_native_direct"] is True
                else "provider_context_rejected_above_or_unverified_admission")
        if arm == "capacity_probe":
            public["not_scored"] = True
            public["parseable_answer_available"] = choice(raw.get("prediction")) is not None
            require("gold" not in raw and "task_success" not in raw, "A capacity probe was scored")
            capacity.append(public)
            continue
        reference, prediction = choice(raw.get("gold")), choice(raw.get("prediction"))
        require(reference is not None, "Reference letter is missing from scored main record")
        require(index not in gold_by_index or gold_by_index[index] == reference, "Reference letters differ across arms")
        gold_by_index[index] = reference
        denied = raw["status"] == "native_admission_denied"
        available = prediction is not None and not denied
        correct = available and prediction == reference
        require(raw.get("semantic_answer_available") is available and raw.get("task_success") is correct,
                "Scored main result disagrees with its letters")
        require(raw.get("correct") is (None if denied else bool(correct)), "Semantic metric differs")
        public.update(
            prediction=prediction, reference_choice=reference,
            answer_available=available, answer_correct_when_available=bool(correct) if available else None,
            native_admission_denied=denied, task_success=bool(correct),
            controller=safe_controller(raw.get("result") or {}),
        )
        if denied:
            require(arm == "direct" and public["measurement"]["fits_native_direct"] is False
                    and usage["physical_generations"] == 0 and prediction is None, "Invalid admission-denied result")
        if arm == "harness" and raw.get("result"):
            require(public["controller"]["mode"] == "raw_source"
                    and public["controller"]["label_free"] and not public["controller"]["reused_direct"],
                    "Harness extension did not use its registered independent raw-source route")
        records.append(public)
    parent_cost_groups = defaultdict(list)
    for entry in recovered_context["parent_money"]["entries"]:
        arm, index_text = entry["id"].split(":")
        if entry["status"] != "not_sent":
            parent_cost_groups[entry["model"], int(index_text), arm].append(entry)
    first_trial_results = []
    for key, original in original_records.items():
        model, index, arm = key
        first_usage = public_ledger(
            original, config["prices"][model], parent_cost_groups[key])
        first_error = safe_error(original.get("error"))
        if key in eligible_recovery_keys:
            require(zero_generation_eligible(original), "Original zero-generation eligibility changed")
            first_error["category"] = "zero_generation_count_network_failure"
        elif first_error and first_error["safe_code"] == "context_length_exceeded":
            first_error["category"] = (
                "provider_context_rejected_within_published_admission"
                if arm == "direct" and measurement(original)["fits_native_direct"] is True
                else "provider_context_rejected_above_or_unverified_admission")
        item = {
            "index": index, "id": original["id"], "requested_model": model, "arm": arm,
            "terminal": True, "status": enum(original["status"]) or "unrecognized_status",
            "actual_execution_freeze_sha256": freeze_hash,
            "original_record_sha256": recovery_protocol["parent_record_sha256"][record_filename(key)],
            "eligible_for_zero_generation_recovery": key in eligible_recovery_keys,
            "measurement": measurement(original), "terminal_error": first_error, **first_usage,
        }
        if arm == "capacity_probe":
            item["not_scored"] = True
        else:
            reference, prediction = choice(original.get("gold")), choice(original.get("prediction"))
            require(reference is not None, "Original scored reference letter is missing")
            available = prediction is not None and original["status"] != "native_admission_denied"
            successful = available and prediction == reference
            require(original["task_success"] is successful
                    and original["semantic_answer_available"] is available,
                    "Original trial's scored short letters disagree")
            item.update(prediction=prediction, reference_choice=reference,
                        answer_available=available, task_success=bool(successful))
        first_trial_results.append(item)
    require(len(first_trial_results) == 75 and len(audit["recovery_samples"]) == 39
            and {record_key(r) for r in audit["recovery_samples"]} == eligible_recovery_keys,
            "Original/recovery artifact coverage differs")
    first_trial_usage = totals(first_trial_results)
    newly_executed = [r for r in records + capacity
                      if r["outcome_origin"] == "zero_generation_network_recovery"]
    recovered_usage = totals(newly_executed)
    # Logical recovered outcomes retain the first failed count queries. Phase
    # totals below keep only new physical/count work so both phases add exactly.
    new_phase_usage = dict(recovered_usage)
    new_phase_usage.update(
        count_queries_including_cache_hits=recovered_usage["recovery_count_queries"],
        count_api_requests=recovered_usage["recovery_count_api_attempts"],
        count_api_errors=recovered_usage["recovery_count_api_errors"],
        count_logical_api_queries=(recovered_usage["count_logical_api_queries"]
                                   - recovered_usage["original_count_api_attempts"]),
        count_logical_api_errors=(recovered_usage["count_logical_api_errors"]
                                 - recovered_usage["original_count_api_errors"]),
        original_count_queries=0, original_count_api_attempts=0, original_count_api_errors=0,
    )
    new_phase_usage["count_cache_hits"] = (
        new_phase_usage["count_queries_including_cache_hits"] - new_phase_usage["count_logical_api_queries"])
    require(all(new_phase_usage[name] >= 0 for name in SUM_FIELDS), "Negative recovery-only usage counter")
    look = {(r["requested_model"], r["index"], r["arm"]): r for r in records}
    summaries = {}
    for model in MODELS:
        summary = {}
        direct_rows = [look[model, i, "direct"] for i in indices]
        harness_rows = [look[model, i, "harness"] for i in indices]
        eligible_ids = {r["index"] for r in direct_rows if r["measurement"]["fits_native_direct"] is True}
        over_ids = {r["index"] for r in direct_rows if r["measurement"]["fits_native_direct"] is False}
        native_pairs, available_pairs = [], []
        for index in indices:
            d, h = look[model, index, "direct"], look[model, index, "harness"]
            if index in eligible_ids:
                native_pairs.append((d["task_success"], h["task_success"]))
            if d["answer_available"] and h["answer_available"]:
                available_pairs.append((d["task_success"], h["task_success"]))
        for arm, rows in (("direct", direct_rows), ("harness", harness_rows)):
            success = sum(r["task_success"] for r in rows)
            available_rows = [r for r in rows if r["answer_available"]]
            item = {
                "n_planned": 12, "n_answer_available": len(available_rows),
                "correct_answers": success, "task_success_rate_all_planned": success / 12,
                "accuracy_among_available_answers_secondary": ratio(
                    sum(r["task_success"] for r in available_rows), len(available_rows)),
                "incorrect_available_answers": sum(not r["task_success"] for r in available_rows),
                "native_admission_denied": sum(r["native_admission_denied"] for r in rows),
                "terminal_errors": sum(r["terminal_error"] is not None for r in rows),
                "status_counts": dict(Counter(r["status"] for r in rows)),
                "terminal_error_category_counts": dict(Counter(
                    r["terminal_error"]["category"] for r in rows if r["terminal_error"])),
                "provider_context_rejected": sum(
                    r["terminal_error"] is not None
                    and r["terminal_error"]["safe_code"] == "context_length_exceeded" for r in rows),
                "provider_context_rejected_within_published_admission": sum(
                    r["terminal_error"] is not None
                    and r["terminal_error"]["category"] == "provider_context_rejected_within_published_admission"
                    for r in rows),
                "infrastructure_terminal_failures": sum(
                    r["terminal_error"] is not None and r["terminal_error"]["category"] in {
                        "authentication_or_model_access", "quota", "rate_limit",
                        "server_error_or_uncertain_delivery", "transport_or_uncertain_delivery"
                    } for r in rows),
                **totals(rows),
            }
            if arm == "direct":
                eligible = [r for r in rows if r["index"] in eligible_ids]
                item.update(
                    n_native_admissible=len(eligible),
                    n_within_published_native_admission=len(eligible),
                    native_eligibility_unknown=12 - len(eligible_ids) - len(over_ids),
                    correct_native_admissible=sum(r["task_success"] for r in eligible),
                    primary_task_success_rate_native_admissible=ratio(
                        sum(r["task_success"] for r in eligible), len(eligible)),
                    failures_or_missing_native_admissible=sum(not r["answer_available"] for r in eligible),
                )
            else:
                item.update(
                    primary_task_success_rate_all_planned=success / 12,
                    descriptive_context_unit_wilson_95=wilson(success, 12),
                    caught_candidate_errors=sum(r["controller"]["caught_candidate_errors"] for r in rows),
                    caught_option_verifier_errors=sum(r["controller"]["caught_option_verifier_errors"] for r in rows),
                    caught_verifier_parse_errors=sum(r["controller"]["caught_verifier_parse_errors"] for r in rows),
                    explicit_verifier_retry_attempts=sum(r["controller"]["explicit_verifier_retry_attempts"] for r in rows),
                    returned_raw_source_results=sum(r["controller"]["mode"] == "raw_source" for r in rows),
                )
            original = audit["summary"][model][arm]
            for key in ("n_planned", "n_answer_available", "correct_answers", "task_success_rate_all_planned",
                        "native_admission_denied", "physical_generations", "charged_or_reserved_tokens"):
                require(item[key] == original[key], "Public summary differs from scored audit: " + key)
            require(same_float(item["accuracy_among_available_answers_secondary"],
                               original["accuracy_among_available_answers"]), "Conditional accuracy differs")
            if arm == "direct":
                for key in ("n_native_admissible", "native_eligibility_unknown", "correct_native_admissible",
                            "primary_task_success_rate_native_admissible"):
                    require(same_float(item[key], original[key]), "Native-admissible metric differs")
            summary[arm] = item
        paired_native = pair_table(native_pairs)
        paired_available = pair_table(available_pairs)
        for actual_pair, name in ((paired_native, "paired_native_admissible"), (paired_available, "paired_available")):
            old = audit["summary"][model][name]
            require(all(actual_pair[k] == old[k if k != "both_unsuccessful" else "both_wrong"]
                        for k in actual_pair), "Paired table differs from scored audit")
        paired_native["failed_or_missing_answers_retained"] = True
        paired_available["secondary_conditioned_on_both_answers_available"] = True
        summary["paired_native_admissible_primary"] = paired_native
        summary["paired_available_secondary"] = paired_available
        over = [r for r in harness_rows if r["index"] in over_ids]
        native_harness = [r for r in harness_rows if r["index"] in eligible_ids]
        length_by_index = {r["index"]: r["measurement"]["full_prompt_admission_input_tokens"] for r in direct_rows}
        summary["above_native_harness"] = {
            "definition": "Actual native-Direct full-prompt admission input plus 1024 output exceeds 1050000",
            "n": len(over), "indices": sorted(over_ids),
            "correct": sum(r["task_success"] for r in over),
            "task_success_rate": ratio(sum(r["task_success"] for r in over), len(over)),
            "answer_available": sum(r["answer_available"] for r in over),
            "descriptive_context_unit_wilson_95": wilson(sum(r["task_success"] for r in over), len(over)),
            "longest_correct": size_result([r for r in over if r["task_success"]], length_by_index),
        }
        summary["harness_on_native_admissible"] = {
            "n": len(native_harness), "correct": sum(r["task_success"] for r in native_harness),
            "task_success_rate": ratio(sum(r["task_success"] for r in native_harness), len(native_harness)),
        }
        provider_rejected_ids = {
            r["index"] for r in direct_rows
            if r["terminal_error"] is not None
            and r["terminal_error"]["safe_code"] == "context_length_exceeded"
        }
        rejected_harness = [r for r in harness_rows if r["index"] in provider_rejected_ids]
        provider_rejected_rows = [r for r in direct_rows if r["index"] in provider_rejected_ids
                                  and count(length_by_index.get(r["index"]))]
        smallest_rejection = (min(provider_rejected_rows, key=lambda r: length_by_index[r["index"]])
                              if provider_rejected_rows else None)
        summary["harness_on_provider_context_rejected_sources"] = {
            "n": len(rejected_harness), "indices": sorted(provider_rejected_ids),
            "correct": sum(r["task_success"] for r in rejected_harness),
            "task_success_rate": ratio(sum(r["task_success"] for r in rejected_harness), len(rejected_harness)),
            "provider_rejections_are_actual_http_context_length_exceeded_not_local_admission_denials": True,
        }
        summary["observed_capacity"] = {
            "unit": "Exact complete native-Direct prompt admission, including formatting and safety margin",
            "maximum_attempted": size_result(harness_rows, length_by_index),
            "maximum_with_harness_answer": size_result([r for r in harness_rows if r["answer_available"]], length_by_index),
            "maximum_with_correct_harness_answer": size_result([r for r in harness_rows if r["task_success"]], length_by_index),
            "maximum_native_admissible": size_result([r for r in direct_rows if r["index"] in eligible_ids], length_by_index),
            "maximum_with_direct_answer": size_result([r for r in direct_rows if r["answer_available"]], length_by_index),
            "maximum_with_correct_direct_answer": size_result([r for r in direct_rows if r["task_success"]], length_by_index),
            "maximum_with_known_direct_processing": size_result(
                [r for r in direct_rows if r["known_usage_tokens"] > 0], length_by_index),
            "minimum_with_actual_direct_context_rejection": size_result(
                [smallest_rejection] if smallest_rejection else [], length_by_index),
            "not_a_proven_absolute_limit": True,
        }
        summary["capacity_probes"] = totals([p for p in capacity if p["requested_model"] == model])
        observations = [event for r in records + capacity if r["requested_model"] == model
                        for event in r["provider_error_observations"]]
        summary["provider_context_numeric_observations"] = {
            "unique_reported_maximum_context_tokens": sorted({
                event["unique_numeric_fields"]["maximum_context_length_tokens"]
                for event in observations if "maximum_context_length_tokens" in event.get("unique_numeric_fields", {})
            }),
            "all_reported_maximum_context_candidates": sorted({
                value for event in observations
                for value in event.get("numeric_candidates", {}).get("maximum_context_length_tokens", [])
            }),
            "generation_error_unique_reported_maximum_context_tokens": sorted({
                event["unique_numeric_fields"]["maximum_context_length_tokens"]
                for event in observations
                if event["stage"] == "generation"
                and "maximum_context_length_tokens" in event.get("unique_numeric_fields", {})
            }),
            "numeric_error_events": sum(bool(event.get("numeric_candidates")) for event in observations),
            "reported_error_fields_are_not_an_independent_service_limit_measurement": True,
        }
        summaries[model] = summary
    all_results = records + capacity
    usage_all = totals(all_results)
    require(usage_all["physical_generations"] == len(
        [e for e in money["entries"] if e["status"] != "not_sent"]), "Global monetary/generation count mismatch")
    upper = sum((Decimal(e["estimated_upper_usd"]) for e in money["entries"]), Decimal(0))
    require(Decimal(money["estimated_upper_usd"]) == upper
            == Decimal(audit["estimated_generation_cost_upper_usd"])
            == Decimal(usage_all["estimated_generation_cost_upper_usd"]) <= Decimal(50),
            "Global conservative cost does not conserve within cap")
    require(Decimal(first_trial_usage["estimated_generation_cost_upper_usd"])
                == recovered_context["parent_upper"]
            and Decimal(recovered_usage["estimated_generation_cost_upper_usd"])
                == recovered_context["new_upper"],
            "Phase-specific public cost totals differ")
    for name in SUM_FIELDS:
        require(usage_all[name] == first_trial_usage[name] + new_phase_usage[name],
                "Original plus recovery physical usage does not conserve")
    require(usage_all["original_count_queries"] == first_trial_usage["count_queries_including_cache_hits"]
            and usage_all["original_count_api_attempts"] == first_trial_usage["count_api_requests"]
            and usage_all["recovery_count_api_attempts"] == recovered_usage["recovery_count_api_attempts"],
            "Original failed count attempts were lost or duplicated")
    preflight_path = formal.parent / "preflight.json"
    preflight = load(preflight_path)
    require(digest(preflight_path) == manifest["preflight_sha256"], "Preflight artifact changed")
    smoke = preflight["preflight"]
    require(len(smoke) == 3 and {r["model"] for r in smoke} == set(MODELS)
            and all(r["ok"] and r["model"] == r["actual_model"] for r in smoke), "Availability probe metadata differs")
    smoke_input = sum(r["usage"]["prompt_tokens"] for r in smoke)
    smoke_output = sum(r["usage"]["completion_tokens"] for r in smoke)
    require(smoke_input == 249 and smoke_output == 15, "Availability preflight token accounting differs")
    sources = []
    for item in selected:
        sources.append({
            "index": item["index"], "id": item["id"], "domain": item["domain"],
            "sub_domain": item["sub_domain"], "difficulty": item["difficulty"], "length_category": item["length"],
            "context_chars": item["context_chars"], "context_sha256": sha(item["context_sha256"]),
            "official_prompt_sha256": sha(item["prompt_sha256"]),
            "reference_choice": gold_by_index[item["index"]],
            "selection_exact_sol_direct_admission_input_tokens":
                item["preflight_native_direct"]["direct_admission_input_tokens"],
        })
    payload = {
        "schema_version": 1, "experiment_id": "longbench_openai_limits_20261102",
        "experiment_date_local": datetime.fromtimestamp(
            freeze["frozen_at_unix"], ZoneInfo("America/Los_Angeles")).date().isoformat(),
        "client_timezone": "America/Los_Angeles", "frozen_at_utc": utc(freeze["frozen_at_unix"]),
        "scored_at_utc": utc(recovered_context["access"]["first_scored_at_unix"]),
        "status": "all_114_terminal_unified_scored_registered_zero_generation_recovery",
        "requested_models": list(MODELS),
        "n_distinct_source_contexts": 12, "main_model_source_arm_records": 72,
        "capacity_probe_records": 3, "logical_terminal_outcomes": 75,
        "original_terminal_records": 75, "new_recovery_terminal_records": 39,
        "all_terminal_records": 114, "freeze_sha256": freeze_hash,
        "logical_main_freeze_sha256": freeze_hash,
        "recovery_freeze_sha256": EXPECTED_RECOVERY_FREEZE,
        "algorithm": {
            "name": "CRU-Harness LongBench multiview raw-evidence controller with option verification",
            "path": "existing_longbench_evidence_controller_with_responses_budget_bridge",
            "controller_selection_and_voting_changed_during_run": False,
            "oolong_v32_structured_fact_path_used": False,
            "direct_result_reused_by_harness": False,
            "evidence_configuration": {
                "chunk_chars": 48000, "candidate_count": 3, "max_windows": 8,
                "window_chars": 4000, "evidence_max_chars": 28000,
                "candidate_max_tokens": 2048, "option_verify_tokens": 2048,
                "explicit_option_verifier_retry_limit": 1,
            },
        },
        "benchmark": {
            "name": "LongBench v2", "official_project": "https://longbench2.github.io/",
            "official_repository": "https://github.com/THUDM/LongBench",
            "dataset": "zai-org/LongBench-v2", "dataset_url": "https://huggingface.co/datasets/zai-org/LongBench-v2",
            "dataset_config": "default", "split": "train", "upstream_revision": None,
            "upstream_revision_limitation": "The original local HF rows cache did not record a revision",
            "raw_cache_sha256_declared": sha(manifest["cache_sha256"]),
            "inputs_sha256": sha(freeze["inputs_sha256"]), "labels_sha256": sha(freeze["labels_sha256"]),
            "manifest_sha256": sha(freeze["manifest_sha256"]),
            "source_selection": manifest["selection_rule"],
            "regression_set": True, "fresh_blind_holdout": False,
            "natural_unmodified_sources": True, "artificial_padding_used": False,
            "selected_domain_counts": dict(Counter(item["domain"] for item in selected)),
            "not_full_503_question_score": True, "official_leaderboard_submission": False,
        },
        "budgets": {"native_direct": DIRECT_BUDGET, "harness": HARNESS_BUDGET,
                    "capacity_probe": "One explicitly submitted above-native request per model with a separate enlarged submission ceiling"},
        "wire_parameters": freeze["wire_parameters"],
        "transport_encoding": {
            **freeze["transport_encoding"], "initial_single_string_limit_chars": 10485760,
            "initial_wire_field_error": "string_above_max_length",
            "text_parts_are_a_wire_fix_not_context_expansion": True,
        },
        "model_specifications": {
            "published_native_context_tokens_per_model": NATIVE, "official_model_urls": MODEL_URLS,
            "published_specification_is_not_verified_service_acceptance": True,
        },
        "metric_definitions": {
            "task_success": "A parseable final A/B/C/D equals the frozen reference letter; failure or missing output is unsuccessful",
            "native_direct_primary": "Correct task delivery among sources within published native admission, retaining provider context rejections, infrastructure failures, and missing outputs; not pure semantic accuracy",
            "native_direct_all12": "Operational task success over all 12, with admission-denied cases separately labeled, not wrong semantic answers",
            "harness_primary": "Correct final choice over all 12 sources, including errors and missing outputs",
            "paired_primary": "Matched native-admissible sources, retaining failures on either arm",
            "paired_available_secondary": "Conditional on both arms returning an answer; not the primary comparison",
            "capacity_probe": "Separate API capacity/transport outcome; never receives a semantic score",
            "length": "Responses input-envelope counts; full admission includes budget notice and 16 safety tokens, unlike raw-text tokenization",
            "coverage": "Bounded evidence retrieval does not imply every source token was read by the model",
        },
        "summary": summaries, "sources": sources, "results": records, "capacity_probes": capacity,
        "usage_all_including_probes": usage_all,
        "original_infrastructure_incomplete_trial": {
            "terminal_records": 75, "infrastructure_incomplete": True,
            "zero_generation_count_network_failures": 39,
            "summary": original_audit["summary"],
            "status_counts": audit["first_trial_status_counts"],
            "usage_including_original_probes": first_trial_usage,
            "results_including_unscored_probes": first_trial_results,
            "scored_only_after_all_114_records_terminal": True,
            "not_an_independent_method_or_tuning_trial": True,
        },
        "registered_zero_generation_recovery": {
            "original_records_preserved": 75, "new_terminal_records": 39,
            "selection_registered_before_any_label_access": True,
            "selection_used_answers_or_correctness": False,
            "eligibility": recovery_protocol["eligibility"],
            "original_generated_or_charged_tasks_retried": 0,
            "generation_retries": 0, "parent_generation_budgets_reset": False,
            "count_max_attempts_per_network_call": 3, "count_retry_delays_seconds": [5, 15],
            "new_generation_cost_cap_usd": recovery_protocol["new_generation_cost_cap_usd"],
            "recovery_frozen_at_utc": utc(recovery_freeze["frozen_at_unix"]),
            "recovery_status_counts": audit["recovery_status_counts"],
            "new_physical_generation_usage_including_probes": new_phase_usage,
            "selected_cases": [{
                "requested_model": job["model"], "index": job["index"], "arm": job["arm"],
                "context_sha256": job["context_sha256"],
                "original_record_sha256": job["parent_record_sha256"],
                "original_status": original_records[record_key(job)]["status"],
                "original_physical_generations": 0, "original_charged_or_reserved_tokens": 0,
                "original_count_queries": len(original_records[record_key(job)].get("count_audit") or []),
            } for job in recovery_protocol["execution_order"]],
            "count_attempt_metric": "Physical count requests include all original misses and recovery retry attempts; cache hits and logical queries are separately counted",
        },
        "cost": {
            "currency": "USD", "definition": freeze["cost_definition"], "cap_usd": "50",
            "estimated_generation_cost_upper_usd": str(upper),
            "parent_generation_cost_upper_usd": str(recovered_context["parent_upper"]),
            "recovery_generation_cost_upper_usd": str(recovered_context["new_upper"]),
            "separate_preflight_cost_upper_usd": str(recovered_context["preflight_upper"]),
            "combined_with_preflight_upper_usd": audit["combined_cost_with_preflight_upper_usd"],
            "global_cap_includes_separate_preflight_reservation": True,
            "price_usd_per_million": config["prices"], "official_price_sources": MODEL_URLS,
            "admission": "(admitted input * max(ordinary,cached,default-30m write) + reserved output * output rate) / 1e6",
            "long_input": "Above 272000 admitted input: whole-request input/cache rates x2 and output rate x1.5",
            "known_usage": "Actual input/output settled at admitted envelope rates",
            "unknown_usage": "Full monetary and token reservations retained; not known actual token usage or an invoice",
            "default_cache_ttl": "30m", "cache_retention_overridden": False,
        },
        "separate_preflight": {
            "availability_probe_models": list(MODELS), "availability_generations": 3,
            "known_input_tokens": smoke_input, "known_output_tokens": smoke_output,
            "known_total_tokens": smoke_input + smoke_output,
            "included_in_formal_scores_or_cost_ledger": False,
            "count_attempts": len(preflight["counts"]),
            "count_failures": sum(r.get("ok") is False for r in preflight["counts"]),
            "count_requests_are_not_generation_usage": True,
            "preflight_sha256": digest(preflight_path),
        },
        "integrity": {
            "complete_72_main_plus_3_probes": True,
            "all_75_terminal_before_label_access": True, "scoring_recomputed_from_short_choices": True,
            "all_75_original_and_39_recovery_terminal_before_label_access": True,
            "all_original_generated_outcomes_preserved_without_rerun": True,
            "zero_generation_recovery_lineage_verified": True,
            "token_and_monetary_ledgers_recomputed": True,
            "known_response_model_and_default_tier_verified": True,
            "unknown_usage_model_identity_not_inferred": True,
            "os_guard_probe_denied": True, "os_guard_is_not_full_access_forensics": True,
            "frozen_source_files": 56, "live_and_snapshot_source_hashes_match": True,
            "base_sources_match_both_phase_snapshots": True,
            "additional_recovery_wrapper_source_files": 1,
            "recovery_wrapper_source_sha256": {
                "scripts/benchmark_longbench_openai_recovery.py": recovery_protocol["wrapper_sha256"]},
            "source_sha256": source_hashes, "local_scored_audit_sha256": digest(formal / "audit.json"),
            "local_recovered_composite_audit_sha256": digest(recovery / "audit.json"),
            "local_cost_ledger_sha256": digest(formal / "cost_ledger.json"),
            "local_recovery_cost_ledger_sha256": digest(recovery / "cost_ledger.json"),
            "local_scoring_access_sha256": digest(formal / "scoring_access.json"),
            "local_recovery_scoring_access_sha256": digest(recovery / "scoring_access.json"),
            "exporter_sha256": digest(Path(__file__)), "raw_artifacts_distributed": False,
        },
        "limitations": [
            "Previously consumed LongBench regression sources; no claim of a new blind holdout or unknown pretraining exposure",
            "12 length-biased sources across six domains; not a full 503 score or an official leaderboard submission",
            "Different native-Direct and bounded-Harness compute ceilings; this is not a same-compute accuracy comparison",
            "One model run per source/arm; 72 main records are not 72 independent sources",
            "Distinct source IDs and hashes do not exclude underlying record overlap; intervals are descriptive context-unit summaries",
            "Maximum attempted, answered, and correct sizes are different observations, not a proven absolute context limit",
            "The model attention window is unchanged; observed extension is system-level task execution over external sources",
            "Long ordered text parts repair a transport field limit without removing provider context limits",
            "The LongBench evidence route differs from the V3.2 OOLONG structured-fact path",
            "Provider-rejected probes can lack usage; retained reservations are conservative accounting, not known incurred tokens",
            "Published-window admission and actual provider acceptance can disagree; actual HTTP context rejections are reported separately",
            "The first trial had 39 zero-generation count-network failures; original partial outcomes are preserved separately from the registered recovery overlay",
            "Recovery changed count-only retry orchestration, not source selection, evidence controller, or model generation policies",
        ],
    }
    ensure_public(payload)
    payload = append_headtail(payload, headtail, root, formal, recovery)
    return append_posthoc_audits(payload, root, formal.parent)


def compatible_choice(text):
    matches = re.findall(r"(?:correct\s+answer\s+is|answer\s+is|answer\s*:|choice)\s*\(?([ABCD])\)?",
                         str(text or ""), flags=re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    matches = re.findall(r"\(([ABCD])\)", str(text or ""), flags=re.IGNORECASE)
    return matches[-1].upper() if matches else None


def strict_official_choice(text):
    clean = str(text or "").replace("*", "").strip()
    for pattern in (r"The correct answer is \(([A-D])\)", r"The correct answer is ([A-D])"):
        match = re.search(pattern, clean)
        if match:
            return match.group(1)
    return None


def append_headtail(p, h, root, formal, recovery):
    """Add the distinct lossy-control arm without changing main trial results."""
    frozen, audit, directory = h["freeze"], h["audit"], h["directory"]
    require(audit["first_trial_audit_sha256"] == digest(formal / "audit.json")
            and audit["recovery_composite_audit_sha256"] == digest(recovery / "audit.json"),
            "Head/tail comparison points to different main audits")
    require(Decimal(frozen["parent_cost_upper_usd"]) == Decimal(p["cost"]["parent_generation_cost_upper_usd"])
            and Decimal(frozen["recovery_cost_upper_usd"]) == Decimal(p["cost"]["recovery_generation_cost_upper_usd"])
            and Decimal(frozen["preflight_cost_upper_usd"]) == Decimal(p["cost"]["separate_preflight_cost_upper_usd"]),
            "Control was registered against different preceding costs")
    planned = frozen["planned_indices"]
    expected = {(model, index) for index in planned for model in MODELS}
    raw_rows = audit["samples"]
    require(len(raw_rows) == frozen["expected_control_records"]
            and {(r["model"], r["index"]) for r in raw_rows} == expected,
            "The conditional head/tail result matrix differs")
    ledger = load(directory / "cost_ledger.json")
    require(ledger["freeze_sha256"] == h["freeze_sha256"]
            and Decimal(ledger["cap_usd"]) == Decimal(frozen["control_cap_usd"]),
            "Head/tail monetary ledger identity differs")
    groups = defaultdict(list)
    for entry in ledger["entries"]:
        arm, index = entry["id"].split(":")
        require(arm == "headtail_control" and (entry["model"], int(index)) in expected,
                "Unexpected head/tail monetary reservation")
        if entry["status"] == "not_sent":
            require(Decimal(entry["estimated_upper_usd"]) == 0, "Unsent control reservation was charged")
        else:
            groups[entry["model"], int(index)].append(entry)
    sources = {r["index"]: r for r in h["manifest"]["selected"]}
    reference = {r["index"]: r["reference_choice"] for r in p["sources"]}
    rows = []
    for raw in raw_rows:
        model, index = raw["model"], raw["index"]
        source = sources[index]
        require(raw["terminal"] is True and raw["freeze_sha256"] == h["freeze_sha256"]
                and raw["parent_freeze_sha256"] == EXPECTED_FREEZE
                and raw["recovery_freeze_sha256"] == EXPECTED_RECOVERY_FREEZE
                and raw["arm"] == "headtail_control" and raw["budget"] == HEADTAIL_BUDGET
                and raw["id"] == source["id"]
                and raw["context_sha256"] == source["trimmed_context_sha256"]
                and raw["original_context_sha256"] == source["original_context_sha256"]
                and raw["source_spans"] == source["source_spans"], "Control terminal source/budget identity differs")
        derived = {**raw, "ledger_error_count": sum(e.get("status") == "error" for e in raw["ledger"])}
        usage = public_ledger(derived, p["cost"]["price_usd_per_million"][model],
                              groups[model, index], phase="headtail")
        predicted, strict = compatible_choice(raw.get("content")), strict_official_choice(raw.get("content"))
        require(raw.get("prediction") == predicted and raw.get("strict_official_prediction") == strict
                and raw.get("strict_format_accepted") is (strict is not None),
                "Control parsing differs from actual returned text")
        gold = choice(raw.get("gold"))
        require(gold == reference[index]
                and raw["correct"] is (predicted == gold)
                and raw["strict_official_correct"] is (strict == gold), "Control short-letter scores differ")
        m = raw.get("measurements") or {}
        admission = m.get("full_prompt_admission_input_tokens")
        if admission is not None:
            require(count(admission) and admission + 1024 <= 800_000
                    and m.get("prepared_admission_input_tokens")
                        == source["direct_admission_tokens_by_model"][model]
                    and m.get("whole_prompt_fits_control_budget") is True,
                    "Control exact prompt admission differs")
        error = safe_error(raw.get("error"))
        if error and error["safe_code"] == "context_length_exceeded":
            error["category"] = "provider_context_rejected_within_headtail_ceiling"
        rows.append({
            "index": index, "id": raw["id"], "requested_model": model, "arm": "headtail_control",
            "terminal": True, "status": enum(raw["status"]) or "unrecognized_status",
            "actual_execution_freeze_sha256": h["freeze_sha256"],
            "original_context_sha256": source["original_context_sha256"],
            "trimmed_context_sha256": source["trimmed_context_sha256"],
            "source_spans": source["source_spans"], "dropped_middle_span": source["dropped_middle_span"],
            "budget": HEADTAIL_BUDGET, "prediction": predicted, "reference_choice": gold,
            "task_success": predicted == gold, "answer_available": predicted is not None,
            "strict_official_prediction": strict, "strict_official_correct": strict == gold,
            "strict_format_accepted": strict is not None, "terminal_error": error,
            "prepared_admission_input_tokens": source["direct_admission_tokens_by_model"][model],
            "actual_admission_input_tokens": admission,
            "provider_error_observations": safe_provider_observations(raw),
            **usage,
        })
    usage = totals(rows)
    upper = sum((Decimal(e["estimated_upper_usd"]) for e in ledger["entries"]), Decimal(0))
    require(upper == Decimal(ledger["estimated_upper_usd"])
            == Decimal(audit["estimated_control_cost_upper_usd"])
            == Decimal(usage["estimated_generation_cost_upper_usd"])
            <= Decimal(frozen["control_cap_usd"]), "Head/tail cost does not conserve")
    main_usage = p["usage_all_including_probes"]
    combined = totals(p["results"] + p["capacity_probes"] + rows)
    for key in SUM_FIELDS:
        require(combined[key] == main_usage[key] + usage[key], "All-phase usage does not conserve")
    total_cost = Decimal(p["cost"]["estimated_generation_cost_upper_usd"]) + upper
    total_with_preflight = total_cost + Decimal(p["cost"]["separate_preflight_cost_upper_usd"])
    require(total_cost == Decimal(combined["estimated_generation_cost_upper_usd"])
            and total_with_preflight == Decimal(audit["combined_cost_with_preflight_upper_usd"]) <= 50,
            "All three phases plus preflight exceed the shared cap")
    summaries = {}
    for model in MODELS:
        group = [r for r in rows if r["requested_model"] == model]
        harness = {r["index"]: r for r in p["results"]
                   if r["requested_model"] == model and r["arm"] == "harness" and r["index"] in planned}
        correct = sum(r["task_success"] for r in group)
        strict_correct = sum(r["strict_official_correct"] for r in group)
        summary = {
            "n_planned": len(group), "primary_correct": correct,
            "primary_task_delivery_rate_all_planned": ratio(correct, len(group)),
            "strict_official_correct": strict_correct,
            "strict_official_rate_all_planned": ratio(strict_correct, len(group)),
            "compatible_answer_available": sum(r["answer_available"] for r in group),
            "strict_format_accepted": sum(r["strict_format_accepted"] for r in group),
            "harness_same_sources_correct": sum(r["task_success"] for r in harness.values()),
            "status_counts": dict(Counter(r["status"] for r in group)),
            "paired_with_harness_compatible_letters": pair_table([
                (r["task_success"], harness[r["index"]]["task_success"]) for r in group]),
            **totals(group),
        }
        original = audit["summary"][model]
        for key in ("n_planned", "primary_correct", "strict_official_correct", "strict_format_accepted",
                    "harness_same_sources_correct", "physical_generations", "charged_or_reserved_tokens",
                    "status_counts"):
            require(summary[key] == original[key], "Head/tail model summary differs: " + key)
        require(same_float(summary["primary_task_delivery_rate_all_planned"],
                           original["primary_accuracy_all_planned"]), "Head/tail primary rate differs")
        summaries[model] = summary
        p["summary"][model]["headtail_control"] = summary
    public_sources = [{
        "index": s["index"], "id": s["id"],
        "original_context_sha256": s["original_context_sha256"],
        "trimmed_context_sha256": s["trimmed_context_sha256"],
        "trimmed_prompt_sha256": s["trimmed_prompt_sha256"],
        "original_context_chars": s["original_context_chars"], "retained_chars": s["retained_chars"],
        "retained_character_fraction": s["retained_chars"] / s["original_context_chars"],
        "source_spans": s["source_spans"], "dropped_middle_span": s["dropped_middle_span"],
        "inserted_characters": 0, "same_trimmed_source_for_all_models": True,
        "prepared_admission_tokens_by_model": s["direct_admission_tokens_by_model"],
    } for s in h["manifest"]["selected"]]
    prep = load(directory / "preparation_count_audit.json")
    preparation_counts = {}
    for model, data in prep.items():
        require(model in MODELS, "Unknown model in control preparation counts")
        attempts = data.get("physical_count_attempts") or []
        preparation_counts[model] = {
            "logical_queries_including_cache": len(data.get("count_audit") or []),
            "physical_http_attempts": len(attempts),
            "retry_attempts": sum(a.get("attempt", 1) > 1 for a in attempts),
            "failed_attempts": sum(a.get("status") == "error" for a in attempts),
            "generation_requested": False,
        }
    p["usage_main_and_recovery_including_probes"] = main_usage
    p["usage_all_including_probes"] = combined
    p["cost"].update(
        main_and_recovery_generation_cost_upper_usd=p["cost"]["estimated_generation_cost_upper_usd"],
        headtail_generation_cost_upper_usd=str(upper),
        estimated_generation_cost_upper_usd=str(total_cost),
        combined_with_preflight_upper_usd=str(total_with_preflight),
    )
    p["headtail_quality_control"] = {
        "status": "budget_skipped" if not rows else "completed",
        "experiment": audit["experiment"], "freeze_sha256": h["freeze_sha256"],
        "frozen_at_utc": utc(frozen["frozen_at_unix"]),
        "registered_at_utc": utc(frozen["registered_at_unix"]),
        "control_kind": audit["control_kind"], "budget": HEADTAIL_BUDGET,
        "planned_indices": planned, "n_planned_records": frozen["expected_control_records"],
        "remaining_before_control_usd": frozen["remaining_before_control_usd"],
        "control_cost_cap_usd": frozen["control_cap_usd"],
        "selection_rule": "Money only: remaining >=13 selects 143+260; >=6.5 selects 143; otherwise explicit budget skip",
        "selection_uses_answers_or_model_outputs": False,
        "no_original_model_answer_rerun": True, "generation_retries": 0,
        "sources": public_sources, "results": rows, "summary": summaries, "usage": usage,
        "count_only_preparation": preparation_counts,
        "not_bit_identical_to_upstream_token_half": True,
        "upstream_pred_commit": frozen["upstream_pred_commit"],
        "upstream_pred_url": frozen["upstream_pred_url"],
        "upstream_code_pin_is_not_dataset_revision": True,
        "comparison_has_unequal_compute": True,
        "compatible_primary_parser": "Local compatible A/B/C/D extraction, same as the main controller",
        "strict_secondary_parser": "Actual head/tail response with stars removed, then the two capitalized The correct answer is patterns; bare A alone is not accepted",
        "main_harness_metric_is_compatible_controller_choice_not_inferred_raw_strict_output": True,
        "code_sha256": h["code_sha256"],
        "prepared_inputs_sha256": frozen["prepared_inputs_sha256"],
        "manifest_sha256": frozen["manifest_sha256"],
        "scoring_access_sha256": digest(directory / "scoring_access.json"),
        "scored_audit_sha256": digest(directory / "audit.json"),
        "cost_ledger_sha256": digest(directory / "cost_ledger.json"),
        "guard_probe_denied": h["guard_verified"],
    }
    p["status"] = "all_phases_terminal_unified_scored_with_recovery_and_conditional_headtail"
    p["headtail_freeze_sha256"] = h["freeze_sha256"]
    p["main_and_recovery_terminal_records"] = p["all_terminal_records"]
    p["additional_headtail_terminal_records"] = len(rows)
    p["all_terminal_records"] += len(rows)
    p["logical_outcomes_including_auxiliary"] = p["logical_terminal_outcomes"] + len(rows)
    p["final_scoring_started_at_utc"] = utc(h["access"]["first_scored_at_unix"])
    p["integrity"]["all_114_and_conditional_headtail_terminal_before_label_access"] = True
    p["integrity"]["headtail_all_phase_shared_cost_cap_verified"] = True
    prep_attempts = sum(v["physical_http_attempts"] for v in preparation_counts.values())
    prep_queries = sum(v["logical_queries_including_cache"] for v in preparation_counts.values())
    p["count_request_scopes"] = {
        "runtime_original_recovery_and_control_physical_http_attempts": combined["count_api_requests"],
        "runtime_logical_queries_including_cache": combined["count_queries_including_cache_hits"],
        "headtail_preparation_physical_http_attempts": prep_attempts,
        "headtail_preparation_logical_queries_including_cache": prep_queries,
        "separate_selection_preflight_recorded_count_attempts": p["separate_preflight"]["count_attempts"],
        "all_recorded_count_http_attempts": (combined["count_api_requests"] + prep_attempts
                                             + p["separate_preflight"]["count_attempts"]),
        "count_requests_are_not_generation_token_usage": True,
    }
    p["limitations"].append(
        "The head/tail quality control uses only the fixed longest one or two sources and a larger 800k prompt budget; its small-sample scores are not pooled into the main 12-source comparison")
    ensure_public(p)
    return p


def append_posthoc_audits(p, root, directory):
    """Publish safe replay receipts; never replace any primary choice or score."""
    trajectory_path = directory / "trajectory_audit_v2.json"
    receipts_path = directory / "strict_choices_private_v2.json"
    trajectory = load(trajectory_path)
    receipts = load(receipts_path)
    cost_path = root / "benchmarks/longbench_openai_score_audit_20261102.json"
    cost_audit = load(cost_path)
    public_trajectory = root / "benchmarks/longbench_openai_trajectory_audit_20261102.json"
    require(digest(public_trajectory) == digest(trajectory_path), "Public trajectory audit differs")
    require(trajectory["freeze_sha256"] == EXPECTED_FREEZE
            and trajectory["recovery_freeze_sha256"] == EXPECTED_RECOVERY_FREEZE
            and trajectory["headtail_freeze_sha256"] == EXPECTED_HEADTAIL_FREEZE
            and trajectory["violations"] == []
            and trajectory["status"] == "pass_available_audits", "Trajectory audit identity or status differs")
    require(trajectory["all_75_original_plus_39_recovery_terminal_and_scored_before_audit"] is True
            and trajectory["all_registered_auxiliary_controls_terminal_and_scored_before_audit"] is True
            and trajectory["labels_or_scored_gold_objects_opened_by_auditor"] is False
            and trajectory["provider_or_tokenizer_calls_made"] == 0
            and trajectory["semantic_entailment_or_label_truth_verified"] is False,
            "Post-hoc audit scope changed")
    require(receipts["contains_raw_reply_or_gold"] is False
            and receipts["created_after_final_score_gates"] is True,
            "Strict receipts have an unsafe scope")
    totals_raw = trajectory["totals"]
    require(len(trajectory["records"]) == totals_raw["all_phase_records"] == 120
            and len(trajectory["logical_records"]) == 75
            and len(trajectory["auxiliary_headtail_records"]) == 6
            and len(receipts["records"]) == 81, "Post-hoc receipt matrix differs")
    history = trajectory["records"]
    replay_passed = sum(r["trajectory"]["status"] == "passed" for r in history)
    replay_unavailable = sum(r["trajectory"]["status"] == "unavailable" for r in history)
    require(totals_raw["trajectory_passed"] == replay_passed
            and totals_raw["trajectory_unavailable"] == replay_unavailable
            and replay_passed + replay_unavailable == len(history), "Trajectory-status phase totals do not conserve")
    for field in ("known_responses_replayed", "unknown_delivery_errors_replayed", "generation_entries_replayed",
                  "count_events_replayed", "responses_wire_request_hashes_checked", "wire_prompt_hashes_checked"):
        require(all(count(r["trajectory"].get(field, 0)) for r in history)
                and totals_raw[field] == sum(r["trajectory"].get(field, 0) for r in history),
                "Trajectory physical-history total does not match per-record sums: " + field)
    require(totals_raw["known_responses_replayed"] + totals_raw["unknown_delivery_errors_replayed"]
                == totals_raw["generation_entries_replayed"], "Replayed generation-kind partition does not conserve")
    require(all(count(r.get("recorded_count_events")) for r in history)
            and totals_raw["recorded_count_events_all_phases"] == sum(r["recorded_count_events"] for r in history)
                == p["usage_all_including_probes"]["count_queries_including_cache_hits"],
            "Recorded count-event phase totals do not conserve")
    actual = p["usage_all_including_probes"]
    require(totals_raw["ledger_passed"] == 120
            and totals_raw["physical_generation_attempts"] == actual["physical_generations"]
            and totals_raw["charged_or_reserved_tokens"] == actual["charged_or_reserved_tokens"]
            and totals_raw["unknown_usage_attempts"] == actual["unknown_usage_requests"],
            "Trajectory and monetary ledger totals disagree")
    require(cost_audit["status"] == "pass" and cost_audit["violations"] == []
            and cost_audit["physical_generations_all_phases"] == actual["physical_generations"]
            and cost_audit["known_input_tokens_all_phases"] == actual["known_input_tokens"]
            and cost_audit["known_output_tokens_all_phases"] == actual["known_output_tokens"]
            and cost_audit["charged_or_reserved_tokens_all_phases"] == actual["charged_or_reserved_tokens"]
            and cost_audit["unknown_usage_requests_all_phases"] == actual["unknown_usage_requests"]
            and Decimal(cost_audit["combined_cost_with_preflight_upper_usd"])
                == Decimal(p["cost"]["combined_with_preflight_upper_usd"]),
            "Independent public cost audit differs")
    cost_source = root / "scripts/audits/audit_longbench_openai_cost_scores.py"
    trajectory_source = root / "scripts/audits/audit_longbench_openai_trajectories.py"
    require(digest(cost_source) == cost_audit["auditor_source_sha256"]
            and digest(trajectory_source) == trajectory["auditor_sha256"], "Published auditor source differs")
    expected_rows = p["results"] + p["capacity_probes"] + p["headtail_quality_control"]["results"]
    expected = {(r["requested_model"], r["index"], r["arm"]): r for r in expected_rows}
    require(len({(r["model"], r["index"], r["arm"]) for r in receipts["records"]}) == 81,
            "Duplicate raw-reply receipt")
    raw_rows = []
    source_by_index = {s["index"]: s for s in p["sources"]}
    for r in receipts["records"]:
        key = r["model"], r["index"], r["arm"]
        require(key in expected, "Unknown raw-reply receipt")
        before, source = expected[key], source_by_index[r["index"]]
        raw_available = r["raw_reply_available"] is True
        provenance = r["selected_provenance"]
        kind = enum(provenance["kind"])
        require(kind is not None and r["id"] == before["id"] and r["record_status"] == before["status"]
                and r["actual_record_freeze_sha256"] == before["actual_execution_freeze_sha256"]
                and r["original_context_sha256"] == source["context_sha256"]
                and choice(r["actual_controller_choice"]) == before.get("prediction"),
                "Raw-reply receipt identity or primary choice differs")
        require(r["source_context_sha256"] == before.get("trimmed_context_sha256", source["context_sha256"]),
                "Raw-reply source binding differs")
        if raw_available:
            raw_hash = sha(r["raw_reply_sha256"])
            if r["arm"] == "harness":
                require(kind == "fixed_actual_selected_candidate_raw_reply"
                        and provenance["strict_parser_used_to_select_candidate"] is False,
                        "Strict extraction reselected a Harness candidate")
        else:
            require(r["raw_reply_sha256"] is None and r["strict_choice"] is None
                    and r["local_raw_reply_choice"] is None, "Unavailable raw reply acquired a choice")
            raw_hash = None
        strict, local = choice(r["strict_choice"]), choice(r["local_raw_reply_choice"])
        is_probe = r["arm"] == "capacity_probe"
        reference = None if is_probe else before["reference_choice"]
        raw_rows.append({
            "index": r["index"], "id": r["id"], "requested_model": r["model"], "arm": r["arm"],
            "status": before["status"], "actual_execution_freeze_sha256": r["actual_record_freeze_sha256"],
            "original_context_sha256": r["original_context_sha256"],
            "raw_reply_available": raw_available, "raw_reply_sha256": raw_hash,
            "availability": "traceable_raw_reply" if raw_available else "unavailable_raw_reply",
            "primary_controller_choice": choice(r["actual_controller_choice"]),
            "selected_raw_compatible_choice": local, "selected_raw_strict_choice": strict,
            "strict_format_accepted": (strict is not None) if raw_available else None,
            "reference_choice": reference,
            "strict_correct_on_traceable_raw_reply": (strict == reference) if raw_available and not is_probe else None,
            "raw_strict_agrees_with_primary": (strict == before.get("prediction")) if raw_available else None,
            "selected_provenance": {
                "kind": kind,
                "candidate_index": provenance.get("candidate_index") if count(provenance.get("candidate_index")) else None,
                "physical_request": provenance.get("physical_request") if count(provenance.get("physical_request")) else None,
                "strict_parser_used_to_select_candidate": False if r["arm"] == "harness" and raw_available else None,
            },
            "not_scored": is_probe,
        })
    summaries = {}
    for model in MODELS:
        summaries[model] = {}
        for arm in ("direct", "harness", "headtail_control"):
            group = [r for r in raw_rows if r["requested_model"] == model and r["arm"] == arm]
            available = [r for r in group if r["raw_reply_available"]]
            strict_correct = sum(r["strict_correct_on_traceable_raw_reply"] is True for r in available)
            primary_group = [r for r in expected_rows if r["requested_model"] == model and r["arm"] == arm]
            summaries[model][arm] = {
                "n_planned": len(group), "raw_replies_traceable": len(available),
                "raw_replies_unavailable": len(group) - len(available),
                "strict_format_accepted_on_traceable": sum(r["strict_format_accepted"] is True for r in available),
                "strict_correct_on_traceable": strict_correct,
                "strict_accuracy_on_traceable_only_secondary": ratio(strict_correct, len(available)),
                "primary_correct_all_planned": sum(r["task_success"] for r in primary_group),
                "strict_vs_primary_disagreements_on_traceable": sum(r["raw_strict_agrees_with_primary"] is False for r in available),
            }
    logical_unavailable = []
    for r in trajectory["logical_records"]:
        if r["trajectory"]["status"] == "unavailable":
            logical_unavailable.append({
                "requested_model": r["model"], "index": r["index"], "id": r["id"], "arm": r["arm"],
                "reason_code": enum(r["trajectory"].get("reason_code")),
                "actual_record_sha256": sha(r["actual_record_sha256"]),
                "ledger_passed": r["ledger_passed"] is True,
            })
    require(len(logical_unavailable) == totals_raw["logical_trajectories_unavailable"] == 1
            and logical_unavailable[0]["requested_model"] == "gpt-6-sol"
            and logical_unavailable[0]["index"] == 251, "Replay limitation changed")
    traceable = [r for r in raw_rows if r["raw_reply_available"]]
    require(len(traceable) == trajectory["strict_parser_summary"]["raw_replies_available"] == 46
            and sum(r["strict_format_accepted"] is True for r in traceable)
                == trajectory["strict_parser_summary"]["official_choice_available"] == 46,
            "Raw strict-parser summary differs")
    p["posthoc_audits"] = {
        "computed_after_all_scoring": True,
        "changes_primary_choices_or_scores": False,
        "independent_cost_score_audit": {
            "repository_path": "benchmarks/longbench_openai_score_audit_20261102.json",
            "sha256": digest(cost_path), "status": "pass",
            "auditor_repository_path": "scripts/audits/audit_longbench_openai_cost_scores.py",
            "auditor_sha256": digest(cost_source),
        },
        "trajectory_audit": {
            "repository_path": "benchmarks/longbench_openai_trajectory_audit_20261102.json",
            "sha256": digest(trajectory_path), "auditor_sha256": digest(trajectory_source),
            "auditor_repository_path": "scripts/audits/audit_longbench_openai_trajectories.py",
            "status": "pass_available_audits_with_unavailable_trajectories",
            "physical_history_records": 120, "ledgers_passed": 120,
            "trajectories_replayed": totals_raw["logical_trajectories_passed"] + totals_raw["auxiliary_trajectories_passed"],
            "trajectories_unavailable": totals_raw["trajectory_unavailable"],
            "main_logical_trajectories_replayed": totals_raw["logical_trajectories_passed"],
            "main_logical_trajectories_unavailable": totals_raw["logical_trajectories_unavailable"],
            "auxiliary_trajectories_replayed": totals_raw["auxiliary_trajectories_passed"],
            "preserved_original_zero_generation_failures": totals_raw["preserved_original_zero_generation_failures"],
            "source_bound_harness_trajectories_replayed": totals_raw["harness_source_view_trajectories_passed"],
            "wire_request_hashes_checked": totals_raw["responses_wire_request_hashes_checked"],
            "physical_generation_entries_replayed": totals_raw["generation_entries_replayed"],
            "known_generation_responses_replayed": totals_raw["known_responses_replayed"],
            "unknown_delivery_errors_replayed": totals_raw["unknown_delivery_errors_replayed"],
            "recorded_runtime_count_events": totals_raw["recorded_count_events_all_phases"],
            "runtime_count_events_replayed": totals_raw["count_events_replayed"],
            "source_bindings_checked": totals_raw["source_bindings_checked"],
            "semantic_entailment_or_reference_truth_verified": False,
            "reuses_frozen_production_selection_parsers_and_controller": True,
            "auditor_model_or_count_api_calls": 0,
            "unavailable_main_logical_cases": logical_unavailable,
        },
        "strict_selected_raw_reply_diagnostic": {
            "scope": "Post-hoc exact-pattern extraction from the original fixed selected raw reply; not an official leaderboard score or a replacement for controller-choice accuracy",
            "strict_parser_used_to_select_candidate": False,
            "receipts_sha256": digest(receipts_path),
            "official_parser_provenance": receipts["official_parser_provenance"],
            "n_receipts_including_unscored_probes": 81,
            "raw_replies_traceable": len(traceable), "raw_replies_unavailable": 81 - len(traceable),
            "strict_format_accepted_on_traceable": sum(r["strict_format_accepted"] is True for r in traceable),
            "strict_vs_primary_disagreements_on_traceable": sum(r["raw_strict_agrees_with_primary"] is False for r in traceable),
            "missing_raw_reply_is_format_failure": False,
            "summary": summaries, "receipts": raw_rows,
        },
    }
    p["limitations"].append(
        "Full trajectory replay is unavailable for 39 preserved original zero-generation failures and Sol Harness index 251 because the original count exception payload was not recorded; all 120 token/money ledgers pass, but unavailable trajectories are not marked passed")
    p["limitations"].append(
        "Strict raw-reply parsing is a retrospective diagnostic on 46 traceable replies; the one missing Harness raw replay is unavailable, not a format failure, and primary controller-choice results remain unchanged")
    ensure_public(p)
    return p


def frac(k, n):
    return f"{k}/{n} ({100*k/n:.2f}%)" if n else "N/A (n=0)"


def percent(value):
    return "N/A" if value is None else f"{100*value:.2f}%"


def money_text(value):
    return f"USD {Decimal(value):.6f}"


def max_text(value):
    return "N/A" if value is None else f'{value["full_native_direct_admission_input_tokens"]:,} (index {value["index"]})'


def official_model_links():
    return " · ".join(f"[{model}]({MODEL_URLS[model]})" for model in MODELS)


def render(p, chinese=False):
    zh = chinese
    lines = [
        "# LongBench v2：原生直连与有界 Harness 极长上下文对照" if zh else
        "# LongBench v2: native Direct versus bounded Harness at extreme lengths",
        "",
        "**简体中文** | [English](longbench_openai_limits_20261102.md)" if zh else
        "**English** | [简体中文](longbench_openai_limits_20261102.zh-CN.md)",
        "",
        ("本实验在12个未填充的自然LongBench来源上，分别运行三个精确模型的完整原生直连和有界证据Harness，共72条主逻辑结果，另有3条独立容量探针。第一阶段75条终态中有39条计数网络失败、没有发送任何生成；随后单独冻结这39条的恢复协议，新增39条执行记录。之后按预登记的长度和剩余资金条件运行独立首尾截断质量对照。全部原始、恢复及条件对照终态齐备（或明确预算跳过）后，才统一读取标签评分。"
         if zh else
         "This experiment pairs full native-context Direct with bounded-evidence Harness on 12 unpadded natural LongBench sources using three exact models: 72 main logical outcomes plus 3 separate capacity probes. The first 75 terminal records included 39 count-network failures with no generation sent. A separately frozen recovery added 39 execution records for those tasks. A distinct head/tail quality control followed the preregistered length-and-money rule. Labels were first opened only after every original, recovery, and conditional-control outcome was terminal or an explicit budget skip was recorded."),
        "",
        ("这是已消费来源的长度偏置回归评测，不是新的盲测、全503题成绩或官方榜单。两条路径使用不同计算预算，不能解释为同计算量准确率提升。"
         if zh else
         "This is a length-biased regression evaluation on previously consumed sources, not a new blind holdout, full 503 score, or official leaderboard result. The arms have different compute budgets; the comparison does not establish same-compute accuracy uplift."),
        "",
        "## 原始不完整试验与登记恢复" if zh else "## Original incomplete trial and registered recovery",
        "",
        ("下表先保留第一阶段的正确交付结果，再给出恢复完成后的主表，避免把基础设施恢复藏成算法准确率提升。39条选择仅依据已冻结的generic-network错误、空生成账本、零物理生成及零令牌费用，不读取输出或正确性。所有已经发送生成的原始任务均未重跑。"
         if zh else
         "The first-trial delivery results are preserved below before the recovered main table, so infrastructure recovery is not presented as an algorithmic accuracy gain. The 39 cases were selected only by the frozen generic-network error, empty generation ledger, zero physical generations, and zero token charges. Selection did not use outputs or correctness. No originally generated task was rerun."),
        "",
        ("| 模型 | 原始Direct正确交付／12 | 原始Harness正确交付／12 | 登记恢复任务数 | 原始生成请求 | 新增恢复生成请求 |"
         if zh else
         "| Model | Original Direct correct delivery /12 | Original Harness correct delivery /12 | Registered recovery tasks | Original generation calls | New recovery generation calls |"),
        "|---|---:|---:|---:|---:|---:|",
    ]
    first = p["original_infrastructure_incomplete_trial"]
    recovery = p["registered_zero_generation_recovery"]
    for model in MODELS:
        initial = first["summary"][model]
        selected_count = sum(r["requested_model"] == model for r in recovery["selected_cases"])
        original_calls = sum(r["physical_generations"] for r in first["results_including_unscored_probes"]
                             if r["requested_model"] == model)
        new_calls = sum(r["physical_generations"] for r in p["results"] + p["capacity_probes"]
                        if r["requested_model"] == model
                        and r["outcome_origin"] == "zero_generation_network_recovery")
        lines.append(f'| {model} | {frac(initial["direct"]["correct_answers"],12)} | '
                     f'{frac(initial["harness"]["correct_answers"],12)} | {selected_count} | '
                     f'{original_calls} | {new_calls} |')
    lines += ["",
        ("| 模型 | Direct恢复indices | Harness恢复indices | 独立探针恢复indices |"
         if zh else
         "| Model | Recovered Direct indices | Recovered Harness indices | Recovered probe indices |"),
        "|---|---|---|---|",
    ]
    for model in MODELS:
        cells = []
        for arm in ("direct", "harness", "capacity_probe"):
            values = sorted(r["index"] for r in recovery["selected_cases"]
                            if r["requested_model"] == model and r["arm"] == arm)
            cells.append(", ".join(str(v) for v in values) or "none")
        lines.append("| " + model + " | " + " | ".join(cells) + " |")
    lines += ["",
        ("原始75条记录和新增39条均不可变。恢复只增加了计数网络请求的最多3次尝试及5／15秒等待、错误数字字段观测；不做生成重试，不修改56份原算法／控制器源码，不重置已消费生成预算。原始39条均为零消耗，因此原任务与恢复的合计仍受同一逻辑预算限制。"
         if zh else
         "The original 75 and new 39 records remain immutable. Recovery added count-only network retries up to 3 attempts with 5/15-second delays and safe numeric error observations. It did not retry generations, modify the 56 original algorithm/controller files, or reset consumed generation budgets. Each selected original task had zero consumption, so parent plus recovery remains within the same logical task budget."),
        "",
        "## 恢复后的主结果与容量拒绝" if zh else "## Recovered main outcomes and capacity denials",
        "",
        ("| 模型 | Direct全部12任务正确交付 | Direct本地容量拒绝 | Direct公开规格准入组正确交付 | Harness全部12正确 | 超公开原生窗子组Harness正确 |"
         if zh else
         "| Model | Direct task success /all 12 | Direct local capacity denials | Direct correct delivery /published-admissible | Harness correct /all 12 | Harness above-published-native subgroup |"),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        s = p["summary"][model]
        d, h, over = s["direct"], s["harness"], s["above_native_harness"]
        lines.append(f'| {model} | {frac(d["correct_answers"],12)} | {d["native_admission_denied"]} | '
                     f'{frac(d["correct_native_admissible"],d["n_native_admissible"])} | '
                     f'{frac(h["correct_answers"],12)} | {frac(over["correct"],over["n"])} |')
    lines += ["",
        ("Direct的全部12题列衡量任务交付，容量拒绝没有语义答案，不标为模型答错。公开规格准入组的主指标保留服务端上下文拒绝、其他API失败、缺失和错误答案在分母中，属于正确交付率而非纯语义准确率。Harness主指标保留全部12题。"
         if zh else
         "The Direct/all-12 column measures operational task success. Admission-denied cases have no semantic answer and are not labeled wrong model answers. The published-admissible primary denominator retains provider context rejections, other API failures, missing outputs, and wrong answers. It is correct-delivery rate, not pure semantic accuracy. Harness retains all 12."),
        "",
        "### Direct拒绝与基础设施失败分解" if zh else "### Direct rejection and infrastructure breakdown",
        "",
        ("| 模型 | 本地超窗拒绝 | 服务端context拒绝 | 其中公开规格准入范围内 | 基础设施终态失败 | 准入未知 | 实际返回答案 |"
         if zh else
         "| Model | Local admission denials | Provider context rejections | Rejected within published admission | Infrastructure terminal failures | Eligibility unknown | Actual returned choices |"),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        d = p["summary"][model]["direct"]
        lines.append(f'| {model} | {d["native_admission_denied"]} | {d["provider_context_rejected"]} | '
                     f'{d["provider_context_rejected_within_published_admission"]} | '
                     f'{d["infrastructure_terminal_failures"]} | {d["native_eligibility_unknown"]} | '
                     f'{d["n_answer_available"]} |')
    lines += ["",
        ("按105万公开规格精确计数可准入，不等于服务端实际接受。若服务端在该范围内返回context_length_exceeded，仍保留原始error状态并单列，不能写成语义答错、本地容量拒绝或已验证105万窗口可用。429和未知网络交付也分别归类。"
         if zh else
         "Exact admission under the published 1.05M specification does not establish actual provider acceptance. A context_length_exceeded response within that range retains its original error status and is listed separately, not as a wrong semantic answer, a local denial, or proof that the 1.05M service window works. Rate/quota 429 and uncertain network delivery are separate categories."),
        "",
        "## 原生可容纳来源的配对比较" if zh else "## Paired comparison on native-admissible sources",
        "",
        ("| 模型 | 主配对n | 双方正确 | 仅Direct正确 | 仅Harness正确 | 双方未成功 | 双方有答案条件配对n（次要） |"
         if zh else
         "| Model | Primary paired n | Both correct | Direct only | Harness only | Both unsuccessful | Both-answers-available n (secondary) |"),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        s = p["summary"][model]
        a, b = s["paired_native_admissible_primary"], s["paired_available_secondary"]
        lines.append(f'| {model} | {a["n"]} | {a["both_correct"]} | {a["direct_only_correct"]} | '
                     f'{a["harness_only_correct"]} | {a["both_unsuccessful"]} | {b["n"]} |')
    lines += ["",
        ("双方有答案的条件统计仅作为次要结果，避免丢弃失败后放大准确率。样本少且每路径只运行一次，不能据此确立稳定排名；容量之外的错误和基础设施限制需分别解释。"
         if zh else
         "Conditioning on both answers being available is secondary, to avoid inflating accuracy by dropping failures. Few sources and one run per arm do not establish stable rankings. Wrong answers, capacity limits, and infrastructure failures are different outcomes."),
        "",
        ("| 模型 | Direct实际context拒绝来源数 | 同批来源Harness正确交付 |"
         if zh else
         "| Model | Sources actually context-rejected by Direct | Harness correct delivery on those sources |"),
        "|---|---:|---:|",
    ]
    for model in MODELS:
        group = p["summary"][model]["harness_on_provider_context_rejected_sources"]
        lines.append(f'| {model} | {group["n"]} | {frac(group["correct"],group["n"])} |')
    lines += ["",
        ("该表只纳入实际HTTP context_length_exceeded，区别于因公开规格超窗而未发送生成的8题容量组。探针另列且不参与此表。"
         if zh else
         "This table includes actual HTTP context_length_exceeded outcomes only, separate from the above-specification group whose Direct generations were not submitted. The separately registered probes are not scored in this table."),
        "",
        "## 已观察到的长度范围" if zh else "## Observed length frontier",
        "",
        ("| 模型 | Direct最大实际返回答案提示 | Direct最短实际上下文拒绝提示 | Harness最大尝试提示 | Harness最大返回答案提示 | Harness最大正确提示 | 最长正确／公开原生窗 | 最长正确／16k工作窗 |"
         if zh else
         "| Model | Largest prompt with an actual Direct answer | Smallest actual Direct context rejection | Largest Harness attempt | Largest with a Harness answer | Largest correct Harness prompt | Longest correct /published-native | Longest correct /16k workspace |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        c = p["summary"][model]["observed_capacity"]
        correct = c["maximum_with_correct_harness_answer"]
        native_ratio = "N/A" if correct is None else f'{correct["relative_to_native_window"]:.2f}x'
        work_ratio = "N/A" if correct is None else f'{correct["relative_to_harness_request_ceiling"]:.2f}x'
        lines.append(f'| {model} | {max_text(c["maximum_with_direct_answer"])} | '
                     f'{max_text(c["minimum_with_actual_direct_context_rejection"])} | {max_text(c["maximum_attempted"])} | '
                     f'{max_text(c["maximum_with_harness_answer"])} | '
                     f'{max_text(correct)} | {native_ratio} | {work_ratio} |')
    lines += ["",
        ("长度单位是完整原生Direct提示的Responses准入输入，含资源说明和16-token安全余量；它不是无格式原文token数。105万原生窗口与16,384实验工作窗是两个不同分母。Direct实际成功与拒绝值只描述此次采样区间，不是精确硬上限；错误消息没有报告数值上限。最大尝试、返回答案和正确答案分别列出；单个多选题答对不能证明稳定理解整个来源，也不证明已找到绝对极限。"
         if zh else
         "Lengths are complete native-Direct Responses admission inputs, including the resource notice and 16 safety tokens, not raw-text tokenization. The 1,050,000 native window and 16,384 experimental workspace are different denominators. Actual Direct acceptance and rejection describe this sampled interval, not a precise hard limit; the errors reported no numeric maximum. Attempted, answered, and correct sizes are separate. A correct multiple-choice answer does not establish reliable understanding of the entire source or an absolute limit."),
        "",
        ("三模型原生规格来源：" if zh else "Official model-window sources: ") + official_model_links(),
        "",
        "## 三条独立API容量探针" if zh else "## Three separate API capacity probes",
        "",
        ("| 模型 | 来源index／准入input tokens | 实际状态 | 安全HTTP/code | 生成错误中报告的context上限 | 物理请求 | 已知tokens | 未知usage预留tokens | 保守费用上界 |"
         if zh else
         "| Model | Source index /admitted input tokens | Actual status | Safe HTTP/code | Context maximum reported in generation error | Physical calls | Known tokens | Unknown-usage reserved tokens | Conservative upper cost |"),
        "|---|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in p["capacity_probes"]:
        error = r["terminal_error"] or {}
        code = f'{error.get("http_status") or "N/A"}/{error.get("safe_code") or "N/A"}'
        maxima = sorted({
            event["unique_numeric_fields"]["maximum_context_length_tokens"]
            for event in r["provider_error_observations"] if event["stage"] == "generation"
            and "maximum_context_length_tokens" in event.get("unique_numeric_fields", {})
        })
        reported_maximum = ", ".join(f"{value:,}" for value in maxima) or "not_reported"
        admitted = r["measurement"]["full_prompt_admission_input_tokens"]
        admitted_text = f"{admitted:,}" if count(admitted) else "N/A"
        lines.append(f'| {r["requested_model"]} | {r["index"]} / {admitted_text} | {r["status"]} | {code} | '
                     f'{reported_maximum} | {r["physical_generations"]} | {r["known_usage_tokens"]:,} | '
                     f'{r["unknown_usage_reserved_tokens"]:,} | {money_text(r["estimated_generation_cost_upper_usd"])} |')
    lines += ["",
        ("探针只核验提供方接受／拒绝和传输结果，不参与准确率。HTTP400只有明确context_length_exceeded才解释为上下文拒绝；429为限流／配额，普通400或传输错误不冒充上下文上限。恢复包装器只保留白名单错误码、参数与数字候选，错误正文不公开；报告中的上限是提供方错误消息给出的数值，不等于独立实测极限。没有usage时保留整额预留，不把预留tokens写成实际推理用量。"
         if zh else
         "Probes test provider acceptance/rejection and transport only and receive no accuracy score. Only explicit context_length_exceeded establishes a context rejection; 429 is a rate/quota event, and generic 400 or transport failures do not establish a context boundary. The recovery observer retains whitelisted codes, parameters, and numeric candidates without error prose. Reported maxima are numbers in provider errors, not an independently measured absolute limit. Missing usage retains a full reservation, not known actual inference usage."),
        "",
        "## 用量与费用" if zh else "## Usage and estimated cost",
        "",
        ("| 模型／路径 | 生成请求 | 已知输入 | 已知输出 | 未知usage请求 | 未知预留tokens | 账本错误请求 | 费用估计上界 |"
         if zh else
         "| Model /arm | Generation calls | Known input | Known output | Unknown-usage calls | Unknown reserved tokens | Ledger error calls | Estimated upper cost |"),
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for arm in ("direct", "harness", "capacity_probes", "headtail_control"):
            s = p["summary"][model][arm]
            lines.append(f'| {model} / {arm} | {s["physical_generations"]} | {s["known_input_tokens"]:,} | '
                         f'{s["known_output_tokens"]:,} | {s["unknown_usage_requests"]} | '
                         f'{s["unknown_usage_reserved_tokens"]:,} | {s["ledger_error_requests"]} | '
                         f'{money_text(s["estimated_generation_cost_upper_usd"])} |')
    all_usage = p["usage_all_including_probes"]
    lines += ["",
        (f'原始、恢复与条件首尾对照各阶段含探针共{all_usage["physical_generations"]}次生成请求，已知输入加输出{all_usage["known_usage_tokens"]:,} tokens，另保留未知usage额度{all_usage["unknown_usage_reserved_tokens"]:,} tokens。各阶段正式生成保守费用上界为{money_text(p["cost"]["estimated_generation_cost_upper_usd"])}；另加预检后合计{money_text(p["cost"]["combined_with_preflight_upper_usd"])}，共同低于USD50全局上限。这些是保守估算，不是实际账单。'
         if zh else
         f'The original, recovery, and conditional head/tail stages including probes made {all_usage["physical_generations"]} generation requests, with {all_usage["known_usage_tokens"]:,} known input/output tokens and {all_usage["unknown_usage_reserved_tokens"]:,} separately reserved unknown-usage tokens. Their formal generation-cost upper estimate is {money_text(p["cost"]["estimated_generation_cost_upper_usd"])}; adding preflight gives {money_text(p["cost"]["combined_with_preflight_upper_usd"])}, within the shared USD 50 global cap. These are conservative estimates, not an invoice.'),
        "",
        ("| 运行阶段（不含准备与预检计数） | 物理生成 | 逻辑计数查询（含缓存） | 实际计数HTTP尝试 | 计数重试尝试 | 生成费用估计上界 |"
         if zh else
         "| Runtime stage | Physical generations | Logical count queries including cache | Physical count HTTP attempts | Count retry attempts | Generation-cost upper estimate |"),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, stage in (
        ("original_infrastructure_incomplete", first["usage_including_original_probes"]),
        ("registered_recovery_new_work", recovery["new_physical_generation_usage_including_probes"]),
        ("headtail_quality_control", p["headtail_quality_control"]["usage"]),
        ("combined", all_usage),
    ):
        lines.append(f'| {name} | {stage["physical_generations"]} | '
                     f'{stage["count_queries_including_cache_hits"]} | {stage["count_api_requests"]} | '
                     f'{stage["count_retry_attempts"]} | {money_text(stage["estimated_generation_cost_upper_usd"])} |')
    lines += ["",
        ("该运行阶段表保留原始失败计数尝试及恢复的逐次计数重试；它不包含单列的候选筛选预检与首尾裁剪准备。下表仅汇总本次实验已记录的计数请求范围，不代表项目全部历史计数。计数请求不占生成token。"
         if zh else
         "This runtime table retains original failed count attempts and every recovery count retry. It excludes the separate selection preflight and the head/tail trimming preparation, which are listed below. The aggregate covers recorded requests in this experiment, not the project's entire counting history. Count requests are not generation-token usage."),
        "",
        ("| 计数范围 | 实际计数HTTP尝试 |" if zh else "| Count-request scope | Physical count HTTP attempts |"),
        "|---|---:|",
        f'| runtime_original_recovery_and_control | {p["count_request_scopes"]["runtime_original_recovery_and_control_physical_http_attempts"]} |',
        f'| headtail_trimming_preparation | {p["count_request_scopes"]["headtail_preparation_physical_http_attempts"]} |',
        f'| separate_candidate_selection_preflight | {p["count_request_scopes"]["separate_selection_preflight_recorded_count_attempts"]} |',
        f'| all_recorded_count_attempts | {p["count_request_scopes"]["all_recorded_count_http_attempts"]} |',
        "",
        ("每次生成前按输入的普通／缓存读／默认30分钟写入最高单价及输出单价持久化预留。输入超过272k时，整条请求输入／缓存单价乘2、输出乘1.5。已知usage按准入价格包络结算，未知usage保留整额；缓存和reasoning细分不重复加到token总量。"
         if zh else
         "Before generation, monetary admission reserves the maximum ordinary/cached/default 30-minute-write input price and the output price. Above 272k input, whole-request input/cache prices double and output prices multiply by 1.5. Known usage settles at the admitted envelope; unknown usage retains the reservation. Cache and reasoning subsets are not added twice."),
        "",
        ("冻结单价与规格来源：" if zh else "Frozen pricing and specification sources: ") + official_model_links(),
        "",
        ("正式实验之外，三次API可用性预检共249输入＋15输出＝264tokens。计数API尝试及初始字段大小错误单列；它们不是生成token用量。"
         if zh else
         "Separate from the formal run, three availability probes used 249 input + 15 output = 264 tokens. Counting attempts and the initial field-size failure are separate; they are not generation-token usage."),
        "",
        "## 控制器捕获的内部错误" if zh else "## Controller-caught internal errors",
        "",
        ("| 模型 | Harness终态错误 | 账本错误请求 | 捕获候选错误 | 捕获选项verifier错误 | 其中解析错误 | 显式verifier重试 |"
         if zh else
         "| Model | Harness terminal errors | Ledger error requests | Caught candidate errors | Caught option-verifier errors | Parse errors among them | Explicit verifier retries |"),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        h = p["summary"][model]["harness"]
        lines.append(f'| {model} | {h["terminal_errors"]} | {h["ledger_error_requests"]} | '
                     f'{h["caught_candidate_errors"]} | {h["caught_option_verifier_errors"]} | '
                     f'{h["caught_verifier_parse_errors"]} | {h["explicit_verifier_retry_attempts"]} |')
    lines += ["",
        ("控制器可在候选或验证器出错后继续交付答案，因此终态无错误不代表内部所有请求成功。以物理账本为调用依据；明确的恢复调用和重试共同计入请求、token和费用预算。同一失败可能同时出现在账本与控制器捕获计数中，两者不可相加为独立失败数。"
         if zh else
         "A controller can still deliver an answer after a candidate or verifier failure, so a clean terminal status does not imply every internal call succeeded. The physical ledger is authoritative; explicit recoveries and retries share request, token, and monetary budgets. One failure can appear in both ledger and caught-error counters, so these are not additive counts of independent failures."),
        "",
        "## 逐来源结果" if zh else "## Per-source results",
        "",
        ("每格为Direct预测／Harness预测，括号为正确标记；容量拒绝显示denied而非答错。A/B/C/D仅是已公开基准的短选项字母，不含选项正文。"
         if zh else
         "Each cell is Direct/Harness predicted choice, with correctness marks. Capacity denial is labeled denied, not an incorrect answer. A/B/C/D are short derived benchmark choice letters without option text."),
        "",
        "| Index / ID | Domain | Reference | gpt-6-sol | gpt-6-luna | gpt-6.1-terra |",
        "|---|---|---|---|---|---|",
    ]
    lookup = {(r["requested_model"], r["index"], r["arm"]): r for r in p["results"]}
    def answer_text(r):
        if r["native_admission_denied"]:
            return "denied"
        if r["terminal_error"] and r["terminal_error"]["safe_code"] == "context_length_exceeded":
            return "provider_rejected"
        if r["terminal_error"] and r["terminal_error"]["http_status"] == 429:
            return "rate_or_quota_limited"
        if not r["answer_available"]:
            return "missing"
        return r["prediction"] + (" ✓" if r["task_success"] else " ✗")
    for source in p["sources"]:
        cells = []
        for model in MODELS:
            cells.append(answer_text(lookup[model, source["index"], "direct"]) + " / "
                         + answer_text(lookup[model, source["index"], "harness"]))
        lines.append(f'| {source["index"]} / {source["id"]} | {source["domain"]} | '
                     f'{source["reference_choice"]} | ' + " | ".join(cells) + " |")
    lines += ["",
        "## 来源、协议与解释边界" if zh else "## Provenance, protocol, and interpretation",
        "",
        ("来源为[LongBench v2官网](https://longbench2.github.io/)、[官方代码](https://github.com/THUDM/LongBench)及[官方数据](https://huggingface.co/datasets/zai-org/LongBench-v2)。HF split名为train。原始缓存没有记录上游revision，本报告保留null并公布缓存、输入和逐来源哈希；不事后补造版本。"
         if zh else
         "Sources: the [official LongBench v2 project](https://longbench2.github.io/), [official code](https://github.com/THUDM/LongBench), and [official dataset](https://huggingface.co/datasets/zai-org/LongBench-v2). The HF split is named train. The original cache did not record an upstream revision; the report retains null and publishes cache/input/per-source hashes rather than inventing a revision."),
        "",
        ("完整直连使用1,050,000单次与累计上限、1次生成、1,024输出上限。Harness使用16,384单次、65,536累计、32次生成、2,048单次输出上限；三候选证据视图及选项支持／反对／无证据验证共享账本。所有模型reasoning=none、store=false、truncation=disabled、default服务档，不发送temperature或seed。不做隐式重试，也不复用Direct答案。"
         if zh else
         "Native Direct has a 1,050,000 per-request/cumulative ceiling, one generation, and 1,024 output ceiling. Harness has 16,384 per request, 65,536 cumulative, 32 generations, and 2,048 output per call. Three evidence views and support/against/unknown option verification share the ledger. All models use reasoning=none, store=false, truncation=disabled, default tier, with no temperature or seed sent. There are no implicit retries or reused Direct answers."),
        "",
        ("API初检发现单字符串10,485,760字符字段限制；长字符串改为同一消息中按原顺序的input_text片段，每片最多10,000,000字符，拼接后与原提示完全相同。计数与生成使用同一无损编码。该修复只解决传输字段限制，不增加模型原生上下文；本轮没有人工重复或填充来源。"
         if zh else
         "Preflight found a 10,485,760-character single-string field limit. Longer text becomes ordered input_text parts within the same message, each at most 10,000,000 characters; concatenation exactly preserves the original prompt. Counting and generation use the same lossless encoding. This repairs a transport field limit, not native context capacity. No source was artificially repeated or padded."),
        "",
        ("本轮运行的是LongBench多窗口证据与选项验证路径，区别于OOLONG的V3.2结构化事实归约路径。完整来源保留在外部，模型只接收有限证据；输出正确不等于模型逐token阅读全部来源。12个长度偏置来源、一次运行和多选题偶然性限制了质量边界的解释。"
         if zh else
         "This uses the LongBench multiview-evidence/option-verification path, distinct from OOLONG V3.2 structured-fact reduction. Full sources remain external and the model receives bounded evidence; a correct answer does not mean the model read every source token. Twelve length-biased sources, a single generated outcome per logical arm, and multiple-choice chance limit quality-frontier conclusions."),
        "",
        "Freeze SHA-256: " + p["freeze_sha256"],
        "Recovery freeze SHA-256: " + p["recovery_freeze_sha256"],
        "",
        ("56份原算法源码、恢复及首尾对照包装器指纹、原始75条结果、39条恢复资格及记录来源、恢复后的72条主逻辑结果、3条探针与6条附加对照，以及精确计数、短选项字母、已知／未知用量与费用定义见[公开JSON](../benchmarks/longbench_openai_limits_20261102.json)。原始问题、来源正文、选项正文、证据窗口、模型原始回复、凭证及本机路径不随报告分发。"
         if zh else
         "The [public JSON](../benchmarks/longbench_openai_limits_20261102.json) contains 56 original algorithm source fingerprints, recovery and head/tail wrapper fingerprints, all 75 original outcomes, the 39-case recovery lineage, 72 recovered main logical outcomes, 3 probes, 6 auxiliary outcomes, exact counts, short choice letters, known/unknown usage, and cost definitions. Raw questions, contexts, option text, evidence windows, model replies, credentials, and private paths are not distributed."),
        "",
        ("公开文件可用标准库独立核验器核对短选项计分、样本矩阵、各阶段用量与费用守恒、代码指纹及隐私字段；它不会调用API或读取私有标签。该核验不是对未公开API原始回复与访问日志的重新执行。"
         if zh else
         "A standard-library public verifier checks short-choice scoring, cohort matrices, phase usage and cost conservation, source fingerprints, and private-field exclusions. It makes no API calls and reads no private labels. This is not a replay of unpublished API responses or access logs."),
        "",
        "```bash",
        "python3 -S scripts/verify_longbench_openai_limits_results.py",
        "```",
        "",
    ]
    return "\n".join(lines) + render_headtail(p, zh) + render_posthoc(p, zh)


def render_headtail(p, chinese=False):
    zh, control = chinese, p["headtail_quality_control"]
    lines = ["", "## 附加质量对照：有损首尾截断 Direct" if zh else
             "## Supplemental quality control: lossy head/tail Direct", ""]
    lines += [
        ("该对照在开标签前，按剩余费用条件固定选择最长来源143和260（足额时），或仅143，或明确预算跳过。每题最多1次生成，完整提示输入加输出上限800,000、输出上限1,024；使用同一份按字符均衡保留的首尾文本，三模型共享保留跨度。它使用大于Harness的输入预算，是独立质量对照，不是原模型完整阅读，也不是同计算量比较。"
         if zh else
         "Before labels opened, the remaining-money rule selected the fixed longest sources 143 and 260 when affordable, only 143 otherwise, or an explicit budget skip. Each source permits one generation, an 800,000 input-plus-output ceiling, and 1,024 output. The same character-balanced prefix/suffix and source spans are shared across models. This is a distinct quality control with a larger input budget than Harness, not full-source reading or a same-compute comparison."),
        "",
    ]
    if control["status"] == "budget_skipped":
        lines += [
            ("本轮已按登记的资金规则跳过该对照，没有生成，也没有可报告的head/tail准确率；跳过不记为0%成绩。"
             if zh else
             "The registered money rule skipped this control. No generation or head/tail accuracy is reported; a skip is not 0% accuracy."), ""]
        return "\n".join(lines)
    lines += [
        ("| 模型 | 计划题数 | Head/tail兼容解析正确 | Head/tail严格原始回复正确 | Harness同批完整来源正确 | Head/tail有可解析答案 |"
         if zh else
         "| Model | Planned sources | Head/tail compatible correct | Head/tail strict actual-text correct | Harness correct on the same full sources | Parseable head/tail outputs |"),
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        s = control["summary"][model]
        n = s["n_planned"]
        lines.append(f'| {model} | {n} | {frac(s["primary_correct"],n)} | '
                     f'{frac(s["strict_official_correct"],n)} | '
                     f'{frac(s["harness_same_sources_correct"],n)} | {s["compatible_answer_available"]} |')
    lines += ["",
        ("主质量对照使用与主实验相同的本地兼容选项提取。严格列只对head/tail的实际模型回复去除星号后匹配两条区分大小写的“The correct answer is”模式；裸A不自动算严格格式合格。Harness主实验是控制器最终选项，不把该单字母再解析成所谓原始回复严格得分。两种解析口径分别列示。"
         if zh else
         "The primary quality comparison uses the same local compatible choice extraction as the main experiment. The strict column strips stars from actual head/tail model text and applies the two capitalized The correct answer is patterns; bare A is not automatically accepted. The main Harness result is its controller choice, which is not reinterpreted as a strict score of raw model text. The parser scopes are separate."),
        "",
        ("| 来源index | 原文字符 | 保留字符／比例 | 保留source spans | 丢弃中段span | 三模型准备阶段准入input tokens |"
         if zh else
         "| Source index | Original characters | Retained characters /fraction | Retained source spans | Dropped middle span | Prepared admission-input tokens by model |"),
        "|---|---:|---:|---|---|---|",
    ]
    for s in control["sources"]:
        counts = ", ".join(f'{m}: {s["prepared_admission_tokens_by_model"][m]:,}' for m in MODELS)
        lines.append(f'| {s["index"]} | {s["original_context_chars"]:,} | '
                     f'{s["retained_chars"]:,} / {100*s["retained_character_fraction"]:.2f}% | '
                     f'{s["source_spans"]} | {s["dropped_middle_span"]} | {counts} |')
    lines += ["",
        ("没有插入新字符；保留首段与尾段直接拼接，中段明确丢弃。原文、截断文本及完整截断提示哈希见JSON。该实现参考上游head/tail策略，但按字符跨度裁剪并精确计数整个请求，不是上游按token取两半的逐字节复现。"
         if zh else
         "No characters are inserted: retained prefix and suffix are concatenated and the middle is explicitly discarded. Original, trimmed-text, and complete trimmed-prompt hashes are in the JSON. This follows the upstream head/tail idea with character spans and exact whole-request counting; it is not a bit-identical reproduction of upstream token-half selection."),
        "",
        ("| 来源／模型 | 参考选项 | 兼容预测 | 严格预测 | 实际状态 | 已知tokens | 未知预留tokens | 费用估计上界 |"
         if zh else
         "| Source /model | Reference | Compatible prediction | Strict prediction | Actual status | Known tokens | Unknown reserved tokens | Upper estimated cost |"),
        "|---|---|---|---|---|---:|---:|---:|",
    ]
    for r in control["results"]:
        lines.append(f'| {r["index"]} / {r["requested_model"]} | {r["reference_choice"]} | '
                     f'{r["prediction"] or "missing"} | {r["strict_official_prediction"] or "not_accepted"} | '
                     f'{r["status"]} | {r["known_usage_tokens"]:,} | {r["unknown_usage_reserved_tokens"]:,} | '
                     f'{money_text(r["estimated_generation_cost_upper_usd"])} |')
    lines += ["",
        ("上游代码固定为[" + control["upstream_pred_commit"] + "](" + control["upstream_pred_url"]
         + ")；这是代码pin，不是数据集revision。仅1或2个来源、一次推理及丢失中段的条件限制了结论，不将附加结果拼进主12题成绩。"
         if zh else
         "The upstream code is pinned at [" + control["upstream_pred_commit"] + "](" + control["upstream_pred_url"]
         + "). This is a code pin, not a dataset revision. Only one or two sources, one generation, and the discarded middle limit the inference; these results are not pooled into the main 12-source score."),
        "",
        "Head/tail freeze SHA-256: " + control["freeze_sha256"], "",
    ]
    return "\n".join(lines)


def render_posthoc(p, chinese=False):
    zh, audits = chinese, p["posthoc_audits"]
    trajectory, strict = audits["trajectory_audit"], audits["strict_selected_raw_reply_diagnostic"]
    lines = ["", "## 完成后的轨迹与原始回复诊断" if zh else
             "## Post-run trajectory and selected-raw-reply diagnostics", ""]
    lines += [
        ("独立[费用与评分审计](../benchmarks/longbench_openai_score_audit_20261102.json)核对了各阶段预算、模型身份、已知／未知用量、费用及主评分；[轨迹审计](../benchmarks/longbench_openai_trajectory_audit_20261102.json)另核对来源绑定、请求哈希及可重放的控制器流程。它们均在最终评分后离线执行，没有新增模型或计数API调用。轨迹重放复用冻结的生产选择、解析器与控制器，不是独立的语义蕴含或参考答案真值核验。"
         if zh else
         "The independent [cost and score audit](../benchmarks/longbench_openai_score_audit_20261102.json) checks phase budgets, model identity, known/unknown usage, costs, and primary scoring. The separate [trajectory audit](../benchmarks/longbench_openai_trajectory_audit_20261102.json) checks source bindings, request hashes, and replayable controller paths. Both ran offline after final scoring, with no additional model or count API calls. Trajectory replay reuses frozen production selection, parsers, and controller; it does not independently verify semantic entailment or reference-answer truth."),
        "",
        ("| 范围 | 记录数 | 完整轨迹重放通过 | 轨迹不可确证 |" if zh else
         "| Scope | Records | Complete trajectory replay passed | Trajectory unavailable |"),
        "|---|---:|---:|---:|",
        f'| original + recovery + auxiliary history | 120 | {trajectory["trajectories_replayed"]} | {trajectory["trajectories_unavailable"]} |',
        f'| main logical outcomes including capacity probes | 75 | {trajectory["main_logical_trajectories_replayed"]} | {trajectory["main_logical_trajectories_unavailable"]} |',
        f'| auxiliary head/tail | 6 | {trajectory["auxiliary_trajectories_replayed"]} | 0 |',
        "",
        ("全部120条token／费用账本核验通过，但不能声称全部轨迹重放通过。40条不可确证轨迹包括保留的39条原始零生成计数故障，以及gpt-6-sol的Harness index251：该原始任务执行过4次生成，缺少一次计数故障的原始异常payload，因此其完整轨迹与选中原始回复无法确证。原始主结果仍为正确且保持不变；账本通过不替代缺失的轨迹证据。"
         if zh else
         "All 120 token/money ledgers pass, but not all trajectories replay completely. The 40 unavailable trajectories comprise 39 preserved original zero-generation count failures and gpt-6-sol Harness index 251. That original task made four generation attempts and lacks an original count-exception payload, so its complete trajectory and selected raw reply cannot be confirmed. Its primary recorded answer remains correct and unchanged; a passing ledger does not replace missing trajectory evidence."),
        "",
        ("严格格式诊断只作用于当时实际固定选中的完整原始回复，绝不按严格解析结果重选候选。共81份逻辑收据（含3条不评分探针）中，可追溯原始回复46份：35条Harness、5条原生Direct及6条首尾对照；46份均可由上游两条严格模式提取，且与原始控制器选项零分歧。缺少原始回复标记为不可用，不当作格式失败。下表是可追溯子集的事后敏感性分析，不替换主准确率，也不是官方榜单成绩。"
         if zh else
         "Strict-format diagnostics apply only to the complete raw reply originally selected, never to a candidate reselected by strict extraction. Of 81 logical receipts including three unscored probes, 46 raw replies are traceable: 35 Harness, five native Direct, and six head/tail. All 46 match the upstream strict patterns with zero disagreement from the recorded controller choices. Missing raw replies are unavailable, not format failures. The table is a retrospective sensitivity analysis on traceable subsets; it does not replace primary accuracy or constitute an official leaderboard score."),
        "",
        ("| 模型／路径 | 计划题数 | 可追溯原始回复 | 严格格式接受 | 严格正确／可追溯子集 | 不可追溯 | 原始主结果正确／全部计划 |"
         if zh else
         "| Model /arm | Planned | Traceable raw replies | Strict format accepted | Strict correct /traceable subset | Unavailable raw replies | Primary correct /all planned |"),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in MODELS:
        for arm in ("direct", "harness", "headtail_control"):
            s = strict["summary"][model][arm]
            lines.append(f'| {model} / {arm} | {s["n_planned"]} | {s["raw_replies_traceable"]} | '
                         f'{s["strict_format_accepted_on_traceable"]} | '
                         f'{frac(s["strict_correct_on_traceable"], s["raw_replies_traceable"])} | '
                         f'{s["raw_replies_unavailable"]} | {frac(s["primary_correct_all_planned"], s["n_planned"])} |')
    lines += ["",
        ("两种分母不能直接当作同一准确率比较。Sol Harness的7／11严格可追溯子集，与8／12主结果不同，是因为index251的原始回复不可追溯，不能据此宣称它格式失败。收据只公开短选项、原始回复哈希、来源哈希及选中位置，不分发模型正文。"
         if zh else
         "The denominators represent different subsets. Sol Harness has 7/11 correct among strict-traceable replies and 8/12 primary correct because index 251 lacks a traceable raw reply; this is not evidence of a formatting failure. Receipts publish only short choices, raw-reply hashes, source hashes, and selected positions, without model text."),
        "",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal", type=Path, default=ROOT / "work/longbench_openai_limits_20261102/formal")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--recovery", type=Path, default=ROOT / "work/longbench_openai_limits_20261102/recovery")
    parser.add_argument("--control", type=Path, default=ROOT / "work/longbench_openai_limits_20261102/headtail")
    parser.add_argument("--overwrite", action="store_true", help="Replace only the three fixed generated report paths")
    args = parser.parse_args()
    root, formal = args.root.resolve(), args.formal.resolve()
    p = build_public(formal, root, args.recovery.resolve(), args.control.resolve())
    english, chinese = render(p), render(p, chinese=True)
    ensure_public([english, chinese])
    outputs = {
        root / "benchmarks/longbench_openai_limits_20261102.json":
            json.dumps(p, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        root / "docs/longbench_openai_limits_20261102.md": english,
        root / "docs/longbench_openai_limits_20261102.zh-CN.md": chinese,
    }
    require(args.overwrite or all(not path.exists() for path in outputs),
            "Refusing to overwrite an existing report without --overwrite")
    for path, value in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(value, encoding="utf-8")
        temporary.replace(path)
    print(json.dumps({"exported": [str(p.relative_to(root)) for p in outputs],
                      "main_logical_records": 72, "unscored_capacity_probes": 3,
                      "original_records_preserved": 75, "additional_recovery_records": 39}))


if __name__ == "__main__":
    main()
