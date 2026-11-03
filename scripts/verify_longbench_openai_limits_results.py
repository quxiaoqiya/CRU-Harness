"""Verify public LongBench arithmetic and source fingerprints without private data.

Uses only Python's standard library, the public JSON/Markdown reports, and the
published repository sources. It never opens source contexts, labels, model
responses, private audits, or credentials. This verifies the published evidence;
it cannot independently reconstruct unpublished API responses or access logs.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
from decimal import Decimal
import hashlib
import json
import math
from pathlib import Path
import re


ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "long_context_agent" / "harness.py").is_file())
MODELS = ("gpt-6-sol", "gpt-6-luna", "gpt-6.1-terra")
INDICES = (143, 260, 453, 449, 438, 33, 147, 257, 395, 335, 35, 251)
FREEZES = (
    "941c86120e66020439571fbe8888999c643ebcf83fc315d7ec7e1e0e1045208d",
    "febefea46171ba7d843104aa8857bff82670b597e6b0bb0de6b640e7c7f312c1",
    "dfb25fdb5cd26067a881dca5747b78ded10c7a8e5389091eacb2529a26890c3f",
)
ADDITIVE = (
    "physical_generations", "known_input_tokens", "known_output_tokens", "known_usage_tokens",
    "unknown_usage_requests", "unknown_usage_reserved_tokens", "charged_or_reserved_tokens",
    "ledger_error_requests", "count_queries_including_cache_hits", "count_api_requests", "count_api_errors",
    "cached_input_tokens", "cache_write_tokens", "cache_write_30m_tokens", "cache_write_12h_tokens",
    "reasoning_tokens", "count_logical_api_queries", "count_logical_api_errors", "count_cache_hits",
    "original_count_queries", "original_count_api_attempts", "original_count_api_errors",
    "recovery_count_queries", "recovery_count_api_attempts", "recovery_count_api_errors", "count_retry_attempts",
    "headtail_count_queries", "headtail_count_api_attempts", "headtail_count_api_errors",
)
FORBIDDEN = {
    "context", "question", "choice_A", "choice_B", "choice_C", "choice_D", "gold", "answer", "answers",
    "labels", "content", "reasoning", "trace", "facts", "record_facts", "evidence_text", "headers",
    "authorization", "api_key", "sample_paths", "raw_cache", "private_labels",
}
PRIVATE = re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{12,}|(?i:Bearer\s+\S+)|"
                     r"/(?:Users|home|root|private|uniparser|mnt)/|"
                     r"\b(?:192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(?:1[6-9]|2\d|3[01])\.\d+\.\d+)\b")


def check(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value):
    return type(value) is int and value >= 0


def sha(value):
    return isinstance(value, str) and re.fullmatch("[a-f0-9]{64}", value) is not None


def near(left, right):
    return left is right if left is None or right is None else math.isclose(left, right, abs_tol=1e-14)


def rate(k, n):
    return k / n if n else None


def scan(value):
    if isinstance(value, dict):
        check(not (FORBIDDEN & set(value)), "Raw text field in public payload")
        for key, child in value.items():
            if key.endswith("_sha256") and isinstance(child, str):
                check(sha(child), "Invalid SHA-256 metadata")
            scan(child)
    elif isinstance(value, list):
        for child in value:
            scan(child)
    elif isinstance(value, str):
        check(not PRIVATE.search(value), "Private path or credential pattern in public payload")


def key(row):
    return row["requested_model"], row["index"], row["arm"]


def usage(rows, declared):
    for field in ADDITIVE:
        check(all(integer(r[field]) for r in rows), "Invalid usage counter: " + field)
        check(sum(r[field] for r in rows) == declared[field], "Usage total mismatch: " + field)
    check(sum((Decimal(r["estimated_generation_cost_upper_usd"]) for r in rows), Decimal(0))
          == Decimal(declared["estimated_generation_cost_upper_usd"]), "Cost total mismatch")


def record(row, source, probe=False):
    check(row["terminal"] is True and row["id"] == source["id"], "Record identity mismatch")
    check(row["requested_model"] in MODELS, "Unknown requested model")
    for field in ADDITIVE:
        check(integer(row[field]), "Invalid record usage: " + field)
    check(row["known_usage_tokens"] == row["known_input_tokens"] + row["known_output_tokens"],
          "Known token accounting mismatch")
    check(row["charged_or_reserved_tokens"] == row["known_usage_tokens"] + row["unknown_usage_reserved_tokens"],
          "Known plus unknown token accounting mismatch")
    check(row["unknown_usage_requests"] <= row["physical_generations"]
          and row["ledger_error_requests"] <= row["physical_generations"], "Invalid request counters")
    check(row["count_logical_api_queries"] + row["count_cache_hits"]
          == row["count_queries_including_cache_hits"], "Count cache conservation failed")
    check(row["count_api_requests"] == row["count_logical_api_queries"] + row["count_retry_attempts"],
          "Count retry conservation failed")
    check(row["within_registered_budget"] is True
          and row["physical_generations"] <= row["budget"]["max_calls"]
          and row["charged_or_reserved_tokens"] <= row["budget"]["total_tokens"], "Registered budget violation")
    check(set(row["actual_model_ids"]) <= {row["requested_model"]}
          and set(row["returned_service_tiers"]) <= {"default", "not_reported"}, "Model or service tier mismatch")
    check(row["known_usage_missing_model_identity"] == 0, "Known response identity missing")
    if probe:
        check(row["not_scored"] is True and not {"reference_choice", "task_success", "answer_correct_when_available"}
              & set(row), "Capacity probe has semantic score")
        return
    pred, ref = row["prediction"], row["reference_choice"]
    check(ref in "ABCD" and ref == source["reference_choice"], "Reference letter mismatch")
    check(pred is None or pred in ("A", "B", "C", "D"), "Prediction is not a short choice")
    available = pred is not None and row["status"] != "native_admission_denied"
    check(row["answer_available"] is available
          and row["task_success"] is (available and pred == ref), "Choice scoring mismatch")
    if "answer_correct_when_available" in row:
        check(row["answer_correct_when_available"] is ((pred == ref) if available else None), "Conditional score mismatch")


def paired(pairs, declared):
    values = Counter((a, b) for a, b in pairs)
    for name, value in (
        ("n", len(pairs)), ("both_correct", values[True, True]),
        ("direct_only_correct", values[True, False]), ("harness_only_correct", values[False, True]),
        ("both_unsuccessful", values[False, False]),
    ):
        check(declared[name] == value, "Paired result mismatch: " + name)


def validate(p, root):
    scan(p)
    check(p["experiment_id"] == "longbench_openai_limits_20261102" and p["requested_models"] == list(MODELS),
          "Unexpected experiment or models")
    check((p["freeze_sha256"], p["recovery_freeze_sha256"], p["headtail_freeze_sha256"]) == FREEZES,
          "Freeze identities changed")
    sources = p["sources"]
    check([s["index"] for s in sources] == list(INDICES) and len({s["context_sha256"] for s in sources}) == 12,
          "Source cohort changed")
    source = {s["index"]: s for s in sources}
    check(len({s["id"] for s in sources}) == 12 and all(sha(s["context_sha256"]) and integer(s["context_chars"])
          for s in sources), "Source identity metadata invalid")
    check(p["benchmark"]["upstream_revision"] is None and p["benchmark"]["regression_set"] is True
          and p["benchmark"]["fresh_blind_holdout"] is False
          and p["benchmark"]["official_leaderboard_submission"] is False, "Provenance scope changed")
    rows, probes = p["results"], p["capacity_probes"]
    matrix = {(m, i, a) for m in MODELS for i in INDICES for a in ("direct", "harness")}
    check(len(rows) == 72 and {key(r) for r in rows} == matrix, "Main matrix incomplete")
    check(len(probes) == 3 and {key(r) for r in probes} == {(m, 35, "capacity_probe") for m in MODELS},
          "Probe matrix incomplete")
    for row in rows + probes:
        record(row, source[row["index"]], probe=row["arm"] == "capacity_probe")
    by = {key(r): r for r in rows}
    for model in MODELS:
        direct = [by[model, i, "direct"] for i in INDICES]
        harness = [by[model, i, "harness"] for i in INDICES]
        s = p["summary"][model]
        native = [r for r in direct if r["measurement"]["fits_native_direct"] is True]
        for arm, group in (("direct", direct), ("harness", harness)):
            stat = s[arm]
            k, available = sum(r["task_success"] for r in group), sum(r["answer_available"] for r in group)
            check(stat["n_planned"] == 12 and stat["correct_answers"] == k
                  and stat["n_answer_available"] == available, "Model-arm scoring total mismatch")
            check(near(stat["task_success_rate_all_planned"], rate(k, 12))
                  and near(stat["accuracy_among_available_answers_secondary"], rate(k, available)), "Model-arm rate mismatch")
            check(stat["status_counts"] == dict(Counter(r["status"] for r in group)), "Status totals mismatch")
            rejected = sum((r.get("terminal_error") or {}).get("safe_code") == "context_length_exceeded" for r in group)
            check(stat["provider_context_rejected"] == rejected
                  and stat["terminal_errors"] == sum(r.get("terminal_error") is not None for r in group),
                  "Provider/infrastructure outcome total mismatch")
            usage(group, stat)
        check(len(native) == 4 and s["direct"]["n_native_admissible"] == 4
              and s["direct"]["correct_native_admissible"] == sum(r["task_success"] for r in native),
              "Published-admissible result mismatch")
        check(near(s["direct"]["primary_task_success_rate_native_admissible"],
                   rate(sum(r["task_success"] for r in native), 4)), "Published-admissible rate mismatch")
        above = [r for r in direct if r["measurement"]["fits_native_direct"] is False]
        correct_above = sum(by[model, r["index"], "harness"]["task_success"] for r in above)
        check(s["above_native_harness"]["indices"] == sorted(r["index"] for r in above)
              and s["above_native_harness"]["n"] == len(above) == 8
              and s["above_native_harness"]["correct"] == correct_above
              and near(s["above_native_harness"]["task_success_rate"], rate(correct_above, 8)), "Above-native mismatch")
        paired([(r["task_success"], by[model, r["index"], "harness"]["task_success"]) for r in native],
               s["paired_native_admissible_primary"])
        both = [(r["task_success"], by[model, r["index"], "harness"]["task_success"]) for r in direct
                if r["answer_available"] and by[model, r["index"], "harness"]["answer_available"]]
        paired(both, s["paired_available_secondary"])
        for value in s["observed_capacity"].values():
            if isinstance(value, dict) and "full_native_direct_admission_input_tokens" in value:
                n = value["full_native_direct_admission_input_tokens"]
                check(near(value["relative_to_native_window"], n / 1050000)
                      and near(value["relative_to_harness_request_ceiling"], n / 16384), "Expansion denominator mismatch")
        lengths = {r["index"]: r["measurement"]["full_prompt_admission_input_tokens"] for r in direct}
        frontiers = {
            "maximum_attempted": harness,
            "maximum_with_harness_answer": [r for r in harness if r["answer_available"]],
            "maximum_with_correct_harness_answer": [r for r in harness if r["task_success"]],
            "maximum_native_admissible": native,
            "maximum_with_direct_answer": [r for r in direct if r["answer_available"]],
            "maximum_with_correct_direct_answer": [r for r in direct if r["task_success"]],
            "maximum_with_known_direct_processing": [r for r in direct if r["known_usage_tokens"] > 0],
            "minimum_with_actual_direct_context_rejection": [r for r in direct
                if (r.get("terminal_error") or {}).get("safe_code") == "context_length_exceeded"],
        }
        for name, group in frontiers.items():
            expected = ((min if name.startswith("minimum_") else max)(
                group, key=lambda r: (lengths[r["index"]], -r["index"])) if group else None)
            value = s["observed_capacity"][name]
            check((value is None) is (expected is None), "Missing frontier metadata")
            if expected is not None:
                check(value["index"] == expected["index"] and value["id"] == expected["id"]
                      and value["full_native_direct_admission_input_tokens"] == lengths[expected["index"]],
                      "Attempted/answered/correct frontier mismatch: " + name)
    first, recovery, h = (p["original_infrastructure_incomplete_trial"], p["registered_zero_generation_recovery"],
                          p["headtail_quality_control"])
    old = first["results_including_unscored_probes"]
    check(len(old) == 75 and len({key(r) for r in old}) == 75, "Original history incomplete")
    selected = recovery["selected_cases"]
    selected_keys = {key(r) for r in selected}
    check(len(selected) == len(selected_keys) == 39 and recovery["selection_used_answers_or_correctness"] is False
          and recovery["original_generated_or_charged_tasks_retried"] == recovery["generation_retries"] == 0,
          "Recovery selection protocol changed")
    old_by = {key(r): r for r in old}
    for case in selected:
        before = old_by[key(case)]
        check(before["eligible_for_zero_generation_recovery"] is True and before["physical_generations"] == 0
              and before["charged_or_reserved_tokens"] == 0 and before["status"] == "error",
              "Recovery includes a consumed generation")
    for row in rows + probes:
        is_recovered = key(row) in selected_keys
        check(row["outcome_origin"] == ("zero_generation_network_recovery" if is_recovered else "original_trial")
              and row["actual_execution_freeze_sha256"] == FREEZES[1 if is_recovered else 0], "Recovery lineage mismatch")
        if not is_recovered:
            check(row["status"] == old_by[key(row)]["status"]
                  and row.get("prediction") == old_by[key(row)].get("prediction"), "Original generated outcome changed")
    usage(old, first["usage_including_original_probes"])
    usage(rows + probes, p["usage_main_and_recovery_including_probes"])
    check(h["planned_indices"] == [143, 260] and h["n_planned_records"] == 6
          and h["selection_uses_answers_or_model_outputs"] is False, "Head/tail preregistration changed")
    aux = h["results"]
    check(len(aux) == 6 and {key(r) for r in aux} == {(m, i, "headtail_control") for m in MODELS for i in (143, 260)},
          "Head/tail matrix incomplete")
    trims = {s["index"]: s for s in h["sources"]}
    for s in trims.values():
        (a, b), (c, d) = s["source_spans"]
        check(a == 0 and d == s["original_context_chars"] == source[s["index"]]["context_chars"]
              and s["dropped_middle_span"] == [b, c] and 0 < b < c < d
              and b + d - c == s["retained_chars"] and abs(b - (d-c)) <= 1
              and s["inserted_characters"] == 0, "Head/tail spans mismatch")
    for row in aux:
        record(row, source[row["index"]])
        check(row["strict_official_correct"] is (row["strict_official_prediction"] == row["reference_choice"])
              and row["strict_format_accepted"] is (row["strict_official_prediction"] is not None), "Strict choice score mismatch")
        check(row["actual_admission_input_tokens"] + 1024 <= 800000
              and row["original_context_sha256"] == source[row["index"]]["context_sha256"]
              and row["trimmed_context_sha256"] == trims[row["index"]]["trimmed_context_sha256"], "Head/tail metadata mismatch")
    for model in MODELS:
        group, s = [r for r in aux if r["requested_model"] == model], h["summary"][model]
        check(s["primary_correct"] == sum(r["task_success"] for r in group)
              and s["strict_official_correct"] == sum(r["strict_official_correct"] for r in group)
              and s["harness_same_sources_correct"] == sum(by[model, i, "harness"]["task_success"] for i in h["planned_indices"]),
              "Head/tail model score mismatch")
        usage(group, s)
    usage(aux, h["usage"])
    usage(rows + probes + aux, p["usage_all_including_probes"])
    phases = (first["usage_including_original_probes"], recovery["new_physical_generation_usage_including_probes"], h["usage"])
    for field in ADDITIVE:
        check(sum(s[field] for s in phases) == p["usage_all_including_probes"][field], "Three-phase conservation failed")
    cost = p["cost"]
    total = sum((Decimal(s["estimated_generation_cost_upper_usd"]) for s in phases), Decimal(0))
    check(total == Decimal(cost["estimated_generation_cost_upper_usd"])
          and total + Decimal(cost["separate_preflight_cost_upper_usd"]) == Decimal(cost["combined_with_preflight_upper_usd"])
          <= Decimal(cost["cap_usd"]) == 50, "Shared money cap/conservation failed")
    scope = p["count_request_scopes"]
    prep = sum(x["physical_http_attempts"] for x in h["count_only_preparation"].values())
    check(scope["all_recorded_count_http_attempts"] == p["usage_all_including_probes"]["count_api_requests"]
          + prep + p["separate_preflight"]["count_attempts"], "Count scope reconciliation failed")
    check(p["all_terminal_records"] == 120 and p["logical_outcomes_including_auxiliary"] == 81,
          "Physical-history/logical-outcome distinction changed")
    check(datetime.fromisoformat(p["frozen_at_utc"]) < datetime.fromisoformat(p["final_scoring_started_at_utc"]),
          "Reported scoring timestamp predates freeze")
    for name in ("all_114_and_conditional_headtail_terminal_before_label_access",
                 "all_original_generated_outcomes_preserved_without_rerun", "token_and_monetary_ledgers_recomputed"):
        check(p["integrity"][name] is True, "Missing declared gate: " + name)
    hashes = h["code_sha256"]
    check(len(hashes) == 58 and len(p["integrity"]["source_sha256"]) == 56, "Source fingerprint count changed")
    for filename, expected in hashes.items():
        path = Path(filename)
        check(not path.is_absolute() and ".." not in path.parts and path.suffix == ".py" and sha(expected),
              "Unsafe source fingerprint path")
        check(hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, "Published source fingerprint differs: " + filename)
    for filename, expected in p["integrity"]["source_sha256"].items():
        check(hashes[filename] == expected, "Base source identity differs between phases")
    posthoc = p["posthoc_audits"]
    check(posthoc["computed_after_all_scoring"] is True and posthoc["changes_primary_choices_or_scores"] is False,
          "Post-hoc diagnostics changed primary outcomes")
    public_audits = {}
    for name in ("independent_cost_score_audit", "trajectory_audit"):
        meta = posthoc[name]
        file = root / meta["repository_path"]
        check(hashlib.sha256(file.read_bytes()).hexdigest() == meta["sha256"], "Public audit fingerprint mismatch")
        doc = json.loads(file.read_text())
        scan(doc)
        check(doc["violations"] == [], "Public audit reports violations")
        public_audits[name] = doc
        check(hashlib.sha256((root / meta["auditor_repository_path"]).read_bytes()).hexdigest() == meta["auditor_sha256"],
              "Post-hoc auditor source fingerprint mismatch")
    independent = public_audits["independent_cost_score_audit"]
    for a, b in (("physical_generations_all_phases", "physical_generations"),
                 ("known_input_tokens_all_phases", "known_input_tokens"),
                 ("known_output_tokens_all_phases", "known_output_tokens"),
                 ("charged_or_reserved_tokens_all_phases", "charged_or_reserved_tokens"),
                 ("unknown_usage_requests_all_phases", "unknown_usage_requests")):
        check(independent[a] == p["usage_all_including_probes"][b], "Independent cost audit disagrees")
    check(independent["combined_cost_with_preflight_upper_usd"] == p["cost"]["combined_with_preflight_upper_usd"],
          "Independent money audit disagrees")
    replay, raw = posthoc["trajectory_audit"], posthoc["strict_selected_raw_reply_diagnostic"]
    replay_artifact = public_audits["trajectory_audit"]
    replay_source = replay_artifact["totals"]
    physical_history = replay_artifact["records"]
    check(len(physical_history) == 120
          and replay_source["trajectory_passed"] == sum(r["trajectory"]["status"] == "passed" for r in physical_history)
          and replay_source["trajectory_unavailable"] == sum(r["trajectory"]["status"] == "unavailable" for r in physical_history),
          "Trajectory status counters omit a phase")
    for field in ("known_responses_replayed", "unknown_delivery_errors_replayed", "generation_entries_replayed",
                  "count_events_replayed", "responses_wire_request_hashes_checked", "wire_prompt_hashes_checked"):
        check(replay_source[field] == sum(r["trajectory"].get(field, 0) for r in physical_history),
              "Trajectory aggregate differs from physical-record sum: " + field)
    check(replay_source["known_responses_replayed"] + replay_source["unknown_delivery_errors_replayed"]
          == replay_source["generation_entries_replayed"], "Replayed generation-kind accounting mismatch")
    check(all(integer(r.get("recorded_count_events")) for r in physical_history)
          and replay_source["recorded_count_events_all_phases"] == sum(r["recorded_count_events"] for r in physical_history)
              == p["usage_all_including_probes"]["count_queries_including_cache_hits"],
          "Runtime recorded-count phase totals mismatch")
    for public_field, audit_field in (
        ("physical_generation_entries_replayed", "generation_entries_replayed"),
        ("known_generation_responses_replayed", "known_responses_replayed"),
        ("unknown_delivery_errors_replayed", "unknown_delivery_errors_replayed"),
        ("recorded_runtime_count_events", "recorded_count_events_all_phases"),
        ("runtime_count_events_replayed", "count_events_replayed"),
    ):
        check(replay[public_field] == replay_source[audit_field], "Public replay scope differs from source audit")
    check(replay["physical_history_records"] == replay["ledgers_passed"] == 120
          and replay["trajectories_replayed"] == 80 and replay["trajectories_unavailable"] == 40
          and replay["main_logical_trajectories_replayed"] == 74
          and replay["main_logical_trajectories_unavailable"] == 1
          and replay["auxiliary_trajectories_replayed"] == 6
          and replay_source["trajectory_unavailable"] == replay["trajectories_unavailable"],
          "Replay pass/unavailable accounting changed")
    case = replay["unavailable_main_logical_cases"]
    check(len(case) == 1 and case[0]["requested_model"] == "gpt-6-sol" and case[0]["index"] == 251
          and case[0]["arm"] == "harness"
          and case[0]["reason_code"] == "recorded_count_failure_lacks_exception_payload",
          "Material unavailable replay case omitted")
    check(replay["semantic_entailment_or_reference_truth_verified"] is False
          and raw["strict_parser_used_to_select_candidate"] is False
          and raw["missing_raw_reply_is_format_failure"] is False, "Diagnostic scope changed")
    receipts = raw["receipts"]
    expected_all = {key(r): r for r in rows + probes + aux}
    check(len(receipts) == 81 and {key(r) for r in receipts} == set(expected_all), "Raw receipt matrix mismatch")
    for r in receipts:
        primary = expected_all[key(r)]
        check(r["primary_controller_choice"] == primary.get("prediction")
              and r["reference_choice"] == primary.get("reference_choice"), "Raw receipt changed primary choice")
        available = r["raw_reply_available"]
        check(type(available) is bool and r["not_scored"] is (r["arm"] == "capacity_probe"), "Invalid receipt scope")
        if available:
            check(sha(r["raw_reply_sha256"]) and r["selected_raw_strict_choice"] in (None, "A", "B", "C", "D")
                  and r["strict_format_accepted"] is (r["selected_raw_strict_choice"] is not None),
                  "Strict traceable receipt malformed")
            check(r["raw_strict_agrees_with_primary"] is (r["selected_raw_strict_choice"] == primary.get("prediction")),
                  "Strict/raw primary agreement mismatch")
            if not r["not_scored"]:
                check(r["strict_correct_on_traceable_raw_reply"] is
                      (r["selected_raw_strict_choice"] == primary["reference_choice"]), "Strict raw score mismatch")
            if r["arm"] == "harness":
                check(r["selected_provenance"]["kind"] == "fixed_actual_selected_candidate_raw_reply"
                      and r["selected_provenance"]["strict_parser_used_to_select_candidate"] is False,
                      "Strict parser reselected a candidate")
        else:
            check(all(r[k] is None for k in ("raw_reply_sha256", "selected_raw_strict_choice", "selected_raw_compatible_choice",
                                             "strict_format_accepted", "strict_correct_on_traceable_raw_reply")),
                  "Unavailable receipt became a format failure")
    available = [r for r in receipts if r["raw_reply_available"]]
    check(len(available) == raw["raw_replies_traceable"] == raw["strict_format_accepted_on_traceable"] == 46
          and raw["raw_replies_unavailable"] == 35
          and sum(r["raw_strict_agrees_with_primary"] is False for r in available)
              == raw["strict_vs_primary_disagreements_on_traceable"] == 0, "Raw parser aggregate mismatch")
    for model in MODELS:
        for arm in ("direct", "harness", "headtail_control"):
            group = [r for r in receipts if r["requested_model"] == model and r["arm"] == arm]
            traced = [r for r in group if r["raw_reply_available"]]
            stat = raw["summary"][model][arm]
            correct = sum(r["strict_correct_on_traceable_raw_reply"] is True for r in traced)
            check(stat["n_planned"] == len(group) and stat["raw_replies_traceable"] == len(traced)
                  and stat["raw_replies_unavailable"] == len(group)-len(traced)
                  and stat["strict_correct_on_traceable"] == correct
                  and near(stat["strict_accuracy_on_traceable_only_secondary"], rate(correct, len(traced)))
                  and stat["primary_correct_all_planned"] == sum(expected_all[key(r)]["task_success"] for r in group),
                  "Traceable-only strict summary mismatch")
    for filename in ("docs/longbench_openai_limits_20261102.md", "docs/longbench_openai_limits_20261102.zh-CN.md"):
        text = (root / filename).read_text()
        scan(text)
        check(all(value in text for value in FREEZES), "Markdown freeze identity missing")
    return {"status": "PASS", "main_records": 72, "probes": 3, "original_records_preserved": 75,
            "zero_generation_recoveries": 39, "headtail_records": 6, "source_fingerprints_verified": 58,
            "posthoc_auditor_fingerprints_verified": 2, "trajectory_replayed": 80, "trajectory_unavailable": 40,
            "strict_raw_replies_traceable": 46,
            "physical_generations": p["usage_all_including_probes"]["physical_generations"],
            "all_recorded_count_http_attempts": scope["all_recorded_count_http_attempts"],
            "conservative_cost_with_preflight_usd": cost["combined_with_preflight_upper_usd"],
            "scope": "Public arithmetic, metadata, privacy patterns and live source hashes; no private API or label replay"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    path = args.json or root / "benchmarks/longbench_openai_limits_20261102.json"
    print(json.dumps(validate(json.loads(path.read_text()), root), sort_keys=True))


if __name__ == "__main__":
    main()
