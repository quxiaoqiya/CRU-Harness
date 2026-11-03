#!/usr/bin/env python3
"""Full LongBench v2 comparison: direct model vs model-first harness.

Runs every question in the local index for one or more model context budgets.
For each question it records the dataset index, source fields, question, the
context size in characters and tokens, the direct result, and the harness
result (routing mode, evidence volume, vote counts, and final answer).

Direct answers are budget-independent, so they are cached once per question
and reused across budgets.  Every path is resumable: existing JSONL records
are skipped, and summaries are rebuilt from disk so a stopped run still
produces complete aggregates for the questions it finished.

This runner never downloads the dataset itself; build the index first with
``scripts/build_longbench_v2_index.py``.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from long_context_agent.harness import LongContextHarness
from long_context_agent.token_budget import AdaptiveTokenEstimator
from scripts.benchmark_longbench_v2_compare import QwenHTTPClient
from scripts.longbench_v2_protocol import (
    LongBenchRow,
    aggregate_option_verifications,
    chunked_document,
    evidence_window_groups,
    extract_choice,
    memory_prompt,
    option_verification_prompt,
    official_prompt,
    parse_option_verification,
    verify_evidence_candidates,
)


def load_index(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(cache_path: Path) -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    if not cache_path.exists():
        raise SystemExit(f"missing row cache: {cache_path} (run scripts/build_longbench_v2_index.py)")
    with cache_path.open(encoding="utf-8") as cache:
        for line in cache:
            record = json.loads(line)
            rows[int(record["index"])] = record
    return rows


def make_row(record: dict[str, Any]) -> LongBenchRow:
    return LongBenchRow(
        item_id=str(record.get("id", "")),
        domain=str(record.get("domain", "")),
        sub_domain=str(record.get("sub_domain", "")),
        difficulty=str(record.get("difficulty", "")),
        length=str(record.get("length", "")),
        question=str(record.get("question", "")),
        choice_a=str(record.get("choice_A", "")),
        choice_b=str(record.get("choice_B", "")),
        choice_c=str(record.get("choice_C", "")),
        choice_d=str(record.get("choice_D", "")),
        answer=str(record.get("answer", "")).upper(),
        context=str(record.get("context", "")),
        offset=int(record.get("index", 0)),
    )


def chunk_items(context: str, chunk_chars: int) -> list[dict[str, Any]]:
    return [
        {"index": index, "text": context[start : start + chunk_chars], "start_char": start}
        for index, start in enumerate(range(0, len(context), chunk_chars))
    ]


def usage_tokens(usage: Any, key: str = "prompt_tokens") -> int:
    if isinstance(usage, dict):
        value = usage.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    return 0


def answer_fields(result: Any, row: LongBenchRow) -> dict[str, Any]:
    choice = extract_choice(result.content) or extract_choice(result.reasoning)
    return {
        "prediction": choice,
        "correct": bool(choice == row.answer) if choice else False,
        "prompt_tokens": usage_tokens(result.usage),
        "completion_tokens": usage_tokens(result.usage, "completion_tokens"),
        "seconds": round(float(result.seconds), 3),
        "fallback_used": bool(result.fallback_used),
        "attempts": int(result.attempts),
        "error": None,
    }


def error_fields(exc: BaseException, *, seconds: float | None = None) -> dict[str, Any]:
    body = getattr(exc, "read", None)
    payload = ""
    if callable(body):
        try:
            payload = body().decode("utf-8", "ignore")[:400]
        except Exception:
            payload = ""
    result: dict[str, Any] = {
        "prediction": None,
        "correct": False,
        "error": {"type": type(exc).__name__, "message": str(exc), "body": payload},
    }
    if seconds is not None:
        result["seconds"] = round(seconds, 3)
    return result


def run_harness(
    row: LongBenchRow,
    document: Any,
    groups: list[dict[str, Any]],
    direct: dict[str, Any] | None,
    *,
    args: argparse.Namespace,
    estimator: AdaptiveTokenEstimator,
    budget: int,
    client: QwenHTTPClient,
) -> dict[str, Any]:
    started = time.perf_counter()
    prompt = official_prompt(row)
    decision = LongContextHarness(
        reader_agent=None,
        max_context_tokens=budget,
        reserve_tokens=args.reserve_tokens,
        read_when_over_budget=False,
        token_estimator=estimator,
    ).prepare(document, full_prompt=prompt, scope=row.domain)
    record: dict[str, Any] = {
        "budget": budget,
        "mode": decision.mode,
        "estimated_prompt_tokens": decision.estimated_prompt_tokens,
        "estimate_method": decision.estimate_method,
        "estimate_scope": decision.estimate_scope,
        "estimator_chars_per_token": round(decision.estimator_chars_per_token, 4),
    }
    if decision.mode == "direct_pass" and direct and not direct.get("error"):
        record.update(dict(direct))
        record.update({"reused_direct": True, "model_calls": 0, "evidence_chars": 0})
        record["seconds"] = direct.get("seconds", 0.0)
        return record
    if decision.mode == "direct_pass":
        # The estimator said the prompt fits, but the provider rejected it.
        # The effective path is evidence-based, so do not report direct_pass.
        record["mode"] = "evidence_fallback"
        record["estimate_mismatch"] = True
    # The estimator said the prompt fits but the provider rejected it, or the
    # prompt is genuinely over budget: answer from bounded evidence instead.
    candidates: list[dict[str, Any]] = []
    candidate_objects: list[Any] = []
    candidate_evidence: list[str] = []
    vote_counts: dict[str, int] = {}
    model_calls = 0
    prompt_tokens = 0
    if not groups:
        return {**record, "prediction": None, "correct": False, "error": {"type": "NoEvidence", "message": "no windows"}}
    for group in groups:
        text = str(group.get("text", ""))
        candidate_evidence.append(text)
        try:
            candidate_prompt = memory_prompt(row, "", text)
            if str(group.get("query_mode")) == "document_coverage":
                candidate_prompt += (
                    "\nThis evidence view deliberately samples multiple distant source documents or regions. "
                    "Compare the documents before choosing an option; do not answer from a single isolated window."
                )
            result = client.complete_answer(
                candidate_prompt,
                max_tokens=args.candidate_max_tokens,
                enable_thinking=False,
                preserve_thinking=False,
            )
            model_calls += 1
            prompt_tokens += usage_tokens(result.usage)
            candidate_objects.append(result)
            choice = extract_choice(result.content) or extract_choice(result.reasoning)
        except Exception as exc:
            candidate_objects.append(None)
            choice = None
            record.setdefault("candidate_errors", []).append(error_fields(exc)["error"])
        vote_weight = 3 if (
            row.domain == "Long Structured Data Understanding"
            and str(group.get("query_mode")) == "structured_records"
        ) else 1
        if choice:
            vote_counts[choice] = vote_counts.get(choice, 0) + vote_weight
        candidates.append(
            {
                "candidate_index": group["candidate_index"],
                "query_mode": group["query_mode"],
                "vote_choice": choice,
                "vote_weight": vote_weight,
                "evidence_chars": len(text),
            }
        )
    if not vote_counts and candidate_evidence:
        # Every independent view failed to emit a letter. One strict final
        # attempt on the joint view avoids recording a recoverable answer as
        # missing, and it is counted as an extra model call.
        window_index = min(2, len(candidate_evidence) - 1)
        try:
            fallback = client.complete_answer(
                memory_prompt(row, "", candidate_evidence[window_index])
                + "\nAnswer with exactly one letter: (A), (B), (C), or (D).",
                max_tokens=512,
                enable_thinking=False,
                preserve_thinking=False,
            )
            model_calls += 1
            prompt_tokens += usage_tokens(fallback.usage)
            fallback_choice = extract_choice(fallback.content) or extract_choice(fallback.reasoning)
            if fallback_choice:
                vote_counts[fallback_choice] = vote_counts.get(fallback_choice, 0) + 1
                candidates.append(
                    {
                        "candidate_index": len(candidates),
                        "query_mode": "fallback_joint",
                        "vote_choice": fallback_choice,
                        "evidence_chars": len(candidate_evidence[window_index]),
                    }
                )
                candidate_evidence.append(candidate_evidence[window_index])
                candidate_objects.append(fallback)
                record["fallback_answer"] = True
        except Exception as exc:
            record.setdefault("candidate_errors", []).append(error_fields(exc)["error"])
    verification = verify_evidence_candidates(
        row,
        [{**candidate, "evidence_text": candidate_evidence[index]} for index, candidate in enumerate(candidates)],
    )
    winner = verification.get("verified_winner")
    candidate_index = verification.get("selected_candidate_index")

    # A lexical evidence check is useful for tie breaking, but it cannot tell
    # whether an option is actually entailed by a retrieved window.  When
    # enabled, ask the provider to label every option independently as
    # support/against/unknown and require source spans through the strict
    # aggregator.  The aggregator may override a majority only when a
    # span-backed option has repeated support and is itself a voted candidate;
    # otherwise the original evidence vote is preserved.
    option_verifications: list[dict[str, Any]] = []
    option_verification_errors: list[dict[str, Any]] = []
    option_verification_retries = 0
    option_aggregate: dict[str, Any] | None = None
    if getattr(args, "harness_option_verify", False) and candidate_evidence:
        for index, evidence_text in enumerate(candidate_evidence):
            try:
                verifier_prompt = option_verification_prompt(row, evidence_text)
                verifier_result = client.complete(
                    verifier_prompt,
                    max_tokens=getattr(args, "harness_option_verify_tokens", 4_096),
                    enable_thinking=False,
                    preserve_thinking=False,
                    reasoning_effort="none",
                )
                model_calls += 1
                prompt_tokens += usage_tokens(verifier_result.usage)
                parsed = parse_option_verification(verifier_result.content or verifier_result.reasoning)
                if parsed is None:
                    # A short, no-thinking retry recovers truncated JSON and
                    # servers that return a prose preamble despite the JSON
                    # instruction. Keep the first response diagnostics local
                    # so the audit can distinguish a recovered parse from a
                    # persistent verifier failure.
                    for _ in range(max(0, int(getattr(args, "harness_option_verify_retries", 1)))):
                        option_verification_retries += 1
                        retry_result = client.complete(
                            verifier_prompt
                            + "\nReturn exactly one compact JSON object. Do not include markdown, prose, or code fences. "
                            "If the source does not decide an option, use status unknown with an empty evidence_spans list.",
                            max_tokens=min(1_536, max(512, int(getattr(args, "harness_option_verify_tokens", 4_096)))),
                            enable_thinking=False,
                            preserve_thinking=False,
                            reasoning_effort=None,
                        )
                        model_calls += 1
                        prompt_tokens += usage_tokens(retry_result.usage)
                        parsed = parse_option_verification(retry_result.content or retry_result.reasoning)
                        if parsed is not None:
                            verifier_result = retry_result
                            break
                    if parsed is None:
                        option_verification_errors.append(
                            {
                                "candidate_index": candidates[index].get("candidate_index", index),
                                "type": "ParseError",
                                "initial_content_chars": len(verifier_result.content or ""),
                                "initial_reasoning_chars": len(verifier_result.reasoning or ""),
                                "initial_preview": (verifier_result.content or verifier_result.reasoning or "")[:240],
                            }
                        )
                        continue
                option_verifications.append(
                    {
                        **parsed,
                        "candidate_index": candidates[index].get("candidate_index", index),
                        # Kept only until aggregation; the output record omits
                        # raw evidence to avoid duplicating the source text.
                        "evidence_text": evidence_text,
                    }
                )
            except Exception as exc:
                option_verification_errors.append(
                    {
                        "candidate_index": candidates[index].get("candidate_index", index),
                        **(error_fields(exc).get("error") or {}),
                    }
                )
        option_aggregate = aggregate_option_verifications(
            row,
            option_verifications,
            vote_counts=verification.get("vote_counts"),
            majority_winner=verification.get("majority_winner"),
        )
        if option_aggregate.get("strong_override"):
            # Only select an existing candidate object.  This guards against a
            # verifier hallucinating a label that no evidence-view model voted
            # for, even if its source spans look plausible.
            option_winner = option_aggregate.get("winner")
            matching = next(
                (
                    (index, candidate)
                    for index, candidate in enumerate(candidates)
                    if candidate.get("vote_choice") == option_winner
                ),
                None,
            )
            if matching is not None:
                candidate_index = matching[0]
                winner = option_winner

    final_result = None
    if candidate_index is not None and int(candidate_index) in range(len(candidate_objects)):
        final_result = candidate_objects[int(candidate_index)]
    prediction = extract_choice(getattr(final_result, "content", "")) if final_result is not None else None
    if not prediction:
        prediction = extract_choice(getattr(final_result, "reasoning", "")) if final_result is not None else None
    if not prediction:
        prediction = winner
    record.update(
        {
            "reused_direct": False,
            "model_calls": model_calls,
            "prompt_tokens": prompt_tokens,
            "evidence_chars": len("\n\n".join(candidate_evidence)),
            "candidate_results": candidates,
            "vote_counts": vote_counts,
            "vote_winner": verification.get("majority_winner"),
            "selected_winner": winner,
            "verification_override": bool(
                verification.get("majority_winner") and winner and winner != verification.get("majority_winner")
            ),
            "verification": {
                key: verification.get(key)
                for key in ("majority_winner", "verified_winner", "verification_margin", "reason")
            },
            "option_verifications": [
                {key: value for key, value in item.items() if key != "evidence_text"}
                for item in option_verifications
            ],
            "option_verification_errors": option_verification_errors,
            "option_verification_retries": option_verification_retries,
            "option_verification": (
                {
                    key: option_aggregate.get(key)
                    for key in (
                        "winner",
                        "semantic_winner",
                        "majority_winner",
                        "strong_override",
                        "status_counts",
                        "option_scores",
                        "reason",
                    )
                }
                if option_aggregate is not None
                else None
            ),
            "option_verification_override": bool(option_aggregate and option_aggregate.get("strong_override")),
            "prediction": prediction,
            "correct": bool(prediction == row.answer) if prediction else False,
            "answer_source": "candidate" if final_result is not None else "verification",
            "seconds": round(time.perf_counter() - started, 3),
            "error": None,
        }
    )
    return record


def stats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [row for row in rows if row.get("prediction")]
    correct = sum(1 for row in scored if row.get("correct"))
    return {
        "n": len(rows),
        "answered": len(scored),
        "correct": correct,
        "accuracy": round(correct / len(scored), 4) if scored else None,
        "accuracy_all": round(correct / len(rows), 4) if rows else None,
        "errors": sum(1 for row in rows if row.get("error")),
    }


def quantiles(values: list[int]) -> dict[str, float]:
    values = sorted(values)
    if not values:
        return {}
    pick = lambda q: values[min(len(values) - 1, int(q * (len(values) - 1)))]
    return {
        "min": values[0],
        "p25": pick(0.25),
        "median": statistics.median(values),
        "p75": pick(0.75),
        "p90": pick(0.90),
        "max": values[-1],
        "mean": round(statistics.mean(values), 1),
    }


def build_sample(
    item: dict[str, Any],
    record: dict[str, Any],
    direct: dict[str, Any] | None,
    harness: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "index": int(item["index"]),
        "id": record.get("id"),
        "domain": record.get("domain"),
        "sub_domain": record.get("sub_domain"),
        "difficulty": record.get("difficulty"),
        "length": record.get("length"),
        "question": record.get("question"),
        "choices": {label: record.get(f"choice_{label}") for label in "ABCD"},
        "answer": str(record.get("answer", "")).upper(),
        "context_chars": int(item.get("context_chars") or len(str(record.get("context", "")))),
        "context_words": int(item.get("context_words") or 0),
        "estimated_prompt_tokens": int(item.get("estimated_prompt_tokens") or 0),
        "context_tokens": int((direct or {}).get("prompt_tokens") or item.get("estimated_prompt_tokens") or 0),
        "context_tokens_source": "provider_prompt_tokens"
        if (direct or {}).get("prompt_tokens")
        else "char_estimate",
        "direct": direct or {},
        "harness": harness,
    }


def effective_mode(record: dict[str, Any]) -> str:
    """Report the path that actually ran, not the pre-flight estimate."""
    if record.get("reused_direct"):
        return "direct_pass"
    mode = str(record.get("mode") or "")
    if mode == "raw_source":
        return "raw_source"
    if mode == "direct_pass":
        return "evidence_fallback"
    return mode or "unknown"


class JsonlSink:
    """Thread-safe append-only JSONL writer."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()

    def append(self, record: dict[str, Any]) -> None:
        with self.lock:
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def load(self) -> dict[int, dict[str, Any]]:
        if not self.path.exists():
            return {}
        records: dict[int, dict[str, Any]] = {}
        with self.path.open(encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue
                records[int(record["index"])] = record
        return records


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, default=Path("docs/longbench_v2_index.json"))
    parser.add_argument("--rows", type=Path, default=Path("data/longbench_v2_rows.jsonl"))
    parser.add_argument("--outdir", type=Path, default=Path("docs/longbench_v2_full"))
    parser.add_argument("--budgets", nargs="+", type=int, default=[65_536])
    parser.add_argument("--limit", type=int, default=0, help="0 means every indexed question")
    parser.add_argument("--indices", nargs="+", type=int, default=[])
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--reserve-tokens", type=int, default=8_192)
    parser.add_argument("--chunk-chars", type=int, default=48_000)
    parser.add_argument("--max-tokens", type=int, default=8_192)
    parser.add_argument("--candidate-max-tokens", type=int, default=2_048)
    parser.add_argument("--evidence-windows", type=int, default=8)
    parser.add_argument("--evidence-window-chars", type=int, default=4_000)
    parser.add_argument("--evidence-max-chars", type=int, default=28_000)
    parser.add_argument("--harness-vote-candidates", type=int, default=3)
    parser.add_argument(
        "--harness-option-verify",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="cross-check each evidence-view answer with option-level support/against/unknown labels",
    )
    parser.add_argument("--harness-option-verify-tokens", type=int, default=4_096)
    parser.add_argument(
        "--harness-option-verify-retries",
        type=int,
        default=1,
        help="compact no-thinking retries after an option-verifier parse failure",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("config/long_context_token_calibration.json"),
        help="per-domain token coefficients measured from provider usage",
    )
    parser.add_argument("--reasoning-effort", default=os.getenv("Q38_REASONING_EFFORT", "medium"))
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--base-url", default=os.getenv("Q38_BASE_URL", "http://127.0.0.1:18016/v1"))
    parser.add_argument("--model", default=os.getenv("Q38_MODEL", "Qwen3.8-Flash-Next-FP8"))
    parser.add_argument("--api-key", default=os.getenv("Q38_API_KEY"))
    parser.add_argument("--timeout", type=float, default=3600.0)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    budgets = list(dict.fromkeys(int(budget) for budget in args.budgets))
    if not args.api_key:
        raise SystemExit("Q38_API_KEY is required")
    index = load_index(args.index)
    if args.indices:
        selected = [item for item in index if int(item["index"]) in set(args.indices)]
    elif args.limit:
        selected = index[: args.limit]
    else:
        selected = index
    if not selected:
        raise SystemExit("no questions selected")
    rows = load_rows(args.rows)
    missing = [item["index"] for item in selected if int(item["index"]) not in rows]
    if missing:
        raise SystemExit(f"row cache is missing indices: {missing[:10]}")

    args.outdir.mkdir(parents=True, exist_ok=True)
    direct_sink = JsonlSink(args.outdir / "direct.jsonl")
    harness_sinks = {budget: JsonlSink(args.outdir / f"harness_{budget}.jsonl") for budget in budgets}
    direct_records = direct_sink.load()
    harness_records = {budget: sink.load() for budget, sink in harness_sinks.items()}

    estimator = AdaptiveTokenEstimator.from_calibration_file(args.calibration)
    estimator_lock = threading.Lock()
    progress = {"done": 0, "errors": 0}
    progress_lock = threading.Lock()

    def sample_task(item: dict[str, Any]) -> None:
        record = rows[int(item["index"])]
        row = make_row(record)
        index_value = int(item["index"])
        client = QwenHTTPClient(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            timeout=args.timeout,
            reasoning_effort=args.reasoning_effort,
        )
        prompt = official_prompt(row)
        direct = direct_records.get(index_value)
        if direct is None:
            direct_started = time.perf_counter()
            try:
                result = client.complete_answer(
                    prompt,
                    max_tokens=args.max_tokens,
                    enable_thinking=args.enable_thinking,
                    preserve_thinking=args.enable_thinking,
                )
                direct = answer_fields(result, row)
                with estimator_lock:
                    estimator.observe_usage(prompt, result.usage, scope=row.domain)
            except Exception as exc:
                direct = error_fields(exc, seconds=time.perf_counter() - direct_started)
            direct = {"index": index_value, **direct}
            direct_records[index_value] = direct
            direct_sink.append(direct)
        document = chunked_document(row, chunk_chars=args.chunk_chars)
        # Building evidence windows is the expensive part of a long row, so
        # skip it when every budget will reuse the direct answer.
        needs_evidence = bool(direct.get("error"))
        if not needs_evidence:
            for budget in budgets:
                if harness_records[budget].get(index_value) is not None:
                    continue
                decision = LongContextHarness(
                    reader_agent=None,
                    max_context_tokens=budget,
                    reserve_tokens=args.reserve_tokens,
                    read_when_over_budget=False,
                    token_estimator=estimator,
                ).prepare(document, full_prompt=prompt, scope=row.domain)
                if decision.mode != "direct_pass":
                    needs_evidence = True
                    break
        groups = (
            evidence_window_groups(
                row,
                chunk_items(row.context, args.chunk_chars),
                candidate_count=args.harness_vote_candidates,
                max_windows=args.evidence_windows,
                window_chars=args.evidence_window_chars,
                max_total_chars=args.evidence_max_chars,
            )
            if needs_evidence
            else []
        )
        for budget in budgets:
            if harness_records[budget].get(index_value) is not None:
                continue
            try:
                outcome = run_harness(
                    row, document, groups, direct, args=args, estimator=estimator, budget=budget, client=client
                )
            except Exception as exc:
                outcome = {"budget": budget, "mode": "error", **error_fields(exc)}
            outcome = {"index": index_value, **outcome}
            harness_records[budget][index_value] = outcome
            harness_sinks[budget].append(outcome)
        with progress_lock:
            progress["done"] += 1
            if (direct or {}).get("error"):
                progress["errors"] += 1
            if progress["done"] % 10 == 0:
                print(
                    json.dumps(
                        {"event": "progress", "done": progress["done"], "total": len(pending), "errors": progress["errors"]}
                    ),
                    flush=True,
                )

    pending = [
        item for item in selected if any(int(item["index"]) not in harness_records[budget] for budget in budgets)
    ]
    print(json.dumps({"event": "start", "questions": len(selected), "pending": len(pending), "budgets": budgets, "jobs": args.jobs}), flush=True)
    started = time.perf_counter()
    if pending:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            futures = {pool.submit(sample_task, item): item for item in pending}
            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as exc:
                    print(json.dumps({"event": "task_error", "index": futures[future]["index"], "error": str(exc)[:300]}), flush=True)

    completed = [
        item
        for item in selected
        if all(int(item["index"]) in harness_records[budget] for budget in budgets)
        and int(item["index"]) in direct_records
    ]
    samples = [
        build_sample(
            item,
            rows[int(item["index"])],
            direct_records.get(int(item["index"])),
            {str(budget): harness_records[budget][int(item["index"])] for budget in budgets},
        )
        for item in completed
    ]
    budget_summaries = []
    for budget in budgets:
        harness_rows = [sample["harness"][str(budget)] for sample in samples]
        modes: dict[str, int] = {}
        for row in harness_rows:
            modes[effective_mode(row)] = modes.get(effective_mode(row), 0) + 1
        by_length: dict[str, dict[str, list[dict[str, Any]]]] = {}
        by_domain: dict[str, dict[str, list[dict[str, Any]]]] = {}
        for sample in samples:
            tier = str(sample.get("length", ""))
            domain = str(sample.get("domain", ""))
            by_length.setdefault(tier, {"direct": [], "harness": []})
            by_domain.setdefault(domain, {"direct": [], "harness": []})
            by_length[tier]["direct"].append(sample["direct"])
            by_length[tier]["harness"].append(sample["harness"][str(budget)])
            by_domain[domain]["direct"].append(sample["direct"])
            by_domain[domain]["harness"].append(sample["harness"][str(budget)])
        budget_summaries.append(
            {
                "budget": budget,
                "direct": stats([sample["direct"] for sample in samples]),
                "harness": stats(harness_rows),
                "modes": modes,
                "by_length": {
                    tier: {"direct": stats(values["direct"]), "harness": stats(values["harness"])}
                    for tier, values in by_length.items()
                },
                "by_domain": {
                    domain: {"direct": stats(values["direct"]), "harness": stats(values["harness"])}
                    for domain, values in by_domain.items()
                },
            }
        )
    summary = {
        "dataset": "LongBench-v2",
        "questions_total": len(selected),
        "questions_completed": len(samples),
        "context_tokens": quantiles([sample["context_tokens"] for sample in samples]),
        "context_chars": quantiles([sample["context_chars"] for sample in samples]),
        "config": {
            "budgets": budgets,
            "reserve_tokens": args.reserve_tokens,
            "chunk_chars": args.chunk_chars,
            "max_tokens": args.max_tokens,
            "candidate_max_tokens": args.candidate_max_tokens,
            "evidence_windows": args.evidence_windows,
            "evidence_window_chars": args.evidence_window_chars,
            "evidence_max_chars": args.evidence_max_chars,
            "harness_vote_candidates": args.harness_vote_candidates,
            "harness_option_verify": args.harness_option_verify,
            "harness_option_verify_tokens": args.harness_option_verify_tokens,
            "harness_option_verify_retries": args.harness_option_verify_retries,
            "reasoning_effort": args.reasoning_effort,
            "enable_thinking": args.enable_thinking,
            "jobs": args.jobs,
            "model": args.model,
            "calibration": str(args.calibration),
            "token_estimator": estimator.state(),
        },
        "elapsed_seconds": round(time.perf_counter() - started, 1),
        "by_budget": budget_summaries,
    }
    (args.outdir / "samples.json").write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    (args.outdir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = args.outdir / "samples.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "index", "id", "domain", "sub_domain", "difficulty", "length", "question", "question_chars",
            "answer", "context_chars", "context_words", "context_tokens", "context_tokens_source",
            "direct_answer", "direct_correct", "direct_error",
        ]
        for budget in budgets:
            fieldnames += [f"harness_{budget}_mode", f"harness_{budget}_answer", f"harness_{budget}_correct", f"harness_{budget}_error"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for sample in samples:
            row_out = {key: sample.get(key) for key in fieldnames if key in sample}
            row_out["question"] = " ".join(str(sample.get("question") or "").split())
            row_out["question_chars"] = len(str(sample.get("question") or ""))
            row_out["direct_answer"] = (sample.get("direct") or {}).get("prediction")
            row_out["direct_correct"] = (sample.get("direct") or {}).get("correct")
            row_out["direct_error"] = ((sample.get("direct") or {}).get("error") or {}).get("type") if (sample.get("direct") or {}).get("error") else ""
            for budget in budgets:
                harness = sample["harness"][str(budget)]
                row_out[f"harness_{budget}_mode"] = harness.get("mode")
                row_out[f"harness_{budget}_answer"] = harness.get("prediction")
                row_out[f"harness_{budget}_correct"] = harness.get("correct")
                row_out[f"harness_{budget}_error"] = (harness.get("error") or {}).get("type") if harness.get("error") else ""
            writer.writerow(row_out)
    print(json.dumps({"event": "summary_written", "path": str(args.outdir / "summary.json"), "completed": len(samples)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
