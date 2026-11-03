"""Compare direct Qwen answers with the local long-context Agent memory flow.

The script keeps the benchmark harness local.  It downloads only the selected
LongBench v2 rows through the public Hugging Face datasets-server API and calls
an OpenAI-compatible endpoint supplied by ``Q38_BASE_URL``/``Q38_API_KEY``.
It never writes to the development host; use an SSH local-forward when the
endpoint is only reachable there.

The Agent path is deliberately explicit:

1. Convert each raw LongBench context into 24k-character IR sections with
   source spans pointing back to the original context.
2. Let ``ReaderAgent`` read each section through a provider-backed
   ``SectionReader`` and persist local checkpoints in a temporary directory.
3. In ``reader`` mode, ask the provider to answer from section summaries and
   evidence anchors, with a small lexical retrieval of traceable raw windows.
   In ``harness`` mode, preserve the full prompt whenever it fits the model
   context and enter the reader path only when routing requires it.

This is a pilot harness, not a leaderboard submission.  It reports direct and
Agent predictions, answer accuracy, calls, latency, and Agent completion state.
"""
from __future__ import annotations

import argparse
import http.client
import json
from numbers import Number
import os
from dataclasses import dataclass
import re
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from independent_ingestion.document_ir import DocumentIR
from long_context_agent import LongContextHarness, ReaderAgent
from reading_memory import MemoryStateStore

try:
    from .longbench_v2_protocol import (
        LongBenchRow,
        chunked_document,
        evidence_windows,
        extract_choice,
        fetch_row,
        memory_prompt,
        memory_text,
        official_prompt,
    )
except ImportError:  # direct ``python scripts/…`` execution
    from scripts.longbench_v2_protocol import (
        LongBenchRow,
        chunked_document,
        evidence_windows,
        extract_choice,
        fetch_row,
        memory_prompt,
        memory_text,
        official_prompt,
    )


DEFAULT_OFFSETS = (1, 4, 6, 8, 9, 11, 13, 15, 20, 21)
_USE_DEFAULT_REASONING_EFFORT = object()
_EMPTY_CONTENT_RECOVERY_SUFFIX = (
    "\n\nFinal answer recovery instruction: the previous generation emitted internal "
    "reasoning without a final answer. Answer the original multiple-choice question "
    "now. Do not provide analysis, reasoning, or explanations. Return exactly one "
    'line in the format "The correct answer is (A)" (using A, B, C, or D).'
)


@dataclass(frozen=True)
class ChatResult:
    content: str
    reasoning: str
    usage: dict[str, Any]
    seconds: float
    # ``complete_answer`` may make a second, no-thinking request when a server
    # spends the whole budget on reasoning and returns an empty final channel.
    # Keep the metadata on the result so the benchmark can report this without
    # changing how predictions are scored.
    fallback_used: bool = False
    attempts: int = 1


class QwenHTTPClient:
    """Minimal OpenAI-compatible client with no SDK dependency."""

    def __init__(self, *, base_url: str, api_key: str, model: str, timeout: float, reasoning_effort: str | None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.reasoning_effort = reasoning_effort
        self.calls = 0

    def complete(
        self,
        prompt: str,
        *,
        max_tokens: int,
        enable_thinking: bool,
        preserve_thinking: bool = True,
        reasoning_effort: str | None | object = _USE_DEFAULT_REASONING_EFFORT,
    ) -> ChatResult:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "temperature": 0.1,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": bool(enable_thinking),
                    "preserve_thinking": bool(preserve_thinking),
                }
            },
        }
        # Some OpenAI-compatible Qwen servers treat reasoning_effort as an
        # instruction to think even when chat_template_kwargs disables
        # thinking.  Omit it in the no-thinking comparison mode so the
        # requested output budget remains available for the final answer.
        requested_effort = self.reasoning_effort if reasoning_effort is _USE_DEFAULT_REASONING_EFFORT else reasoning_effort
        # An explicit ``reasoning_effort`` override is also useful when the
        # caller disables thinking but needs a structured JSON response. Some
        # Qwen-compatible servers otherwise spend the whole short budget in a
        # hidden reasoning channel despite chat_template_kwargs=false.
        if requested_effort and (enable_thinking or reasoning_effort is not _USE_DEFAULT_REASONING_EFFORT):
            body["reasoning_effort"] = requested_effort
        request = Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        started = time.perf_counter()
        self.calls += 1
        last_error: BaseException | None = None
        for attempt in range(3):
            try:
                with urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
                break
            except (HTTPError, URLError, TimeoutError, OSError, http.client.HTTPException) as exc:
                last_error = exc
                status = getattr(exc, "code", None)
                retryable = status in {429, 500, 502, 503, 504} or status is None
                if attempt >= 2 or not retryable:
                    raise
                time.sleep(1.0 * (2**attempt))
        else:  # pragma: no cover - loop either breaks or raises above
            raise last_error or RuntimeError("model request failed")
        message = (payload.get("choices") or [{}])[0].get("message") or {}
        return ChatResult(
            content=str(message.get("content") or ""),
            # Qwen-compatible servers have used both names over time.  Keep a
            # single normalized reasoning field for diagnostics and recovery.
            reasoning=str(message.get("reasoning") or message.get("reasoning_content") or ""),
            usage=dict(payload.get("usage") or {}),
            seconds=time.perf_counter() - started,
        )

    @staticmethod
    def _merge_usage(*usages: Mapping[str, Any]) -> dict[str, Any]:
        """Add numeric usage counters while retaining non-numeric metadata."""
        merged: dict[str, Any] = {}
        for usage in usages:
            for key, value in usage.items():
                if isinstance(value, Number) and not isinstance(value, bool):
                    previous = merged.get(key)
                    if isinstance(previous, Number) and not isinstance(previous, bool):
                        merged[key] = previous + value
                    else:
                        merged[key] = value
                elif key not in merged:
                    merged[key] = value
        return merged

    def complete_answer(
        self,
        prompt: str,
        *,
        max_tokens: int,
        enable_thinking: bool,
        preserve_thinking: bool = True,
    ) -> ChatResult:
        """Return a final answer, recovering from reasoning-only responses.

        Some Qwen OpenAI-compatible servers can exhaust ``max_tokens`` in the
        reasoning channel before writing ``message.content``.  Scoring that
        response as a missing answer penalizes whichever path happened to hit
        the server's reasoning cutoff.  Both the direct and Agent paths call
        this helper, so recovery has identical semantics for the comparison.
        """
        primary = self.complete(
            prompt,
            max_tokens=max_tokens,
            enable_thinking=enable_thinking,
            preserve_thinking=preserve_thinking,
        )
        if primary.content.strip():
            return primary

        # A short, no-thinking retry leaves the full answer budget available
        # for the one-line choice and omits reasoning_effort explicitly.  Cap
        # it because LongBench only needs a choice, while respecting very small
        # caller budgets.
        retry_max_tokens = max(1, min(int(max_tokens), 2048))
        recovery = self.complete(
            prompt + _EMPTY_CONTENT_RECOVERY_SUFFIX,
            max_tokens=retry_max_tokens,
            enable_thinking=False,
            preserve_thinking=False,
            reasoning_effort=None,
        )
        reasoning_parts = [part for part in (primary.reasoning, recovery.reasoning) if part]
        return ChatResult(
            content=recovery.content,
            reasoning="\n".join(reasoning_parts),
            usage=self._merge_usage(primary.usage, recovery.usage),
            seconds=primary.seconds + recovery.seconds,
            fallback_used=True,
            attempts=2,
        )


def _json_object(text: str) -> dict[str, Any] | None:
    """Parse a JSON object from a provider response, including code fences."""
    cleaned = (text or "").strip()
    if not cleaned:
        return None
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
        return dict(value) if isinstance(value, Mapping) else None
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start < 0 or end <= start:
            return None
        try:
            value = json.loads(cleaned[start : end + 1])
            return dict(value) if isinstance(value, Mapping) else None
        except json.JSONDecodeError:
            return None


class LongBenchMemoryProvider:
    """Provider adapter used by ``ReaderAgent`` for one benchmark row."""

    def __init__(
        self,
        client: QwenHTTPClient,
        row: LongBenchRow,
        *,
        chunk_max_tokens: int,
        final_max_tokens: int,
        enable_thinking: bool,
        chunk_thinking: bool,
        chunk_chars: int = 24_000,
        evidence_window_count: int = 4,
        evidence_window_chars: int = 5_000,
        evidence_max_chars: int = 18_000,
    ):
        self.client = client
        self.row = row
        self.chunk_max_tokens = chunk_max_tokens
        self.final_max_tokens = final_max_tokens
        self.enable_thinking = enable_thinking
        self.chunk_thinking = chunk_thinking
        self.chunk_chars = chunk_chars
        self.evidence_window_count = evidence_window_count
        self.evidence_window_chars = evidence_window_chars
        self.evidence_max_chars = evidence_max_chars
        self.chunk_calls = 0
        self.chunk_results: list[dict[str, Any]] = []
        self._source_search_from = 0

    def complete(self, task: str, prompt: str, *, schema: Mapping[str, Any] | None = None, **_: Any) -> dict[str, Any]:
        if task != "text_reader":
            return {"summary": prompt[:1000], "anchors": []}
        index = self.chunk_calls
        self.chunk_calls += 1
        chunk_prompt = (
            "You are the chunk reader inside an evidence-aware long-document Agent. "
            "Read only the chunk below and preserve facts useful for answering the "
            "multiple-choice question. Return a JSON object with keys summary, "
            "anchors, conflicts, and missing_fields. Keep summary below 900 chars; "
            "anchors should contain at most five concise factual anchors.\n\n"
            f"Question: {self.row.question.strip()}\n"
            f"(A) {self.row.choice_a.strip()}\n(B) {self.row.choice_b.strip()}\n"
            f"(C) {self.row.choice_c.strip()}\n(D) {self.row.choice_d.strip()}\n\n"
            f"Chunk index: {index}\n<chunk>\n{prompt}\n</chunk>"
        )
        try:
            result = self.client.complete(
                chunk_prompt,
                max_tokens=self.chunk_max_tokens,
                enable_thinking=self.chunk_thinking,
                preserve_thinking=self.chunk_thinking,
            )
            parsed = _json_object(result.content)
            if parsed is None:
                parsed = {"summary": (result.content or result.reasoning)[-1000:], "anchors": []}
        except (HTTPError, URLError, TimeoutError) as exc:
            parsed = {"summary": f"chunk provider error: {type(exc).__name__}", "anchors": [], "conflicts": [str(exc)]}
        parsed.setdefault("summary", "")
        parsed.setdefault("anchors", [])
        parsed.setdefault("conflicts", [])
        parsed.setdefault("missing_fields", [])
        safe_anchors: list[dict[str, Any]] = []
        for anchor in parsed.get("anchors", []) or []:
            if not isinstance(anchor, Mapping):
                continue
            item = dict(anchor)
            item.setdefault("anchor_type", "benchmark_evidence")
            item.setdefault("label", f"chunk-{index}")
            item.setdefault("normalized_value", item.get("label", f"chunk-{index}"))
            item.setdefault("confidence", 0.5)
            item.setdefault("status", "candidate")
            # SectionReader adds a section-scoped evidence ref when the model
            # does not provide one.  Model-generated block ids are not stable
            # IR ids, so do not let them leak into the memory graph.
            item.pop("evidence_refs", None)
            safe_anchors.append(item)
        parsed["anchors"] = safe_anchors
        # Keep the exact bounded section text alongside its model summary.  It
        # is used only by the final lexical retrieval lane; checkpoints still
        # persist the compact structured MemoryPoint as before.
        # ReaderAgent normally visits fixed chunks in source order.  Resolve
        # the exact source offset from the text as a guard against future
        # retrieval-driven reordering; fall back to the call order when a
        # provider has transformed the section text.
        source_start = self.row.context.find(prompt, self._source_search_from)
        if source_start < 0:
            source_start = index * self.chunk_chars
        else:
            self._source_search_from = source_start + len(prompt)
        source_index = source_start // self.chunk_chars if self.chunk_chars else index
        self.chunk_results.append(
            {
                "index": source_index,
                "call_index": index,
                "summary": str(parsed.get("summary", ""))[:1000],
                "text": prompt,
                "start_char": source_start,
                "end_char": source_start + len(prompt),
            }
        )
        return parsed

    def seed_source_document(self, document: DocumentIR) -> None:
        """Index raw IR blocks for the harness without model summarization."""
        self.chunk_results = []
        blocks_by_id = {getattr(block, "block_id", ""): block for block in getattr(document, "blocks", []) or []}
        ordered_ids = getattr(document, "reading_order", []) or list(blocks_by_id)
        for call_index, block_id in enumerate(ordered_ids):
            block = blocks_by_id.get(block_id)
            text = str(getattr(block, "text", "")) if block is not None else ""
            if not text:
                continue
            spans = getattr(block, "source_spans", []) or []
            source_start = int(getattr(spans[0], "start_char", call_index * self.chunk_chars)) if spans else call_index * self.chunk_chars
            self.chunk_results.append(
                {
                    "index": source_start // self.chunk_chars if self.chunk_chars else call_index,
                    "call_index": call_index,
                    "summary": "",
                    "text": text,
                    "start_char": source_start,
                    "end_char": source_start + len(text),
                }
            )

    def answer(
        self,
        memory: Any,
        *,
        evidence_text: str | None = None,
        enable_thinking: bool | None = None,
        preserve_thinking: bool | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Answer from memory/evidence with an optional final-call policy.

        Harness overflow routes can contain enough retrieved evidence to answer
        with a short choice.  They may therefore disable extended reasoning
        without changing the old ReaderAgent defaults used by the baseline.
        """
        text = memory_text(memory)
        evidence = evidence_text
        if evidence is None:
            evidence = evidence_windows(
                self.row,
                self.chunk_results,
                max_windows=self.evidence_window_count,
                window_chars=self.evidence_window_chars,
                max_total_chars=self.evidence_max_chars,
            )
        use_thinking = self.enable_thinking if enable_thinking is None else bool(enable_thinking)
        preserve = use_thinking if preserve_thinking is None else bool(preserve_thinking)
        return self.client.complete_answer(
            memory_prompt(self.row, text, evidence),
            max_tokens=self.final_max_tokens if max_tokens is None else int(max_tokens),
            enable_thinking=use_thinking,
            preserve_thinking=preserve,
        )


def _error_payload(exc: BaseException) -> dict[str, Any]:
    body = ""
    if isinstance(exc, HTTPError):
        try:
            body = exc.read().decode("utf-8", "replace")[:2000]
        except Exception:
            body = ""
    return {"type": type(exc).__name__, "message": str(exc), "body": body}


def run_sample(row: LongBenchRow, client: QwenHTTPClient, args: argparse.Namespace) -> dict[str, Any]:
    item: dict[str, Any] = row.metadata()
    item["direct"] = {}
    item["agent"] = {}
    direct: ChatResult | None = None
    try:
        direct = client.complete_answer(
            official_prompt(row),
            max_tokens=args.max_tokens,
            enable_thinking=args.enable_thinking,
            preserve_thinking=args.enable_thinking,
        )
        direct_choice = extract_choice(direct.content)
        item["direct"] = {
            "prediction": direct_choice,
            "correct": direct_choice == row.answer,
            "answer_source": "content" if direct.content.strip() else "none",
            "reasoning_only_choice": extract_choice(direct.reasoning) if not direct.content.strip() else None,
            "seconds": round(direct.seconds, 3),
            "usage": direct.usage,
            "content_preview": direct.content[:1000],
            "reasoning_chars": len(direct.reasoning),
            "fallback_used": direct.fallback_used,
            "attempts": direct.attempts,
        }
    except Exception as exc:
        item["direct"] = {"error": _error_payload(exc)}

    calls_before_agent = client.calls
    provider = LongBenchMemoryProvider(
        client,
        row,
        chunk_max_tokens=args.chunk_max_tokens,
        final_max_tokens=args.final_max_tokens,
        enable_thinking=args.enable_thinking,
        chunk_thinking=args.chunk_thinking,
        chunk_chars=args.chunk_chars,
        evidence_window_count=args.evidence_windows,
        evidence_window_chars=args.evidence_window_chars,
        evidence_max_chars=args.evidence_max_chars,
    )
    started = time.perf_counter()
    try:
        document: DocumentIR = chunked_document(row, chunk_chars=args.chunk_chars)
        with tempfile.TemporaryDirectory(prefix="longbench-v2-agent-") as checkpoint_root:
            store = MemoryStateStore(checkpoint_root)
            reader = ReaderAgent(
                provider=provider,
                state_store=store,
                max_context_chars=args.chunk_chars,
            )
            harness_mode = "reader_agent"
            estimated_prompt_tokens = None
            if args.agent_mode == "harness":
                prepared = LongContextHarness(
                    reader,
                    max_context_tokens=args.model_context_tokens,
                    reserve_tokens=args.harness_reserve_tokens,
                    chars_per_token=args.harness_chars_per_token,
                    read_when_over_budget=False,
                ).prepare(document, full_prompt=official_prompt(row), resume=False)
                harness_mode = prepared.mode
                estimated_prompt_tokens = prepared.estimated_prompt_tokens
                memory, state = prepared.memory, prepared.state
            else:
                _, memory, state = reader.read(document, resume=False)

            # A direct-pass harness deliberately reuses the direct result. It
            # avoids asking the model to summarize a source it can already see
            # in full, and keeps this path a routing decision rather than a
            # second cognitive layer.
            if harness_mode == "direct_pass":
                final = direct or client.complete_answer(
                    official_prompt(row),
                    max_tokens=args.final_max_tokens,
                    enable_thinking=args.enable_thinking,
                    preserve_thinking=args.enable_thinking,
                )
            elif harness_mode == "raw_source":
                provider.seed_source_document(document)
                final = provider.answer(None, enable_thinking=False, preserve_thinking=False, max_tokens=2048)
            else:
                final = provider.answer(memory)
            agent_choice = extract_choice(final.content)
            item["agent"] = {
                "prediction": agent_choice,
                "correct": agent_choice == row.answer,
                "answer_source": "content" if final.content.strip() else "none",
                "reasoning_only_choice": extract_choice(final.reasoning) if not final.content.strip() else None,
                "seconds": round(time.perf_counter() - started, 3),
                "status": state.status.value if state is not None else "direct_pass",
                "harness_mode": harness_mode,
                "reused_direct": harness_mode == "direct_pass",
                "estimated_prompt_tokens": estimated_prompt_tokens,
                "visited_sections": len(state.visited_sections) if state is not None else 0,
                "completed_chunks": len(memory.completed_chunks) if memory is not None else 0,
                "chunk_calls": provider.chunk_calls,
                "model_calls": client.calls - calls_before_agent,
                "usage": final.usage,
                "content_preview": final.content[:1000],
                "reasoning_chars": len(final.reasoning),
                "memory_chars": len(memory_text(memory)) if memory is not None else 0,
                "evidence_chars": (
                    len(
                        evidence_windows(
                            row,
                            provider.chunk_results,
                            max_windows=args.evidence_windows,
                            window_chars=args.evidence_window_chars,
                            max_total_chars=args.evidence_max_chars,
                        )
                    )
                    if memory is not None or harness_mode == "raw_source"
                    else 0
                ),
                "fallback_used": final.fallback_used,
                "attempts": final.attempts,
            }
    except Exception as exc:
        item["agent"] = {"error": _error_payload(exc), "seconds": round(time.perf_counter() - started, 3), "chunk_calls": provider.chunk_calls}
    return item


def summarize(samples: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(key: str) -> dict[str, Any]:
        rows = [x.get(key, {}) for x in samples]
        scored = [x for x in rows if x.get("prediction")]
        missing = sum(1 for x in rows if x.get("answer_source") != "content")
        fallbacks = sum(bool(x.get("fallback_used")) for x in rows)
        correct = sum(bool(x.get("correct")) for x in scored)
        return {
            "attempted": len(rows),
            "scored": len(scored),
            "missing_final": missing,
            "fallbacks": fallbacks,
            "correct": correct,
            "accuracy": round(correct / len(scored), 4) if scored else None,
            "accuracy_all_attempts": round(correct / len(rows), 4) if rows else None,
        }

    return {"samples": len(samples), "direct": stats("direct"), "agent": stats("agent")}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offsets", nargs="+", type=int, default=list(DEFAULT_OFFSETS))
    parser.add_argument("--max-samples", type=int, default=10)
    parser.add_argument("--base-url", default=os.getenv("Q38_BASE_URL", "http://127.0.0.1:18016/v1"))
    parser.add_argument("--model", default=os.getenv("Q38_MODEL", "Qwen3.8-Flash-Next-FP8"))
    parser.add_argument("--api-key", default=os.getenv("Q38_API_KEY"))
    parser.add_argument("--chunk-chars", type=int, default=24_000)
    parser.add_argument(
        "--evidence-windows",
        type=int,
        default=4,
        help="maximum raw source windows appended to final Agent QA",
    )
    parser.add_argument(
        "--evidence-window-chars",
        type=int,
        default=5_000,
        help="maximum characters per retrieved source window",
    )
    parser.add_argument(
        "--evidence-max-chars",
        type=int,
        default=18_000,
        help="total character budget for retrieved source windows",
    )
    parser.add_argument("--chunk-max-tokens", type=int, default=4096)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--final-max-tokens", type=int, default=8192)
    parser.add_argument(
        "--agent-mode",
        choices=("reader", "harness"),
        default="harness",
        help="harness is model-first; reader keeps the original map-reduce baseline",
    )
    parser.add_argument("--model-context-tokens", type=int, default=262_144)
    parser.add_argument("--harness-reserve-tokens", type=int, default=8_192)
    parser.add_argument("--harness-chars-per-token", type=float, default=4.0)
    parser.add_argument("--reasoning-effort", default=os.getenv("Q38_REASONING_EFFORT", "xhigh"))
    parser.add_argument("--enable-thinking", action=argparse.BooleanOptionalAction, default=os.getenv("Q38_ENABLE_THINKING", "true").lower() == "true")
    parser.add_argument("--chunk-thinking", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--output", type=Path, default=Path("docs/longbench_v2_compare_10.json"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not args.api_key:
        raise SystemExit("Q38_API_KEY is required; pass --api-key or set it in the local shell")
    offsets = list(dict.fromkeys(args.offsets))[: max(0, args.max_samples)]
    client = QwenHTTPClient(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        timeout=args.timeout,
        reasoning_effort=args.reasoning_effort,
    )
    samples: list[dict[str, Any]] = []

    def write_report() -> dict[str, Any]:
        report = {
            "benchmark": "LongBench-v2",
            "mode": "direct_qwen_vs_reader_agent_memory",
            "config": {
                "model": args.model,
                "base_url": args.base_url,
                "chunk_chars": args.chunk_chars,
                "evidence_windows": args.evidence_windows,
                "evidence_window_chars": args.evidence_window_chars,
                "evidence_max_chars": args.evidence_max_chars,
                "chunk_max_tokens": args.chunk_max_tokens,
                "max_tokens": args.max_tokens,
                "final_max_tokens": args.final_max_tokens,
                "agent_mode": args.agent_mode,
                "model_context_tokens": args.model_context_tokens,
                "harness_reserve_tokens": args.harness_reserve_tokens,
                "harness_chars_per_token": args.harness_chars_per_token,
                "enable_thinking": args.enable_thinking,
                "chunk_thinking": args.chunk_thinking,
                "reasoning_effort": args.reasoning_effort,
                "empty_content_recovery": {
                    "enabled": True,
                    "thinking": False,
                    "max_tokens_cap": 2048,
                },
                "offsets": offsets,
            },
            "summary": summarize(samples),
            "samples": samples,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        return report

    for offset in offsets:
        try:
            row = fetch_row(offset)
            print(json.dumps({"event": "sample_start", **row.metadata()}, ensure_ascii=False), flush=True)
            result = run_sample(row, client, args)
        except Exception as exc:
            result = {"offset": offset, "error": _error_payload(exc)}
        samples.append(result)
        print(json.dumps({"event": "sample_done", **result}, ensure_ascii=False), flush=True)
        partial = write_report()
        print(json.dumps({"event": "partial_report", "path": str(args.output), "summary": partial["summary"]}, ensure_ascii=False), flush=True)
    report = write_report()
    print(json.dumps({"event": "report_written", "path": str(args.output), "summary": report["summary"]}, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
