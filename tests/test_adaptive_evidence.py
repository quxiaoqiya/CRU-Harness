from __future__ import annotations

import json
import re

from long_context_agent.adaptive_evidence import (
    AdaptiveConfig,
    EvidenceGap,
    EvidenceWindow,
    SourceIndex,
    run_adaptive,
    source_coverage,
    validate_supported_spans,
    _allocate,
)


class CappedFakeClient:
    def __init__(self, responder, *, calls=5, tokens=12000, context=1400):
        self.responder = responder
        self.remaining_calls = calls
        self.remaining_tokens = tokens
        self.context_tokens = context
        self.messages = []

    def count_text(self, text):
        return max(1, (len(text) + 3) // 4)

    def count_messages(self, messages):
        return 5 + sum(4 + self.count_text(m["content"]) for m in messages)

    def can_call(self, messages, max_tokens):
        needed = self.count_messages(messages) + max_tokens
        return self.remaining_calls > 0 and needed <= min(self.remaining_tokens, self.context_tokens)

    def complete(self, messages, max_tokens=1024):
        assert self.can_call(messages, max_tokens)
        self.messages.append(messages)
        response = self.responder(messages, len(self.messages))
        self.remaining_calls -= 1
        self.remaining_tokens -= self.count_messages(messages) + self.count_text(response)
        assert self.remaining_tokens >= 0
        return response


def _header(messages):
    return re.findall(r"\[(source_chars \d+:\d+)", messages[-1]["content"])[0]


CONFIG = AdaptiveConfig(block_chars=80, halo_chars=0, max_windows_per_round=1, max_rounds=3, review_output_tokens=128, answer_output_tokens=64)


def test_followup_gap_changes_retrieved_source_and_resolves_dependency():
    context = "start lookup: the next identifier is QUARTZ77.\n".ljust(80) + "irrelevant padding. ".ljust(80) * 225 + "QUARTZ77 record: the final destination is coral."

    def respond(messages, number):
        if number == 1:
            return json.dumps({"answer": "", "needs": [{"id": "destination", "query": "QUARTZ77 record final destination", "reason": "Resolve the identifier learned from the first source"}], "supported_spans": [_header(messages)], "complete": False})
        if number == 2:
            assert "final destination is coral" in messages[-1]["content"]
            return json.dumps({"answer": "coral", "needs": [], "supported_spans": [_header(messages)], "complete": True})
        return "coral"

    client = CappedFakeClient(respond)
    result = run_adaptive({"id": "new-task", "context": context, "question": "Resolve the start lookup."}, client, config=CONFIG)
    reviews = [t for t in result["trace"] if t.get("action") == "review"]
    assert result["answer"] == "coral"
    assert result["evidence_complete"]
    assert reviews[1]["gaps_before"][0]["query"] == "QUARTZ77 record final destination"
    assert reviews[0]["new_spans"] != reviews[1]["new_spans"]
    assert reviews[0]["allocation"][0]["tokens"] > 0
    assert result["stop_reason"] == "evidence_complete"


def test_invalid_source_claim_cannot_complete_or_resolve_a_gap():
    def respond(messages, number):
        if "Return one JSON object" not in messages[0]["content"]:
            return "tentative"
        return json.dumps({"answer": "tentative", "needs": [], "supported_spans": ["source_chars 999999:1000000"], "complete": True})

    client = CappedFakeClient(respond, calls=3)
    result = run_adaptive({"context": "source fact. " * 2000, "question": "What is the source fact?"}, client, config=CONFIG)
    assert result["evidence_complete"] is False
    assert result["supported_spans"] == []
    review = result["trace"][0]
    assert review["invalid_spans"] == ["source_chars 999999:1000000"]
    assert not review["resolved_gap_ids"]
    assert any(g["id"] == "source_provenance" for g in review["gaps_after"])


def test_last_call_is_reserved_for_an_answer_not_another_review():
    def respond(messages, number):
        if number == 1:
            return json.dumps({"answer": "", "needs": [{"id": "missing", "query": "other evidence", "reason": "still missing"}], "supported_spans": [], "complete": False})
        assert "Return only the requested answer" in messages[0]["content"]
        return "best answer"

    client = CappedFakeClient(respond, calls=2)
    result = run_adaptive({"context": "padding information. " * 1000, "question": "Find a specific fact."}, client, config=CONFIG)
    assert result["answer"] == "best answer"
    assert len(client.messages) == 2
    assert result["stop_reason"] == "final_call_reserved"
    assert result["trace"][-1]["action"] == "final_answer"


def test_small_aggregate_budget_does_not_spend_unavailable_tokens():
    client = CappedFakeClient(lambda messages, number: "answer", calls=3, tokens=1100)
    result = run_adaptive({"context": "irrelevant. " * 2000, "question": "Name the item."}, client, config=CONFIG)
    assert client.remaining_tokens >= 0
    assert client.remaining_calls >= 0
    assert result["trace"][-1]["action"] in {"final_answer", "final_answer_unavailable"}


def test_retrieval_excludes_seen_spans_and_finalizes_when_no_novelty():
    index = SourceIndex("first key here. second key there.", block_chars=80, halo_chars=0)
    first = index.rank(EvidenceGap("key", "key"))
    assert first
    assert not index.rank(EvidenceGap("key", "key"), {(first[0].start, first[0].end)})

    client = CappedFakeClient(lambda messages, number: json.dumps({"answer": "", "needs": [], "supported_spans": [], "complete": False}) if number == 1 else "unknown", context=850)
    # The question alone makes direct input exceed the small per-call window;
    # the empty source then has no retrievable novelty.
    result = run_adaptive({"context": "", "question": "q " * 2000}, client, config=CONFIG)
    assert result["stop_reason"] in {"token_budget_reserved", "no_novel_evidence"}
    assert client.remaining_tokens >= 0


def test_full_input_passes_through_without_a_review():
    client = CappedFakeClient(lambda messages, number: "42")
    result = run_adaptive({"context": "The value is 42.", "question": "What is the value?"}, client, config=CONFIG)
    assert result["mode"] == "direct_pass"
    assert result["answer"] == "42"
    assert len(client.messages) == 1


def test_span_must_equal_an_exposed_header_not_just_be_in_document():
    windows = [EvidenceWindow(10, 50, "raw", "gap", 1)]
    valid, invalid = validate_supported_spans(["source_chars 10:50", "source_chars 11:49", {"start": 0, "end": 100}], windows)
    assert valid == ["source_chars 10:50"]
    assert len(invalid) == 2


def test_high_priority_gap_receives_more_evidence_without_starving_other_gap():
    index = SourceIndex("alpha source. ".ljust(80) * 4 + "beta source. ".ljust(80) * 4, block_chars=80, halo_chars=0)
    client = CappedFakeClient(lambda messages, number: "")
    selected, allocations = _allocate(index, [EvidenceGap("alpha", "alpha", kind="conflict", priority=3), EvidenceGap("beta", "beta", priority=1)], set(), client, 180, 10)
    counts = {g: sum(w.gap_id == g for w in selected) for g in ("alpha", "beta")}
    assert allocations[0]["tokens"] > allocations[1]["tokens"]
    assert counts["alpha"] > counts["beta"] >= 1
    assert sum(client.count_text(w.render()) for w in selected) <= 180


def test_aggregate_claim_cannot_complete_without_full_source_coverage():
    def respond(messages, number):
        if "Return one JSON object" in messages[0]["content"]:
            return json.dumps({"answer": "12", "needs": [], "supported_spans": [_header(messages)], "resolved_gap_ids": ["question"], "complete": True})
        return "12"

    client = CappedFakeClient(respond, calls=3)
    result = run_adaptive({"context": "arbitrary source records. ".ljust(80) * 240, "question": "How many records are in the whole source?"}, client, config=CONFIG)
    assert result["coverage_required"]
    assert result["coverage"]["fraction"] < 1
    assert not result["evidence_complete"]
    assert any(g["kind"] == "coverage" for g in result["unresolved_gaps"])
    assert not result["trace"][0]["resolved_gap_ids"]


def test_coverage_uses_union_of_spans_and_reports_unread_ranges():
    coverage = source_coverage([(0, 10), (8, 20), (30, 40)], 50)
    assert coverage["covered_chars"] == 30
    assert coverage["unread_ranges"] == [[20, 30], [40, 50]]
    assert coverage["fraction"] == 0.6


def test_coverage_retrieval_spreads_over_distant_source_bins():
    index = SourceIndex("generic source. ".ljust(80) * 80, block_chars=80, halo_chars=0)
    windows = index.rank(EvidenceGap("all", "generic", kind="coverage"))[:8]
    assert len({w.start // 800 for w in windows}) == 8
