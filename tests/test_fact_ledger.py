from datetime import date
import hashlib
import json

import pytest

from long_context_agent.fact_ledger import FactLedger, normalize_date, parse_records


def log(rows):
    return f"The following lines contain {len(rows)} records, one per line.\n\n" + "".join(
        f"Date: {day} || User: {user} || Instance: {body}\n" for day, user, body in rows)


def sample():
    return parse_records(log([
        ("Jan 01, 2024", "7", "first"),
        ("Jan 02, 2024", "8", "second"),
        ("Jan 03, 2024", "7", "third"),
        ("Jan 04, 2024", "9", "fourth"),
    ]))


def test_parser_binds_exact_offsets_and_does_not_infer_labels():
    text = log([("Jan 01, 2024", "7", "Contains || a delimiter"), ("Jan 02, 2024", "8", "second")])
    data = parse_records(text)
    assert data.complete and len(data.records) == 2
    assert data.source_sha256 == hashlib.sha256(text.encode()).hexdigest()
    for record in data.records:
        assert record.text == text[record.start:record.end]
        assert "label" not in record.fields
    assert data.records[0].fields["body"] == "Contains || a delimiter"
    assert data.records[0].fields["date_iso"] == "2024-01-01"
    assert data.records[0].record_id == parse_records(text).records[0].record_id


def test_overlapping_windows_same_record_is_counted_once():
    data = sample()
    ledger = FactLedger(data)
    record = data.records[0]
    assert ledger.add(record.record_id, "positive", source_span=record.source_span)["status"] == "accepted"
    assert ledger.add(record.record_id, "Positive", source_span=(record.start, record.end),
                      provenance={"window": "overlapping second exposure"})["status"] == "duplicate"
    result = ledger.aggregate("count", target="positive")
    assert result["known_count"] == 1 and result["unknown_count"] == 3
    assert result["lower_bound"] == 1 and result["upper_bound"] == 4
    assert result["value"] is None and not result["complete"]


def test_same_text_at_distinct_positions_is_still_distinct_physical_record():
    text = log([("Jan 01, 2024", "7", "same"), ("Jan 01, 2024", "7", "same")])
    data = parse_records(text)
    assert len({r.record_id for r in data.records}) == 2
    assert FactLedger(data).aggregate("count")["value"] == 2


def test_conflict_stays_unknown_after_repeated_majority_label():
    data = sample()
    ledger = FactLedger(data)
    rid = data.records[0].record_id
    ledger.add(rid, "positive")
    assert ledger.add(rid, "negative")["status"] == "conflict"
    assert ledger.add(rid, "positive")["status"] == "conflict"
    result = ledger.aggregate("group_counts", labels=["positive", "negative"])
    assert result["known_count"] == 0 and result["conflict_count"] == 1
    assert result["unknown_count"] == 4 and not result["complete"]


def test_wrong_source_hash_or_span_cannot_enter_facts():
    data = sample()
    ledger = FactLedger(data)
    record = data.records[0]
    assert ledger.add(record.record_id, "yes", source_sha256="another-source")["status"] == "rejected"
    assert ledger.add(record.record_id, "yes", source_span=(record.start, record.end - 1))["status"] == "rejected"
    assert ledger.add("nonexistent", "yes")["status"] == "rejected"
    assert ledger.facts == {}


def test_filtered_scope_complete_does_not_require_unrelated_record_labels():
    data = sample()
    ledger = FactLedger(data)
    selected = data.select({"user_ids": [7]})
    assert len(selected) == 2
    for record in selected:
        ledger.add(record.record_id, "a")
    result = ledger.aggregate("group_counts", filters={"user": "7"}, labels=["a", "b"])
    assert result["complete"] and result["coverage_complete"]
    assert result["counts"] == {"a": 2, "b": 0}
    assert result["known_count"] == 2
    assert ledger.aggregate("group_counts", labels=["a", "b"])["unknown_count"] == 2


def test_invalid_date_or_missing_filter_fields_do_not_silently_exclude():
    data = parse_records('{"user":"7","date":"not-a-date","text":"one"}\n'
                         '{"date":"2024-01-01","text":"two"}\n')
    scope = data.scope({"user": "7", "date_before": "2024-02-01"})
    assert len(scope.uncertain_records) == 2
    result = FactLedger(data).aggregate("count", filters={"user": "7"})
    assert result["lower_bound"] == 1 and result["upper_bound"] == 2
    assert not result["complete"]
    with pytest.raises(ValueError):
        data.select({"date_before": "nonsense"})


def test_malformed_data_or_declared_record_shortfall_cannot_close_scope():
    text = "The following lines contain 3 records.\nDate: Jan 01, 2024 || User: 7 || Instance: first\nMALFORMED DATA\n"
    data = parse_records(text)
    assert not data.complete and data.unparsed_ranges
    ledger = FactLedger(data)
    ledger.add(data.records[0].record_id, "a")
    result = ledger.aggregate("count", target="a", filters={"user": "7"})
    assert result["value"] is None and not result["complete"]
    assert result["upper_bound"] is None
    short = parse_records("The following lines contain 2 records.\nDate: Jan 01, 2024 || User: 7 || Instance: one\n")
    assert not short.complete and short.scope().unindexed_count == 1


def test_unknown_target_count_is_not_zero():
    ledger = FactLedger(sample())
    result = ledger.aggregate("count", labels=["a", "b"], target="a")
    assert result["counts"]["a"] == 0
    assert result["lower_bound"] == 0 and result["upper_bound"] == 4
    assert result["value"] is None and result["unknown_count"] == 4


def test_mode_can_close_from_bounds_but_not_from_sample_majority_alone():
    data = sample()
    ledger = FactLedger(data)
    ledger.add(data.records[0].record_id, "a")
    assert not ledger.aggregate("mode", labels=["a", "b"])["complete"]
    ledger.add(data.records[1].record_id, "a")
    assert not ledger.aggregate("mode", labels=["a", "b"])["complete"]
    ledger.add(data.records[2].record_id, "a")
    result = ledger.aggregate("mode", labels=["a", "b"])
    assert result["complete"] and result["value"] == "a"
    assert not result["coverage_complete"] and result["unknown_count"] == 1


def test_least_common_uses_declared_taxonomy_including_zero_categories():
    data = sample()
    ledger = FactLedger(data)
    for record, label in zip(data.records, ["a", "a", "b", "b"]):
        ledger.add(record.record_id, label)
    result = ledger.aggregate("least_common", labels=["a", "b", "c"])
    assert result["complete"] and result["value"] == "c"
    assert result["counts"] == {"a": 2, "b": 2, "c": 0}


def test_compare_counts_uses_unknown_mass_and_can_certify_unequal_counts():
    data = sample()
    ledger = FactLedger(data)
    for record in data.records[:2]:
        ledger.add(record.record_id, "a")
    assert not ledger.aggregate("compare_counts", compare=["a", "b"])["complete"]
    ledger.add(data.records[2].record_id, "a")
    result = ledger.aggregate("compare_counts", compare=["a", "b"])
    assert result["complete"] and result["value"] == "more common than"
    ledger.add(data.records[3].record_id, "b")
    assert ledger.aggregate("count", target="b")["value"] == 1


def test_frequency_and_strict_temporal_filters():
    data = sample()
    ledger = FactLedger(data)
    for record, label in zip(data.records[:2], ["a", "b"]):
        ledger.add(record.record_id, label)
    partial = ledger.aggregate("frequency", target="a")
    assert partial["lower_bound"] == .25 and partial["upper_bound"] == .75
    assert partial["value"] is None
    before = ledger.aggregate("frequency", filters={"date_before": "2024-01-03"}, target="a")
    assert before["complete"] and before["value"] == .5
    assert len(data.select({"date_after": "2024-01-03"})) == 1
    assert len(data.select({"date_from": "2024-01-03", "date_to": "2024-01-04"})) == 2


def test_earliest_is_complete_only_after_unknown_prefix_is_resolved():
    data = sample()
    ledger = FactLedger(data)
    ledger.add(data.records[1].record_id, "target")
    pending = ledger.aggregate("earliest", target="target")
    assert not pending["complete"] and pending["candidate_value"] == "2024-01-02"
    ledger.add(data.records[0].record_id, "other")
    result = ledger.aggregate("earliest", target="target")
    assert result["complete"] and result["value"] == "2024-01-02"
    assert not result["coverage_complete"] and result["unknown_count"] == 2


def test_latest_and_date_only_do_not_require_unneeded_classifications():
    data = sample()
    ledger = FactLedger(data)
    assert ledger.aggregate("earliest")["value"] == "2024-01-01"
    ledger.add(data.records[-2].record_id, "target")
    assert not ledger.aggregate("latest", target="target")["complete"]
    ledger.add(data.records[-1].record_id, "other")
    assert ledger.aggregate("latest", target="target")["value"] == "2024-01-03"


def test_numeric_aggregates_are_deterministic_and_partial_is_not_exact():
    data = parse_records('{"user":7,"value":2,"text":"one"}\n{"user":7,"value":4,"text":"two"}\n')
    ledger = FactLedger(data)
    for op, expected in [("sum", 6), ("mean", 3), ("min", 2), ("max", 4)]:
        result = ledger.aggregate(op, field="value")
        assert result["complete"] and result["value"] == expected
    unknown = FactLedger(sample())
    unknown.add(unknown.collection.records[0].record_id, 10, field="value")
    result = unknown.aggregate("sum", field="value")
    assert not result["complete"] and result["value"] is None
    assert result["partial_value"] == 10
    assert result["upper_bound"] is None and result["lower_bound"] is None


def test_source_explicit_label_conflict_is_not_overridden_by_model():
    data = parse_records('{"label":"a","text":"one"}\n')
    ledger = FactLedger(data)
    assert ledger.aggregate("count", target="a")["value"] == 1
    assert ledger.add(data.records[0].record_id, "b")["status"] == "conflict"
    assert not ledger.aggregate("group_counts")["complete"]


def test_invalid_values_rejected_compact_state_and_closed_empty_scope():
    data = sample()
    ledger = FactLedger(data)
    assert ledger.add(data.records[0].record_id, float("nan"))["status"] == "rejected"
    ledger.add(data.records[0].record_id, "a")
    assert ledger.aggregate("count", filters={"user": "absent"})["value"] == 0
    state = ledger.state(compact=True)
    assert "scope_record_ids" not in state and "facts" not in state
    assert state["fact_samples"] and state["fact_digest"]
    json.dumps(state, allow_nan=False)
    assert normalize_date(date(2024, 1, 2)) == "2024-01-02"


def test_out_of_taxonomy_value_can_be_revisited():
    data = sample()
    ledger = FactLedger(data)
    ledger.add(data.records[0].record_id, "outside")
    assert ledger.unresolved_records(labels=["a", "b"])[0].record_id == data.records[0].record_id


def test_case_variants_across_records_share_one_semantic_label():
    data = sample()
    ledger = FactLedger(data)
    for record, label in zip(data.records, ["Positive", "positive", "POSITIVE", "negative"]):
        ledger.add(record.record_id, label)
    assert ledger.aggregate("count", target="POSITIVE")["value"] == 3
    assert ledger.aggregate("mode")["value"] == "positive"
    assert ledger.aggregate("compare_counts", compare=["Positive", "Negative"])["value"] == "more common than"


def test_malformed_source_records_before_first_valid_record_cannot_be_ignored():
    data = parse_records("Date: malformed record\nDate: 2024-01-02 || User: 7 || Instance: valid\n")
    assert len(data.records) == 1 and not data.complete and data.unparsed_ranges
    assert not FactLedger(data).aggregate("count")["complete"]
    nonfinite = parse_records('{"value":NaN,"text":"invalid"}\n{"value":1,"text":"valid"}\n')
    assert len(nonfinite.records) == 1 and not nonfinite.complete


def test_unsupported_filter_operator_cannot_silently_match_every_record():
    with pytest.raises(ValueError):
        sample().select({"user": {"contains": "7"}})
    with pytest.raises(ValueError):
        sample().select({"body": {"contains": "first"}})
