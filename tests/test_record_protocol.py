from dataclasses import replace
import hashlib
import json

import pytest

from long_context_agent.fact_ledger import FactLedger, parse_records
from long_context_agent.record_protocol import (
    ProjectionError, body_projection, decode_classifications, project_records,
)


def records(bodies=("first", "second")):
    return parse_records("".join(f"Date: 2024-01-0{i + 1} || User: 987654321 || Instance: {body}\n"
                                 for i, body in enumerate(bodies))).records


def test_projection_exposes_only_alias_and_complete_body():
    source_records = records()
    projection = project_records(source_records, labels=["positive", "negative"])
    assert projection.rows == [{"id": "r0", "text": "first"}, {"id": "r1", "text": "second"}]
    assert "987654321" not in projection.wire_text and "2024-01" not in projection.wire_text
    assert "source_chars" not in projection.wire_text
    assert projection.wire_sha256 == hashlib.sha256(projection.wire_text.encode()).hexdigest()
    for alias, record in zip(("r0", "r1"), source_records):
        ref = projection.bindings[alias]
        assert ref["source_span"] == record.source_span and ref["source_sha256"] == record.source_sha256
        start, end = ref["body_start"] - record.start, ref["body_end"] - record.start
        assert record.text[start:end] == record.fields["body"]
        assert ref["projection_sha256"] == projection.wire_sha256


def test_short_body_cannot_bind_to_an_identical_user_header_value():
    record = parse_records("Date: 2024-01-01 || User: 7 || Instance: 7\n").records[0]
    body, (start, end) = body_projection(record)
    assert body == "7" and start > record.text.index("Instance:")
    assert record.text[start - record.start:end - record.start] == "7"


def test_long_multiline_and_unicode_body_is_never_truncated():
    body = "long " * 7000 + "\nsecond\x85line\u2028end"
    projection = project_records(records((body, "last")), labels=["a"])
    assert projection.rows[0]["text"] == body
    assert projection.audit()["body_characters"] == len(body) + len("last")


def test_json_body_span_is_exact_or_explicitly_rejected():
    plain = parse_records('{"user":"first","text":"first"}\n').records[0]
    assert body_projection(plain)[1][0] == plain.text.index('"text":"') + len('"text":"')
    escaped = parse_records('{"text":"first\\nsecond"}\n').records[0]
    with pytest.raises(ProjectionError, match="body_not_verbatim_source"):
        project_records([escaped])
    marker = parse_records('{"text":"x || Instance: y"}\n').records[0]
    assert project_records([marker]).rows[0]["text"] == "x || Instance: y"


def test_projection_rejects_unbound_payload_and_malformed_record_range():
    record = records()[0]
    with pytest.raises(ProjectionError, match="body_not_verbatim_source"):
        project_records([replace(record, fields={"body": "not in source"})])
    with pytest.raises(ProjectionError, match="invalid_record_span"):
        project_records([replace(record, end=record.end + 1)])
    with pytest.raises(ProjectionError, match="missing_string_body"):
        project_records([replace(record, fields={})])


def test_duplicate_identity_and_mixed_documents_reject_projection():
    record = records()[0]
    with pytest.raises(ProjectionError, match="duplicate_source_record"):
        project_records([record, record])
    with pytest.raises(ProjectionError, match="mixed_source_documents"):
        project_records([record, replace(records()[1], source_sha256="another")])


def test_compact_values_bind_position_and_identity_locally():
    source = records()
    projection = project_records(source, labels=["a", "b", "c"])
    decoded = decode_classifications({"values": [2, 0]}, projection)
    assert not decoded["rejected"] and not decoded["batch_rejected"]
    assert [item["value"] for item in decoded["items"]] == ["c", "a"]
    for item, record in zip(decoded["items"], source):
        assert item["source_span"] == record.source_span
        assert item["record_id"] == record.record_id
        assert item["provenance"]["body_sha256"] == hashlib.sha256(record.fields["body"].encode()).hexdigest()


@pytest.mark.parametrize("values", [[0], [0, 1, 0], "0,1", None])
def test_length_mismatch_rejects_whole_vector_without_alignment_repair(values):
    projection = project_records(records(), labels=["a", "b"])
    decoded = decode_classifications({"values": values}, projection)
    assert decoded["batch_rejected"] and decoded["items"] == []
    assert len(decoded["rejected"]) == 2
    assert {error["reason"] for error in decoded["rejected"]} == {"vector_length_mismatch"}


@pytest.mark.parametrize("value,reason", [(-1, "taxonomy_index_out_of_range"), (2, "taxonomy_index_out_of_range"),
                                          (True, "taxonomy_index_not_integer"), (1.0, "taxonomy_index_not_integer"),
                                          ("a", "taxonomy_index_not_integer")])
def test_invalid_index_is_explicit_and_other_positions_do_not_shift(value, reason):
    projection = project_records(records(), labels=["a", "b"])
    decoded = decode_classifications({"values": [value, 1]}, projection)
    assert [(item["id"], item["value"]) for item in decoded["items"]] == [("r1", "b")]
    error = decoded["rejected"][0]
    assert error["reason"] == reason and error["index"] == 0 and error["alias"] == "r0"
    assert error["record_id"] == projection.bindings["r0"]["record_id"]


def test_null_stays_unknown_with_diagnostic_and_does_not_become_other():
    collection = parse_records("Date: 2024-01-01 || User: 1 || Instance: one\n")
    projection = project_records(collection.records, labels=["a", "__other__"])
    decoded = decode_classifications({"values": [None]}, projection)
    assert decoded["items"][0]["value"] is None
    assert decoded["rejected"][0]["reason"] == "unknown_value"
    item = decoded["items"][0]
    ledger = FactLedger(collection)
    ledger.add(item["record_id"], item["value"], source_span=item["source_span"], provenance=item["provenance"])
    assert ledger.aggregate("group_counts")["unknown_count"] == 1


@pytest.mark.parametrize("value,accepted", [(3, True), (2.5, True), (None, True),
                                          (False, False), ("3", False), (float("nan"), False), (float("inf"), False)])
def test_numeric_fields_require_typed_finite_numbers(value, accepted):
    projection = project_records(records(("one",)), field="value")
    decoded = decode_classifications({"values": [value]}, projection)
    assert bool(decoded["items"]) == accepted
    json.dumps(decoded, allow_nan=False)


def test_legacy_without_span_uses_binding_and_supplied_wrong_span_rejects():
    projection = project_records(records(), labels=["a", "b"])
    result = decode_classifications({"records": [{"id": "r0", "value": "A"},
                                                 {"id": "r1", "value": "b", "source_span": "source_chars 0:1"}]}, projection)
    assert result["items"][0]["source_span"] == projection.bindings["r0"]["source_span"]
    assert result["items"][0]["value"] == "a"
    assert result["rejected"][0]["reason"] == "source_span_mismatch"


def test_legacy_duplicate_aliases_are_all_rejected_and_missing_is_explicit():
    projection = project_records(records(), labels=["a", "b"])
    result = decode_classifications({"records": [{"id": "r0", "value": "a"},
                                                 {"id": "r0", "value": "b"},
                                                 {"id": "invented", "value": "a"}]}, projection)
    assert result["items"] == []
    assert [e["reason"] for e in result["rejected"]] == ["duplicate_alias", "duplicate_alias", "unknown_alias", "missing_alias"]


@pytest.mark.parametrize("spoof", [{"record_id": "invented"}, {"source_sha256": "another"}])
def test_optional_legacy_identity_claims_cannot_override_binding(spoof):
    projection = project_records(records(("one",)), labels=["a"])
    assert not decode_classifications({"records": [{"id": "r0", "value": "a", **spoof}]}, projection)["items"]


def test_schema_changes_and_projection_tampering_fail_closed():
    projection = project_records(records(), labels=["a", "b"])
    assert decode_classifications({"values": [0, 1]}, projection, labels=["b", "a"])["batch_rejected"]
    assert decode_classifications({"values": [0, 1]}, replace(projection, wire_text="[]"))["batch_rejected"]
    assert decode_classifications({"values": [0, 1]}, replace(projection, bindings_json="{}"))["batch_rejected"]
    with pytest.raises(ProjectionError, match="ambiguous_taxonomy"):
        project_records(records(), labels=["A", "a"])


def test_projection_properties_do_not_allow_mutating_local_identity():
    projection = project_records(records(), labels=["a", "b"])
    projection.rows[0]["text"] = "tampered"
    projection.bindings["r0"]["record_id"] = "tampered"
    assert projection.rows[0]["text"] == "first"
    assert decode_classifications({"values": [0, 1]}, projection)["items"][0]["record_id"] != "tampered"


def test_wire_and_reply_reduce_structural_cost_without_losing_body():
    source = records(("first", "second"))
    projection = project_records(source, labels=["a", "b"])
    old_input = json.dumps([{"id": f"r{i}", "source_span": r.source_span, "text": r.text} for i, r in enumerate(source)])
    old_reply = json.dumps({"records": [{"id": f"r{i}", "source_span": r.source_span, "value": "a"} for i, r in enumerate(source)]})
    assert len(projection.wire_text) < len(old_input) / 2
    assert len(json.dumps({"values": [0, 0]})) < len(old_reply) / 3


def test_ambiguous_or_wrong_top_level_reply_is_not_silently_reinterpreted():
    projection = project_records(records(), labels=["a", "b"])
    for reply in ([0, 1], "not JSON", {"values": [0, 1], "records": []}):
        assert decode_classifications(reply, projection)["batch_rejected"]


def test_preferred_pairs_bind_reordered_values_by_alias_not_position():
    projection = project_records(records(), labels=["a", "b", "c"])
    result = decode_classifications({"records": [["r1", 2], ["r0", 0]]}, projection)
    assert result["format"] == "compact_alias_pairs_v1" and not result["rejected"]
    assert [(item["id"], item["value"]) for item in result["items"]] == [("r1", "c"), ("r0", "a")]
    assert result["items"][0]["record_id"] == projection.bindings["r1"]["record_id"]
    assert result["items"][0]["provenance"]["protocol"] == "compact_alias_pairs_v1"


def test_omitted_pair_leaves_only_that_alias_unknown_without_realignment():
    projection = project_records(records(), labels=["a", "b"])
    result = decode_classifications({"records": [["r1", 1]]}, projection)
    assert not result["batch_rejected"]
    assert [(item["id"], item["value"]) for item in result["items"]] == [("r1", "b")]
    assert [(error["reason"], error["alias"]) for error in result["rejected"]] == [("missing_alias", "r0")]


def test_duplicate_pairs_reject_every_vote_for_that_alias():
    projection = project_records(records(), labels=["a", "b"])
    result = decode_classifications({"records": [["r0", 0], ["r1", 1], ["r0", 0]]}, projection)
    assert [item["id"] for item in result["items"]] == ["r1"]
    assert [error["reason"] for error in result["rejected"]] == ["duplicate_alias", "duplicate_alias"]


@pytest.mark.parametrize("value,reason", [(9, "taxonomy_index_out_of_range"),
                                          ("a", "taxonomy_index_not_integer"), (True, "taxonomy_index_not_integer"),
                                          (None, "unknown_value")])
def test_pair_invalid_or_unknown_value_never_becomes_other(value, reason):
    projection = project_records(records(), labels=["a", "__other__"])
    result = decode_classifications({"records": [["r0", value], ["r1", 0]]}, projection)
    assert result["rejected"][0]["reason"] == reason
    assert result["items"][-1]["id"] == "r1" and result["items"][-1]["value"] == "a"
    if value is not None:
        assert len(result["items"]) == 1


def test_pair_format_empty_and_malformed_rows_are_explicit():
    projection = project_records(records(), labels=["a", "b"])
    empty = decode_classifications({"records": []}, projection)
    assert empty["format"] == "compact_alias_pairs_v1"
    assert [error["reason"] for error in empty["rejected"]] == ["missing_alias", "missing_alias"]
    malformed = decode_classifications({"records": [["r0", 0, "extra"], [1, 0], ["invented", 0], ["r1", 1]]}, projection)
    assert [item["id"] for item in malformed["items"]] == ["r1"]
    assert [error["reason"] for error in malformed["rejected"]] == ["invalid_alias_pair", "alias_not_string", "unknown_alias", "missing_alias"]
    mixed = decode_classifications({"records": [["r0", 0], {"id": "r1", "value": "b"}]}, projection)
    assert mixed["batch_rejected"] and mixed["rejected"][0]["reason"] == "mixed_record_formats"


def test_numeric_alias_pairs_keep_typed_numeric_values_and_legacy_labels_stay_compatible():
    projection = project_records(records(), field="value")
    result = decode_classifications({"records": [["r1", 2.5], ["r0", 4]]}, projection)
    assert [item["value"] for item in result["items"]] == [2.5, 4]
    labels = project_records(records(), labels=["a", "b"])
    legacy = decode_classifications({"records": [{"id": "r0", "value": "a"}, {"id": "r1", "value": "B"}]}, labels)
    assert legacy["format"] == "legacy_records_v1" and not legacy["rejected"]


def test_sixty_four_pairs_can_recover_valid_records_after_an_omission():
    source = records(tuple(f"example {i}" for i in range(64)))
    projection = project_records(source, labels=["a", "b"])
    reply = {"records": [[f"r{i}", i % 2] for i in range(64) if i != 7]}
    result = decode_classifications(reply, projection)
    assert len(result["items"]) == 63
    assert result["rejected"][0]["alias"] == "r7"
    assert next(item for item in result["items"] if item["id"] == "r63")["record_id"] == source[63].record_id
    assert len(json.dumps(reply, separators=(",", ":"))) < 800
