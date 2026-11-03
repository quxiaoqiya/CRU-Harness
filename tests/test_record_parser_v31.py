"""Source-format regression tests; synthetic fixtures only."""
from long_context_agent.fact_ledger import parse_records, FactLedger


def record(day, user, body):
    return f"Date: {day} || User: {user} || Instance: {body}\n"


def envelope(rows, *, noun="records"):
    intro = f"The following lines contain {len(rows)} {noun}, one per line.\n\nCheck every source record.\n\n"
    footer = intro.replace("The following lines", "Recall: the preceding lines", 1)
    return intro + "".join(rows) + footer


def test_repeated_wrapper_is_count_checked_and_source_bound():
    text = envelope([record("2024-01-01", 1, "first"), record("2024-01-02", 2, "second")], noun="long reports")
    collection = parse_records(text)
    assert collection.complete and collection.declared_count == 2
    assert collection.unparsed_ranges == () and len(collection.ignored_wrapper_ranges) == 2
    footer = next(e for e in collection.boundary_evidence if e["kind"] == "count_checked_footer")
    assert footer["wrapper_mirrors_prefix"] and footer["parsed_records"] == 2
    assert text[footer["source_start"]:footer["source_end"]].startswith("Recall:")
    for entry in collection.records:
        assert entry.text == text[entry.start:entry.end]
        assert "Recall:" not in entry.fields["body"]


def test_multiline_bodies_extend_to_next_header_and_validated_footer():
    text = envelope([record("2024-01-01", 1, "first line\nsecond line"),
                     record("2024-01-02", 2, "another line\nfinal continuation")])
    collection = parse_records(text)
    assert collection.complete and len(collection.records) == 2
    assert collection.records[0].fields["body"] == "first line\nsecond line"
    assert collection.records[0].fields["instance"] == "first line\nsecond line"
    assert collection.records[1].fields["body"] == "another line\nfinal continuation"
    assert collection.diagnostics["continuation_records"] == 2


def test_unicode_body_separators_are_never_physical_record_boundaries():
    text = envelope([record("2024-01-01", 1, "A\x85B\u2028C\vD\fE"), record("2024-01-02", 2, "end")])
    collection = parse_records(text)
    assert collection.complete
    assert collection.records[0].fields["body"] == "A\x85B\u2028C\vD\fE"


def test_explicit_minimal_footer_can_close_without_mirrored_prefix():
    text = record("2024-01-01", 1, "first") + "Recall: the preceding lines contain 1 record.\n"
    collection = parse_records(text)
    assert collection.complete and collection.declared_count == 1
    assert any(e.get("standalone_explicit_boundary") for e in collection.boundary_evidence)


def test_mismatched_footer_count_stays_unknown():
    text = record("2024-01-01", 1, "first") + "Recall: the preceding lines contain 2 records.\n"
    collection = parse_records(text)
    assert not collection.complete and collection.unparsed_ranges
    assert not collection.ignored_wrapper_ranges
    assert FactLedger(collection).aggregate("count")["upper_bound"] is None


def test_unknown_line_after_valid_footer_cannot_disappear():
    text = envelope([record("2024-01-01", 1, "first")]) + "UNKNOWN EXTRA DATA\n"
    collection = parse_records(text)
    assert not collection.complete and collection.unparsed_ranges
    assert not collection.diagnostics["footer_boundary_validated"]
    assert any("UNKNOWN EXTRA DATA" in text[a:b] for a, b in collection.unparsed_ranges)


def test_unrecognized_trailing_prose_remains_unknown():
    text = record("2024-01-01", 1, "first") + "This may contain a missing source record.\n"
    collection = parse_records(text)
    assert not collection.complete
    assert collection.records[0].fields["body"] == "first"
    assert collection.unparsed_ranges


def test_malformed_header_is_not_swallowed_as_body():
    text = envelope([record("2024-01-01", 1, "first\nDate: broken || User: missing"),
                     record("2024-01-02", 2, "second")])
    collection = parse_records(text)
    assert not collection.complete
    assert collection.records[0].fields["body"] == "first"
    assert any("Date: broken" in text[a:b] for a, b in collection.unparsed_ranges)


def test_jsonl_continues_to_reject_malformed_nonrecord_lines():
    text = '{"user":1,"text":"first"}\nUNKNOWN\n{"user":2,"text":"second"}\n'
    collection = parse_records(text)
    assert not collection.complete and len(collection.records) == 2


def test_field_scope_can_close_only_after_boundary_validation():
    text = envelope([record("2024-01-01", 1, "first"), record("2024-01-02", 2, "second")])
    collection = parse_records(text)
    scope = collection.scope({"user_ids": [1], "date_from": "2024-01-01", "date_to": "2024-01-01"})
    assert scope.complete and len(scope.records) == 1 and not scope.uncertain_records
    ledger = FactLedger(collection)
    ledger.add(scope.records[0].record_id, "target")
    result = ledger.aggregate("count", filters={"user": 1}, target="target")
    assert result["complete"] and result["value"] == 1


def test_conflicting_prefix_declaration_cannot_be_closed_by_matching_footer():
    text = "The following lines contain 2 records.\n" + record("2024-01-01", 1, "first")
    text += "Recall: the preceding lines contain 1 record.\n"
    collection = parse_records(text)
    assert not collection.complete and not collection.diagnostics["declared_count_matches"]


def test_crlf_spans_are_exact_and_last_bounded_body_keeps_newlines():
    text = envelope([record("2024-01-01", 1, "first\nsecond")]).replace("\n", "\r\n")
    collection = parse_records(text)
    assert collection.complete
    assert collection.records[0].fields["body"] == "first\r\nsecond"
    assert collection.records[0].text == text[collection.records[0].start:collection.records[0].end]
