"""Source-bound facts and deterministic aggregation, without provider calls.

Physical source records are identified by document hash and exact source span.
Repeated observations do not create new records. A disagreeing observation
remains a conflict; the most recent model output is never silently preferred.
Partial counts are lower bounds, not exact answers. Completeness certifies
coverage and arithmetic over accepted observations, not the semantic accuracy
of a model-generated label.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping


_SPAN = re.compile(r"(?:source_chars\s*)?(\d+)\s*:\s*(\d+)")
_HEADER = re.compile(r"^[ \t]*(?:Date|User|Record(?:_?ID)?|ID)\s*:", re.I)
_PAIR = re.compile(r"^\s*([A-Za-z][A-Za-z0-9 _-]*)\s*:\s*(.*?)\s*$", re.S)
_BODY_KEYS = ("body", "instance", "text", "content", "message")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def normalize_date(value: Any) -> str | None:
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    raw = str(value or "").strip()
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).date().isoformat()
    except ValueError:
        pass
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%b %d %Y", "%B %d %Y", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    return None


def _normal_fields(values: Mapping[str, Any]) -> dict[str, Any]:
    fields = {str(k).strip().lower().replace(" ", "_").replace("-", "_"): v for k, v in values.items()}
    for alias in ("user_id", "userid", "user"):
        if alias in fields:
            fields["user"] = str(fields[alias]).strip()
            break
    for alias in ("date", "timestamp", "time"):
        if alias in fields:
            fields["date"] = str(fields[alias]).strip()
            fields["date_iso"] = normalize_date(fields[alias])
            break
    for key in _BODY_KEYS:
        if key in fields:
            fields["body"] = str(fields[key])
            break
    return fields


@dataclass(frozen=True)
class SourceRecord:
    record_id: str
    start: int
    end: int
    text: str
    fields: Mapping[str, Any]
    source_sha256: str

    @property
    def source_span(self) -> str:
        return f"source_chars {self.start}:{self.end}"

    @property
    def source_start(self) -> int:
        return self.start

    @property
    def source_end(self) -> int:
        return self.end

    def to_dict(self) -> dict[str, Any]:
        return {"record_id": self.record_id, "start": self.start, "end": self.end,
                "source_span": self.source_span, "text": self.text,
                "fields": dict(self.fields), "source_sha256": self.source_sha256}


@dataclass(frozen=True)
class ScopeSelection:
    records: tuple[SourceRecord, ...]
    uncertain_records: tuple[SourceRecord, ...]
    complete: bool
    unindexed_count: int | None


def _membership(record: SourceRecord, filters: Mapping[str, Any]) -> bool | None:
    """Three-valued filter evaluation. Missing fields cannot silently exclude."""
    uncertain = False
    for raw_key, expected in filters.items():
        key = str(raw_key).lower()
        if isinstance(expected, Mapping) and (not expected or set(expected) - {"eq", "in"}):
            raise ValueError(f"Unsupported filter operators for: {key}")
        if key in {"user", "user_id", "user_ids", "users"}:
            actual = record.fields.get("user")
            if actual is None:
                uncertain = True
            else:
                groups = ([expected["eq"]],) if isinstance(expected, Mapping) and "eq" in expected else ()
                if isinstance(expected, Mapping) and "in" in expected:
                    groups += (expected["in"],)
                elif not isinstance(expected, Mapping):
                    groups = (expected if isinstance(expected, (list, tuple, set)) else [expected],)
                if any(str(actual) not in {str(v).strip() for v in group} for group in groups):
                    return False
        elif key in {"date_before", "date_after", "date_from", "date_to", "date_on"}:
            actual = record.fields.get("date_iso")
            target = normalize_date(expected)
            if target is None:
                raise ValueError(f"Invalid date filter: {key}")
            if actual is None:
                uncertain = True
                continue
            ok = {"date_before": actual < target, "date_after": actual > target,
                  "date_from": actual >= target, "date_to": actual <= target,
                  "date_on": actual == target}[key]
            if not ok:
                return False
        else:
            if key not in record.fields:
                uncertain = True
                continue
            actual = record.fields[key]
            if isinstance(expected, Mapping):
                if "eq" in expected and actual != expected["eq"]:
                    return False
                if "in" in expected and actual not in expected["in"]:
                    return False
            elif isinstance(expected, (list, tuple, set)):
                if actual not in expected:
                    return False
            elif str(actual) != str(expected):
                return False
    return None if uncertain else True


@dataclass(frozen=True)
class RecordCollection:
    records: tuple[SourceRecord, ...]
    source_sha256: str
    complete: bool
    unparsed_ranges: tuple[tuple[int, int], ...] = ()
    declared_count: int | None = None
    parser: str = "unsupported"
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
    ignored_wrapper_ranges: tuple[tuple[int, int], ...] = ()
    boundary_evidence: tuple[Mapping[str, Any], ...] = ()

    @property
    def by_id(self) -> dict[str, SourceRecord]:
        return {record.record_id: record for record in self.records}

    def scope(self, filters: Mapping[str, Any] | None = None) -> ScopeSelection:
        selected, uncertain = [], []
        for record in self.records:
            match = _membership(record, filters or {})
            if match is True:
                selected.append(record)
            elif match is None:
                uncertain.append(record)
        missing = 0 if self.complete else (
            max(0, self.declared_count - len(self.records))
            if self.declared_count is not None and not self.unparsed_ranges else None)
        return ScopeSelection(tuple(selected), tuple(uncertain), self.complete and not uncertain, missing)

    def select(self, filters: Mapping[str, Any] | None = None) -> tuple[SourceRecord, ...]:
        return self.scope(filters).records


def _record_header(content: str) -> tuple[dict[str, Any], str, int | None] | None:
    """A header must carry actual fields; a Date-like malformed line is not body."""
    if content.lstrip().startswith("{"):
        try:
            value = json.loads(content)
            if isinstance(value, dict) and value:
                _canonical(value)
                return _normal_fields(value), "jsonl", None
        except (ValueError, TypeError):
            pass
    if not _HEADER.match(content):
        return None
    parts = content.split("||")
    if len(parts) < 2:
        return None
    values: dict[str, Any] = {}
    body_start = None
    for index, part in enumerate(parts):
        match = _PAIR.match(part)
        if not match:
            return None
        key, value = match.group(1).strip().lower(), match.group(2).strip()
        if key in values:
            return None
        if key in _BODY_KEYS:
            body_start = sum(len(v) + 2 for v in parts[:index]) + match.start(2)
            values[key] = content[body_start:].strip()
            break
        values[key] = value
    if body_start is None and "value" not in values and "label" not in values:
        return None
    return _normal_fields(values), "field_log", body_start


def _declared_record_counts(text: str) -> list[int]:
    """Counts of source rows, independently of their descriptive noun phrase."""
    counts = [int(m.group(1).replace(",", "")) for m in re.finditer(
        r"(?im)^[ \t]*(?:The following lines|Recall:\s*the preceding lines)\s+contain\s+([\d,]+)\s+[^\r\n]+", text)]
    counts.extend(int(m.group(1).replace(",", "")) for m in re.finditer(
        r"\b(?:contain(?:s)?|consist(?:s)?\s+of|has|are)\s+([\d,]+)\s+(?:pairs?\s+of\s+sentences|records?|examples?|instances?|questions?|reviews?|entries|datapoints?|lines)\b",
        text, re.I))
    return counts


def _footer_boundary(context: str, lines, drafts, prefix: str):
    """Recognize a count-checked wrapper, never an arbitrary unknown suffix.

    A full Recall wrapper must mirror the source's leading wrapper. A standalone
    minimal ``Recall ... N records`` is also an explicit format boundary. Extra
    nonwrapper lines invalidate the entire footer rather than disappearing.
    """
    if not drafts:
        return None
    nonblank_prefix = [re.sub(r"\s+", " ", line.strip()) for line in prefix.split("\n") if line.strip()]
    for index in range(drafts[-1][0] + 1, len(lines)):
        start, _, raw = lines[index]
        first = raw.rstrip("\r\n").strip()
        match = re.match(r"^Recall:\s*the preceding lines\s+contain\s+([\d,]+)\s+", first, re.I)
        if not match:
            continue
        count = int(match.group(1).replace(",", ""))
        tail_lines = [re.sub(r"\s+", " ", line.strip()) for line in context[start:].split("\n") if line.strip()]
        transformed = list(tail_lines)
        transformed[0] = re.sub(r"^Recall:\s*the preceding lines", "The following lines", transformed[0], flags=re.I)
        mirrored = bool(nonblank_prefix and transformed == nonblank_prefix)
        minimal = len(tail_lines) == 1 and bool(re.fullmatch(
            r"Recall:\s*the preceding lines\s+contain\s+[\d,]+\s+(?:records?|entries|examples?|instances?)\.?", first, re.I))
        if count == len(drafts) and (mirrored or minimal):
            return start, {"kind": "count_checked_footer", "source_start": start,
                           "source_end": len(context), "declared_records": count,
                           "parsed_records": len(drafts), "wrapper_mirrors_prefix": mirrored,
                           "standalone_explicit_boundary": minimal}
    return None


def parse_records(context: str) -> RecordCollection:
    """Index structured records and retain their complete, exact source bodies.

    Physical line scanning uses CR/LF only: Unicode NEL and separators in a body
    are data, not new records. A field-log body may span lines until the next
    valid header or a count-checked wrapper. Apparent malformed headers remain
    unknown. An unrecognized trailing suffix remains unknown, rather than
    being silently reclassified as a wrapper or the final record's body.
    """
    context = str(context)
    sha = hashlib.sha256(context.encode()).hexdigest()
    lines = [(m.start(), m.end(), m.group()) for m in re.finditer(r"[^\r\n]*(?:\r\n|\r|\n|$)", context) if m.end() > m.start()]
    drafts = []
    malformed: set[int] = set()
    for index, (start, end, raw) in enumerate(lines):
        content = raw.rstrip("\r\n")
        if not content.strip():
            continue
        parsed = _record_header(content)
        if parsed is not None:
            drafts.append((index, start, end, *parsed))
        elif _HEADER.match(content) or content.lstrip().startswith(("{", "[")):
            malformed.add(index)
    prefix = context[:drafts[0][1]] if drafts else context
    declared_counts = _declared_record_counts(prefix)
    footer = _footer_boundary(context, lines, drafts, prefix)
    footer_start = footer[0] if footer else len(context)
    evidence: list[Mapping[str, Any]] = []
    ignored = []
    if drafts and prefix.strip() and not any(i < drafts[0][0] for i in malformed):
        ignored.append((0, drafts[0][1]))
        evidence.append({"kind": "leading_wrapper", "source_start": 0,
                         "source_end": drafts[0][1], "declared_records": declared_counts})
    if footer:
        ignored.append((footer_start, len(context)))
        evidence.append(footer[1])
        declared_counts.append(footer[1]["declared_records"])
    unknown_indices = {i for i in malformed if lines[i][0] < footer_start}
    records: list[SourceRecord] = []
    parser_types = set()
    continuation_records = 0
    for position, draft in enumerate(drafts):
        index, start, line_end, fields, parser, body_start = draft
        next_index = drafts[position + 1][0] if position + 1 < len(drafts) else len(lines)
        limit = drafts[position + 1][1] if position + 1 < len(drafts) else footer_start
        gap_indices = [i for i in range(index + 1, next_index) if lines[i][0] < limit]
        nonblank = [i for i in gap_indices if lines[i][2].strip()]
        bad = [i for i in gap_indices if i in malformed]
        # A Recall-looking marker that fails validation must not be swallowed
        # as ordinary continuation text, even when its stated count is wrong.
        bad.extend(i for i in gap_indices if re.match(r"^[ \t]*Recall:", lines[i][2], re.I))
        bounded_body = position + 1 < len(drafts) or footer is not None
        end = line_end
        if parser == "field_log" and body_start is not None and bounded_body:
            end = lines[min(bad)][0] if bad else limit
            if bad:
                unknown_indices.update(i for i in nonblank if i >= min(bad))
            if any(lines[i][0] < end for i in nonblank):
                continuation_records += 1
            fields = dict(fields)
            # _normal_fields also adds the canonical ``body`` alias; update
            # the actual source field as well as that alias without losing
            # continuation text from Instance/Text/Content payloads.
            body_key = next((key for key in _BODY_KEYS if key != "body" and key in fields), "body")
            fields[body_key] = context[start + body_start:end].strip()
            fields["body"] = fields[body_key]
        else:
            unknown_indices.update(nonblank)
        identity = hashlib.sha256(f"{sha}:{start}:{end}".encode()).hexdigest()[:16]
        records.append(SourceRecord("r" + identity, start, end, context[start:end], fields, sha))
        parser_types.add(parser)
        evidence.append({"kind": "record_boundary", "record_id": records[-1].record_id,
                         "source_start": start, "source_end": end, "header_end": line_end,
                         "body_continued": end > line_end,
                         "end_basis": "malformed_header" if bad else
                                      "next_header" if position + 1 < len(drafts) else
                                      "count_checked_footer" if footer else "physical_line"})
    unknown = [(lines[i][0], lines[i][1]) for i in sorted(unknown_indices)]
    declared_count = declared_counts[0] if declared_counts else None
    count_matches = all(count == len(records) for count in declared_counts)
    complete = bool(records) and not unknown and count_matches
    if not records:
        unknown = [(0, len(context))] if context.strip() else []
        complete = not context.strip()
    return RecordCollection(
        tuple(records), sha, complete, tuple(unknown), declared_count,
        "+".join(sorted(parser_types)) or "unsupported",
        {"parsed_records": len(records), "declared_count_matches": count_matches,
         "declared_record_counts": declared_counts, "unparsed_nonblank_lines": len(unknown),
         "continuation_records": continuation_records, "footer_boundary_validated": footer is not None,
         "line_break_policy": "CR/LF; Unicode NEL and separators retained as body data"},
        tuple(ignored), tuple(evidence))


@dataclass
class FactCell:
    candidates: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)

    @property
    def conflict(self) -> bool:
        return len(self.candidates) > 1

    @property
    def value(self) -> Any:
        return next(iter(self.candidates.values())) if len(self.candidates) == 1 else None


class FactLedger:
    def __init__(self, collection: RecordCollection):
        self.collection = collection
        self._by_id = collection.by_id
        self._cells: dict[tuple[str, str], FactCell] = {}
        self.events: list[dict[str, Any]] = []

    @property
    def facts(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for (rid, name), cell in self._cells.items():
            result.setdefault(rid, {})[name] = {
                "value": cell.value, "status": "conflict" if cell.conflict else "known",
                "candidates": list(cell.candidates.values()), "evidence": list(cell.evidence)}
        return result

    def add(self, record_id: str, value: Any, *, field: str = "label",
            source_span: str | tuple[int, int] | list[int] | None = None,
            source_sha256: str | None = None, provenance: Any = None) -> dict[str, Any]:
        record = self._by_id.get(str(record_id))
        if str(field) == "label" and isinstance(value, str):
            value = value.strip()
        status, reason = "rejected", ""
        if record is None:
            reason = "unknown_record_id"
        elif source_sha256 is not None and source_sha256 != self.collection.source_sha256:
            reason = "source_hash_mismatch"
        else:
            span = source_span if source_span is not None else (record.start, record.end)
            if isinstance(span, str):
                match = _SPAN.fullmatch(span.strip())
                parsed = (int(match.group(1)), int(match.group(2))) if match else None
            else:
                parsed = tuple(span) if isinstance(span, (list, tuple)) and len(span) == 2 else None
            if parsed != (record.start, record.end):
                reason = "source_span_mismatch"
            elif value is None:
                status, reason = "unknown", "no_fact_value"
            else:
                try:
                    identity = _canonical(value.casefold() if field == "label" and isinstance(value, str) else value)
                    _canonical(provenance)
                except (ValueError, TypeError):
                    reason = "non_json_or_nonfinite_value"
                else:
                    key = (record.record_id, str(field))
                    cell = self._cells.setdefault(key, FactCell())
                    if not cell.candidates and field in record.fields and record.fields[field] is not None:
                        parsed_value = record.fields[field]
                        parsed_key = parsed_value.casefold() if field == "label" and isinstance(parsed_value, str) else parsed_value
                        cell.candidates[_canonical(parsed_key)] = parsed_value
                    duplicate = identity in cell.candidates
                    if not duplicate:
                        cell.candidates[identity] = value
                    ref = {"source_sha256": self.collection.source_sha256,
                           "source_span": record.source_span, "provenance": provenance}
                    if _canonical(ref) not in {_canonical(v) for v in cell.evidence}:
                        cell.evidence.append(ref)
                    status = "conflict" if cell.conflict else "duplicate" if duplicate else "accepted"
        event = {"record_id": str(record_id), "field": str(field), "status": status}
        if reason:
            event["reason"] = reason
        self.events.append(event)
        return event

    def _value(self, record: SourceRecord, name: str, labels: list[str] | None = None) -> tuple[bool, Any, bool]:
        cell = self._cells.get((record.record_id, name))
        conflict = bool(cell and cell.conflict)
        value = cell.value if cell else record.fields.get(name)
        if conflict or value is None:
            return False, None, conflict
        if labels is not None:
            match = [label for label in labels if str(label).strip().casefold() == str(value).strip().casefold()]
            if len(match) != 1:
                return False, None, False
            value = match[0]
        elif name == "label" and isinstance(value, str):
            value = value.strip().casefold()
        return True, value, False

    def unresolved_records(self, filters: Mapping[str, Any] | None = None, *, field: str = "label",
                           labels: Iterable[str] | None = None) -> tuple[SourceRecord, ...]:
        taxonomy = list(labels) if labels is not None else None
        return tuple(record for record in self.collection.select(filters) if not self._value(record, field, taxonomy)[0])

    def state(self, filters: Mapping[str, Any] | None = None, *, field: str = "label", compact: bool = False) -> dict[str, Any]:
        result = self.aggregate("group_counts", filters=filters, field=field)
        records = self.collection.select(filters)
        facts = []
        for record in records:
            known, value, conflict = self._value(record, field)
            if known or conflict:
                facts.append({"record_id": record.record_id, "value": value,
                              "status": "conflict" if conflict else "known",
                              "source_span": record.source_span,
                              "date": record.fields.get("date_iso"), "user": record.fields.get("user")})
        result["fact_digest"] = hashlib.sha256(_canonical(facts).encode()).hexdigest()
        if compact:
            result.pop("scope_record_ids", None)
            result["fact_samples"] = facts[:8]
            result["unresolved_sample_ids"] = [r.record_id for r in self.unresolved_records(filters, field=field)[:16]]
        else:
            result["facts"] = facts
        return result

    def aggregate(self, operation: str, *, filters: Mapping[str, Any] | None = None,
                  field: str = "label", labels: Iterable[str] | None = None,
                  target: Any = None, compare: Iterable[str] | None = None) -> dict[str, Any]:
        operation = str(operation).lower()
        taxonomy = list(dict.fromkeys(str(v) for v in labels)) if labels is not None else None
        scope = self.collection.scope(filters)
        counts: Counter[str] = Counter({label: 0 for label in taxonomy or []})
        known_values: list[tuple[SourceRecord, Any]] = []
        unknown_records: list[SourceRecord] = []
        conflicts = 0
        for record in scope.records:
            known, value, conflict = self._value(record, field, taxonomy)
            conflicts += int(conflict)
            if known:
                key = str(value) if isinstance(value, (str, int, float, bool)) else _canonical(value)
                counts[key] += 1
                known_values.append((record, value))
            else:
                unknown_records.append(record)
        potential = None if scope.unindexed_count is None else scope.unindexed_count + len(scope.uncertain_records)
        unknown = len(unknown_records)
        mass = None if potential is None else unknown + potential
        covered = scope.complete and unknown == 0
        bounds = {label: {"lower": n, "upper": n + mass if mass is not None else None}
                  for label, n in counts.items()}
        result: dict[str, Any] = {
            "operation": operation, "field": field, "filters": dict(filters or {}),
            "value": None, "counts": dict(counts), "bounds": bounds,
            "known_count": len(known_values), "unknown_count": unknown,
            "conflict_count": conflicts, "total_count": len(scope.records),
            "scope_uncertain_count": len(scope.uncertain_records),
            "unindexed_count": scope.unindexed_count,
            "scope_complete": scope.complete, "coverage_complete": covered,
            "complete": False, "source_sha256": self.collection.source_sha256,
            "semantic_verification": "not_provided",
            "scope_record_ids": [r.record_id for r in scope.records],
        }
        def count_key(value):
            key = str(value)
            if field == "label":
                matches = [name for name in counts if name.strip().casefold() == key.strip().casefold()]
                if len(matches) == 1:
                    return matches[0]
            return key
        if operation == "count" and target is None:
            upper = len(scope.records) + potential if potential is not None else None
            result.update(value=len(scope.records) if scope.complete else None,
                          lower_bound=len(scope.records), upper_bound=upper, complete=scope.complete,
                          coverage_complete=scope.complete, requires_classification=False,
                          known_count=len(scope.records), unknown_count=0, conflict_count=0)
        elif operation in {"count", "frequency"}:
            if target is None:
                raise ValueError(f"{operation} requires target")
            lower = counts.get(count_key(target), 0)
            upper = lower + mass if mass is not None else None
            if operation == "frequency":
                maximum_total = len(scope.records) + potential if potential is not None else None
                low_frequency = lower / maximum_total if maximum_total else 0.0
                high_frequency = upper / maximum_total if maximum_total and upper is not None else 1.0
                if scope.complete and not scope.records:
                    low_frequency = high_frequency = None
                result.update(value=lower / len(scope.records) if covered and scope.records else None,
                              lower_bound=low_frequency, upper_bound=high_frequency, complete=covered)
            else:
                result.update(value=lower if covered else None, lower_bound=lower, upper_bound=upper, complete=covered)
        elif operation == "group_counts":
            result.update(value=dict(counts) if covered else None, complete=covered)
        elif operation in {"mode", "least_common"}:
            # Without a closed taxonomy an unseen class could be the answer.
            candidates = taxonomy or list(counts)
            if covered and not scope.records:
                result.update(value=None, ties=[], no_match=True, complete=True)
            elif covered:
                if candidates:
                    optimum = (max if operation == "mode" else min)(counts[v] for v in candidates)
                    winners = [v for v in candidates if counts[v] == optimum]
                    result.update(value=winners[0] if len(winners) == 1 else winners, ties=winners, complete=True)
                else:
                    result.update(value=None, ties=[], complete=True)
            elif taxonomy and mass is not None and scope.complete:
                winners = []
                for label in candidates:
                    others = [v for v in candidates if v != label]
                    certified = (all(counts[label] > counts[v] + mass for v in others)
                                 if operation == "mode" else
                                 all(counts[label] + mass < counts[v] for v in others))
                    if certified:
                        winners.append(label)
                if len(winners) == 1:
                    result.update(value=winners[0], ties=winners, complete=True)
        elif operation == "compare_counts":
            pair = list(compare or [])
            if len(pair) != 2:
                raise ValueError("compare_counts requires two labels in compare")
            left, right = count_key(pair[0]), count_key(pair[1])
            a, b = counts.get(left, 0), counts.get(right, 0)
            au, bu = (a + mass, b + mass) if mass is not None else (None, None)
            result["comparison_bounds"] = {left: {"lower": a, "upper": au}, right: {"lower": b, "upper": bu}}
            if scope.complete and bu is not None and a > bu:
                result.update(value="more common than", complete=True)
            elif scope.complete and au is not None and au < b:
                result.update(value="less common than", complete=True)
            elif covered:
                result.update(value="more common than" if a > b else "less common than" if a < b else "same frequency as", complete=True)
        elif operation in {"earliest", "latest"}:
            self._date_result(result, scope, known_values, unknown_records, field, target, operation)
        elif operation in {"sum", "mean", "min", "max"}:
            numeric = [value for _, value in known_values
                       if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)]
            invalid_numeric = len(known_values) - len(numeric)
            numeric_complete = covered and invalid_numeric == 0
            partial = (sum(numeric) if operation == "sum" else
                       sum(numeric) / len(numeric) if operation == "mean" and numeric else
                       min(numeric) if operation == "min" and numeric else
                       max(numeric) if operation == "max" and numeric else None)
            result.update(value=partial if numeric_complete else None,
                          partial_value=partial, complete=numeric_complete,
                          coverage_complete=numeric_complete,
                          known_count=len(numeric), unknown_count=unknown + invalid_numeric,
                          lower_bound=partial if numeric_complete or operation == "max" else None,
                          upper_bound=partial if numeric_complete or operation == "min" else None)
        else:
            raise ValueError(f"Unsupported deterministic operation: {operation}")
        return result

    def _date_result(self, result, scope, known_values, unknown_records, field, target, operation):
        values = {r.record_id: value for r, value in known_values}
        matches, blockers = [], list(scope.uncertain_records)
        for record in scope.records:
            known = record.record_id in values
            matches_target = target is None or (known and (
                str(values[record.record_id]).strip().casefold() == str(target).strip().casefold()
                if field == "label" else str(values[record.record_id]) == str(target)))
            if target is not None and known and not matches_target:
                continue
            day = record.fields.get("date_iso")
            if matches_target and day is not None:
                matches.append((day, record.record_id))
            else:
                blockers.append(record)
        if not matches:
            result.update(no_match=scope.complete and not blockers,
                          complete=scope.complete and not blockers, value=None)
            return
        best = (min if operation == "earliest" else max)(day for day, _ in matches)
        relevant_unknown = [r for r in blockers if r.fields.get("date_iso") is None or (
            r.fields["date_iso"] < best if operation == "earliest" else r.fields["date_iso"] > best)]
        decisive = scope.complete and not relevant_unknown
        result.update(value=best if decisive else None, candidate_value=best,
                      complete=decisive, prefix_complete=decisive,
                      blocking_record_ids=[r.record_id for r in relevant_unknown],
                      matching_record_ids=[rid for day, rid in matches if day == best])


__all__ = ["SourceRecord", "RecordCollection", "ScopeSelection", "FactLedger", "parse_records", "normalize_date"]
