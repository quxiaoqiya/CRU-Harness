"""Compact record classification with locally bound source identity.

Only aliases and complete, verbatim bodies cross the model boundary. Full
record identity remains local; compact alias pairs cannot invent source spans.
Projection hashes cover the exact canonical JSON rows used on the wire, not
the full record metadata that was deliberately excluded from that payload.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Any, Iterable, Mapping

from .fact_ledger import SourceRecord


_BODY = re.compile(r"(?:^|\|\|)[ \t]*(?:body|instance|text|content|message)[ \t]*:[ \t]*", re.I)
_SPAN = re.compile(r"(?:source_chars\s*)?(\d+)\s*:\s*(\d+)")


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class ProjectionError(ValueError):
    def __init__(self, reason: str, *, record_id: str = ""):
        self.reason = reason
        self.record_id = record_id
        super().__init__(f"{reason}: {record_id}" if record_id else reason)


@dataclass(frozen=True)
class RecordProjection:
    wire_text: str
    wire_sha256: str
    bindings_json: str
    field: str
    labels: tuple[str, ...]
    bindings_sha256: str = ""

    @property
    def rows(self) -> list[dict[str, Any]]:
        return json.loads(self.wire_text)

    @property
    def bindings(self) -> dict[str, dict[str, Any]]:
        return json.loads(self.bindings_json)

    def audit(self) -> dict[str, Any]:
        refs = list(self.bindings.values())
        return {"protocol": "compact_alias_pairs_v1", "wire_projection_sha256": self.wire_sha256,
                "bindings_sha256": self.bindings_sha256,
                "wire_characters": len(self.wire_text), "records": len(refs),
                "body_characters": sum(ref["body_end"] - ref["body_start"] for ref in refs),
                "bindings": self.bindings}


def body_projection(record: SourceRecord) -> tuple[str, tuple[int, int]]:
    """Return an untruncated body and a verified, contiguous source slice.

    Field-log markers disambiguate a short body also occurring in User/Date
    metadata. A decoded escaped JSON string is not a verbatim source slice;
    that representation is rejected instead of assigning a fictitious span.
    """
    if record.start < 0 or record.end - record.start != len(record.text):
        raise ProjectionError("invalid_record_span", record_id=record.record_id)
    body = record.fields.get("body")
    if not isinstance(body, str):
        raise ProjectionError("missing_string_body", record_id=record.record_id)
    if not body:
        raise ProjectionError("empty_body", record_id=record.record_id)
    is_json = record.text.lstrip().startswith("{")
    marker = None if is_json else _BODY.search(record.text)
    relative = None
    if marker is not None:
        suffix = record.text[marker.end():]
        relative = marker.end() + len(suffix) - len(suffix.lstrip())
    elif is_json:
        # Strict JSONL support where the body is literally present in the
        # source string. Select its actual key/value token, not another field
        # coincidentally containing the same short text.
        for match in re.finditer(r'"(?:body|instance|text|content|message)"\s*:\s*("(?:[^"\\]|\\.)*")', record.text, re.I):
            try:
                decoded = json.loads(match.group(1))
            except ValueError:
                continue
            raw_body = match.group(1)[1:-1]
            if decoded == body and raw_body == body:
                relative = match.start(1) + 1
                break
    if relative is None or record.text[relative:relative + len(body)] != body:
        raise ProjectionError("body_not_verbatim_source", record_id=record.record_id)
    return body, (record.start + relative, record.start + relative + len(body))


def project_records(records: Iterable[SourceRecord], *, field: str = "label",
                    labels: Iterable[str] | None = None) -> RecordProjection:
    taxonomy = tuple(str(value) for value in labels or ())
    if any(not value.strip() for value in taxonomy) or len({v.strip().casefold() for v in taxonomy}) != len(taxonomy):
        raise ProjectionError("ambiguous_taxonomy")
    rows, bindings = [], {}
    seen = set()
    hashes = set()
    for index, record in enumerate(records):
        if record.record_id in seen:
            raise ProjectionError("duplicate_source_record", record_id=record.record_id)
        seen.add(record.record_id)
        hashes.add(record.source_sha256)
        body, (start, end) = body_projection(record)
        alias = f"r{index}"
        rows.append({"id": alias, "text": body})
        bindings[alias] = {"alias": alias, "record_id": record.record_id,
                           "source_span": record.source_span, "source_sha256": record.source_sha256,
                           "body_source_span": f"source_chars {start}:{end}",
                           "body_start": start, "body_end": end, "body_sha256": _sha(body)}
    if len(hashes) > 1:
        raise ProjectionError("mixed_source_documents")
    wire = _canonical(rows)
    digest = _sha(wire)
    for binding in bindings.values():
        binding["projection_sha256"] = digest
    binding_wire = _canonical(bindings)
    return RecordProjection(wire, digest, binding_wire, str(field), taxonomy, _sha(binding_wire))


def _span_matches(value: Any, binding: Mapping[str, Any]) -> bool:
    if not isinstance(value, str):
        return False
    match = _SPAN.fullmatch(value.strip())
    return bool(match and f"source_chars {int(match.group(1))}:{int(match.group(2))}" == binding["source_span"])


def decode_classifications(reply: Mapping[str, Any] | str, projection: RecordProjection, *,
                           field: str | None = None, labels: Iterable[str] | None = None) -> dict[str, Any]:
    """Decode alias pairs, compatible vectors or legacy named records.

    A vector length mismatch rejects the entire batch. Nulls stay unknown;
    invalid indices never become ``other``. Legacy omissions are explicit and
    every occurrence of a duplicated alias is rejected, avoiding double votes.
    """
    bindings = projection.bindings
    aliases = [row["id"] for row in projection.rows]
    field = projection.field if field is None else str(field)
    taxonomy = projection.labels if labels is None else tuple(str(v) for v in labels)
    result: dict[str, Any] = {"items": [], "rejected": [], "format": "unknown", "batch_rejected": False,
                              "projection_sha256": projection.wire_sha256}

    def reject(reason, alias=None, index=None, previous_value=None, **extra):
        try:
            _canonical(previous_value)
        except (TypeError, ValueError):
            previous_value = {"type": type(previous_value).__name__, "representation": repr(previous_value)[:200]}
        item = {"reason": reason, "alias": alias, "index": index, "previous_value": previous_value, **extra}
        if alias in bindings:
            item["record_id"] = bindings[alias]["record_id"]
        result["rejected"].append(item)

    if field != projection.field or taxonomy != projection.labels:
        reject("projection_schema_mismatch")
        result["batch_rejected"] = True
        return result
    if _sha(projection.wire_text) != projection.wire_sha256 or _sha(projection.bindings_json) != projection.bindings_sha256:
        reject("projection_hash_mismatch")
        result["batch_rejected"] = True
        return result
    if len(aliases) != len(set(aliases)) or set(aliases) != set(bindings):
        reject("projection_identity_mismatch")
        result["batch_rejected"] = True
        return result
    if any(_sha(row["text"]) != bindings[row["id"]]["body_sha256"] for row in projection.rows):
        reject("projection_body_mismatch")
        result["batch_rejected"] = True
        return result
    if isinstance(reply, str):
        try:
            reply = json.loads(reply)
        except (ValueError, TypeError):
            reply = None
    if not isinstance(reply, Mapping):
        reject("invalid_reply_object")
        result["batch_rejected"] = True
        return result

    def add(alias, value, index, *, compact, item=None):
        binding = bindings[alias]
        if item is not None and "source_span" in item and not _span_matches(item["source_span"], binding):
            reject("source_span_mismatch", alias, index, value)
            return
        if item is not None and "record_id" in item and item["record_id"] != binding["record_id"]:
            reject("record_identity_mismatch", alias, index, value)
            return
        if item is not None and "source_sha256" in item and item["source_sha256"] != binding["source_sha256"]:
            reject("source_hash_mismatch", alias, index, value)
            return
        if value is None:
            reject("unknown_value", alias, index, None)
        elif field == "label":
            if compact:
                if isinstance(value, bool) or not isinstance(value, int):
                    reject("taxonomy_index_not_integer", alias, index, value)
                    return
                if not 0 <= value < len(taxonomy):
                    reject("taxonomy_index_out_of_range", alias, index, value, taxonomy_size=len(taxonomy))
                    return
                value = taxonomy[value]
            else:
                matched = [label for label in taxonomy if isinstance(value, str) and label.strip().casefold() == value.strip().casefold()]
                if len(matched) != 1:
                    reject("value_outside_taxonomy", alias, index, value)
                    return
                value = matched[0]
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or (isinstance(value, float) and not math.isfinite(value)):
            reject("numeric_value_required", alias, index, value)
            return
        result["items"].append({"id": alias, "record_id": binding["record_id"],
                                "source_span": binding["source_span"], "value": value,
                                "provenance": {**binding, "protocol": result["format"]}})

    if "values" in reply and "records" in reply:
        reject("ambiguous_reply_formats")
        result["batch_rejected"] = True
        return result
    if "values" in reply:
        result["format"] = "compact_values_v1"
        values = reply["values"]
        if not isinstance(values, list) or len(values) != len(aliases):
            for index, alias in enumerate(aliases):
                reject("vector_length_mismatch", alias, index,
                       expected_length=len(aliases), actual_length=len(values) if isinstance(values, list) else None)
            result["batch_rejected"] = True
            return result
        for index, (alias, value) in enumerate(zip(aliases, values)):
            add(alias, value, index, compact=True)
    elif isinstance(reply.get("records"), list):
        records = reply["records"]
        has_pairs = any(isinstance(item, list) for item in records)
        has_dicts = any(isinstance(item, Mapping) for item in records)
        if has_pairs and has_dicts:
            reject("mixed_record_formats")
            result["batch_rejected"] = True
            return result
        if has_pairs or not has_dicts:
            result["format"] = "compact_alias_pairs_v1"
            counts = Counter(item[0] for item in records
                             if isinstance(item, list) and len(item) == 2 and isinstance(item[0], str))
            for index, item in enumerate(records):
                if not isinstance(item, list) or len(item) != 2:
                    reject("invalid_alias_pair", index=index, previous_value=item)
                    continue
                alias, value = item
                if not isinstance(alias, str):
                    reject("alias_not_string", index=index, previous_value=value)
                elif alias not in bindings:
                    reject("unknown_alias", alias, index, value)
                elif counts[alias] > 1:
                    reject("duplicate_alias", alias, index, value)
                else:
                    add(alias, value, index, compact=True)
            for index, alias in enumerate(aliases):
                if not counts[alias]:
                    reject("missing_alias", alias, index)
            return result
        result["format"] = "legacy_records_v1"
        counts = Counter(str(item.get("id")) for item in records if isinstance(item, Mapping))
        for index, item in enumerate(records):
            if not isinstance(item, Mapping):
                reject("invalid_legacy_record", index=index)
                continue
            alias = str(item.get("id", ""))
            value = item.get("value", item.get("label"))
            if alias not in bindings:
                reject("unknown_alias", alias, index, value)
            elif counts[alias] > 1:
                reject("duplicate_alias", alias, index, value)
            else:
                add(alias, value, index, compact=False, item=item)
        for index, alias in enumerate(aliases):
            if not counts[alias]:
                reject("missing_alias", alias, index)
    else:
        reject("missing_values_or_records")
        result["batch_rejected"] = True
    return result


__all__ = ["ProjectionError", "RecordProjection", "body_projection", "project_records", "decode_classifications"]
