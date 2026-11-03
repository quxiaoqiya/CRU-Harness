from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping


_ALLOWED_STATUS = {"confirmed", "validated", "mature"}
_BLOCKED_STATUS = {"unresolved", "conflict", "missing", "candidate", "rejected"}


def _get(value: Any, key: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(key, default)
    return getattr(value, key, default)


def _anchors(memory: Any) -> list[Any]:
    value = _get(memory, "anchors", None)
    if value is None:
        value = _get(memory, "points", [])
    return list(value or [])


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


@dataclass(frozen=True)
class ProjectionResult:
    document_id: str
    manifest: str
    records: tuple[dict[str, Any], ...] = ()
    blockers: tuple[str, ...] = ()
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    @property
    def ready(self) -> bool:
        return not self.blockers and bool(self.records)

    def for_duckdb(self) -> dict[str, Any]:
        """Return only product records and the shared manifest digest."""
        return {"manifest": self.manifest, "document_id": self.document_id, "records": list(self.records)}

    def for_neo4j(self) -> dict[str, Any]:
        """Neo4j receives the exact same manifest and record identities."""
        return self.for_duckdb()


def _record_from_anchor(anchor: Any) -> dict[str, Any] | None:
    status = str(_get(anchor, "status", "")).lower()
    if status not in _ALLOWED_STATUS or status in _BLOCKED_STATUS:
        return None
    evidence_items = []
    for item in (_get(anchor, "evidence_refs", ()) or ()):
        if hasattr(item, "to_dict"):
            evidence_items.append(item.to_dict())
        elif isinstance(item, Mapping):
            evidence_items.append(dict(item))
        elif str(item):
            evidence_items.append(str(item))
    evidence = tuple(evidence_items)
    if not evidence:
        return None
    value = _get(anchor, "normalized_value", None)
    if value is None or value == "":
        return None
    uncertainty = _get(anchor, "uncertainty", None)
    if uncertainty not in (None, "", False, 0):
        # A bounded uncertainty can be published only when explicitly resolved.
        if not bool(_get(anchor, "uncertainty_resolved", False)):
            return None
    anchor_type = str(_get(anchor, "anchor_type", "field"))
    label = str(_get(anchor, "label", "")).strip()
    if not label:
        return None
    normalized = {"label": label, "type": anchor_type, "value": value, "evidence_refs": list(evidence)}
    # The product boundary intentionally drops memory internals and model output.
    return normalized


def project_document_memory(memory: Any) -> ProjectionResult:
    """Project mature, evidenced anchors; fail closed on incomplete memory.

    This function never mutates or serializes the private memory object. The
    resulting manifest is suitable for both DuckDB and Neo4j writers.
    """
    document_id = str(_get(memory, "document_id", "")).strip()
    blockers: list[str] = []
    if not document_id:
        blockers.append("missing_document_id")
    if not bool(_get(memory, "complete", _get(memory, "is_complete", False))):
        blockers.append("document_memory_incomplete")
    raw = []
    rejected = 0
    for anchor in _anchors(memory):
        record = _record_from_anchor(anchor)
        if record is None:
            rejected += 1
        else:
            raw.append(record)
    seen: set[str] = set()
    records: list[dict[str, Any]] = []
    duplicate_count = 0
    for record in raw:
        identity = hashlib.sha256(_canonical(record).encode()).hexdigest()
        if identity in seen:
            duplicate_count += 1
            continue
        seen.add(identity)
        record = dict(record)
        record["record_id"] = hashlib.sha256((document_id + identity).encode()).hexdigest()[:24]
        records.append(record)
    if not records:
        blockers.append("no_mature_evidenced_records")
    payload = {"document_id": document_id, "records": records, "version": "product_projection_v1"}
    manifest = hashlib.sha256(_canonical(payload).encode()).hexdigest()
    diagnostics = {"rejected_anchor_count": rejected, "duplicate_count": duplicate_count, "record_count": len(records)}
    return ProjectionResult(document_id, manifest, tuple(records), tuple(dict.fromkeys(blockers)), diagnostics)


def manifests_match(*projections: ProjectionResult) -> bool:
    """Check that all downstream projections use one release manifest."""
    manifests = {p.manifest for p in projections if p is not None}
    return len(manifests) == 1
