"""Source-bound reuse of labels for byte-identical record bodies.

This is an equivalence index, not a semantic classifier. Only the complete body
sent by the record protocol is compared; no whitespace, punctuation, Unicode,
case or sentence normalization is permitted. Distinct physical source records
remain distinct facts and counts after one representative is classified.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Iterable

from .fact_ledger import SourceRecord
from .record_protocol import ProjectionError, body_projection


@dataclass(frozen=True)
class BodyBinding:
    record: SourceRecord
    body_sha256: str
    body_start: int
    body_end: int

    @property
    def body_source_span(self):
        return f"source_chars {self.body_start}:{self.body_end}"


class ExactBodyIndex:
    """Build immutable exact-body groups inside one query's selected scope.

    Group dictionary equality uses the entire Python string. SHA-256 values
    describe provenance; they are never the sole equality predicate, so even a
    digest collision cannot join different text. Unprojectable rows are kept
    as singleton records and still pass through the normal rejection path.
    """

    def __init__(self, records: Iterable[SourceRecord]):
        self.groups: dict[str, tuple[BodyBinding, ...]] = {}
        self.by_id: dict[str, tuple[BodyBinding, ...]] = {}
        self.binding_by_id: dict[str, BodyBinding] = {}
        drafts: dict[str, list[BodyBinding]] = {}
        self.unprojectable_ids = []
        for record in records:
            try:
                body, (start, end) = body_projection(record)
            except ProjectionError:
                self.unprojectable_ids.append(record.record_id)
                continue
            binding = BodyBinding(record, hashlib.sha256(body.encode('utf-8')).hexdigest(), start, end)
            drafts.setdefault(body, []).append(binding)
            self.binding_by_id[record.record_id] = binding
        for body, values in drafts.items():
            group = tuple(values)
            # The body string is the actual key; SHA hashes are audit metadata.
            self.groups[body] = group
            for binding in group:
                self.by_id[binding.record.record_id] = group

    def representatives(self, ordered: Iterable[SourceRecord], pending_ids: set[str]):
        """Return at most one unresolved representative of each exact group.

        Preserve the controller's deterministic date/hash order; repeated
        source copies cannot use up another call after a failed representative.
        The caller accounts attempts for every group member together.
        """
        seen = set()
        result = []
        for record in ordered:
            if record.record_id not in pending_ids:
                continue
            group = self.by_id.get(record.record_id)
            key = group[0].record.record_id if group else record.record_id
            if key in seen:
                continue
            seen.add(key)
            result.append(record)
        return result

    def group(self, record_id: str):
        return self.by_id.get(record_id, ())

    def summary(self):
        physical = sum(len(group) for group in self.groups.values()) + len(self.unprojectable_ids)
        return {
            'enabled': True, 'equality': 'exact_utf8_body',
            'normalization': 'none', 'source_records': physical,
            'unique_projectable_bodies': len(self.groups),
            'coalesced_duplicates': sum(len(group) - 1 for group in self.groups.values()),
            'unprojectable_records': len(self.unprojectable_ids),
            'largest_group': max((len(group) for group in self.groups.values()), default=0),
            'classification_assumption': 'one_query_body_only_atomic_field',
            'semantic_truth_verified': False,
        }

    def fanout_provenance(self, representative_id: str, target_id: str, projection: dict):
        representative = self.binding_by_id[representative_id]
        target = self.binding_by_id[target_id]
        if self.by_id[representative_id] is not self.by_id[target_id]:
            raise ValueError('fanout_target_not_in_exact_body_group')
        return {
            'stage': 'exact_body_fanout', 'equality': 'exact_utf8_body',
            'representative_record_id': representative_id,
            'representative_source_span': representative.record.source_span,
            'representative_projection': projection,
            'body_sha256': target.body_sha256,
            'body_source_span': target.body_source_span,
            'body_start': target.body_start, 'body_end': target.body_end,
            'body_characters': target.body_end - target.body_start,
            'metadata_used': False,
        }
