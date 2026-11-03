"""Provider-neutral document intermediate representation.

The IR deliberately contains only stable, serialisable evidence metadata.  It
does not expose parser-specific objects or UniParser identifiers.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from typing import Any, Mapping, Sequence


def stable_id(document_id: str, kind: str, key: str) -> str:
    """Return a deterministic identifier scoped to a source document."""
    digest = hashlib.sha256(f"{document_id}|{kind}|{key}".encode("utf-8")).hexdigest()
    return f"{kind}_{digest[:20]}"


@dataclass(frozen=True)
class SourceSpan:
    start_char: int = 0
    end_char: int = 0
    start_token: int | None = None
    end_token: int | None = None
    page: int | None = None
    source_hash: str = ""
    parser_version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Block:
    block_id: str
    text: str
    kind: str = "paragraph"
    section_id: str | None = None
    source_spans: list[SourceSpan] = field(default_factory=list)
    figure_refs: list[str] = field(default_factory=list)
    table_refs: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Section:
    section_id: str
    title: str
    level: int = 1
    parent_id: str | None = None
    block_ids: list[str] = field(default_factory=list)
    source_spans: list[SourceSpan] = field(default_factory=list)


@dataclass
class Table:
    table_id: str
    caption: str = ""
    rows: list[list[str]] = field(default_factory=list)
    section_id: str | None = None
    source_spans: list[SourceSpan] = field(default_factory=list)


@dataclass
class Figure:
    figure_id: str
    caption: str = ""
    asset_ref: str | None = None
    panel_ids: list[str] = field(default_factory=list)
    section_id: str | None = None
    source_spans: list[SourceSpan] = field(default_factory=list)


@dataclass
class Equation:
    equation_id: str
    content: str
    section_id: str | None = None
    source_spans: list[SourceSpan] = field(default_factory=list)


@dataclass
class Reference:
    reference_id: str
    text: str
    source_spans: list[SourceSpan] = field(default_factory=list)


@dataclass
class Provenance:
    source_name: str = ""
    source_uri: str | None = None
    parser: str = ""
    parser_version: str = "1"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentIR:
    document_id: str
    source_sha256: str
    source_type: str
    parser_version: str
    pages: list[dict[str, Any]] = field(default_factory=list)
    sections: list[Section] = field(default_factory=list)
    blocks: list[Block] = field(default_factory=list)
    tables: list[Table] = field(default_factory=list)
    figures: list[Figure] = field(default_factory=list)
    equations: list[Equation] = field(default_factory=list)
    reading_order: list[str] = field(default_factory=list)
    source_spans: list[SourceSpan] = field(default_factory=list)
    asset_refs: list[str] = field(default_factory=list)
    references: list[Reference] = field(default_factory=list)
    provenance: Provenance = field(default_factory=Provenance)

    @classmethod
    def make_document_id(cls, source_sha256: str, source_type: str) -> str:
        return stable_id(source_sha256, "document", source_type)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DocumentIR":
        def spans(items: Sequence[Mapping[str, Any]]) -> list[SourceSpan]:
            return [SourceSpan(**dict(item)) for item in items]

        sections = [Section(**{**dict(x), "source_spans": spans(x.get("source_spans", []))}) for x in value.get("sections", [])]
        blocks = [Block(**{**dict(x), "source_spans": spans(x.get("source_spans", []))}) for x in value.get("blocks", [])]
        tables = [Table(**{**dict(x), "source_spans": spans(x.get("source_spans", []))}) for x in value.get("tables", [])]
        figures = [Figure(**{**dict(x), "source_spans": spans(x.get("source_spans", []))}) for x in value.get("figures", [])]
        equations = [Equation(**{**dict(x), "source_spans": spans(x.get("source_spans", []))}) for x in value.get("equations", [])]
        references = [Reference(**{**dict(x), "source_spans": spans(x.get("source_spans", []))}) for x in value.get("references", [])]
        root_spans = spans(value.get("source_spans", []))
        provenance = Provenance(**dict(value.get("provenance", {})))
        return cls(document_id=value["document_id"], source_sha256=value["source_sha256"], source_type=value["source_type"], parser_version=value["parser_version"], pages=list(value.get("pages", [])), sections=sections, blocks=blocks, tables=tables, figures=figures, equations=equations, reading_order=list(value.get("reading_order", [])), source_spans=root_spans, asset_refs=list(value.get("asset_refs", [])), references=references, provenance=provenance)

    def block(self, block_id: str) -> Block:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        raise KeyError(block_id)

