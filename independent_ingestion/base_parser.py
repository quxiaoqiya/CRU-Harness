from __future__ import annotations

import re
from typing import Iterable

from .document_ir import Block, DocumentIR, Figure, Provenance, Section, SourceSpan, Table, stable_id
from .source_loader import SourcePayload


PARSER_VERSION = "independent-ir/1.0"


def _span(payload: SourcePayload, start: int, end: int, page: int | None = None) -> SourceSpan:
    # Token coordinates are deterministic and parser-neutral.  They are
    # approximate for binary/PDF sources, but remain useful for evidence
    # windows and are always tied to the immutable source hash.
    decoded = payload.data.decode("utf-8", errors="replace")
    start_token = len(re.findall(r"\S+", decoded[: max(0, start)]))
    end_token = start_token + len(re.findall(r"\S+", decoded[max(0, start): max(0, end)]))
    return SourceSpan(start_char=start, end_char=end, start_token=start_token, end_token=end_token, page=page, source_hash=payload.sha256, parser_version=PARSER_VERSION)


def text_ir(payload: SourcePayload, text: str, *, parser: str, source_type: str | None = None, metadata: dict | None = None) -> DocumentIR:
    doc_id = DocumentIR.make_document_id(payload.sha256, source_type or payload.source_type)
    sections: list[Section] = []
    blocks: list[Block] = []
    reading: list[str] = []
    current: Section | None = None
    offset = 0
    section_re = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
    for index, raw in enumerate(text.splitlines(keepends=True)):
        line = raw.rstrip("\r\n")
        start, end = offset, offset + len(line)
        offset += len(raw)
        if not line.strip():
            continue
        match = section_re.match(line) or re.match(r"^(?:\\(?:section|subsection|subsubsection)\{(.+?)\})$", line)
        if match:
            if match.lastindex == 2:
                level, title = len(match.group(1)), match.group(2)
            else:
                level, title = 1, match.group(1)
            parent = next((s.section_id for s in reversed(sections) if s.level < level), None)
            sid = stable_id(doc_id, "section", f"{index}:{title}")
            current = Section(sid, title.strip(), level, parent, source_spans=[_span(payload, start, end)])
            sections.append(current)
            continue
        if current is None:
            sid = stable_id(doc_id, "section", "root")
            current = Section(sid, "Document", 1, None, source_spans=[_span(payload, 0, 0)])
            sections.append(current)
        bid = stable_id(doc_id, "block", f"{start}:{end}:{line}")
        block = Block(bid, line.strip(), section_id=current.section_id, source_spans=[_span(payload, start, end)])
        blocks.append(block)
        current.block_ids.append(bid)
        reading.append(bid)
    # Parse simple pipe tables in the block stream and image/caption hints.
    tables: list[Table] = []
    figures: list[Figure] = []
    for block in blocks:
        if "|" in block.text and block.text.count("|") >= 2:
            rows = [[cell.strip() for cell in block.text.strip("|").split("|")]]
            tid = stable_id(doc_id, "table", block.block_id)
            tables.append(Table(tid, rows=rows, section_id=block.section_id, source_spans=block.source_spans.copy()))
            block.table_refs.append(tid)
        if re.match(r"(?i)^(figure|fig\\.?|图)\\s*", block.text):
            fid = stable_id(doc_id, "figure", block.block_id)
            figures.append(Figure(fid, caption=block.text, section_id=block.section_id, source_spans=block.source_spans.copy()))
            block.figure_refs.append(fid)
    return DocumentIR(document_id=doc_id, source_sha256=payload.sha256, source_type=source_type or payload.source_type, parser_version=PARSER_VERSION, pages=[{"page": 1, "block_ids": reading}] if reading else [], sections=sections, blocks=blocks, tables=tables, figures=figures, reading_order=reading, source_spans=[_span(payload, 0, len(text))], provenance=Provenance(source_name=payload.source_name, source_uri=payload.source_uri, parser=parser, parser_version=PARSER_VERSION, metadata=metadata or {}))

