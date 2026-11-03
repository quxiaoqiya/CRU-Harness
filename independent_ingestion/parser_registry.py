"""Extensible parser registry and the public ``parse`` entry point."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .document_ir import DocumentIR
from .source_loader import SourcePayload, load_source
from .text_parser import parse_text
from .markdown_parser import parse_markdown
from .html_parser import parse_html
from .xml_parser import parse_xml
from .latex_parser import parse_latex
from .pdf_parser import parse_pdf
from .image_asset_parser import parse_image, parse_image_directory

Parser = Callable[[SourcePayload], DocumentIR]

class ParserRegistry:
    def __init__(self):
        self._parsers: dict[str, Parser] = {}
    def register(self, source_type: str, parser: Parser) -> None:
        self._parsers[source_type.lower()] = parser
    def parse(self, source: Any, source_type: str | None = None) -> DocumentIR:
        payload = load_source(source, source_type)
        try:
            parser = self._parsers[payload.source_type]
        except KeyError as exc:
            raise ValueError(f"no parser registered for {payload.source_type}") from exc
        return parser(payload)

default_registry = ParserRegistry()
for _type, _parser in {"text": parse_text, "markdown": parse_markdown, "html": parse_html, "xml": parse_xml, "latex": parse_latex, "pdf": parse_pdf, "image": parse_image, "image_directory": parse_image_directory}.items():
    default_registry.register(_type, _parser)

def parse(source: Any, source_type: str | None = None) -> DocumentIR:
    return default_registry.parse(source, source_type)

