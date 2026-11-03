"""Independent, local document ingestion.

No UniParser import or environment variable is required by this package.
"""
from .document_ir import *
from .parser_registry import ParserRegistry, default_registry, parse
from .source_loader import load_source

__all__ = ["DocumentIR", "ParserRegistry", "default_registry", "parse", "load_source"]

