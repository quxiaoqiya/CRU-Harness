"""Local source loading with content hashing and no network dependencies."""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourcePayload:
    data: bytes
    source_name: str
    source_uri: str | None
    source_type: str
    sha256: str


def infer_source_type(name: str, data: bytes = b"") -> str:
    suffix = Path(name).suffix.lower().lstrip(".")
    if suffix in {"md", "markdown"}: return "markdown"
    if suffix in {"html", "htm"}: return "html"
    if suffix in {"xml", "tei"}: return "xml"
    if suffix in {"tex", "latex"}: return "latex"
    if suffix == "pdf" or data.startswith(b"%PDF"): return "pdf"
    if suffix in {"png", "jpg", "jpeg", "tif", "tiff", "webp", "gif", "bmp"}: return "image"
    return "text"


def load_source(source: str | Path | bytes, source_type: str | None = None) -> SourcePayload:
    if isinstance(source, Path) or (isinstance(source, str) and Path(source).exists()):
        path = Path(source)
        if path.is_dir():
            files = sorted(p for p in path.rglob("*") if p.is_file() and infer_source_type(p.name) == "image")
            # A directory is represented by a deterministic manifest, never
            # by copying the image bytes into IR/checkpoints.
            rows = []
            for item in files:
                digest = hashlib.sha256(item.read_bytes()).hexdigest()
                rows.append(f"{item.relative_to(path)}\t{digest}")
            data = "\n".join(rows).encode("utf-8")
            name, uri = path.name, str(path)
            if source_type is None:
                source_type = "image_directory"
        else:
            data = path.read_bytes()
            name, uri = path.name, str(path)
    elif isinstance(source, bytes):
        data, name, uri = source, "memory", None
    else:
        data, name, uri = str(source).encode("utf-8"), "memory.txt", None
    kind = (source_type or infer_source_type(name, data)).lower()
    return SourcePayload(data=data, source_name=name, source_uri=uri, source_type=kind, sha256=hashlib.sha256(data).hexdigest())

