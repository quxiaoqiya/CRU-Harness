from .document_ir import DocumentIR, Figure, Provenance, SourceSpan, stable_id
from .source_loader import SourcePayload
from pathlib import Path
import hashlib

def parse_image(payload: SourcePayload):
    doc_id = DocumentIR.make_document_id(payload.sha256, "image")
    fid = stable_id(doc_id, "figure", payload.source_name)
    fig = Figure(fid, caption=payload.source_name, asset_ref=payload.sha256, source_spans=[SourceSpan(0, len(payload.data), 0, 0, None, payload.sha256, "independent-ir/1.0")])
    return DocumentIR(document_id=doc_id, source_sha256=payload.sha256, source_type="image", parser_version="independent-ir/1.0", figures=[fig], asset_refs=[payload.sha256], provenance=Provenance(source_name=payload.source_name, source_uri=payload.source_uri, parser="image", parser_version="independent-ir/1.0"))


def parse_image_directory(payload: SourcePayload):
    """Index an image directory without embedding or copying raw image bytes."""
    root = Path(payload.source_uri or "")
    doc_id = DocumentIR.make_document_id(payload.sha256, "image_directory")
    figures, assets = [], []
    for path in sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower().lstrip(".") in {"png", "jpg", "jpeg", "tif", "tiff", "webp", "gif", "bmp"}):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        fid = stable_id(doc_id, "figure", str(path.relative_to(root)))
        figures.append(Figure(fid, caption=path.name, asset_ref=digest, source_spans=[SourceSpan(0, path.stat().st_size, 0, 0, None, digest, "independent-ir/1.0")]))
        assets.append(digest)
    return DocumentIR(document_id=doc_id, source_sha256=payload.sha256, source_type="image_directory", parser_version="independent-ir/1.0", figures=figures, asset_refs=assets, provenance=Provenance(source_name=payload.source_name, source_uri=payload.source_uri, parser="image_directory", parser_version="independent-ir/1.0", metadata={"figure_count": len(figures)}))

