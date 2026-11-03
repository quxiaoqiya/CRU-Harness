from .base_parser import text_ir
from .source_loader import SourcePayload

def parse_pdf(payload: SourcePayload):
    text = ""
    error = ""
    try:
        from pypdf import PdfReader
        import io
        import contextlib
        # pypdf can print parser diagnostics for deliberately minimal/offline
        # fixtures; keep those diagnostics inside provenance rather than the
        # command output stream.
        with contextlib.redirect_stderr(io.StringIO()):
            reader = PdfReader(io.BytesIO(payload.data))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        pages = len(reader.pages)
    except Exception as exc:
        pages = 0
        text = payload.data.decode("utf-8", errors="ignore")
        error = type(exc).__name__
    metadata = {"page_count": pages, "fallback": bool(error)}
    if error:
        metadata["fallback_error"] = error
    return text_ir(payload, text, parser="pdf", source_type="pdf", metadata=metadata)

