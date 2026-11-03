from .base_parser import text_ir
from .source_loader import SourcePayload

def parse_markdown(payload: SourcePayload):
    return text_ir(payload, payload.data.decode("utf-8", errors="replace"), parser="markdown", source_type="markdown")

