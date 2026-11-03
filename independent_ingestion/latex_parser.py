import re
from .base_parser import text_ir
from .source_loader import SourcePayload

def parse_latex(payload: SourcePayload):
    text = payload.data.decode("utf-8", errors="replace")
    text = re.sub(r"%.*", "", text)
    text = re.sub(r"\\(?:cite|ref|label)\{([^}]+)\}", r"[\1]", text)
    return text_ir(payload, text, parser="latex", source_type="latex")

