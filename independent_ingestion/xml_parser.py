import xml.etree.ElementTree as ET
from .base_parser import text_ir
from .source_loader import SourcePayload

def parse_xml(payload: SourcePayload):
    raw = payload.data.decode("utf-8", errors="replace")
    try:
        root = ET.fromstring(raw)
        text = "\n".join(t.strip() for t in root.itertext() if t.strip())
    except ET.ParseError:
        text = raw
    return text_ir(payload, text, parser="xml", source_type="xml")

