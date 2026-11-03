from html.parser import HTMLParser
from .base_parser import text_ir
from .source_loader import SourcePayload

class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__(); self.parts=[]; self.ignore=0
    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style", "noscript"}: self.ignore += 1
        if tag in {"p", "div", "section", "h1", "h2", "h3", "h4", "li", "tr"}: self.parts.append("\n")
    def handle_endtag(self, tag):
        if tag in {"script", "style", "noscript"} and self.ignore: self.ignore -= 1
        if tag in {"p", "div", "section", "h1", "h2", "h3", "h4", "li", "tr"}: self.parts.append("\n")
    def handle_data(self, data):
        if not self.ignore: self.parts.append(data)

def parse_html(payload: SourcePayload):
    parser = _TextExtractor(); parser.feed(payload.data.decode("utf-8", errors="replace"))
    return text_ir(payload, "\n".join(line.strip() for line in "".join(parser.parts).splitlines() if line.strip()), parser="html", source_type="html")

