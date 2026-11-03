from independent_ingestion import parse

def test_markdown_ir_has_stable_ids_and_provenance():
    one = parse("# A\n\nBody", source_type="markdown")
    two = parse("# A\n\nBody", source_type="markdown")
    assert one.document_id == two.document_id
    assert [b.block_id for b in one.blocks] == [b.block_id for b in two.blocks]
    assert one.blocks[0].source_spans[0].source_hash == one.source_sha256

def test_parser_replacement():
    from independent_ingestion import ParserRegistry
    registry = ParserRegistry()
    registry.register("text", lambda payload: parse(payload.data, source_type="text"))
    assert registry.parse("hello", source_type="text").source_type == "text"


def test_supported_local_source_types_are_uniparser_free():
    from independent_ingestion import parse

    cases = {
        "text": b"plain text",
        "markdown": b"# Heading\n\nBody",
        "html": b"<h1>Heading</h1><p>Body</p>",
        "xml": b"<article><title>Heading</title><p>Body</p></article>",
        "latex": b"\\section{Heading}\nBody",
        # The parser degrades to a deterministic text IR when an optional PDF
        # backend is unavailable; this still exercises the independent path.
        "pdf": b"%PDF-1.4\nminimal fixture",
        "image": b"\x89PNG\r\nfixture",
    }
    for source_type, data in cases.items():
        document = parse(data, source_type=source_type)
        assert document.source_type == source_type
        assert document.source_sha256
        assert document.provenance.parser


def test_image_directory_is_indexed_without_copying_assets(tmp_path):
    (tmp_path / "one.png").write_bytes(b"png")
    (tmp_path / "two.jpg").write_bytes(b"jpg")
    document = parse(tmp_path)
    assert document.source_type == "image_directory"
    assert len(document.figures) == 2
    assert all(figure.asset_ref for figure in document.figures)

