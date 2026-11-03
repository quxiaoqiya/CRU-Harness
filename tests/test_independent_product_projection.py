from product_projection import project_document_memory, manifests_match


def test_projection_fail_closed_and_shared_manifest():
    memory = {
        "document_id": "doc-1",
        "complete": True,
        "anchors": [
            {"anchor_type": "measurement", "label": "bandgap", "normalized_value": 1.8, "status": "confirmed", "evidence_refs": ["b1"]},
            {"anchor_type": "measurement", "label": "unknown", "normalized_value": 2, "status": "unresolved", "evidence_refs": ["b2"]},
            {"anchor_type": "measurement", "label": "no-evidence", "normalized_value": 3, "status": "confirmed", "evidence_refs": []},
        ],
    }
    result = project_document_memory(memory)
    assert result.ready
    assert len(result.records) == 1
    assert result.for_duckdb()["manifest"] == result.for_neo4j()["manifest"]
    assert manifests_match(result, result)


def test_incomplete_memory_never_ready():
    result = project_document_memory({"document_id": "doc-2", "complete": False, "anchors": []})
    assert not result.ready
    assert "document_memory_incomplete" in result.blockers


def test_projection_drops_private_memory_fields():
    memory = {
        "document_id": "doc-3",
        "complete": True,
        "summary": "private chunk summary",
        "model_response": {"secret": True},
        "anchors": [{"anchor_type": "entity", "label": "sample", "normalized_value": "S1", "status": "mature", "evidence_refs": ["b1"]}],
    }
    payload = project_document_memory(memory).for_duckdb()
    assert "summary" not in payload and "model_response" not in payload
    assert "memory_version" not in payload
