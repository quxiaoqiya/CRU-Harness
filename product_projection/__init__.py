"""Fail-closed projection from private reading memory to product records."""
from .projector import ProjectionResult, project_document_memory, manifests_match

__all__ = ["ProjectionResult", "project_document_memory", "manifests_match"]
