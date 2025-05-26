from __future__ import annotations

import uuid


def generate_document_id() -> str:
    """Generate a new document ID."""
    return f"doc_{uuid.uuid4().hex[:12]}"


def generate_chunk_id(document_id: str, index: int) -> str:
    """Generate a new chunk ID."""
    return f"chunk_{document_id}_{index}"


__all__ = ["generate_document_id", "generate_chunk_id"]
