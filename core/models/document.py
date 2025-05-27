from __future__ import annotations

from datetime import datetime
import hashlib
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class Document(BaseModel):
    """Represents an uploaded document."""

    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension (pdf, docx, txt, md)")
    file_size: int = Field(..., description="File size in bytes")
    content: str = Field(..., description="Full text content")
    upload_date: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_ids: List[str] = Field(default_factory=list)

    @field_validator("file_type")
    def validate_file_type(cls, v: str) -> str:
        allowed = ["pdf", "docx", "txt", "md"]
        if v.lower() not in allowed:
            raise ValueError(f"File type must be one of {allowed}")
        return v.lower()

    @field_validator("file_size")
    def validate_file_size(cls, v: int) -> int:
        max_size = 100 * 1024 * 1024  # 100MB
        if v > max_size:
            raise ValueError(f"File size exceeds maximum of {max_size} bytes")
        return v

    def generate_id(self) -> str:
        """Generate unique ID based on content hash."""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return f"doc_{content_hash[:12]}"


class DocumentChunk(BaseModel):
    """Represents a chunk of document for RAG."""

    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Position in document")
    start_char: int = Field(..., description="Starting character position")
    end_char: int = Field(..., description="Ending character position")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("content")
    def validate_content_length(cls, v: str) -> str:
        if len(v.strip()) < 10:
            raise ValueError("Chunk content too short")
        return v

    def generate_id(self) -> str:
        """Generate unique chunk ID."""
        return f"chunk_{self.document_id}_{self.chunk_index}"


__all__ = ["Document", "DocumentChunk"]
