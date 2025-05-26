from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, validator


class SearchQuery(BaseModel):
    """Represents a search query."""

    query_text: str = Field(..., min_length=1)
    max_results: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filter_file_types: Optional[List[str]] = None
    filter_date_range: Optional[tuple[datetime, datetime]] = None
    include_metadata: bool = Field(default=True)

    @validator("query_text")
    def clean_query(cls, v: str) -> str:
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty")
        return cleaned


class SearchResult(BaseModel):
    """Represents a single search result."""

    chunk_id: str
    document_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    content: str
    highlighted_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @property
    def source_info(self) -> Dict[str, Any]:
        """Get source document information."""
        return {
            "filename": self.metadata.get("filename", "Unknown"),
            "file_type": self.metadata.get("file_type", "Unknown"),
            "chunk_index": self.metadata.get("chunk_index", 0),
        }


class SearchResponse(BaseModel):
    """Complete search response."""

    query: SearchQuery
    results: List[SearchResult]
    total_results: int
    search_time_ms: float

    def format_for_display(self) -> List[Dict[str, Any]]:
        """Format results for Gradio display."""
        formatted = []
        for result in self.results:
            formatted.append(
                {
                    "source": result.source_info["filename"],
                    "score": f"{result.score:.2%}",
                    "content": result.highlighted_content or result.content[:200] + "...",
                    "metadata": result.metadata,
                }
            )
        return formatted


__all__ = ["SearchQuery", "SearchResult", "SearchResponse"]
