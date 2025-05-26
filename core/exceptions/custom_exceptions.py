from __future__ import annotations

from typing import Any, Dict


class SemanticScoutError(Exception):
    """Base exception for all custom errors."""

    def __init__(self, message: str, details: Dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


class DocumentProcessingError(SemanticScoutError):
    """Raised when document processing fails."""


class EmbeddingError(SemanticScoutError):
    """Raised when embedding generation fails."""


class VectorStoreError(SemanticScoutError):
    """Raised when vector database operations fail."""


class ValidationError(SemanticScoutError):
    """Raised when input validation fails."""


class SearchError(SemanticScoutError):
    """Raised when search operations fail."""


class ChatError(SemanticScoutError):
    """Raised when chat operations fail."""


class RateLimitError(EmbeddingError):
    """Raised when API rate limit is hit."""

    def __init__(self, retry_after: int = 60) -> None:
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds")
        self.retry_after = retry_after


__all__ = [
    "SemanticScoutError",
    "DocumentProcessingError",
    "EmbeddingError",
    "VectorStoreError",
    "ValidationError",
    "SearchError",
    "ChatError",
    "RateLimitError",
]
