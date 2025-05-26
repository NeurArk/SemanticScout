from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any, Dict

from core.exceptions import DocumentProcessingError


class BaseExtractor(ABC):
    """Abstract base class for document extractors."""

    @abstractmethod
    def can_extract(self, file_path: str) -> bool:
        """Return True if this extractor can handle the given file."""
        raise NotImplementedError

    @abstractmethod
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text and metadata from the file."""
        raise NotImplementedError

    def validate_file(self, file_path: str) -> None:
        """Validate that the file exists and is within size limits."""
        if not os.path.exists(file_path):
            raise DocumentProcessingError(f"File not found: {file_path}")

        if os.path.getsize(file_path) > 100 * 1024 * 1024:
            raise DocumentProcessingError("File too large (max 100MB)")
