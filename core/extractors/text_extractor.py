from __future__ import annotations

import logging
from typing import Any, Dict

import chardet

from core.exceptions import DocumentProcessingError

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class TextExtractor(BaseExtractor):
    """Extract text from plain text files (.txt, .md)."""

    def can_extract(self, file_path: str) -> bool:
        return file_path.lower().endswith((".txt", ".md"))

    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text content with encoding detection."""
        self.validate_file(file_path)

        try:
            with open(file_path, "rb") as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result.get("encoding") or "utf-8"

            with open(file_path, "r", encoding=encoding) as f:
                content = f.read()

            if not content.strip():
                raise DocumentProcessingError("File is empty")

            metadata = {
                "encoding": encoding,
                "line_count": len(content.splitlines()),
                "char_count": len(content),
            }

            return {"content": content, "metadata": metadata}
        except Exception as exc:  # pragma: no cover
            logger.error("Text extraction failed: %s", exc)
            raise DocumentProcessingError(f"Failed to extract text: {exc}") from exc
