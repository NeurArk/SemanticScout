from __future__ import annotations

# mypy: ignore-errors

import logging
from typing import Any, Dict

import fitz  # PyMuPDF

from core.exceptions import DocumentProcessingError

from .base_extractor import BaseExtractor

logger = logging.getLogger(__name__)


class PDFExtractor(BaseExtractor):
    """Extract text from PDF files using PyMuPDF."""

    def can_extract(self, file_path: str) -> bool:
        return file_path.lower().endswith(".pdf")

    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text and metadata from a PDF file."""
        self.validate_file(file_path)

        try:
            text_content: list[dict[str, Any]] = []
            metadata: Dict[str, Any] = {}

            with fitz.open(file_path) as pdf:
                metadata = {
                    "page_count": len(pdf),
                    "title": pdf.metadata.get("title", ""),
                    "author": pdf.metadata.get("author", ""),
                    "subject": pdf.metadata.get("subject", ""),
                    "creator": pdf.metadata.get("creator", ""),
                }

                for page_num, page in enumerate(pdf):
                    try:
                        text = page.get_text()
                        if text.strip():
                            text_content.append({"page": page_num + 1, "content": text})
                    except Exception as exc:  # pragma: no cover - log only
                        logger.warning(
                            "Failed to extract page %s: %s", page_num + 1, exc
                        )
                        continue

            if not text_content:
                raise DocumentProcessingError("No text content found in PDF")

            full_text = "\n\n".join(
                [f"[Page {p['page']}]\n{p['content']}" for p in text_content]
            )

            return {"content": full_text, "metadata": metadata, "pages": text_content}
        except DocumentProcessingError:
            raise
        except Exception as exc:  # pragma: no cover - external library errors
            logger.error("PDF extraction failed: %s", exc)
            raise DocumentProcessingError(f"Failed to extract PDF: {exc}") from exc
