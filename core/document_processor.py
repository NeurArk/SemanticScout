from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import List, Tuple

from core.exceptions import DocumentProcessingError
from core.models.document import Document, DocumentChunk

from .extractors import (
    DOCXExtractor,
    MetadataExtractor,
    PDFExtractor,
    TextExtractor,
)
from .chunking import TextChunker
from .processors import ContentCleaner

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """Main document processing orchestrator."""

    def __init__(self) -> None:
        self.extractors = [PDFExtractor(), DOCXExtractor(), TextExtractor()]
        self.metadata_extractor = MetadataExtractor()
        self.chunker = TextChunker()
        self.cleaner = ContentCleaner()

    def process_document(self, file_path: str) -> Tuple[Document, List[DocumentChunk]]:
        """Process a document and return the Document object and its chunks."""
        logger.info("Processing document: %s", file_path)
        extractor = self._get_extractor(file_path)
        if extractor is None:
            raise DocumentProcessingError(f"No extractor found for file: {file_path}")

        extracted = extractor.extract(file_path)
        content = extracted["content"]
        metadata = extracted.get("metadata", {})
        metadata.update(self.metadata_extractor.extract(file_path))

        cleaned_content = self.cleaner.clean_text(content)

        file_path_obj = Path(file_path)
        document = Document(
            id=self._generate_document_id(cleaned_content),
            filename=file_path_obj.name,
            file_type=file_path_obj.suffix.lower().lstrip("."),
            file_size=file_path_obj.stat().st_size,
            content=cleaned_content,
            metadata=metadata,
        )

        chunks = self.chunker.chunk_document(document.id, cleaned_content)
        document.chunk_ids = [chunk.id for chunk in chunks]

        logger.info("Document processed: %s with %s chunks", document.id, len(chunks))
        return document, chunks

    def _get_extractor(self, file_path: str):
        for extractor in self.extractors:
            if extractor.can_extract(file_path):
                return extractor
        return None

    def _generate_document_id(self, content: str) -> str:
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"doc_{content_hash[:12]}"
