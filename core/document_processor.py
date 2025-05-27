from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Callable, List, Optional, Tuple
import asyncio

from core.exceptions import DocumentProcessingError
from core.models.document import Document, DocumentChunk

from .extractors import (
    DOCXExtractor,
    MetadataExtractor,
    PDFExtractor,
    TextExtractor,
)
from .extractors.base_extractor import BaseExtractor
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

    def process_document(
        self,
        file_path: str,
        *,
        retries: int = 3,
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> Tuple[Document, List[DocumentChunk]]:
        """Process a document and return the Document object and its chunks."""
        logger.info("Processing document: %s", file_path)
        if progress_callback:
            progress_callback(0.0)
        extractor = self._get_extractor(file_path)
        if extractor is None:
            raise DocumentProcessingError(f"No extractor found for file: {file_path}")

        attempt = 0
        while True:
            try:
                extracted = extractor.extract(file_path)
                break
            except Exception as exc:  # pragma: no cover
                attempt += 1
                logger.warning("Extraction attempt %s failed: %s", attempt, exc)
                if attempt > retries:
                    raise DocumentProcessingError(str(exc)) from exc
        if progress_callback:
            progress_callback(0.3)

        content = extracted["content"]
        metadata = extracted.get("metadata", {})
        metadata.update(self.metadata_extractor.extract(file_path))

        cleaned_content = self.cleaner.clean_text(content)
        if progress_callback:
            progress_callback(0.6)

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
        if progress_callback:
            progress_callback(0.9)

        logger.info("Document processed: %s with %s chunks", document.id, len(chunks))
        if progress_callback:
            progress_callback(1.0)
        return document, chunks

    async def process_document_async(
        self,
        file_path: str,
        *,
        retries: int = 3,
        progress_callback: Optional[Callable[[float], None]] = None,
        timeout: int = 30,
    ) -> Tuple[Document, List[DocumentChunk]]:
        """Asynchronously process a document with optional timeout."""

        return await asyncio.wait_for(
            asyncio.to_thread(
                self.process_document,
                file_path,
                retries=retries,
                progress_callback=progress_callback,
            ),
            timeout=timeout,
        )

    async def process_documents_async(
        self,
        file_paths: List[str],
        *,
        timeout: int = 30,
    ) -> List[Tuple[Document, List[DocumentChunk]]]:
        """Process multiple documents concurrently."""

        tasks = [
            self.process_document_async(path, timeout=timeout) for path in file_paths
        ]
        return await asyncio.gather(*tasks)

    def _get_extractor(self, file_path: str) -> BaseExtractor | None:
        for extractor in self.extractors:
            if extractor.can_extract(file_path):
                return extractor
        return None

    def _generate_document_id(self, content: str) -> str:
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"doc_{content_hash[:12]}"
