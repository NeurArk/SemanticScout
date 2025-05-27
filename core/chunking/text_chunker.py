from __future__ import annotations

# mypy: ignore-errors

import logging
from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter

try:  # pragma: no cover - optional for offline environments
    import tiktoken
except Exception:  # pragma: no cover - tiktoken may fail to download model
    tiktoken = None

from core.models.document import DocumentChunk

logger = logging.getLogger(__name__)


class TextChunker:
    """Smart text chunking with overlap for RAG."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        if tiktoken is not None:
            try:
                self.encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:  # pragma: no cover - fallback when no network
                self.encoding = None
        else:
            self.encoding = None
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def _token_length(self, text: str) -> int:
        if self.encoding is None:
            return len(text)
        return len(self.encoding.encode(text))

    def chunk_document(self, document_id: str, content: str) -> List[DocumentChunk]:
        """Split document into overlapping chunks."""
        if not content.strip():
            return []

        texts = self.splitter.split_text(content)
        chunks: List[DocumentChunk] = []
        char_index = 0
        for i, text in enumerate(texts):
            start_char = content.find(text, char_index)
            end_char = start_char + len(text)
            char_index = start_char + len(text) - self.chunk_overlap

            chunk = DocumentChunk(
                id=f"chunk_{document_id}_{i}",
                document_id=document_id,
                content=text,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    "token_count": self._token_length(text),
                    "chunk_total": len(texts),
                },
            )
            chunks.append(chunk)

        logger.info("Created %s chunks for document %s", len(chunks), document_id)
        return chunks
