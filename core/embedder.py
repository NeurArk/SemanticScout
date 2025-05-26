from __future__ import annotations

from typing import List, Dict, Any
import logging

from core.models.document import Document, DocumentChunk
from core.exceptions import EmbeddingError
from config.settings import get_settings
from .embedding.openai_embedder import OpenAIEmbedder
from .embedding.embedding_cache import EmbeddingCache
from .embedding.batch_processor import BatchProcessor

logger = logging.getLogger(__name__)
settings = get_settings()


class EmbeddingService:
    """Main service for generating and managing embeddings."""

    def __init__(self) -> None:
        self.embedder = OpenAIEmbedder()
        self.cache = EmbeddingCache(
            cache_dir=settings.cache_dir,
            max_size=settings.cache_max_size,
        )
        self.batch_processor = BatchProcessor(
            self.embedder,
            self.cache,
            batch_size=settings.embedding_batch_size,
        )
        self.model = settings.embedding_model

    def embed_document(self, document: Document, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for all chunks of a document."""
        logger.info("Embedding document %s with %s chunks", document.id, len(chunks))
        if not chunks:
            return []
        try:
            embedded_chunks = self.batch_processor.process_chunks(chunks, self.model)
            missing = [c.id for c in embedded_chunks if c.embedding is None]
            if missing:
                logger.warning("Missing embeddings for chunks: %s", missing)
            success_count = sum(1 for c in embedded_chunks if c.embedding is not None)
            logger.info("Successfully embedded %s/%s chunks", success_count, len(chunks))
            return embedded_chunks
        except Exception as exc:
            logger.error("Document embedding failed: %s", exc)
            raise EmbeddingError(f"Failed to embed document {document.id}: {exc}") from exc

    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        cached = self.cache.get(query, self.model)
        if cached is not None:
            return cached
        try:
            embedding = self.embedder.generate_embedding(query)
            self.cache.set(query, self.model, embedding)
            return embedding
        except Exception as exc:
            logger.error("Query embedding failed: %s", exc)
            raise EmbeddingError(f"Failed to embed query: {exc}") from exc

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return self.cache.get_stats()

    def estimate_cost(self, text_length: int) -> float:
        """Estimate embedding cost based on text length."""
        tokens = text_length / 4
        cost_per_million = 0.13
        return (tokens / 1_000_000) * cost_per_million
