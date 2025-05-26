from __future__ import annotations

from typing import List
import logging

from core.models.document import DocumentChunk
from .openai_embedder import OpenAIEmbedder
from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)


class BatchProcessor:
    """Efficient batch processing for embeddings."""

    def __init__(self, embedder: OpenAIEmbedder, cache: EmbeddingCache, batch_size: int = 100) -> None:
        self.embedder = embedder
        self.cache = cache
        self.batch_size = batch_size

    def process_chunks(self, chunks: List[DocumentChunk], model: str) -> List[DocumentChunk]:
        """Process chunks in batches, using cache when possible."""
        cached_chunks: List[DocumentChunk] = []
        uncached_chunks: List[DocumentChunk] = []

        for chunk in chunks:
            embedding = self.cache.get(chunk.content, model)
            if embedding is not None:
                chunk.embedding = embedding
                cached_chunks.append(chunk)
            else:
                uncached_chunks.append(chunk)

        logger.info("Cache hits: %s, misses: %s", len(cached_chunks), len(uncached_chunks))

        for i in range(0, len(uncached_chunks), self.batch_size):
            batch = uncached_chunks[i : i + self.batch_size]
            texts = [c.content for c in batch]
            try:
                embeddings = self.embedder.generate_embeddings_batch(texts)
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
                    self.cache.set(chunk.content, model, embedding)
                logger.info("Processed batch %s (%s chunks)", i // self.batch_size + 1, len(batch))
            except Exception as exc:
                logger.error("Batch processing failed: %s", exc)
                for chunk in batch:
                    try:
                        embedding = self.embedder.generate_embedding(chunk.content)
                        chunk.embedding = embedding
                        self.cache.set(chunk.content, model, embedding)
                    except Exception as exc2:
                        logger.error("Failed to embed chunk %s: %s", chunk.id, exc2)

        return chunks
