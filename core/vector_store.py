from __future__ import annotations

# mypy: ignore-errors

from typing import Any, Dict, List
import logging
import time

from core.models.document import Document, DocumentChunk
from core.models.search import SearchQuery, SearchResponse
from core.exceptions.custom_exceptions import VectorStoreError
from config.settings import get_settings

from .vectordb.chroma_manager import ChromaManager
from .vectordb.collection_manager import CollectionManager
from .vectordb.query_builder import QueryBuilder

logger = logging.getLogger(__name__)
settings = get_settings()


class VectorStore:
    """Store and retrieve document embeddings using ChromaDB."""

    def __init__(self) -> None:
        self.chroma_manager = ChromaManager(
            persist_directory=settings.chroma_persist_dir
        )
        self.collection = self.chroma_manager.get_or_create_collection(
            name="semantic_scout_docs",
            metadata={
                "description": "Document embeddings for semantic search",
                "embedding_model": settings.embedding_model,
                "embedding_dimension": settings.embedding_dimension,
            },
        )
        self.collection_manager = CollectionManager(self.collection)
        self.query_builder = QueryBuilder(self.collection)
        self._search_cache: Dict[str, SearchResponse] = {}
        self._cache_size = 100

    def health_check(self) -> bool:
        """Check database connectivity."""
        try:
            self.chroma_manager.list_collections()
            return True
        except Exception as exc:  # pragma: no cover - simple check
            logger.error("Vector store health check failed: %s", exc)
            return False

    def clear_search_cache(self) -> None:
        """Clear cached search responses."""
        self._search_cache.clear()

    def store_document(self, document: Document, chunks: List[DocumentChunk]) -> None:
        """Store a document and its chunks in the vector database."""
        logger.info("Storing document %s with %s chunks", document.id, len(chunks))
        deleted = self.collection_manager.delete_document(document.id)
        if deleted > 0:
            logger.info(
                "Removed %s existing chunks for document %s", deleted, document.id
            )
        self.collection_manager.add_documents(document, chunks)

    def search(
        self, query_embedding: List[float], search_query: SearchQuery
    ) -> SearchResponse:
        """Search for similar chunks."""
        cache_key = f"{hash(tuple(query_embedding))}:{search_query.model_dump_json()}"
        if cache_key in self._search_cache:
            logger.debug("Search cache hit")
            return self._search_cache[cache_key]

        start = time.time()
        results = self.query_builder.search(query_embedding, search_query)
        duration = (time.time() - start) * 1000
        response = SearchResponse(
            query=search_query,
            results=results,
            total_results=len(results),
            search_time_ms=duration,
        )
        logger.info(
            "Search completed in %.2fms, found %s results", duration, len(results)
        )
        self._search_cache[cache_key] = response
        if len(self._search_cache) > self._cache_size:
            self._search_cache.pop(next(iter(self._search_cache)))
        return response

    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[DocumentChunk]:
        """Retrieve chunks by their IDs."""
        try:
            results = self.collection.get(
                ids=chunk_ids,
                include=["documents", "metadatas", "embeddings"],
            )
            chunks: List[DocumentChunk] = []
            for i, chunk_id in enumerate(results["ids"]):
                chunks.append(
                    DocumentChunk(
                        id=chunk_id,
                        document_id=results["metadatas"][i]["document_id"],
                        content=results["documents"][i],
                        chunk_index=results["metadatas"][i]["chunk_index"],
                        start_char=results["metadatas"][i]["start_char"],
                        end_char=results["metadatas"][i]["end_char"],
                        embedding=(
                            results["embeddings"][i]
                            if results.get("embeddings") is not None
                            else None
                        ),
                        metadata=results["metadatas"][i],
                    )
                )
            return chunks
        except Exception as exc:
            logger.error("Failed to retrieve chunks: %s", exc)
            raise VectorStoreError(f"Chunk retrieval failed: {exc}") from exc

    def delete_document(self, document_id: str) -> bool:
        """Delete a document and its chunks."""
        try:
            deleted = self.collection_manager.delete_document(document_id)
            return deleted > 0
        except Exception as exc:  # pragma: no cover - wrapper
            logger.error("Failed to delete document: %s", exc)
            return False

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Return summary of stored documents."""
        try:
            all_metadata = self.collection.get(include=["metadatas"])["metadatas"]
            documents: Dict[str, Dict[str, Any]] = {}
            for metadata in all_metadata:
                doc_id = metadata.get("document_id")
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        "document_id": doc_id,
                        "filename": metadata.get("filename", "Unknown"),
                        "file_type": metadata.get("file_type", "Unknown"),
                        "chunk_count": 0,
                    }
                if doc_id:
                    documents[doc_id]["chunk_count"] += 1
            return list(documents.values())
        except Exception as exc:  # pragma: no cover - wrapper
            logger.error("Failed to get documents: %s", exc)
            return []

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the vector store."""
        stats = self.collection_manager.get_stats()
        stats["persist_directory"] = str(self.chroma_manager.persist_directory)
        return stats

    def _get_collection_size(self) -> int:
        """Return size on disk of the vector store in bytes."""
        total = 0
        try:
            for path in self.chroma_manager.persist_directory.rglob("*"):
                if path.is_file():
                    total += path.stat().st_size
        except Exception as exc:  # pragma: no cover - simple helper
            logger.error("Failed to calculate collection size: %s", exc)
        return total

    def get_statistics(self) -> Dict[str, Any]:
        """Return extended statistics for analytics display."""
        try:
            stats = self.get_stats()
            documents = self.get_all_documents()
            stats["pdf_count"] = sum(
                1 for doc in documents if doc.get("file_type") == "pdf"
            )
            stats["docx_count"] = sum(
                1 for doc in documents if doc.get("file_type") == "docx"
            )
            stats["txt_count"] = sum(
                1 for doc in documents if doc.get("file_type") == "txt"
            )
            stats["collection_size"] = self._get_collection_size()
            return stats
        except Exception as exc:  # pragma: no cover - wrapper
            logger.error("Failed to get statistics: %s", exc)
            return {"error": str(exc)}

    def clear(self) -> None:
        """Remove all documents from the vector store."""
        try:
            self.chroma_manager.reset_database()
            self.collection = self.chroma_manager.get_or_create_collection(
                name="semantic_scout_docs",
                metadata={
                    "description": "Document embeddings for semantic search",
                    "embedding_model": settings.embedding_model,
                    "embedding_dimension": settings.embedding_dimension,
                },
            )
            self.collection_manager = CollectionManager(self.collection)
            self.query_builder = QueryBuilder(self.collection)
            self.clear_search_cache()
            logger.info("Vector store cleared")
        except Exception as exc:  # pragma: no cover - simple wrapper
            logger.error("Failed to clear vector store: %s", exc)
            raise VectorStoreError(f"Failed to clear store: {exc}") from exc
