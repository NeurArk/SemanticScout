from __future__ import annotations

from typing import Any, Dict, List
import logging

import chromadb

from core.models.document import Document, DocumentChunk
from core.exceptions.custom_exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class CollectionManager:
    """Handle operations on a ChromaDB collection."""

    def __init__(self, collection: chromadb.Collection) -> None:
        self.collection = collection

    def add_documents(self, document: Document, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to the collection."""
        if not chunks:
            return

        ids: List[str] = []
        embeddings: List[List[float]] = []
        documents: List[str] = []
        metadatas: List[Dict[str, Any]] = []

        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning("Skipping chunk %s - no embedding", chunk.id)
                continue
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.content)
            metadata = {
                "document_id": document.id,
                "filename": document.filename,
                "file_type": document.file_type,
                "file_size": document.file_size,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata,
            }
            metadatas.append(metadata)

        if not ids:
            logger.warning("No chunks with embeddings for document %s", document.id)
            return

        try:
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            logger.info("Added %s chunks from document %s", len(ids), document.id)
        except Exception as exc:
            logger.error("Failed to add documents: %s", exc)
            raise VectorStoreError(f"Failed to store document chunks: {exc}") from exc

    def delete_document(self, document_id: str) -> int:
        """Remove all chunks for a document."""
        try:
            results = self.collection.get(where={"document_id": document_id})
            if results["ids"]:
                self.collection.delete(ids=results["ids"])
                logger.info(
                    "Deleted %s chunks for document %s", len(results["ids"]), document_id
                )
                return len(results["ids"])
            return 0
        except Exception as exc:
            logger.error("Failed to delete document: %s", exc)
            raise VectorStoreError(f"Failed to delete document: {exc}") from exc

    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Retrieve all chunks for a document."""
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas", "embeddings"],
            )
            chunks: List[Dict[str, Any]] = []
            for i in range(len(results["ids"])):
                chunks.append(
                    {
                        "id": results["ids"][i],
                        "content": results["documents"][i],
                        "metadata": results["metadatas"][i],
                        "embedding": results["embeddings"][i]
                        if results.get("embeddings")
                        else None,
                    }
                )
            return chunks
        except Exception as exc:
            logger.error("Failed to get document chunks: %s", exc)
            raise VectorStoreError(f"Failed to retrieve chunks: {exc}") from exc

    def get_stats(self) -> Dict[str, Any]:
        """Return statistics about the collection."""
        try:
            count = self.collection.count()
            all_metadata = self.collection.get(include=["metadatas"])["metadatas"]
            unique_docs = {m.get("document_id") for m in all_metadata if m}
            return {
                "total_chunks": count,
                "total_documents": len(unique_docs),
                "collection_name": self.collection.name,
            }
        except Exception as exc:  # pragma: no cover - simple wrapper
            logger.error("Failed to get stats: %s", exc)
            return {"error": str(exc)}
