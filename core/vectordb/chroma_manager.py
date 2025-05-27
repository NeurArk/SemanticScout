from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

import chromadb
from chromadb.config import Settings

from core.exceptions.custom_exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class ChromaManager:
    """Manage ChromaDB client and collections."""

    def __init__(self, persist_directory: str = "./data/chroma_db") -> None:
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)

        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            logger.info("ChromaDB initialized at %s", self.persist_directory)
        except Exception as exc:  # pragma: no cover - initialization rarely fails
            logger.error("Failed to initialize ChromaDB: %s", exc)
            raise VectorStoreError(f"ChromaDB initialization failed: {exc}") from exc

    def get_or_create_collection(
        self, name: str, metadata: Optional[Dict[str, Any]] = None
    ) -> chromadb.Collection:
        """Return existing collection or create a new one."""
        try:
            collection = self.client.get_collection(name=name)
            logger.info("Retrieved existing collection: %s", name)
            return collection
        except Exception:
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {"description": "Document embeddings"},
            )
            logger.info("Created new collection: %s", name)
            return collection

    def delete_collection(self, name: str) -> None:
        """Delete a collection by name."""
        try:
            self.client.delete_collection(name=name)
            logger.info("Deleted collection: %s", name)
        except Exception as exc:  # pragma: no cover - simple wrapper
            logger.error("Failed to delete collection: %s", exc)
            raise VectorStoreError(f"Collection deletion failed: {exc}") from exc

    def list_collections(self) -> List[str]:
        """List available collections."""
        return [col.name for col in self.client.list_collections()]

    def reset_database(self) -> None:
        """Reset the entire Chroma database."""
        try:
            self.client.reset()
            logger.warning("ChromaDB has been reset")
        except Exception as exc:  # pragma: no cover - rarely used
            logger.error("Failed to reset ChromaDB: %s", exc)
            raise VectorStoreError(f"Database reset failed: {exc}") from exc

    def health_check(self) -> bool:
        """Check database availability."""
        try:
            self.client.list_collections()
            return True
        except Exception as exc:  # pragma: no cover - simple
            logger.error("ChromaDB health check failed: %s", exc)
            return False

    def backup_database(self, backup_path: str) -> None:
        """Create a simple backup of the persist directory."""
        try:
            import shutil

            shutil.make_archive(backup_path, "zip", self.persist_directory)
            logger.info("Database backup created at %s.zip", backup_path)
        except Exception as exc:
            logger.error("Failed to backup database: %s", exc)
            raise VectorStoreError(f"Backup failed: {exc}") from exc
