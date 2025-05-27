from __future__ import annotations

from typing import Any, Dict, List, Optional
import logging

from core.models.search import SearchQuery, SearchResult
from core.exceptions.custom_exceptions import VectorStoreError

logger = logging.getLogger(__name__)


class QueryBuilder:
    """Build and execute vector similarity queries."""

    def __init__(self, collection) -> None:
        self.collection = collection

    def search(self, query_embedding: List[float], search_query: SearchQuery) -> List[SearchResult]:
        """Execute similarity search and return results."""
        where_clause = self._build_where_clause(search_query)
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_query.max_results,
                where=where_clause or None,
                include=["documents", "metadatas", "distances"],
            )
            search_results: List[SearchResult] = []
            if results.get("ids") and results["ids"][0]:
                for i, chunk_id in enumerate(results["ids"][0]):
                    distance = results["distances"][0][i]
                    score = 1 - distance
                    if score < search_query.similarity_threshold:
                        continue
                    result = SearchResult(
                        chunk_id=chunk_id,
                        document_id=results["metadatas"][0][i]["document_id"],
                        score=score,
                        content=results["documents"][0][i],
                        metadata=results["metadatas"][0][i],
                    )
                    search_results.append(result)
            logger.info("Found %s results above threshold", len(search_results))
            return search_results
        except Exception as exc:
            logger.error("Search failed: %s", exc)
            raise VectorStoreError(f"Vector search failed: {exc}") from exc

    def _build_where_clause(self, search_query: SearchQuery) -> Optional[Dict[str, Any]]:
        """Construct where clause for metadata filtering."""
        conditions: List[Dict[str, Any]] = []
        if search_query.filter_file_types:
            conditions.append({"file_type": {"$in": search_query.filter_file_types}})
        if search_query.filter_date_range:
            # Example placeholder; requires timestamp metadata
            start_date, end_date = search_query.filter_date_range
            conditions.append({"upload_date": {"$gte": start_date.timestamp(), "$lte": end_date.timestamp()}})
        if not conditions:
            return None
        if len(conditions) == 1:
            return conditions[0]
        return {"$and": conditions}
