from __future__ import annotations

from unittest.mock import Mock

from core.vectordb.query_builder import QueryBuilder
from core.models.search import SearchQuery


def test_build_where_clause() -> None:
    qb = QueryBuilder(Mock())
    query = SearchQuery(query_text="test", filter_file_types=["pdf"], max_results=5)
    clause = qb._build_where_clause(query)
    assert clause == {"file_type": {"$in": ["pdf"]}}


def test_search_results() -> None:
    collection = Mock()
    collection.query.return_value = {
        "ids": [["chunk1"]],
        "documents": [["content"]],
        "metadatas": [[{"document_id": "doc1", "chunk_index": 0, "start_char": 0, "end_char": 10}]],
        "distances": [[0.1]],
    }
    qb = QueryBuilder(collection)
    query = SearchQuery(query_text="test", similarity_threshold=0.0)
    results = qb.search([0.1, 0.1, 0.1], query)
    assert len(results) == 1
    assert results[0].document_id == "doc1"
