from __future__ import annotations

import pytest
from unittest.mock import Mock, patch

from core.embedder import EmbeddingService
from core.models.document import Document, DocumentChunk


@pytest.fixture
def mock_openai() -> Mock:
    with patch("openai.OpenAI") as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 3072)]
        mock_instance.embeddings.create.return_value = mock_response
        yield mock_instance


def test_embed_query_with_cache(mock_openai: Mock) -> None:
    service = EmbeddingService()
    service.cache._memory_cache.clear()
    service.cache.cache_dir = None
    embedding1 = service.embed_query("test query")
    assert mock_openai.embeddings.create.call_count == 1
    embedding2 = service.embed_query("test query")
    assert mock_openai.embeddings.create.call_count == 1
    assert embedding1 == embedding2


def test_batch_processing(mock_openai: Mock) -> None:
    service = EmbeddingService()
    service.cache._memory_cache.clear()
    service.cache.cache_dir = None
    chunks = [
        DocumentChunk(
            id=f"chunk_{i}",
            document_id="doc_1",
            content=f"Test content {i}",
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
        )
        for i in range(150)
    ]
    mock_openai.embeddings.create.return_value.data = [
        Mock(embedding=[0.1] * 3072) for _ in range(100)
    ]
    embedded = service.embed_document(
        Document(
            id="doc_1",
            filename="test.txt",
            file_type="txt",
            file_size=1000,
            content="test",
        ),
        chunks,
    )
    assert mock_openai.embeddings.create.call_count == 2
    assert all(chunk.embedding is not None for chunk in embedded)


def test_rate_limit_handling(mock_openai: Mock) -> None:
    import openai
    import httpx

    service = EmbeddingService()
    service.cache._memory_cache.clear()
    service.cache.cache_dir = None
    resp = httpx.Response(429, request=httpx.Request("POST", "http://test"))
    mock_openai.embeddings.create.side_effect = openai.RateLimitError(
        "Rate limit", response=resp, body=None
    )
    with pytest.raises(Exception):
        service.embed_query("test")


def test_cache_hit_rate(mock_openai: Mock) -> None:
    service = EmbeddingService()
    service.cache.clear()
    service.cache.cache_dir = None

    service.embed_query("a")
    service.embed_query("a")
    stats = service.get_cache_stats()
    assert stats["hits"] == 1
    assert stats["hit_rate"] == 0.5
