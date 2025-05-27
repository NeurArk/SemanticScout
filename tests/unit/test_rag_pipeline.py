from __future__ import annotations

from unittest.mock import Mock, patch

import pytest

from core.models.search import SearchQuery, SearchResult, SearchResponse
from core.models.chat import ChatMessage
from core.rag_pipeline import RAGPipeline


@pytest.fixture
def mock_openai() -> Mock:
    with patch("openai.OpenAI") as mock_cls:
        client = Mock()
        mock_cls.return_value = client

        # embeddings
        embed_resp = Mock()
        embed_resp.data = [Mock(embedding=[0.1] * 3072)]
        client.embeddings.create.return_value = embed_resp

        # chat completions
        chat_resp = Mock()
        chat_resp.choices = [Mock(message=Mock(content="Mock answer"))]
        client.chat.completions.create.return_value = chat_resp

        yield client


@pytest.fixture
def mock_vector_store() -> Mock:
    with patch("core.rag_pipeline.VectorStore") as mock_cls:
        store = Mock()
        mock_cls.return_value = store
        yield store


def test_rag_pipeline_success(mock_openai: Mock, mock_vector_store: Mock) -> None:
    search_result = SearchResult(
        chunk_id="c1",
        document_id="d1",
        score=0.9,
        content="chunk text",
        metadata={"filename": "doc1.txt"},
    )
    search_response = SearchResponse(
        query=SearchQuery(query_text="q"),
        results=[search_result],
        total_results=1,
        search_time_ms=1.0,
    )
    mock_vector_store.search.return_value = search_response

    rag = RAGPipeline()
    answer, sources = rag.query("What is this?", history=[ChatMessage(role="user", content="hi")])

    assert "doc1.txt" in answer
    assert sources == ["doc1.txt"]


def test_rag_pipeline_no_results(mock_openai: Mock, mock_vector_store: Mock) -> None:
    search_response = SearchResponse(
        query=SearchQuery(query_text="q"),
        results=[],
        total_results=0,
        search_time_ms=1.0,
    )
    mock_vector_store.search.return_value = search_response

    rag = RAGPipeline()
    answer, sources = rag.query("No info")

    assert sources == []
    assert "couldn't find" in answer.lower()

