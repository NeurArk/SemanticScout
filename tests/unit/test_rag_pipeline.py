from __future__ import annotations

from unittest.mock import Mock, patch, call

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


@patch("core.rag_pipeline.adaptive_analyzer")
def test_rag_pipeline_with_adaptive_threshold(mock_analyzer: Mock, mock_openai: Mock, mock_vector_store: Mock) -> None:
    """Test that RAG pipeline uses adaptive threshold."""
    # Setup mocks
    mock_analyzer.expand_query.return_value = "attention mechanism"
    mock_analyzer.get_adaptive_threshold.return_value = 0.25
    
    search_result = SearchResult(
        chunk_id="c1",
        document_id="d1",
        score=0.3,
        content="attention is all you need",
        metadata={"filename": "attention.pdf"},
    )
    search_response = SearchResponse(
        query=SearchQuery(query_text="attention mechanism"),
        results=[search_result],
        total_results=1,
        search_time_ms=1.0,
    )
    mock_vector_store.search.return_value = search_response
    
    # Execute
    rag = RAGPipeline()
    answer, sources = rag.query("attention")
    
    # Verify adaptive features were used
    mock_analyzer.expand_query.assert_called_once_with("attention")
    mock_analyzer.get_adaptive_threshold.assert_called_once_with("attention")
    
    # Verify search was called with expanded query
    search_call = mock_vector_store.search.call_args[0][1]
    assert search_call.query_text == "attention mechanism"
    assert search_call.similarity_threshold == 0.25
    assert search_call.max_results == 10  # Increased for better selection
    
    assert sources == ["attention.pdf"]


@patch("core.rag_pipeline.adaptive_analyzer")
def test_rag_pipeline_fallback_search(mock_analyzer: Mock, mock_openai: Mock, mock_vector_store: Mock) -> None:
    """Test fallback search with lower threshold."""
    # Setup mocks
    mock_analyzer.expand_query.return_value = "test query"
    mock_analyzer.get_adaptive_threshold.return_value = 0.3
    
    # First search returns no results
    empty_response = SearchResponse(
        query=SearchQuery(query_text="test query"),
        results=[],
        total_results=0,
        search_time_ms=1.0,
    )
    
    # Fallback search returns results
    search_result = SearchResult(
        chunk_id="c1",
        document_id="d1",
        score=0.2,
        content="test content",
        metadata={"filename": "test.pdf"},
    )
    fallback_response = SearchResponse(
        query=SearchQuery(query_text="test query"),
        results=[search_result],
        total_results=1,
        search_time_ms=1.0,
    )
    
    mock_vector_store.search.side_effect = [empty_response, fallback_response]
    
    # Execute
    rag = RAGPipeline()
    answer, sources = rag.query("test query")
    
    # Verify fallback was triggered
    assert mock_vector_store.search.call_count == 2
    
    # Check second call used lower threshold
    second_call = mock_vector_store.search.call_args_list[1][0][1]
    assert second_call.similarity_threshold == 0.15
    
    assert sources == ["test.pdf"]

