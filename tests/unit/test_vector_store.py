from __future__ import annotations

import importlib
from typing import List
import importlib

import pytest

from config.settings import get_settings
from core.models.document import Document, DocumentChunk
from core.models.search import SearchQuery


@pytest.fixture
def vector_store(tmp_path, monkeypatch):
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path))
    get_settings.cache_clear()
    import core.vector_store as vs

    importlib.reload(vs)
    store = vs.VectorStore()
    yield store


@pytest.fixture
def sample_document() -> Document:
    return Document(
        id="doc_123",
        filename="test.pdf",
        file_type="pdf",
        file_size=1000,
        content="Test content",
    )


@pytest.fixture
def sample_chunks() -> List[DocumentChunk]:
    return [
        DocumentChunk(
            id=f"chunk_{i}",
            document_id="doc_123",
            content=f"Test chunk {i}",
            chunk_index=i,
            start_char=i * 100,
            end_char=(i + 1) * 100,
            embedding=[0.1] * 3072,
        )
        for i in range(3)
    ]


def test_store_document(vector_store, sample_document, sample_chunks):
    vector_store.store_document(sample_document, sample_chunks)
    docs = vector_store.get_all_documents()
    assert len(docs) == 1
    assert docs[0]["document_id"] == "doc_123"
    assert docs[0]["chunk_count"] == 3


def test_search_documents(vector_store):
    query = SearchQuery(
        query_text="test query", max_results=5, similarity_threshold=0.7
    )
    query_embedding = [0.1] * 3072
    response = vector_store.search(query_embedding, query)
    assert response.total_results == 0
    assert response.search_time_ms > 0
    response2 = vector_store.search(query_embedding, query)
    assert response2.search_time_ms <= response.search_time_ms


def test_health_check(vector_store):
    assert vector_store.health_check()


def test_delete_and_get_chunks(vector_store, sample_document, sample_chunks):
    vector_store.store_document(sample_document, sample_chunks)
    ids = [c.id for c in sample_chunks]
    retrieved = vector_store.get_chunks_by_ids(ids)
    assert len(retrieved) == len(ids)
    assert retrieved[0].id == ids[0]
    assert vector_store.delete_document(sample_document.id) is True
    assert vector_store.get_all_documents() == []


def test_get_statistics(vector_store, sample_document, sample_chunks):
    vector_store.store_document(sample_document, sample_chunks)

    doc2 = Document(
        id="doc_456",
        filename="other.docx",
        file_type="docx",
        file_size=2000,
        content="More text",
    )
    chunks2 = [
        DocumentChunk(
            id=f"chunk_b{i}",
            document_id="doc_456",
            content=f"Chunk text {i}",
            chunk_index=i,
            start_char=i * 50,
            end_char=(i + 1) * 50,
            embedding=[0.1] * 3072,
        )
        for i in range(2)
    ]

    vector_store.store_document(doc2, chunks2)

    stats = vector_store.get_statistics()

    assert stats["total_documents"] == 2
    assert stats["total_chunks"] == len(sample_chunks) + len(chunks2)
    assert stats["pdf_count"] == 1
    assert stats["docx_count"] == 1
    assert stats["txt_count"] == 0
    assert "collection_size" in stats
