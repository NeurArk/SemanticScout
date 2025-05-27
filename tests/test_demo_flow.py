import pytest
from unittest.mock import Mock, patch
from pathlib import Path
import fitz

from core.document_processor import DocumentProcessor
from core.embedder import EmbeddingService
from core.rag_pipeline import RAGPipeline


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> str:
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World")
    doc.save(str(pdf_path))
    return str(pdf_path)


@pytest.fixture()
def mock_openai() -> Mock:
    with patch("openai.OpenAI") as mock_cls:
        client = Mock()
        mock_cls.return_value = client

        embed_resp = Mock()
        embed_resp.data = [Mock(embedding=[0.1] * 3072)]
        client.embeddings.create.return_value = embed_resp

        chat_resp = Mock()
        chat_resp.choices = [Mock(message=Mock(content="Test answer"))]
        client.chat.completions.create.return_value = chat_resp
        yield client


def test_complete_demo_flow(sample_pdf: str, mock_openai: Mock, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path))
    from config.settings import get_settings
    get_settings.cache_clear()
    import importlib
    import core.vector_store as vs
    importlib.reload(vs)

    processor = DocumentProcessor()
    embedder = EmbeddingService()
    vector_store = vs.VectorStore()
    rag = RAGPipeline()

    doc, chunks = processor.process_document(sample_pdf)
    assert doc is not None
    assert len(chunks) > 0

    embedded = embedder.embed_document(doc, chunks)
    assert all(c.embedding is not None for c in embedded)

    vector_store.store_document(doc, embedded)

    answer, sources = rag.query("What is this document about?")
    assert len(answer) > 0
    assert len(sources) >= 0


def test_no_document_handling(mock_openai: Mock, tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("CHROMA_PERSIST_DIR", str(tmp_path))
    from config.settings import get_settings
    get_settings.cache_clear()
    import importlib
    import core.vector_store as vs
    importlib.reload(vs)
    rag = RAGPipeline()
    answer, sources = rag.query("Tell me about the contract")
    assert "couldn't find" in answer.lower()
    assert sources == []


def test_error_handling(mock_openai: Mock) -> None:
    processor = DocumentProcessor()
    with pytest.raises(Exception):
        processor.process_document("does_not_exist.pdf")
