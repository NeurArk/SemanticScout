from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import app
from core.models.document import Document, DocumentChunk


@patch("app.vector_store")
@patch("app.embedder")
@patch("app.doc_processor")
def test_process_file_success(mock_processor: Mock, mock_embedder: Mock, mock_store: Mock) -> None:
    app.uploaded_files.clear()
    file_obj = SimpleNamespace(name="/tmp/test.txt")

    doc = Document(
        id="d1",
        filename="test.txt",
        file_type="txt",
        file_size=10,
        content="content",
    )
    chunk = DocumentChunk(
        id="c1",
        document_id="d1",
        content="chunk text",
        chunk_index=0,
        start_char=0,
        end_char=10,
        embedding=[0.1],
    )

    mock_processor.process_document.return_value = (doc, [chunk])
    mock_embedder.embed_document.return_value = [chunk]

    status = app.process_file(file_obj)
    assert "Successfully processed" in status
    assert "test.txt" in app.uploaded_files
    mock_store.store_document.assert_called_once_with(doc, [chunk])


def test_process_file_no_file() -> None:
    status = app.process_file(None)
    assert status == "No file uploaded"


@patch("app.rag_pipeline")
def test_chat_response(mock_rag: Mock) -> None:
    mock_rag.query.return_value = ("Answer", ["doc1.txt"])
    history = [["hi", "hello"]]
    response = app.chat_response("question", history)
    assert response == "Answer"
    mock_rag.query.assert_called_once()


@patch("app.vector_store")
def test_clear_all_documents(mock_store: Mock) -> None:
    app.uploaded_files["file.txt"] = {"doc_id": "d1", "chunks": 1}
    mock_store.clear.return_value = None
    msg = app.clear_all_documents()
    assert "cleared" in msg
    assert app.uploaded_files == {}
    mock_store.clear.assert_called_once()

