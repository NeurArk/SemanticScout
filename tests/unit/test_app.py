from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch
import pandas as pd

import app
from core.models.document import Document, DocumentChunk


@patch("app.adaptive_analyzer")
@patch("app.vector_store")
@patch("app.embedder")
@patch("app.doc_processor")
def test_process_file_success(
    mock_processor: Mock, mock_embedder: Mock, mock_store: Mock, mock_analyzer: Mock
) -> None:
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

    status, _ = app.process_file(file_obj)
    assert "Successfully processed" in status
    assert "test.txt" in app.uploaded_files
    mock_store.store_document.assert_called_once_with(doc, [chunk])


def test_process_file_no_file() -> None:
    status, _ = app.process_file(None)
    assert status == app.get_upload_status()


@patch("app.rag_pipeline")
def test_chat_response(mock_rag: Mock) -> None:
    mock_rag.query.return_value = ("Answer", ["doc1.txt"])
    history = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"}
    ]
    response = app.chat_response("question", history)
    assert response == "Answer"
    mock_rag.query.assert_called_once()


@patch("app.adaptive_analyzer")
@patch("app.vector_store")
def test_clear_all_documents(mock_store: Mock, mock_analyzer: Mock) -> None:
    app.uploaded_files["file.txt"] = {"doc_id": "d1", "chunks": 1}
    mock_store.clear.return_value = None
    msg = app.clear_all_documents()
    assert "cleared" in msg
    assert app.uploaded_files == {}
    mock_store.clear.assert_called_once()


@patch("app.vector_store")
def test_get_system_stats(mock_store: Mock) -> None:
    mock_store.get_statistics.return_value = {
        "total_documents": 2,
        "total_chunks": 5,
        "collection_size": 123,
        "pdf_count": 1,
        "docx_count": 1,
        "txt_count": 0,
    }

    stats = app.get_system_stats()

    assert "Documents: 2" in stats
    assert "Total Chunks: 5" in stats
    assert "PDF: 1" in stats
    mock_store.get_statistics.assert_called_once()


@patch("app.vector_store")
def test_get_system_stats_error(mock_store: Mock) -> None:
    mock_store.get_statistics.side_effect = Exception("fail")
    stats = app.get_system_stats()
    assert "unavailable" in stats.lower()


def test_create_document_type_chart() -> None:
    # Clear and setup uploaded_files
    app.uploaded_files.clear()
    app.uploaded_files["doc1.pdf"] = {"doc_id": "d1", "chunks": 3, "file_size": 1000}
    app.uploaded_files["doc2.pdf"] = {"doc_id": "d2", "chunks": 3, "file_size": 1000}
    app.uploaded_files["test.docx"] = {"doc_id": "d3", "chunks": 5, "file_size": 1000}
    app.uploaded_files["readme.md"] = {"doc_id": "d4", "chunks": 2, "file_size": 500}

    df = app.create_document_type_chart()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 4  # PDF, DOCX, TXT, MD
    assert df.loc[df["Type"] == "PDF", "Count"].iloc[0] == 2
    assert df.loc[df["Type"] == "DOCX", "Count"].iloc[0] == 1
    assert df.loc[df["Type"] == "MD", "Count"].iloc[0] == 1
    assert df.loc[df["Type"] == "TXT", "Count"].iloc[0] == 0


def test_create_document_scatter() -> None:
    # Clear and setup uploaded_files
    app.uploaded_files.clear()
    app.uploaded_files["doc.pdf"] = {
        "doc_id": "d1",
        "chunks": 3,
        "file_size": 2097152  # 2MB in bytes
    }
    app.uploaded_files["test.docx"] = {
        "doc_id": "d2", 
        "chunks": 5,
        "file_size": 1048576  # 1MB in bytes
    }
    app.uploaded_files["readme.md"] = {
        "doc_id": "d3",
        "chunks": 2,
        "file_size": 524288  # 0.5MB in bytes
    }

    df = app.create_document_scatter()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # Now we have 3 documents including MD
    assert "Chunks" in df.columns
    assert "Size (MB)" in df.columns
    assert "Type" in df.columns
    assert "Filename" in df.columns
    
    # Check first document
    pdf_row = df[df["Filename"] == "doc.pdf"]
    assert len(pdf_row) == 1
    assert pdf_row["Chunks"].iloc[0] == 3
    assert pdf_row["Size (MB)"].iloc[0] == 2.0
    assert pdf_row["Type"].iloc[0] == "PDF"
    
    # Check second document
    docx_row = df[df["Filename"] == "test.docx"]
    assert len(docx_row) == 1
    assert docx_row["Chunks"].iloc[0] == 5
    assert docx_row["Size (MB)"].iloc[0] == 1.0
    assert docx_row["Type"].iloc[0] == "DOCX"
    
    # Check third document (MD)
    md_row = df[df["Filename"] == "readme.md"]
    assert len(md_row) == 1
    assert md_row["Chunks"].iloc[0] == 2
    assert md_row["Size (MB)"].iloc[0] == 0.5
    assert md_row["Type"].iloc[0] == "MD"


def test_create_plotly_scatter() -> None:
    # Clear and setup uploaded_files
    app.uploaded_files.clear()
    app.uploaded_files["doc.pdf"] = {
        "doc_id": "d1",
        "chunks": 3,
        "file_size": 2097152  # 2MB in bytes
    }

    fig = app.create_plotly_scatter()
    assert fig is not None
    assert hasattr(fig, 'data')
    assert hasattr(fig, 'layout')
    
    # Test empty case
    app.uploaded_files.clear()
    fig_empty = app.create_plotly_scatter()
    assert fig_empty is not None
