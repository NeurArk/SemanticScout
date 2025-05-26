from __future__ import annotations

import fitz
import pytest
from pathlib import Path

from core.document_processor import DocumentProcessor
from core.exceptions import DocumentProcessingError
from core.chunking import TextChunker


@pytest.fixture()
def sample_pdf(tmp_path: Path) -> str:
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Hello World")
    doc.save(str(pdf_path))
    return str(pdf_path)


def test_process_pdf_document(sample_pdf: str) -> None:
    processor = DocumentProcessor()
    doc, chunks = processor.process_document(sample_pdf)

    assert doc.file_type == "pdf"
    assert len(chunks) > 0
    assert all(chunk.document_id == doc.id for chunk in chunks)


def test_process_unsupported_file() -> None:
    processor = DocumentProcessor()
    with pytest.raises(DocumentProcessingError):
        processor.process_document("test.xyz")


def test_chunk_overlap() -> None:
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    text = "A" * 250
    chunks = chunker.chunk_document("test_doc", text)

    assert len(chunks) >= 2
    if len(chunks) >= 2:
        overlap = chunks[0].content[-20:]
        assert overlap in chunks[1].content
