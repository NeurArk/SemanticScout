from __future__ import annotations

import fitz
import pytest
from pathlib import Path
from typing import Any, Dict, List

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


def test_progress_callback(sample_pdf: str) -> None:
    processor = DocumentProcessor()
    progress: List[float] = []

    def cb(value: float) -> None:
        progress.append(value)

    processor.process_document(sample_pdf, progress_callback=cb)
    assert progress and progress[-1] == 1.0


def test_retry_logic(monkeypatch, sample_pdf: str) -> None:
    processor = DocumentProcessor()
    call_count = {
        "count": 0,
    }

    class FailingExtractor:
        def can_extract(self, path: str) -> bool:
            return True

        def extract(self, path: str) -> Dict[str, Any]:
            call_count["count"] += 1
            if call_count["count"] < 2:
                raise DocumentProcessingError("fail")
            return {"content": "This is test content after retry", "metadata": {}}

    monkeypatch.setattr(processor, "_get_extractor", lambda _p: FailingExtractor())
    doc, _ = processor.process_document(sample_pdf, retries=2)
    assert doc.filename.endswith("test.pdf")
