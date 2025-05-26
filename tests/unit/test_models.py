import pytest

from core.models.document import Document, DocumentChunk
from core.models.chat import ChatMessage, ChatContext, MessageRole


def test_document_validation() -> None:
    """Validate Document model."""
    doc = Document(
        id="test123",
        filename="test.pdf",
        file_type="pdf",
        file_size=1000,
        content="Test content",
    )
    assert doc.file_type == "pdf"

    with pytest.raises(ValueError):
        Document(
            id="test123",
            filename="test.exe",
            file_type="exe",
            file_size=1000,
            content="Test",
        )


def test_document_chunk_validation() -> None:
    """Validate DocumentChunk model."""
    chunk = DocumentChunk(
        id="chunk1",
        document_id="doc1",
        content="This is a valid chunk of text",
        chunk_index=0,
        start_char=0,
        end_char=10,
    )
    assert chunk.generate_id() == "chunk_doc1_0"

    with pytest.raises(ValueError):
        DocumentChunk(
            id="c2",
            document_id="doc1",
            content="short",
            chunk_index=1,
            start_char=0,
            end_char=5,
        )


def test_chat_context_formatting() -> None:
    """Test context formatting for LLM."""
    context = ChatContext(
        messages=[
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!"),
        ]
    )
    formatted = context.format_for_llm()
    assert len(formatted) == 3
    assert formatted[0]["role"] == "system"
