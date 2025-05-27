# PR9: Essential Testing - Demo Reliability Focus

## Overview
Add minimal but effective tests to ensure the demo works flawlessly. Not comprehensive testing, just demo reliability.

## Goal
Ensure zero failures during demos by testing critical paths.

## Test Structure (Keep it Simple)

```
tests/
├── test_demo_flow.py       # Main demo scenarios
└── test_integration.py     # Basic integration test
```

## Essential Tests Only

### 1. Demo Flow Test (`tests/test_demo_flow.py`)

```python
import pytest
from pathlib import Path
from core.document_processor import DocumentProcessor
from core.embedder import EmbeddingService
from core.vector_store import VectorStore
from core.rag_pipeline import RAGPipeline

@pytest.fixture
def sample_pdf(tmp_path):
    """Create a simple test PDF."""
    # Use existing sample or create minimal PDF
    return "samples/test_contract.pdf"

def test_complete_demo_flow(sample_pdf):
    """Test the complete upload → process → chat flow."""
    
    # 1. Initialize services
    processor = DocumentProcessor()
    embedder = EmbeddingService()
    vector_store = VectorStore()
    rag = RAGPipeline()
    
    # 2. Process document
    doc, chunks = processor.process_document(sample_pdf)
    assert doc is not None
    assert len(chunks) > 0
    
    # 3. Generate embeddings
    embedded = embedder.embed_document(doc, chunks)
    assert all(chunk.embedding is not None for chunk in embedded)
    
    # 4. Store in vector DB
    vector_store.add_document(doc, embedded)
    
    # 5. Test RAG query
    answer, sources = rag.query("What is this document about?")
    assert len(answer) > 0
    assert len(sources) > 0

def test_no_document_handling():
    """Test graceful handling when no documents exist."""
    rag = RAGPipeline()
    answer, sources = rag.query("Tell me about the contract")
    
    assert "couldn't find" in answer.lower()
    assert len(sources) == 0

def test_error_handling():
    """Test that errors don't crash the system."""
    processor = DocumentProcessor()
    
    # Test with non-existent file
    with pytest.raises(Exception):
        processor.process_document("does_not_exist.pdf")
```

### 2. Quick Integration Test (`tests/test_integration.py`)

```python
import pytest
from app import process_file, chat_response, clear_all_documents

def test_gradio_functions():
    """Test main Gradio callback functions work."""
    
    # Test clear function
    result = clear_all_documents()
    assert "cleared" in result.lower()
    
    # Test chat with no documents
    response = chat_response("Hello", [])
    assert len(response) > 0
    assert "error" not in response.lower()
```

## What to Test

1. **Happy Path**: Upload → Process → Query → Get Answer
2. **No Documents**: Graceful "not found" responses
3. **Bad Input**: Invalid files, empty queries
4. **API Errors**: Mock OpenAI errors (optional)

## What NOT to Test

- ❌ Every edge case
- ❌ Performance metrics
- ❌ Complex mocking scenarios
- ❌ UI components (Gradio handles this)
- ❌ Token counting accuracy

## Running Tests

```bash
# Run only demo-critical tests
pytest tests/test_demo_flow.py -v

# Skip slow tests if any
pytest -m "not slow"

# Run with coverage (optional)
pytest --cov=core --cov-report=term-missing
```

## Mock Strategies (Keep Simple)

```python
# Simple mock for OpenAI during tests
@pytest.fixture
def mock_openai(monkeypatch):
    def mock_create(*args, **kwargs):
        return type('obj', (object,), {
            'choices': [type('obj', (object,), {
                'message': type('obj', (object,), {
                    'content': 'Test response based on documents.'
                })()
            })()]
        })()
    
    monkeypatch.setattr("openai.OpenAI.chat.completions.create", mock_create)
```

## Success Criteria

- [ ] Demo flow test passes
- [ ] No errors when documents missing
- [ ] Tests run in < 30 seconds
- [ ] CI/CD stays green

## CI Configuration (`.github/workflows/ci.yml`)

Already configured to run tests with dummy OpenAI key. No changes needed.

Remember: These tests are about preventing demo failures, not achieving 100% coverage.