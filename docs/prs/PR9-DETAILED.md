# PR9: Testing & Quality Assurance - Detailed Implementation Guide

## Overview
This PR implements comprehensive testing suite including unit, integration, and E2E tests.

## Test Structure
```
tests/
├── conftest.py              # Pytest fixtures
├── unit/
│   ├── test_models.py       # Model validation tests
│   ├── test_document_processor.py
│   ├── test_embedder.py
│   ├── test_vector_store.py
│   ├── test_chat_engine.py
│   └── test_search_engine.py
├── integration/
│   ├── test_rag_pipeline.py
│   ├── test_document_flow.py
│   └── test_chat_flow.py
├── e2e/
│   └── test_gradio_app.py
└── fixtures/
    ├── sample_docs/
    └── mock_responses.py
```

## Key Test Implementations

### 1. Fixtures (`tests/conftest.py`)

```python
import pytest
from unittest.mock import Mock, patch
import tempfile
from pathlib import Path

@pytest.fixture
def mock_openai():
    """Mock OpenAI API responses."""
    with patch('openai.OpenAI') as mock:
        client = Mock()
        mock.return_value = client
        
        # Mock embeddings
        embedding_response = Mock()
        embedding_response.data = [Mock(embedding=[0.1] * 3072)]
        client.embeddings.create.return_value = embedding_response
        
        # Mock chat
        chat_response = Mock()
        chat_response.choices = [Mock(message=Mock(content="Test response"))]
        chat_response.usage = Mock(total_tokens=100)
        client.chat.completions.create.return_value = chat_response
        
        yield client

@pytest.fixture
def sample_pdf(tmp_path):
    """Create sample PDF for testing."""
    # Use reportlab or similar to create test PDF
    pdf_path = tmp_path / "test.pdf"
    pdf_path.write_text("Sample PDF content")
    return str(pdf_path)

@pytest.fixture
def test_vector_store(tmp_path):
    """Create test vector store."""
    from core.vector_store import VectorStore
    import os
    os.environ['CHROMA_PERSIST_DIR'] = str(tmp_path / "chroma")
    return VectorStore()
```

### 2. Unit Tests Example

```python
# tests/unit/test_chat_engine.py
import pytest
from core.chat_engine import ChatEngine
from core.models.chat import MessageRole

def test_chat_without_documents(mock_openai, test_vector_store):
    """Test chat behavior when no documents are uploaded."""
    
    # Mock empty vector store
    test_vector_store.get_stats = Mock(return_value={'total_documents': 0})
    
    # Create chat engine
    engine = ChatEngine(test_vector_store, Mock())
    
    # Test response
    response, sources = engine.chat("What is machine learning?")
    
    assert "upload some documents" in response.lower()
    assert len(sources) == 0
    assert len(engine.conversation_manager.messages) == 2  # user + assistant

def test_chat_with_rag(mock_openai, test_vector_store):
    """Test chat with document context."""
    
    # Mock vector store with documents
    test_vector_store.get_stats = Mock(return_value={'total_documents': 5})
    
    # Mock search results
    mock_results = [
        Mock(
            chunk_id="chunk_1",
            document_id="doc_1", 
            score=0.9,
            content="Machine learning is a type of AI...",
            metadata={'filename': 'ml_basics.pdf'}
        )
    ]
    
    test_vector_store.search = Mock(return_value=Mock(results=mock_results))
    
    # Create chat engine with mocked embedding service
    embedding_service = Mock()
    embedding_service.embed_query = Mock(return_value=[0.1] * 3072)
    
    engine = ChatEngine(test_vector_store, embedding_service)
    
    # Test response
    response, sources = engine.chat("What is machine learning?")
    
    assert "machine learning" in response.lower()
    assert len(sources) > 0
    assert mock_openai.chat.completions.create.called
```

### 3. Integration Tests

```python
# tests/integration/test_document_flow.py
import pytest
from core.document_processor import DocumentProcessor
from core.embedder import EmbeddingService
from core.vector_store import VectorStore

@pytest.mark.integration
def test_complete_document_flow(sample_pdf, mock_openai, test_vector_store):
    """Test complete document processing flow."""
    
    # Initialize services
    processor = DocumentProcessor()
    embedder = EmbeddingService()
    
    # Process document
    document, chunks = processor.process_document(sample_pdf)
    
    assert document is not None
    assert len(chunks) > 0
    
    # Generate embeddings
    embedded_chunks = embedder.embed_document(document, chunks)
    
    assert all(chunk.embedding is not None for chunk in embedded_chunks)
    
    # Store in vector database
    test_vector_store.store_document(document, embedded_chunks)
    
    # Verify storage
    stats = test_vector_store.get_stats()
    assert stats['total_documents'] == 1
    assert stats['total_chunks'] == len(chunks)
```

### 4. E2E Tests

```python
# tests/e2e/test_gradio_app.py
import pytest
from gradio.test_client import TestClient

@pytest.mark.e2e
def test_chat_interface():
    """Test Gradio chat interface."""
    from app import create_app
    
    app = create_app()
    client = TestClient(app)
    
    # Test chat submission
    response = client.predict(
        "Hello, how are you?",
        [],  # chat history
        api_name="/chat"
    )
    
    assert response is not None
    assert len(response[0]) == 1  # One message added
    assert response[1] == ""  # Input cleared

@pytest.mark.e2e 
def test_document_upload():
    """Test document upload flow."""
    from app import create_app
    
    app = create_app()
    client = TestClient(app)
    
    # Test file upload
    with open("tests/fixtures/sample.pdf", "rb") as f:
        response = client.predict(
            f,
            api_name="/upload"
        )
    
    assert "✅" in response  # Success indicator
```

### 5. Performance Tests

```python
# tests/performance/test_performance.py
import pytest
import time
from locust import HttpUser, task, between

class SemanticScoutUser(HttpUser):
    wait_time = between(1, 3)
    
    @task
    def search_documents(self):
        self.client.post("/api/search", json={
            "query": "test query",
            "max_results": 10
        })
    
    @task
    def chat_interaction(self):
        self.client.post("/api/chat", json={
            "message": "What is this document about?"
        })

# Run with: locust -f tests/performance/test_performance.py
```

## Test Configuration

### pytest.ini
```ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --verbose
    --cov=core
    --cov-report=html
    --cov-report=term-missing
    --cov-fail-under=80
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow tests
```

### Coverage Requirements
- Core modules: 90%+
- UI components: 80%+
- Overall: 80%+

## CI/CD Integration

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.11'
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install -r requirements-dev.txt
      - name: Run tests
        run: |
          pytest tests/unit -v
          pytest tests/integration -v --cov
      - name: Upload coverage
        uses: codecov/codecov-action@v2
```

## Success Criteria

1. ✅ 80%+ code coverage achieved
2. ✅ All unit tests pass
3. ✅ Integration tests validate workflows
4. ✅ E2E tests confirm UI functionality
5. ✅ Performance tests meet requirements
6. ✅ Mocked OpenAI API to avoid costs
7. ✅ CI/CD pipeline configured
8. ✅ Test reports generated