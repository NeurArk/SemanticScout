# SemanticScout - Testing Strategy

**Version**: 1.0  
**Date**: May 2025  
**Status**: Comprehensive Test Plan

## 🧪 Testing Philosophy

SemanticScout's testing strategy ensures reliability, performance, and user satisfaction through comprehensive automated testing, manual validation, and continuous integration practices. Our approach balances thorough coverage with practical development velocity.

### Testing Principles
1. **Quality First**: Every feature must be thoroughly tested before release
2. **Automated Testing**: Prefer automated tests for regression prevention
3. **User-Centric**: Test from user perspective, not just technical functionality
4. **Performance Aware**: Include performance testing in all test scenarios
5. **Fail Fast**: Catch issues early in the development cycle

## 🏗️ Test Architecture

### Testing Pyramid
```
                    /\
                   /  \
                  /    \
                 / E2E  \     <- End-to-End Tests (5%)
                /________\
               /          \
              /Integration \   <- Integration Tests (20%)
             /______________\
            /                \
           /   Unit Tests     \  <- Unit Tests (75%)
          /____________________\
```

### Test Environment Hierarchy
```
Production Environment
         ↑
Staging Environment (Pre-production testing)
         ↑
Test Environment (Integration testing)
         ↑
Development Environment (Unit testing)
```

## 🔬 Unit Testing Strategy

### Framework and Tools
```python
# Testing stack
pytest                 # Primary testing framework  
pytest-cov            # Coverage reporting
pytest-mock          # Mocking utilities
pytest-asyncio       # Async testing
factory-boy           # Test data factories
freezegun              # Time manipulation
responses             # HTTP mocking
```

### Unit Test Structure
```python
# tests/unit/test_document_processor.py
import pytest
from unittest.mock import Mock, patch
from core.document_processor import DocumentProcessor, ProcessingResult
from core.exceptions import DocumentProcessingError

class TestDocumentProcessor:
    
    @pytest.fixture
    def document_processor(self):
        """Create DocumentProcessor instance for testing"""
        return DocumentProcessor()
    
    @pytest.fixture
    def sample_pdf_path(self, tmp_path):
        """Create a sample PDF file for testing"""
        pdf_file = tmp_path / "sample.pdf"
        pdf_file.write_bytes(b"Sample PDF content")
        return str(pdf_file)
    
    def test_process_pdf_document_success(self, document_processor, sample_pdf_path):
        """Test successful PDF processing"""
        with patch('core.document_processor.PyMuPDF') as mock_pymupdf:
            mock_pymupdf.extract_text.return_value = "Sample PDF text content"
            
            result = document_processor.process_document(sample_pdf_path, "pdf")
            
            assert result.success is True
            assert result.document_id is not None
            assert len(result.chunks) > 0
            assert "pdf" in result.metadata["file_type"]
    
    def test_process_invalid_file_format(self, document_processor):
        """Test handling of invalid file formats"""
        with pytest.raises(DocumentProcessingError) as exc_info:
            document_processor.process_document("test.xyz", "xyz")
        
        assert "Unsupported file format" in str(exc_info.value)
    
    def test_process_corrupted_file(self, document_processor, tmp_path):
        """Test handling of corrupted files"""
        corrupted_file = tmp_path / "corrupted.pdf"
        corrupted_file.write_bytes(b"Not a valid PDF")
        
        result = document_processor.process_document(str(corrupted_file), "pdf")
        
        assert result.success is False
        assert "corrupted" in result.error_message.lower()
    
    @patch('core.document_processor.OpenAI')
    def test_embedding_generation_failure(self, mock_openai, document_processor, sample_pdf_path):
        """Test handling of embedding generation failures"""
        mock_openai.embeddings.create.side_effect = Exception("API Error")
        
        result = document_processor.process_document(sample_pdf_path, "pdf")
        
        assert result.success is False
        assert "embedding" in result.error_message.lower()
```

### Test Coverage Requirements
```python
# pytest.ini
[tool:pytest]
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
addopts = 
    --cov=core
    --cov=config
    --cov-report=html:htmlcov
    --cov-report=term-missing
    --cov-fail-under=80
    --strict-markers
    --disable-warnings
markers =
    unit: Unit tests
    integration: Integration tests
    e2e: End-to-end tests
    slow: Slow running tests
    api: API-related tests
```

### Mock Strategies
```python
# tests/fixtures/mocks.py
import pytest
from unittest.mock import Mock, MagicMock

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    client = Mock()
    client.embeddings.create.return_value = Mock(
        data=[Mock(embedding=[0.1] * 3072)]
    )
    client.chat.completions.create.return_value = Mock(
        choices=[Mock(message=Mock(content="Test response"))]
    )
    return client

@pytest.fixture
def mock_chroma_collection():
    """Mock ChromaDB collection for testing"""
    collection = Mock()
    collection.add.return_value = None
    collection.query.return_value = {
        'ids': [['doc1', 'doc2']],
        'distances': [[0.1, 0.2]],
        'documents': [['Sample document 1', 'Sample document 2']],
        'metadatas': [[{'type': 'pdf'}, {'type': 'docx'}]]
    }
    return collection

@pytest.fixture
def sample_documents():
    """Generate sample documents for testing"""
    return [
        {
            'id': 'doc1',
            'content': 'This is a sample document about machine learning.',
            'metadata': {'type': 'pdf', 'filename': 'ml_paper.pdf'}
        },
        {
            'id': 'doc2', 
            'content': 'Another document discussing artificial intelligence.',
            'metadata': {'type': 'docx', 'filename': 'ai_report.docx'}
        }
    ]
```

## 🔗 Integration Testing Strategy

### Integration Test Scope
```python
# tests/integration/test_search_workflow.py
import pytest
from core.application import SemanticScoutApp
from core.models import SearchQuery
import tempfile
import os

class TestSearchWorkflow:
    """Test complete search workflow integration"""
    
    @pytest.fixture
    def app_instance(self):
        """Create application instance with test configuration"""
        with tempfile.TemporaryDirectory() as temp_dir:
            app = SemanticScoutApp(
                chroma_persist_dir=os.path.join(temp_dir, "chroma"),
                upload_dir=os.path.join(temp_dir, "uploads")
            )
            yield app
    
    @pytest.fixture
    def uploaded_documents(self, app_instance):
        """Upload sample documents for testing"""
        documents = []
        
        # Create sample PDF
        pdf_content = b"Sample PDF about machine learning algorithms"
        pdf_result = app_instance.upload_document(
            file_content=pdf_content,
            filename="ml_algorithms.pdf",
            file_type="pdf"
        )
        documents.append(pdf_result)
        
        # Create sample text file
        txt_content = "Text document discussing neural networks and deep learning"
        txt_result = app_instance.upload_document(
            file_content=txt_content.encode(),
            filename="neural_networks.txt", 
            file_type="txt"
        )
        documents.append(txt_result)
        
        return documents
    
    def test_end_to_end_search_workflow(self, app_instance, uploaded_documents):
        """Test complete search workflow"""
        # Wait for processing to complete
        app_instance.wait_for_processing()
        
        # Execute search
        query = SearchQuery(
            query="machine learning",
            limit=10,
            threshold=0.5
        )
        
        results = app_instance.search(query)
        
        # Verify results
        assert len(results.results) > 0
        assert results.query_time_ms < 2000  # Performance requirement
        assert all(result.score >= 0.5 for result in results.results)
        
        # Verify content relevance
        result_texts = [r.content.lower() for r in results.results]
        assert any("machine learning" in text for text in result_texts)
    
    def test_similarity_search_integration(self, app_instance, uploaded_documents):
        """Test document similarity functionality"""
        doc_id = uploaded_documents[0].document_id
        
        similar_docs = app_instance.find_similar_documents(doc_id, limit=5)
        
        assert len(similar_docs) >= 0  # May be empty if no similar docs
        for doc in similar_docs:
            assert doc.document_id != doc_id  # Should not return the same document
            assert 0.0 <= doc.score <= 1.0
```

### Database Integration Tests
```python
# tests/integration/test_vector_store.py
import pytest
from core.vector_store import ChromaVectorStore
import tempfile
import os

class TestVectorStoreIntegration:
    
    @pytest.fixture
    def vector_store(self):
        """Create vector store with temporary directory"""
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ChromaVectorStore(persist_directory=temp_dir)
            yield store
    
    def test_document_storage_and_retrieval(self, vector_store):
        """Test storing and retrieving documents"""
        # Add documents
        documents = [
            {
                'id': 'doc1',
                'content': 'Machine learning is a subset of artificial intelligence',
                'embedding': [0.1] * 3072,
                'metadata': {'type': 'pdf'}
            }
        ]
        
        result = vector_store.add_documents(documents)
        assert result.success is True
        assert result.stored_count == 1
        
        # Search documents
        query_embedding = [0.1] * 3072
        search_results = vector_store.search(query_embedding, limit=5)
        
        assert len(search_results) == 1
        assert search_results[0].document_id == 'doc1'
        assert search_results[0].score > 0.9  # Should be very similar
    
    def test_concurrent_operations(self, vector_store):
        """Test concurrent database operations"""
        import threading
        import time
        
        results = []
        errors = []
        
        def add_document(doc_id):
            try:
                doc = {
                    'id': doc_id,
                    'content': f'Document {doc_id} content',
                    'embedding': [0.1] * 3072,
                    'metadata': {'type': 'test'}
                }
                result = vector_store.add_documents([doc])
                results.append(result)
            except Exception as e:
                errors.append(e)
        
        # Create multiple threads
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_document, args=(f'doc_{i}',))
            threads.append(thread)
            thread.start()
        
        # Wait for completion
        for thread in threads:
            thread.join()
        
        # Verify results
        assert len(errors) == 0
        assert len(results) == 5
        assert all(r.success for r in results)
```

## 🌐 End-to-End Testing Strategy

### E2E Test Framework
```python
# tests/e2e/test_gradio_interface.py
import pytest
import requests
import time
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

class TestGradioInterface:
    """End-to-end tests for Gradio interface"""
    
    @pytest.fixture(scope="class")
    def driver(self):
        """Setup Selenium WebDriver"""
        options = webdriver.ChromeOptions()
        options.add_argument("--headless")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        
        driver = webdriver.Chrome(options=options)
        driver.implicitly_wait(10)
        
        yield driver
        driver.quit()
    
    @pytest.fixture(scope="class")
    def app_url(self):
        """Application URL for testing"""
        return "http://localhost:7860"
    
    def test_application_loads(self, driver, app_url):
        """Test that application loads successfully"""
        driver.get(app_url)
        
        # Wait for page to load
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.TAG_NAME, "body"))
        )
        
        # Verify title
        assert "SemanticScout" in driver.title
        
        # Verify main components are present
        assert driver.find_element(By.XPATH, "//h1[contains(text(), 'SemanticScout')]")
    
    def test_document_upload_flow(self, driver, app_url):
        """Test document upload functionality"""
        driver.get(app_url)
        
        # Navigate to upload tab
        upload_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Documents')]"))
        )
        upload_tab.click()
        
        # Find file upload component
        file_input = driver.find_element(By.CSS_SELECTOR, "input[type='file']")
        
        # Upload a test file (create temporary file)
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("This is a test document for SemanticScout.")
            test_file_path = f.name
        
        file_input.send_keys(test_file_path)
        
        # Wait for upload to complete
        WebDriverWait(driver, 30).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(text(), 'Upload complete')]"))
        )
        
        # Cleanup
        import os
        os.unlink(test_file_path)
    
    def test_search_functionality(self, driver, app_url):
        """Test search functionality"""
        driver.get(app_url)
        
        # Navigate to search tab
        search_tab = WebDriverWait(driver, 10).until(
            EC.element_to_be_clickable((By.XPATH, "//button[contains(text(), 'Search')]"))
        )
        search_tab.click()
        
        # Find search input
        search_input = driver.find_element(By.CSS_SELECTOR, "textarea[placeholder*='Ask anything']")
        search_input.clear()
        search_input.send_keys("machine learning algorithms")
        
        # Click search button
        search_button = driver.find_element(By.XPATH, "//button[contains(text(), 'Search')]")
        search_button.click()
        
        # Wait for results
        WebDriverWait(driver, 15).until(
            EC.presence_of_element_located((By.XPATH, "//div[contains(@class, 'results')]"))
        )
        
        # Verify results are displayed
        results = driver.find_elements(By.CSS_SELECTOR, ".search-result")
        assert len(results) >= 0  # May be empty if no documents uploaded
```

### API End-to-End Tests
```python
# tests/e2e/test_api_endpoints.py
import pytest
import requests
import json
import time

class TestAPIEndpoints:
    """Test API endpoints end-to-end"""
    
    @pytest.fixture
    def api_base_url(self):
        return "http://localhost:8000/api/v1"
    
    @pytest.fixture
    def auth_headers(self):
        return {"X-API-Key": "test-api-key"}
    
    def test_health_endpoint(self, api_base_url):
        """Test health check endpoint"""
        response = requests.get(f"{api_base_url}/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "timestamp" in data
        assert "checks" in data
    
    def test_document_upload_api(self, api_base_url, auth_headers):
        """Test document upload via API"""
        # Create test file
        test_content = "This is a test document for API testing."
        files = {
            'file': ('test.txt', test_content, 'text/plain')
        }
        
        response = requests.post(
            f"{api_base_url}/documents",
            files=files,
            headers=auth_headers
        )
        
        assert response.status_code == 201
        data = response.json()
        assert "id" in data
        assert data["filename"] == "test.txt"
        assert data["processing_status"] in ["pending", "processing", "completed"]
        
        return data["id"]
    
    def test_search_api(self, api_base_url, auth_headers):
        """Test search API endpoint"""
        search_payload = {
            "query": "test document",
            "limit": 10,
            "threshold": 0.5
        }
        
        response = requests.post(
            f"{api_base_url}/search",
            json=search_payload,
            headers=auth_headers
        )
        
        assert response.status_code == 200
        data = response.json()
        assert "results" in data
        assert "total_count" in data
        assert "query_time_ms" in data
        assert isinstance(data["results"], list)
```

## 🚀 Performance Testing Strategy

### Load Testing with Locust
```python
# tests/performance/locustfile.py
from locust import HttpUser, task, between
import json
import random

class SemanticScoutUser(HttpUser):
    wait_time = between(1, 3)
    
    def on_start(self):
        """Setup for each user"""
        self.auth_headers = {"X-API-Key": "test-api-key"}
    
    @task(3)
    def search_documents(self):
        """Simulate document search"""
        queries = [
            "machine learning algorithms",
            "artificial intelligence applications", 
            "data science techniques",
            "neural network architectures",
            "deep learning frameworks"
        ]
        
        query = random.choice(queries)
        payload = {
            "query": query,
            "limit": 10,
            "threshold": 0.7
        }
        
        with self.client.post(
            "/api/v1/search",
            json=payload,
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                data = response.json()
                if data.get("query_time_ms", 0) > 2000:
                    response.failure("Search took too long")
                else:
                    response.success()
            else:
                response.failure(f"Search failed: {response.status_code}")
    
    @task(1)
    def upload_document(self):
        """Simulate document upload"""
        test_content = f"Test document {random.randint(1, 1000)} with random content."
        files = {
            'file': (f'test_{random.randint(1, 1000)}.txt', test_content, 'text/plain')
        }
        
        with self.client.post(
            "/api/v1/documents",
            files=files,
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 201:
                response.success()
            else:
                response.failure(f"Upload failed: {response.status_code}")
    
    @task(1)
    def get_stats(self):
        """Check system statistics"""
        with self.client.get(
            "/api/v1/analytics/stats",
            headers=self.auth_headers,
            catch_response=True
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Stats request failed: {response.status_code}")
```

### Memory and Resource Testing
```python
# tests/performance/test_memory_usage.py
import pytest
import psutil
import time
from core.application import SemanticScoutApp

class TestMemoryUsage:
    """Test memory usage and resource management"""
    
    def test_memory_usage_during_document_processing(self):
        """Monitor memory usage during document processing"""
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        app = SemanticScoutApp()
        
        # Process multiple documents
        for i in range(10):
            large_content = "Large document content " * 1000
            app.upload_document(
                file_content=large_content.encode(),
                filename=f"large_doc_{i}.txt",
                file_type="txt"
            )
        
        # Wait for processing
        time.sleep(30)
        
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        # Assert memory usage is reasonable (< 500MB increase)
        assert memory_increase < 500, f"Memory usage increased by {memory_increase}MB"
    
    def test_search_performance_with_large_dataset(self):
        """Test search performance with many documents"""
        app = SemanticScoutApp()
        
        # Add many documents
        for i in range(100):
            content = f"Document {i} about machine learning and artificial intelligence"
            app.upload_document(
                file_content=content.encode(),
                filename=f"doc_{i}.txt",
                file_type="txt"
            )
        
        # Wait for processing
        time.sleep(60)
        
        # Measure search performance
        start_time = time.time()
        results = app.search("machine learning")
        search_time = (time.time() - start_time) * 1000  # ms
        
        assert search_time < 2000, f"Search took {search_time}ms (too slow)"
        assert len(results.results) > 0, "No results returned"
```

## 🔍 Test Data Management

### Test Data Factories
```python
# tests/factories.py
import factory
from datetime import datetime
import random

class DocumentFactory(factory.Factory):
    class Meta:
        model = dict
    
    id = factory.Sequence(lambda n: f"doc_{n}")
    filename = factory.LazyAttribute(lambda obj: f"{obj.id}.pdf")
    file_type = factory.Iterator(["pdf", "docx", "txt"])
    file_size = factory.LazyFunction(lambda: random.randint(1000, 100000))
    upload_date = factory.LazyFunction(datetime.now)
    content = factory.Faker('text', max_nb_chars=2000)
    metadata = factory.LazyFunction(lambda: {
        "author": factory.Faker('name'),
        "created_date": factory.Faker('date_time_this_year')
    })

class SearchQueryFactory(factory.Factory):
    class Meta:
        model = dict
    
    query = factory.Iterator([
        "machine learning algorithms",
        "artificial intelligence applications",
        "deep learning neural networks",
        "natural language processing",
        "computer vision techniques"
    ])
    limit = factory.LazyFunction(lambda: random.randint(5, 20))
    threshold = factory.LazyFunction(lambda: random.uniform(0.5, 0.9))
    filters = factory.LazyFunction(dict)

# Usage in tests
def test_with_factory_data():
    documents = DocumentFactory.create_batch(10)
    query = SearchQueryFactory.create()
    # Use in tests...
```

### Test Database Fixtures
```python
# tests/fixtures/database.py
import pytest
import tempfile
import shutil
from core.vector_store import ChromaVectorStore

@pytest.fixture(scope="session")
def test_database():
    """Create test database for session"""
    temp_dir = tempfile.mkdtemp()
    db = ChromaVectorStore(persist_directory=temp_dir)
    
    # Add sample data
    sample_documents = [
        {
            'id': 'sample_1',
            'content': 'This is a document about machine learning algorithms and their applications.',
            'embedding': [0.1] * 3072,
            'metadata': {'type': 'pdf', 'topic': 'ml'}
        },
        {
            'id': 'sample_2',
            'content': 'Artificial intelligence and its impact on modern technology.',
            'embedding': [0.2] * 3072,
            'metadata': {'type': 'docx', 'topic': 'ai'}
        }
    ]
    
    db.add_documents(sample_documents)
    
    yield db
    
    # Cleanup
    shutil.rmtree(temp_dir)

@pytest.fixture
def clean_database():
    """Create clean database for each test"""
    temp_dir = tempfile.mkdtemp()
    db = ChromaVectorStore(persist_directory=temp_dir)
    
    yield db
    
    shutil.rmtree(temp_dir)
```

## 📊 Test Reporting and Metrics

### Coverage Reporting
```python
# .coveragerc
[run]
source = core, config
omit = 
    */tests/*
    */venv/*
    */migrations/*
    setup.py

[report]
precision = 2
show_missing = True
skip_covered = False

[html]
directory = htmlcov
```

### Test Metrics Collection
```python
# tests/conftest.py
import pytest
import time
import json
from pathlib import Path

@pytest.fixture(autouse=True)
def test_metrics(request):
    """Collect test execution metrics"""
    start_time = time.time()
    
    yield
    
    end_time = time.time()
    duration = end_time - start_time
    
    # Collect metrics
    metrics = {
        'test_name': request.node.name,
        'duration_seconds': duration,
        'status': 'passed' if not hasattr(request.node, 'rep_call') else 'failed',
        'timestamp': start_time
    }
    
    # Save metrics
    metrics_file = Path('test_metrics.jsonl')
    with open(metrics_file, 'a') as f:
        f.write(json.dumps(metrics) + '\n')

# Generate test report
def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Generate test summary report"""
    passed = len(terminalreporter.stats.get('passed', []))
    failed = len(terminalreporter.stats.get('failed', []))
    skipped = len(terminalreporter.stats.get('skipped', []))
    
    print(f"\n=== Test Summary ===")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Skipped: {skipped}")
    print(f"Total: {passed + failed + skipped}")
```

## 🔄 Continuous Integration Testing

### GitHub Actions Workflow
```yaml
# .github/workflows/test.yml
name: Test Suite

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: [3.11, 3.12]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v4
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Cache dependencies
      uses: actions/cache@v3
      with:
        path: ~/.cache/pip
        key: ${{ runner.os }}-pip-${{ hashFiles('**/requirements.txt') }}
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install -r requirements-test.txt
    
    - name: Run linting
      run: |
        flake8 core/ config/ tests/
        black --check core/ config/ tests/
        isort --check-only core/ config/ tests/
    
    - name: Run unit tests
      run: |
        pytest tests/unit/ -v --cov=core --cov=config
    
    - name: Run integration tests
      run: |
        pytest tests/integration/ -v
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    
    - name: Upload coverage reports
      uses: codecov/codecov-action@v3
      with:
        file: ./coverage.xml
        flags: unittests
        name: codecov-umbrella

  e2e-test:
    runs-on: ubuntu-latest
    needs: test
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Set up Python 3.12
      uses: actions/setup-python@v4
      with:
        python-version: 3.12
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install selenium
    
    - name: Setup Chrome
      uses: browser-actions/setup-chrome@latest
    
    - name: Start application
      run: |
        python app.py &
        sleep 30
      env:
        OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
    
    - name: Run E2E tests
      run: |
        pytest tests/e2e/ -v
```

## 📋 Test Execution Guidelines

### Local Testing Commands
```bash
# Run all tests
pytest

# Run specific test types
pytest tests/unit/ -m unit
pytest tests/integration/ -m integration
pytest tests/e2e/ -m e2e

# Run with coverage
pytest --cov=core --cov-report=html

# Run performance tests
locust -f tests/performance/locustfile.py --host=http://localhost:7860

# Run tests in parallel
pytest -n auto

# Run specific test file
pytest tests/unit/test_document_processor.py -v

# Run tests matching pattern
pytest -k "test_search" -v
```

### Test Environment Setup
```bash
# Create test environment
python -m venv test_env
source test_env/bin/activate

# Install test dependencies
pip install -r requirements-test.txt

# Set test environment variables
export ENVIRONMENT=test
export OPENAI_API_KEY=test-key
export LOG_LEVEL=DEBUG
```

### Pre-commit Testing
```bash
#!/bin/bash
# scripts/pre-commit-test.sh

echo "Running pre-commit tests..."

# Linting
echo "Running linting..."
flake8 core/ config/ tests/
if [ $? -ne 0 ]; then
    echo "Linting failed!"
    exit 1
fi

# Type checking
echo "Running type checks..."
mypy core/ config/
if [ $? -ne 0 ]; then
    echo "Type checking failed!"
    exit 1
fi

# Unit tests
echo "Running unit tests..."
pytest tests/unit/ --cov=core --cov-fail-under=80
if [ $? -ne 0 ]; then
    echo "Unit tests failed!"
    exit 1
fi

echo "All pre-commit tests passed!"
```

---

*This comprehensive testing strategy ensures SemanticScout maintains high quality, reliability, and performance throughout its development lifecycle.*