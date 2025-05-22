# SemanticScout - AI Agent Development Guidelines

**Version**: 1.0  
**Date**: May 2025  
**Status**: Agent Ready

## 🤖 Agent Overview

This file provides comprehensive guidelines for AI development agents working on SemanticScout. These instructions ensure consistent, high-quality development aligned with project goals and technical standards.

## 🎯 Project Context

### Project Mission
SemanticScout is a professional-grade semantic document search application designed for:
- **Enterprise demonstrations**: Showcasing AI capabilities to potential clients
- **Portfolio enhancement**: Demonstrating advanced technical skills for freelance platforms
- **Technology validation**: Proving competency with latest 2025 AI/ML technologies

### Success Definition
A successful implementation must be:
- **Professional**: Enterprise-grade appearance and functionality
- **Reliable**: Stable operation suitable for live demonstrations
- **Performant**: Response times meeting business requirements
- **Maintainable**: Clean, documented code for future enhancement

## 📁 Codebase Structure

### Primary Working Directories
```
SemanticScout/
├── core/                    # Core business logic - PRIMARY FOCUS
├── config/                  # Configuration management
├── tests/                   # Comprehensive test suite
├── docs/                    # Complete documentation (READ FIRST)
├── app.py                   # Main Gradio application entry point
└── requirements.txt         # Dependencies (keep updated)
```

### Documentation Priority
**ALWAYS read documentation before coding**:
1. `docs/PRD.md` - Product requirements and business context
2. `docs/TECHNICAL_STACK.md` - Technology choices and dependencies
3. `docs/ARCHITECTURE.md` - System design and component structure
4. `docs/API_SPECIFICATION.md` - API contracts and data models
5. `docs/UI_GUIDELINES.md` - User interface standards
6. `TODO.md` - Current development tasks and priorities

## 🛠️ Development Standards

### Code Quality Requirements

#### Python Standards
```python
# REQUIRED: Type hints for all functions
def process_document(file_path: str, file_type: str) -> ProcessingResult:
    """
    Process uploaded document and extract content.
    
    Args:
        file_path: Absolute path to document file
        file_type: File extension (pdf, docx, txt, md)
    
    Returns:
        ProcessingResult with success status and extracted content
    
    Raises:
        DocumentProcessingError: When file cannot be processed
    """
    pass

# REQUIRED: Comprehensive error handling
try:
    result = risky_operation()
except SpecificException as e:
    logger.error(f"Operation failed: {e}")
    raise ProcessingError(f"Friendly error message: {e}")

# REQUIRED: Logging for all significant operations
import logging
logger = logging.getLogger(__name__)

def important_function():
    logger.info("Starting important operation")
    # ... operation logic
    logger.debug(f"Operation completed with result: {result}")
```

#### Dependency Management
```python
# ALWAYS use exact versions from TECHNICAL_STACK.md
# NEVER upgrade dependencies without documentation update
# ALWAYS test after dependency changes

# Example from requirements.txt
langchain>=0.1.0
openai>=1.12.0
chromadb>=0.4.0
```

### Testing Requirements

#### Test Coverage Mandate
- **Minimum 80% coverage** for all new code
- **Unit tests** for all business logic functions
- **Integration tests** for external API interactions
- **Mock external dependencies** (OpenAI API, file system)

#### Test Structure
```python
# tests/unit/test_document_processor.py
import pytest
from unittest.mock import Mock, patch
from core.document_processor import DocumentProcessor

class TestDocumentProcessor:
    @pytest.fixture
    def document_processor(self):
        return DocumentProcessor()
    
    def test_process_pdf_success(self, document_processor):
        # REQUIRED: Test happy path
        pass
    
    def test_process_invalid_format(self, document_processor):
        # REQUIRED: Test error cases
        pass
    
    @patch('core.document_processor.OpenAI')
    def test_api_failure_handling(self, mock_openai, document_processor):
        # REQUIRED: Test external dependency failures
        pass
```

## 🔧 Implementation Guidelines

### Configuration Management
```python
# ALWAYS use centralized configuration
from config.settings import settings

# NEVER hardcode values
openai_client = OpenAI(api_key=settings.openai_api_key)

# ALWAYS validate configuration on startup
def validate_config():
    if not settings.openai_api_key:
        raise ConfigurationError("OpenAI API key required")
```

### Error Handling Strategy
```python
# CREATE specific exceptions for different failure modes
class DocumentProcessingError(Exception):
    """Raised when document processing fails"""
    pass

class EmbeddingError(Exception):
    """Raised when embedding generation fails"""
    pass

# PROVIDE actionable error messages to users
try:
    process_document(file_path)
except DocumentProcessingError as e:
    return ProcessingResult(
        success=False,
        error_message="Unable to process document. Please check file format and try again."
    )
```

### Performance Requirements
```python
# ENFORCE performance requirements in code
import time
from functools import wraps

def performance_monitor(max_seconds: float):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time
            
            if duration > max_seconds:
                logger.warning(f"{func.__name__} took {duration:.2f}s (max: {max_seconds}s)")
            
            return result
        return wrapper
    return decorator

@performance_monitor(2.0)  # Search must complete in 2 seconds
def search_documents(query: str) -> SearchResponse:
    pass
```

## 🎨 User Interface Standards

### Gradio Component Guidelines
```python
import gradio as gr
from config.settings import settings

# ALWAYS use consistent theme
theme = gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="gray"
)

# PROVIDE helpful labels and descriptions
search_input = gr.Textbox(
    label="Search Query",
    placeholder="Ask anything about your documents...",
    info="Use natural language to find relevant content"
)

# IMPLEMENT proper error display
def handle_search_error(error: Exception) -> str:
    logger.error(f"Search failed: {error}")
    return "❌ Search failed. Please try again or contact support."
```

### User Experience Requirements
- **Immediate feedback**: Show loading states for all operations
- **Clear error messages**: User-friendly descriptions, not technical details
- **Progress indicators**: For long-running operations (document processing)
- **Consistent styling**: Follow UI_GUIDELINES.md color palette and typography

## 🔌 External Integration Standards

### OpenAI API Integration
```python
import openai
from tenacity import retry, stop_after_attempt, wait_exponential

class EmbeddingService:
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10)
    )
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding with retry logic"""
        try:
            response = self.client.embeddings.create(
                model=settings.openai_embedding_model,
                input=text,
                dimensions=settings.embedding_dimensions
            )
            return response.data[0].embedding
        except openai.RateLimitError:
            logger.warning("Rate limit hit, retrying...")
            raise
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {e}")
```

### ChromaDB Integration
```python
import chromadb
from chromadb.config import Settings as ChromaSettings

class VectorStore:
    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=settings.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False)
        )
        self.collection = self._get_or_create_collection()
    
    def _get_or_create_collection(self):
        """Initialize collection with proper configuration"""
        return self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "SemanticScout document collection"}
        )
```

## 📊 Testing and Validation

### Development Workflow
1. **Read TODO.md** to understand current development phase
2. **Implement feature** following architecture specifications
3. **Write comprehensive tests** with 80%+ coverage
4. **Validate performance** against requirements
5. **Update documentation** if APIs change
6. **Test integration** with existing components

### Documentation Maintenance Requirements
**CRITICAL**: You MUST maintain documentation accuracy throughout development:

1. **TODO.md Maintenance**:
   - **ALWAYS** update TODO.md when completing tasks
   - **MARK COMPLETED** tasks immediately after finishing
   - **ADD NEW TASKS** if scope evolves during development
   - **UPDATE STATUS** of in-progress tasks regularly
   - **ESTIMATE REMAINING** work accurately

2. **README.md Maintenance**:
   - **UPDATE** setup instructions as project evolves
   - **ADD** new features to feature list
   - **MODIFY** installation requirements when dependencies change
   - **KEEP** usage examples current with actual implementation
   - **MAINTAIN** project description accuracy

3. **Technical Documentation Updates**:
   - **TECHNICAL_STACK.md**: Update if dependencies or versions change
   - **ARCHITECTURE.md**: Modify if system design evolves
   - **API_SPECIFICATION.md**: Update if API contracts change
   - **UI_GUIDELINES.md**: Revise if interface standards change
   - **DEPLOYMENT.md**: Update if deployment process changes

3. **Documentation Sync Process**:
   - **BEFORE** implementing major changes, review affected docs
   - **DURING** development, note documentation changes needed
   - **AFTER** completing features, update ALL affected documentation
   - **VALIDATE** that documentation matches actual implementation

4. **Documentation Quality Standards**:
   - **ACCURACY**: All code examples must work as written
   - **COMPLETENESS**: All new features must be documented
   - **CONSISTENCY**: Maintain same format and style throughout
   - **TIMELINESS**: Update docs within same PR as code changes

### Validation Commands
```bash
# ALWAYS run before committing
pytest tests/ --cov=core --cov-fail-under=80
flake8 core/ config/ tests/
black --check core/ config/ tests/
mypy core/ config/

# PERFORMANCE validation
python scripts/performance_test.py

# INTEGRATION validation
python scripts/health_check.py
```

### Mock Strategy for Testing
```python
# MOCK external APIs consistently
@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock:
        mock.return_value.embeddings.create.return_value = Mock(
            data=[Mock(embedding=[0.1] * 3072)]
        )
        yield mock

# TEST error scenarios
def test_openai_rate_limit_handling(mock_openai):
    mock_openai.return_value.embeddings.create.side_effect = openai.RateLimitError("Rate limit")
    # Verify retry logic works correctly
```

## 🚨 Critical Requirements

### Security Constraints
- **NEVER commit API keys** to version control
- **ALWAYS validate file uploads** (type, size, content)
- **SANITIZE user inputs** before processing
- **USE environment variables** for all secrets

### Performance Constraints
- **Document processing**: < 30 seconds per file
- **Search operations**: < 2 seconds response time
- **Memory usage**: < 2GB for 1000 documents
- **UI responsiveness**: < 500ms for interactions

### Quality Constraints
- **Test coverage**: Minimum 80% for all modules
- **Error handling**: All external dependencies must have error handling
- **Logging**: All significant operations must be logged
- **Documentation**: All public APIs must have docstrings

## 🔄 Development Process

### Git Workflow and Branch Strategy

#### Branch Management
- **Main branch**: Contains production-ready code, always stable
- **Feature branches**: Create new branch for each PR from main
- **No develop branch**: Work directly with main and feature branches

#### Agent Workflow
1. **Start from main branch** 
2. **Create new feature branch** for current PR (format: `feature/PR{X}-description`)
3. **Complete entire PR scope** in that branch
4. **Push branch when ready** for PR creation

### Feature Implementation Process
1. **Understand requirements** from PRD.md and TODO.md
2. **Create feature branch** following naming convention
3. **Review architecture** in ARCHITECTURE.md for design guidance
4. **Implement core logic** with proper error handling and logging
5. **Write comprehensive tests** including edge cases
6. **Add user interface** following UI_GUIDELINES.md
7. **Validate performance** against documented requirements
8. **Update documentation** if needed
9. **Prepare PR** with clear description and validation

### Code Review Checklist
- [ ] Follows Python PEP 8 style guidelines
- [ ] Has comprehensive type hints
- [ ] Includes proper error handling
- [ ] Has meaningful logging statements
- [ ] Meets performance requirements
- [ ] Has 80%+ test coverage
- [ ] Follows security best practices
- [ ] Updates documentation if needed

### Commit Standards
```bash
# USE conventional commit format
git commit -m "feat: implement document upload with validation

- Add multi-format file upload support
- Implement file validation (type, size)
- Add progress tracking for large files
- Include comprehensive error handling

Closes #issue-number"
```

## 🎯 Success Validation

### Before Declaring Complete
- [ ] **All tests pass** including unit, integration, and E2E
- [ ] **Performance requirements met** with evidence
- [ ] **Error scenarios handled** gracefully
- [ ] **User experience polished** and professional
- [ ] **Documentation updated** and accurate
- [ ] **Demo-ready** for client presentations

### Demo Readiness Checklist
- [ ] Application starts without errors
- [ ] Sample documents process successfully
- [ ] Search returns relevant results
- [ ] Visualizations render correctly
- [ ] Error states display user-friendly messages
- [ ] Performance meets requirements under normal load

## 🆘 Troubleshooting

### Common Issues and Solutions

#### OpenAI API Issues
```python
# Handle rate limits gracefully
except openai.RateLimitError:
    logger.warning("Rate limit hit, implementing backoff")
    time.sleep(60)  # Wait before retry

# Handle API key issues
except openai.AuthenticationError:
    raise ConfigurationError("Invalid OpenAI API key")
```

#### ChromaDB Issues
```python
# Handle database connection issues
try:
    collection = client.get_collection("documents")
except chromadb.errors.InvalidCollectionException:
    collection = client.create_collection("documents")
```

#### Memory Issues
```python
# Process large files in chunks
def process_large_file(file_path: str) -> Iterator[str]:
    with open(file_path, 'r', encoding='utf-8') as f:
        while True:
            chunk = f.read(8192)  # 8KB chunks
            if not chunk:
                break
            yield chunk
```

### Getting Help
1. **Check documentation** in docs/ directory first
2. **Review similar implementations** in existing code
3. **Check TODO.md** for context on current development phase
4. **Validate configuration** using scripts/health_check.py

---

*These guidelines ensure consistent, high-quality development of SemanticScout. Follow them precisely to create a professional, demonstrable application that meets all business and technical requirements.*