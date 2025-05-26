# PR2: Core Data Models & Exceptions - Detailed Implementation Guide

## Overview
This PR establishes the foundation data models for the chat-with-documents application, including models for chat functionality, document processing, and search.

## File Structure
```
core/
├── models/
│   ├── __init__.py
│   ├── document.py      # Document and chunk models
│   ├── chat.py          # Chat-related models
│   ├── search.py        # Search query and results
│   └── visualization.py # Visualization data structures
├── exceptions/
│   ├── __init__.py
│   └── custom_exceptions.py
└── utils/
    ├── __init__.py
    ├── validation.py
    └── text_processing.py
```

## Detailed Implementation

### 1. Document Models (`core/models/document.py`)

```python
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Dict, Any
from datetime import datetime
import hashlib

class Document(BaseModel):
    """Represents an uploaded document."""
    id: str = Field(..., description="Unique document identifier")
    filename: str = Field(..., description="Original filename")
    file_type: str = Field(..., description="File extension (pdf, docx, txt, md)")
    file_size: int = Field(..., description="File size in bytes")
    content: str = Field(..., description="Full text content")
    upload_date: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    chunk_ids: List[str] = Field(default_factory=list)
    
    @validator('file_type')
    def validate_file_type(cls, v):
        allowed = ['pdf', 'docx', 'txt', 'md']
        if v.lower() not in allowed:
            raise ValueError(f"File type must be one of {allowed}")
        return v.lower()
    
    @validator('file_size')
    def validate_file_size(cls, v):
        max_size = 100 * 1024 * 1024  # 100MB
        if v > max_size:
            raise ValueError(f"File size exceeds maximum of {max_size} bytes")
        return v
    
    def generate_id(self) -> str:
        """Generate unique ID based on content hash."""
        content_hash = hashlib.sha256(self.content.encode()).hexdigest()
        return f"doc_{content_hash[:12]}"

class DocumentChunk(BaseModel):
    """Represents a chunk of document for RAG."""
    id: str = Field(..., description="Unique chunk identifier")
    document_id: str = Field(..., description="Parent document ID")
    content: str = Field(..., description="Chunk text content")
    chunk_index: int = Field(..., description="Position in document")
    start_char: int = Field(..., description="Starting character position")
    end_char: int = Field(..., description="Ending character position")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @validator('content')
    def validate_content_length(cls, v):
        if len(v.strip()) < 10:
            raise ValueError("Chunk content too short")
        return v
    
    def generate_id(self) -> str:
        """Generate unique chunk ID."""
        return f"chunk_{self.document_id}_{self.chunk_index}"
```

### 2. Chat Models (`core/models/chat.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum

class MessageRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"

class ChatMessage(BaseModel):
    """Represents a single chat message."""
    role: MessageRole
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
class ChatContext(BaseModel):
    """Context for generating chat responses."""
    messages: List[ChatMessage]
    retrieved_chunks: List[DocumentChunk] = Field(default_factory=list)
    max_context_length: int = Field(default=8000)
    
    def get_context_string(self) -> str:
        """Format retrieved chunks as context."""
        if not self.retrieved_chunks:
            return ""
        
        context_parts = []
        for chunk in self.retrieved_chunks:
            source = chunk.metadata.get('filename', 'Unknown')
            context_parts.append(f"[Source: {source}]\n{chunk.content}\n")
        
        return "\n---\n".join(context_parts)
    
    def format_for_llm(self) -> List[Dict[str, str]]:
        """Format messages for OpenAI API."""
        formatted = []
        
        # Add system message with context if available
        context = self.get_context_string()
        if context:
            system_content = f"""You are a helpful AI assistant with access to the user's documents.
            
Based on the following document excerpts, answer the user's questions:

{context}

If the answer isn't in the provided context, say so."""
        else:
            system_content = "You are a helpful AI assistant. The user hasn't uploaded any documents yet."
        
        formatted.append({"role": "system", "content": system_content})
        
        # Add conversation history
        for msg in self.messages:
            formatted.append({"role": msg.role.value, "content": msg.content})
        
        return formatted
```

### 3. Search Models (`core/models/search.py`)

```python
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
from datetime import datetime

class SearchQuery(BaseModel):
    """Represents a search query."""
    query_text: str = Field(..., min_length=1)
    max_results: int = Field(default=10, ge=1, le=50)
    similarity_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    filter_file_types: Optional[List[str]] = None
    filter_date_range: Optional[tuple[datetime, datetime]] = None
    include_metadata: bool = Field(default=True)
    
    @validator('query_text')
    def clean_query(cls, v):
        cleaned = v.strip()
        if not cleaned:
            raise ValueError("Query cannot be empty")
        return cleaned

class SearchResult(BaseModel):
    """Represents a single search result."""
    chunk_id: str
    document_id: str
    score: float = Field(..., ge=0.0, le=1.0)
    content: str
    highlighted_content: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @property
    def source_info(self) -> Dict[str, Any]:
        """Get source document information."""
        return {
            "filename": self.metadata.get("filename", "Unknown"),
            "file_type": self.metadata.get("file_type", "Unknown"),
            "chunk_index": self.metadata.get("chunk_index", 0)
        }

class SearchResponse(BaseModel):
    """Complete search response."""
    query: SearchQuery
    results: List[SearchResult]
    total_results: int
    search_time_ms: float
    
    def format_for_display(self) -> List[Dict[str, Any]]:
        """Format results for Gradio display."""
        formatted = []
        for result in self.results:
            formatted.append({
                "source": result.source_info["filename"],
                "score": f"{result.score:.2%}",
                "content": result.highlighted_content or result.content[:200] + "...",
                "metadata": result.metadata
            })
        return formatted
```

### 4. Visualization Models (`core/models/visualization.py`)

```python
from pydantic import BaseModel, Field
from typing import List, Tuple, Dict, Any, Optional

class DocumentNode(BaseModel):
    """Represents a document in visualization."""
    document_id: str
    label: str
    position: Tuple[float, float]
    size: float = Field(default=10.0)
    color: str = Field(default="#3498db")
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentEdge(BaseModel):
    """Represents similarity between documents."""
    source_id: str
    target_id: str
    weight: float = Field(..., ge=0.0, le=1.0)
    label: Optional[str] = None

class VisualizationData(BaseModel):
    """Data for document visualization."""
    nodes: List[DocumentNode]
    edges: List[DocumentEdge]
    layout_type: str = Field(default="force")
    
    def to_plotly_format(self) -> Dict[str, Any]:
        """Convert to Plotly graph format."""
        # Implementation for Plotly visualization
        pass
```

### 5. Custom Exceptions (`core/exceptions/custom_exceptions.py`)

```python
class SemanticScoutError(Exception):
    """Base exception for all custom errors."""
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.details = details or {}

class DocumentProcessingError(SemanticScoutError):
    """Raised when document processing fails."""
    pass

class EmbeddingError(SemanticScoutError):
    """Raised when embedding generation fails."""
    pass

class VectorStoreError(SemanticScoutError):
    """Raised when vector database operations fail."""
    pass

class ValidationError(SemanticScoutError):
    """Raised when input validation fails."""
    pass

class SearchError(SemanticScoutError):
    """Raised when search operations fail."""
    pass

class ChatError(SemanticScoutError):
    """Raised when chat operations fail."""
    pass

class RateLimitError(EmbeddingError):
    """Raised when API rate limit is hit."""
    def __init__(self, retry_after: int = 60):
        super().__init__(f"Rate limit exceeded. Retry after {retry_after} seconds")
        self.retry_after = retry_after
```

### 6. Validation Utilities (`core/utils/validation.py`)

```python
import os
import magic
from typing import List, Tuple

ALLOWED_EXTENSIONS = ['pdf', 'docx', 'txt', 'md']
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

def validate_file_type(file_path: str) -> Tuple[bool, str]:
    """Validate file type using magic numbers."""
    if not os.path.exists(file_path):
        return False, "File not found"
    
    # Check extension
    ext = os.path.splitext(file_path)[1].lower().lstrip('.')
    if ext not in ALLOWED_EXTENSIONS:
        return False, f"File type '{ext}' not supported"
    
    # Check MIME type
    mime = magic.from_file(file_path, mime=True)
    valid_mimes = {
        'pdf': 'application/pdf',
        'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'txt': 'text/plain',
        'md': 'text/plain'
    }
    
    expected_mime = valid_mimes.get(ext)
    if mime != expected_mime and not (ext in ['txt', 'md'] and mime.startswith('text/')):
        return False, f"File content doesn't match extension"
    
    return True, "Valid"

def validate_file_size(file_path: str) -> Tuple[bool, str]:
    """Validate file size."""
    size = os.path.getsize(file_path)
    if size > MAX_FILE_SIZE:
        return False, f"File size {size} exceeds maximum {MAX_FILE_SIZE}"
    return True, "Valid"

def sanitize_text(text: str) -> str:
    """Clean and sanitize text content."""
    # Remove null bytes
    text = text.replace('\x00', '')
    # Normalize whitespace
    text = ' '.join(text.split())
    return text.strip()
```

### 7. Text Processing (`core/utils/text_processing.py`)

```python
import re
from typing import List, Tuple

def clean_text(text: str) -> str:
    """Clean text for processing."""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters but keep punctuation
    text = re.sub(r'[^\w\s\.\,\!\?\-\:\;\'"]+', '', text)
    return text.strip()

def extract_sentences(text: str) -> List[str]:
    """Extract sentences from text."""
    # Simple sentence splitting
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def calculate_overlap(text: str, start: int, end: int, overlap_size: int) -> Tuple[int, int]:
    """Calculate chunk boundaries with overlap."""
    # Implementation for overlap calculation
    pass
```

## Testing Requirements

Create comprehensive tests for all models and utilities:

```python
# tests/unit/test_models.py
import pytest
from core.models.document import Document, DocumentChunk
from core.models.chat import ChatMessage, ChatContext, MessageRole

def test_document_validation():
    # Test valid document
    doc = Document(
        id="test123",
        filename="test.pdf",
        file_type="pdf",
        file_size=1000,
        content="Test content"
    )
    assert doc.file_type == "pdf"
    
    # Test invalid file type
    with pytest.raises(ValueError):
        Document(
            id="test123",
            filename="test.exe",
            file_type="exe",
            file_size=1000,
            content="Test"
        )

def test_chat_context_formatting():
    # Test context formatting for LLM
    context = ChatContext(
        messages=[
            ChatMessage(role=MessageRole.USER, content="Hello"),
            ChatMessage(role=MessageRole.ASSISTANT, content="Hi there!")
        ]
    )
    formatted = context.format_for_llm()
    assert len(formatted) == 3  # system + 2 messages
    assert formatted[0]["role"] == "system"
```

## Integration Points

These models will be used by:
- **PR3**: Document processing will create Document and DocumentChunk instances
- **PR4**: Embedding service will update DocumentChunk with embeddings
- **PR5**: Vector store will persist and retrieve these models
- **PR6**: Search engine will use SearchQuery and return SearchResult
- **PR7**: Gradio UI will display these models
- **PR8**: Visualization will use VisualizationData

## Success Criteria

1. All models have complete Pydantic validation
2. Type hints are comprehensive
3. Docstrings explain each field
4. Validation catches edge cases
5. Models are serializable to JSON
6. Test coverage > 95% for models
7. Utilities handle errors gracefully