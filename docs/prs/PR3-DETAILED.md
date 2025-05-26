# PR3: Document Processing Engine - Detailed Implementation Guide

## Overview
This PR implements the document processing pipeline that extracts and prepares text from various file formats for the chat and search features.

## Prerequisites
- PR2 completed (models and exceptions defined)
- Dependencies: PyMuPDF, python-docx, Unstructured

## File Structure
```
core/
├── document_processor.py    # Main processing orchestrator
├── extractors/
│   ├── __init__.py
│   ├── pdf_extractor.py    # PDF text extraction
│   ├── docx_extractor.py   # Word document extraction
│   ├── text_extractor.py   # TXT/MD extraction
│   └── base_extractor.py   # Abstract base class
├── chunking/
│   ├── __init__.py
│   ├── text_chunker.py     # Smart text chunking
│   └── chunk_manager.py    # Chunk overlap handling
└── processors/
    ├── __init__.py
    └── content_cleaner.py  # Text cleaning utilities
```

## Detailed Implementation

### 1. Base Extractor (`core/extractors/base_extractor.py`)

```python
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from core.models.document import Document
from core.exceptions.custom_exceptions import DocumentProcessingError

class BaseExtractor(ABC):
    """Abstract base class for document extractors."""
    
    @abstractmethod
    def can_extract(self, file_path: str) -> bool:
        """Check if this extractor can handle the file."""
        pass
    
    @abstractmethod
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text and metadata from file."""
        pass
    
    def validate_file(self, file_path: str) -> None:
        """Common file validation logic."""
        import os
        if not os.path.exists(file_path):
            raise DocumentProcessingError(f"File not found: {file_path}")
        
        if os.path.getsize(file_path) > 100 * 1024 * 1024:  # 100MB
            raise DocumentProcessingError("File too large (max 100MB)")
```

### 2. PDF Extractor (`core/extractors/pdf_extractor.py`)

```python
import pymupdf
from typing import Dict, Any
from .base_extractor import BaseExtractor
from core.exceptions.custom_exceptions import DocumentProcessingError
import logging

logger = logging.getLogger(__name__)

class PDFExtractor(BaseExtractor):
    """Extract text from PDF files using PyMuPDF."""
    
    def can_extract(self, file_path: str) -> bool:
        return file_path.lower().endswith('.pdf')
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text and metadata from PDF."""
        self.validate_file(file_path)
        
        try:
            text_content = []
            metadata = {}
            
            with pymupdf.open(file_path) as pdf:
                # Extract metadata
                metadata = {
                    'page_count': len(pdf),
                    'title': pdf.metadata.get('title', ''),
                    'author': pdf.metadata.get('author', ''),
                    'subject': pdf.metadata.get('subject', ''),
                    'creator': pdf.metadata.get('creator', ''),
                }
                
                # Extract text from each page
                for page_num, page in enumerate(pdf):
                    try:
                        text = page.get_text()
                        if text.strip():
                            text_content.append({
                                'page': page_num + 1,
                                'content': text
                            })
                    except Exception as e:
                        logger.warning(f"Failed to extract page {page_num + 1}: {e}")
                        continue
            
            if not text_content:
                raise DocumentProcessingError("No text content found in PDF")
            
            # Combine all pages
            full_text = "\n\n".join([f"[Page {p['page']}]\n{p['content']}" 
                                    for p in text_content])
            
            return {
                'content': full_text,
                'metadata': metadata,
                'pages': text_content
            }
            
        except Exception as e:
            logger.error(f"PDF extraction failed: {e}")
            raise DocumentProcessingError(f"Failed to extract PDF: {str(e)}")
```

### 3. DOCX Extractor (`core/extractors/docx_extractor.py`)

```python
from docx import Document as DocxDocument
from typing import Dict, Any
from .base_extractor import BaseExtractor
from core.exceptions.custom_exceptions import DocumentProcessingError
import logging

logger = logging.getLogger(__name__)

class DOCXExtractor(BaseExtractor):
    """Extract text from Word documents."""
    
    def can_extract(self, file_path: str) -> bool:
        return file_path.lower().endswith('.docx')
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text and metadata from DOCX."""
        self.validate_file(file_path)
        
        try:
            doc = DocxDocument(file_path)
            
            # Extract text from paragraphs
            paragraphs = []
            for para in doc.paragraphs:
                if para.text.strip():
                    paragraphs.append(para.text)
            
            # Extract text from tables
            table_texts = []
            for table in doc.tables:
                table_text = []
                for row in table.rows:
                    row_text = [cell.text.strip() for cell in row.cells]
                    if any(row_text):
                        table_text.append(' | '.join(row_text))
                if table_text:
                    table_texts.append('\n'.join(table_text))
            
            # Combine all content
            full_text = '\n\n'.join(paragraphs)
            if table_texts:
                full_text += '\n\n[Tables]\n' + '\n\n'.join(table_texts)
            
            # Extract metadata
            metadata = {
                'paragraph_count': len(paragraphs),
                'table_count': len(doc.tables),
                'author': doc.core_properties.author or '',
                'title': doc.core_properties.title or '',
                'created': str(doc.core_properties.created) if doc.core_properties.created else '',
            }
            
            return {
                'content': full_text,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"DOCX extraction failed: {e}")
            raise DocumentProcessingError(f"Failed to extract DOCX: {str(e)}")
```

### 4. Text Extractor (`core/extractors/text_extractor.py`)

```python
from typing import Dict, Any
from .base_extractor import BaseExtractor
from core.exceptions.custom_exceptions import DocumentProcessingError
import chardet
import logging

logger = logging.getLogger(__name__)

class TextExtractor(BaseExtractor):
    """Extract text from plain text files (TXT, MD)."""
    
    def can_extract(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.txt', '.md'))
    
    def extract(self, file_path: str) -> Dict[str, Any]:
        """Extract text content with encoding detection."""
        self.validate_file(file_path)
        
        try:
            # Detect encoding
            with open(file_path, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding'] or 'utf-8'
            
            # Read with detected encoding
            with open(file_path, 'r', encoding=encoding) as f:
                content = f.read()
            
            if not content.strip():
                raise DocumentProcessingError("File is empty")
            
            metadata = {
                'encoding': encoding,
                'line_count': len(content.splitlines()),
                'char_count': len(content)
            }
            
            return {
                'content': content,
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            raise DocumentProcessingError(f"Failed to extract text: {str(e)}")
```

### 5. Text Chunker (`core/chunking/text_chunker.py`)

```python
from typing import List, Tuple
from langchain.text_splitter import RecursiveCharacterTextSplitter
from core.models.document import DocumentChunk
import tiktoken
import logging

logger = logging.getLogger(__name__)

class TextChunker:
    """Smart text chunking with overlap for RAG."""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.encoding = tiktoken.encoding_for_model("gpt-4")
        
        # Initialize splitter
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=self._token_length,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def _token_length(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def chunk_document(self, document_id: str, content: str) -> List[DocumentChunk]:
        """Split document into chunks."""
        if not content.strip():
            return []
        
        # Split text
        texts = self.splitter.split_text(content)
        
        # Create chunks
        chunks = []
        char_index = 0
        
        for i, text in enumerate(texts):
            # Find actual position in original content
            start_char = content.find(text, char_index)
            end_char = start_char + len(text)
            char_index = start_char + len(text) - self.chunk_overlap
            
            chunk = DocumentChunk(
                id=f"chunk_{document_id}_{i}",
                document_id=document_id,
                content=text,
                chunk_index=i,
                start_char=start_char,
                end_char=end_char,
                metadata={
                    'token_count': self._token_length(text),
                    'chunk_total': len(texts)
                }
            )
            chunks.append(chunk)
        
        logger.info(f"Created {len(chunks)} chunks for document {document_id}")
        return chunks
```

### 6. Document Processor (`core/document_processor.py`)

```python
from typing import List, Dict, Any, Optional
from pathlib import Path
import hashlib
import logging
from datetime import datetime

from core.models.document import Document, DocumentChunk
from core.exceptions.custom_exceptions import DocumentProcessingError
from .extractors import PDFExtractor, DOCXExtractor, TextExtractor
from .chunking.text_chunker import TextChunker
from .processors.content_cleaner import ContentCleaner

logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Main document processing orchestrator."""
    
    def __init__(self):
        self.extractors = [
            PDFExtractor(),
            DOCXExtractor(),
            TextExtractor()
        ]
        self.chunker = TextChunker()
        self.cleaner = ContentCleaner()
    
    def process_document(self, file_path: str) -> Tuple[Document, List[DocumentChunk]]:
        """Process a document and return Document + chunks."""
        logger.info(f"Processing document: {file_path}")
        
        # Find appropriate extractor
        extractor = self._get_extractor(file_path)
        if not extractor:
            raise DocumentProcessingError(f"No extractor found for file: {file_path}")
        
        # Extract content
        extracted = extractor.extract(file_path)
        content = extracted['content']
        metadata = extracted.get('metadata', {})
        
        # Clean content
        cleaned_content = self.cleaner.clean_text(content)
        
        # Create document
        file_path_obj = Path(file_path)
        document = Document(
            id=self._generate_document_id(cleaned_content),
            filename=file_path_obj.name,
            file_type=file_path_obj.suffix.lower().lstrip('.'),
            file_size=file_path_obj.stat().st_size,
            content=cleaned_content,
            metadata=metadata
        )
        
        # Create chunks
        chunks = self.chunker.chunk_document(document.id, cleaned_content)
        document.chunk_ids = [chunk.id for chunk in chunks]
        
        logger.info(f"Document processed: {document.id} with {len(chunks)} chunks")
        return document, chunks
    
    def _get_extractor(self, file_path: str):
        """Find the appropriate extractor for file."""
        for extractor in self.extractors:
            if extractor.can_extract(file_path):
                return extractor
        return None
    
    def _generate_document_id(self, content: str) -> str:
        """Generate unique document ID from content."""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        return f"doc_{content_hash[:12]}"
```

### 7. Content Cleaner (`core/processors/content_cleaner.py`)

```python
import re
from typing import Optional

class ContentCleaner:
    """Clean and normalize text content."""
    
    def clean_text(self, text: str) -> str:
        """Clean text for processing."""
        if not text:
            return ""
        
        # Remove null bytes
        text = text.replace('\x00', '')
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove control characters
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\r\t')
        
        # Normalize line breaks
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)
        
        # Remove excessive line breaks
        text = re.sub(r'\n{3,}', '\n\n', text)
        
        return text.strip()
```

## Testing Requirements

```python
# tests/unit/test_document_processor.py
import pytest
from pathlib import Path
from core.document_processor import DocumentProcessor
from core.exceptions.custom_exceptions import DocumentProcessingError

@pytest.fixture
def sample_pdf(tmp_path):
    # Create sample PDF for testing
    pdf_path = tmp_path / "test.pdf"
    # ... create PDF content
    return str(pdf_path)

def test_process_pdf_document(sample_pdf):
    processor = DocumentProcessor()
    doc, chunks = processor.process_document(sample_pdf)
    
    assert doc.file_type == "pdf"
    assert len(chunks) > 0
    assert all(chunk.document_id == doc.id for chunk in chunks)

def test_process_unsupported_file():
    processor = DocumentProcessor()
    with pytest.raises(DocumentProcessingError):
        processor.process_document("test.xyz")

def test_chunk_overlap():
    from core.chunking.text_chunker import TextChunker
    chunker = TextChunker(chunk_size=100, chunk_overlap=20)
    
    text = "A" * 250  # Text longer than 2 chunks
    chunks = chunker.chunk_document("test_doc", text)
    
    # Verify overlap exists
    assert len(chunks) >= 2
    if len(chunks) >= 2:
        overlap = chunks[0].content[-20:]
        assert overlap in chunks[1].content
```

## Integration Points

- **PR2 Models**: Uses Document and DocumentChunk models
- **PR4 Embeddings**: Processed chunks will be embedded
- **PR5 Vector Store**: Documents and chunks will be stored
- **PR6 RAG**: Chunks will be retrieved for chat context
- **PR7 UI**: Upload progress will be displayed

## Performance Considerations

1. **Streaming**: Process large files in chunks to avoid memory issues
2. **Async Processing**: Use async for multiple file uploads
3. **Progress Tracking**: Emit progress events for UI updates
4. **Error Recovery**: Continue processing other files if one fails

## Success Criteria

1. ✅ All supported formats extract successfully
2. ✅ Large files (up to 100MB) process without memory issues
3. ✅ Chunks maintain semantic coherence
4. ✅ Metadata is preserved and accessible
5. ✅ Processing completes within 30 seconds
6. ✅ Error handling is comprehensive
7. ✅ Progress can be tracked in real-time