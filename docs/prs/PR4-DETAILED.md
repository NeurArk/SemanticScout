# PR4: Embedding Generation Service - Detailed Implementation Guide

## Overview
This PR implements the embedding generation service using OpenAI's text-embedding-3-large model, with caching and batch processing for efficiency.

## Prerequisites
- PR2 completed (models defined)
- PR3 completed (document chunks available)
- OpenAI API key configured

## File Structure
```
core/
├── embedder.py              # Main embedding service
├── embedding/
│   ├── __init__.py
│   ├── openai_embedder.py   # OpenAI API integration
│   ├── batch_processor.py   # Batch processing logic
│   └── embedding_cache.py   # Cache implementation
└── utils/
    ├── token_counter.py     # Token counting utilities
    └── rate_limiter.py      # Rate limiting logic
```

## Detailed Implementation

### 1. OpenAI Embedder (`core/embedding/openai_embedder.py`)

```python
from typing import List, Optional, Dict, Any
import openai
from tenacity import retry, stop_after_attempt, wait_exponential
import logging
from core.exceptions.custom_exceptions import EmbeddingError, RateLimitError
from config.settings import settings

logger = logging.getLogger(__name__)

class OpenAIEmbedder:
    """OpenAI embedding generation with retry logic."""
    
    def __init__(self):
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.embedding_model  # text-embedding-3-large
        self.dimension = settings.embedding_dimension  # 3072
        
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True
    )
    def generate_embedding(self, text: str) -> List[float]:
        """Generate embedding for single text."""
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=text,
                dimensions=self.dimension
            )
            return response.data[0].embedding
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit: {e}")
            raise RateLimitError(retry_after=60)
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise EmbeddingError(f"Failed to generate embedding: {str(e)}")
    
    def generate_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for multiple texts."""
        if not texts:
            return []
        
        try:
            response = self.client.embeddings.create(
                model=self.model,
                input=texts,
                dimensions=self.dimension
            )
            return [item.embedding for item in response.data]
            
        except openai.RateLimitError as e:
            logger.warning(f"Rate limit hit in batch: {e}")
            # Fall back to individual processing
            embeddings = []
            for text in texts:
                emb = self.generate_embedding(text)
                embeddings.append(emb)
            return embeddings
```

### 2. Embedding Cache (`core/embedding/embedding_cache.py`)

```python
from typing import Optional, List, Dict
import hashlib
import time
from functools import lru_cache
import pickle
import os
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class EmbeddingCache:
    """LRU cache for embeddings with optional disk persistence."""
    
    def __init__(self, cache_dir: Optional[str] = None, max_size: int = 1000):
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_size = max_size
        self._memory_cache: Dict[str, Dict] = {}
        self._access_times: Dict[str, float] = {}
        
        # Create cache directory if specified
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            self._load_disk_cache()
    
    def _generate_key(self, text: str, model: str) -> str:
        """Generate cache key from text and model."""
        content = f"{model}:{text}"
        return hashlib.sha256(content.encode()).hexdigest()
    
    def get(self, text: str, model: str) -> Optional[List[float]]:
        """Retrieve embedding from cache."""
        key = self._generate_key(text, model)
        
        # Check memory cache
        if key in self._memory_cache:
            self._access_times[key] = time.time()
            logger.debug(f"Cache hit (memory): {key[:8]}...")
            return self._memory_cache[key]['embedding']
        
        # Check disk cache
        if self.cache_dir:
            disk_path = self.cache_dir / f"{key}.pkl"
            if disk_path.exists():
                try:
                    with open(disk_path, 'rb') as f:
                        data = pickle.load(f)
                    
                    # Add to memory cache
                    self._add_to_memory(key, data)
                    logger.debug(f"Cache hit (disk): {key[:8]}...")
                    return data['embedding']
                except Exception as e:
                    logger.warning(f"Failed to load from disk cache: {e}")
        
        logger.debug(f"Cache miss: {key[:8]}...")
        return None
    
    def set(self, text: str, model: str, embedding: List[float]) -> None:
        """Store embedding in cache."""
        key = self._generate_key(text, model)
        data = {
            'text': text[:100],  # Store preview only
            'model': model,
            'embedding': embedding,
            'timestamp': time.time()
        }
        
        # Add to memory cache
        self._add_to_memory(key, data)
        
        # Save to disk if enabled
        if self.cache_dir:
            try:
                disk_path = self.cache_dir / f"{key}.pkl"
                with open(disk_path, 'wb') as f:
                    pickle.dump(data, f)
            except Exception as e:
                logger.warning(f"Failed to save to disk cache: {e}")
    
    def _add_to_memory(self, key: str, data: Dict) -> None:
        """Add to memory cache with LRU eviction."""
        # Evict oldest if at capacity
        if len(self._memory_cache) >= self.max_size:
            oldest_key = min(self._access_times, key=self._access_times.get)
            del self._memory_cache[oldest_key]
            del self._access_times[oldest_key]
        
        self._memory_cache[key] = data
        self._access_times[key] = time.time()
    
    def _load_disk_cache(self) -> None:
        """Load recent items from disk cache."""
        if not self.cache_dir:
            return
        
        cache_files = list(self.cache_dir.glob("*.pkl"))
        cache_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        # Load most recent items up to max_size
        for cache_file in cache_files[:self.max_size]:
            try:
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                key = cache_file.stem
                self._memory_cache[key] = data
                self._access_times[key] = cache_file.stat().st_mtime
            except Exception as e:
                logger.warning(f"Failed to load cache file {cache_file}: {e}")
        
        logger.info(f"Loaded {len(self._memory_cache)} items from disk cache")
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics."""
        return {
            'memory_items': len(self._memory_cache),
            'disk_items': len(list(self.cache_dir.glob("*.pkl"))) if self.cache_dir else 0,
            'max_size': self.max_size
        }
```

### 3. Batch Processor (`core/embedding/batch_processor.py`)

```python
from typing import List, Tuple, Dict
import logging
from core.models.document import DocumentChunk
from .openai_embedder import OpenAIEmbedder
from .embedding_cache import EmbeddingCache

logger = logging.getLogger(__name__)

class BatchProcessor:
    """Efficient batch processing for embeddings."""
    
    def __init__(self, embedder: OpenAIEmbedder, cache: EmbeddingCache, 
                 batch_size: int = 100):
        self.embedder = embedder
        self.cache = cache
        self.batch_size = batch_size
    
    def process_chunks(self, chunks: List[DocumentChunk], 
                      model: str) -> List[DocumentChunk]:
        """Process chunks in batches, using cache when possible."""
        
        # Separate cached and uncached chunks
        cached_chunks = []
        uncached_chunks = []
        
        for chunk in chunks:
            embedding = self.cache.get(chunk.content, model)
            if embedding is not None:
                chunk.embedding = embedding
                cached_chunks.append(chunk)
            else:
                uncached_chunks.append(chunk)
        
        logger.info(f"Cache hits: {len(cached_chunks)}, misses: {len(uncached_chunks)}")
        
        # Process uncached chunks in batches
        for i in range(0, len(uncached_chunks), self.batch_size):
            batch = uncached_chunks[i:i + self.batch_size]
            texts = [chunk.content for chunk in batch]
            
            try:
                embeddings = self.embedder.generate_embeddings_batch(texts)
                
                # Assign embeddings and update cache
                for chunk, embedding in zip(batch, embeddings):
                    chunk.embedding = embedding
                    self.cache.set(chunk.content, model, embedding)
                
                logger.info(f"Processed batch {i//self.batch_size + 1} "
                          f"({len(batch)} chunks)")
                
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Fall back to individual processing
                for chunk in batch:
                    try:
                        embedding = self.embedder.generate_embedding(chunk.content)
                        chunk.embedding = embedding
                        self.cache.set(chunk.content, model, embedding)
                    except Exception as e2:
                        logger.error(f"Failed to embed chunk {chunk.id}: {e2}")
                        # Continue with other chunks
        
        return chunks
```

### 4. Main Embedding Service (`core/embedder.py`)

```python
from typing import List, Optional, Dict, Any
import logging
from pathlib import Path
from core.models.document import Document, DocumentChunk
from core.exceptions.custom_exceptions import EmbeddingError
from .embedding.openai_embedder import OpenAIEmbedder
from .embedding.embedding_cache import EmbeddingCache
from .embedding.batch_processor import BatchProcessor
from config.settings import settings

logger = logging.getLogger(__name__)

class EmbeddingService:
    """Main service for generating and managing embeddings."""
    
    def __init__(self):
        self.embedder = OpenAIEmbedder()
        self.cache = EmbeddingCache(
            cache_dir=settings.cache_dir,
            max_size=settings.cache_max_size
        )
        self.batch_processor = BatchProcessor(
            self.embedder, 
            self.cache,
            batch_size=settings.embedding_batch_size
        )
        self.model = settings.embedding_model
    
    def embed_document(self, document: Document, 
                      chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Generate embeddings for all chunks of a document."""
        logger.info(f"Embedding document {document.id} with {len(chunks)} chunks")
        
        if not chunks:
            return []
        
        try:
            # Process chunks in batches
            embedded_chunks = self.batch_processor.process_chunks(chunks, self.model)
            
            # Verify all chunks have embeddings
            missing = [c.id for c in embedded_chunks if c.embedding is None]
            if missing:
                logger.warning(f"Missing embeddings for chunks: {missing}")
            
            success_count = sum(1 for c in embedded_chunks if c.embedding is not None)
            logger.info(f"Successfully embedded {success_count}/{len(chunks)} chunks")
            
            return embedded_chunks
            
        except Exception as e:
            logger.error(f"Document embedding failed: {e}")
            raise EmbeddingError(f"Failed to embed document {document.id}: {str(e)}")
    
    def embed_query(self, query: str) -> List[float]:
        """Generate embedding for a search query."""
        # Check cache first
        cached = self.cache.get(query, self.model)
        if cached is not None:
            return cached
        
        try:
            embedding = self.embedder.generate_embedding(query)
            self.cache.set(query, self.model, embedding)
            return embedding
        except Exception as e:
            logger.error(f"Query embedding failed: {e}")
            raise EmbeddingError(f"Failed to embed query: {str(e)}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        return self.cache.get_stats()
    
    def estimate_cost(self, text_length: int) -> float:
        """Estimate embedding cost based on text length."""
        # Rough estimation: ~4 chars per token
        tokens = text_length / 4
        # Cost per 1M tokens (check current OpenAI pricing)
        cost_per_million = 0.13  # USD for text-embedding-3-large
        return (tokens / 1_000_000) * cost_per_million
```

### 5. Token Counter (`core/utils/token_counter.py`)

```python
import tiktoken
from typing import List

class TokenCounter:
    """Count tokens for cost estimation."""
    
    def __init__(self, model: str = "text-embedding-3-large"):
        # Use cl100k_base encoding for embeddings
        self.encoding = tiktoken.get_encoding("cl100k_base")
    
    def count_tokens(self, text: str) -> int:
        """Count tokens in text."""
        return len(self.encoding.encode(text))
    
    def count_tokens_batch(self, texts: List[str]) -> int:
        """Count total tokens in multiple texts."""
        return sum(self.count_tokens(text) for text in texts)
```

## Testing Requirements

```python
# tests/unit/test_embedder.py
import pytest
from unittest.mock import Mock, patch
from core.embedder import EmbeddingService
from core.models.document import Document, DocumentChunk

@pytest.fixture
def mock_openai():
    with patch('openai.OpenAI') as mock:
        mock_instance = Mock()
        mock.return_value = mock_instance
        
        # Mock embedding response
        mock_response = Mock()
        mock_response.data = [Mock(embedding=[0.1] * 3072)]
        mock_instance.embeddings.create.return_value = mock_response
        
        yield mock_instance

def test_embed_query_with_cache(mock_openai):
    service = EmbeddingService()
    
    # First call - should hit API
    embedding1 = service.embed_query("test query")
    assert mock_openai.embeddings.create.call_count == 1
    
    # Second call - should hit cache
    embedding2 = service.embed_query("test query")
    assert mock_openai.embeddings.create.call_count == 1  # No additional call
    assert embedding1 == embedding2

def test_batch_processing(mock_openai):
    service = EmbeddingService()
    
    # Create test chunks
    chunks = [
        DocumentChunk(
            id=f"chunk_{i}",
            document_id="doc_1",
            content=f"Test content {i}",
            chunk_index=i,
            start_char=i*100,
            end_char=(i+1)*100
        )
        for i in range(150)  # More than batch size
    ]
    
    # Mock batch response
    mock_openai.embeddings.create.return_value.data = [
        Mock(embedding=[0.1] * 3072) for _ in range(100)
    ]
    
    embedded = service.embed_document(
        Document(id="doc_1", filename="test.txt", file_type="txt", 
                file_size=1000, content="test"),
        chunks
    )
    
    # Should call API twice (100 + 50)
    assert mock_openai.embeddings.create.call_count == 2
    assert all(chunk.embedding is not None for chunk in embedded)

def test_rate_limit_handling(mock_openai):
    import openai
    service = EmbeddingService()
    
    # Simulate rate limit error
    mock_openai.embeddings.create.side_effect = openai.RateLimitError("Rate limit")
    
    with pytest.raises(Exception):  # Should retry and eventually fail
        service.embed_query("test")
```

## Configuration

Add to `config/settings.py`:

```python
# Embedding settings
embedding_model: str = "text-embedding-3-large"
embedding_dimension: int = 3072
embedding_batch_size: int = 100

# Cache settings
cache_dir: Optional[str] = "data/embedding_cache"
cache_max_size: int = 1000
cache_ttl: int = 3600  # 1 hour

# Rate limiting
rate_limit_delay: int = 4  # seconds
max_retries: int = 3
```

## Success Criteria

1. ✅ Embeddings generated successfully for all chunk types
2. ✅ Cache reduces API calls by 80%+ on repeated content
3. ✅ Batch processing optimizes API usage
4. ✅ Rate limiting handled gracefully
5. ✅ Cost tracking implemented
6. ✅ All chunks get embeddings (with retry)
7. ✅ Performance: < 5s for 100 chunks (with cache)