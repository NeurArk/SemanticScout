# PR5: Vector Database Integration - Detailed Implementation Guide

## Overview
This PR implements ChromaDB integration for storing document embeddings and enabling similarity search.

## Prerequisites
- PR2-4 completed (models, processing, embeddings)
- ChromaDB installed

## File Structure
```
core/
├── vector_store.py          # Main vector store service
├── vectordb/
│   ├── __init__.py
│   ├── chroma_manager.py    # ChromaDB client management
│   ├── collection_manager.py # Collection operations
│   └── query_builder.py     # Search query construction
└── utils/
    └── vector_utils.py      # Vector operations utilities
```

## Detailed Implementation

### 1. ChromaDB Manager (`core/vectordb/chroma_manager.py`)

```python
import chromadb
from chromadb.config import Settings
from typing import Optional, Dict, Any
import logging
from pathlib import Path
from core.exceptions.custom_exceptions import VectorStoreError

logger = logging.getLogger(__name__)

class ChromaManager:
    """Manages ChromaDB client and connections."""
    
    def __init__(self, persist_directory: str = "./data/chroma_db"):
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        
        try:
            self.client = chromadb.PersistentClient(
                path=str(self.persist_directory),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            logger.info(f"ChromaDB initialized at {self.persist_directory}")
        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise VectorStoreError(f"ChromaDB initialization failed: {str(e)}")
    
    def get_or_create_collection(self, name: str, 
                               metadata: Optional[Dict[str, Any]] = None) -> chromadb.Collection:
        """Get or create a collection."""
        try:
            # Try to get existing collection
            collection = self.client.get_collection(name=name)
            logger.info(f"Retrieved existing collection: {name}")
            return collection
        except:
            # Create new collection
            collection = self.client.create_collection(
                name=name,
                metadata=metadata or {"description": "Document embeddings"}
            )
            logger.info(f"Created new collection: {name}")
            return collection
    
    def delete_collection(self, name: str) -> None:
        """Delete a collection."""
        try:
            self.client.delete_collection(name=name)
            logger.info(f"Deleted collection: {name}")
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")
            raise VectorStoreError(f"Collection deletion failed: {str(e)}")
    
    def list_collections(self) -> List[str]:
        """List all collections."""
        return [col.name for col in self.client.list_collections()]
    
    def reset_database(self) -> None:
        """Reset entire database (use with caution)."""
        try:
            self.client.reset()
            logger.warning("ChromaDB has been reset")
        except Exception as e:
            logger.error(f"Failed to reset ChromaDB: {e}")
            raise VectorStoreError(f"Database reset failed: {str(e)}")
```

### 2. Collection Manager (`core/vectordb/collection_manager.py`)

```python
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from core.models.document import Document, DocumentChunk
from core.exceptions.custom_exceptions import VectorStoreError
import logging

logger = logging.getLogger(__name__)

class CollectionManager:
    """Manages operations on ChromaDB collections."""
    
    def __init__(self, collection: chromadb.Collection):
        self.collection = collection
    
    def add_documents(self, document: Document, chunks: List[DocumentChunk]) -> None:
        """Add document chunks to collection."""
        if not chunks:
            return
        
        # Prepare data for ChromaDB
        ids = []
        embeddings = []
        documents = []
        metadatas = []
        
        for chunk in chunks:
            if chunk.embedding is None:
                logger.warning(f"Skipping chunk {chunk.id} - no embedding")
                continue
            
            ids.append(chunk.id)
            embeddings.append(chunk.embedding)
            documents.append(chunk.content)
            
            # Prepare metadata
            metadata = {
                "document_id": document.id,
                "filename": document.filename,
                "file_type": document.file_type,
                "chunk_index": chunk.chunk_index,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                **chunk.metadata
            }
            metadatas.append(metadata)
        
        if not ids:
            logger.warning(f"No chunks with embeddings for document {document.id}")
            return
        
        try:
            # Add to ChromaDB
            self.collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas
            )
            logger.info(f"Added {len(ids)} chunks from document {document.id}")
            
        except Exception as e:
            logger.error(f"Failed to add documents: {e}")
            raise VectorStoreError(f"Failed to store document chunks: {str(e)}")
    
    def delete_document(self, document_id: str) -> int:
        """Delete all chunks for a document."""
        try:
            # Query chunks for this document
            results = self.collection.get(
                where={"document_id": document_id}
            )
            
            if results['ids']:
                self.collection.delete(ids=results['ids'])
                logger.info(f"Deleted {len(results['ids'])} chunks for document {document_id}")
                return len(results['ids'])
            return 0
            
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            raise VectorStoreError(f"Failed to delete document: {str(e)}")
    
    def get_document_chunks(self, document_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document."""
        try:
            results = self.collection.get(
                where={"document_id": document_id},
                include=["documents", "metadatas", "embeddings"]
            )
            
            chunks = []
            for i in range(len(results['ids'])):
                chunks.append({
                    'id': results['ids'][i],
                    'content': results['documents'][i],
                    'metadata': results['metadatas'][i],
                    'embedding': results['embeddings'][i] if results['embeddings'] else None
                })
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to get document chunks: {e}")
            raise VectorStoreError(f"Failed to retrieve chunks: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            # Get total count
            count = self.collection.count()
            
            # Get unique documents
            all_metadata = self.collection.get(include=["metadatas"])['metadatas']
            unique_docs = set(m.get('document_id') for m in all_metadata if m)
            
            return {
                'total_chunks': count,
                'total_documents': len(unique_docs),
                'collection_name': self.collection.name
            }
            
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {'error': str(e)}
```

### 3. Query Builder (`core/vectordb/query_builder.py`)

```python
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from core.models.search import SearchQuery, SearchResult
import logging

logger = logging.getLogger(__name__)

class QueryBuilder:
    """Builds and executes vector similarity queries."""
    
    def __init__(self, collection):
        self.collection = collection
    
    def search(self, query_embedding: List[float], 
              search_query: SearchQuery) -> List[SearchResult]:
        """Execute similarity search."""
        
        # Build where clause for filters
        where_clause = self._build_where_clause(search_query)
        
        try:
            # Execute query
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=search_query.max_results,
                where=where_clause if where_clause else None,
                include=["documents", "metadatas", "distances"]
            )
            
            # Convert to SearchResult objects
            search_results = []
            
            if results['ids'] and results['ids'][0]:
                for i, chunk_id in enumerate(results['ids'][0]):
                    # Calculate similarity score (1 - distance for cosine)
                    distance = results['distances'][0][i]
                    score = 1 - distance  # Convert distance to similarity
                    
                    # Skip if below threshold
                    if score < search_query.similarity_threshold:
                        continue
                    
                    result = SearchResult(
                        chunk_id=chunk_id,
                        document_id=results['metadatas'][0][i]['document_id'],
                        score=score,
                        content=results['documents'][0][i],
                        metadata=results['metadatas'][0][i]
                    )
                    search_results.append(result)
            
            logger.info(f"Found {len(search_results)} results above threshold")
            return search_results
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Vector search failed: {str(e)}")
    
    def _build_where_clause(self, search_query: SearchQuery) -> Optional[Dict[str, Any]]:
        """Build ChromaDB where clause from search filters."""
        conditions = []
        
        # File type filter
        if search_query.filter_file_types:
            conditions.append({
                "file_type": {"$in": search_query.filter_file_types}
            })
        
        # Date range filter (if metadata includes dates)
        if search_query.filter_date_range:
            start_date, end_date = search_query.filter_date_range
            # This assumes metadata includes upload_date
            # ChromaDB doesn't support date comparisons directly
            # Would need to store as timestamp
        
        if not conditions:
            return None
        
        # Combine conditions with AND
        if len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}
```

### 4. Main Vector Store Service (`core/vector_store.py`)

```python
from typing import List, Dict, Any, Optional
import logging
from core.models.document import Document, DocumentChunk
from core.models.search import SearchQuery, SearchResult, SearchResponse
from core.exceptions.custom_exceptions import VectorStoreError
from .vectordb.chroma_manager import ChromaManager
from .vectordb.collection_manager import CollectionManager
from .vectordb.query_builder import QueryBuilder
from config.settings import settings
import time

logger = logging.getLogger(__name__)

class VectorStore:
    """Main vector store service for document storage and retrieval."""
    
    def __init__(self):
        self.chroma_manager = ChromaManager(
            persist_directory=settings.chroma_persist_dir
        )
        self.collection = self.chroma_manager.get_or_create_collection(
            name="semantic_scout_docs",
            metadata={
                "description": "Document embeddings for semantic search",
                "embedding_model": settings.embedding_model,
                "embedding_dimension": settings.embedding_dimension
            }
        )
        self.collection_manager = CollectionManager(self.collection)
        self.query_builder = QueryBuilder(self.collection)
    
    def store_document(self, document: Document, chunks: List[DocumentChunk]) -> None:
        """Store document and its chunks in vector database."""
        logger.info(f"Storing document {document.id} with {len(chunks)} chunks")
        
        # Remove existing chunks if any
        deleted = self.collection_manager.delete_document(document.id)
        if deleted > 0:
            logger.info(f"Removed {deleted} existing chunks for document {document.id}")
        
        # Add new chunks
        self.collection_manager.add_documents(document, chunks)
    
    def search(self, query_embedding: List[float], 
              search_query: SearchQuery) -> SearchResponse:
        """Search for similar documents."""
        start_time = time.time()
        
        # Execute search
        results = self.query_builder.search(query_embedding, search_query)
        
        # Create response
        search_time_ms = (time.time() - start_time) * 1000
        
        response = SearchResponse(
            query=search_query,
            results=results,
            total_results=len(results),
            search_time_ms=search_time_ms
        )
        
        logger.info(f"Search completed in {search_time_ms:.2f}ms, "
                   f"found {len(results)} results")
        
        return response
    
    def get_chunks_by_ids(self, chunk_ids: List[str]) -> List[DocumentChunk]:
        """Retrieve specific chunks by IDs (for RAG context)."""
        try:
            results = self.collection.get(
                ids=chunk_ids,
                include=["documents", "metadatas", "embeddings"]
            )
            
            chunks = []
            for i, chunk_id in enumerate(results['ids']):
                chunk = DocumentChunk(
                    id=chunk_id,
                    document_id=results['metadatas'][i]['document_id'],
                    content=results['documents'][i],
                    chunk_index=results['metadatas'][i]['chunk_index'],
                    start_char=results['metadatas'][i]['start_char'],
                    end_char=results['metadatas'][i]['end_char'],
                    embedding=results['embeddings'][i] if results['embeddings'] else None,
                    metadata=results['metadatas'][i]
                )
                chunks.append(chunk)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Failed to retrieve chunks: {e}")
            raise VectorStoreError(f"Chunk retrieval failed: {str(e)}")
    
    def delete_document(self, document_id: str) -> bool:
        """Delete a document and all its chunks."""
        try:
            deleted = self.collection_manager.delete_document(document_id)
            return deleted > 0
        except Exception as e:
            logger.error(f"Failed to delete document: {e}")
            return False
    
    def get_all_documents(self) -> List[Dict[str, Any]]:
        """Get summary of all stored documents."""
        try:
            # Get all unique document IDs
            all_metadata = self.collection.get(include=["metadatas"])['metadatas']
            
            # Group by document
            documents = {}
            for metadata in all_metadata:
                doc_id = metadata.get('document_id')
                if doc_id and doc_id not in documents:
                    documents[doc_id] = {
                        'document_id': doc_id,
                        'filename': metadata.get('filename', 'Unknown'),
                        'file_type': metadata.get('file_type', 'Unknown'),
                        'chunk_count': 0
                    }
                if doc_id:
                    documents[doc_id]['chunk_count'] += 1
            
            return list(documents.values())
            
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = self.collection_manager.get_stats()
        stats['persist_directory'] = str(self.chroma_manager.persist_directory)
        return stats
```

## Testing

```python
# tests/unit/test_vector_store.py
import pytest
from unittest.mock import Mock, patch
from core.vector_store import VectorStore
from core.models.document import Document, DocumentChunk
from core.models.search import SearchQuery

@pytest.fixture
def sample_document():
    return Document(
        id="doc_123",
        filename="test.pdf",
        file_type="pdf",
        file_size=1000,
        content="Test content"
    )

@pytest.fixture
def sample_chunks():
    return [
        DocumentChunk(
            id=f"chunk_{i}",
            document_id="doc_123",
            content=f"Test chunk {i}",
            chunk_index=i,
            start_char=i*100,
            end_char=(i+1)*100,
            embedding=[0.1] * 3072
        )
        for i in range(3)
    ]

def test_store_document(sample_document, sample_chunks):
    store = VectorStore()
    
    # Store document
    store.store_document(sample_document, sample_chunks)
    
    # Verify stored
    docs = store.get_all_documents()
    assert len(docs) == 1
    assert docs[0]['document_id'] == "doc_123"
    assert docs[0]['chunk_count'] == 3

def test_search_documents():
    store = VectorStore()
    
    # Create search query
    query = SearchQuery(
        query_text="test query",
        max_results=5,
        similarity_threshold=0.7
    )
    
    # Mock embedding
    query_embedding = [0.1] * 3072
    
    # Search (will be empty initially)
    response = store.search(query_embedding, query)
    assert response.total_results == 0
    assert response.search_time_ms > 0
```

## Success Criteria

1. ✅ Documents and chunks stored persistently
2. ✅ Similarity search returns relevant results
3. ✅ Search completes in < 2 seconds
4. ✅ Supports 1000+ documents efficiently
5. ✅ Metadata filtering works correctly
6. ✅ Document deletion removes all chunks
7. ✅ Database persists across restarts