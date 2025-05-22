# SemanticScout - API Specification

**Version**: 1.0  
**Date**: May 2025  
**Status**: Design Complete

## 📋 Overview

SemanticScout provides both internal API interfaces for component communication and optional external REST API endpoints for programmatic integration. This specification covers all API contracts, data models, and integration patterns.

## 🏗️ Internal API Architecture

### Core Service Interfaces

#### Document Processing Service
```python
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from dataclasses import dataclass

@dataclass
class ProcessingResult:
    success: bool
    document_id: str
    chunks: List[str]
    metadata: Dict[str, Any]
    error_message: Optional[str] = None

class DocumentProcessor(ABC):
    @abstractmethod
    def process_document(self, file_path: str, file_type: str) -> ProcessingResult:
        """Process uploaded document and extract content"""
        pass
    
    @abstractmethod
    def validate_file(self, file_path: str) -> bool:
        """Validate file format and integrity"""
        pass
    
    @abstractmethod
    def extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract document metadata"""
        pass
```

#### Embedding Service
```python
@dataclass
class EmbeddingResult:
    success: bool
    embeddings: List[List[float]]
    model_used: str
    dimensions: int
    error_message: Optional[str] = None

class EmbeddingService(ABC):
    @abstractmethod
    def generate_embeddings(self, texts: List[str]) -> EmbeddingResult:
        """Generate embeddings for text chunks"""
        pass
    
    @abstractmethod
    def generate_query_embedding(self, query: str) -> List[float]:
        """Generate embedding for search query"""
        pass
    
    @abstractmethod
    def get_embedding_dimensions(self) -> int:
        """Get embedding vector dimensions"""
        pass
```

#### Vector Store Service
```python
@dataclass
class SearchResult:
    document_id: str
    chunk_id: str
    content: str
    score: float
    metadata: Dict[str, Any]

@dataclass
class StorageResult:
    success: bool
    stored_count: int
    error_message: Optional[str] = None

class VectorStore(ABC):
    @abstractmethod
    def add_documents(self, documents: List[Dict[str, Any]]) -> StorageResult:
        """Add documents to vector store"""
        pass
    
    @abstractmethod
    def search(self, query_embedding: List[float], 
               limit: int = 10, threshold: float = 0.7) -> List[SearchResult]:
        """Perform similarity search"""
        pass
    
    @abstractmethod
    def delete_document(self, document_id: str) -> bool:
        """Remove document from store"""
        pass
    
    @abstractmethod
    def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        pass
```

#### Search Engine Service
```python
@dataclass
class SearchQuery:
    query: str
    filters: Optional[Dict[str, Any]] = None
    limit: int = 10
    threshold: float = 0.7
    include_metadata: bool = True

@dataclass
class SearchResponse:
    results: List[SearchResult]
    total_count: int
    query_time_ms: float
    embedding_time_ms: float

class SearchEngine(ABC):
    @abstractmethod
    def search(self, query: SearchQuery) -> SearchResponse:
        """Execute semantic search"""
        pass
    
    @abstractmethod
    def suggest_similar_documents(self, document_id: str, 
                                  limit: int = 5) -> List[SearchResult]:
        """Find similar documents"""
        pass
```

## 🌐 External REST API (Future Implementation)

### API Base Configuration
```yaml
openapi: 3.0.3
info:
  title: SemanticScout API
  description: Semantic document search and analysis API
  version: 1.0.0
  contact:
    name: NeurArk
    url: https://neurark.com

servers:
  - url: http://localhost:8000/api/v1
    description: Development server
  - url: https://semanticscout.neurark.com/api/v1
    description: Production server

security:
  - ApiKeyAuth: []
```

### Authentication
```yaml
components:
  securitySchemes:
    ApiKeyAuth:
      type: apiKey
      in: header
      name: X-API-Key
      description: API key for authentication
```

### Document Management Endpoints

#### Upload Document
```yaml
/documents:
  post:
    summary: Upload and process document
    description: Upload a document for semantic indexing
    requestBody:
      required: true
      content:
        multipart/form-data:
          schema:
            type: object
            properties:
              file:
                type: string
                format: binary
                description: Document file (PDF, DOCX, TXT)
              metadata:
                type: object
                description: Additional document metadata
            required:
              - file
    responses:
      201:
        description: Document uploaded and processed successfully
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DocumentResponse'
      400:
        description: Invalid file format or size
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/ErrorResponse'
      413:
        description: File too large
      500:
        description: Processing error
```

#### List Documents
```yaml
/documents:
  get:
    summary: List all documents
    description: Retrieve list of uploaded documents
    parameters:
      - name: limit
        in: query
        schema:
          type: integer
          minimum: 1
          maximum: 100
          default: 20
      - name: offset
        in: query
        schema:
          type: integer
          minimum: 0
          default: 0
      - name: file_type
        in: query
        schema:
          type: string
          enum: [pdf, docx, txt, md]
    responses:
      200:
        description: List of documents
        content:
          application/json:
            schema:
              type: object
              properties:
                documents:
                  type: array
                  items:
                    $ref: '#/components/schemas/DocumentSummary'
                total:
                  type: integer
                limit:
                  type: integer
                offset:
                  type: integer
```

#### Get Document Details
```yaml
/documents/{document_id}:
  get:
    summary: Get document details
    parameters:
      - name: document_id
        in: path
        required: true
        schema:
          type: string
    responses:
      200:
        description: Document details
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/DocumentResponse'
      404:
        description: Document not found
```

#### Delete Document
```yaml
/documents/{document_id}:
  delete:
    summary: Delete document
    parameters:
      - name: document_id
        in: path
        required: true
        schema:
          type: string
    responses:
      204:
        description: Document deleted successfully
      404:
        description: Document not found
```

### Search Endpoints

#### Semantic Search
```yaml
/search:
  post:
    summary: Perform semantic search
    description: Search documents using natural language queries
    requestBody:
      required: true
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/SearchRequest'
    responses:
      200:
        description: Search results
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SearchResponse'
      400:
        description: Invalid search query
```

#### Similar Documents
```yaml
/documents/{document_id}/similar:
  get:
    summary: Find similar documents
    parameters:
      - name: document_id
        in: path
        required: true
        schema:
          type: string
      - name: limit
        in: query
        schema:
          type: integer
          minimum: 1
          maximum: 20
          default: 5
      - name: threshold
        in: query
        schema:
          type: number
          minimum: 0.0
          maximum: 1.0
          default: 0.7
    responses:
      200:
        description: Similar documents
        content:
          application/json:
            schema:
              type: array
              items:
                $ref: '#/components/schemas/SearchResult'
```

### Analytics Endpoints

#### Collection Statistics
```yaml
/analytics/stats:
  get:
    summary: Get collection statistics
    responses:
      200:
        description: Collection statistics
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/CollectionStats'
```

#### Search Analytics
```yaml
/analytics/searches:
  get:
    summary: Get search analytics
    parameters:
      - name: start_date
        in: query
        schema:
          type: string
          format: date
      - name: end_date
        in: query
        schema:
          type: string
          format: date
    responses:
      200:
        description: Search analytics data
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SearchAnalytics'
```

## 📊 Data Models

### Core Schemas
```yaml
components:
  schemas:
    DocumentResponse:
      type: object
      properties:
        id:
          type: string
          description: Unique document identifier
        filename:
          type: string
          description: Original filename
        file_type:
          type: string
          enum: [pdf, docx, txt, md]
        file_size:
          type: integer
          description: File size in bytes
        upload_date:
          type: string
          format: date-time
        processing_status:
          type: string
          enum: [pending, processing, completed, failed]
        chunk_count:
          type: integer
          description: Number of text chunks
        metadata:
          type: object
          description: Document metadata
        content_preview:
          type: string
          description: First 500 characters
      required:
        - id
        - filename
        - file_type
        - upload_date
        - processing_status

    DocumentSummary:
      type: object
      properties:
        id:
          type: string
        filename:
          type: string
        file_type:
          type: string
        file_size:
          type: integer
        upload_date:
          type: string
          format: date-time
        processing_status:
          type: string
        chunk_count:
          type: integer

    SearchRequest:
      type: object
      properties:
        query:
          type: string
          description: Natural language search query
          minLength: 1
          maxLength: 1000
        filters:
          type: object
          properties:
            file_type:
              type: array
              items:
                type: string
                enum: [pdf, docx, txt, md]
            date_range:
              type: object
              properties:
                start:
                  type: string
                  format: date
                end:
                  type: string
                  format: date
            file_size:
              type: object
              properties:
                min:
                  type: integer
                max:
                  type: integer
        limit:
          type: integer
          minimum: 1
          maximum: 50
          default: 10
        threshold:
          type: number
          minimum: 0.0
          maximum: 1.0
          default: 0.7
        include_content:
          type: boolean
          default: true
      required:
        - query

    SearchResponse:
      type: object
      properties:
        results:
          type: array
          items:
            $ref: '#/components/schemas/SearchResult'
        total_count:
          type: integer
          description: Total number of matching documents
        query_time_ms:
          type: number
          description: Query execution time in milliseconds
        embedding_time_ms:
          type: number
          description: Query embedding time in milliseconds
        query:
          type: string
          description: Original search query

    SearchResult:
      type: object
      properties:
        document_id:
          type: string
        chunk_id:
          type: string
        score:
          type: number
          minimum: 0.0
          maximum: 1.0
          description: Similarity score
        content:
          type: string
          description: Matching text content
        highlighted_content:
          type: string
          description: Content with search terms highlighted
        context_before:
          type: string
          description: Text before the match
        context_after:
          type: string
          description: Text after the match
        document_metadata:
          type: object
          properties:
            filename:
              type: string
            file_type:
              type: string
            upload_date:
              type: string
              format: date-time
        chunk_metadata:
          type: object
          properties:
            chunk_index:
              type: integer
            start_char:
              type: integer
            end_char:
              type: integer

    CollectionStats:
      type: object
      properties:
        total_documents:
          type: integer
        total_chunks:
          type: integer
        total_embeddings:
          type: integer
        file_type_distribution:
          type: object
          additionalProperties:
            type: integer
        average_document_size:
          type: number
        storage_size_mb:
          type: number
        last_updated:
          type: string
          format: date-time

    SearchAnalytics:
      type: object
      properties:
        total_searches:
          type: integer
        successful_searches:
          type: integer
        average_response_time_ms:
          type: number
        most_common_queries:
          type: array
          items:
            type: object
            properties:
              query:
                type: string
              count:
                type: integer
        search_volume_by_date:
          type: array
          items:
            type: object
            properties:
              date:
                type: string
                format: date
              count:
                type: integer

    ErrorResponse:
      type: object
      properties:
        error:
          type: string
          description: Error type
        message:
          type: string
          description: Human-readable error message
        details:
          type: object
          description: Additional error details
        timestamp:
          type: string
          format: date-time
        request_id:
          type: string
          description: Unique request identifier
      required:
        - error
        - message
        - timestamp
```

## 🔧 Implementation Examples

### Python Client SDK
```python
from typing import List, Optional, Dict, Any
import requests
from dataclasses import dataclass

@dataclass
class SemanticScoutConfig:
    base_url: str
    api_key: str
    timeout: int = 30

class SemanticScoutClient:
    def __init__(self, config: SemanticScoutConfig):
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            'X-API-Key': config.api_key,
            'Content-Type': 'application/json'
        })
    
    def upload_document(self, file_path: str, 
                       metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Upload and process a document"""
        with open(file_path, 'rb') as f:
            files = {'file': f}
            data = {'metadata': metadata} if metadata else {}
            
            response = self.session.post(
                f"{self.config.base_url}/documents",
                files=files,
                data=data,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            return response.json()
    
    def search(self, query: str, limit: int = 10, 
               threshold: float = 0.7, filters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Perform semantic search"""
        payload = {
            'query': query,
            'limit': limit,
            'threshold': threshold,
            'filters': filters or {}
        }
        
        response = self.session.post(
            f"{self.config.base_url}/search",
            json=payload,
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_document(self, document_id: str) -> Dict[str, Any]:
        """Get document details"""
        response = self.session.get(
            f"{self.config.base_url}/documents/{document_id}",
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def find_similar_documents(self, document_id: str, 
                              limit: int = 5) -> List[Dict[str, Any]]:
        """Find similar documents"""
        response = self.session.get(
            f"{self.config.base_url}/documents/{document_id}/similar",
            params={'limit': limit},
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics"""
        response = self.session.get(
            f"{self.config.base_url}/analytics/stats",
            timeout=self.config.timeout
        )
        response.raise_for_status()
        return response.json()

# Usage example
config = SemanticScoutConfig(
    base_url="http://localhost:8000/api/v1",
    api_key="your-api-key-here"
)
client = SemanticScoutClient(config)

# Upload document
result = client.upload_document("document.pdf")
document_id = result['id']

# Search documents
search_results = client.search(
    query="machine learning algorithms",
    limit=10,
    threshold=0.8
)

# Find similar documents
similar_docs = client.find_similar_documents(document_id)
```

### JavaScript/TypeScript Client
```typescript
interface SemanticScoutConfig {
  baseUrl: string;
  apiKey: string;
  timeout?: number;
}

interface SearchRequest {
  query: string;
  limit?: number;
  threshold?: number;
  filters?: Record<string, any>;
}

class SemanticScoutClient {
  private config: SemanticScoutConfig;
  
  constructor(config: SemanticScoutConfig) {
    this.config = { timeout: 30000, ...config };
  }
  
  private async request(endpoint: string, options: RequestInit = {}): Promise<any> {
    const url = `${this.config.baseUrl}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'X-API-Key': this.config.apiKey,
        'Content-Type': 'application/json',
        ...options.headers
      }
    });
    
    if (!response.ok) {
      throw new Error(`API Error: ${response.status} ${response.statusText}`);
    }
    
    return response.json();
  }
  
  async uploadDocument(file: File, metadata?: Record<string, any>) {
    const formData = new FormData();
    formData.append('file', file);
    if (metadata) {
      formData.append('metadata', JSON.stringify(metadata));
    }
    
    return this.request('/documents', {
      method: 'POST',
      body: formData,
      headers: {} // Remove Content-Type for FormData
    });
  }
  
  async search(request: SearchRequest) {
    return this.request('/search', {
      method: 'POST',
      body: JSON.stringify(request)
    });
  }
  
  async getDocument(documentId: string) {
    return this.request(`/documents/${documentId}`);
  }
  
  async findSimilarDocuments(documentId: string, limit: number = 5) {
    return this.request(`/documents/${documentId}/similar?limit=${limit}`);
  }
  
  async getStats() {
    return this.request('/analytics/stats');
  }
}
```

## 🔒 Security Specifications

### API Authentication
```python
# API Key validation middleware
from functools import wraps
from flask import request, jsonify

def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        if not validate_api_key(api_key):
            return jsonify({'error': 'Invalid API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def validate_api_key(api_key: str) -> bool:
    # Implement API key validation logic
    return api_key in get_valid_api_keys()
```

### Rate Limiting
```python
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# Apply specific limits to endpoints
@app.route('/api/v1/search', methods=['POST'])
@limiter.limit("20 per minute")
@require_api_key
def search():
    # Search implementation
    pass

@app.route('/api/v1/documents', methods=['POST'])
@limiter.limit("10 per minute")
@require_api_key
def upload_document():
    # Upload implementation
    pass
```

### Input Validation
```python
from marshmallow import Schema, fields, validate

class SearchRequestSchema(Schema):
    query = fields.Str(required=True, validate=validate.Length(min=1, max=1000))
    limit = fields.Int(validate=validate.Range(min=1, max=50), missing=10)
    threshold = fields.Float(validate=validate.Range(min=0.0, max=1.0), missing=0.7)
    filters = fields.Dict(missing=dict)

def validate_search_request(data):
    schema = SearchRequestSchema()
    return schema.load(data)
```

## 📈 Performance Specifications

### Response Time Targets
- **Document Upload**: < 30 seconds for files up to 100MB
- **Search Queries**: < 2 seconds for response
- **Document Retrieval**: < 500ms for metadata
- **Statistics**: < 1 second for collection stats

### Throughput Targets
- **Concurrent Searches**: 10 per second per instance
- **Document Processing**: 5 documents per minute per instance
- **API Requests**: 100 requests per minute per API key

### Caching Strategy
```python
from functools import lru_cache
import redis

# Redis cache for frequent queries
redis_client = redis.Redis(host='localhost', port=6379, db=0)

@lru_cache(maxsize=1000)
def get_cached_embedding(text: str) -> List[float]:
    """Cache embeddings to avoid recomputation"""
    cache_key = f"embedding:{hash(text)}"
    cached = redis_client.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    embedding = generate_embedding(text)
    redis_client.setex(cache_key, 3600, json.dumps(embedding))  # 1 hour TTL
    return embedding
```

---

*This API specification provides comprehensive integration capabilities while maintaining security, performance, and usability standards for SemanticScout.*