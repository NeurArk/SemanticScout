# SemanticScout - System Architecture Documentation

**Version**: 1.0  
**Date**: May 2025  
**Status**: Design Complete

## 🏛️ Architecture Overview

SemanticScout follows a modular, layered architecture designed for semantic document search and analysis. The system is built with clear separation of concerns, enabling maintainability, testability, and future extensibility.

## 📐 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Presentation Layer                        │
│                         (Gradio UI)                             │
├─────────────────────────────────────────────────────────────────┤
│                       Application Layer                         │
│              (Business Logic & Orchestration)                   │
├─────────────────────────────────────────────────────────────────┤
│                        Service Layer                            │
│         (Document Processing, Search, Visualization)            │
├─────────────────────────────────────────────────────────────────┤
│                      Infrastructure Layer                       │
│        (Vector DB, File Storage, External APIs)                 │
└─────────────────────────────────────────────────────────────────┘
```

## 🏗️ Detailed Component Architecture

### 1. Presentation Layer

#### Gradio Interface Components
```
app.py (Main Application)
├── DocumentUploadInterface
│   ├── FileUploader (multi-format support)
│   ├── UploadProgress (real-time feedback)
│   └── DocumentPreview (content preview)
├── SearchInterface
│   ├── SearchBox (natural language queries)
│   ├── FilterPanel (type, date, size filters)
│   └── AdvancedOptions (search parameters)
├── ResultsInterface
│   ├── ResultsList (scored results)
│   ├── DocumentViewer (content display)
│   └── SimilarityScore (relevance indicators)
└── VisualizationInterface
    ├── DocumentCloud (2D scatter plot)
    ├── SimilarityNetwork (interactive graph)
    └── ClusteringView (thematic groups)
```

### 2. Application Layer

#### Core Application Services
```
core/
├── application.py (Main App Controller)
│   ├── DocumentManager (upload orchestration)
│   ├── SearchOrchestrator (query processing)
│   ├── VisualizationManager (chart generation)
│   └── ConfigurationManager (settings)
├── exceptions.py (Custom Error Handling)
├── validators.py (Input Validation)
└── utils.py (Common Utilities)
```

### 3. Service Layer

#### Document Processing Service
```
core/document_processor.py
├── DocumentExtractor
│   ├── PDFExtractor (PyMuPDF + Unstructured)
│   ├── WordExtractor (python-docx + Unstructured)
│   ├── TextExtractor (plain text processing)
│   └── MetadataExtractor (file properties)
├── ContentValidator
│   ├── FormatValidator (file type checking)
│   ├── SizeValidator (file size limits)
│   └── ContentValidator (malware/corruption check)
└── DocumentChunker
    ├── SemanticChunker (meaning-based splitting)
    ├── OverlapManager (chunk overlap handling)
    └── MetadataPreserver (chunk metadata)
```

#### Embedding Service
```
core/embedder.py
├── EmbeddingGenerator
│   ├── OpenAIEmbedder (text-embedding-3-large)
│   ├── BatchProcessor (efficient bulk processing)
│   └── CacheManager (embedding cache)
├── EmbeddingValidator
│   ├── DimensionValidator (vector size check)
│   ├── QualityScorer (embedding quality metrics)
│   └── SimilarityCalculator (cosine similarity)
└── EmbeddingOptimizer
    ├── DimensionReducer (optional dimension reduction)
    ├── NormalizationHandler (vector normalization)
    └── CompressionManager (storage optimization)
```

#### Vector Storage Service
```
core/vector_store.py
├── ChromaManager
│   ├── CollectionManager (document collections)
│   ├── IndexManager (vector indexing)
│   └── QueryProcessor (similarity search)
├── MetadataManager
│   ├── DocumentMetadata (file properties)
│   ├── ChunkMetadata (text segments)
│   └── EmbeddingMetadata (vector properties)
└── PersistenceManager
    ├── DatabasePersister (local storage)
    ├── BackupManager (data backup)
    └── RecoveryHandler (error recovery)
```

#### Search Engine Service
```
core/search_engine.py
├── QueryProcessor
│   ├── QueryAnalyzer (intent understanding)
│   ├── QueryEmbedder (query vectorization)
│   └── QueryOptimizer (search optimization)
├── SearchExecutor
│   ├── VectorSearcher (similarity search)
│   ├── MetadataFilter (attribute filtering)
│   └── ResultRanker (relevance scoring)
└── ResultProcessor
    ├── ContextExtractor (relevant passages)
    ├── HighlightGenerator (text highlighting)
    └── ScoreNormalizer (confidence scoring)
```

### 4. Visualization Service

#### Visualization Engine
```
core/visualizer.py
├── DimensionalityReducer
│   ├── UMAPReducer (2D/3D projection)
│   ├── TSNEReducer (alternative reduction)
│   └── PCAReducer (linear reduction)
├── NetworkGenerator
│   ├── SimilarityGraphBuilder (document relationships)
│   ├── ClusterDetector (community detection)
│   └── LayoutOptimizer (graph layout)
└── PlotGenerator
    ├── ScatterPlotBuilder (document clouds)
    ├── NetworkPlotBuilder (interactive graphs)
    └── HeatmapBuilder (similarity matrices)
```

## 🔄 Data Flow Architecture

### Document Ingestion Flow
```
1. File Upload (Gradio) 
   ↓
2. File Validation (DocumentProcessor)
   ↓
3. Content Extraction (Unstructured/PyMuPDF)
   ↓
4. Text Chunking (SemanticChunker)
   ↓
5. Embedding Generation (OpenAI API)
   ↓
6. Vector Storage (ChromaDB)
   ↓
7. Index Update (SearchEngine)
   ↓
8. UI Feedback (Progress Update)
```

### Search Query Flow
```
1. Query Input (Gradio Search Box)
   ↓
2. Query Processing (QueryProcessor)
   ↓
3. Query Embedding (OpenAI API)
   ↓
4. Vector Search (ChromaDB)
   ↓
5. Result Filtering (MetadataFilter)
   ↓
6. Result Ranking (ResultRanker)
   ↓
7. Context Extraction (ContextExtractor)
   ↓
8. UI Display (ResultsInterface)
```

### Visualization Flow
```
1. Data Request (UI Component)
   ↓
2. Vector Retrieval (ChromaDB)
   ↓
3. Dimensionality Reduction (UMAP)
   ↓
4. Clustering Analysis (NetworkX)
   ↓
5. Plot Generation (Plotly)
   ↓
6. Interactive Display (Gradio Plot)
```

## 🗃️ Data Models

### Document Model
```python
@dataclass
class Document:
    id: str
    filename: str
    file_type: str
    file_size: int
    upload_date: datetime
    content: str
    metadata: Dict[str, Any]
    chunks: List[DocumentChunk]
    embedding_id: Optional[str] = None
    
@dataclass
class DocumentChunk:
    id: str
    document_id: str
    content: str
    start_char: int
    end_char: int
    chunk_index: int
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Search Models
```python
@dataclass
class SearchQuery:
    query_text: str
    filters: Dict[str, Any]
    limit: int = 10
    threshold: float = 0.7
    include_metadata: bool = True
    
@dataclass
class SearchResult:
    document_id: str
    chunk_id: str
    score: float
    content: str
    highlighted_content: str
    metadata: Dict[str, Any]
    context_before: str
    context_after: str
```

### Visualization Models
```python
@dataclass
class VisualizationData:
    document_ids: List[str]
    coordinates: List[Tuple[float, float]]
    labels: List[str]
    colors: List[str]
    sizes: List[float]
    metadata: Dict[str, Any]
    
@dataclass
class NetworkData:
    nodes: List[NetworkNode]
    edges: List[NetworkEdge]
    clusters: List[DocumentCluster]
    layout: str = "force"
```

## 🔌 Integration Architecture

### External API Integration
```
OpenAI API Integration
├── Authentication (API Key Management)
├── Rate Limiting (Request Throttling)
├── Error Handling (Retry Logic)
├── Response Caching (Performance Optimization)
└── Cost Monitoring (Usage Tracking)

LangChain Integration
├── Document Loaders (Multi-format Support)
├── Text Splitters (Chunk Management)
├── Vector Stores (ChromaDB Wrapper)
├── Embeddings (OpenAI Integration)
└── Chains (Search Workflows)
```

### Internal Service Communication
```
Service Communication Pattern: Direct Method Calls
├── Synchronous Operations (Gradio UI Interactions)
├── Asynchronous Operations (Background Processing)
├── Error Propagation (Exception Handling)
└── Event Logging (Operation Tracking)

Note: No REST API - All interactions through Gradio interface
```

## 🏗️ File Structure

### Complete Project Structure
```
SemanticScout/
├── app.py                          # Main Gradio application
├── requirements.txt                # Python dependencies
├── .env.example                    # Environment variables template
├── .gitignore                     # Git ignore patterns
├── README.md                      # Project documentation
│
├── core/                          # Core business logic
│   ├── __init__.py
│   ├── application.py             # Main application controller
│   ├── document_processor.py      # Document processing service
│   ├── embedder.py               # Embedding generation service
│   ├── vector_store.py           # Vector database management
│   ├── search_engine.py          # Search functionality
│   ├── visualizer.py             # Visualization generation
│   ├── exceptions.py             # Custom exceptions
│   ├── validators.py             # Input validation
│   ├── models.py                 # Data models
│   └── utils.py                  # Utility functions
│
├── config/                        # Configuration management
│   ├── __init__.py
│   ├── settings.py               # Application settings
│   ├── logging.py                # Logging configuration
│   └── constants.py              # Application constants
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── unit/                     # Unit tests
│   ├── integration/              # Integration tests
│   ├── fixtures/                 # Test fixtures
│   └── conftest.py              # Pytest configuration
│
├── data/                          # Data storage
│   ├── uploads/                  # Uploaded documents
│   ├── chroma_db/               # Vector database
│   ├── cache/                   # Temporary cache
│   └── logs/                    # Application logs
│
├── docs/                          # Documentation
│   ├── PRD.md                    # Product requirements
│   ├── TECHNICAL_STACK.md        # Technology stack
│   ├── ARCHITECTURE.md           # System architecture
│   ├── UI_GUIDELINES.md          # UI/UX guidelines
│   ├── DEPLOYMENT.md             # Deployment guide
│   └── TESTING_STRATEGY.md       # Testing approach
│
├── scripts/                       # Utility scripts
│   ├── setup.py                 # Environment setup
│   ├── migrate_db.py            # Database migration
│   └── health_check.py          # System health check
│
└── assets/                        # Static assets
    ├── images/                   # UI images
    ├── styles/                   # CSS styles
    └── icons/                    # Application icons
```

## ⚡ Performance Architecture

### Caching Strategy
```
Multi-Level Caching
├── Memory Cache (LRU Cache)
│   ├── Embedding Cache (recently computed)
│   ├── Search Results Cache (frequent queries)
│   └── Document Metadata Cache (file properties)
├── Disk Cache (File System)
│   ├── Processed Documents (parsed content)
│   ├── Generated Embeddings (vector cache)
│   └── Visualization Data (plot cache)
└── Database Cache (ChromaDB)
    ├── Vector Index Cache (search optimization)
    ├── Metadata Index Cache (filter optimization)
    └── Query Result Cache (repeated searches)
```

### Async Processing
```
Background Processing
├── Document Upload Processing
│   ├── Async File Processing (non-blocking)
│   ├── Progress Tracking (status updates)
│   └── Error Handling (graceful failures)
├── Embedding Generation
│   ├── Batch Processing (API efficiency)
│   ├── Rate Limiting (API constraints)
│   └── Retry Logic (failure recovery)
└── Search Operations
    ├── Concurrent Searches (parallel processing)
    ├── Result Streaming (progressive loading)
    └── Cache Warming (precomputed results)
```

## 🔒 Security Architecture

### Data Protection
```
Security Layers
├── Input Validation
│   ├── File Type Validation (allowed formats)
│   ├── File Size Validation (size limits)
│   ├── Content Scanning (malware detection)
│   └── Query Sanitization (injection prevention)
├── API Security
│   ├── API Key Management (environment variables)
│   ├── Rate Limiting (abuse prevention)
│   ├── Request Validation (parameter checking)
│   └── Error Masking (information disclosure)
└── Data Storage
    ├── Local Storage Only (no cloud by default)
    ├── Temporary Files (automatic cleanup)
    ├── Encrypted API Keys (secure storage)
    └── Access Control (file permissions)
```

## 📊 Monitoring Architecture

### Observability Stack
```
Monitoring Components
├── Application Metrics
│   ├── Response Time Tracking (operation latency)
│   ├── Error Rate Monitoring (failure tracking)
│   ├── Resource Usage (memory, CPU)
│   └── API Usage (OpenAI call tracking)
├── Business Metrics
│   ├── Document Processing Stats (success rate)
│   ├── Search Query Analytics (usage patterns)
│   ├── User Interaction Metrics (engagement)
│   └── System Performance (throughput)
└── Health Checks
    ├── Database Connectivity (ChromaDB status)
    ├── API Availability (OpenAI API status)
    ├── Storage Health (disk space)
    └── Memory Usage (resource monitoring)
```

## 🔄 Deployment Architecture

### Environment Configuration
```
Deployment Environments
├── Development
│   ├── Local ChromaDB (file-based)
│   ├── Local File Storage (./data/)
│   ├── Debug Logging (verbose output)
│   └── Development API Keys (lower limits)
├── Demo/Staging
│   ├── Temporary Storage (session-based)
│   ├── Limited Resources (cost optimization)
│   ├── Basic Monitoring (essential metrics)
│   └── Shared API Keys (demo accounts)
└── Production (Future)
    ├── Cloud Vector DB (scalable storage)
    ├── Object Storage (S3/GCS)
    ├── Comprehensive Monitoring (full observability)
    └── Production API Keys (high limits)
```

## 🧪 Testing Architecture

### Test Strategy
```
Testing Layers
├── Unit Tests
│   ├── Service Layer Tests (business logic)
│   ├── Model Tests (data validation)
│   ├── Utility Tests (helper functions)
│   └── Integration Tests (component interaction)
├── Integration Tests
│   ├── API Integration (OpenAI calls)
│   ├── Database Integration (ChromaDB operations)
│   ├── File Processing (document handling)
│   └── Search Workflow (end-to-end search)
└── UI Tests
    ├── Gradio Interface Tests (component testing)
    ├── User Workflow Tests (interaction testing)
    ├── Visualization Tests (chart generation)
    └── Error Handling Tests (failure scenarios)
```

---

*This architecture is designed for scalability, maintainability, and professional demonstration while keeping complexity manageable for a demo application.*