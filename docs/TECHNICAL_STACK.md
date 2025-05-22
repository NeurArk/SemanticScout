# SemanticScout - Technical Stack Documentation

**Version**: 1.0  
**Date**: May 2025  
**Status**: Production Ready

## 🏗️ Technology Stack Overview

SemanticScout leverages the latest 2025 AI/ML technologies to deliver a professional-grade semantic search experience. Our stack prioritizes stability, performance, and developer experience while maintaining cost-effectiveness for demo purposes.

## 🧠 AI/ML Framework Layer

### Primary Framework: LangChain + LangGraph

- **LangChain**: `^0.1.0` - Core framework for LLM applications
  - Document loading and processing pipelines
  - Vector store abstractions and integrations
  - Chain composition for complex workflows
  - Native OpenAI integration
- **LangGraph**: Latest stable - Stateful, multi-actor applications
  - Cyclical graph workflows
  - Advanced memory management
  - Agent orchestration
  - Semantic search in BaseStore

**Justification**: Most mature and stable ecosystem in 2025 with extensive community support and production-ready features.

### Language Model: OpenAI GPT-4.1

- **Model**: `gpt-4.1` (latest available)
- **Usage**: Query understanding and result processing
- **Rate Limits**: 10,000 RPM for tier 1 accounts
- **Cost**: ~$0.01 per 1K tokens (input), ~$0.03 per 1K tokens (output)

## 🔢 Embeddings Layer

### OpenAI Text Embeddings

- **Model**: `text-embedding-3-large`
- **Dimensions**: 3072 (configurable down to 256)
- **Performance**: 54.9% MIRACL score vs 31.4% previous generation
- **Cost**: $0.00013 per 1K tokens
- **Context Length**: 8191 tokens

**Key Features**:

- Superior multilingual support
- Dimensional reduction capability
- Best-in-class semantic understanding
- Production-ready reliability

## 🗄️ Vector Database Layer

### ChromaDB

- **Version**: `^0.5.10`
- **Type**: Open-source vector database
- **Storage**: Local filesystem with optional cloud deployment
- **Features**:
  - Native Python integration
  - Automatic persistence
  - Metadata filtering
  - Distance metrics (cosine, euclidean, manhattan)
  - Collections management

**Justification**: Perfect balance of simplicity and functionality for demo applications with professional capabilities.

## 📄 Document Processing Layer

### Primary: Unstructured

- **Package**: `unstructured[all-docs]`
- **Capabilities**:
  - Multi-format support (PDF, DOCX, TXT, HTML, etc.)
  - OCR integration
  - Layout analysis
  - Table extraction
  - Hierarchical document structure

### Secondary: PyMuPDF

- **Package**: `PyMuPDF ^1.25.0`
- **Capabilities**:
  - High-performance PDF processing
  - Text extraction with positioning
  - Metadata extraction
  - Image extraction
  - PyMuPDF4LLM for RAG optimization

### Tertiary: Docling (IBM)

- **Usage**: Advanced document structure analysis
- **Features**:
  - DocLayNet AI model for layout analysis
  - TableFormer for table structure recognition
  - DOCX, PPTX, HTML support
  - Export to Markdown and structured formats

## 🎨 User Interface Layer

### Gradio

- **Version**: Latest stable
- **Type**: Python-native UI framework
- **Theme**: `gr.themes.Soft()` with custom corporate styling
- **Components**:
  - `gr.File()` for multi-format uploads
  - `gr.Textbox()` for search queries
  - `gr.Gallery()` for document previews
  - `gr.Plot()` for interactive visualizations
  - `gr.Dataframe()` for results display

**Justification**: Optimal for ML demo applications with minimal setup overhead and professional appearance.

## 📊 Visualization Layer

### Plotly

- **Package**: `plotly`
- **Usage**: Interactive visualizations and dashboards
- **Features**:
  - Real-time updates
  - 3D plotting capabilities
  - Export functionality (PNG, SVG, PDF)
  - Mobile-responsive charts

### NetworkX

- **Package**: `networkx ^3.4.2`
- **Usage**: Graph analysis and network visualization
- **Features**:
  - Graph algorithms
  - Layout algorithms
  - Community detection
  - Centrality measures

### Dimensionality Reduction

- **UMAP**: `umap-learn ^0.5.7` - Superior to t-SNE for 2025
- **Scikit-learn**: `^1.6.1` - Additional ML utilities

## 🛠️ Development Tools

### Core Python

- **Version**: Python 3.11+ (recommended 3.12)
- **Package Manager**: pip with requirements.txt
- **Virtual Environment**: venv or conda

### Development Dependencies

```
# Core ML/AI
langchain
langchain-community
langchain-openai
langgraph
openai

# Vector Database
chromadb

# Document Processing
unstructured[all-docs]
PyMuPDF
python-docx

# UI Framework
gradio

# Visualization
plotly
networkx
umap-learn
scikit-learn

# Utilities
python-dotenv
pydantic
fastapi  # For future API endpoints
uvicorn   # ASGI server
```

## 🔧 Configuration Management

### Environment Variables

```bash
# OpenAI Configuration
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4.1
OPENAI_EMBEDDING_MODEL=text-embedding-3-large

# Application Configuration
APP_NAME=SemanticScout
APP_VERSION=1.0.0
DEBUG=false
LOG_LEVEL=INFO

# Storage Configuration
CHROMA_PERSIST_DIR=./data/chroma_db
UPLOAD_DIR=./data/uploads
MAX_FILE_SIZE=100MB
SUPPORTED_FORMATS=pdf,docx,txt,md

# UI Configuration
GRADIO_THEME=soft
GRADIO_SHARE=false
GRADIO_PORT=7860
```

### Pydantic Settings

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    openai_model: str = "gpt-4.1"
    embedding_model: str = "text-embedding-3-large"
    embedding_dimensions: int = 3072
    chroma_persist_dir: str = "./data/chroma_db"
    max_file_size: int = 100 * 1024 * 1024  # 100MB

    class Config:
        env_file = ".env"
```

## 🚀 Deployment Options

### Local Development

- **Environment**: Local Python environment
- **Database**: Local ChromaDB
- **Storage**: Local filesystem
- **Access**: localhost:7860

### Demo Deployment

- **Platform**: Hugging Face Spaces (recommended)
- **Runtime**: Gradio app
- **Storage**: Temporary (session-based)
- **Cost**: Free tier available

### Production Options

- **Cloud**: AWS/GCP/Azure with Docker
- **Database**: ChromaDB Cloud or Pinecone
- **Storage**: S3/GCS/Azure Blob
- **Monitoring**: Weights & Biases or similar

## 📈 Performance Specifications

### Target Performance

- **Document Processing**: < 30 seconds per document
- **Search Response**: < 2 seconds
- **UI Responsiveness**: < 100ms interactions
- **Memory Usage**: < 2GB RAM for 1000 documents
- **Storage**: ~10MB per 100 documents

### Scalability Limits (Demo)

- **Max Documents**: 1000 (recommended for demo)
- **Max File Size**: 100MB per file
- **Concurrent Users**: 1 (demo limitation)
- **API Calls**: Rate limited by OpenAI tier

## 🔒 Security Considerations

### API Key Management

- Environment variables only
- No hardcoded keys in source code
- Local .env file for development
- Secrets management for production

### Data Privacy

- Local document processing
- No document upload to external services (except OpenAI for embeddings)
- User data not persisted in demo mode
- GDPR-compliant data handling

### Input Validation

- File type validation
- File size limits
- Content sanitization
- Error handling for malicious files

## 🧪 Testing Framework

### Unit Testing

- **Framework**: pytest
- **Coverage**: pytest-cov
- **Mocking**: pytest-mock for API calls

### Integration Testing

- **Database**: Test ChromaDB instance
- **File Processing**: Sample documents
- **API**: Mock OpenAI responses

### UI Testing

- **Framework**: Gradio testing utilities
- **Browser**: Selenium for E2E (optional)

## 📊 Monitoring and Logging

### Application Logging

- **Library**: Python logging
- **Format**: Structured JSON logs
- **Levels**: DEBUG, INFO, WARNING, ERROR, CRITICAL

### Metrics Collection

- **Response times**: Per operation timing
- **Error rates**: Failed operations tracking
- **Usage patterns**: Search query analysis
- **Resource usage**: Memory and CPU monitoring

### Health Checks

- **API connectivity**: OpenAI API status
- **Database**: ChromaDB connection
- **Storage**: Filesystem availability
- **Memory**: RAM usage monitoring

## 🔄 Version Management

### Semantic Versioning

- **Format**: MAJOR.MINOR.PATCH
- **Current**: 1.0.0 (MVP release)
- **Breaking Changes**: MAJOR version bump
- **New Features**: MINOR version bump
- **Bug Fixes**: PATCH version bump

### Dependency Updates

- **Strategy**: Conservative updates for stability
- **Testing**: Full test suite before version bumps
- **Security**: Immediate updates for security patches
- **Compatibility**: Maintain Python 3.11+ support

## 📋 Architecture Decisions

### Framework Choice Rationale

1. **LangChain over LlamaIndex**:

   - Better ecosystem maturity
   - More extensive documentation
   - Stronger community support

2. **Gradio over Streamlit**:

   - Better for ML demos
   - Faster development cycle
   - Less complexity for single-user apps

3. **ChromaDB over Pinecone**:

   - Cost-effective for demos
   - Local development friendly
   - Open-source flexibility

4. **OpenAI over Open Source LLMs**:
   - Reliability and performance
   - Comprehensive API ecosystem
   - Professional demo requirements

---

_This technical stack has been optimized for 2025 standards and production readiness while maintaining demo simplicity._
