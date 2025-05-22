# 🔍 SemanticScout

> **AI-Powered Semantic Document Search Engine**  
> Transform how you discover and explore documents through intelligent semantic understanding

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![OpenAI GPT-4.1](https://img.shields.io/badge/OpenAI-GPT--4.1-green.svg)](https://platform.openai.com/)
[![LangChain](https://img.shields.io/badge/LangChain-latest-orange.svg)](https://python.langchain.com/)
[![Gradio](https://img.shields.io/badge/Gradio-latest-purple.svg)](https://gradio.app/)

## 🎯 Overview

SemanticScout is a cutting-edge semantic search system that revolutionizes document discovery by understanding **conceptual meaning** rather than relying on exact keyword matches. Built with the latest 2025 AI technologies, it provides enterprise-grade capabilities in an intuitive interface perfect for demonstrations and professional applications.

### ✨ Key Features

- **🧠 Intelligent Processing**: Multi-format document support (PDF, DOCX, TXT, MD)
- **🔍 Semantic Search**: Natural language queries with contextual understanding
- **📊 Visual Analytics**: Interactive document relationship visualization
- **⚡ Real-time Results**: Sub-2-second search responses with relevance scoring
- **🎨 Professional UI**: Modern Gradio interface optimized for demos
- **🛡️ Enterprise Ready**: Secure, scalable architecture with comprehensive testing

## 🚀 Quick Start

### Prerequisites

- **Python 3.11+** (3.12 recommended)
- **OpenAI API Key** ([Get yours here](https://platform.openai.com/api-keys))
- **Git** for version control

### Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/NeurArk/SemanticScout.git
   cd SemanticScout
   ```

2. **Create virtual environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env and add your OpenAI API key
   ```

5. **Launch the application**:
   ```bash
   python app.py
   ```

6. **Open your browser** to `http://localhost:7860`

## 💡 Usage

### Document Upload
- Drag & drop or click to upload PDF, DOCX, TXT, or MD files
- Maximum file size: 100MB
- Real-time processing with progress indicators

### Semantic Search
- Enter natural language queries like:
  - *"Documents about machine learning optimization techniques"*
  - *"Research papers discussing environmental sustainability"*
  - *"Technical specifications for database design"*

### Results Exploration
- Browse ranked results with relevance scores
- View contextual excerpts with highlighted matches
- Explore document relationships through interactive visualizations

## 🏗️ Architecture

### Core Technologies

| Component | Technology | Version | Purpose |
|-----------|------------|---------|---------|
| **AI Framework** | LangChain + LangGraph | Latest | RAG pipeline orchestration |
| **Language Model** | OpenAI GPT-4.1 | Latest | Query understanding & processing |
| **Embeddings** | text-embedding-3-large | 3072-dim | Semantic vector generation |
| **Vector DB** | ChromaDB | Latest | Efficient similarity search |
| **UI Framework** | Gradio | Latest | Interactive web interface |
| **Visualization** | Plotly + NetworkX | Latest | Document relationship graphs |

### System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │────│   Processing    │────│   Vector        │
│   Upload        │    │   Pipeline      │    │   Storage       │
│   (Multi-format)│    │   (Extraction)  │    │   (ChromaDB)    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Gradio UI     │────│   Search Engine │────│   OpenAI API    │
│   (User Interface)   │   (Semantic)    │    │   (GPT-4.1)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📊 Performance

- **Processing Speed**: < 30 seconds per document
- **Search Latency**: < 2 seconds average response time
- **Accuracy**: 85%+ relevance score on semantic queries
- **Scalability**: Optimized for 1000+ documents
- **Memory Usage**: < 2GB RAM for typical workloads

## 🛠️ Development

### Project Structure

```
SemanticScout/
├── app.py                 # Main application entry point
├── src/                   # Source code
│   ├── core/             # Core business logic
│   ├── processing/       # Document processing
│   ├── search/           # Search engine
│   ├── ui/               # User interface
│   └── utils/            # Utilities
├── tests/                # Test suites
├── docs/                 # Documentation
├── data/                 # Local data storage
└── requirements.txt      # Dependencies
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src

# Run specific test category
pytest tests/unit/
pytest tests/integration/
```

### Code Quality

```bash
# Format code
black src/ tests/

# Check types
mypy src/

# Lint code
flake8 src/ tests/

# Run all checks
pre-commit run --all-files
```

## 🚀 Deployment

### Local Demo
Perfect for client presentations and development:
```bash
python app.py --host 0.0.0.0 --port 7860
```

### Hugging Face Spaces
Free hosting for demos:
1. Fork this repository
2. Create a new Space on Hugging Face
3. Connect your GitHub repository
4. Add your OpenAI API key to Space secrets

### Docker Production
```bash
docker build -t semantic-scout .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key semantic-scout
```

## 📈 Use Cases

### 🏢 Enterprise Applications
- **Document Discovery**: Find relevant documents across large repositories
- **Knowledge Management**: Organize and search company knowledge bases
- **Research Assistance**: Accelerate literature reviews and research
- **Compliance**: Locate policy documents and regulatory information

### 🎯 Demo Scenarios
- **Client Presentations**: Showcase AI capabilities with real documents
- **Technical Interviews**: Demonstrate semantic search understanding
- **Portfolio Projects**: Highlight modern AI/ML development skills
- **Proof of Concepts**: Validate semantic search for specific domains

## 📚 Documentation

- **[Product Requirements](docs/PRD.md)**: Complete product specification
- **[Technical Architecture](docs/ARCHITECTURE.md)**: Detailed system design
- **[API Reference](docs/API_SPECIFICATION.md)**: REST API documentation
- **[Deployment Guide](docs/DEPLOYMENT.md)**: Production deployment instructions
- **[Development Roadmap](docs/TODO.md)**: Feature development timeline

## 🤝 Contributing

We welcome contributions! Please see our development guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Development Guidelines
- Follow the existing code style and patterns
- Add tests for new features
- Update documentation for significant changes
- Ensure all CI checks pass

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Professional Portfolio

**SemanticScout** showcases advanced AI/ML capabilities including:
- **RAG (Retrieval Augmented Generation)** implementation
- **Vector database** integration and optimization
- **Modern LLM** application development
- **Production-ready** software architecture
- **Enterprise-grade** user experience design

Perfect for demonstrating expertise in:
- 🤖 **AI/ML Engineering**
- 🔧 **Python Development**
- 🏗️ **System Architecture**
- 🎨 **UI/UX Design**
- 📊 **Data Visualization**

## 🔗 Links

- **Live Demo**: [Coming Soon]
- **Documentation**: [docs/](docs/)
- **GitHub**: [https://github.com/NeurArk/SemanticScout](https://github.com/NeurArk/SemanticScout)
- **Portfolio**: [NeurArk](https://github.com/NeurArk)

---

**Built with ❤️ by [NeurArk](https://www.neurark.com) • Powered by OpenAI GPT-4.1 & LangChain**

*Transform your document search experience with the power of semantic AI.*