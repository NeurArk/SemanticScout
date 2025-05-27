# CLAUDE.md - SemanticScout Project Context

## IMPORTANT: Git Commit Rules
**NEVER** add the following to git commits:
- 🤖 Generated with [Claude Code](https://claude.ai/code)
- Co-Authored-By: Claude <noreply@anthropic.com>

Keep commits clean and professional without AI attribution.

## Project Overview
SemanticScout is a **"Chat with your Documents"** demo application combining:
- **GPT-4.1 Chat Interface**: Conversational AI to discuss documents
- **Semantic Search**: Embedding-based document retrieval
- **RAG Pipeline**: Retrieval Augmented Generation
- **Gradio UI**: Professional interface (NO REST API)

## Current Development Status
- PR1: ✅ Complete (Project foundation)
- PR2-10: To be implemented by coding agent

## PR Implementation Guides
Detailed implementation guides are available in `docs/prs/`:
- PR2-DETAILED.md: Core Models (with chat models)
- PR3-DETAILED.md: Document Processing
- PR4-DETAILED.md: Embedding Generation
- PR5-DETAILED.md: Vector Database Integration
- PR6-DETAILED.md: Chat Engine & RAG Implementation
- PR7-DETAILED.md: Gradio UI (Chat-first design)
- PR8-DETAILED.md: Visualization & Analytics
- PR9-DETAILED.md: Testing & Quality Assurance
- PR10-DETAILED.md: Deployment & Documentation

These guides provide complete code examples and step-by-step instructions.

## Key Technical Decisions

### Core Features
1. **Chat functionality** with GPT-4.1
2. **Semantic search** with text-embedding-3-large
3. **Document processing** for PDF, DOCX, TXT, MD
4. **Vector storage** with ChromaDB
5. **Visualization** of document relationships

### Technical Parameters
- Chunk size: 1000 tokens (200 overlap)
- Retrieval: Top 5 documents for context
- Chat context: 8000 tokens max
- Search results: 10 documents
- Processing timeout: 30s per document

### Architecture Choices
- **Frontend**: Gradio only (no REST API)
- **Backend**: Direct Python service calls
- **Storage**: Local ChromaDB + file system
- **Models**: OpenAI GPT-4.1 + embeddings

## Common Issues to Watch

### Documentation Alignment
1. **No REST API** despite old references
2. **Chat feature** is primary (not just search)
3. **Branch strategy**: main + feature branches only
4. **Demo focus**: Not production-ready features

### Development Priorities
1. Chat experience quality
2. RAG accuracy
3. Visual polish for demos
4. Fast response times
5. Clear error handling

### Cost Optimization
- Mock OpenAI in tests
- Cache embeddings aggressively
- Limit context window size
- Use batch processing

## How to Assist Developer

### When Reviewing Code
- Ensure chat functionality is implemented
- Verify RAG pipeline correctness
- Check Gradio UI matches guidelines
- Confirm no REST API code
- Validate test mocking

### Key Files to Monitor
- `app.py` - Main Gradio application
- `core/chat_engine.py` - Chat with RAG
- `core/search_engine.py` - Semantic search
- `config/settings.py` - Configuration

### Demo Scenarios
1. **Legal Documents**: Contract analysis
2. **Research Papers**: Literature review
3. **Technical Docs**: API documentation chat
4. **Business Reports**: Financial analysis

## Project Goals
- **Primary**: Impressive client demos
- **Secondary**: Portfolio showcase
- **Tertiary**: Reusable RAG template

Remember: This is a DEMO showcasing "chat with documents" capabilities, not a production system.