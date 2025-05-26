# PR6: Chat Engine & Search Implementation (RAG) - Detailed Implementation Guide

## Overview
This PR implements the core RAG (Retrieval Augmented Generation) pipeline, combining GPT-4.1 chat capabilities with semantic search to enable conversations about documents.

## Prerequisites
- PR2-5 completed (models, processing, embeddings, vector store)
- OpenAI API key with GPT-4.1 access

## File Structure
```
core/
├── chat_engine.py           # Main chat orchestrator
├── search_engine.py         # Search functionality
├── rag/
│   ├── __init__.py
│   ├── rag_pipeline.py      # RAG orchestration
│   ├── context_builder.py   # Context preparation
│   ├── prompt_manager.py    # Prompt templates
│   └── response_formatter.py # Response formatting
└── chat/
    ├── __init__.py
    ├── conversation_manager.py # Chat history
    └── gpt_client.py        # GPT-4.1 client
```

## Detailed Implementation

### 1. GPT Client (`core/chat/gpt_client.py`)

```python
from typing import List, Dict, Optional
import openai
from openai import OpenAI
import logging
from core.exceptions.custom_exceptions import ChatError
from config.settings import settings

logger = logging.getLogger(__name__)

class GPTClient:
    """GPT-4.1 client for chat completions."""
    
    def __init__(self):
        self.client = OpenAI(api_key=settings.openai_api_key)
        self.model = settings.chat_model  # gpt-4.1
        self.temperature = settings.chat_temperature
        self.max_tokens = settings.chat_max_tokens
    
    def generate_response(self, messages: List[Dict[str, str]], 
                         temperature: Optional[float] = None) -> str:
        """Generate chat response from GPT-4.1."""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature or self.temperature,
                max_tokens=self.max_tokens,
                n=1,
                stream=False
            )
            
            content = response.choices[0].message.content
            
            # Log token usage
            if response.usage:
                logger.info(f"Token usage - Prompt: {response.usage.prompt_tokens}, "
                          f"Completion: {response.usage.completion_tokens}, "
                          f"Total: {response.usage.total_tokens}")
            
            return content
            
        except openai.RateLimitError as e:
            logger.error(f"Rate limit error: {e}")
            raise ChatError("API rate limit reached. Please try again later.")
        except openai.APIError as e:
            logger.error(f"OpenAI API error: {e}")
            raise ChatError(f"Chat generation failed: {str(e)}")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            raise ChatError(f"An unexpected error occurred: {str(e)}")
    
    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Estimate token count for messages."""
        import tiktoken
        
        encoding = tiktoken.encoding_for_model(self.model)
        
        # Estimate tokens (rough approximation)
        token_count = 0
        for message in messages:
            token_count += len(encoding.encode(message.get('content', '')))
            token_count += 4  # Message overhead
        
        token_count += 2  # Priming tokens
        
        return token_count
```

### 2. Conversation Manager (`core/chat/conversation_manager.py`)

```python
from typing import List, Optional
from datetime import datetime
from core.models.chat import ChatMessage, MessageRole, ChatContext
from core.models.document import DocumentChunk
import logging

logger = logging.getLogger(__name__)

class ConversationManager:
    """Manages chat conversation history and context."""
    
    def __init__(self, max_history: int = 10):
        self.messages: List[ChatMessage] = []
        self.max_history = max_history
    
    def add_message(self, role: MessageRole, content: str) -> ChatMessage:
        """Add a message to the conversation."""
        message = ChatMessage(
            role=role,
            content=content,
            timestamp=datetime.now()
        )
        
        self.messages.append(message)
        
        # Trim history if needed
        if len(self.messages) > self.max_history * 2:  # Keep some buffer
            self.messages = self.messages[-self.max_history:]
        
        logger.debug(f"Added {role} message, total messages: {len(self.messages)}")
        return message
    
    def get_context(self, retrieved_chunks: List[DocumentChunk], 
                   max_messages: Optional[int] = None) -> ChatContext:
        """Get chat context with retrieved documents."""
        # Get recent messages
        messages = self.messages[-(max_messages or self.max_history):]
        
        return ChatContext(
            messages=messages,
            retrieved_chunks=retrieved_chunks
        )
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.messages = []
        logger.info("Conversation history cleared")
    
    def get_history_summary(self) -> str:
        """Get a summary of conversation history."""
        if not self.messages:
            return "No conversation history."
        
        summary_parts = []
        for msg in self.messages[-5:]:  # Last 5 messages
            role = "User" if msg.role == MessageRole.USER else "Assistant"
            preview = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            summary_parts.append(f"{role}: {preview}")
        
        return "\n".join(summary_parts)
```

### 3. Context Builder (`core/rag/context_builder.py`)

```python
from typing import List, Tuple, Optional
from core.models.document import DocumentChunk
from core.models.search import SearchResult
import logging

logger = logging.getLogger(__name__)

class ContextBuilder:
    """Builds context from retrieved documents for RAG."""
    
    def __init__(self, max_context_length: int = 8000):
        self.max_context_length = max_context_length
    
    def build_context(self, search_results: List[SearchResult], 
                     max_chunks: int = 5) -> Tuple[str, List[DocumentChunk]]:
        """Build context string from search results."""
        if not search_results:
            return "", []
        
        # Sort by relevance score
        sorted_results = sorted(search_results, key=lambda r: r.score, reverse=True)
        
        # Take top chunks
        selected_results = sorted_results[:max_chunks]
        
        # Build context
        context_parts = []
        used_chunks = []
        current_length = 0
        
        for result in selected_results:
            # Create source citation
            source_info = f"[Source: {result.metadata.get('filename', 'Unknown')}, " \
                         f"Page/Section: {result.metadata.get('chunk_index', 'N/A')}]"
            
            # Add to context if within limit
            chunk_text = f"{source_info}\n{result.content}\n"
            chunk_length = len(chunk_text)
            
            if current_length + chunk_length > self.max_context_length:
                logger.warning(f"Context length limit reached, using {len(used_chunks)} chunks")
                break
            
            context_parts.append(chunk_text)
            used_chunks.append(result)
            current_length += chunk_length
        
        # Join with separators
        context = "\n---\n".join(context_parts)
        
        logger.info(f"Built context with {len(used_chunks)} chunks, "
                   f"total length: {current_length} chars")
        
        return context, used_chunks
    
    def format_sources(self, chunks: List[SearchResult]) -> str:
        """Format source citations for display."""
        if not chunks:
            return ""
        
        sources = []
        seen_files = set()
        
        for chunk in chunks:
            filename = chunk.metadata.get('filename', 'Unknown')
            if filename not in seen_files:
                seen_files.add(filename)
                sources.append(f"📄 {filename}")
        
        return "Sources: " + ", ".join(sources)
```

### 4. RAG Pipeline (`core/rag/rag_pipeline.py`)

```python
from typing import Optional, Tuple, List
from core.models.chat import ChatContext, MessageRole
from core.models.search import SearchQuery, SearchResult
from core.vector_store import VectorStore
from core.embedder import EmbeddingService
from .context_builder import ContextBuilder
from .prompt_manager import PromptManager
from core.chat.gpt_client import GPTClient
import logging

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Orchestrates the RAG process for chat responses."""
    
    def __init__(self, vector_store: VectorStore, 
                 embedding_service: EmbeddingService,
                 gpt_client: GPTClient):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.gpt_client = gpt_client
        self.context_builder = ContextBuilder()
        self.prompt_manager = PromptManager()
    
    def generate_response(self, query: str, 
                         chat_context: ChatContext) -> Tuple[str, List[SearchResult]]:
        """Generate response using RAG pipeline."""
        logger.info(f"Processing query: {query[:100]}...")
        
        # Step 1: Retrieve relevant documents
        retrieved_chunks = self._retrieve_documents(query)
        
        # Step 2: Build context
        context_str, used_chunks = self.context_builder.build_context(retrieved_chunks)
        
        # Step 3: Prepare messages for GPT
        messages = self._prepare_messages(query, context_str, chat_context)
        
        # Step 4: Generate response
        response = self.gpt_client.generate_response(messages)
        
        # Step 5: Add source citations if chunks were used
        if used_chunks:
            sources = self.context_builder.format_sources(used_chunks)
            response = f"{response}\n\n{sources}"
        
        logger.info(f"Generated response using {len(used_chunks)} sources")
        
        return response, used_chunks
    
    def _retrieve_documents(self, query: str) -> List[SearchResult]:
        """Retrieve relevant documents for the query."""
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query)
            
            # Create search query
            search_query = SearchQuery(
                query_text=query,
                max_results=10,  # Retrieve more, select best later
                similarity_threshold=0.7
            )
            
            # Search
            search_response = self.vector_store.search(query_embedding, search_query)
            
            return search_response.results
            
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return []
    
    def _prepare_messages(self, query: str, context: str, 
                         chat_context: ChatContext) -> List[Dict[str, str]]:
        """Prepare messages for GPT including context."""
        
        # Get system prompt
        system_prompt = self.prompt_manager.get_system_prompt(
            has_context=bool(context)
        )
        
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add context if available
        if context:
            context_message = self.prompt_manager.format_context_message(context)
            messages.append({"role": "system", "content": context_message})
        
        # Add conversation history (limited)
        for msg in chat_context.messages[-4:]:  # Last 4 messages
            messages.append({
                "role": msg.role.value,
                "content": msg.content
            })
        
        # Add current query
        messages.append({"role": "user", "content": query})
        
        return messages
```

### 5. Search Engine (`core/search_engine.py`)

```python
from typing import List, Optional
from core.models.search import SearchQuery, SearchResponse, SearchResult
from core.vector_store import VectorStore
from core.embedder import EmbeddingService
import logging
import time

logger = logging.getLogger(__name__)

class SearchEngine:
    """Semantic search engine for documents."""
    
    def __init__(self, vector_store: VectorStore, 
                 embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
    
    def search(self, query_text: str, 
              max_results: int = 10,
              file_types: Optional[List[str]] = None) -> SearchResponse:
        """Perform semantic search."""
        logger.info(f"Searching for: {query_text}")
        
        start_time = time.time()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_service.embed_query(query_text)
            
            # Create search query
            search_query = SearchQuery(
                query_text=query_text,
                max_results=max_results,
                filter_file_types=file_types,
                similarity_threshold=0.5  # Lower threshold for search
            )
            
            # Execute search
            response = self.vector_store.search(query_embedding, search_query)
            
            # Enhance results with highlighting
            for result in response.results:
                result.highlighted_content = self._highlight_content(
                    result.content, 
                    query_text
                )
            
            logger.info(f"Search completed: {response.total_results} results "
                       f"in {response.search_time_ms:.2f}ms")
            
            return response
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            # Return empty response
            return SearchResponse(
                query=SearchQuery(query_text=query_text),
                results=[],
                total_results=0,
                search_time_ms=(time.time() - start_time) * 1000
            )
    
    def _highlight_content(self, content: str, query: str, 
                          context_length: int = 200) -> str:
        """Create highlighted snippet around query terms."""
        # Simple highlighting - in production use more sophisticated approach
        query_words = query.lower().split()
        content_lower = content.lower()
        
        # Find first occurrence of any query word
        best_pos = -1
        for word in query_words:
            pos = content_lower.find(word)
            if pos != -1 and (best_pos == -1 or pos < best_pos):
                best_pos = pos
        
        if best_pos == -1:
            # No match found, return beginning
            return content[:context_length] + "..."
        
        # Extract context around match
        start = max(0, best_pos - context_length // 2)
        end = min(len(content), best_pos + context_length // 2)
        
        snippet = content[start:end]
        if start > 0:
            snippet = "..." + snippet
        if end < len(content):
            snippet = snippet + "..."
        
        return snippet
```

### 6. Chat Engine (`core/chat_engine.py`)

```python
from typing import Tuple, List, Optional
from core.models.chat import MessageRole
from core.models.search import SearchResult
from core.vector_store import VectorStore
from core.embedder import EmbeddingService
from .rag.rag_pipeline import RAGPipeline
from .chat.conversation_manager import ConversationManager
from .chat.gpt_client import GPTClient
import logging

logger = logging.getLogger(__name__)

class ChatEngine:
    """Main chat engine combining conversation and RAG."""
    
    def __init__(self, vector_store: VectorStore, 
                 embedding_service: EmbeddingService):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.gpt_client = GPTClient()
        self.conversation_manager = ConversationManager()
        self.rag_pipeline = RAGPipeline(
            vector_store,
            embedding_service,
            self.gpt_client
        )
    
    def chat(self, user_input: str) -> Tuple[str, List[SearchResult]]:
        """Process user input and generate response."""
        
        # Add user message to history
        self.conversation_manager.add_message(MessageRole.USER, user_input)
        
        # Check if documents are available
        doc_stats = self.vector_store.get_stats()
        has_documents = doc_stats.get('total_documents', 0) > 0
        
        if not has_documents:
            # No documents - provide helpful response
            response = "I don't have any documents to reference yet. " \
                      "Please upload some documents first so I can help you " \
                      "explore and answer questions about them."
            sources = []
        else:
            # Get chat context
            chat_context = self.conversation_manager.get_context([])
            
            # Generate response using RAG
            response, sources = self.rag_pipeline.generate_response(
                user_input, 
                chat_context
            )
        
        # Add assistant response to history
        self.conversation_manager.add_message(MessageRole.ASSISTANT, response)
        
        return response, sources
    
    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_manager.clear_history()
        logger.info("Chat history cleared")
    
    def get_history_summary(self) -> str:
        """Get conversation history summary."""
        return self.conversation_manager.get_history_summary()
```

### 7. Prompt Manager (`core/rag/prompt_manager.py`)

```python
class PromptManager:
    """Manages prompts for the RAG system."""
    
    def get_system_prompt(self, has_context: bool) -> str:
        """Get system prompt based on context availability."""
        if has_context:
            return """You are a helpful AI assistant with access to the user's documents. 
Your role is to answer questions based on the provided document context.

Guidelines:
1. Base your answers primarily on the provided document excerpts
2. If the answer is in the documents, cite the specific source
3. If the answer isn't in the documents, clearly state this
4. Be concise but thorough in your responses
5. Maintain a professional and helpful tone"""
        else:
            return """You are a helpful AI assistant. The user hasn't uploaded any documents yet, 
so you cannot answer questions about specific documents. Instead:
1. Explain that documents need to be uploaded first
2. Describe what types of questions you can answer once documents are available
3. Be encouraging and helpful about the document upload process"""
    
    def format_context_message(self, context: str) -> str:
        """Format context for inclusion in prompt."""
        return f"""Here are the relevant document excerpts to answer the user's question:

{context}

Use these excerpts to provide an accurate, well-sourced answer."""
```

## Testing

```python
# tests/unit/test_chat_engine.py
import pytest
from unittest.mock import Mock, patch
from core.chat_engine import ChatEngine

@pytest.fixture
def mock_dependencies():
    vector_store = Mock()
    embedding_service = Mock()
    return vector_store, embedding_service

def test_chat_without_documents(mock_dependencies):
    vector_store, embedding_service = mock_dependencies
    vector_store.get_stats.return_value = {'total_documents': 0}
    
    engine = ChatEngine(vector_store, embedding_service)
    response, sources = engine.chat("What is machine learning?")
    
    assert "upload some documents" in response.lower()
    assert len(sources) == 0

def test_chat_with_documents(mock_dependencies):
    vector_store, embedding_service = mock_dependencies
    vector_store.get_stats.return_value = {'total_documents': 5}
    
    # Mock search results
    mock_results = [Mock(score=0.9, content="Machine learning is...")]
    vector_store.search.return_value = Mock(results=mock_results)
    
    with patch('core.chat.gpt_client.GPTClient.generate_response') as mock_gpt:
        mock_gpt.return_value = "Based on the documents, machine learning is..."
        
        engine = ChatEngine(vector_store, embedding_service)
        response, sources = engine.chat("What is machine learning?")
        
        assert "machine learning" in response.lower()
        assert len(sources) > 0
```

## Success Criteria

1. ✅ Chat responds accurately using document context
2. ✅ GPT-4.1 integration works smoothly
3. ✅ RAG pipeline retrieves relevant chunks
4. ✅ Sources are properly cited in responses
5. ✅ Conversation history is maintained
6. ✅ Fallback behavior when no documents
7. ✅ Search functionality works independently
8. ✅ Response time < 5 seconds typically