# PR6: Chat Engine & RAG Implementation - Simplified Guide

## Overview
Implement a simple but effective chat system that answers questions using document context.

## Goal
Create a chat interface that:
1. Takes user questions
2. Finds relevant document chunks
3. Uses GPT-4 to answer based on those chunks
4. Shows which documents were used

## Simplified File Structure
```
core/
├── chat_engine.py      # Simple chat with RAG
└── rag_pipeline.py     # Combines search + chat
```

## Implementation Guide

### 1. Chat Engine (`core/chat_engine.py`)

```python
from typing import List, Optional
import openai
from core.models.chat import ChatMessage
from config.settings import get_settings

class ChatEngine:
    """Simple chat engine using GPT-4."""
    
    def __init__(self):
        settings = get_settings()
        self.client = openai.OpenAI(api_key=settings.openai_api_key)
        self.model = settings.openai_model
    
    def chat(
        self, 
        query: str, 
        context_chunks: List[str],
        history: Optional[List[ChatMessage]] = None
    ) -> str:
        """Generate response with document context."""
        
        # Build context from chunks
        context = "\n\n".join([
            f"[Document excerpt {i+1}]:\n{chunk}" 
            for i, chunk in enumerate(context_chunks[:5])
        ])
        
        # Simple system message
        system_msg = """You are a helpful assistant that answers questions based on provided documents.
When answering, mention which document excerpt you're using.
If the documents don't contain the answer, say so clearly."""
        
        messages = [{"role": "system", "content": system_msg}]
        
        # Add last 3 messages from history
        if history:
            for msg in history[-3:]:
                messages.append({"role": msg.role, "content": msg.content})
        
        # Add current question with context
        user_msg = f"""Documents:
{context}

Question: {query}"""
        
        messages.append({"role": "user", "content": user_msg})
        
        # Get response
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"
```

### 2. RAG Pipeline (`core/rag_pipeline.py`)

```python
from typing import List, Tuple
from core.chat_engine import ChatEngine
from core.vector_store import VectorStore
from core.models.chat import ChatMessage

class RAGPipeline:
    """Simple RAG pipeline combining search and chat."""
    
    def __init__(self):
        self.chat_engine = ChatEngine()
        self.vector_store = VectorStore()
    
    def query(
        self, 
        question: str,
        history: Optional[List[ChatMessage]] = None
    ) -> Tuple[str, List[str]]:
        """
        Answer question using RAG.
        Returns: (answer, source_documents)
        """
        
        # Search for relevant chunks
        search_results = self.vector_store.search(
            query_text=question,
            limit=5
        )
        
        if not search_results:
            return "I couldn't find any relevant information in the documents.", []
        
        # Extract text and sources
        chunks = [result.content for result in search_results]
        sources = list(set([
            result.metadata.get('filename', 'Unknown') 
            for result in search_results
        ]))
        
        # Generate answer
        answer = self.chat_engine.chat(
            query=question,
            context_chunks=chunks,
            history=history
        )
        
        # Add sources to answer if not already mentioned
        if sources and not any(src in answer for src in sources):
            answer += f"\n\nSources: {', '.join(sources)}"
        
        return answer, sources
```

## Integration Example

```python
# In your Gradio app
from core.rag_pipeline import RAGPipeline
from core.models.chat import ChatMessage

# Initialize
rag = RAGPipeline()

# Use in chat interface
def chat_fn(message, history):
    # Convert history to ChatMessage objects
    chat_history = []
    for h in history:
        chat_history.append(ChatMessage(role="user", content=h[0]))
        chat_history.append(ChatMessage(role="assistant", content=h[1]))
    
    # Get response
    answer, sources = rag.query(message, chat_history)
    
    return answer
```

## Testing Guide

### Basic Test Cases

```python
def test_rag_pipeline():
    """Test the complete RAG flow."""
    rag = RAGPipeline()
    
    # Test with a simple question
    answer, sources = rag.query("What is the main topic?")
    assert isinstance(answer, str)
    assert isinstance(sources, list)
    
    # Test with no relevant docs
    answer, _ = rag.query("Random question about xyz123")
    assert "couldn't find" in answer.lower()
```

## Key Simplifications from Original

1. **No complex conversation management** - Just pass last 3 messages
2. **No token counting** - Let OpenAI handle limits
3. **No response formatting** - Keep responses natural
4. **No fallback strategies** - Simple "not found" message
5. **No streaming** - Complete responses only
6. **No cost tracking** - Not needed for demo

## Success Criteria

- [ ] User asks question → Gets answer with sources
- [ ] Works with multiple document types
- [ ] Handles "not found" gracefully
- [ ] Response time < 3 seconds
- [ ] No errors during typical use

## Common Issues & Solutions

1. **Rate limits**: Add simple retry with 1 second delay
2. **Large contexts**: Limit to 5 chunks max
3. **No relevant docs**: Return friendly "not found" message
4. **Token limits**: Use shorter chunks (500 chars)

## Demo Scenarios

1. **Legal Q&A**: "What are the termination clauses?"
2. **Technical Docs**: "How do I install the software?"
3. **Research**: "What were the study findings?"
4. **Multi-doc**: "Compare the budgets across documents"

Remember: Keep it simple. The goal is to impress in a demo, not build a production system.