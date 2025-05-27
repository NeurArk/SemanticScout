# PR7: Gradio Interface - Simple Demo-Focused Guide

## Overview
Create a clean, professional Gradio interface that impresses during demos. Focus on reliability over features.

## Goal
Build an interface where:
1. Users can drag & drop documents
2. Chat about their documents naturally
3. See which sources were used
4. Everything works smoothly

## Single File Approach
```
app.py              # Everything in one file for simplicity
```

## Implementation Guide

### Complete Gradio App (`app.py`)

```python
import gradio as gr
import os
from typing import List, Tuple
from pathlib import Path

from core.document_processor import DocumentProcessor
from core.embedder import EmbeddingService
from core.vector_store import VectorStore
from core.rag_pipeline import RAGPipeline
from core.models.chat import ChatMessage

# Initialize services
doc_processor = DocumentProcessor()
embedder = EmbeddingService()
vector_store = VectorStore()
rag_pipeline = RAGPipeline()

# Store uploaded files info
uploaded_files = {}

def process_file(file) -> str:
    """Process uploaded file and add to vector store."""
    try:
        if file is None:
            return "No file uploaded"
        
        # Save uploaded file
        file_path = file.name
        filename = Path(file_path).name
        
        # Check if already processed
        if filename in uploaded_files:
            return f"✓ {filename} already processed"
        
        # Process document
        doc, chunks = doc_processor.process_document(file_path)
        
        # Generate embeddings
        embedded_chunks = embedder.embed_document(doc, chunks)
        
        # Store in vector database
        vector_store.add_document(doc, embedded_chunks)
        
        # Track uploaded file
        uploaded_files[filename] = {
            'doc_id': doc.id,
            'chunks': len(chunks)
        }
        
        return f"✓ Successfully processed {filename} ({len(chunks)} chunks)"
        
    except Exception as e:
        return f"❌ Error processing file: {str(e)}"

def chat_response(message: str, history: List[List[str]]) -> str:
    """Generate chat response using RAG."""
    try:
        # Convert history to ChatMessage format
        chat_history = []
        for user_msg, assistant_msg in history:
            chat_history.append(ChatMessage(role="user", content=user_msg))
            chat_history.append(ChatMessage(role="assistant", content=assistant_msg))
        
        # Get response from RAG pipeline
        answer, sources = rag_pipeline.query(message, chat_history)
        
        return answer
        
    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"

def get_document_list() -> str:
    """Get list of uploaded documents."""
    if not uploaded_files:
        return "No documents uploaded yet"
    
    doc_list = "📄 **Uploaded Documents:**\n\n"
    for filename, info in uploaded_files.items():
        doc_list += f"• {filename} ({info['chunks']} chunks)\n"
    
    return doc_list

def clear_all_documents() -> str:
    """Clear all documents from vector store."""
    global uploaded_files
    
    try:
        # Clear vector store
        vector_store.clear()
        
        # Clear tracking
        uploaded_files = {}
        
        return "✓ All documents cleared"
    except Exception as e:
        return f"❌ Error clearing documents: {str(e)}"

# Create Gradio interface
with gr.Blocks(title="SemanticScout - Chat with your Documents") as app:
    gr.Markdown(
        """
        # 🔍 SemanticScout
        ### Chat naturally with your documents using AI
        
        Upload PDFs, Word docs, or text files and ask questions about their content.
        """
    )
    
    with gr.Row():
        with gr.Column(scale=3):
            # Chat interface
            chatbot = gr.Chatbot(
                height=500,
                show_label=False,
                elem_id="chatbot"
            )
            
            msg = gr.Textbox(
                label="Ask a question about your documents",
                placeholder="e.g., What are the main findings? What does the contract say about termination?",
                lines=2
            )
            
            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Chat")
        
        with gr.Column(scale=1):
            # Document management
            gr.Markdown("### 📁 Document Management")
            
            file_upload = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                type="filepath"
            )
            
            upload_status = gr.Textbox(
                label="Status",
                interactive=False,
                lines=2
            )
            
            doc_list = gr.Markdown(get_document_list())
            
            refresh_btn = gr.Button("Refresh List", size="sm")
            clear_docs_btn = gr.Button("Clear All Documents", variant="stop", size="sm")
    
    # Event handlers
    def respond(message, chat_history):
        bot_message = chat_response(message, chat_history)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    # File upload handler
    file_upload.change(
        fn=process_file,
        inputs=[file_upload],
        outputs=[upload_status]
    ).then(
        fn=get_document_list,
        outputs=[doc_list]
    )
    
    # Chat handlers
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)
    
    # Document management handlers
    refresh_btn.click(
        fn=get_document_list,
        outputs=[doc_list]
    )
    
    clear_docs_btn.click(
        fn=clear_all_documents,
        outputs=[upload_status]
    ).then(
        fn=get_document_list,
        outputs=[doc_list]
    )

# Launch configuration
if __name__ == "__main__":
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,  # Set to True for public URL
        show_error=True
    )
```

## Styling (Optional CSS)

```python
# Add custom CSS for professional look
css = """
#chatbot {
    border-radius: 10px;
    border: 1px solid #e0e0e0;
}
.message {
    padding: 10px;
    margin: 5px;
    border-radius: 5px;
}
"""

# Add to Blocks:
with gr.Blocks(title="SemanticScout", css=css) as app:
    ...
```

## Key Features for Demo Success

1. **Drag & Drop Upload**: Works instantly, shows progress
2. **Clear Status Messages**: ✓ for success, ❌ for errors
3. **Document List**: Always visible, shows what's loaded
4. **Clean Chat**: Focus on conversation, not UI complexity
5. **Error Handling**: Graceful messages, no crashes

## What We're NOT Building

- ❌ Complex search interface (search is through chat)
- ❌ Document preview/viewer
- ❌ User authentication
- ❌ Advanced filters
- ❌ Export functionality
- ❌ Settings/configuration

## Testing the Interface

```python
# Quick test script
def test_ui_flow():
    """Test basic UI flow works."""
    
    # 1. Upload a test PDF
    test_file = "samples/test.pdf"
    status = process_file(test_file)
    assert "Successfully processed" in status
    
    # 2. Ask a question
    response = chat_response("What is this document about?", [])
    assert len(response) > 0
    
    # 3. Clear documents
    clear_status = clear_all_documents()
    assert "cleared" in clear_status
```

## Demo Script

1. **Opening**: "Let me show you SemanticScout - upload any document and chat with it naturally"

2. **Upload**: Drag & drop a PDF - "Watch how quickly it processes this contract"

3. **First Question**: "What are the payment terms in this contract?"
   - Show how it finds specific information
   - Point out source attribution

4. **Follow-up**: "Are there any penalties for late payment?"
   - Demonstrate context understanding

5. **Multiple Docs**: Upload another document
   - "Now let's compare across documents"
   - "What are the differences in payment terms between the two contracts?"

## Common Demo Issues & Fixes

1. **Slow Upload**: Pre-process documents before demo
2. **API Errors**: Have offline fallback responses
3. **Empty Results**: Prepare documents with rich content
4. **UI Glitches**: Test on demo machine beforehand

## Success Metrics

- [ ] Zero errors during 15-minute demo
- [ ] Upload → Process → Chat in < 30 seconds
- [ ] Professional appearance
- [ ] Smooth interactions
- [ ] Clear value demonstration

Remember: Less is more. A simple interface that works perfectly beats a complex one with bugs.