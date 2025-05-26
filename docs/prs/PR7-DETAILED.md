# PR7: Gradio Chat & Search Interface - Detailed Implementation Guide

## Overview
This PR implements the Gradio web interface with chat as the primary feature and search as secondary, following the UI guidelines for a professional appearance.

## Prerequisites
- PR2-6 completed (all backend functionality)
- Gradio installed
- UI_GUIDELINES.md reviewed

## File Structure
```
app.py                       # Main Gradio application
ui/
├── __init__.py
├── components/
│   ├── chat_interface.py    # Chat UI component
│   ├── document_manager.py  # Document upload/management
│   ├── search_interface.py  # Search UI component
│   └── visualization.py     # Visualization component
├── theme.py                 # Custom Gradio theme
├── callbacks.py             # Event handlers
└── utils.py                 # UI utilities
```

## Detailed Implementation

### 1. Custom Theme (`ui/theme.py`)

```python
import gradio as gr

def create_custom_theme():
    """Create professional theme based on UI guidelines."""
    
    return gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="gray",
        neutral_hue="gray",
        text_size="md",
        spacing_size="md",
        radius_size="md",
        font=gr.themes.GoogleFont("Inter")
    ).set(
        # Primary colors
        body_background_fill="*neutral_50",
        body_background_fill_dark="*neutral_900",
        
        # Chat interface colors
        chatbot_code_background_color="*neutral_100",
        chatbot_code_background_color_dark="*neutral_800",
        
        # Button styles
        button_primary_background_fill="*primary_500",
        button_primary_background_fill_hover="*primary_600",
        button_primary_text_color="white",
        
        # Input styles
        input_background_fill="white",
        input_border_color="*neutral_300",
        input_border_color_focus="*primary_500",
        
        # Panel styles
        panel_background_fill="white",
        panel_border_color="*neutral_200",
        
        # Shadow and elevation
        shadow_drop="0 2px 4px 0 rgb(0 0 0 / 0.05)",
        shadow_drop_lg="0 10px 15px -3px rgb(0 0 0 / 0.1)",
    )
```

### 2. Chat Interface (`ui/components/chat_interface.py`)

```python
import gradio as gr
from typing import List, Tuple, Optional
from core.chat_engine import ChatEngine
import logging

logger = logging.getLogger(__name__)

class ChatInterface:
    """Chat interface component."""
    
    def __init__(self, chat_engine: ChatEngine):
        self.chat_engine = chat_engine
    
    def create_interface(self):
        """Create the chat interface component."""
        
        with gr.Column(scale=2, elem_id="chat-column"):
            gr.Markdown("## 💬 Chat with your Documents")
            
            # Chatbot display
            chatbot = gr.Chatbot(
                label="Conversation",
                elem_id="chatbot",
                height=500,
                show_label=False,
                avatar_images=("🧑", "🤖")
            )
            
            # Chat input
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your message",
                    placeholder="Ask anything about your documents...",
                    lines=2,
                    scale=4,
                    elem_id="chat-input"
                )
                
                submit_btn = gr.Button(
                    "Send", 
                    variant="primary",
                    scale=1,
                    elem_id="send-button"
                )
            
            # Chat controls
            with gr.Row():
                clear_btn = gr.Button("🗑️ Clear Chat", size="sm")
                examples = gr.Examples(
                    examples=[
                        "What are the main topics in these documents?",
                        "Summarize the key findings",
                        "What does the document say about...",
                        "Compare the information across documents"
                    ],
                    inputs=msg_input,
                    label="Example questions"
                )
            
            # Status display
            status = gr.Markdown("", elem_id="chat-status")
            
            return chatbot, msg_input, submit_btn, clear_btn, status
    
    def handle_message(self, message: str, history: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], str]:
        """Handle chat message submission."""
        if not message.strip():
            return history, ""
        
        try:
            # Generate response
            response, sources = self.chat_engine.chat(message)
            
            # Update history
            history.append((message, response))
            
            return history, ""
            
        except Exception as e:
            logger.error(f"Chat error: {e}")
            error_msg = "I encountered an error processing your message. Please try again."
            history.append((message, error_msg))
            return history, ""
    
    def clear_chat(self) -> List:
        """Clear chat history."""
        self.chat_engine.clear_history()
        return []
```

### 3. Document Manager (`ui/components/document_manager.py`)

```python
import gradio as gr
from typing import List, Optional, Dict, Any
from pathlib import Path
import logging
from core.document_processor import DocumentProcessor
from core.embedder import EmbeddingService
from core.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DocumentManager:
    """Document upload and management interface."""
    
    def __init__(self, document_processor: DocumentProcessor,
                 embedding_service: EmbeddingService,
                 vector_store: VectorStore):
        self.document_processor = document_processor
        self.embedding_service = embedding_service
        self.vector_store = vector_store
    
    def create_interface(self):
        """Create document management interface."""
        
        with gr.Column(elem_id="document-column"):
            gr.Markdown("## 📄 Document Management")
            
            # Upload interface
            with gr.Box():
                file_upload = gr.File(
                    label="Upload Documents",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                    file_count="multiple",
                    elem_id="file-upload"
                )
                
                upload_btn = gr.Button(
                    "📤 Process Documents",
                    variant="primary",
                    elem_id="upload-button"
                )
            
            # Progress display
            progress = gr.Progress(elem_id="upload-progress")
            status = gr.Markdown("", elem_id="upload-status")
            
            # Document list
            with gr.Box():
                gr.Markdown("### 📚 Uploaded Documents")
                doc_list = gr.DataFrame(
                    headers=["Filename", "Type", "Chunks", "Status"],
                    datatype=["str", "str", "number", "str"],
                    elem_id="document-list",
                    interactive=False
                )
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", size="sm")
                    delete_btn = gr.Button("🗑️ Delete Selected", size="sm", variant="stop")
            
            # Stats display
            stats = gr.JSON(label="Storage Statistics", elem_id="storage-stats")
            
            return file_upload, upload_btn, progress, status, doc_list, refresh_btn, delete_btn, stats
    
    def process_documents(self, files: List[str], progress=gr.Progress()) -> str:
        """Process uploaded documents."""
        if not files:
            return "No files selected."
        
        results = []
        total_files = len(files)
        
        for i, file_path in enumerate(files):
            progress((i + 1) / total_files, f"Processing {Path(file_path).name}...")
            
            try:
                # Process document
                document, chunks = self.document_processor.process_document(file_path)
                
                # Generate embeddings
                embedded_chunks = self.embedding_service.embed_document(document, chunks)
                
                # Store in vector database
                self.vector_store.store_document(document, embedded_chunks)
                
                results.append(f"✅ {document.filename}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                results.append(f"❌ {Path(file_path).name}: {str(e)}")
        
        return "\n".join(results)
    
    def get_document_list(self) -> List[List[Any]]:
        """Get list of stored documents."""
        try:
            documents = self.vector_store.get_all_documents()
            
            return [
                [
                    doc['filename'],
                    doc['file_type'].upper(),
                    doc['chunk_count'],
                    "✅ Indexed"
                ]
                for doc in documents
            ]
        except Exception as e:
            logger.error(f"Failed to get documents: {e}")
            return []
    
    def get_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""
        try:
            return self.vector_store.get_stats()
        except Exception as e:
            logger.error(f"Failed to get stats: {e}")
            return {"error": str(e)}
```

### 4. Main Application (`app.py`)

```python
import gradio as gr
import logging
from pathlib import Path
from ui.theme import create_custom_theme
from ui.components.chat_interface import ChatInterface
from ui.components.document_manager import DocumentManager
from ui.components.search_interface import SearchInterface
from core.document_processor import DocumentProcessor
from core.embedder import EmbeddingService
from core.vector_store import VectorStore
from core.chat_engine import ChatEngine
from core.search_engine import SearchEngine
from config.settings import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_app():
    """Create and configure the Gradio application."""
    
    # Initialize services
    logger.info("Initializing services...")
    
    document_processor = DocumentProcessor()
    embedding_service = EmbeddingService()
    vector_store = VectorStore()
    chat_engine = ChatEngine(vector_store, embedding_service)
    search_engine = SearchEngine(vector_store, embedding_service)
    
    # Create UI components
    chat_interface = ChatInterface(chat_engine)
    doc_manager = DocumentManager(document_processor, embedding_service, vector_store)
    search_interface = SearchInterface(search_engine)
    
    # Create custom theme
    theme = create_custom_theme()
    
    # Build interface
    with gr.Blocks(
        title="SemanticScout - Chat with your Documents",
        theme=theme,
        css="""
        #chat-column { flex: 2; }
        #document-column { flex: 1; }
        #chatbot { height: 500px; }
        #chat-input { font-size: 16px; }
        .gradio-container { max-width: 1400px; margin: auto; }
        """
    ) as app:
        
        # Header
        gr.Markdown(
            """
            # 🔍 SemanticScout
            ### AI-Powered Document Chat & Search
            
            Upload your documents and start asking questions. I'll help you explore and understand your content.
            """,
            elem_id="header"
        )
        
        # Main layout
        with gr.Row():
            # Left: Chat interface (primary)
            chat_ui = chat_interface.create_interface()
            chatbot, msg_input, submit_btn, clear_btn, chat_status = chat_ui
            
            # Right: Document management + Search
            with gr.Column(scale=1):
                with gr.Tabs():
                    # Documents tab
                    with gr.TabItem("📄 Documents"):
                        doc_ui = doc_manager.create_interface()
                        (file_upload, upload_btn, progress, upload_status, 
                         doc_list, refresh_btn, delete_btn, stats) = doc_ui
                    
                    # Search tab
                    with gr.TabItem("🔍 Search"):
                        search_ui = search_interface.create_interface()
                        search_input, search_btn, search_results = search_ui
                    
                    # Visualization tab (placeholder)
                    with gr.TabItem("📊 Visualize"):
                        gr.Markdown("Document visualization coming soon...")
        
        # Event handlers
        
        # Chat events
        submit_btn.click(
            fn=chat_interface.handle_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
            show_progress=True
        )
        
        msg_input.submit(
            fn=chat_interface.handle_message,
            inputs=[msg_input, chatbot],
            outputs=[chatbot, msg_input],
            show_progress=True
        )
        
        clear_btn.click(
            fn=chat_interface.clear_chat,
            outputs=[chatbot]
        )
        
        # Document events
        upload_btn.click(
            fn=doc_manager.process_documents,
            inputs=[file_upload],
            outputs=[upload_status],
            show_progress=True
        ).then(
            fn=doc_manager.get_document_list,
            outputs=[doc_list]
        ).then(
            fn=doc_manager.get_stats,
            outputs=[stats]
        )
        
        refresh_btn.click(
            fn=doc_manager.get_document_list,
            outputs=[doc_list]
        )
        
        # Search events
        search_btn.click(
            fn=search_interface.search,
            inputs=[search_input],
            outputs=[search_results],
            show_progress=True
        )
        
        # Load initial data
        app.load(
            fn=doc_manager.get_document_list,
            outputs=[doc_list]
        )
        
        app.load(
            fn=doc_manager.get_stats,
            outputs=[stats]
        )
    
    return app

# Entry point
if __name__ == "__main__":
    logger.info("Starting SemanticScout...")
    
    app = create_app()
    
    app.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        favicon_path="assets/favicon.ico" if Path("assets/favicon.ico").exists() else None
    )
```

### 5. Search Interface (`ui/components/search_interface.py`)

```python
import gradio as gr
from typing import List, Dict, Any
from core.search_engine import SearchEngine
import logging

logger = logging.getLogger(__name__)

class SearchInterface:
    """Search interface component."""
    
    def __init__(self, search_engine: SearchEngine):
        self.search_engine = search_engine
    
    def create_interface(self):
        """Create search interface."""
        
        gr.Markdown("### 🔍 Semantic Search")
        
        search_input = gr.Textbox(
            label="Search Query",
            placeholder="Search your documents...",
            lines=2
        )
        
        with gr.Row():
            search_btn = gr.Button("Search", variant="primary")
            
            # Filters
            file_type_filter = gr.CheckboxGroup(
                choices=["PDF", "DOCX", "TXT", "MD"],
                label="File Types",
                value=None
            )
        
        # Results display
        search_results = gr.HTML(
            label="Search Results",
            elem_id="search-results"
        )
        
        return search_input, search_btn, search_results
    
    def search(self, query: str) -> str:
        """Execute search and format results."""
        if not query.strip():
            return "<p>Enter a search query.</p>"
        
        try:
            # Execute search
            response = self.search_engine.search(query, max_results=10)
            
            # Format results as HTML
            if not response.results:
                return "<p>No results found.</p>"
            
            html_parts = [f"<h4>Found {response.total_results} results in {response.search_time_ms:.0f}ms</h4>"]
            
            for i, result in enumerate(response.results, 1):
                source = result.source_info
                html_parts.append(f"""
                <div class="search-result">
                    <h5>{i}. {source['filename']} (Score: {result.score:.2%})</h5>
                    <p>{result.highlighted_content}</p>
                    <small>Chunk {source['chunk_index']}</small>
                </div>
                <hr>
                """)
            
            return "".join(html_parts)
            
        except Exception as e:
            logger.error(f"Search error: {e}")
            return f"<p>Search error: {str(e)}</p>"
```

## CSS Styling

```css
/* Custom CSS for professional appearance */
.gradio-container {
    font-family: 'Inter', sans-serif;
}

#header {
    text-align: center;
    padding: 2rem 0;
    border-bottom: 1px solid #e5e7eb;
}

#chatbot {
    border-radius: 8px;
    border: 1px solid #e5e7eb;
}

#chat-input {
    border-radius: 6px;
}

#send-button {
    min-width: 100px;
}

.search-result {
    padding: 1rem;
    margin-bottom: 1rem;
    background: #f9fafb;
    border-radius: 6px;
}

.search-result h5 {
    margin: 0 0 0.5rem 0;
    color: #1f2937;
}

.search-result p {
    margin: 0.5rem 0;
    color: #4b5563;
}

.search-result small {
    color: #9ca3af;
}
```

## Success Criteria

1. ✅ Chat interface is prominent and intuitive
2. ✅ Document upload with drag-and-drop works
3. ✅ Real-time processing feedback
4. ✅ Search complements chat functionality
5. ✅ Professional appearance matches UI guidelines
6. ✅ Responsive design works on different screens
7. ✅ All interactions provide immediate feedback
8. ✅ Error states handled gracefully