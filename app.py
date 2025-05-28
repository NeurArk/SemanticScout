"""Gradio application for SemanticScout."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import gradio as gr
import pandas as pd
import logging
import plotly.express as px

from config.logging import setup_logging
from core.document_processor import DocumentProcessor
from core.embedder import EmbeddingService
from core.vector_store import VectorStore
from core.rag_pipeline import RAGPipeline
from core.models.chat import ChatMessage


setup_logging()

logger = logging.getLogger(__name__)

doc_processor = DocumentProcessor()
embedder = EmbeddingService()
vector_store = VectorStore()
rag_pipeline = RAGPipeline()


uploaded_files: dict[str, dict[str, int]] = {}

def sync_uploaded_files():
    """Synchronize uploaded_files with vector store on startup."""
    global uploaded_files
    try:
        docs = vector_store.get_all_documents()
        for doc in docs:
            filename = doc.get("filename", "Unknown")
            uploaded_files[filename] = {
                "doc_id": doc.get("document_id", ""),
                "chunks": doc.get("chunk_count", 0),
                "file_size": doc.get("file_size", 0)  # Try to get file size
            }
        logger.info(f"Synchronized {len(uploaded_files)} documents from vector store")
    except Exception as exc:
        logger.error(f"Failed to sync uploaded files: {exc}")

# We'll sync after the app is defined to ensure components can use the data


def process_file(file: gr.FileData | None) -> tuple[str, gr.FileData | None]:
    """Process uploaded file and add it to the vector store."""

    if file is None:
        return get_upload_status(), None

    file_path = file.name
    filename = Path(file_path).name

    if filename in uploaded_files:
        return f"✓ {filename} already processed\n{get_upload_status()}", None

    try:
        doc, chunks = doc_processor.process_document(file_path)
        embedded = embedder.embed_document(doc, chunks)
        vector_store.store_document(doc, embedded)
        uploaded_files[filename] = {"doc_id": doc.id, "chunks": len(chunks), "file_size": doc.file_size}
        return f"✓ Successfully processed {filename} ({len(chunks)} chunks)\n{get_upload_status()}", None
    except Exception as exc:  # pragma: no cover - gradio will show error
        return f"❌ Error processing file: {exc}\n{get_upload_status()}", None

def get_upload_status() -> str:
    """Get current upload status."""
    if not uploaded_files:
        return "No files uploaded"
    return f"📚 {len(uploaded_files)} document{'s' if len(uploaded_files) > 1 else ''} uploaded"



def chat_response(message: str, history: List[Dict[str, Any]]) -> str:
    """Return chat response using the RAG pipeline."""

    chat_history: List[ChatMessage] = []
    for msg in history:
        if msg["role"] == "user":
            chat_history.append(ChatMessage(role="user", content=msg["content"]))
        elif msg["role"] == "assistant":
            chat_history.append(ChatMessage(role="assistant", content=msg["content"]))

    answer, _sources = rag_pipeline.query(message, chat_history)
    return answer


def get_document_list() -> str:
    """Return markdown list of uploaded documents."""

    if not uploaded_files:
        return "No documents uploaded yet"

    doc_lines = ["📄 **Uploaded Documents:**\n"]
    for filename, info in uploaded_files.items():
        doc_lines.append(f"• {filename} ({info['chunks']} chunks)")
    return "\n".join(doc_lines)


def clear_all_documents() -> str:
    """Remove all documents from the vector store."""
    import time
    
    global uploaded_files
    try:
        # Clear the vector store
        vector_store.clear()
        
        # Clear uploaded files
        uploaded_files.clear()
        
        # Give ChromaDB time to clean up
        time.sleep(0.5)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        return "✓ All documents cleared - Database reset successfully"
    except Exception as exc:  # pragma: no cover - gradio will show error
        return f"❌ Error clearing documents: {exc}"


def get_system_stats() -> str:
    """Get simple system statistics."""
    try:
        # Get stats from vector store
        stats = vector_store.get_statistics()
        
        # Also calculate from uploaded_files for accuracy
        total_docs = len(uploaded_files)
        total_chunks = sum(info['chunks'] for info in uploaded_files.values())
        
        # Count file types from uploaded_files
        pdf_count = sum(1 for f in uploaded_files if f.endswith('.pdf'))
        docx_count = sum(1 for f in uploaded_files if f.endswith('.docx'))
        txt_count = sum(1 for f in uploaded_files if f.endswith('.txt'))
        md_count = sum(1 for f in uploaded_files if f.endswith('.md'))
        
        # Use the maximum of both sources
        total_docs = max(stats.get('total_documents', 0), total_docs)
        total_chunks = max(stats.get('total_chunks', 0), total_chunks)

        return f"""
        📊 **System Statistics**

        • Documents: {total_docs}
        • Total Chunks: {total_chunks}
        • Vector Store Size: {stats.get('collection_size', 0)}

        **Document Types:**
        • PDF: {max(stats.get('pdf_count', 0), pdf_count)}
        • DOCX: {max(stats.get('docx_count', 0), docx_count)}
        • TXT: {max(stats.get('txt_count', 0), txt_count)}
        • MD: {md_count}
        """
    except Exception:  # pragma: no cover - simple fallback
        return "📊 Statistics unavailable"


def create_document_type_chart() -> pd.DataFrame:
    """Return data for bar chart of document types."""
    try:
        # Count from uploaded_files
        pdf_count = sum(1 for f in uploaded_files if f.endswith('.pdf'))
        docx_count = sum(1 for f in uploaded_files if f.endswith('.docx'))
        txt_count = sum(1 for f in uploaded_files if f.endswith('.txt'))
        md_count = sum(1 for f in uploaded_files if f.endswith('.md'))
        
        return pd.DataFrame(
            {
                "Type": ["PDF", "DOCX", "TXT", "MD"],
                "Count": [pdf_count, docx_count, txt_count, md_count],
            }
        )
    except Exception as exc:  # pragma: no cover - simple fallback
        logger.error("Failed to create bar chart: %s", exc)
        return pd.DataFrame({"Type": [], "Count": []})


def create_plotly_scatter():
    """Create a plotly scatter plot that adapts to theme."""
    try:
        df = create_document_scatter()
        
        # Theme-adaptive colors
        grid_color = 'rgba(128, 128, 128, 0.2)'  # Semi-transparent gray for grid
        
        if df.empty or len(df) == 0:
            # Create empty plot with message
            fig = px.scatter(
                pd.DataFrame({"x": [0], "y": [0], "text": ["No documents uploaded"]}),
                x="x", y="y", text="text",
                template="plotly"  # Default template for better theme adaptation
            )
            fig.update_layout(
                title="Document Size vs Chunk Count",
                xaxis_title="Chunks",
                yaxis_title="Size (MB)",
                showlegend=False,
                height=400,
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
                xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
                yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
            )
            fig.update_traces(marker=dict(size=20, color='rgba(128, 128, 128, 0.3)'))
            # Add annotation
            fig.add_annotation(
                x=0, y=0,
                text="Upload documents to see data",
                showarrow=False,
                font=dict(size=14)
            )
        else:
            fig = px.scatter(
                df, 
                x="Chunks", 
                y="Size (MB)", 
                color="Type",
                hover_data=["Filename"],
                title="Document Size vs Chunk Count",
                height=400,
                template="plotly",  # Default template for better theme adaptation
                color_discrete_sequence=px.colors.qualitative.Set2  # Good contrast in both themes
            )
            # Layout customization
            fig.update_layout(
                xaxis=dict(
                    title="Number of Chunks",
                    gridcolor=grid_color,
                    zerolinecolor=grid_color
                ),
                yaxis=dict(
                    title="File Size (MB)",
                    gridcolor=grid_color,
                    zerolinecolor=grid_color
                ),
                plot_bgcolor='rgba(0,0,0,0)',  # Transparent background
                paper_bgcolor='rgba(0,0,0,0)',  # Transparent paper
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1,
                    bgcolor='rgba(0,0,0,0)'  # Transparent legend background
                )
            )
            # Make points larger and more visible
            fig.update_traces(marker=dict(size=12, line=dict(width=1, color='rgba(128, 128, 128, 0.5)')))
        
        return fig
    except Exception as exc:
        logger.error(f"Failed to create plotly scatter: {exc}", exc_info=True)
        # Return error figure
        grid_color = 'rgba(128, 128, 128, 0.2)'
        fig = px.scatter(
            pd.DataFrame({"x": [0], "y": [0]}),
            x="x", y="y",
            title="Error creating plot",
            template="plotly"
        )
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color),
            yaxis=dict(gridcolor=grid_color, zerolinecolor=grid_color)
        )
        fig.add_annotation(
            x=0, y=0,
            text=f"Error: {str(exc)}",
            showarrow=False,
            font=dict(size=12, color="rgba(255, 0, 0, 0.8)")
        )
        return fig


def create_document_scatter() -> pd.DataFrame:
    """Return data for scatter plot of file size vs chunk count."""
    try:
        # Always create a fresh dataframe
        data = []
        
        if uploaded_files:
            for filename, info in uploaded_files.items():
                # Extract data with validation
                try:
                    chunks = int(info.get('chunks', 0))
                    file_size_bytes = float(info.get('file_size', 0))
                    size_mb = round(file_size_bytes / (1024.0 * 1024.0), 2)
                    
                    # Skip if invalid data
                    if chunks <= 0 or size_mb < 0:
                        logger.warning(f"Skipping {filename}: invalid data (chunks={chunks}, size={size_mb})")
                        continue
                    
                    # Determine file type from extension
                    file_ext = Path(filename).suffix.lower()
                    if file_ext == '.pdf':
                        file_type = 'PDF'
                    elif file_ext == '.docx':
                        file_type = 'DOCX'
                    elif file_ext == '.txt':
                        file_type = 'TXT'
                    elif file_ext == '.md':
                        file_type = 'MD'
                    else:
                        file_type = 'Other'
                    
                    # Add to data list
                    data.append({
                        "Chunks": chunks,
                        "Size (MB)": size_mb,
                        "Type": file_type,
                        "Filename": Path(filename).name
                    })
                except (ValueError, TypeError) as e:
                    logger.warning(f"Error processing {filename}: {e}")
                    continue
        
        # Create dataframe from list of dicts
        if data:
            df = pd.DataFrame(data)
        else:
            # Return empty dataframe with correct structure
            df = pd.DataFrame({
                "Chunks": pd.Series(dtype='int64'),
                "Size (MB)": pd.Series(dtype='float64'),
                "Type": pd.Series(dtype='str'),
                "Filename": pd.Series(dtype='str')
            })
        
        logger.info(f"Scatter plot data: {len(df)} rows, columns: {df.columns.tolist()}")
        if len(df) > 0:
            logger.info(f"Data types: {df.dtypes.to_dict()}")
        
        return df
        
    except Exception as exc:
        logger.error(f"Failed to create scatter plot: {exc}", exc_info=True)
        # Return a valid empty dataframe with correct dtypes
        return pd.DataFrame({
            "Chunks": pd.Series(dtype='int64'),
            "Size (MB)": pd.Series(dtype='float64'),
            "Type": pd.Series(dtype='str'),
            "Filename": pd.Series(dtype='str')
        })


def initialize_app_data():
    """Initialize app data by syncing with vector store."""
    try:
        sync_uploaded_files()
        logger.info("App data initialized successfully")
    except Exception as e:
        logger.warning(f"Failed to initialize app data: {e}")
        uploaded_files.clear()

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

/* Plotly theme adaptation for dark mode */
.dark .js-plotly-plot .plotly .text,
.dark .js-plotly-plot .plotly .xtitle,
.dark .js-plotly-plot .plotly .ytitle,
.dark .js-plotly-plot .plotly .gtitle,
.dark .js-plotly-plot .plotly .xtick text,
.dark .js-plotly-plot .plotly .ytick text,
.dark .js-plotly-plot .plotly .legendtext,
.dark .js-plotly-plot .plotly .annotation-text,
.dark .js-plotly-plot .plotly .g-gtitle text,
.dark .js-plotly-plot .plotly .g-xtitle text,
.dark .js-plotly-plot .plotly .g-ytitle text,
.dark .js-plotly-plot .plotly .legendtitletext,
.dark .js-plotly-plot .plotly .g-legendtitle text {
    fill: #e5e7eb !important;
}

/* Plotly theme adaptation for light mode - ensure default colors */
.js-plotly-plot .plotly .text,
.js-plotly-plot .plotly .xtitle,
.js-plotly-plot .plotly .ytitle,
.js-plotly-plot .plotly .gtitle,
.js-plotly-plot .plotly .xtick text,
.js-plotly-plot .plotly .ytick text,
.js-plotly-plot .plotly .legendtext,
.js-plotly-plot .plotly .annotation-text,
.js-plotly-plot .plotly .g-gtitle text,
.js-plotly-plot .plotly .g-xtitle text,
.js-plotly-plot .plotly .g-ytitle text,
.js-plotly-plot .plotly .legendtitletext,
.js-plotly-plot .plotly .g-legendtitle text {
    fill: #2e2e2e;
}
"""


with gr.Blocks(title="SemanticScout - Chat with your Documents", css=css) as app:
    gr.Markdown(
        """
        # 🔍 SemanticScout
        ### Chat naturally with your documents using AI

        Upload PDFs, Word docs (.docx), text files (.txt), or Markdown files (.md) and ask questions about their content.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                height=500, 
                show_label=False, 
                elem_id="chatbot",
                type="messages",
                show_copy_button=True
            )

            msg = gr.Textbox(
                label="Ask a question about your documents",
                placeholder=(
                    "e.g., What are the main findings? What does the contract say about termination?"
                ),
                lines=1,
            )

            with gr.Row():
                submit = gr.Button("Send", variant="primary")
                clear = gr.Button("Clear Chat")

        with gr.Column(scale=1):
            gr.Markdown("### 📁 Document Management")

            file_upload = gr.File(
                label="Upload Document",
                file_types=[".pdf", ".docx", ".txt", ".md"],
                type="filepath",
            )

            upload_status = gr.Textbox(label="Status", interactive=False, lines=2, value=get_upload_status())

            doc_list = gr.Markdown(get_document_list())

            refresh_btn = gr.Button("Refresh List", size="sm")
            clear_docs_btn = gr.Button("Clear All Documents", variant="stop", size="sm")

    with gr.Tab("Analytics"):
        stats_display = gr.Markdown(get_system_stats())
        # Initialize with empty dataframe - will be populated by app.load()
        bar_chart = gr.BarPlot(
            value=pd.DataFrame({"Type": [], "Count": []}),
            x="Type",
            y="Count",
            title="Documents by Type",
        )
        # Use gr.Plot instead of gr.ScatterPlot for better compatibility
        scatter_plot = gr.Plot(
            label="Document Size vs Chunk Count",
            show_label=True
        )
        refresh_stats = gr.Button("Refresh Stats")

        refresh_stats.click(fn=get_system_stats, outputs=[stats_display])
        refresh_stats.click(fn=create_document_type_chart, outputs=[bar_chart])
        refresh_stats.click(fn=create_plotly_scatter, outputs=[scatter_plot])

    def respond(
        user_message: str, chat_history: List[Dict[str, Any]]
    ) -> tuple[str, List[Dict[str, Any]]]:
        bot_message = chat_response(user_message, chat_history)
        chat_history.append({"role": "user", "content": user_message})
        chat_history.append({"role": "assistant", "content": bot_message})
        return "", chat_history

    file_upload.change(
        fn=process_file, inputs=[file_upload], outputs=[upload_status, file_upload]
    ).then(
        fn=get_document_list, outputs=[doc_list]
    ).then(
        fn=get_system_stats, outputs=[stats_display]
    ).then(
        fn=create_document_type_chart, outputs=[bar_chart]
    ).then(
        fn=create_plotly_scatter, outputs=[scatter_plot]
    )

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    refresh_btn.click(fn=get_document_list, outputs=[doc_list]).then(
        fn=get_system_stats, outputs=[stats_display]
    ).then(
        fn=create_document_type_chart, outputs=[bar_chart]
    ).then(
        fn=create_plotly_scatter, outputs=[scatter_plot]
    )
    
    clear_docs_btn.click(fn=clear_all_documents, outputs=[upload_status]).then(
        fn=get_document_list, outputs=[doc_list]
    ).then(
        fn=get_system_stats, outputs=[stats_display]
    ).then(
        fn=create_document_type_chart, outputs=[bar_chart]
    ).then(
        fn=create_plotly_scatter, outputs=[scatter_plot]
    )
    
    # Initialize data and update UI on app load
    app.load(
        fn=lambda: (
            get_document_list(),
            get_system_stats(),
            create_document_type_chart(),
            create_plotly_scatter(),
            get_upload_status()
        ),
        outputs=[doc_list, stats_display, bar_chart, scatter_plot, upload_status]
    )

if __name__ == "__main__":
    # Initialize data before launching
    initialize_app_data()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
