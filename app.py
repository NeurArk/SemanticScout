"""Gradio application for SemanticScout."""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any, Tuple
import time
from datetime import datetime
import numpy as np

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
from core.utils.adaptive_search import adaptive_analyzer


setup_logging()

logger = logging.getLogger(__name__)

doc_processor = DocumentProcessor()
embedder = EmbeddingService()
vector_store = VectorStore()
rag_pipeline = RAGPipeline(vector_store=vector_store)


uploaded_files: dict[str, dict[str, Any]] = {}

def sync_uploaded_files():
    """Synchronize uploaded_files with vector store on startup."""
    global uploaded_files
    try:
        docs = vector_store.get_all_documents()
        for doc in docs:
            filename = doc.get("filename", "Unknown")
            file_type = doc.get("file_type", "unknown")
            if not file_type or file_type == "Unknown":
                # Extract from filename if not stored
                file_type = Path(filename).suffix.lower().replace('.', '')
            
            uploaded_files[filename] = {
                "doc_id": doc.get("document_id", ""),
                "chunks": doc.get("chunk_count", 0),
                "file_size": doc.get("file_size", 0),
                "upload_time": datetime.now().isoformat(),  # Use current time for existing docs
                "file_type": file_type
            }
        logger.info(f"Synchronized {len(uploaded_files)} documents from vector store")
    except Exception as exc:
        logger.error(f"Failed to sync uploaded files: {exc}")

# We'll sync after the app is defined to ensure components can use the data

def format_relative_time(iso_time_str: str) -> str:
    """Convert ISO timestamp to relative time (e.g., '2 minutes ago')."""
    try:
        upload_time = datetime.fromisoformat(iso_time_str)
        now = datetime.now()
        diff = now - upload_time
        
        seconds = diff.total_seconds()
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            minutes = int(seconds / 60)
            return f"{minutes} min{'s' if minutes > 1 else ''} ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours} hour{'s' if hours > 1 else ''} ago"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days} day{'s' if days > 1 else ''} ago"
        else:
            return upload_time.strftime('%Y-%m-%d')
    except:
        return ""


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
        uploaded_files[filename] = {
            "doc_id": doc.id, 
            "chunks": len(chunks), 
            "file_size": doc.file_size,
            "upload_time": datetime.now().isoformat(),
            "file_type": Path(filename).suffix.lower().replace('.', '')
        }
        # Reset adaptive search cache after adding documents
        adaptive_analyzer.reset_cache()
        return f"✓ Successfully processed {filename} ({len(chunks)} chunks)\n{get_upload_status()}", None
    except Exception as exc:  # pragma: no cover - gradio will show error
        return f"❌ Error processing file: {exc}\n{get_upload_status()}", None

def get_upload_status() -> str:
    """Get comprehensive upload status with statistics."""
    if not uploaded_files:
        return "📚 No documents uploaded yet"
    
    total_docs = len(uploaded_files)
    total_chunks = sum(info['chunks'] for info in uploaded_files.values())
    total_size = sum(info.get('file_size', 0) for info in uploaded_files.values())
    total_size_mb = total_size / (1024 * 1024) if total_size > 0 else 0
    
    # Count by type
    type_counts = {}
    for info in uploaded_files.values():
        file_type = info.get('file_type', 'unknown').upper()
        type_counts[file_type] = type_counts.get(file_type, 0) + 1
    
    # Build status lines
    status_lines = []
    status_lines.append(f"📚 **{total_docs} document{'s' if total_docs > 1 else ''}** • {total_chunks} chunks • {total_size_mb:.1f} MB")
    
    if type_counts:
        type_parts = []
        for file_type, count in sorted(type_counts.items()):
            emoji = {
                'PDF': '📄',
                'DOCX': '📝', 
                'TXT': '📋',
                'MD': '📓'
            }.get(file_type, '📎')
            type_parts.append(f"{emoji} {file_type}: {count}")
        status_lines.append(" | ".join(type_parts))
    
    return "\n".join(status_lines)



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




def get_document_list_filtered(search_query: str = "") -> Tuple[str, list]:
    """Return filtered and formatted list of uploaded documents with choices for checkbox."""
    if not uploaded_files:
        return "No documents uploaded yet", []
    
    # Filter documents based on search query
    filtered_files = {}
    for filename, info in uploaded_files.items():
        if search_query.lower() in filename.lower():
            filtered_files[filename] = info
    
    total_filtered = len(filtered_files)
    
    if not filtered_files:
        return f"No documents matching '{search_query}'", []
    
    # Sort by upload time (newest first)
    sorted_files = sorted(
        filtered_files.items(),
        key=lambda x: x[1].get('upload_time', ''),
        reverse=True
    )
    
    # Build markdown lines and checkbox choices
    doc_lines = []
    checkbox_choices = []  # [(label, value), ...]
    
    # Show search results count if searching
    if search_query:
        doc_lines.append(f"**🔍 Showing {total_filtered} of {len(uploaded_files)} documents**\n")
    
    # Group by file type
    by_type = {}
    for filename, info in sorted_files:
        file_type = info.get('file_type', 'unknown').upper()
        if file_type not in by_type:
            by_type[file_type] = []
        by_type[file_type].append((filename, info))
    
    # Build document list with better formatting
    for file_type in sorted(by_type.keys()):
        emoji = {
            'PDF': '📄',
            'DOCX': '📝',
            'TXT': '📋',
            'MD': '📓'
        }.get(file_type, '📎')
        
        # Add type header
        doc_lines.append(f"\n**{emoji} {file_type} Files ({len(by_type[file_type])})**\n")
        
        for filename, info in by_type[file_type]:
            chunks = info['chunks']
            size_mb = info.get('file_size', 0) / (1024 * 1024)
            upload_time = info.get('upload_time', '')
            time_str = format_relative_time(upload_time) if upload_time else ""
            
            # Use code block for better alignment
            doc_lines.append(f"`{Path(filename).name}`")
            doc_lines.append(f"   {chunks} chunk{'s' if chunks != 1 else ''} · {size_mb:.2f} MB · {time_str}")
            doc_lines.append("")  # Empty line for spacing
            
            # Add to checkbox choices - format: (display_label, actual_filename)
            display_name = Path(filename).name
            if len(display_name) > 40:
                display_name = display_name[:37] + "..."
            label = f"{emoji} {display_name}"
            checkbox_choices.append((label, filename))
    
    return "\n".join(doc_lines).strip(), checkbox_choices


def delete_documents(filenames: list) -> tuple[str, str, str, list]:
    """Delete multiple documents from the vector store."""
    global uploaded_files
    
    if not filenames:
        doc_list, choices = get_document_list_filtered()
        return "Please select at least one document to delete", get_upload_status(), doc_list, choices
    
    deleted = []
    errors = []
    
    for filename in filenames:
        if filename not in uploaded_files:
            errors.append(f"{Path(filename).name}: not found")
            continue
            
        try:
            doc_id = uploaded_files[filename]['doc_id']
            vector_store.delete_document(doc_id)
            del uploaded_files[filename]
            deleted.append(Path(filename).name)
        except Exception as exc:
            errors.append(f"{Path(filename).name}: {exc}")
    
    adaptive_analyzer.reset_cache()
    doc_list, choices = get_document_list_filtered()
    
    # Build status message
    status_parts = []
    if deleted:
        status_parts.append(f"✓ Deleted {len(deleted)} document{'s' if len(deleted) > 1 else ''}: {', '.join(deleted)}")
    if errors:
        status_parts.append(f"❌ Errors: {'; '.join(errors)}")
    
    status = "\n".join(status_parts) if status_parts else "No documents deleted"
    return status, get_upload_status(), doc_list, choices


def delete_multiple_documents(filenames: List[str]) -> tuple[str, str, str, list]:
    """Delete multiple documents from the vector store."""
    global uploaded_files
    
    if not filenames:
        doc_list, choices = get_document_list_filtered()
        return "Please select documents to delete", get_upload_status(), doc_list, choices
    
    deleted_count = 0
    failed_count = 0
    messages = []
    
    for filename in filenames:
        if filename in uploaded_files:
            try:
                doc_id = uploaded_files[filename]['doc_id']
                vector_store.delete_document(doc_id)
                del uploaded_files[filename]
                deleted_count += 1
            except Exception as exc:
                failed_count += 1
                messages.append(f"Failed to delete {Path(filename).name}: {exc}")
    
    adaptive_analyzer.reset_cache()
    doc_list, choices = get_document_list_filtered()
    
    status_msg = f"✓ Deleted {deleted_count} document(s)"
    if failed_count > 0:
        status_msg += f", {failed_count} failed"
    if messages:
        status_msg += "\n" + "\n".join(messages[:3])  # Show first 3 error messages
    
    return status_msg, get_upload_status(), doc_list, choices


def clear_all_documents() -> str:
    """Remove all documents from the vector store."""
    import time
    
    global uploaded_files
    try:
        # Count documents before clearing
        doc_count = len(uploaded_files)
        
        # Clear the vector store
        vector_store.clear()
        
        # Clear uploaded files
        uploaded_files.clear()
        
        # Give ChromaDB time to clean up
        time.sleep(0.5)
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Reset adaptive search cache after clearing documents
        adaptive_analyzer.reset_cache()
        
        if doc_count > 0:
            return f"✓ Successfully cleared {doc_count} document{'s' if doc_count > 1 else ''}\\n{get_upload_status()}"
        else:
            return get_upload_status()
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
                hover_data={
                    "Filename": True,
                    "Chunks": False,  # Hide jittered value
                    "Size (MB)": False,  # Hide jittered value
                    "Original_Chunks": ":d",  # Show original value
                    "Original_Size": ":.3f"  # Show original value
                },
                labels={
                    "Original_Chunks": "Chunks",
                    "Original_Size": "Size (MB)"
                },
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
            # Create a counter for points at same coordinates
            coord_counts = {}
            
            for filename, info in uploaded_files.items():
                # Extract data with validation
                try:
                    chunks = int(info.get('chunks', 0))
                    file_size_bytes = float(info.get('file_size', 0))
                    size_mb = round(file_size_bytes / (1024.0 * 1024.0), 4)  # More precision
                    
                    # Skip if invalid data
                    if chunks <= 0:
                        logger.warning(f"Skipping {filename}: invalid data (chunks={chunks})")
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
                    
                    # Count occurrences at same coordinate
                    coord_key = (chunks, size_mb)
                    if coord_key not in coord_counts:
                        coord_counts[coord_key] = 0
                    else:
                        coord_counts[coord_key] += 1
                    
                    # Apply jitter for overlapping points
                    jitter_chunks = chunks
                    jitter_size = size_mb
                    
                    if coord_counts[coord_key] > 0:
                        # Add jitter: small random offset to prevent overlap
                        np.random.seed(hash(filename) % 2**32)  # Consistent jitter per file
                        jitter_chunks = chunks + np.random.uniform(-0.03, 0.03)
                        jitter_size = size_mb + np.random.uniform(-0.002, 0.002)
                        # Ensure no negative values
                        jitter_size = max(0, jitter_size)
                    
                    # Add to data list
                    data.append({
                        "Chunks": jitter_chunks,
                        "Size (MB)": jitter_size,
                        "Type": file_type,
                        "Filename": Path(filename).name,
                        "Original_Chunks": chunks,
                        "Original_Size": size_mb
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
                "Chunks": pd.Series(dtype='float64'),
                "Size (MB)": pd.Series(dtype='float64'),
                "Type": pd.Series(dtype='str'),
                "Filename": pd.Series(dtype='str'),
                "Original_Chunks": pd.Series(dtype='int64'),
                "Original_Size": pd.Series(dtype='float64')
            })
        
        logger.info(f"Scatter plot data: {len(df)} rows, columns: {df.columns.tolist()}")
        if len(df) > 0:
            logger.info(f"Data types: {df.dtypes.to_dict()}")
        
        return df
        
    except Exception as exc:
        logger.error(f"Failed to create scatter plot: {exc}", exc_info=True)
        # Return a valid empty dataframe with correct dtypes
        return pd.DataFrame({
            "Chunks": pd.Series(dtype='float64'),
            "Size (MB)": pd.Series(dtype='float64'),
            "Type": pd.Series(dtype='str'),
            "Filename": pd.Series(dtype='str'),
            "Original_Chunks": pd.Series(dtype='int64'),
            "Original_Size": pd.Series(dtype='float64')
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

/* Status textbox styling */
.status-box textarea {
    font-size: 14px;
    line-height: 1.6;
}

/* Document list styling */
.document-list {
    font-size: 14px;
    line-height: 1.6;
}

.document-list code {
    font-size: 13px;
    font-weight: 500;
}

/* Checkbox group styling */
.doc-checkboxes {
    max-height: 300px;
    overflow-y: auto;
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 4px;
    padding: 8px;
    margin: 8px 0;
}

.doc-checkboxes label {
    display: block;
    padding: 4px 0;
    cursor: pointer;
}

.doc-checkboxes label:hover {
    background-color: rgba(128, 128, 128, 0.1);
    border-radius: 3px;
}

/* Delete status styling */
.delete-status textarea {
    border: none;
    background: transparent;
    font-size: 14px;
    padding: 8px 0;
}

/* Search box inside accordion */
.search-in-accordion input {
    margin-bottom: 1em;
}

/* Accordion styling */
.accordion {
    margin-top: 0.5em;
}

/* Delete status message styling */
.delete-status textarea {
    border: none !important;
    background: transparent !important;
    padding: 0.5em 0 !important;
    resize: none !important;
}

.delete-status textarea:not(:empty) {
    color: #10b981;  /* Success green */
    font-weight: 500;
}

.delete-status textarea:not(:empty)[value*="❌"],
.delete-status textarea:not(:empty)[value*="Error"],
.delete-status textarea:not(:empty)[value*="failed"] {
    color: #ef4444;  /* Error red */
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

            upload_status = gr.Textbox(
                label="📊 Status", 
                interactive=False, 
                lines=3, 
                value=get_upload_status(),
                elem_classes=["status-box"]
            )
            
            # Document list in accordion with integrated search
            with gr.Accordion("📂 Document List", open=False, elem_classes=["accordion"]) as doc_accordion:
                # Search box inside accordion
                search_box = gr.Textbox(
                    label="",
                    placeholder="🔍 Search documents...",
                    container=False,
                    elem_classes=["search-in-accordion"]
                )
                
                doc_list_display = gr.Markdown(
                    get_document_list_filtered("")[0],
                    elem_classes=["document-list"]
                )
                
                # Delete section with checkboxes
                gr.Markdown("---")  # Separator
                gr.Markdown("**🗑 Delete Documents**")
                gr.Markdown("Select documents to delete:")
                
                # CheckboxGroup for document selection
                doc_checkboxes = gr.CheckboxGroup(
                    label="",
                    choices=get_document_list_filtered("")[1],
                    value=[],
                    elem_classes=["doc-checkboxes"],
                    container=False
                )
                
                with gr.Row():
                    delete_btn = gr.Button("🗑 Delete Selected", variant="stop", scale=2)
                    select_all_btn = gr.Button("✅ Select All", variant="secondary", scale=1)
                    clear_selection_btn = gr.Button("❌ Clear Selection", variant="secondary", scale=1)
                
                delete_status = gr.Textbox(
                    label="", 
                    visible=True, 
                    interactive=False,
                    show_label=False,
                    elem_classes=["delete-status"],
                    value=""
                )
            
            with gr.Row():
                refresh_btn = gr.Button("Refresh", size="sm")
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
    
    # Helper functions for checkbox selection
    def select_all_documents(current_choices):
        """Select all documents in the checkbox group."""
        return [choice[1] for choice in current_choices]  # Return all values
    
    def clear_selection():
        """Clear all selections."""
        return []

    
    # Update document search in real-time and auto-open accordion
    def search_and_update(query):
        doc_list, choices = get_document_list_filtered(query)
        # Return the document list, choices for checkboxes, and True to open accordion if searching
        return doc_list, gr.CheckboxGroup(choices=choices, value=[]), gr.Accordion(open=bool(query))
    
    search_box.change(
        fn=search_and_update,
        inputs=[search_box], 
        outputs=[doc_list_display, doc_checkboxes, doc_accordion]
    )
    
    # Handle document deletion (now supports multiple selection)
    delete_btn.click(
        fn=delete_documents,
        inputs=[doc_checkboxes],
        outputs=[delete_status, upload_status, doc_list_display, doc_checkboxes]
    )
    
    # Select all button - get choices from search state
    def select_all_from_search(search_query):
        _, choices = get_document_list_filtered(search_query)
        return [choice[1] for choice in choices]  # Return all values
    
    select_all_btn.click(
        fn=select_all_from_search,
        inputs=[search_box],
        outputs=[doc_checkboxes]
    )
    
    # Clear selection button  
    clear_selection_btn.click(
        fn=lambda: [],
        outputs=[doc_checkboxes]
    )
    
    file_upload.change(
        fn=process_file, inputs=[file_upload], outputs=[upload_status, file_upload]
    ).then(
        fn=lambda: get_document_list_filtered(""), 
        outputs=[doc_list_display, doc_checkboxes]
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

    refresh_btn.click(
        fn=get_upload_status, outputs=[upload_status]
    ).then(
        fn=lambda: get_document_list_filtered(""), 
        outputs=[doc_list_display, doc_checkboxes]
    ).then(
        fn=get_system_stats, outputs=[stats_display]
    ).then(
        fn=create_document_type_chart, outputs=[bar_chart]
    ).then(
        fn=create_plotly_scatter, outputs=[scatter_plot]
    )
    
    clear_docs_btn.click(fn=clear_all_documents, outputs=[upload_status]).then(
        fn=lambda: get_document_list_filtered(""), 
        outputs=[doc_list_display, doc_checkboxes]
    ).then(
        fn=lambda: "", outputs=[search_box]
    ).then(
        fn=get_system_stats, outputs=[stats_display]
    ).then(
        fn=create_document_type_chart, outputs=[bar_chart]
    ).then(
        fn=create_plotly_scatter, outputs=[scatter_plot]
    )
    
    # Initialize data and update UI after JS injection
    app.load(
        fn=lambda: (
            *get_document_list_filtered(""),
            get_system_stats(),
            create_document_type_chart(),
            create_plotly_scatter(),
            get_upload_status()
        ),
        outputs=[doc_list_display, doc_checkboxes, stats_display, bar_chart, scatter_plot, upload_status]
    )

if __name__ == "__main__":
    # Initialize data before launching
    initialize_app_data()
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)
