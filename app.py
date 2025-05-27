"""Gradio application for SemanticScout."""

from __future__ import annotations

from pathlib import Path
from typing import List

import gradio as gr

from config.logging import setup_logging
from core.document_processor import DocumentProcessor
from core.embedder import EmbeddingService
from core.vector_store import VectorStore
from core.rag_pipeline import RAGPipeline
from core.models.chat import ChatMessage


setup_logging()


doc_processor = DocumentProcessor()
embedder = EmbeddingService()
vector_store = VectorStore()
rag_pipeline = RAGPipeline()


uploaded_files: dict[str, dict[str, int]] = {}


def process_file(file: gr.FileData | None) -> str:
    """Process uploaded file and add it to the vector store."""

    if file is None:
        return "No file uploaded"

    file_path = file.name
    filename = Path(file_path).name

    if filename in uploaded_files:
        return f"✓ {filename} already processed"

    try:
        doc, chunks = doc_processor.process_document(file_path)
        embedded = embedder.embed_document(doc, chunks)
        vector_store.store_document(doc, embedded)
        uploaded_files[filename] = {"doc_id": doc.id, "chunks": len(chunks)}
        return f"✓ Successfully processed {filename} ({len(chunks)} chunks)"
    except Exception as exc:  # pragma: no cover - gradio will show error
        return f"❌ Error processing file: {exc}"


def chat_response(message: str, history: List[List[str]]) -> str:
    """Return chat response using the RAG pipeline."""

    chat_history: List[ChatMessage] = []
    for user_msg, assistant_msg in history:
        chat_history.append(ChatMessage(role="user", content=user_msg))
        chat_history.append(ChatMessage(role="assistant", content=assistant_msg))

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

    global uploaded_files
    try:
        vector_store.clear()
        uploaded_files = {}
        return "✓ All documents cleared"
    except Exception as exc:  # pragma: no cover - gradio will show error
        return f"❌ Error clearing documents: {exc}"


def get_system_stats() -> str:
    """Get simple system statistics."""
    try:
        stats = vector_store.get_statistics()

        return f"""
        📊 **System Statistics**

        • Documents: {stats.get('total_documents', 0)}
        • Total Chunks: {stats.get('total_chunks', 0)}
        • Vector Store Size: {stats.get('collection_size', 0)}

        **Document Types:**
        • PDF: {stats.get('pdf_count', 0)}
        • DOCX: {stats.get('docx_count', 0)}
        • TXT: {stats.get('txt_count', 0)}
        """
    except Exception:  # pragma: no cover - simple fallback
        return "📊 Statistics unavailable"


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


with gr.Blocks(title="SemanticScout - Chat with your Documents", css=css) as app:
    gr.Markdown(
        """
        # 🔍 SemanticScout
        ### Chat naturally with your documents using AI

        Upload PDFs, Word docs, or text files and ask questions about their content.
        """
    )

    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(height=500, show_label=False, elem_id="chatbot")

            msg = gr.Textbox(
                label="Ask a question about your documents",
                placeholder=(
                    "e.g., What are the main findings? What does the contract say about termination?"
                ),
                lines=2,
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

            upload_status = gr.Textbox(label="Status", interactive=False, lines=2)

            doc_list = gr.Markdown(get_document_list())

            refresh_btn = gr.Button("Refresh List", size="sm")
            clear_docs_btn = gr.Button("Clear All Documents", variant="stop", size="sm")

    with gr.Tab("Analytics"):
        stats_display = gr.Markdown(get_system_stats())
        refresh_stats = gr.Button("Refresh Stats")

        refresh_stats.click(
            fn=get_system_stats,
            outputs=[stats_display]
        )

    def respond(user_message: str, chat_history: List[List[str]]) -> tuple[str, List[List[str]]]:
        bot_message = chat_response(user_message, chat_history)
        chat_history.append([user_message, bot_message])
        return "", chat_history

    file_upload.change(fn=process_file, inputs=[file_upload], outputs=[upload_status]).then(
        fn=get_document_list, outputs=[doc_list]
    )

    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    submit.click(respond, [msg, chatbot], [msg, chatbot])
    clear.click(lambda: None, None, chatbot, queue=False)

    refresh_btn.click(fn=get_document_list, outputs=[doc_list])
    clear_docs_btn.click(fn=clear_all_documents, outputs=[upload_status]).then(
        fn=get_document_list, outputs=[doc_list]
    )

if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860, share=False, show_error=True)

