# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "gradio",
#     "llama-index-core",
#     "llama-index-llms-ollama",
#     "llama-index-embeddings-huggingface",
#     "llama-index-readers-file",
#     "pypdf",
#     "sentence-transformers",
# ]
# ///
"""
Local RAG Application with Gradio

A fully local RAG (Retrieval-Augmented Generation) pipeline using:
- Ollama for the LLM (runs locally, no API keys needed)
- HuggingFace Embeddings for document embedding
- LlamaIndex for orchestrating the RAG pipeline
- Gradio for the web interface

Run with: uv run local_rag_gradio.py
"""

import gradio as gr
from pathlib import Path
from typing import Generator

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Global state for the index and query engine
_current_index = None
_current_files_hash = None


def setup_models(model_name: str, temperature: float) -> None:
    """Configure LLM and embedding models."""
    Settings.llm = Ollama(
        model=model_name,
        request_timeout=120.0,
        temperature=temperature,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
    )


def get_files_hash(file_paths: list[str]) -> str:
    """Generate a hash for cache invalidation."""
    import hashlib
    content = "".join(sorted(file_paths))
    return hashlib.md5(content.encode()).hexdigest()


def load_and_index_documents(file_paths: list[str]) -> VectorStoreIndex:
    """Load documents and create vector index."""
    global _current_index, _current_files_hash

    files_hash = get_files_hash(file_paths)

    # Return cached index if files haven't changed
    if _current_index is not None and _current_files_hash == files_hash:
        return _current_index

    documents = SimpleDirectoryReader(input_files=file_paths).load_data()
    _current_index = VectorStoreIndex.from_documents(documents, show_progress=True)
    _current_files_hash = files_hash

    return _current_index


def format_sources(source_nodes) -> str:
    """Format source nodes as markdown."""
    if not source_nodes:
        return ""

    sources_text = "\n\n---\n**Sources:**\n"
    for i, node in enumerate(source_nodes, 1):
        score = node.score if hasattr(node, 'score') and node.score else 0.0
        text_preview = node.text[:300].replace("\n", " ") + "..."
        sources_text += f"\n**[{i}]** (score: {score:.3f})\n> {text_preview}\n"

    return sources_text


def chat_with_rag(
    message: dict,
    history: list,
    model_name: str,
    temperature: float,
    top_k: int,
    show_sources: bool,
) -> Generator[str, None, None]:
    """
    Process a chat message using RAG.

    Args:
        message: Dict with 'text' and 'files' keys
        history: List of previous messages
        model_name: Ollama model to use
        temperature: LLM temperature
        top_k: Number of chunks to retrieve
        show_sources: Whether to show source chunks
    """
    # Setup models with current settings
    setup_models(model_name, temperature)

    # Collect all uploaded files from history and current message
    all_files = []

    # Get files from history
    for msg in history:
        if msg["role"] == "user":
            content = msg.get("content", [])
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "file":
                        file_path = item.get("file", {}).get("path")
                        if file_path and file_path not in all_files:
                            all_files.append(file_path)

    # Get files from current message
    if message.get("files"):
        for file_path in message["files"]:
            if file_path not in all_files:
                all_files.append(file_path)

    # Check for default document if no files uploaded
    if not all_files:
        default_path = Path(__file__).parent / "assets-resources" / "attention_paper.pdf"
        if default_path.exists():
            all_files.append(str(default_path))
            yield "Using sample document (Attention paper)...\n\n"
        else:
            yield "Please upload a PDF document to get started."
            return

    # Get the query text
    query_text = message.get("text", "").strip()
    if not query_text:
        yield "Please enter a question about the document."
        return

    # Load documents and create index
    yield "Loading documents and creating index...\n\n"

    try:
        index = load_and_index_documents(all_files)
        query_engine = index.as_query_engine(similarity_top_k=top_k)

        # Query the index
        yield "Thinking...\n\n"
        response = query_engine.query(query_text)

        # Build response with optional sources
        result = str(response)

        if show_sources and hasattr(response, 'source_nodes'):
            result += format_sources(response.source_nodes)

        yield result

    except Exception as e:
        yield f"Error: {str(e)}\n\nMake sure Ollama is running with the {model_name} model pulled."


def create_demo() -> gr.Blocks:
    """Create and configure the Gradio interface."""

    with gr.Blocks(
        title="Local RAG Chat",
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(
            """
            # 📚 Local RAG Chat
            Ask questions about your PDF documents using a fully local AI pipeline.

            **Requirements:** [Ollama](https://ollama.ai) installed with model pulled (e.g., `ollama pull llama3.2`)
            """
        )

        with gr.Row():
            # Sidebar controls
            with gr.Column(scale=1):
                gr.Markdown("### ⚙️ Configuration")

                model_dropdown = gr.Dropdown(
                    choices=["llama3.2", "llama3.1", "mistral", "phi3", "gemma2"],
                    value="llama3.2",
                    label="Ollama Model",
                    info="Select the model for generation",
                )

                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.1,
                    step=0.1,
                    label="Temperature",
                    info="Lower = factual, Higher = creative",
                )

                top_k_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=3,
                    step=1,
                    label="Retrieved Chunks (top_k)",
                    info="Number of relevant chunks to retrieve",
                )

                show_sources_checkbox = gr.Checkbox(
                    value=True,
                    label="Show Sources",
                    info="Display retrieved source chunks",
                )

                gr.Markdown(
                    """
                    ---
                    ### 📄 Documents
                    Upload PDFs using the chat input below, or use the sample document.
                    """
                )

            # Main chat area
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    height=500,
                    placeholder="Upload a PDF and ask questions about it...",
                    type="messages",
                )

                chat_interface = gr.ChatInterface(
                    fn=chat_with_rag,
                    chatbot=chatbot,
                    multimodal=True,
                    textbox=gr.MultimodalTextbox(
                        file_types=[".pdf"],
                        placeholder="Upload a PDF and/or ask a question...",
                        sources=["upload"],
                    ),
                    additional_inputs=[
                        model_dropdown,
                        temperature_slider,
                        top_k_slider,
                        show_sources_checkbox,
                    ],
                    submit_btn="Send",
                    retry_btn="🔄 Retry",
                    undo_btn="↩️ Undo",
                    clear_btn="🗑️ Clear",
                )

    return demo


if __name__ == "__main__":
    demo = create_demo()
    demo.launch()
