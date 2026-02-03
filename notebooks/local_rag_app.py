# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "streamlit",
#     "llama-index-core",
#     "llama-index-llms-ollama",
#     "llama-index-embeddings-huggingface",
#     "llama-index-readers-file",
#     "pypdf",
#     "sentence-transformers",
# ]
# ///
"""
Local RAG Application with Streamlit

A fully local RAG (Retrieval-Augmented Generation) pipeline using:
- Ollama for the LLM (runs locally, no API keys needed)
- HuggingFace Embeddings for document embedding
- LlamaIndex for orchestrating the RAG pipeline

Run with: uv run streamlit run local_rag_app.py
"""

import streamlit as st
import tempfile
import os
from pathlib import Path

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# Page configuration
st.set_page_config(
    page_title="Local RAG Chat",
    page_icon="📚",
    layout="wide",
)

st.title("📚 Local RAG Chat")
st.markdown("Ask questions about your PDF documents using a fully local AI pipeline.")


# Sidebar for configuration
with st.sidebar:
    st.header("Configuration")

    # Model selection
    ollama_model = st.selectbox(
        "Ollama Model",
        options=["llama3.2", "llama3.1", "mistral", "phi3", "gemma2"],
        index=0,
        help="Select the Ollama model to use for generation",
    )

    # Temperature slider
    temperature = st.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.1,
        help="Lower = more factual, Higher = more creative",
    )

    # Number of chunks to retrieve
    top_k = st.slider(
        "Retrieved Chunks (top_k)",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of relevant chunks to retrieve for each query",
    )

    st.divider()

    # File uploader
    st.header("Document")
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a PDF document to query",
    )

    # Option to use default document
    use_default = st.checkbox(
        "Use sample document (Attention paper)",
        value=uploaded_file is None,
        disabled=uploaded_file is not None,
    )

    st.divider()
    st.markdown("""
    **Requirements:**
    - [Ollama](https://ollama.ai) installed
    - Model pulled: `ollama pull llama3.2`
    """)


@st.cache_resource
def setup_models(model_name: str, temp: float):
    """Configure LLM and embedding models."""
    Settings.llm = Ollama(
        model=model_name,
        request_timeout=120.0,
        temperature=temp,
    )
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5",
    )
    return True


@st.cache_resource
def create_index(_documents, doc_hash: str):
    """Create vector index from documents. Uses doc_hash for cache invalidation."""
    with st.spinner("Creating index... (embedding documents)"):
        index = VectorStoreIndex.from_documents(
            _documents,
            show_progress=True,
        )
    return index


def load_documents(file_path: str):
    """Load documents from a PDF file."""
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    return documents


def get_document_hash(file_content: bytes) -> str:
    """Generate a hash for cache invalidation."""
    import hashlib
    return hashlib.md5(file_content).hexdigest()


# Main application logic
def main():
    # Setup models
    setup_models(ollama_model, temperature)

    # Determine which document to use
    documents = None
    doc_hash = None

    if uploaded_file is not None:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(uploaded_file.getvalue())
            tmp_path = tmp.name

        doc_hash = get_document_hash(uploaded_file.getvalue())
        documents = load_documents(tmp_path)
        st.sidebar.success(f"Loaded {len(documents)} pages")

    elif use_default:
        default_path = Path(__file__).parent / "assets-resources" / "attention_paper.pdf"
        if default_path.exists():
            with open(default_path, "rb") as f:
                doc_hash = get_document_hash(f.read())
            documents = load_documents(str(default_path))
            st.sidebar.success(f"Loaded {len(documents)} pages from sample")
        else:
            st.error(f"Sample document not found at: {default_path}")
            return
    else:
        st.info("Please upload a PDF document or select the sample document.")
        return

    # Create index
    index = create_index(documents, doc_hash)

    # Create query engine
    query_engine = index.as_query_engine(similarity_top_k=top_k)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message:
                with st.expander("View Sources"):
                    for i, source in enumerate(message["sources"]):
                        st.markdown(f"**Source {i+1}** (score: {source['score']:.3f})")
                        st.text(source["text"][:500] + "...")
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about the document..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = query_engine.query(prompt)

            st.markdown(response.response)

            # Extract sources
            sources = []
            for node in response.source_nodes:
                sources.append({
                    "score": node.score,
                    "text": node.text,
                })

            # Show sources
            with st.expander("View Sources"):
                for i, source in enumerate(sources):
                    st.markdown(f"**Source {i+1}** (score: {source['score']:.3f})")
                    st.text(source["text"][:500] + "...")
                    st.divider()

        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": response.response,
            "sources": sources,
        })


if __name__ == "__main__":
    main()
