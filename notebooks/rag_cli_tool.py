# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "llama-index>=0.14.25",
#     "llama-index-core>=0.14.25",
#     "llama-index-embeddings-huggingface>=0.8.0",
#     "llama-index-llms-ollama>=0.11.0",
#     "llama-index-readers-file",
#     "pypdf>=6.19.0",
#     "sentence-transformers>=6.1.0",
# ]
# ///

from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
import sys
import argparse

MODEL_NAME = "gemma4"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

Settings.llm = Ollama(
    model=MODEL_NAME,           # Use gemma4 (course default, = gemma4:e4b)
    request_timeout=120.0,       # Timeout for generation
    temperature=0.1,             # Low temperature for factual responses
)

# COnfigure the embedding model - runs locally via sentence-transformers
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,  # 384-dim, ~130MB
)

print("LLM and Embedding model configured!")

def load_files(folder_path: str, exts: list[str] | None = None):
    """Load every supported file in a folder into LlamaIndex Documents.

    Pattern from the LlamaIndex SimpleDirectoryReader docs:
    https://developers.llamaindex.ai/python/framework/module_guides/loading/simpledirectoryreader/
    """
    reader = SimpleDirectoryReader(
        input_dir=folder_path,  # Folder to scan
    )
    # The reader picks a parser per extension (.pdf → pypdf, .docx, .md, .txt, ...).
    # Each Document carries metadata (file_name, file_path, page_label, ...) used later for citations.
    documents = reader.load_data()

    # PDFs become one Document per page, so this counts pages, not files.
    print(f"Loaded {len(documents)} documents from {len(reader.input_files)} files")
    return documents

def chunk_embed_index(documents: list):
    """Chunk, embed, and index the documents"""
    # Create the index - this embeds all chunks
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,  # Show embedding progress
    )
    # Optional persistence - uncomment to save the index to disk for later use
    # index.storage_context.persist(persist_dir="./storage/attention_paper")

    print("\nIndex created successfully!") 
    
    return index

def create_query_engine_retrieval(index,k_param=3):
    query_engine = index.as_query_engine(
    similarity_top_k=k_param,  # Number of chunks to retrieve
)
    return query_engine


def query(query_engine, prompt: str) -> str:
    """Query the index and return the answer"""
    response = query_engine.query(prompt)
    return response.response



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG CLI Tool")
    parser.add_argument("query", type=str, help="Query to ask the model")
    parser.add_argument("--folder", type=str, required=True, help="Path to the folder containing documents")
    # loading the documents from the folder
    folder_path = parser.parse_args().folder
    docs = load_files(folder_path)
    print(docs)
    # creating the vector store with the embedded documents
    index = chunk_embed_index(docs)
    # Creating the query engine for retrieval
    query_engine = create_query_engine_retrieval(index)
    prompt = parser.parse_args().query
    answer = query(query_engine, prompt)
    print("Answer:")
    print(answer)    


