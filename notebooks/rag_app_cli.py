from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
import sys

MODEL_NAME = "llama3.2"
EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"

# Configure the LLM - Ollama runs locally on port 11434
Settings.llm = Ollama(
    model=MODEL_NAME,           # Use llama3.2 (3B params, good balance)
    request_timeout=120.0,       # Timeout for generation
    temperature=0.1,             # Low temperature for factual responses
)

# Configure the embedding model - runs locally via sentence-transformers
Settings.embed_model = HuggingFaceEmbedding(
    model_name=EMBEDDING_MODEL_NAME,  # 384-dim, ~130MB
)

print("LLM and Embedding model configured!")





# 1. **Load** - Read the PDF document

def load_pdf(pdf_path: str="./assets-resources/attention_paper.pdf"):
    """Load the PDF document"""
    # Load the PDF - using the attention paper as our example
    documents = SimpleDirectoryReader(
        input_files=[pdf_path]
    ).load_data()

    print(f"Loaded {len(documents)} pages from the PDF")
    
    return documents

# 2. **Chunk** - Split into manageable pieces
# 3. **Embed** - Convert chunks to vector representations
# 4. **Index** - Store vectors for efficient retrieval
def chunk_embed_index(documents):
    # Create the index - this embeds all chunks
    index = VectorStoreIndex.from_documents(
        documents,
        show_progress=True,  # Show embedding progress
    )

    print("\nIndex created successfully!") 
    
    return index

def create_query_engine_retrieval(index,k_param=3):
    query_engine = index.as_query_engine(
    similarity_top_k=k_param,  # Number of chunks to retrieve
)
    return query_engine

# 5. **Query** - Retrieve relevant chunks and generate answers
def query(query_engine, prompt: str) -> str:
    response = query_engine.query(prompt)
    
    return response.response

if __name__=="__main__":
    # takes input pdf path as command line argument
    if len(sys.argv) != 2:
        print("Usage: python rag_app_local_v1.py <pdf_path>")
        sys.exit(1)
    pdf_path = sys.argv[1]
    # load the pdf
    documents = load_pdf(pdf_path)
    # chunk the pdf
    index = chunk_embed_index(documents)
    # create the query engine
    query_engine = create_query_engine_retrieval(index)
    # query the pdf
    prompt = input("Enter your prompt or quit (q): ")
    if prompt == "q":
        sys.exit(0)
    while prompt != "q":
        try:
            response = query(query_engine, prompt)
            print(response)
            prompt = input("Enter your prompt or quit (q): ")
        except Exception as e:
            print(f"Error: {e}")
            prompt = input("Enter your prompt or quit (q): ")
    sys.exit(0)
