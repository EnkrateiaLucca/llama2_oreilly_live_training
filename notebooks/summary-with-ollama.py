import ollama
from langchain_community.document_loaders import PyPDFLoader
import sys

MODEL = "llama3.1:8b"
SYSTEM_PROMPT = """
You are a summarization engine.
You take in documents and output bullet points
summaries of their contents.
"""
TEMP=0

def load_document(file_path: str, num_pages: int) -> str:
    """Loads the content of a document given its file path."""
    if file_path.endswith('.txt'):
        with open(file_path, 'r') as f:
            return f.read()
    elif file_path.endswith('.pdf'):
        docs = PyPDFLoader(file_path).load()
        docs_str =  "\n\n".join(doc.page_content for doc in docs[:num_pages])
        return docs_str
    

def summarize_bullet_points(prompt: str) -> str:
    """Summarizes a document with bullet points."""
    response = ollama.chat(model=MODEL,
                           messages=[
                               {"role": "system", "content": SYSTEM_PROMPT},
                               {"role": "user", "content": prompt}
                           ],)
    return response['message']['content']

file_contents = load_document(file_path=sys.argv[1], num_pages=3)

# print(load_document(file_path=sys.argv[1]))
# print(file_contents)
prompt = f"Summarize this document: \n\n {file_contents}" 
print(summarize_bullet_points(file_contents))