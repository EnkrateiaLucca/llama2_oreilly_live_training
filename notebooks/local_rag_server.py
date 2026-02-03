# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "fastapi",
#     "uvicorn",
#     "python-multipart",
#     "llama-index-core",
#     "llama-index-llms-ollama",
#     "llama-index-embeddings-huggingface",
#     "llama-index-readers-file",
#     "pypdf",
#     "sentence-transformers",
# ]
# ///
"""
Local RAG Application with FastAPI + HTML

A fully local RAG (Retrieval-Augmented Generation) pipeline using:
- Ollama for the LLM (runs locally, no API keys needed)
- HuggingFace Embeddings for document embedding
- LlamaIndex for orchestrating the RAG pipeline
- FastAPI for the backend server
- Vanilla HTML/CSS/JS for the frontend

Run with: uv run local_rag_server.py
Then open: http://localhost:8000
"""

import hashlib
import tempfile
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.huggingface import HuggingFaceEmbedding


# ============================================================================
# Application State
# ============================================================================

app = FastAPI(title="Local RAG Chat")

# Store sessions with their indexes
sessions: dict[str, dict] = {}


# ============================================================================
# RAG Functions
# ============================================================================

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


def get_file_hash(content: bytes) -> str:
    """Generate hash for cache invalidation."""
    return hashlib.md5(content).hexdigest()


def load_and_index_document(file_path: str) -> VectorStoreIndex:
    """Load document and create vector index."""
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    index = VectorStoreIndex.from_documents(documents, show_progress=True)
    return index


# ============================================================================
# API Endpoints
# ============================================================================

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a PDF document and create an index."""
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    content = await file.read()
    file_hash = get_file_hash(content)

    # Check if we already have this document indexed
    for session_id, session in sessions.items():
        if session.get("file_hash") == file_hash:
            return JSONResponse({
                "session_id": session_id,
                "message": f"Document already indexed ({session['page_count']} pages)",
                "filename": session["filename"],
            })

    # Save to temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    # Create index
    try:
        index = load_and_index_document(tmp_path)
        page_count = len(index.docstore.docs)

        session_id = str(uuid.uuid4())[:8]
        sessions[session_id] = {
            "index": index,
            "file_hash": file_hash,
            "filename": file.filename,
            "page_count": page_count,
            "messages": [],
        }

        return JSONResponse({
            "session_id": session_id,
            "message": f"Successfully indexed {page_count} pages",
            "filename": file.filename,
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error indexing document: {str(e)}")


@app.post("/api/use-sample")
async def use_sample_document():
    """Use the sample Attention paper document."""
    default_path = Path(__file__).parent / "assets-resources" / "attention_paper.pdf"

    if not default_path.exists():
        raise HTTPException(status_code=404, detail="Sample document not found")

    with open(default_path, "rb") as f:
        content = f.read()

    file_hash = get_file_hash(content)

    # Check if already indexed
    for session_id, session in sessions.items():
        if session.get("file_hash") == file_hash:
            return JSONResponse({
                "session_id": session_id,
                "message": f"Sample document already indexed ({session['page_count']} pages)",
                "filename": "attention_paper.pdf",
            })

    # Create index
    index = load_and_index_document(str(default_path))
    page_count = len(index.docstore.docs)

    session_id = str(uuid.uuid4())[:8]
    sessions[session_id] = {
        "index": index,
        "file_hash": file_hash,
        "filename": "attention_paper.pdf",
        "page_count": page_count,
        "messages": [],
    }

    return JSONResponse({
        "session_id": session_id,
        "message": f"Successfully indexed {page_count} pages",
        "filename": "attention_paper.pdf",
    })


@app.post("/api/query")
async def query_document(
    session_id: str = Form(...),
    query: str = Form(...),
    model: str = Form("llama3.2"),
    temperature: float = Form(0.1),
    top_k: int = Form(3),
):
    """Query the indexed document."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Please upload a document first.")

    session = sessions[session_id]

    # Setup models with current settings
    setup_models(model, temperature)

    try:
        query_engine = session["index"].as_query_engine(similarity_top_k=top_k)
        response = query_engine.query(query)

        # Extract sources
        sources = []
        if hasattr(response, 'source_nodes'):
            for node in response.source_nodes:
                sources.append({
                    "score": node.score if hasattr(node, 'score') and node.score else 0.0,
                    "text": node.text[:500] + "..." if len(node.text) > 500 else node.text,
                })

        # Store in message history
        session["messages"].append({"role": "user", "content": query})
        session["messages"].append({
            "role": "assistant",
            "content": str(response),
            "sources": sources,
        })

        return JSONResponse({
            "response": str(response),
            "sources": sources,
        })

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error querying document: {str(e)}. Make sure Ollama is running with the {model} model."
        )


@app.get("/api/history/{session_id}")
async def get_history(session_id: str):
    """Get chat history for a session."""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")

    return JSONResponse({
        "messages": sessions[session_id]["messages"],
        "filename": sessions[session_id]["filename"],
    })


# ============================================================================
# HTML Frontend
# ============================================================================

HTML_CONTENT = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Local RAG Chat</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            justify-content: center;
            padding: 20px;
        }

        .container {
            display: flex;
            gap: 20px;
            max-width: 1200px;
            width: 100%;
        }

        .sidebar {
            width: 280px;
            background: white;
            border-radius: 16px;
            padding: 24px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            height: fit-content;
        }

        .main {
            flex: 1;
            background: white;
            border-radius: 16px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            display: flex;
            flex-direction: column;
            max-height: 90vh;
        }

        h1 {
            color: #333;
            margin-bottom: 8px;
            font-size: 1.5rem;
        }

        .subtitle {
            color: #666;
            font-size: 0.9rem;
            margin-bottom: 24px;
        }

        .section-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: #444;
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        label {
            display: block;
            font-size: 0.85rem;
            color: #555;
            margin-bottom: 6px;
        }

        select, input[type="range"] {
            width: 100%;
            padding: 8px 12px;
            border: 1px solid #ddd;
            border-radius: 8px;
            font-size: 0.9rem;
            margin-bottom: 16px;
        }

        input[type="range"] {
            padding: 0;
        }

        .range-value {
            font-size: 0.85rem;
            color: #667eea;
            font-weight: 600;
        }

        .divider {
            height: 1px;
            background: #eee;
            margin: 20px 0;
        }

        .upload-area {
            border: 2px dashed #ddd;
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            cursor: pointer;
            transition: all 0.3s;
            margin-bottom: 12px;
        }

        .upload-area:hover {
            border-color: #667eea;
            background: #f8f9ff;
        }

        .upload-area.dragover {
            border-color: #667eea;
            background: #f0f3ff;
        }

        .upload-icon {
            font-size: 2rem;
            margin-bottom: 8px;
        }

        .upload-text {
            color: #666;
            font-size: 0.9rem;
        }

        .btn {
            width: 100%;
            padding: 10px 16px;
            border: none;
            border-radius: 8px;
            font-size: 0.9rem;
            cursor: pointer;
            transition: all 0.3s;
        }

        .btn-secondary {
            background: #f0f0f0;
            color: #333;
        }

        .btn-secondary:hover {
            background: #e0e0e0;
        }

        .status {
            margin-top: 16px;
            padding: 12px;
            border-radius: 8px;
            font-size: 0.85rem;
            display: none;
        }

        .status.success {
            display: block;
            background: #d4edda;
            color: #155724;
        }

        .status.error {
            display: block;
            background: #f8d7da;
            color: #721c24;
        }

        .status.loading {
            display: block;
            background: #fff3cd;
            color: #856404;
        }

        /* Chat Area */
        .chat-header {
            padding: 20px 24px;
            border-bottom: 1px solid #eee;
        }

        .chat-header h2 {
            color: #333;
            font-size: 1.25rem;
        }

        .chat-messages {
            flex: 1;
            overflow-y: auto;
            padding: 20px 24px;
        }

        .message {
            margin-bottom: 20px;
            animation: fadeIn 0.3s ease;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .message.user {
            display: flex;
            justify-content: flex-end;
        }

        .message-content {
            max-width: 80%;
            padding: 12px 16px;
            border-radius: 16px;
            line-height: 1.5;
        }

        .message.user .message-content {
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border-bottom-right-radius: 4px;
        }

        .message.assistant .message-content {
            background: #f5f5f5;
            color: #333;
            border-bottom-left-radius: 4px;
        }

        .sources-toggle {
            margin-top: 8px;
            font-size: 0.8rem;
            color: #667eea;
            cursor: pointer;
            display: inline-block;
        }

        .sources-toggle:hover {
            text-decoration: underline;
        }

        .sources {
            display: none;
            margin-top: 12px;
            padding: 12px;
            background: #fff;
            border-radius: 8px;
            font-size: 0.85rem;
            border: 1px solid #eee;
        }

        .sources.show {
            display: block;
        }

        .source-item {
            margin-bottom: 12px;
            padding-bottom: 12px;
            border-bottom: 1px solid #eee;
        }

        .source-item:last-child {
            margin-bottom: 0;
            padding-bottom: 0;
            border-bottom: none;
        }

        .source-score {
            font-weight: 600;
            color: #667eea;
            margin-bottom: 4px;
        }

        .source-text {
            color: #666;
            font-size: 0.8rem;
            line-height: 1.4;
        }

        .chat-input-area {
            padding: 20px 24px;
            border-top: 1px solid #eee;
        }

        .chat-input-container {
            display: flex;
            gap: 12px;
        }

        .chat-input {
            flex: 1;
            padding: 12px 16px;
            border: 2px solid #eee;
            border-radius: 24px;
            font-size: 0.95rem;
            outline: none;
            transition: border-color 0.3s;
        }

        .chat-input:focus {
            border-color: #667eea;
        }

        .chat-input:disabled {
            background: #f5f5f5;
            cursor: not-allowed;
        }

        .send-btn {
            padding: 12px 24px;
            background: linear-gradient(135deg, #667eea, #764ba2);
            color: white;
            border: none;
            border-radius: 24px;
            font-size: 0.95rem;
            cursor: pointer;
            transition: transform 0.2s, box-shadow 0.2s;
        }

        .send-btn:hover:not(:disabled) {
            transform: scale(1.05);
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .send-btn:disabled {
            opacity: 0.6;
            cursor: not-allowed;
        }

        .typing-indicator {
            display: none;
            padding: 12px 16px;
            background: #f5f5f5;
            border-radius: 16px;
            width: fit-content;
        }

        .typing-indicator.show {
            display: block;
        }

        .typing-dot {
            display: inline-block;
            width: 8px;
            height: 8px;
            background: #999;
            border-radius: 50%;
            margin-right: 4px;
            animation: typingBounce 1.4s infinite ease-in-out;
        }

        .typing-dot:nth-child(1) { animation-delay: 0s; }
        .typing-dot:nth-child(2) { animation-delay: 0.2s; }
        .typing-dot:nth-child(3) { animation-delay: 0.4s; }

        @keyframes typingBounce {
            0%, 60%, 100% { transform: translateY(0); }
            30% { transform: translateY(-8px); }
        }

        .empty-state {
            text-align: center;
            color: #999;
            padding: 40px;
        }

        .empty-state-icon {
            font-size: 3rem;
            margin-bottom: 16px;
        }

        @media (max-width: 768px) {
            .container {
                flex-direction: column;
            }
            .sidebar {
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="sidebar">
            <h1>📚 Local RAG Chat</h1>
            <p class="subtitle">Ask questions about your PDF documents</p>

            <div class="section-title">Configuration</div>

            <label for="model">Ollama Model</label>
            <select id="model">
                <option value="llama3.2" selected>llama3.2</option>
                <option value="llama3.1">llama3.1</option>
                <option value="mistral">mistral</option>
                <option value="phi3">phi3</option>
                <option value="gemma2">gemma2</option>
            </select>

            <label for="temperature">Temperature: <span class="range-value" id="temp-value">0.1</span></label>
            <input type="range" id="temperature" min="0" max="1" step="0.1" value="0.1">

            <label for="top_k">Retrieved Chunks: <span class="range-value" id="topk-value">3</span></label>
            <input type="range" id="top_k" min="1" max="10" step="1" value="3">

            <div class="divider"></div>

            <div class="section-title">Document</div>

            <div class="upload-area" id="upload-area">
                <div class="upload-icon">📄</div>
                <div class="upload-text">Drop PDF here or click to upload</div>
                <input type="file" id="file-input" accept=".pdf" hidden>
            </div>

            <button class="btn btn-secondary" id="sample-btn">
                Use Sample (Attention Paper)
            </button>

            <div class="status" id="status"></div>
        </div>

        <div class="main">
            <div class="chat-header">
                <h2>💬 Chat</h2>
            </div>

            <div class="chat-messages" id="messages">
                <div class="empty-state">
                    <div class="empty-state-icon">📝</div>
                    <p>Upload a PDF or use the sample document to start chatting</p>
                </div>
            </div>

            <div class="chat-input-area">
                <div class="chat-input-container">
                    <input type="text" class="chat-input" id="query-input"
                           placeholder="Ask a question about the document..."
                           disabled>
                    <button class="send-btn" id="send-btn" disabled>Send</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        // State
        let sessionId = null;

        // DOM Elements
        const uploadArea = document.getElementById('upload-area');
        const fileInput = document.getElementById('file-input');
        const sampleBtn = document.getElementById('sample-btn');
        const status = document.getElementById('status');
        const messages = document.getElementById('messages');
        const queryInput = document.getElementById('query-input');
        const sendBtn = document.getElementById('send-btn');
        const tempSlider = document.getElementById('temperature');
        const tempValue = document.getElementById('temp-value');
        const topkSlider = document.getElementById('top_k');
        const topkValue = document.getElementById('topk-value');

        // Event Listeners
        tempSlider.addEventListener('input', () => {
            tempValue.textContent = tempSlider.value;
        });

        topkSlider.addEventListener('input', () => {
            topkValue.textContent = topkSlider.value;
        });

        uploadArea.addEventListener('click', () => fileInput.click());

        uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            uploadArea.classList.add('dragover');
        });

        uploadArea.addEventListener('dragleave', () => {
            uploadArea.classList.remove('dragover');
        });

        uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            uploadArea.classList.remove('dragover');
            const file = e.dataTransfer.files[0];
            if (file && file.type === 'application/pdf') {
                uploadFile(file);
            }
        });

        fileInput.addEventListener('change', () => {
            if (fileInput.files[0]) {
                uploadFile(fileInput.files[0]);
            }
        });

        sampleBtn.addEventListener('click', useSampleDocument);

        sendBtn.addEventListener('click', sendQuery);

        queryInput.addEventListener('keypress', (e) => {
            if (e.key === 'Enter' && !sendBtn.disabled) {
                sendQuery();
            }
        });

        // Functions
        function setStatus(message, type) {
            status.textContent = message;
            status.className = 'status ' + type;
        }

        async function uploadFile(file) {
            setStatus('Uploading and indexing document...', 'loading');

            const formData = new FormData();
            formData.append('file', file);

            try {
                const response = await fetch('/api/upload', {
                    method: 'POST',
                    body: formData
                });

                const data = await response.json();

                if (response.ok) {
                    sessionId = data.session_id;
                    setStatus(`✓ ${data.message}`, 'success');
                    enableChat();
                    clearMessages();
                } else {
                    setStatus(`✗ ${data.detail}`, 'error');
                }
            } catch (error) {
                setStatus(`✗ Error: ${error.message}`, 'error');
            }
        }

        async function useSampleDocument() {
            setStatus('Loading sample document...', 'loading');

            try {
                const response = await fetch('/api/use-sample', {
                    method: 'POST'
                });

                const data = await response.json();

                if (response.ok) {
                    sessionId = data.session_id;
                    setStatus(`✓ ${data.message}`, 'success');
                    enableChat();
                    clearMessages();
                } else {
                    setStatus(`✗ ${data.detail}`, 'error');
                }
            } catch (error) {
                setStatus(`✗ Error: ${error.message}`, 'error');
            }
        }

        function enableChat() {
            queryInput.disabled = false;
            sendBtn.disabled = false;
            queryInput.focus();
        }

        function clearMessages() {
            messages.innerHTML = '';
        }

        function addMessage(role, content, sources = null) {
            const div = document.createElement('div');
            div.className = `message ${role}`;

            let html = `<div class="message-content">${escapeHtml(content)}`;

            if (sources && sources.length > 0) {
                const sourcesId = 'sources-' + Date.now();
                html += `<div class="sources-toggle" onclick="toggleSources('${sourcesId}')">📎 View Sources (${sources.length})</div>`;
                html += `<div class="sources" id="${sourcesId}">`;
                sources.forEach((source, i) => {
                    html += `
                        <div class="source-item">
                            <div class="source-score">Source ${i + 1} (score: ${source.score.toFixed(3)})</div>
                            <div class="source-text">${escapeHtml(source.text)}</div>
                        </div>
                    `;
                });
                html += '</div>';
            }

            html += '</div>';
            div.innerHTML = html;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        function addTypingIndicator() {
            const div = document.createElement('div');
            div.className = 'message assistant';
            div.id = 'typing';
            div.innerHTML = `
                <div class="typing-indicator show">
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                    <span class="typing-dot"></span>
                </div>
            `;
            messages.appendChild(div);
            messages.scrollTop = messages.scrollHeight;
        }

        function removeTypingIndicator() {
            const typing = document.getElementById('typing');
            if (typing) typing.remove();
        }

        async function sendQuery() {
            const query = queryInput.value.trim();
            if (!query || !sessionId) return;

            addMessage('user', query);
            queryInput.value = '';
            queryInput.disabled = true;
            sendBtn.disabled = true;

            addTypingIndicator();

            const formData = new FormData();
            formData.append('session_id', sessionId);
            formData.append('query', query);
            formData.append('model', document.getElementById('model').value);
            formData.append('temperature', document.getElementById('temperature').value);
            formData.append('top_k', document.getElementById('top_k').value);

            try {
                const response = await fetch('/api/query', {
                    method: 'POST',
                    body: formData
                });

                removeTypingIndicator();

                const data = await response.json();

                if (response.ok) {
                    addMessage('assistant', data.response, data.sources);
                } else {
                    addMessage('assistant', `Error: ${data.detail}`);
                }
            } catch (error) {
                removeTypingIndicator();
                addMessage('assistant', `Error: ${error.message}`);
            }

            queryInput.disabled = false;
            sendBtn.disabled = false;
            queryInput.focus();
        }

        function escapeHtml(text) {
            const div = document.createElement('div');
            div.textContent = text;
            return div.innerHTML;
        }

        window.toggleSources = function(id) {
            const sources = document.getElementById(id);
            sources.classList.toggle('show');
        };
    </script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main HTML page."""
    return HTML_CONTENT


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("\n📚 Local RAG Chat Server")
    print("=" * 40)
    print("Open http://localhost:8000 in your browser")
    print("=" * 40 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
