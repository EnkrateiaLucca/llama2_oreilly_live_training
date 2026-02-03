# /// script
# requires-python = ">=3.12"
# dependencies = ["ollama", "pypdf2", "streamlit"]
# ///
"""
Agentic RAG Streamlit UI - Interactive PDF search with a ReAct agent.

Usage:
    uv run streamlit run agentic_rag_ui.py
"""

import glob
import os
import subprocess
from typing import Any

import ollama
import streamlit as st


# =============================================================================
# Tools
# =============================================================================

def find_pdf_files(directory: str = "./assets-resources/pdfs") -> str:
    """Find all PDF files in the specified directory."""
    try:
        pdf_files = glob.glob(f"{directory}/**/*.pdf", recursive=True)
        if not pdf_files:
            return f"No PDF files found in {directory}"
        return "\n".join(pdf_files)
    except Exception as e:
        return f"Error finding PDF files: {str(e)}"


def search_pdf(pdf_path: str, search_pattern: str, context_lines: int = 3) -> str:
    """Search for a pattern in a PDF by converting to text first."""
    try:
        output_dir = "text_files"
        os.makedirs(output_dir, exist_ok=True)

        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        text_file_path = os.path.join(output_dir, f"{base_name}.txt")

        subprocess.run(
            ["pdftotext", "-layout", pdf_path, text_file_path],
            capture_output=True,
            text=True,
            check=True
        )

        search_result = subprocess.run(
            ["grep", "-i", "-C", str(context_lines), search_pattern, text_file_path],
            capture_output=True,
            text=True
        )

        if search_result.returncode == 0:
            return f"Found matches for '{search_pattern}' in {pdf_path}:\n\n{search_result.stdout}"
        elif search_result.returncode == 1:
            return f"No matches found for '{search_pattern}' in {pdf_path}"
        else:
            return f"Error searching: {search_result.stderr}"

    except subprocess.CalledProcessError as e:
        return f"Error converting PDF: {e.stderr}"
    except FileNotFoundError:
        return "Error: pdftotext not found. Install poppler-utils (brew install poppler)"
    except Exception as e:
        return f"Error: {str(e)}"


def read_full_pdf(pdf_path: str) -> str:
    """Convert a PDF file to text and return the full content."""
    try:
        output_dir = "text_files"
        os.makedirs(output_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(pdf_path))[0]
        text_file_path = os.path.join(output_dir, f"{base_name}.txt")

        subprocess.run(
            ["pdftotext", "-layout", pdf_path, text_file_path],
            capture_output=True,
            text=True,
            check=True
        )

        with open(text_file_path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except subprocess.CalledProcessError as e:
        return f"Error converting PDF: {e.stderr}"
    except FileNotFoundError:
        return "Error: pdftotext not found. Install poppler-utils (brew install poppler)"
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# Tool Schemas for Ollama
# =============================================================================

TOOLS = [
    {
        'type': 'function',
        'function': {
            'name': 'find_pdf_files',
            'description': 'Finds all PDF files in the specified directory. Use this first to discover available PDFs.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'directory': {
                        'type': 'string',
                        'description': 'Directory path to search (default: ./assets-resources/pdfs)'
                    }
                },
                'required': []
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'search_pdf',
            'description': 'Search for a keyword or phrase in a PDF file. Returns matching excerpts with context.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'pdf_path': {
                        'type': 'string',
                        'description': 'Path to the PDF file to search'
                    },
                    'search_pattern': {
                        'type': 'string',
                        'description': 'The keyword or phrase to search for'
                    },
                    'context_lines': {
                        'type': 'integer',
                        'description': 'Lines of context around matches (default: 3)'
                    }
                },
                'required': ['pdf_path', 'search_pattern']
            }
        }
    },
    {
        'type': 'function',
        'function': {
            'name': 'read_full_pdf',
            'description': 'Read the full text content of a PDF file.',
            'parameters': {
                'type': 'object',
                'properties': {
                    'pdf_path': {
                        'type': 'string',
                        'description': 'Path to the PDF file'
                    }
                },
                'required': ['pdf_path']
            }
        }
    }
]

TOOLS_MAP = {
    'find_pdf_files': find_pdf_files,
    'search_pdf': search_pdf,
    'read_full_pdf': read_full_pdf,
}

SYSTEM_PROMPT = """You are a helpful assistant that can search through PDF documents to answer questions.

Use the available tools to:
1. First, find what PDFs are available using find_pdf_files(directory="./assets-resources/pdfs")
2. Search for relevant keywords in PDFs using search_pdf(pdf_path, search_pattern, context_lines=3)
3. Read the full text content of a PDF file using read_full_pdf(pdf_path) when needed.

Think carefully and use tools as needed. When you have enough information, provide a clear final answer."""


# =============================================================================
# Agent Logic
# =============================================================================

def execute_tool(tool_name: str, arguments: dict[str, Any]) -> str:
    """Execute a tool and return its result."""
    if tool_name not in TOOLS_MAP:
        return f"Error: Tool '{tool_name}' not found"

    try:
        result = TOOLS_MAP[tool_name](**arguments)
        return str(result)
    except Exception as e:
        return f"Error executing {tool_name}: {str(e)}"


def run_agent(user_query: str, model: str, max_turns: int, status_container):
    """Run the agent with streaming status updates."""
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': user_query}
    ]

    reasoning_steps = []

    for turn in range(max_turns):
        with status_container:
            st.write(f"**Turn {turn + 1}/{max_turns}**")

        response = ollama.chat(
            model=model,
            messages=messages,
            tools=TOOLS
        )

        assistant_message = response['message']

        if 'tool_calls' in assistant_message and assistant_message['tool_calls']:
            messages.append(assistant_message)

            for tool_call in assistant_message['tool_calls']:
                tool_name = tool_call['function']['name']
                arguments = tool_call['function']['arguments']

                step = {
                    'turn': turn + 1,
                    'tool': tool_name,
                    'arguments': arguments,
                }

                with status_container:
                    st.write(f"🔧 **Tool Call:** `{tool_name}`")
                    st.code(str(arguments), language="json")

                tool_result = execute_tool(tool_name, arguments)
                step['result'] = tool_result

                preview = tool_result[:300] + "..." if len(tool_result) > 300 else tool_result
                with status_container:
                    st.write("**Result:**")
                    st.code(preview, language="text")

                messages.append({'role': 'tool', 'content': tool_result})
                reasoning_steps.append(step)

        elif 'content' in assistant_message:
            return assistant_message['content'], reasoning_steps

    return "Max turns reached. Unable to complete the task.", reasoning_steps


# =============================================================================
# Streamlit UI
# =============================================================================

def main():
    st.set_page_config(
        page_title="Agentic RAG",
        page_icon="🔍",
        layout="wide"
    )

    st.title("🔍 Agentic RAG - PDF Search Agent")
    st.markdown("""
    Ask questions about the PDF documents. The agent will autonomously search
    and retrieve relevant information using a ReAct (Reasoning + Acting) loop.
    """)

    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")

        model = st.text_input(
            "Ollama Model",
            value="mistral-small3.2",
            help="The Ollama model to use for the agent"
        )

        max_turns = st.slider(
            "Max Turns",
            min_value=1,
            max_value=20,
            value=10,
            help="Maximum number of reasoning-action turns"
        )

        st.divider()

        st.header("📄 Available PDFs")
        pdfs = find_pdf_files()
        if "No PDF files found" not in pdfs:
            for pdf in pdfs.split("\n"):
                st.text(f"• {os.path.basename(pdf)}")
        else:
            st.warning(pdfs)

        st.divider()

        st.header("🛠️ Available Tools")
        st.markdown("""
        - **find_pdf_files**: Discover available PDFs
        - **search_pdf**: Search for keywords in a PDF
        - **read_full_pdf**: Read entire PDF content
        """)

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "reasoning" in message and message["reasoning"]:
                with st.expander("🔍 View Reasoning Steps"):
                    for step in message["reasoning"]:
                        st.write(f"**Turn {step['turn']}** - Tool: `{step['tool']}`")
                        st.code(str(step['arguments']), language="json")
                        preview = step['result'][:200] + "..." if len(step['result']) > 200 else step['result']
                        st.code(preview, language="text")
                        st.divider()

    # Chat input
    if prompt := st.chat_input("Ask a question about the PDFs..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            status_container = st.container()

            with st.spinner("Agent is thinking..."):
                answer, reasoning = run_agent(prompt, model, max_turns, status_container)

            # Clear the status and show final answer
            status_container.empty()
            st.markdown(answer)

            if reasoning:
                with st.expander("🔍 View Reasoning Steps"):
                    for step in reasoning:
                        st.write(f"**Turn {step['turn']}** - Tool: `{step['tool']}`")
                        st.code(str(step['arguments']), language="json")
                        preview = step['result'][:200] + "..." if len(step['result']) > 200 else step['result']
                        st.code(preview, language="text")
                        st.divider()

        # Add assistant message to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "reasoning": reasoning
        })


if __name__ == "__main__":
    main()
