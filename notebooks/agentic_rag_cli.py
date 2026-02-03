# /// script
# requires-python = ">=3.12"
# dependencies = ["ollama", "pypdf2"]
# ///
"""
Agentic RAG CLI - A simple ReAct agent for searching PDFs.

Usage:
    uv run agentic_rag_cli.py "What is LoRA?"
    uv run agentic_rag_cli.py "Summarize the available PDFs" --model qwen3
    uv run agentic_rag_cli.py "Search for fine-tuning" --max-turns 5 --quiet
"""

import argparse
import glob
import os
import subprocess
from typing import Any
import ollama


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


# =============================================================================
# Agent
# =============================================================================

SYSTEM_PROMPT = """You are a helpful assistant that can search through PDF documents to answer questions.

Use the available tools to:
1. First, find what PDFs are available using find_pdf_files(directory="./assets-resources/pdfs")
2. Search for relevant keywords in PDFs using search_pdf(pdf_path, search_pattern, context_lines=3)
3. Read the full text content of a PDF file using read_full_pdf(pdf_path) when needed.

Think carefully and use tools as needed. When you have enough information, provide a clear final answer."""


class SimpleAgent:
    """A simple ReAct agent for PDF search."""

    def __init__(self, model: str = "mistral-small3.2", max_turns: int = 10, verbose: bool = True):
        self.model = model
        self.max_turns = max_turns
        self.verbose = verbose
        self.tools_map = {
            'find_pdf_files': find_pdf_files,
            'search_pdf': search_pdf,
            'read_full_pdf': read_full_pdf,
        }

    def execute_tool(self, tool_name: str, arguments: dict[str, Any]) -> str:
        """Execute a tool and return its result."""
        if tool_name not in self.tools_map:
            return f"Error: Tool '{tool_name}' not found"

        try:
            result = self.tools_map[tool_name](**arguments)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {str(e)}"

    def run(self, user_query: str) -> str:
        """Run the agent with a user query using the ReAct loop."""
        messages = [
            {'role': 'system', 'content': SYSTEM_PROMPT},
            {'role': 'user', 'content': user_query}
        ]

        if self.verbose:
            print(f"\n{'='*60}")
            print(f"User Query: {user_query}")
            print(f"{'='*60}\n")

        for turn in range(self.max_turns):
            if self.verbose:
                print(f"\n--- Turn {turn + 1}/{self.max_turns} ---")

            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=TOOLS
            )

            assistant_message = response['message']

            if 'tool_calls' in assistant_message and assistant_message['tool_calls']:
                messages.append(assistant_message)

                for tool_call in assistant_message['tool_calls']:
                    tool_name = tool_call['function']['name']
                    arguments = tool_call['function']['arguments']

                    if self.verbose:
                        print(f"\n🔧 Tool Call: {tool_name}")
                        print(f"   Arguments: {arguments}")

                    tool_result = self.execute_tool(tool_name, arguments)

                    if self.verbose:
                        preview = tool_result[:200] + "..." if len(tool_result) > 200 else tool_result
                        print(f"   Result: {preview}")

                    messages.append({'role': 'tool', 'content': tool_result})

            elif 'content' in assistant_message:
                final_answer = assistant_message['content']

                if self.verbose:
                    print(f"\n✅ Final Answer (after {turn + 1} turns):")
                    print(f"\n{final_answer}")
                    print(f"\n{'='*60}\n")

                return final_answer

        return "Max turns reached. Unable to complete the task."


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Agentic RAG CLI - Search PDFs with a ReAct agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run agentic_rag_cli.py "What is LoRA?"
  uv run agentic_rag_cli.py "Summarize the PDFs" --model qwen3
  uv run agentic_rag_cli.py "Search for fine-tuning" --quiet
        """
    )
    parser.add_argument("query", help="Your question or task for the agent")
    parser.add_argument("--model", "-m", default="mistral-small3.2",
                        help="Ollama model to use (default: mistral-small3.2)")
    parser.add_argument("--max-turns", "-t", type=int, default=10,
                        help="Maximum reasoning turns (default: 10)")
    parser.add_argument("--quiet", "-q", action="store_true",
                        help="Only show final answer (no reasoning steps)")

    args = parser.parse_args()

    agent = SimpleAgent(
        model=args.model,
        max_turns=args.max_turns,
        verbose=not args.quiet
    )

    answer = agent.run(args.query)

    if args.quiet:
        print(answer)


if __name__ == "__main__":
    main()
