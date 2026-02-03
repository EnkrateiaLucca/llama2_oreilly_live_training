# OReilly Live-Training: "Getting Started with Llama3"

Repository for the oreilly live training course: "Getting Started with Llama3": https://learning.oreilly.com/live-events/getting-started-with-llama-2/0636920098588/

## Setup

**Conda**

- Install [anaconda](https://www.anaconda.com/download)
- This repo was tested on a Mac with python=3.10.
- Create an environment: `conda create -n oreilly-llama3 python=3.10`
- Activate your environment with: `conda activate oreilly-llama3`
- Install requirements with: `pip install -r requirements/requirements.txt`
- Setup your openai [API key](https://platform.openai.com/)

**Pip**


1. **Create a Virtual Environment:**
    Navigate to your project directory. Make sure you hvae python3.10 installed!
    If using Python 3's built-in `venv`:
    ```bash
    python -m venv oreilly-llama3
    ```
    If you're using `virtualenv`:
    ```bash
    virtualenv oreilly-llama3
    ```

2. **Activate the Virtual Environment:**
    - **On Windows:**
      ```bash
      .\oreilly-llama3\Scripts\activate
      ```
    - **On macOS and Linux:**
      ```bash
      source oreilly-llama3/bin/activate
      ```

3. **Install Dependencies from `requirements.txt`:**
    ```bash
    pip install python-dotenv
    pip install -r requirements/requirements.txt
    ```

4. Setup your openai [API key](https://platform.openai.com/)

Remember to deactivate the virtual environment once you're done by simply typing:
```bash
deactivate
```

## Setup your .env file

- Change the `.env.example` file to `.env` and add your OpenAI API key.

## To use this Environment with Jupyter Notebooks:

- ```pip install jupyter```
- ```python3 -m ipykernel install --user --name=oreilly-llama3```


## Notebooks

### Core Learning Path

These notebooks follow a structured learning path from basics to advanced topics:

#### 1. Getting Started with Local LLMs

1. [Quickstart with Ollama](notebooks/1.0-quickstart-ollama.ipynb) - Get started running local LLMs using Ollama

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/1.0-quickstart-ollama.ipynb)

#### 2. RAG (Retrieval-Augmented Generation)

2. [Introduction to RAG](notebooks/2.0-introduction-to-rag.ipynb) - Learn the fundamentals of RAG with interactive visualizations of embeddings and chunking

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/2.0-introduction-to-rag.ipynb)

3. [Local RAG with Llama 3](notebooks/2.1-local-rag.ipynb) - Build a complete local RAG system using Llama 3 and PDF documents

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/2.1-local-rag-with-llama3.ipynb)

#### 3. Tool Calling and Structured Outputs

4. [Tool Calling with Ollama](notebooks/3.0-tool-calling-ollama.ipynb) - Learn how to implement tool calling with local LLMs (Gmail integration example)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/3.0-tool-calling-ollama.ipynb)

5. [Llama 3.1 Structured Outputs](notebooks/3.1-llama31-structured-outputs.ipynb) - Generate structured outputs using Pydantic models with Llama 3.1

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/3.1-llama31-structured-outputs.ipynb)

6. [Local Agent from Scratch](notebooks/3.2-local-agent-from-scratch.ipynb) - Build a simple agent from scratch using tool calling

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/3.2-local-agent-from-scratch.ipynb)

#### 4. Agentic RAG

7. [Simple Agentic RAG](notebooks/4.0-simple-agentic-rag.ipynb) - Build a ReAct-based agentic RAG system from scratch

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.0-simple-agentic-rag.ipynb)

#### 5. Fine-Tuning

8. [Fine-Tuning Llama 3: What You Need to Know](notebooks/6.0-fine-tuning-llama3-what-you-need-to-know.md) - Comprehensive guide to fine-tuning concepts (LoRA, QLoRA, PEFT)

9. [Fine-Tuning Walkthrough with Hugging Face](notebooks/6.1-fine-tuning-walkthrough-hugging-face.ipynb) - Practical fine-tuning implementation

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/6.1-fine-tuning-walkthrough-hugging-face.ipynb)

10. [Quantization Precision Format Code Explanation](notebooks/6.2-quantization-precision-format-code-explanation.ipynb) - Deep dive into model quantization

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/6.2-quantization-precision-format-code-explanation.ipynb)

#### 6. Advanced Topics

11. [GUI for Llama 3 Options](notebooks/7.0-gui-for-llama3-options.ipynb) - Explore different GUI options for working with Llama models

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/7.0-gui-for-llama3-options.ipynb)

12. [Best Local LLMs in Practice (2025 Edition)](notebooks/8.0-best-local-models-examples.ipynb) - Compare and explore the best local models available

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/8.0-best-local-models-examples.ipynb)

13. [vLLM Setup Guide](notebooks/vllm-setup-guide.ipynb) - Complete guide to setting up and using vLLM for high-performance inference

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/vllm-setup-guide.ipynb)

### Legacy Notebooks

Older versions and experimental notebooks are available in the `notebooks/legacy-notebooks/` directory.

## Additional Resources

### Model Guides
- **[LLM Model Sizes Guide](llm-model-sizes-guide.md)** - Comprehensive guide to different model sizes and their use cases
- **[Best Local Models 2025](best-local-models-2025.md)** - Updated guide to the top-performing open-source models that can run locally with <64GB RAM, including Qwen, DeepSeek, Mixtral, and others

### Key Features of the 2025 Model Guide:
- **Performance Benchmarks**: Latest benchmark scores for reasoning, coding, and multilingual tasks
- **Hardware Requirements**: Detailed RAM and GPU requirements for each model
- **Deployment Instructions**: Step-by-step setup for Ollama, LM Studio, and other tools  
- **Use Case Recommendations**: Which models work best for specific applications
- **Model Comparisons**: Side-by-side analysis of capabilities and trade-offs

### Top Models Covered Beyond Llama:
- **Qwen2.5 Series** (Alibaba) - Exceptional multilingual and reasoning capabilities
- **DeepSeek-V3 & DeepSeek-Coder** - Specialized programming and development
- **Mixtral 8x22B** - Efficient Mixture of Experts architecture
- **Gemma 2** (Google) - Efficient and safety-focused models
- **Command-R+** (Cohere) - Optimized for RAG and tool use
- **Yi-Large** (01.AI) - Strong bilingual performance