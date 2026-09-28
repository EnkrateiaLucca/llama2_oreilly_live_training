# OReilly Live-Training: "Getting Started with Llama and Other Local Models"

Repository for the oreilly live training course: "Getting Started with Llama and Other Local Models": https://learning.oreilly.com/live-events/getting-started-with-llama-2/0636920098588/

## Setup

> **Windows Users:** See [WINDOWS-SETUP.md](WINDOWS-SETUP.md) for Windows-specific instructions and a compatible requirements file that addresses known compatibility issues.

**Conda**

- Install [anaconda](https://www.anaconda.com/download)
- This repo was tested on a Mac with python=3.10.
- Create an environment: `conda create -n oreilly-llama3 python=3.10`
- Activate your environment with: `conda activate oreilly-llama3`
- Install requirements with: `pip install -r requirements/requirements.txt`
- (Optional) Setup your openai [API key](https://platform.openai.com/). It's only needed for the cloud-comparison cells in `3.0-tool-calling-ollama.ipynb` and the live tool-calling demo (`live-demo-intro-agents-tool-calling.ipynb`); everything else runs locally.

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

4. (Optional) Setup your openai [API key](https://platform.openai.com/). Only needed for the cloud-comparison cells in `3.0-tool-calling-ollama.ipynb` and the live tool-calling demo (`live-demo-intro-agents-tool-calling.ipynb`).

Remember to deactivate the virtual environment once you're done by simply typing:
```bash
deactivate
```

## Setup your .env file

- Change the `.env.example` file to `.env` and add your OpenAI API key (optional; see above).

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

3. [Local RAG with Gemma 4](notebooks/2.1-local-rag.ipynb) - Build a complete local RAG system using Gemma 4 and PDF documents

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/2.1-local-rag.ipynb)

#### 3. Tool Calling and Structured Outputs

4. [Tool Calling with Ollama](notebooks/3.0-tool-calling-ollama.ipynb) - Learn how to implement tool calling with local LLMs (Gmail integration example)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/3.0-tool-calling-ollama.ipynb)

5. [Local Agents: Structured Outputs to Tool Calling](notebooks/3.1-local-agents-intro.ipynb) - Build up to local agents through structured outputs with Pydantic and tool calling

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/3.1-local-agents-intro.ipynb)

#### 4. Agentic RAG

6. [Simple Agentic RAG](notebooks/4.0-simple-agentic-rag.ipynb) - Build a ReAct-based agentic RAG system from scratch

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.0-simple-agentic-rag.ipynb)

#### 4.5 Local Agents

7. [Useful Local Agents](notebooks/5.0-useful-local-agents.ipynb) - Run Hermes Agent with Gemma 4, vibe-check agentic task quality, and learn when to route to the cloud

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/5.0-useful-local-agents.ipynb)

#### 5. Fine-Tuning

8. [Fine-Tuning Llama 3: What You Need to Know](notebooks/6.0-fine-tuning-llama3-what-you-need-to-know.md) - Comprehensive guide to fine-tuning concepts (LoRA, QLoRA, PEFT)

9. [Fine-Tuning Walkthrough with Hugging Face](notebooks/6.1-fine-tuning-walkthrough-hugging-face.ipynb) - Practical fine-tuning implementation

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/6.1-fine-tuning-walkthrough-hugging-face.ipynb)

10. [Quantization Precision Format Code Explanation](notebooks/6.2-quantization-precision-format-code-explanation.ipynb) - Deep dive into model quantization

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/6.2-quantization-precision-format-code-explanation.ipynb)

#### 6. Advanced Topics

11. [GUI Options for Local Models](best-local-models-2026.md#deployment-tools) - LM Studio, Open WebUI (with the Docker command) and other front-ends for local models

12. [Best Local LLMs in Practice (2026 Edition)](notebooks/8.0-best-local-models-examples.ipynb) - Compare and explore the best local models available

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/8.0-best-local-models-examples.ipynb)

13. [vLLM Setup Guide](notebooks/vllm-setup-guide.ipynb) - Complete guide to setting up and using vLLM for high-performance inference

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/vllm-setup-guide.ipynb)

### Legacy Notebooks

Older versions and experimental notebooks are available in the `notebooks/legacy-notebooks/` directory.

## Additional Resources

### Model Guides
- **[Best Local Models 2026](best-local-models-2026.md)** - Which open models to run locally by hardware tier (Gemma 4, Qwen 3.8, DeepSeek-R1, Phi-4 and others), with deployment tools and use-case picks
