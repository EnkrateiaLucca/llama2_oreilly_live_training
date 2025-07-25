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

Here are the notebooks available in the `notebooks/` folder:

## Notebooks

Here are the notebooks available in the `notebooks/` folder:

1. [Llama 3 Quick Start](notebooks/1.0-Llama3-Quick-Start.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/1.0-Llama3-Quick-Start.ipynb)

2. [Llama 3 Vector Similarity Search](notebooks/2.0-llama3-vector-similarity-search.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/2.0-llama3-vector-similarity-search.ipynb)

3. [Llama 3 Chat PDF Intro](notebooks/3.0-llama3-chat-pdf-intro.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/3.0-llama3-chat-pdf-intro.ipynb)

4. [PrivateGPT with Llama 3 Quick Setup](notebooks/3.2-privateGPT-with-llama3-quick-setup.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/3.2-privateGPT-with-llama3-quick-setup.ipynb)

5. [Tool Calling Ollama](notebooks/4.0-tool-calling-ollama.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.0-tool-calling-ollama.ipynb)

6. [Llama 3 Groq Function Calling](notebooks/4.1-llama3-groq-function-calling.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.1-llama3-groq-function-calling.ipynb)

7. [Llama 3.1 LlamaCPP Tool Calling](notebooks/4.1-llama31-llamacpp-tool-calling.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.1-llama31-llamacpp-tool-calling.ipynb)

8. [Llama 3.1 Tool Calling](notebooks/4.2-llama31-tool-calling.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.2-llama31-tool-calling.ipynb)

9. [Tool Calling Agent](notebooks/4.3-tool-calling-agent.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.3-tool-calling-agent.ipynb)

10. [Llama 3.1 Structured Outputs](notebooks/4.4-llama31-structured-outputs.ipynb)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.4-llama31-structured-outputs.ipynb)

11. [Llama 3 Agentic RAG](notebooks/5.0-llama3-agentic-rag.ipynb)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/5.0-llama3-agentic-rag.ipynb)

12. [Fine Tuning Llama 3: What You Need to Know](notebooks/6.0-fine-tuning-llama3-what-you-need-to-know.ipynb)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/6.0-fine-tuning-llama3-what-you-need-to-know.ipynb)

13. [Fine Tuning Walkthrough Hugging Face](notebooks/6.1-fine-tuning-walkthrough-hugging-face.ipynb)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/6.1-fine-tuning-walkthrough-hugging-face.ipynb)

14. [Quantization Precision Format Code Explanation](notebooks/6.2-quantization-precision-format-code-explanation.ipynb)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/6.2-quantization-precision-format-code-explanation.ipynb)

15. [GUI for Llama 3 Options](notebooks/7.0-gui-for-llama3-options.ipynb)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/7.0-gui-for-llama3-options.ipynb)

16. [Best Local LLMs in Practice (2025 Edition)](notebooks/8.0-best-local-models-examples.ipynb)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/8.0-best-local-models-examples.ipynb)

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