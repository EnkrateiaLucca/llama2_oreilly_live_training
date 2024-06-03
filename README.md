# OReilly Live-Training: "Getting Started with Llama2"

Repository for the oreilly live training course: "Getting Started with Llama3": https://learning.oreilly.com/live-events/getting-started-with-llama-2/0636920098588/

## Setup

**Conda**

- Install [anaconda](https://www.anaconda.com/download)
- This repo was tested on a Mac with python=3.10.
- Create an environment: `conda create -n oreilly-llama2 python=3.10`
- Activate your environment with: `conda activate oreilly-llama2`
- Install requirements with: `pip install -r requirements/requirements.txt`
- Setup your openai [API key](https://platform.openai.com/)

**Pip**


1. **Create a Virtual Environment:**
    Navigate to your project directory. Make sure you hvae python3.10 installed!
    If using Python 3's built-in `venv`:
    ```bash
    python -m venv oreilly-llama2
    ```
    If you're using `virtualenv`:
    ```bash
    virtualenv oreilly-llama2
    ```

2. **Activate the Virtual Environment:**
    - **On Windows:**
      ```bash
      .\oreilly-llama2\Scripts\activate
      ```
    - **On macOS and Linux:**
      ```bash
      source oreilly-llama2/bin/activate
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
- ```python3 -m ipykernel install --user --name=oreilly-llama2```


## Notebooks

Here are the notebooks available in the `notebooks/` folder:

1. [Getting Started with Llama 2](notebooks/1.0-ls-Getting-Started-With-Llama2.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/1.0-ls-Getting-Started-With-Llama2.ipynb)

2. [Hello Llama Local](notebooks/1.1-HelloLlamaLocal.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/1.1-HelloLlamaLocal.ipynb)

3. [Ollama Quick Setup](notebooks/1.2-ollama_quick_setup.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/1.2-ollama_quick_setup.ipynb)

4. [Query Docs with Llama 2](notebooks/3.0-ls-Query-Docs-Llama2.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/2.0-ls-Query-Docs-Llama2.ipynb)

5. [LangChain Llama2 QA CSV](notebooks/2.1-Langchain-Llama2-Qa-Csv.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/2.1-Langchain-Llama2-Qa-Csv.ipynb)

6. [Quiz from Source Llama2 - Llama Index](notebooks/3.0-quiz_from_source_llama2-llama-index.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/3.0-quiz_from_source_llama2-llama-index.ipynb)

7. [Structured Output Langchain Llama2](notebooks/3.1-structured-output-langchain-llama2.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/3.1-structured-output-langchain-llama2.ipynb)

8. [Fine Tuning Llama2](notebooks/4.0-ls-Fine-Tuning-Llama2.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.0-ls-Fine-Tuning-Llama2.ipynb)

9. [Quantization Basics](notebooks/4.1-quantization-basics.ipynb)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/

10. [Fine Tuning Llama2](notebooks/4.0-ls-Fine-Tuning-Llama2.ipynb)

    [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.0-ls-Fine-Tuning-Llama2.ipynb)