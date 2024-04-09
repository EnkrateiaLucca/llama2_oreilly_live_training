# OReilly Live-Training: "Getting Started with Llama2"

Repository for the oreilly live training course: "Getting Started with Llama2": https://learning.oreilly.com/live-events/getting-started-with-llama-2/0636920098588/

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

2. [Prompt Engineering Basics](notebooks/2.0-ls-Prompt-Engineering-Llama2.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/2.0-ls-Prompt-Engineering-Llama2.ipynb)

3. [Query Docs with Llama 2](notebooks/3.0-ls-Query-Docs-Llama2.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/3.0-ls-Query-Docs-Llama2.ipynb)

4. [Fine Tuning Llama2](notebooks/4.0-ls-Fine-Tuning-Llama2.ipynb)

   [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/EnkrateiaLucca/llama2_oreilly_live_training/blob/main/notebooks/4.0-ls-Fine-Tuning-Llama2.ipynb)


