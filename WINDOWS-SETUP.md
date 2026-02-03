# Windows Setup Guide: "Getting Started with Llama 3"

This guide provides Windows-specific setup instructions for the O'Reilly Live Training course.

## Prerequisites

- Windows 10/11
- Python 3.10 (recommended)
- [Ollama for Windows](https://ollama.com/download/windows)

## Known Windows Compatibility Issues

The main requirements file includes packages that have Unix-only dependencies. This guide addresses:

| Package | Issue | Solution |
|---------|-------|----------|
| `gpt4all==2.8.0` | No Windows wheels | Use `gpt4all>=2.8.2` |
| `uvloop` | Unix-only (no Windows support) | Use `winloop` as alternative |
| `bitsandbytes` | Limited Windows support | Optional - requires proper CUDA setup |

## Setup Instructions

### Option 1: Using pip (Recommended for Windows)

1. **Install Python 3.10**

   Download from [python.org](https://www.python.org/downloads/) and ensure "Add Python to PATH" is checked during installation.

2. **Create a Virtual Environment**

   Open Command Prompt or PowerShell and navigate to your project directory:

   ```powershell
   python -m venv oreilly-llama3
   ```

3. **Activate the Virtual Environment**

   ```powershell
   .\oreilly-llama3\Scripts\activate
   ```

4. **Install Dependencies**

   ```powershell
   pip install python-dotenv
   pip install -r requirements/requirements-windows.txt
   ```

   If `requirements-windows.txt` doesn't exist yet, you can compile it:

   ```powershell
   pip install uv
   uv pip compile ./requirements/requirements-windows.in -o ./requirements/requirements-windows.txt
   pip install -r requirements/requirements-windows.txt
   ```

5. **Deactivate when done**

   ```powershell
   deactivate
   ```

### Option 2: Using Conda

1. **Install Anaconda or Miniconda**

   Download from [anaconda.com](https://www.anaconda.com/download)

2. **Create and Activate Environment**

   ```powershell
   conda create -n oreilly-llama3 python=3.10
   conda activate oreilly-llama3
   ```

3. **Install Dependencies**

   ```powershell
   pip install -r requirements/requirements-windows.txt
   ```

## Setup OpenAI API Key

1. Get your API key from [platform.openai.com](https://platform.openai.com/)

2. Create a `.env` file in the project root (copy from `.env.example`):

   ```powershell
   copy .env.example .env
   ```

3. Edit `.env` and add your API key:

   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Setup Jupyter Notebooks

```powershell
pip install jupyter
python -m ipykernel install --user --name=oreilly-llama3
```

## Install Ollama for Windows

1. Download Ollama from [ollama.com/download/windows](https://ollama.com/download/windows)
2. Run the installer
3. After installation, pull the Llama 3 model:

   ```powershell
   ollama pull llama3
   ```

## Troubleshooting

### "No wheels with a matching platform tag" Error

This typically means a package doesn't have Windows binaries. The `requirements-windows.in` file has been adjusted for known problematic packages.

### bitsandbytes Issues

`bitsandbytes` has limited Windows support. If you encounter issues:

1. **Option A**: Skip it (comment out in requirements) - most notebooks will still work
2. **Option B**: Install with CUDA support:
   ```powershell
   pip install bitsandbytes-windows
   ```
   Note: Requires NVIDIA GPU with CUDA installed.

### uvloop Error

If you see errors about `uvloop`, it's Unix-only. The Windows requirements use `winloop` instead. If issues persist:

```powershell
pip uninstall uvloop
pip install winloop
```

### GPT4All Installation

Ensure you're using version 2.8.2 or later:

```powershell
pip install gpt4all>=2.8.2
```

## Notebooks That May Need Adjustment on Windows

Most notebooks should work without modification. If you encounter issues:

- **Fine-tuning notebooks**: May require additional CUDA configuration for GPU acceleration
- **bitsandbytes quantization**: Works best on Linux; consider using cloud environments like Google Colab for these specific notebooks

## Additional Resources

- [Ollama Windows Documentation](https://github.com/ollama/ollama/blob/main/docs/windows.md)
- [Course README](README.md) - Main course information and notebook descriptions
