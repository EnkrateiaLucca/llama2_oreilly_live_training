# Best Open-Source LLMs for Local Deployment (2026 Edition)

*Updated September 2026 — Covers models runnable on consumer hardware up to 48GB VRAM*

---

## Course Default: Gemma 4

**Gemma 4** (Google DeepMind, 2026) is the primary model for this course. One family covers every student regardless of hardware, is multimodal by default, and carries a clean Apache 2.0 license with no usage restrictions.

| Variant | VRAM / RAM | Ollama Tag | Use When |
|---------|-----------|------------|----------|
| `gemma4:e2b` | CPU, ≤8GB RAM | `gemma4:e2b` | Very constrained hardware |
| `gemma4:e4b` | CPU, 8–16GB RAM | `gemma4` (= `gemma4:e4b`) | Course default; no GPU available |
| `gemma4:12b` | 16GB VRAM or unified memory | `gemma4:12b` | Most students (16 GB laptops) |
| `gemma4:26b` | ~18GB VRAM (19GB download) | `gemma4:26b` | 24GB GPUs; MoE, fast per token |
| `gemma4:31b` | 24GB VRAM (20GB download) | `gemma4:31b` | Prosumer GPUs, best quality |

- **Multimodal** — image input on all sizes; audio input on E2B, E4B and 12B only (clips up to 30 s). Ollama's `gemma4` tags currently advertise text + image only, so use Hugging Face Transformers for audio
- **Apache 2.0** — no MAU caps, no EU geographic restrictions, safe for commercial products
- **26B (MoE)** activates only 3.8B parameters per token, so it's fast, but all weights still load: ~18GB VRAM at Q4 (19GB download)

```bash
ollama pull gemma4        # course default (= gemma4:e4b), CPU-friendly
ollama pull gemma4:12b    # 16GB VRAM / unified memory (most students)
ollama pull gemma4:26b    # ~18GB VRAM
ollama pull gemma4:31b    # 24GB VRAM
```

---

## Model Tiers by Hardware

### Tier 1 — CPU Only (8–16GB RAM)

**Start here:** `ollama pull qwen3:4b` (general) or `ollama pull phi4-mini` (reasoning/math)

| Model | Size | License | Ollama Tag | Strength | Notes |
|-------|------|---------|------------|----------|-------|
| **Qwen3 4B** ★ | 4B | Apache 2.0 | `qwen3:4b` | General, multilingual | 2.5GB disk; hybrid thinking/non-thinking mode; 100+ languages |
| **Phi-4 Mini** ★ | 3.8B | MIT | `phi4-mini` | Reasoning, math | MMLU 68.5%; 128K context; ~20–30 tok/s on CPU |
| Gemma 4 E4B | ~4.5B active (8B MoE) | Apache 2.0 | `gemma4:e4b` | Multimodal (image + audio) | 9.6GB disk; course default (`gemma4`); only CPU option with native multimodal |
| Gemma 4 E2B | ~2.3B active (5.1B MoE) | Apache 2.0 | `gemma4:e2b` | Ultra-lightweight | 7.2GB disk; best for ≤8GB RAM systems |
| Qwen3 1.7B | 1.7B | Apache 2.0 | `qwen3:1.7b` | Minimal footprint | 1.4GB disk; Raspberry Pi / embedded |
| DeepSeek-R1 7B (distill) | 7B | MIT | `deepseek-r1:7b` | Chain-of-thought reasoning | 4.7GB disk; distilled from 671B; strong math; slow on CPU |

**Picking a model at Tier 1:**
- Default general use → **Qwen3 4B**: smallest footprint, multilingual, hybrid thinking mode
- STEM, math, or structured reasoning → **Phi-4 Mini**: best reasoning at this size class
- Need image or audio input on CPU → **Gemma 4 E4B**: only option with native multimodal here

---

### Tier 2 — Consumer GPU (6–16GB VRAM)

**Start here:** `ollama pull phi4` (if 10–16GB VRAM) or `ollama pull qwen3:8b` (if 8GB VRAM)

| Model | Size | License | Ollama Tag | Strength | Notes |
|-------|------|---------|------------|----------|-------|
| **Phi-4 14B** ★ | 14B dense | MIT | `phi4` | Reasoning, STEM | 9.1GB disk; MMLU ~80%; matches Llama 3.3 70B on reasoning at 5× smaller |
| **Qwen3 8B** ★ | 8B | Apache 2.0 | `qwen3:8b` | Coding, multilingual | 5.2GB disk; fits 8GB VRAM; hybrid thinking mode; tool calling |
| **Gemma 4 12B** ★ | 12B dense ("12B Unified") | Apache 2.0 | `gemma4:12b` | General, multimodal | 7.6GB download; targets 16GB VRAM / unified memory; 256K context; image + audio |
| Qwen3 14B | 14B | Apache 2.0 | `qwen3:14b` | Long context, coding | 9.3GB disk; solid all-rounder |
| DeepSeek-R1 14B (distill) | 14B | MIT | `deepseek-r1:14b` | Reasoning, math | 9.0GB disk; chain-of-thought; fits 12GB VRAM |
| Phi-4 Mini Reasoning | 3.8B | MIT | `phi4-mini-reasoning:3.8b` | Math specialist | MATH-500: 94.6%; ideal for STEM tutoring on 6GB VRAM |
| Mistral Small 3.2 | 24B dense | Apache 2.0 | `mistral-small3.2:24b` | Multilingual, vision | 128K context; text + image; 1.9M Ollama pulls; fits 16GB VRAM at Q4 |
| LFM2 (Liquid AI) | 24B total / 2B active | Apache 2.0 | `lfm2` | Fast inference | MoE; 2B active → very fast despite 24B total; 1.1M Ollama pulls |

**Picking a model at Tier 2:**
- Best overall quality on 10–16GB VRAM → **Phi-4 14B**: punches far above its weight
- Only have 8GB VRAM → **Qwen3 8B**: fast, capable, best fit for that constraint
- Want best quality + multimodal on 16GB (GPU or laptop unified memory) → **Gemma 4 12B**: built for 16GB machines, image + audio input

---

### Tier 3 — Prosumer GPU (20–24GB VRAM)

**Start here:** `ollama pull qwen3.8:27b` (coding/agentic) or `ollama pull gemma4:31b` (general + multimodal)

| Model | Size | License | Ollama Tag | Strength | Notes |
|-------|------|---------|------------|----------|-------|
| **Qwen3.8 27B** ★ | 27B dense | Apache 2.0 | `qwen3.8:27b` | Coding, agentic | 18GB download; August 2026, newer than Qwen3.6 27B; vision; 256K context |
| **Gemma 4 31B** ★ | 30.7B dense | Apache 2.0 | `gemma4:31b` | General, multimodal | MMLU-Pro 85.2%; AIME 89.2%; 20GB disk at Q4 |
| Gemma 4 26B (MoE) | 25.2B total / 3.8B active | Apache 2.0 | `gemma4:26b` | Fast general, image input | 19GB download; ~18GB VRAM at Q4; 256K context; no audio |
| **DeepSeek-R1 32B (distill)** ★ | 32B | MIT | `deepseek-r1:32b` | Math, reasoning | Best reasoning model fitting 24GB VRAM; 20GB disk |
| Qwen3 32B | 32B | Apache 2.0 | `qwen3:32b` | General, multilingual | 20GB disk; hybrid thinking mode |
| Qwen3 30B (MoE) | 30B total / 3B active | Apache 2.0 | `qwen3:30b` | Fast inference | 19GB disk; 3B active → very fast for quality level; 256K context |
| Qwen3-Coder-Next | 80B total / 3B active MoE | Apache 2.0 | `qwen3-coder-next` | Code generation | 1.3M Ollama pulls; 3B active → fast despite 80B total |

**Picking a model at Tier 3:**
- Coding, agents, tool use → **Qwen3.8 27B**: newest Qwen 27B, 256K context
- General + multimodal → **Gemma 4 31B**: best multimodal quality that fits 24GB VRAM
- Pure reasoning / math / science → **DeepSeek-R1 32B (distill)**: strongest chain-of-thought at this tier

---

### Tier 4 — Workstation / Multi-GPU (48GB+ VRAM)

All of these are self-hostable, but need workstation-class or multi-GPU hardware.

**Start here:** `ollama pull deepseek-r1:70b` (43GB, MIT, single A100 80GB)

| Model | Size | License | Ollama Tag | Hardware | Strength |
|-------|------|---------|------------|----------|---------|
| Qwen3.5 122B (MoE) | 122B total / 10B active | Apache 2.0 | `qwen3.5:122b` | 2× A100 80GB | Strong coding + reasoning; good VRAM efficiency |
| **DeepSeek-R1 70B (distill)** ★ | 70B | MIT | `deepseek-r1:70b` | 43GB, 1× A100 80GB | MMLU 90.8%; best self-hostable reasoning model |
| **Llama 3.3 70B** ★ | 70B dense | Llama 3.3 Community | `llama3.3:70b` | 43GB | MMLU 86%; strong coding; 3.8M Ollama pulls |
| Llama 4 Scout | 109B total / 17B active | Llama 4 Community | `llama4:16x17b` | 67GB | 10M context; multimodal; EU restrictions apply |

---

## Model Family Reference

### Gemma 4 (Google, 2026)
Apache 2.0. Five variants: E2B, E4B, 12B Unified (June 2026, built for 16GB machines), 26B MoE, 31B dense. Image input (and video as frame sequences) on all sizes; audio input on E2B, E4B and 12B only. Ollama's `gemma4` tags currently advertise text + image only, so run audio through Hugging Face Transformers. The 26B MoE activates only 3.8B parameters per token, which makes it fast, but it still needs ~18GB VRAM at Q4.

### Qwen 3 / 3.5 / 3.6 / 3.8 (Alibaba, 2025–2026)
Apache 2.0. Qwen3 launched April 2025 with 8 dense and MoE sizes (0.6B–235B) introducing a **hybrid thinking/non-thinking mode** — toggle with a system prompt. Qwen3.5 expanded multimodal capabilities and added larger MoE variants up to 397B. Qwen3.6-27B (April 2026) is a 27B dense coding-focused model. Qwen3.8-27B (August 2026, Apache 2.0, `qwen3.8:27b`) is the newer 27B release; Qwen3.6 remains available on Ollama.

### DeepSeek R1 / V4 (DeepSeek, 2025–2026)
DeepSeek-R1 (January 2025, MIT) is the reference reasoning model. All distilled variants (7B, 14B, 32B, 70B) are MIT-licensed. DeepSeek V4 Pro (April 2026, MIT) scales to 1.6T parameters with 49B active; it needs data-centre hardware, so it's out of scope for local use.

### Phi-4 (Microsoft, 2025)
MIT license. Phi-4 14B achieves MMLU 80.4% — competitive with models 5× its size. Phi-4 Mini (3.8B) brings strong reasoning to sub-4GB disk. Phi-4 Mini Reasoning (April 2025) is a math specialist scoring 94.6% on MATH-500.

### Llama 3.3 / 4 (Meta, 2024–2026)
Llama 3.3 70B (late 2024) remains competitive on coding benchmarks. Llama 4 Scout and Maverick (April 2025) are multimodal MoE models with 17B active parameters and context windows up to 10M tokens. **License note:** restricts EU-domiciled users and companies with 700M+ MAU.

### Mistral (2025–2026)
Mistral Small 3.2 (24B, Apache 2.0, `mistral-small3.2:24b`) is the current small-tier model with vision. Magistral (24B, Apache 2.0, `magistral`) is the thinking/reasoning variant. Mistral Medium 3.5 (April 2026, 128B dense) is the flagship self-hostable model with a built-in coding agent at 77.6% SWE-bench Verified.

---

## Quick Start

### Install Ollama

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

### Course Default (Gemma 4)

```python
import ollama

response = ollama.chat(
    model='gemma4',
    messages=[
        {'role': 'user', 'content': 'Explain the difference between RAG and fine-tuning.'}
    ]
)
print(response['message']['content'])
```

### Gemma 4 Multimodal (Image Input)

```python
import ollama
import base64

with open('image.jpg', 'rb') as f:
    image_data = base64.b64encode(f.read()).decode()

response = ollama.chat(
    model='gemma4',
    messages=[
        {
            'role': 'user',
            'content': 'Describe what you see in this image.',
            'images': [image_data]
        }
    ]
)
print(response['message']['content'])
```

### Qwen3 with Thinking Mode

```python
import ollama

response = ollama.chat(
    model='qwen3:8b',
    messages=[
        {'role': 'system', 'content': 'You are a helpful assistant. /think'},
        {'role': 'user', 'content': 'Write a Python function to find all prime numbers up to N using the Sieve of Eratosthenes.'}
    ]
)
print(response['message']['content'])
```

### OpenAI-Compatible Endpoint (Drop-in Replacement)

```python
from openai import OpenAI

client = OpenAI(base_url='http://localhost:11434/v1', api_key='ollama')

response = client.chat.completions.create(
    model='gemma4',
    messages=[
        {'role': 'user', 'content': 'What is retrieval-augmented generation?'}
    ]
)
print(response.choices[0].message.content)
```

---

## Deployment Tools

### Ollama — Recommended for most users
Always-on OpenAI-compatible endpoint at `localhost:11434/v1`. Default choice for the course.

```bash
ollama serve                    # start server (auto-starts on install)
ollama pull gemma4              # download the course default (= gemma4:e4b)
ollama list                     # list downloaded models
ollama run gemma4               # interactive chat in terminal
```

### LM Studio — Best for beginners and model discovery
Point-and-click GUI with a built-in Hugging Face model browser. Best model discovery experience, especially on Apple Silicon. API server must be started manually. 3M+ cumulative downloads.

### Open WebUI — Chat interface for Ollama
Browser-based ChatGPT-style UI that connects to any Ollama or OpenAI-compatible backend. Includes RAG, multi-user auth, voice I/O, and a plugin system. Install after Ollama:

```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

### llama-cpp-python — Advanced / in-process inference

For loading any GGUF directly from Hugging Face, in-process Python inference with no daemon, or AMD GPU with Vulkan support.

```bash
pip install llama-cpp-python                  # CPU only
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python  # CUDA GPU
```

```python
from llama_cpp import Llama

llm = Llama.from_pretrained(
    repo_id='unsloth/gemma-4-12b-it-GGUF',
    filename='*Q4_K_M.gguf',
    n_gpu_layers=-1
)

output = llm.create_chat_completion(messages=[
    {'role': 'user', 'content': 'Explain quantization in LLMs.'}
])
print(output['choices'][0]['message']['content'])
```

### vLLM — Production serving (concurrent users)
Required when serving 5+ simultaneous users. Linux + NVIDIA GPU only. Delivers 2,300+ tok/s on H100. Not needed for local development — graduate to this when moving to production.

```bash
pip install vllm
vllm serve google/gemma-4-E4B-it   # OpenAI-compatible server on http://localhost:8000
```

See the [vLLM quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart/) for flags (`--host`, `--port`, `--max-model-len`).

---

## Model Selection by Use Case

| Use Case | Best Pick | Alternative |
|----------|-----------|-------------|
| Course default / general | `gemma4` (= `gemma4:e4b`, CPU) or `gemma4:12b` (16GB) | `qwen3:8b` |
| Coding + agentic workflows | `qwen3.8:27b` | `qwen3:8b` |
| Math / science / reasoning | `deepseek-r1:32b` | `phi4-mini` (CPU) |
| Multimodal (image/audio) | `gemma4:12b` (audio via HF Transformers) | `gemma4:e4b` (CPU) |
| Multilingual | `qwen3:4b` or `qwen3:8b` | `gemma4:12b` |
| Minimal hardware (≤8GB RAM) | `qwen3:4b` | `gemma4:e2b` |
| Best quality, any hardware | `gemma4:31b` (24GB VRAM) | `deepseek-r1:70b` (A100) |

---

## Hardware Configuration Guide

**CPU only / integrated graphics (8–16GB RAM)**
→ `gemma4:e4b` or `qwen3:4b`

**Entry gaming GPU (RTX 4060 / 6–8GB VRAM)**
→ `qwen3:8b` or `gemma4:e4b`

**Mid-range GPU (RTX 4070 / 10–12GB VRAM)**
→ `phi4` (14B) or `qwen3:14b`

**16GB VRAM or 16GB unified-memory laptop**
→ `gemma4:12b`

**High-end GPU (RTX 4090 / 24GB VRAM)**
→ `gemma4:31b`, `gemma4:26b`, `qwen3.8:27b` or `deepseek-r1:32b`

**Prosumer / workstation (48GB+ VRAM)**
→ `deepseek-r1:70b` or `llama3.3:70b`

---

## Keeping Updated

- **Chatbot Arena (overall quality):** https://huggingface.co/spaces/lmsys/chatbot-arena-leaderboard
- **Artificial Analysis (open-source rankings):** https://artificialanalysis.ai
- **Ollama model library:** https://ollama.com/library
- **Hugging Face trending:** https://huggingface.co/models?sort=trending
- **Community:** r/LocalLLaMA (636k+ members)
