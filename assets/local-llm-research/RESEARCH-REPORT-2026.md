# Local LLM Research Report — May 2026

*Prepared for O'Reilly Live Training: Local LLMs with Llama*
*Research date: May 8, 2026*

---

## Course Direction Note

**Primary model for the 2026 course: Gemma 4 (replacing Llama 3)**

Lucas's decision. The reasoning is documented in Section C but summarized here:

Llama 3 made sense as the course anchor in 2023–2024 because it was the leading open model and had the widest community adoption. That's no longer true. Gemma 4 is the better anchor for five concrete reasons:

1. **Same model family across all compute tiers** — E2B and E4B run on CPU-only machines, 26B MoE runs on 8GB VRAM consumer GPUs, 31B dense runs on prosumer GPUs. One model family covers every student regardless of hardware, which makes it unique in the current landscape.
2. **Multimodal by default** — image and audio input across all Gemma 4 variants. Llama 3 was text-only at small sizes. Teaching multimodal workflows no longer requires a separate model.
3. **Clean commercial license (Apache 2.0)** — no MAU caps, no EU geographic restrictions, no usage policy edge cases. Students can build commercial products without license anxiety. Llama 4's Community License has restrictions that create unnecessary complexity.
4. **Strong benchmark performance** — Gemma 4 31B ranks #3 among all open-weight models on Chatbot Arena, above Llama 4 Maverick. Gemma 4 26B (MoE) runs on 8GB VRAM and outperforms models that require twice the VRAM.
5. **Active Google development** — Google is iterating fast on the Gemma family with regular releases, strong Ollama integration, and official documentation. The model will stay current throughout the course's lifecycle.

**What this means for notebooks:** all quickstart examples, RAG demos, tool calling examples, and agent workflows default to `ollama pull gemma4:26b` (or `gemma4:e4b` for CPU-only students). Llama 3 references in existing notebooks should be updated but can remain as "also works with" alternatives.

---

## Executive Summary

The open-source LLM ecosystem has changed substantially since the course materials were last updated (early 2025). Four key findings drive the course update recommendations below:

1. **Ollama is the best starting point for most users.** It has become the de facto infrastructure standard across the ecosystem. llama-cpp-python is the better choice for power users who want maximum hardware control, in-process Python inference, or access to any GGUF from Hugging Face without a model registry.

2. **For RAG, LlamaIndex + ChromaDB is the strongest general-purpose Python stack.** simonwillison/llm is the best lightweight personal CLI option. smolagents is worth knowing for agentic multi-step RAG. The main maintenance change needed is a ChromaDB version bump (`0.5.5` → `0.6.x`).

3. **Model recommendations need a full refresh.** Gemma 4, Qwen 3/3.5/3.6, Phi-4, and DeepSeek R1/V4 have redefined quality expectations at every compute tier. The `best-local-models-2025.md` guide is materially outdated.

4. **Ollama + Open WebUI is the dominant community stack.** LM Studio is the entry point for beginners. GPT4All has effectively stalled. The course should add a mention of Open WebUI as the companion UI for Ollama.

---

## Course Update Recommendations

| Priority | Change | Notebook / File |
|----------|--------|----------------|
| High | Replace model guide with 2026 tier table from Section C | `best-local-models-2025.md` → rename to `best-local-models-2026.md` |
| High | Update ChromaDB pin from `0.5.5` → `0.6.x` and test | `notebooks/2.1-local-rag.ipynb`, requirements files |
| Medium | Add Open WebUI to the GUI options notebook | `notebooks/7.0-gui-for-llama3-options.ipynb` |
| Medium | Add Gemma 4, Qwen3, Phi-4 model examples to best-models notebook | `notebooks/8.0-best-local-models-examples.ipynb` |
| Low | Add a note on llama-cpp-python for the advanced/quantization module | `notebooks/6.2-quantization-precision-format-code-explanation.ipynb` |
| Low | Remove GPT4All from GUI options (stalled since Feb 2025) | `notebooks/7.0-gui-for-llama3-options.ipynb` |

---

## A. Ollama vs llama-cpp-python: Decision

### Project Health (as of May 2026)

| Metric | Ollama | llama-cpp-python |
|--------|--------|-----------------|
| GitHub stars | 171,012 | 10,279 |
| Last release | v0.23.2 — May 7, 2026 | v0.3.22 — May 2, 2026 |
| Release cadence (Jan–May 2026) | ~2–3 releases/week | ~1 release/week |
| Active maintenance | Yes — Ollama Inc., paid team | Yes — community; one primary maintainer; 709 open issues |

### Feature Comparison

| Feature | Ollama | llama-cpp-python |
|---------|--------|-----------------|
| Install | Single binary (`curl \| sh` / `brew` / Windows installer) or `pip install ollama` for the client | `pip install llama-cpp-python` CPU-only by default; GPU needs env flags or separate wheels per CUDA version |
| OpenAI-compat API | Yes — full `/v1/chat/completions`, `/v1/embeddings` at `:11434/v1`; also Anthropic Messages API (Jan 2026) | Yes — via `llama_cpp.server` FastAPI; start with `python -m llama_cpp.server --model model.gguf` |
| Streaming | Yes — SSE, including streaming tool calls | Yes — both Python API and HTTP server |
| Tool calling | Yes — native, first-class; parallel + streaming tool calls | Yes — via `tools` parameter; model-specific chat templates |
| Multimodal | Yes — dedicated multimodal engine (May 2025); supports Qwen2.5-VL, Gemma 3/4, Llama 4; known bug with split-file mmproj (April 2026) | Yes — via libmtmd (April 2025); handles split mmproj files correctly where Ollama fails; requires `--mmproj` path |
| Model library | Curated registry (~200+ models, 100M+ pull events); all pre-quantized GGUF from Ollama CDN | Any GGUF from Hugging Face or local disk; all quantization levels (Q2_K–Q8_0, IQ variants, F16) |
| GPU support | NVIDIA CUDA, AMD ROCm, Apple Metal (MLX runner added March 2026); Vulkan requires source build | NVIDIA CUDA, AMD ROCm, Apple Metal, Vulkan (built-in), Intel SYCL; explicit `--n-gpu-layers` control |
| Performance vs raw llama.cpp | 10–15% overhead (Go HTTP server layer); MLX runner on Apple Silicon closes gap | Near-zero overhead in-process; HTTP server mode comparable to Ollama for single-user load |
| Structured output | Yes — JSON Schema constraints (Dec 2024) | Yes — grammar-based sampling; JSON Schema support available |

### Practitioner Sentiment

On r/LocalLLaMA and tech blogs, the consensus is: **Ollama for getting started quickly, llama.cpp for power users who need raw hardware/quantization control.** The decodesfuture.com benchmark (Jan 2026) reports Ollama setup time ~1 minute vs 10–20 minutes for llama.cpp. The insiderllm.com guide (Apr 2026) notes "Ollama's killer feature isn't performance — it's convenience" but warns "the common mistake is using Ollama for production APIs" — vLLM is the upgrade path when serving concurrent users. The Towards AI benchmark (Apr 2026) confirms Ollama "collapses at 5 concurrent users," making llama-cpp-python's server mode or vLLM better choices for anyone building a multi-user app.

llama-cpp-python has a steeper install (CPU-only by default, separate wheel per CUDA version) but gives you direct Python access to model internals, works on every hardware including Vulkan/AMD out of the box, and lets you load any GGUF from Hugging Face with no registry dependency. Users who move from Ollama to llama-cpp typically do so once they need that level of control.

### Which to use

**Start with Ollama if:** you want the fastest path to a running local LLM, you're building something that needs an always-on OpenAI-compatible endpoint, or you want a curated model registry with one-command downloads. This covers most use cases.

**Move to llama-cpp-python if:** you want to load arbitrary GGUF files from Hugging Face (not just Ollama's registry), you need in-process Python inference with no separate server daemon (`Llama(model_path=...)`), you're on Linux with an AMD GPU and need Vulkan support out of the box, you're building an embedded app where a background server is impractical, or you need fine-grained GPU layer control for performance work.

**Use raw llama.cpp (no Python wrapper) if:** you want the absolute minimum overhead, you're targeting non-Python environments, or you're running on edge hardware (Raspberry Pi, Android, WebAssembly).

### Sources
- GitHub Ollama (fetched May 8 2026): https://github.com/ollama/ollama
- GitHub llama-cpp-python (fetched May 8 2026): https://github.com/abetlen/llama-cpp-python
- "llama.cpp vs Ollama vs vLLM: 2026 Comparison" (Jan 31 2026): https://www.decodesfuture.com/articles/llama-cpp-vs-ollama-vs-vllm-local-llm-stack-guide
- "Fastest Local LLM Setup: Ollama vs vLLM vs llama.cpp" (Apr 22 2026): https://insiderllm.com/guides/llamacpp-vs-ollama-vs-vllm/
- "I Tested Ollama vs vLLM vs llama.cpp: The 'Easiest' One Collapses at 5 Concurrent Users" (Apr 2026): https://pub.towardsai.net/i-tested-ollama-vs-vllm-vs-llama-cpp-the-easiest-one-collapses-at-5-concurrent-users-d4f8e0e84886
- Ollama tool calling docs: https://docs.ollama.com/capabilities/tool-calling
- Ollama blog (multimodal engine, streaming tool calls, Anthropic API): https://ollama.com/blog

---

## B. Local RAG Tools: Practical Comparison

### Tool Summaries

**simonwillison/llm + sqlite-vec**
CLI tool (v0.31, April 2026), actively maintained with 1,055 commits and a large plugin ecosystem. Supports RAG via `llm embed` / `llm embed-multi` backed by SQLite via the `sqlite-vec` extension (added August 2024), with local embeddings via `llm-sentence-transformers` and Ollama via the `llm-ollama` plugin. Primarily a CLI/scripting tool — the intended workflow is shell commands and pipes rather than Python library imports. Best fit for users who want to add semantic search to personal documents with minimal infrastructure: one SQLite file, no vector database to manage, and a single command to query. Has a Python API but it's lower-level than LlamaIndex or LangChain.

**semtools (run-llama)**
Rust CLI tool (v3.0.1, March 2026), maintained by the LlamaIndex org, for semantic search and document parsing from the command line. Key constraint: `semtools ask` **requires an OpenAI API key** and `semtools parse` **requires a LlamaParse API key** — it is not fully local. Designed for coding agents and MCP integration (e.g., letting an AI coding assistant search your codebase semantically). Not a Python library. Worth knowing as a developer tool for semantic search in agentic workflows, but not for building local-first RAG applications.

**smolagents RAG (HuggingFace)**
Agent framework (v1.24.0, January 2026, 27k stars). Supports Ollama via LiteLLM. The official RAG example uses **BM25 retrieval (no vector embeddings)** by default — fast to set up but no semantic search. For semantic retrieval, add `sentence-transformers` and a vector store. The key distinction: smolagents frames RAG as an *agentic* problem — the model decides when to search, what to search for, and how to chain multiple retrievals. This is a genuinely different and more powerful pattern than a fixed pipeline (query → retrieve → answer). If you're building an application where users ask complex multi-step questions over a document corpus, smolagents is worth exploring. If you want a straightforward RAG pipeline you can understand step by step, LlamaIndex is cleaner.

**LlamaIndex (current: llama-index-core)**
Underwent a major architectural split in v0.10 (late 2023): the monolithic `llama-index` package became `llama-index-core` plus 300+ separately installable integration packages. The current `llama-index-core` is v0.14.21 (April 2026). The Ollama integration (`llama-index-llms-ollama` v0.10.1, `llama-index-embeddings-ollama` v0.9.0, both March 2026) is first-class. Local HuggingFace embeddings (`llama-index-embeddings-huggingface`) are well-supported. Features include hybrid search (BM25 + vector), reranking modules, and the Workflows event-driven architecture (v0.11). The most complete Python RAG framework for building production applications: handles document ingestion, chunking, embedding, retrieval, reranking, and generation in one coherent API.

### Comparison Matrix

| Tool | Setup Complexity | Ollama Compat | Local Embeddings | Reranking | Best For |
|------|-----------------|---------------|-----------------|-----------|----------|
| simonwillison/llm + sqlite-vec | Medium (multi-plugin CLI) | Yes (llm-ollama plugin) | Yes (llm-sentence-transformers) | No | Personal CLI scripting |
| semtools | Low (Rust/npm install) | No | Partial (search only; `ask` needs OpenAI key) | No | CLI/MCP agent workflows |
| smolagents RAG | Medium (agent + BM25 setup) | Yes (via LiteLLM) | Partial (BM25 default; semantic needs extra work) | No | Agentic multi-step RAG |
| LlamaIndex (llama-index-core) | Low-Medium (split packages) | Yes (first-class) | Yes (HuggingFace or Ollama embed) | Yes (built-in) | Teaching + app development |
| LangChain | Medium-High (verbose boilerplate) | Yes (OllamaEmbeddings) | Yes | Yes (extra setup) | Complex pipeline prototyping |

### Which to use

**For personal document search (no coding required beyond setup):** simonwillison/llm + sqlite-vec — store embeddings in a SQLite file, search with one command, pipe results to any Ollama model; zero infrastructure overhead. The right tool if you want to query your own notes, PDFs, or files from the terminal without building an application.

**For building a RAG application in Python:** LlamaIndex (llama-index-core) — the most complete framework for the full RAG pipeline (ingest → chunk → embed → retrieve → rerank → generate), with first-class Ollama and local embedding support. Strong choice whether you're prototyping or going to production.

**For agentic RAG (model decides when and how to search):** smolagents — best when your application involves complex multi-step questions where a fixed retrieve-then-answer pattern isn't enough. The model can issue multiple queries, decide what's relevant, and chain reasoning steps. Higher complexity but more powerful for the right use case.

**For developer tooling / codebase search via MCP:** semtools — designed to give AI coding assistants (like Cursor or Continue.dev) semantic search over your codebase. Not for building applications, but a useful personal productivity tool if you work with AI-assisted coding.

**Note on the LlamaIndex package name change:** if you used `llama-index` before 2024, it is now `llama-index-core` plus separate integration packages. Install as: `pip install llama-index-core llama-index-llms-ollama llama-index-embeddings-huggingface chromadb`. Also bump ChromaDB from `0.5.5` to `0.6.x` if using older pinned requirements.

### Sources
- https://github.com/run-llama/semtools (v3.0.1, March 2026)
- https://github.com/huggingface/smolagents (v1.24.0, January 2026)
- https://huggingface.co/docs/smolagents/en/examples/rag
- https://github.com/simonw/llm (v0.31, April 2026)
- https://pypi.org/project/llama-index-core/ (v0.14.21, April 2026)
- https://pypi.org/project/llama-index-llms-ollama/ (v0.10.1, March 2026)
- https://pypi.org/project/llama-index-embeddings-ollama/ (v0.9.0, March 2026)
- https://www.llamaindex.ai/blog/llamaindex-v0-10-838e735948f8
- https://www.firecrawl.dev/blog/best-open-source-rag-frameworks
- https://latenode.com/blog/langchain-vs-llamaindex-2025-complete-rag-framework-comparison

---

## C. Top Models for Commercial Use by Compute Tier (May 2026)

*Source: Artificial Analysis Intelligence Index, Ollama library download counts, Hugging Face trending — checked May 8, 2026.*
*License: Apache 2.0 and MIT = safe for commercial use. Llama Community = commercial-OK under 700M MAU, EU restrictions apply. "Cloud-only" = weights not locally runnable; served via provider API.*

---

### Tier 1 — CPU Only (8–16GB RAM)

**→ Start here:** `ollama pull qwen3:4b` (general use) or `ollama pull phi4-mini` (reasoning/math)

**Top picks:**
- **Qwen3 4B** — best all-rounder at this tier. 2.5GB on disk, Apache 2.0, hybrid thinking/non-thinking mode, 100+ languages. The default choice for most CPU-only users.
- **Phi-4 Mini (3.8B)** — best reasoning and math at this tier. MIT license, MMLU 68.5%, 128K context, ~20–30 tok/s on CPU. Prefer this over Qwen3 4B if your use case is STEM, code, or structured reasoning.
- **Gemma 4 E4B** — only option with native image + audio input at this tier. Choose it specifically if you need multimodal on CPU; otherwise Qwen3 4B is faster and smaller.

| Model | Size | License | Ollama Tag | Strength | Notes |
|-------|------|---------|------------|----------|-------|
| **Qwen3 4B** ★ | 4B | Apache 2.0 | `qwen3:4b` | General, multilingual | 2.5GB disk; hybrid thinking mode; 100+ languages |
| **Phi-4 Mini** ★ | 3.8B | MIT | `phi4-mini` | Reasoning, math | MMLU 68.5%; 128K context; ~20–30 tok/s on CPU |
| Gemma 4 E4B | ~4.5B active (8B MoE) | Apache 2.0 | `gemma4:e4b` | Multimodal (image + audio) | 9.6GB disk; designed for edge; 128K context |
| Gemma 4 E2B | ~2.3B active (5.1B MoE) | Apache 2.0 | `gemma4:e2b` | Ultra-lightweight, native audio | 7.2GB disk; best for ≤8GB RAM systems |
| Qwen3 1.7B | 1.7B | Apache 2.0 | `qwen3:1.7b` | Minimal footprint | 1.4GB disk; embedded/Raspberry Pi |
| DeepSeek-R1 7B (distill) | 7B | MIT | `deepseek-r1:7b` | Chain-of-thought reasoning | 4.7GB disk; distilled from 671B; strong math; slow on CPU |

---

### Tier 2 — Consumer GPU (6–16GB VRAM)

**→ Start here:** `ollama pull phi4` (14B, best quality-per-VRAM) or `ollama pull qwen3:8b` (if you have ≤8GB VRAM)

**Top picks:**
- **Phi-4 14B** — the standout at this tier. MIT license, MMLU ~80%, matches Llama 3.3 70B on many reasoning benchmarks at 5× smaller size, 9.1GB disk. Best choice if you have 10–16GB VRAM.
- **Qwen3 8B** — best fit for 8GB VRAM cards. 5.2GB disk, Apache 2.0, hybrid thinking mode, 100+ languages, tool calling. Fast and capable well above its size.
- **Gemma 4 26B (MoE)** — a surprise: 26B total parameters but only 3.8B active per token thanks to MoE. Fits 8GB VRAM at 4-bit quantization, includes multimodal (image input), and ranks #3 among all open models on the Chatbot Arena leaderboard. Try this if you have 8–16GB VRAM and want the highest benchmark quality.

| Model | Size | License | Ollama Tag | Strength | Notes |
|-------|------|---------|------------|----------|-------|
| **Phi-4 14B** ★ | 14B dense | MIT | `phi4` | Reasoning, STEM | 9.1GB disk; MMLU ~80%; best quality-per-VRAM in class |
| **Qwen3 8B** ★ | 8B | Apache 2.0 | `qwen3:8b` | Coding, multilingual | 5.2GB disk; fits 8GB VRAM; hybrid thinking mode |
| **Gemma 4 26B (MoE)** ★ | 26B total / 3.8B active | Apache 2.0 | `gemma4:26b` | General, multimodal | 18GB disk; fits 8GB VRAM at Q4; 256K context; #3 Arena |
| Qwen3 14B | 14B | Apache 2.0 | `qwen3:14b` | Long context, coding | 9.3GB disk; solid all-rounder |
| DeepSeek-R1 14B (distill) | 14B | MIT | `deepseek-r1:14b` | Reasoning, math | 9.0GB disk; chain-of-thought; fits 12GB VRAM |
| Phi-4 Mini Reasoning | 3.8B | MIT | `phi4-mini-reasoning:3.8b` | Math specialist | MATH-500: 94.6%; ideal for tutoring/STEM on 6GB VRAM |
| Mistral Small 3.2 | 24B dense | Apache 2.0 | `mistral-small3.2:24b` | Multilingual, vision | 128K context; text + image; 1.9M Ollama pulls; fits 16GB VRAM at Q4 |
| LFM2 (Liquid AI) | 24B total / 2B active | Apache 2.0 | `lfm2` | Fast inference, efficiency | MoE; 1.1M Ollama pulls; very low active parameter count → fast |

---

### Tier 3 — Prosumer GPU (20–24GB VRAM)

**→ Start here:** `ollama pull qwen3.6:27b` (coding/agentic) or `ollama pull gemma4:31b` (general + multimodal)

**Top picks:**
- **Qwen3.6 27B** — ranked #8 on Artificial Analysis (score 46), SWE-bench Verified 77.2%, vision-capable, 17GB disk at Q4. The clear pick for anything coding or agentic at this tier. Apache 2.0.
- **Gemma 4 31B** — ranked #11 on Artificial Analysis (score 39), MMLU-Pro 85.2%, AIME 89.2%, native multimodal. Best general-purpose + multimodal choice at this tier. Apache 2.0.
- **DeepSeek-R1 32B (distill)** — if your use case is pure reasoning, math, or science: this is the strongest reasoning model that fits 24GB VRAM. MIT license.

| Model | Size | License | Ollama Tag | Strength | Notes |
|-------|------|---------|------------|----------|-------|
| **Qwen3.6 27B** ★ | 27B dense | Apache 2.0 | `qwen3.6:27b` | Coding, agentic | AA score 46; SWE-bench 77.2%; vision; 256K context; Apr 2026 |
| **Gemma 4 31B** ★ | 30.7B dense | Apache 2.0 | `gemma4:31b` | General, multimodal | AA score 39; MMLU-Pro 85.2%; AIME 89.2%; 20GB disk at Q4 |
| **DeepSeek-R1 32B (distill)** ★ | 32B | MIT | `deepseek-r1:32b` | Math, reasoning | Best reasoning model fitting 24GB VRAM; 20GB disk |
| Qwen3 32B | 32B | Apache 2.0 | `qwen3:32b` | General, multilingual | 20GB disk; hybrid thinking mode |
| Qwen3-Coder-Next | 80B total / 3B active MoE | Apache 2.0 | `qwen3-coder-next` | Code generation | 1.3M Ollama pulls; 3B active → fast despite 80B total |
| Qwen3 30B (MoE) | 30B total / 3B active | Apache 2.0 | `qwen3:30b` | Fast inference | 19GB disk; very fast for quality level; 256K context |

---

### Tier 4 — Multi-GPU / Cloud (48GB+ VRAM)

*Most of these require multi-GPU infrastructure or run as cloud-routed models. Flagged accordingly.*

**→ Best self-hostable pick:** `ollama pull deepseek-r1:70b` (43GB, MIT, fully local) or `ollama pull llama3.3:70b` (43GB, coding)
**→ Best cloud-access pick:** `ollama pull kimi-k2.6` (cloud-routed, #1 on Artificial Analysis)

**Top picks:**
- **Kimi K2.6 (Moonshot AI)** — co-ranked #1 on Artificial Analysis (score 54). 1T total / 32B active MoE, SWE-bench 80.2%, AIME 2026 96.4%. Modified MIT license. Cloud-only on Ollama but accessible via API or Hugging Face weights for those with the infrastructure.
- **DeepSeek-R1 70B (distill)** — the best fully self-hostable model at this tier. 43GB disk, MIT license, MMLU 90.8%, strongest open reasoning model under 100B. Runs on a single A100 80GB or two 40GB cards.
- **DeepSeek V4 Pro** — ranked #3 on Artificial Analysis (score 52), frontier-level coding and math (SWE-bench 80.6%), MIT license. Requires 8× H200 for self-hosting; accessible via API otherwise.

| Model | AA Score | Size | License | Ollama Tag | Self-hostable? | Strength |
|-------|----------|------|---------|------------|----------------|---------|
| **Kimi K2.6** ★ (Moonshot AI) | 54 | 1T total / 32B active | Modified MIT | `kimi-k2.6` (cloud) | No (cloud-only weights) | SWE-bench 80.2%; AIME 96.4%; agentic coding |
| MiMo-V2.5-Pro (Xiaomi) | 54 | 1.02T total / 42B active | MIT | Not on Ollama | Yes (HF weights, massive infra) | 1M context; 40–60% more token-efficient than GPT-5 on agentic evals |
| DeepSeek V4 Pro | 52 | 1.6T total / 49B active | MIT | `deepseek-v4-pro` (cloud) | Yes (8× H200) | SWE-bench 80.6%; MMLU-Pro 87.5% |
| Qwen3.6 Max / Plus | 52 / 50 | — | Apache 2.0 | `qwen3.6` (cloud) | No (preview/cloud only) | Frontier reasoning; not yet released locally |
| Qwen3.5 122B (MoE) | 42 | 122B total / 10B active | Apache 2.0 | `qwen3.5:122b` | Yes (2× A100 80GB) | Strong coding + reasoning; good VRAM efficiency via MoE |
| Qwen3.5 397B (MoE) | 45 | 397B total / 17B active | Apache 2.0 | `qwen3.5:397b` (cloud) | Yes (multi-GPU) | Frontier reasoning; AIME competitive with o3-mini |
| GLM-5.1 (Z.AI) | 51 | 754B MoE | MIT | `glm-5.1` (cloud) | Yes (massive infra) | #1 open model on SWE-bench Pro; agentic engineering |
| MiniMax-M2.7 | 50 | 229B dense | **Custom — NO commercial self-hosting** | `minimax-m2.7` (cloud) | No without MiniMax approval | SWE-Pro 56.22%; low API cost ($0.30/M input) |
| **DeepSeek-R1 70B (distill)** ★ | — | 70B | MIT | `deepseek-r1:70b` | **Yes (43GB, 1× A100 80GB)** | MMLU 90.8%; best self-hostable reasoning model |
| **Llama 3.3 70B** ★ | — | 70B dense | Llama 3.3 Community | `llama3.3:70b` | **Yes (43GB)** | MMLU 86%; best open coding model; 3.8M Ollama pulls |
| Llama 4 Scout | 14 | 109B total / 17B active | Llama 4 Community | `llama4:16x17b` | Yes (67GB) | 10M context; multimodal; EU restrictions apply |
| NVIDIA Nemotron 3 Super | 36 | 120B total / 12B active | NVIDIA Open (commercial OK) | `nemotron-3-super` | Yes (87GB, 2× H100) | #1 on DeepResearch Bench; 91.75% RULER 1M-context |
| Mistral Medium 3.5 | — | 128B dense | Modified MIT | `mistral-medium-3.5` | Yes (80GB) | SWE-bench 77.6%; built-in coding agent; 256K context |

### Notable Model Family Updates

**Gemma 4 (Google, April 2, 2026)**
Released under Apache 2.0 with four variants (E2B, E4B, 26B MoE, 31B dense), adding native multimodal support (images, video, audio) across all sizes — a major leap from Gemma 3's text-only smaller models. The 31B dense variant ranks #3 among all open models on the Arena leaderboard, beating Llama 4 Maverick on math and coding despite being far smaller. The MoE 26B activates only 3.8B parameters per token, making it runnable on consumer GPUs.

**Qwen 3 / Qwen 3.5 / Qwen 3.6 (Alibaba, 2025–2026)**
Qwen3 launched April 29, 2025 with eight dense and MoE sizes (0.6B–235B), introducing a hybrid "thinking/non-thinking" mode toggle. Qwen3.5 (early 2026) expanded to multimodal capabilities and larger MoE variants up to 397B-A17B. Qwen3.6-27B (April 22, 2026) is a dense 27B model achieving 77.2% SWE-bench Verified — outperforming models many times its size on coding. All releases under Apache 2.0.

**Llama 3.x / Llama 4 (Meta, 2025–2026)**
Llama 3.3 70B (late 2024) remains competitive on coding benchmarks. Llama 4 Scout and Maverick (April 2025) are the first natively multimodal Meta models, using MoE with 17B active parameters and context windows of 10M and 1M tokens. Llama 4 Behemoth (~2T parameters) was previewed but weights unreleased as of May 2026. License restricts EU-domiciled users and companies with 700M+ MAU.

**DeepSeek (V3 → V4, 2024–2026)**
DeepSeek V3 (late 2024) established that open-weight models could match frontier proprietary models on coding at a fraction of compute cost. DeepSeek V4 Pro (April 2026, MIT) scales to 1.6T parameters with 49B active, achieving 80.6% SWE-bench Verified and 87.5% MMLU-Pro — top-ranked open model as of May 2026. DeepSeek-R1 (January 2025) remains the reference reasoning model; all distilled variants (1.5B–671B) are MIT-licensed.

**Phi-4 (Microsoft, 2025)**
Phi-4 (14B, late 2024/early 2025, MIT) achieves MMLU 80.4% — competitive with models 5× its size. Phi-4-mini (3.8B) brought strong reasoning to the sub-4GB tier. Phi-4-mini-reasoning (April 2025) is a math specialist scoring 94.6% on MATH-500. No Phi-5 as of May 2026.

**Mistral (2025–2026)**
Shifted to Apache 2.0 starting with Mistral Small 3 (24B, January 2025). The current small-tier model is **Mistral Small 3.2** (24B, Apache 2.0, 1.9M Ollama pulls), which added vision and improved instruction following over 3.1. **Magistral** (24B, Apache 2.0, `magistral` on Ollama, 1.4M pulls) is Mistral's thinking/reasoning variant in the same size class. Mistral Medium 3.5 (April 2026, Modified MIT, 128B dense) is the flagship self-hostable model with a built-in coding agent achieving 77.6% SWE-bench Verified.

### Sources
- https://till-freitag.com/en/blog/open-source-llm-comparison
- https://lushbinary.com/blog/best-open-source-llms-april-2026-comparison-guide/ (April 2026)
- https://codersera.com/blog/best-open-source-llm-2026-llama-4-qwen-3-5-deepseek-v4-gemma-4-mistral/ (May 2026)
- https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/ (April 2026)
- https://qwenlm.github.io/blog/qwen3/ (April 2025)
- https://qwen.ai/blog?id=qwen3.6-27b (April 2026)
- https://ai.meta.com/blog/llama-4-multimodal-intelligence/ (April 2025)
- https://api-docs.deepseek.com/news/news260424 (April 2026)
- https://techcommunity.microsoft.com/blog/educatordeveloperblog/welcome-to-the-new-phi-4-models---microsoft-phi-4-mini--phi-4-multimodal/4386037
- https://mistral.ai/news/mistral-3 (December 2025)
- https://ollama.com/library (checked May 2026)
- https://benchlm.ai/blog/posts/best-open-source-llm
- https://artificialanalysis.ai (open-source model quality rankings, May 2026)
- https://ollama.com/library (download counts checked May 8, 2026)
- https://huggingface.co/XiaomiMiMo/MiMo-V2.5-Pro (MiMo-V2.5-Pro)
- https://huggingface.co/MiniMaxAI/MiniMax-M2.7 (MiniMax-M2.7 license and specs)
- https://huggingface.co/moonshotai/Kimi-K2.6 (Kimi K2.6)
- https://ollama.com/library/nemotron-3-super (NVIDIA Nemotron 3 Super)
- https://ollama.com/library/kimi-k2.6 (Kimi K2.6 Ollama tag)
- https://mistral.ai/news/mistral-small-3 (Mistral Small 3.2)

---

## D. Community-Validated Tools for Local LLM Experimentation

### Tool Overview

| Tool | Platforms | Target User | API Compat | Maintained | Community Sentiment |
|------|-----------|-------------|------------|------------|---------------------|
| Ollama | Mac / Win / Linux | Developers, API builders | ✓ (always-on at `:11434/v1`) | Very active (171k GitHub stars, weekly releases) | "De facto standard"; praised for zero-config CLI and ecosystem integrations; criticized for no built-in chat UI and occasional memory-overload crashes |
| LM Studio | Mac / Win / Linux | Beginners, model explorers | ✓ (must be manually started) | Active (proprietary; 3M+ cumulative downloads) | "Best model discovery experience"; praised for polished GUI and HuggingFace catalog; flagged for 200–400 MB RAM overhead and API server session timeout bugs |
| Open WebUI | Mac / Win / Linux (Docker or pip) | Developers wanting a chat UI | ✓ (connects to Ollama or any OpenAI-compatible backend) | Very active (135k GitHub stars, multiple updates/week) | Standard Ollama companion; praised for RAG, multi-user auth, voice I/O, plugin system; criticized for fast-moving releases requiring version-pinning |
| vLLM | Linux only | Production/backend teams | ✓ (OpenAI-compatible) | Very active (45k GitHub stars, v0.17.1 March 2026) | "In a different category" for throughput (2,300+ tok/s on H100); community consensus: graduate to vLLM only when serving concurrent users at scale |
| llama.cpp (raw) | All platforms + mobile / Raspberry Pi | Power users, embedded devs | ✓ (server mode; some streaming/tools quirks) | Very active (75k GitHub stars, MIT; underlying engine for Ollama and LM Studio) | "The engine underneath almost everything"; praised for literal any-hardware support; no GUI, steeper setup than Ollama |
| Hugging Face Transformers | Mac / Win / Linux (Python) | Researchers, fine-tuners | ✗ (Python library, no HTTP API by default) | Very active | Preferred for fine-tuning, PEFT, model internals; community: use Ollama for inference, HF Transformers only when training/fine-tuning |
| Jan.ai | Mac / Win / Linux | Privacy-focused users, teams | ✓ (OpenAI + Anthropic Messages API) | Active (~25k GitHub stars, AGPL-3.0) | "Open-source alternative to LM Studio"; MCP integration; smaller community; recommended when open-source verifiability is a requirement |
| GPT4All | Mac / Win / Linux | Beginners | ✓ (OpenAI-compatible endpoint added) | **Stalled** — last release v3.10.0, Feb 25 2025 | Community raised "Is GPT4All dead?" in August 2025; no new model support since Feb 2025; still functional for LocalDocs RAG but superseded |
| AnythingLLM | Mac / Win / Linux / Docker | Business teams, document-heavy RAG | ✓ (delegates to Ollama/OpenAI backends) | Active (59k GitHub stars) | Praised for multi-user workspaces, RBAC, and superior RAG UX for non-technical teams |
| llamafile (Mozilla) | All platforms, single executable | Educators, zero-install demos | ✓ (OpenAI + Anthropic Messages API, v0.10.0) | Active (Metal returned Dec 2025, CUDA returned Feb 2026) | Zero-install approach; ideal for course demos and air-gapped environments; not intended for production serving |
| text-generation-webui | Mac / Win / Linux | Power users, fine-tuners | ✓ (OpenAI-compatible) | Active but complex | Still best for ExLlama v2/AutoGPTQ/LoRA fine-tuning; community consensus: Ollama has displaced it for most inference use cases |

### Community Consensus Summary

**Ollama has become the infrastructure default for local LLM work.** Across r/LocalLLaMA (636k+ members), HackerNews threads, and developer blogs, Ollama appears in official documentation from Meta, Google, and Alibaba, and is the assumed backend in LangChain/LlamaIndex tutorials. Multiple 2026 comparisons independently conclude Ollama is "the de facto standard for developer tooling integrations." Real-world warts reported by practitioners: memory-overload crashes when too many models are loaded, and LM Studio has "a very annoying hard timeout of 2–3 minutes on its API server" — solvable issues, not fatal ones.

**The dominant community stack is Ollama + Open WebUI, with LM Studio as the beginner on-ramp.** The techsy.io ranked list, mljourney comparison, and kunalganglani 2026 setup guide converge on the same architecture: Ollama for model serving and API access, Open WebUI for browser-based chat (especially for non-coding users or for RAG), and LM Studio as the recommended entry point for users who do not want to touch a terminal. Multiple sources note these are complementary — "most power users keep both installed." r/LocalLLaMA singles out LM Studio specifically for Apple Silicon users and model discovery, and Ollama for everything programmatic.

**vLLM, Hugging Face Transformers, and raw llama.cpp occupy specialized niches.** The dev.to inference engine comparison (2026) frames vLLM as the tool you adopt "when concurrent requests must fly," not for solo experimentation. HF Transformers is consistently described as right only when you need fine-tuning or model internals — not for inference serving. GPT4All, while historically important as an early popularizer, has stalled; a GitHub issue from August 2025 explicitly asks "Is GPT4All dead?" with no maintainer response resolving the concern.

### Tool Map by User Profile

**"I just want to chat with a local model, no coding"**
→ **LM Studio** or **Jan.ai** — point-and-click model download, built-in chat UI, no terminal needed. LM Studio is more polished (especially on Apple Silicon); Jan.ai is fully open-source if that matters to you. Both let you browse and download models from Hugging Face directly.

**"I want to run models from a browser and use them like ChatGPT, locally"**
→ **Open WebUI** on top of Ollama — full chat interface with RAG, voice I/O, multi-user support, and a plugin system. Install Ollama first (`curl -fsSL https://ollama.com/install.sh | sh`), then Open WebUI with one Docker command.

**"I'm a developer and want an API I can call from my code"**
→ **Ollama** — always-on OpenAI-compatible endpoint at `localhost:11434/v1`. Drop-in replacement for the OpenAI Python SDK: just set `base_url="http://localhost:11434/v1"`. Every major framework (LangChain, LlamaIndex, Continue.dev) targets this endpoint first.

**"I'm building an application and need to serve it to multiple users"**
→ **Ollama** for development and single-user load; **vLLM** when you need to handle concurrent requests at scale (5+ simultaneous users). vLLM requires a dedicated Linux machine with NVIDIA GPU but delivers 2,300+ tokens/sec on H100-class hardware.

**"I want maximum control — custom GGUF files, specific quantization, no daemon process"**
→ **llama-cpp-python** — load any GGUF from Hugging Face directly (`Llama(model_path="...")`), explicit control over GPU layers, Vulkan support on AMD without recompiling, near-zero Python overhead. Steeper install but no restrictions on which models you can run.

**"I work with sensitive documents and need everything air-gapped / verifiable"**
→ **Jan.ai** (open-source, AGPL, auditable) or **llamafile** (single binary, zero install, download and run on any OS). llamafile is especially good for sharing a model with colleagues who shouldn't need to install anything.

**"I need fine-tuning or access to model internals (logits, embeddings, training)"**
→ **Hugging Face Transformers** — the only option here. Not for inference serving, but essential for any fine-tuning, PEFT, LoRA, or custom training work.

**"I'm doing research or power-user workflows on a high-end machine"**
→ **raw llama.cpp** for maximum performance and hardware flexibility; combine with **text-generation-webui** if you want ExLlama v2 or AutoGPTQ backends alongside a UI.

### Tools to Drop from the Radar

**GPT4All** — Last substantive release February 2025; community raised active-maintenance concerns August 2025; no model support for models released after early 2025. Still functional for its specific LocalDocs RAG use case but superseded by Ollama + Open WebUI for general use.

**text-generation-webui (oobabooga)** — Still maintained; still best for ExLlama v2, AutoGPTQ, or LoRA fine-tuning from the same tool. But community consensus is that Ollama has displaced it for the vast majority of inference use cases. The complex setup is not justified unless students specifically need these advanced backends.

**PrivateGPT** — Appears in some older guides; rarely mentioned in 2025–2026 community discussions. AnythingLLM and Open WebUI have absorbed its use case.

### Sources
- https://techsy.io/en/blog/best-tools-run-llms-locally (2026)
- https://mljourney.com/ollama-vs-lm-studio-in-2026-which-should-you-use/ (2026)
- https://appstackbuilder.com/blog/ollama-vs-lm-studio-2026 (2026)
- https://pinggy.io/blog/top_5_local_llm_tools_and_models/ (2026)
- https://dev.to/agdex_ai/5-best-open-source-llm-inference-engines-in-2026-vllm-ollama-llamacpp-more-2811 (2026)
- https://www.kunalganglani.com/blog/running-local-llms-2026-hardware-setup-guide (2026)
- https://wz-it.com/en/blog/open-webui-vs-anythingllm-comparison/ (2025)
- https://github.com/nomic-ai/gpt4all/issues/3605 — "Is GPT4All dead?" (August 2025)
- https://github.com/nomic-ai/gpt4all/releases (last release v3.10.0, Feb 25 2025)
- https://news.ycombinator.com/item?id=44837130 — Ask HN: Are you running local LLMs? (2025)
- https://craftrigs.com/articles/106-llamafile-0-10-0-gpu-speed-review/ (2026)
- https://collabnix.com/hugging-face-vs-ollama-the-complete-technical-deep-dive-guide-for-local-ai-development-in-2025/ (2025)

---

*End of report. See Course Update Recommendations table at the top for action items.*
