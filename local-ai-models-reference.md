# Local AI Models — Quick Reference

A short, practical shortlist of local models and what each is best for. Transcribed from the "Local AI Models" course slide (2026-08-14). Split between the presenter's day-to-day picks on a Mac Studio and other popular models worth knowing.

## Lucas's picks (Mac Studio)

| Model | Size / VRAM | Best for |
|-------|-------------|----------|
| **Qwen 3.6 35B** | 35B | General-purpose daily driver |
| **Gemma 4 26B** | 26B | Fast, agentic tool use |

## Other popular models

| Model | Size / VRAM | Best for |
|-------|-------------|----------|
| **Qwen3.5 27B** | ~16GB+ VRAM — runs on M4 Pro/Max | General-purpose alternative |
| **Qwen 3 8B** | ~8GB VRAM — runs on any modern laptop | Budget / lightweight tasks |
| **Hermes 4 36B** | ~22GB — 24GB GPU minimum | Nous's own tool-calling model |
| **Mistral Small 24B** | ~14–16GB VRAM | Efficient all-rounder |
| **Llama 4 Maverick** | 400B+ params — requires very powerful device, or cloud recommended | Strongest open tool-calling |
| **DeepSeek V4** | 1.6T params — requires very powerful device, or cloud recommended | Frontier coding / reasoning |
| **GLM-5** | 744B params, 1M context — requires very powerful device, or cloud recommended | Repo-scale agent loops |

## Notes

- **Sizing rule of thumb:** anything at or under ~24GB VRAM runs comfortably on a single consumer GPU or a well-specced Apple Silicon machine. The 400B+ / 1T+ / 744B models (Llama 4 Maverick, DeepSeek V4, GLM-5) are frontier-scale — expect to run them via cloud/API rather than fully local unless you have serious hardware.
- **Budget fallback:** Qwen 3 8B (~8GB VRAM) is the go-to for constrained laptops and cheap VPS boxes.
- **Tool use:** Gemma 4 26B and Hermes 4 36B are the strongest tool-calling picks in the locally-runnable range.
