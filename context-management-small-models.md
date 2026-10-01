# Context Management for Small-Model Agent Harnesses (Python)

A compressed, opinionated reference for building agent loops around 1B–8B local models (Gemma 3, Llama 3.2, Qwen 2.5/3, Phi-4-mini, SmolLM2) served via Ollama / llama.cpp / vLLM.

> Note on "Gemma 4": as of early 2026 the latest publicly released family is **Gemma 3** (1B / 4B / 12B / 27B, March 2025). Treat "Gemma 4" as either a typo for Gemma 3 or an unreleased version — pin to Gemma 3 in code.

---

## 1. The core problem: nominal ≠ effective context

| Model | Nominal window | Practical "good" working zone* |
|---|---|---|
| Gemma 3 4B | 128K | ~4–8K |
| Llama 3.2 3B | 128K | ~4–8K |
| Qwen 2.5 7B | 128K (YaRN) | ~8–16K |
| Phi-4-mini 3.8B | 128K | ~4–8K |
| SmolLM2 1.7B | 8K | ~2–4K |

\* Rule of thumb from RULER / NoLiMa-style long-context benchmarks: small models lose >30% accuracy well before their advertised window. **Design for the working zone, not the marketing number.**

Quantization makes this worse — Q4_K_M typically costs another 5–15% on long-context recall vs Q8/F16. If you must run quantized + long context, prefer Q5_K_M or Q6_K.

---

## 2. The four levers (in priority order)

Most "my agent breaks after 5 turns" problems are solved by levers 1–2. Reach for 3–4 only when needed.

### Lever 1 — Shrink what enters the window

- **Trim tool outputs aggressively** before re-injection. A `requests.get()` body or a 50-line `ls` output is 90% noise. Extract structured fields (`title`, `top_3_results`, `error_only`) and discard the rest.
- **Cap per-turn tool result size** (e.g. 500–1000 tokens). Spill the full result to disk; pass back a path + summary.
- **One tool per turn** for <7B models. Parallel/batched tool use is brittle below Qwen 2.5 7B level.

```python
def trim_tool_result(raw: str, max_tokens: int = 800) -> str:
    if len(raw) // 4 <= max_tokens:           # ~4 chars/token heuristic
        return raw
    head = raw[: max_tokens * 2]
    tail = raw[-max_tokens * 2 :]
    return f"{head}\n...[truncated {len(raw)} chars]...\n{tail}"
```

### Lever 2 — Sliding window with pinned anchors

The simplest pattern that works. Keep:

1. System prompt (pinned)
2. Original user task (pinned)
3. Last N turns (rolling, FIFO)
4. A short "running notes" scratchpad the model edits each turn

```python
def assemble_messages(system, task, history, scratchpad, n_recent=4):
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"TASK: {task}\n\nNOTES:\n{scratchpad}"},
        *history[-n_recent * 2:],   # last N user+assistant pairs
    ]
```

Why this beats naive concatenation: pinning the task fights the "lost in the middle" effect, and the scratchpad gives the model a place to compress its own state.

### Lever 3 — Summarization / compaction triggers

Fire compaction when **any** of these hit:

- Total tokens > 70% of working zone
- History length > 8 turns
- Tool result was >2K tokens

Use a *separate, smaller* model call (same Gemma 3 4B is fine — it's cheap locally) with a templated prompt that outputs **structured** notes, not prose. Structured beats prose because the next turn re-reads it.

```python
COMPACT_PROMPT = """Compress the conversation below into JSON:
{{"facts_learned": [...], "open_questions": [...], "next_step": "..."}}.
Drop pleasantries, raw tool dumps, and resolved sub-goals.

CONVERSATION:
{history}"""
```

### Lever 4 — External memory (only when needed)

For chats >20 turns or doc-grounded agents:

- **SQLite + FTS5** for keyed scratch facts (overkill-free, no embedding model needed)
- **Vector store** (Chroma, LanceDB, sqlite-vec) only when you need semantic recall across sessions
- **File-as-memory** (`scratchpad.md` the agent reads/writes) is underrated — works great with small models because writing to a known location is simpler than retrieval

Letta (ex-MemGPT) implements the canonical "OS-style paging" version of this; worth reading the source even if you don't adopt it.

---

## 3. Python stack picks (early 2026)

| Need | Pick | Why |
|---|---|---|
| Local inference | **Ollama** for prototyping, **llama.cpp server** or **vLLM** for production | Ollama owns the DX; vLLM owns throughput |
| Structured output | **Outlines** or **llama.cpp grammars** (GBNF) | Hard-constrains JSON — critical for small models. `format=json` in Ollama is weaker. |
| Agent loop | Hand-rolled `while` loop > frameworks for small models | LangGraph / LlamaIndex assume frontier-model reliability. Roll your own ~80-line loop. |
| Memory layer | **Letta** if you need it, else SQLite | Skip vector DBs until proven necessary |
| Validation | **Pydantic** for tool args + results | Reject malformed tool calls; ask model to retry with the validation error appended |

**ReAct is a trap below 7B.** Don't use ReAct-style "Thought / Action / Observation" prose templates with Gemma 3 4B — they hallucinate tool calls mid-thought. Instead:

- Use the model's **native tool-calling format** (Qwen 2.5, Llama 3.2, Gemma 3 all have one)
- Or force JSON via grammar/Outlines
- Single-tool-per-turn, hard cap on total iterations (5–8)

---

## 4. KV-cache reuse (the underused win)

Local inference gives you something API users don't: **direct KV-cache control**.

- **llama.cpp**: `--prompt-cache <file>` persists the cache across calls; same prefix → near-instant prefill.
- **vLLM**: automatic prefix caching (`enable_prefix_caching=True`) — system prompt + tool definitions stay cached across requests.
- **Ollama**: caches the last conversation per model automatically; keep messages append-only (don't rewrite history) to preserve hits.

Practical implication: put **stable content first** (system prompt → tool schemas → pinned task), volatile content last. A reordered history busts the cache.

---

## 5. Failure modes & quick fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| Agent loops on same tool call | No state diff between turns | Append `"PREVIOUS_ATTEMPTS: [...]"` block; lower temp to 0.3 |
| Output truncates mid-JSON | Hitting `num_predict` / `max_tokens` | Set explicitly; don't rely on defaults (Ollama default is 128) |
| Quality cliff after ~3K tokens | Effective-context limit | Trigger compaction at 2.5K, not at the model's nominal max |
| Hallucinated tool names | ReAct-style prose prompting | Switch to grammar-constrained JSON tool calls |
| Tool args off-by-one / wrong types | Small-model JSON drift | Pydantic-validate; on failure, re-prompt with the validation error |
| Repeats system instructions in output | System prompt too long / placed last | Shorten; put first; use `<<<` style delimiters |

---

## 6. Minimal agent-loop skeleton

```python
# /// script
# requires-python = ">=3.12"
# dependencies = ["ollama", "pydantic"]
# ///
import json, ollama
from pydantic import BaseModel, ValidationError

MODEL = "gemma3:4b"
MAX_STEPS = 6
WORKING_ZONE_TOKENS = 4000

class ToolCall(BaseModel):
    name: str
    args: dict

def approx_tokens(msgs): return sum(len(m["content"]) for m in msgs) // 4

def compact(history):
    out = ollama.chat(model=MODEL, messages=[
        {"role": "system", "content": "Compress to JSON: facts, open_questions, next_step."},
        {"role": "user", "content": json.dumps(history)},
    ], options={"temperature": 0.1})
    return out["message"]["content"]

def run(task, tools):
    system = "Reply ONLY with JSON: {\"name\": ..., \"args\": {...}} or {\"final\": \"...\"}."
    history, scratch = [], ""
    for step in range(MAX_STEPS):
        msgs = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"TASK: {task}\nNOTES: {scratch}"},
            *history[-8:],
        ]
        if approx_tokens(msgs) > WORKING_ZONE_TOKENS * 0.7:
            scratch = compact(history); history = history[-2:]
        r = ollama.chat(model=MODEL, messages=msgs, format="json",
                        options={"temperature": 0.2})["message"]["content"]
        data = json.loads(r)
        if "final" in data: return data["final"]
        try:
            call = ToolCall(**data)
            result = tools[call.name](**call.args)
        except (ValidationError, KeyError) as e:
            result = f"ERROR: {e}"
        history += [{"role": "assistant", "content": r},
                    {"role": "user", "content": f"TOOL_RESULT: {str(result)[:1500]}"}]
    return "max_steps_reached"
```

Run with `uv run agent.py`. Swap `gemma3:4b` for `qwen2.5:7b` if you want the next tier of reliability.

---

## 7. Cheat sheet

1. **Pin the task at the top of every turn.** Small models forget what they're doing.
2. **Trim tool results to <1K tokens before re-injection.**
3. **Compact at 70% of working zone, not at the nominal max.**
4. **Single tool per turn, hard step cap.**
5. **Grammar-constrained JSON > ReAct prose** for <7B.
6. **Append-only history** so KV-prefix caching hits.
7. **Q5_K_M minimum** if you care about long-context behavior.
8. **Pydantic-validate tool args; re-prompt with the error** on failure.

---

## References to chase

- Google, *Gemma 3 Technical Report* (March 2025) — context window + recall tradeoffs
- NVIDIA, *RULER: What's the Real Context Size of Your Long-Context Language Models?* (2024)
- Adams et al., *NoLiMa: Long-Context Evaluation Beyond Literal Matching* (2025)
- Letta (ex-MemGPT) docs — `github.com/letta-ai/letta`
- llama.cpp `--prompt-cache` and GBNF grammar docs
- vLLM automatic prefix caching: `docs.vllm.ai`
- Outlines structured generation: `github.com/dottxt-ai/outlines`
