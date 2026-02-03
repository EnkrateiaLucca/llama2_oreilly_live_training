# LLM Hallucination Metrics for Evaluation Pipelines

There are **multiple** hallucination metrics used in LLM evaluation pipelines—no single universal metric exists. The field uses a combination of benchmarks, model-based detectors, and scoring methods.

---

## Benchmarks (Test Datasets)

- **TruthfulQA** — Evaluates whether LLMs produce answers that mimic human false beliefs; tests factual accuracy across domains
- **HaluEval** — Large-scale benchmark with 35K generated and human-annotated hallucinated samples; covers QA, dialogue, and summarization tasks
- **HalluLens** — Distinguishes between intrinsic (contradicting source) and extrinsic (unverifiable) hallucinations; uses dynamic test generation to prevent data leakage
- **Mu-SHROOM** — SemEval benchmark for multilingual hallucination evaluation
- **CCHall** — ACL 2025 benchmark for multimodal reasoning hallucinations
- **REFIND** — Uses Context Sensitivity Ratio (CSR) comparing token probabilities with/without retrieved documents

---

## Model-Based Detection Metrics

- **HHEM (Hughes Hallucination Evaluation Model)** — Vectara's efficient classifier that scores factual consistency in summarization tasks; used in their public Hallucination Leaderboard
- **SelfCheckGPT** — Generates multiple stochastic samples and checks consistency; computes BertScore, n-gram overlap, and QA-based scores
- **LLM-Check** — Analyzes internal LLM representations (eigenanalysis of hidden states/attention) plus uncertainty quantification of output tokens
- **QAG Score (Question-Answer Generation)** — Uses LLM reasoning with closed yes/no questions to evaluate factuality without direct score generation

---

## Traditional & Computed Metrics

- **CHAIR (Caption Hallucination Assessment with Image Relevance)** — CHAIRi (per-instance) and CHAIRs (per-sentence) measure object hallucination in image captioning
- **Hallucination Rate** — Simple percentage of responses containing hallucinated content
- **Expected Calibration Error (ECE)** — Measures gap between model confidence and actual accuracy
- **NLI-based Factuality Scores** — Uses natural language inference models to check if generated text is entailed by source documents
- **Context Sensitivity Ratio (CSR)** — Compares generation probability with vs. without context to flag likely hallucinations

---

## Leaderboards & Aggregated Evaluation

- **Vectara Hallucination Leaderboard** — Ranks LLMs on summarization faithfulness using HHEM-2.3
- **Hugging Face Hallucinations Leaderboard** — Evaluates across NQ Open, TriviaQA, TruthfulQA (MC1, MC2, Generative) with 8-shot and 64-shot settings

---

## Key Challenges

- No universal metric works across all domains and task types
- Traditional lexical metrics (BLEU, ROUGE, METEOR) fail to capture semantic grounding
- Subtle, high-confidence hallucinations remain difficult to detect
- Standard training rewards confident guessing over admitting uncertainty

---

## Sources

- [Comprehensive Survey of Hallucination in LLMs (arXiv, Oct 2025)](https://arxiv.org/abs/2510.06265)
- [HalluLens Benchmark (ACL 2025)](https://arxiv.org/html/2504.17550v1)
- [Vectara Hallucination Leaderboard (GitHub)](https://github.com/vectara/hallucination-leaderboard)
- [HaluEval Benchmark (GitHub)](https://github.com/RUCAIBox/HaluEval)
- [Hugging Face Hallucinations Leaderboard](https://huggingface.co/blog/leaderboard-hallucinations)
- [Frontiers in AI Survey on LLM Hallucinations (2025)](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2025.1622292/full)
- [Awesome Hallucination Detection Papers (EdinburghNLP)](https://github.com/EdinburghNLP/awesome-hallucination-detection)
- [Confident AI LLM Evaluation Guide](https://www.confident-ai.com/blog/llm-evaluation-metrics-everything-you-need-for-llm-evaluation)
