Spike S5 runs (2026-10-21), RTX 3090, CUDA provider, ortgenai-fp32x, Microsoft.ML.OnnxRuntimeGenAI.Cuda 0.17.1,
greedy, thinking off, max 256 new tokens. Model directory: the published ortgenai-fp32x files plus
transformers-5.18.0/tokenizer.json and Qwen/Qwen3.5-0.8B@2fc06364 tokenizer_config.json (GenAI's guidance needs both).

- runs.jsonl / summary.md: free vs guided (`--out runs.jsonl`, two chunks of 50 cases).
- aa.jsonl / summary-aa.md: A/A noise control (`--aa`), free vs a repeat of free.

Regenerate a summary with `python3 -I summarize.py results/runs.jsonl`.
