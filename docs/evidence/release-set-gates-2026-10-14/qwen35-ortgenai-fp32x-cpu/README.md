# qwen3.5-0.8b `ortgenai-fp32x-cpu` on the P-030 release sets

Run on 2026-10-14 at `10c016c` (main after #26). The release prompts and the reference fixture are at `qwen3.5-0.8b/release/release-prompts/` (32 oasst1 root prompts, 64 new tokens each). The tokenizer release set is `qwen3.5-0.8b/release/tokenizer-release/` (232 holdout rows), and GSM8K is `qwen3.5-0.8b/release/task-quality/gsm8k-test-v1/` (50 problems). The evaluator verified the protected evidence against Git. Nothing under `protected/` was changed.

| Gate | Set | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- | --- |
| G1 | tokenizer release (232 rows) | **passed** | ids exact 232/232 (`discrete.prompt_ids`). Tokens, masks, offsets in three units and decode, with and without special tokens, are also exact (measured). | [`g1-tokenizer-release/report.md`](g1-tokenizer-release/report.md) |
| G1 | release prompts (32) | **passed** | owned template rendering exact 32/32; owned tokenizer ids exact 32/32 | [`g1-g2-release-prompts/report.md`](g1-g2-release-prompts/report.md) |
| G2 | release prompts (32) | **not-run** | Greedy ids exact on 32/32, with no divergence despite reference steps with margins down to 5.7e-4. Finish reason 32/32. Step-0 argmax 32/32. Step-0 top-5 log-probability 4.54e-4 (2e-3). Step-0 log-probability on the reference top 64: 5.87e-4 (bound 5e-3; full vocabulary not run, below). Streaming reassembly and usage exact. Provider options `{}`. **Not run:** the full-vocabulary step-0 bound, and the three runtime-semantics keys (cached vs uncached, chunked prefill, concurrency). | same |
| G3 | GSM8K, 50 problems, greedy | **passed** | exact match **0.64** (32/50; minimum 0.60). The transformers CPU fp32 reference also scores 0.64, and no row's correctness differs from it (`reference-scores/transformers-cpu-fp32.json`). The budget names the data digest. | [`g3-gsm8k/report.md`](g3-gsm8k/report.md) |

## Why G2 is not-run

1. **Runtime-semantics keys cannot run on the release prompts.** The profile's binding tolerances include `float.cached_vs_uncached_logprob_max_abs`, `float.chunked_vs_one_call_prefill_logprob_max_abs` and `discrete.concurrent_requests`. `Starlings.Qwen35.Fidelity --modes semantics` produces them only on the development prompts. Its checks pick prompts by id (`en-short`, `en-list`, `zh`, `code-cs`, `thinking-on`, `multi-turn`, `long-license`), because they rely on properties of those prompts: a stop-terminated reference for the max-token boundary, a multi-turn prompt, a long prompt for chunked prefill. On the release prompts the run stops with `Sequence contains no matching element`. All 32 release prompts are single-turn, 64 tokens, and finish on `length`. Redesigning those checks would change what the evidence measures, so it is not done here, and I did not mix a development `semantics.json` into the release run. The maintainer has to decide whether these keys are fixture-independent profile checks (evaluated from a separate development-prompt run) or need a release-prompt design. Until then, no `ortgenai-*` profile can pass G2 on the release prompts.
2. **The full step-0 vectors were not reproduced bit for bit.** The fixture keeps the top 64 per prompt and the SHA-256 of each full vector. `run_reference.py` on the release prompts (`/home/mark/Projects/AI-models/gen/venvs/gen-ref`: torch 2.14.1+cu130 run on CPU, transformers 5.18.0, 4 threads) gave vectors whose digests differ from the recorded ones, so the evaluator refused them. The reference used torch 2.14.1+cpu with 6 threads. The step-0 check was therefore measured on the 64 reference ids only.

## Commands

```sh
RP=model-pipeline/protected/qwen3.5-0.8b/release/release-prompts
MAN=model-pipeline/registry/qwen3.5-0.8b/ortgenai-fp32x-cpu/manifest.yaml
TOK=model-pipeline/protected/qwen3.5-0.8b/release/tokenizer-release/tokenizer.jsonl
dotnet ai-dotnet/tools/Starlings.Fidelity.Tool/bin/Release/net10.0/Starlings.Fidelity.Tool.dll tokenizer --model qwen35 \
  --tokenizer <snapshot>/transformers-5.18.0/tokenizer.json --fixtures $TOK --out <run>/g1-tokenizer
model-pipeline evaluate $MAN --tokenizer-fixtures $TOK --results <run>/g1-tokenizer --out <report>
dotnet ai-dotnet/tools/Starlings.Qwen35.Fidelity/bin/Release/net10.0/Starlings.Qwen35.Fidelity.dll --bundle <snapshot> --variant fp32x \
  --device cpu --prompts $RP/prompts-release-v1.jsonl --reference $RP/reference --out <run>/run-parity --modes parity --threads 4
model-pipeline evaluate $MAN --fixtures $RP/reference --results <run>/run-parity --out <report> --require-gates G1,G2
DS=model-pipeline/protected/qwen3.5-0.8b/release/task-quality/gsm8k-test-v1/prompts.jsonl
dotnet .../Starlings.Qwen35.Fidelity.dll generation --bundle <snapshot> --prompts $DS --out <run>/g3 --variant fp32x --device cpu --threads 4
model-pipeline evaluate $MAN --g3 generation --dataset $DS --results <run>/g3 --out <report>
```

Candidate digests (outside Git): G1 `tokenizer.candidate.jsonl` `76903d5f28ac648fbad8691b5466366c3689c86da06f7cf017f01660e05e6501`, parity `results.jsonl` `b9362347b01a1e916e0c639e3c1fd97dda57aa896f6cbab59592addc8f3902d6`. The GSM8K `generations.jsonl` digest is in `g3-gsm8k/report.json`. Environment: i7-5820K, .NET 10.0.12, ONNX Runtime GenAI 0.17.1 CPU, 4 threads. The machine was shared with soak measurements, so these wall-clock times are not benchmarks: release-prompt parity 415 s, GSM8K 2,676 s, reference step-0 run 758 s.
