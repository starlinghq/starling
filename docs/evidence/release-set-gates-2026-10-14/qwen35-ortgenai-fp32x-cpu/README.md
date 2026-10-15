# qwen3.5-0.8b `ortgenai-fp32x-cpu` on the P-030 release sets

Run on 2026-10-14 at `10c016c` (main after #26). The release prompts and the reference fixture are at `qwen3.5-0.8b/release/release-prompts/` (32 oasst1 root prompts, 64 new tokens each). The tokenizer release set is `qwen3.5-0.8b/release/tokenizer-release/` (232 holdout rows), and GSM8K is `qwen3.5-0.8b/release/task-quality/gsm8k-test-v1/` (50 problems). The evaluator verified the protected evidence against Git. Nothing under `protected/` was changed.

| Gate | Set | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- | --- |
| G1 | tokenizer release (232 rows) | **passed** | ids exact 232/232 (`discrete.prompt_ids`). Tokens, masks, offsets in three units and decode, with and without special tokens, are also exact (measured). | [`g1-tokenizer-release/report.md`](g1-tokenizer-release/report.md) |
| G1 | release prompts (32) | **passed** | owned template rendering exact 32/32; owned tokenizer ids exact 32/32 | [`g1-g2-release-prompts/report.md`](g1-g2-release-prompts/report.md) |
| G2 | release prompts (32) | **not-run** | Greedy ids exact on 32/32, with no divergence despite reference steps with margins down to 5.7e-4. Finish reason 32/32. Step-0 argmax 32/32. Step-0 top-5 log-probability 4.54e-4 (2e-3). Step-0 log-probability on the reference top 64: 5.87e-4 (bound 5e-3; full vocabulary not run, below). Streaming reassembly and usage exact. Provider options `{}`. **Not run:** the full-vocabulary step-0 bound, and the three runtime-semantics keys (cached vs uncached, chunked prefill, concurrency). | same |
| G2 | release prompts (32) | **passed** (rerun below; the first run's not-run is kept in the section that follows) | full-vocabulary step 0 9.27e-4 (5e-3); cached vs uncached 2.67e-5 (2e-4); chunked vs one-call prefill 0 (1e-4); concurrency equal; greedy 32/32 exact | [`g2-release-prompts-fixture-semantics/report.md`](g2-release-prompts-fixture-semantics/report.md) |
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

## G2 on the release prompts, rerun (2026-10-14)

**G2 passed** on the release prompts, exit 0, at `e360188` (branch `qwen-release-semantics`). The two reasons for the first run's not-run are resolved:

1. **Full step-0 reference vectors.** `run_reference.py` regenerated them in the environment the fixture records (`release-prompts/reference/environment.json`): torch 2.14.1+cpu, transformers 5.18.0, tokenizers 0.23.2, numpy 2.4.6, safetensors 0.8.0, no flash-linear-attention, Python 3.11.16, CPU, `--threads 6`. All 32 `step0/<id>.f32` are byte-identical to the SHA-256 the fixture records per case (`step0_sha256`, and `step0_full_vectors.sha256`), and the evaluator checked each file against them (`--reference-tensors`). The vectors stay outside Git (`~/.cache/starling-release-refs/qwen-release-t6`). The first run's mismatch came from the torch +cu130 wheel at 4 threads.
2. **Runtime-semantics keys.** `Starlings.Qwen35.Fidelity --modes fixture-semantics` runs the three checks that feed G2 (cached vs uncached, chunked vs one-call prefill, concurrency) on any generation fixture, with the cases chosen from the fixture by a fixed rule that `semantics.json` records under `selection`: cached vs uncached on the case with the largest smallest reference top-2 margin over its first 12 steps (`oasst1-68a06244`); concurrency on the six cases with the largest smallest margin over 24 steps (four in parallel on two threads, two interleaved); chunked prefill on the longest prompt (`oasst1-330fc1cf`, 92 tokens: chunks 64 and 32) and the second longest (`oasst1-3c5b52fd`, 83 tokens: chunk 7). These checks compare with the reference's greedy prefix, so the rule avoids near ties; the greedy-ids check keeps every case. The development-only checks of `--modes semantics` (max-token boundary, stop sequences, cancellation, sampling, text path) bind no tolerance key and are not part of this run. The evaluator now refuses a `semantics.json` that names cases outside the evaluated fixture, so a development-prompt run cannot stand in for the release prompts.

| Check (binding key) | Measured (bound) | Status |
| --- | --- | --- |
| Template rendering, prompt ids (G1) | 32/32 exact | passed |
| Greedy ids (`exact_except_listed_margin_below_1e-3`) | 32/32 exact | passed |
| Finish reason, step-0 argmax | 32/32 | passed |
| Step-0 top-5 log-probability | 4.54e-4 (2e-3) | passed |
| Step-0 log-probability, full vocabulary | 9.27e-4 (5e-3); top 64 5.87e-4 | passed |
| Cached vs uncached (12 steps, `oasst1-68a06244`) | 2.67e-5 (2e-4), prefix equal | passed |
| Chunked vs one-call prefill (`oasst1-330fc1cf` chunks 64 and 32, `oasst1-3c5b52fd` chunk 7) | 0 (1e-4), tokens equal | passed |
| Concurrency (4 parallel on 2 threads, 2 interleaved) | equal to the reference prefix | passed |
| Streaming reassembly, usage, provider options `{}` | exact | passed |

Bounds from `protected/qwen3.5-0.8b/ortgenai-fp32x-cpu/tolerances.yaml`; nothing under `protected/` changed. Report: [`g2-release-prompts-fixture-semantics/report.md`](g2-release-prompts-fixture-semantics/report.md) (its date is UTC, 2026-10-15).

```sh
RP=model-pipeline/protected/qwen3.5-0.8b/release/release-prompts
<venv: torch==2.14.1+cpu transformers==5.18.0 tokenizers==0.23.2 numpy==2.4.6 safetensors==0.8.0>/bin/python model-pipeline/reference/generation/run_reference.py \
  --model-dir <Qwen/Qwen3.5-0.8B@2fc06364 snapshot> --revision 2fc06364715b967f1860aea9cf38778875588b17 --prompts $RP/prompts-release-v1.jsonl --out <ref> --device cpu --threads 6
dotnet .../Starlings.Qwen35.Fidelity.dll --bundle <snapshot> --variant fp32x --device cpu --prompts $RP/prompts-release-v1.jsonl \
  --reference $RP/reference --out <run> --modes parity,fixture-semantics --threads 4
model-pipeline evaluate model-pipeline/registry/qwen3.5-0.8b/ortgenai-fp32x-cpu/manifest.yaml --fixtures $RP/reference --results <run> \
  --reference-tensors <ref> --out <report> --require-gates G1,G2
```

Wall clock (shared machine, not a benchmark): reference regeneration 307 s; runner 265 s (parity and fixture-semantics, 4 threads).

