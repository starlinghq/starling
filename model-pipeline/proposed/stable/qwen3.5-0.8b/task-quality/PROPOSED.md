# Proposed: G3 generation task set and budget, Qwen3.5-0.8B (GSM8K)

Status: **proposed**. Not accepted and not under `model-pipeline/protected/`. The evaluator cannot gate on it yet: `model-pipeline evaluate --g3` accepts `retrieval` and `ner` only, and `task-quality.yaml` has no `generation` key. The budget below is written as a proposal for that future key, not as a file.

## Set: `gsm8k-test-v1/`

`prompts.jsonl`: 50 GSM8K test problems in the prompts format of `reference/generation/prompts-v1.jsonl` plus `answer`. Each is one user turn: the question, a newline, and "Please reason step by step, and put your final answer within \boxed{}." (the instruction Qwen's model cards use for math); `chat_template_kwargs: {enable_thinking: false}`; greedy; `max_new_tokens` 512.

| Source | Revision | File | License (evidence) |
| --- | --- | --- | --- |
| `openai/gsm8k` (`main`) | `740312add88f781978c0658806c59bc2815b9866` | `main/test-00000-of-00001.parquet` (SHA-256 in `reference/stable/sources.lock.json`) | MIT (card front matter; card text "licensed under the MIT License") |

Selection (`reference/stable/select_generation.py`, seed 20261014): the test rows in file order, one seeded shuffle, first 50. Answers are the number after `####` with commas removed. Metric (`reference/stable/score_generation.py`, fixed before any run): exact match of the last `\boxed{...}` in the output, else the last number, compared numerically.

GSM8K is a public benchmark and may be in the model's training data; that does not matter for a budget that compares the port with the same model's reference.

## Reference score

transformers 5.18.0 on PyTorch 2.14.1 CPU, fp32, greedy (`reference/generation/run_reference.py`; Qwen3.5's linear-attention layers on transformers' reference PyTorch kernels): **exact match 0.64 (32 of 50)**. 43 outputs stop at end of turn, 7 reach 512 tokens. 16 of the 50 outputs have at least one greedy step whose top-2 logit margin is below 0.01, so a backend with slightly different arithmetic can take another path on those rows. The run took 52 minutes on the shared workstation; the .NET ORT GenAI fp32x CPU run is expected to be of the same order (the development benchmark measured 4 to 5 decode tokens/s under load). Scores: `reference-scores/transformers-cpu-fp32.json` (task SHA-256 `c60ad0bc…4884`, results SHA-256 `a105d827…e6e5`; the results file stays outside Git).

Proposed budget (for a future `generation:` entry of `task-quality.yaml`):

| Profile | Minimum `exact_match` | Basis |
| --- | --- | --- |
| `ortgenai-fp32x-cpu`, `ortgenai-fp32x-cuda` | 0.60 | reference minus 2 problems (each problem is 0.02); greedy fp32 should follow the reference except at near-ties, and a flip at a near-tie can go either way |
| `ortgenai-fp16x-cuda` | 0.56 | reference minus 4 problems: an approximate variant (18/22 to 19/22 greedy agreement on the development prompts), budgeted on task quality as plan 10.1 asks |

## Commands

```sh
<generation venv>/bin/python model-pipeline/reference/generation/run_reference.py --model-dir <Qwen/Qwen3.5-0.8B@2fc06364 snapshot> \
  --revision 2fc06364715b967f1860aea9cf38778875588b17 --prompts gsm8k-test-v1/prompts.jsonl --out <ref run> --device cpu --threads 6
python model-pipeline/reference/stable/score_generation.py --task gsm8k-test-v1/prompts.jsonl --results <ref run>/results.jsonl --out ref.json
# .NET candidate (the same prompts file; the tool writes output_ids):
dotnet run --project ai-dotnet/tools/Starlings.Qwen35.Fidelity -c Release -- --bundle <snapshot> --variant fp32x --device cpu \
  --prompts gsm8k-test-v1/prompts.jsonl --reference <ref run> --out <run> --modes parity
python model-pipeline/reference/stable/score_generation.py --task gsm8k-test-v1/prompts.jsonl --results <run>/results.jsonl \
  --tokenizer <snapshot>/transformers-5.18.0/tokenizer.json --baseline <ref run>/results.jsonl --out net.json
```

`--baseline` lists the rows whose correctness flips between the reference and the candidate, so a budget miss can be traced to specific near-tie divergences (the per-step margins are in the reference `results.jsonl`).

## Decisions for the reviewer

1. Accept GSM8K (MIT) as the Qwen G3 set and the exact-match metric.
2. The budget values (above) and the evaluator workstream for a `generation` G3 task.
3. Whether fp16x CUDA gets the same budget or a wider one as an approximate variant (plan 10.1).
