# Qwen3.5 batched generation: CPU spike evidence (2026-10-17)

Supports `docs/plan/qwen35-batched-generation.md`. This is a research spike, not a qualification run. No product code changed, and `batch` stays Unsupported.

## Environment

- ONNX Runtime GenAI 0.17.1 (`Microsoft.ML.OnnxRuntimeGenAI`, CPU natives, which bring ONNX Runtime 1.30.0), .NET 10, Linux x64. The managed API was called directly, without Starling's `TextGenerator`.
- Bundle `ortgenai-fp32x` of Qwen/Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17`, from the local cache. It is the same bundle as the Stable CPU profile.
- Overlays: `pad_token_id` 248319 (`Qwen35ChatModel.PadTokenIdOverlay`) and `intra_op_num_threads` 8. CPU provider only (`CUDA_VISIBLE_DEVICES` empty).
- Prompt ids and reference output ids come from `model-pipeline/protected/qwen3.5-0.8b/reference/generation.jsonl` (read only). The run was greedy, with 32 new tokens.
- Host: 12 CPUs shared with other jobs (load average 23 to 28 during the timing run). Treat the timings as indicative only.

## Files

- `batch-experiment/`: the program (`Program.cs`, `qwen-batch.csproj`). It is a standalone project and not part of `AI.slnx`. To run it: `dotnet build -c Release`, then `dotnet bin/Release/net10.0/linux-x64/qwen-batch.dll <ortgenai-fp32x dir> <generation.jsonl> 32 8 [timing]`.
- `agreement-cpu.txt`: batch of 2 compared with single runs (token agreement, agreement with the reference, step-0 logits).
- `timing-cpu.txt`: wall time for one pair run three ways: sequential singles, two concurrent singles, and one batch of 2.
- `inspect_graph.py`, `graph-fp32x.txt`: the operators and stateful-op inputs of the fp32x graph. The script loads only the graph, without weights. The fp16x graph has the same wiring.

## Results

| Pair (prompt tokens) | Padding | Shorter row vs single | Longer row vs single |
| --- | --- | --- | --- |
| en-list (22) + en-list (22) | none | 32/32 identical, step-0 max abs 0 | 32/32 identical, 0 |
| en-short (24) + special-endoftext (24) | none | 8/8 identical (EOS), 0 | 32/32 identical, 0 |
| en-list (22) + en-math (40) | left | diverges at step 0, step-0 max abs 17.3 | 32/32 identical, 0 |
| en-list (22) + en-math (40) | right | token 1 equal, then EOS at step 1, step-0 max abs 21.4 | 32/32 identical, 0 |
| en-short (24) + code-cs (80) | left | diverges at step 0: it re-emits the prompt's `<think>\n\n</think>\n\n` block before the answer, step-0 max abs 16.8 | 32/32 identical, 0 |
| en-short (24) + code-cs (80) | right | token 1 equal, then EOS at step 1, step-0 max abs 15.6 | 32/32 identical, 0 |

All single runs matched the reference over the compared prefix.

Timing (minimum of 2 repetitions, 32 new tokens):

| Pair | Sequential singles | 2 concurrent singles | Batch of 2 |
| --- | --- | --- | --- |
| en-list + en-list (no padding) | 13.68 s | 9.59 s | 8.92 s |
| en-list + en-math (left padding; batched output wrong) | 23.98 s | 15.87 s | 13.97 s |
