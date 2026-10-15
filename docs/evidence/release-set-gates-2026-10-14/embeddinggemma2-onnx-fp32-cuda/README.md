# embeddinggemma-2 `onnx-fp32-cuda` on the P-030 release sets

Run on 2026-10-14 at `472e187` on the RTX 3090, GPU 0 only. The sets are the same as for the CPU profile ([`../embeddinggemma2-onnx-fp32-cpu/`](../embeddinggemma2-onnx-fp32-cpu/README.md)). Every stage ran on `CUDAExecutionProvider` with `use_tf32: 0`.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 tokenizer holdout (232 rows) | **passed** | ids and masks exact 232/232 (device-independent; evaluated against this manifest) | [`g1-tokenizer-holdout/report.md`](g1-tokenizer-holdout/report.md) |
| G2 text-encoder release set | **not-run**: no committed `embeddings.jsonl` (see the CPU README) | Measured, not gate evidence, against the regenerated reference whose SHA-256 equals the recorded `c2ffe5e4…`, on a scratch copy with `--protected-base none`: stage 1 ids exact 696/696; .NET embedding from ids and from text 9.8e-7 (5e-6), min cosine 0.999999999995 (0.9999999); C# vs Python ORT CUDA 2.6e-7 (1e-6); token embeddings per case 7.97e-6 relative (1e-4); no failing check | [`g2-measured-not-gate/report.md`](g2-measured-not-gate/report.md) |
| G3 MIRACL | **passed** | nDCG@10 **0.8325** (minimum 0.826), MRR@10 **0.8136** (minimum 0.808), equal to the CPU run; producer on CUDA device 0 | [`g3-retrieval/report.md`](g3-retrieval/report.md) |

Wall clock: .NET stages 39.3 s, Python stage 2 (CUDA, `requirements-cuda.txt`) 155.6 s, G3 producer 52.3 s.
