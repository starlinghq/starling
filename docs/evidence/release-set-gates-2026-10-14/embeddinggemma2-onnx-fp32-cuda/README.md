# embeddinggemma-2 `onnx-fp32-cuda` on the P-030 release sets

Run on 2026-10-14 at `472e187` on the RTX 3090, GPU 0 only. The sets are the same as for the CPU profile ([`../embeddinggemma2-onnx-fp32-cpu/`](../embeddinggemma2-onnx-fp32-cpu/README.md)). Every stage ran on `CUDAExecutionProvider` with `use_tf32: 0`.

| Gate | Result | Key numbers (bound) | Report |
| --- | --- | --- | --- |
| G1 tokenizer holdout (232 rows) | **passed** | ids and masks exact 232/232 (device-independent; evaluated against this manifest) | [`g1-tokenizer-holdout/report.md`](g1-tokenizer-holdout/report.md) |
| G2 text-encoder release set (696 cases) | **passed** (rerun, below) | stage 1 ids exact 696/696; .NET embedding from ids and from text max abs **9.8e-7** (5e-6), min cosine 0.999999999995 (0.9999999); Python ORT CUDA embedding 9.8e-7 (5e-6); C# vs Python ORT CUDA 2.6e-7 (1e-6); token embeddings per case 7.97e-6 relative (1e-4); padded batches within 7.4e-7 of single (5e-6). The first run was not-run (measured only, [`g2-measured-not-gate/report.md`](g2-measured-not-gate/report.md)) | [`g2-release/report.md`](g2-release/report.md) |
| G3 MIRACL | **passed** | nDCG@10 **0.8325** (minimum 0.826), MRR@10 **0.8136** (minimum 0.808), equal to the CPU run; producer on CUDA device 0 | [`g3-retrieval/report.md`](g3-retrieval/report.md) |

Wall clock: .NET stages 39.3 s, Python stage 2 (CUDA, `requirements-cuda.txt`) 155.6 s, G3 producer 52.3 s.

## G2 rerun

The reference and the method are the same as on CPU ([`../embeddinggemma2-onnx-fp32-cpu/README.md`](../embeddinggemma2-onnx-fp32-cpu/README.md), "G2 rerun"). The reference is the outside-Git `embeddings.jsonl` regenerated at 6 threads, SHA-256 `c2ffe5e45be11904b7ad0277758c0d8bafc48c6c65dcc87ff932ccbaf574285c`, and `--reference-embeddings` bound it to the protected `reference.json`.

- **Producers.** They ran at `303d7ca` on the RTX 3090, GPU 0 only (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`, `use_tf32: 0`, `NVIDIA_TF32_OVERRIDE=0`). Runs: `Starlings.Fidelity.Tool.Cuda embeddinggemma2 --device cuda:0 --threads 6` (34 s), then `stage_embeddinggemma2_export.py --provider cuda` (103 s).
- **Evaluation.** `model-pipeline evaluate .../onnx-fp32-cuda/manifest.yaml --fixtures model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release --reference-embeddings <file> --require-gates G1,G2`, with protected evidence `verified`. **Exit 0, first divergence none, G1 passed, G2 passed.**
- **Candidate stage results (outside Git):** `tokenizer` `0dbe7c8e…`, `export` `c5dbdece…`, `dotnet` `ac29f768…`.
