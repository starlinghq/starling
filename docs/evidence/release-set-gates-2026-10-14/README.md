# Gates G1, G2 and G3 on the P-030 release sets (2026-10-14)

These runs are the first .NET evaluation on the release sets, G3 datasets and quality budgets the maintainer accepted on 2026-10-14 (`docs/kickoff.md`, "Release sets, G3 datasets and budgets"; layout in `model-pipeline/protected/README.md`). Each profile directory holds a README with commands, digests, metrics against budgets, maximum errors against bounds and wall clock, next to the unedited `report.md` and `report.json` of `model-pipeline evaluate`. Candidate outputs (stage results, embeddings, predictions, generations) stay outside Git, and the READMEs give their SHA-256.

The release sets are addressed explicitly. Staged G1/G2 use `model-pipeline evaluate --fixtures <protected release dir>` (#24). G1 on a tokenizer holdout or release file uses `--tokenizer-fixtures <protected .jsonl>` (#24). G3 uses `--g3 <task> --dataset <protected dataset>`, and #26 makes the accepted retrieval and NER budgets bind. The C# producers take the same paths: #27 adds `--max-image-pixels` and #29 adds `--device` for the G3 producers. The evaluator reads only `model-pipeline/protected/` and verifies every protected file it reads against Git. Bounds and budgets come only from each profile's protected `tolerances.yaml` and `task-quality.yaml`. Nothing under `model-pipeline/protected/` was changed, and nothing was tuned.

A manifest gate entry becomes `passed` only where the evaluator reported that gate passed on the release set, and the entry gets this evidence path and the date 2026-10-14. Other gates keep their entry, with a note. G0, G4 to G7 and maturity are unchanged.

| Profile | G1 | G2 | G3 (budget) | README |
| --- | --- | --- | --- | --- |
| e5 `onnx-fp32-cpu` | passed | passed (1.46e-7 / 1e-5) | passed: nDCG@10 0.7978 ≥ 0.792, MRR@10 0.7717 ≥ 0.765 | [e5-onnx-fp32-cpu](e5-onnx-fp32-cpu/README.md) |
| e5 `onnx-fp32-cuda` | passed | passed (1.2e-7 / 1e-5) | passed: 0.7978 / 0.7717 | [e5-onnx-fp32-cuda](e5-onnx-fp32-cuda/README.md) |
| EmbeddingGemma 2 `onnx-fp32-cpu` | passed | **not-run**: no committed `embeddings.jsonl` (measured 5.1e-7 / 5e-6 against the regenerated reference with the recorded digest) | passed: 0.8325 ≥ 0.826, 0.8136 ≥ 0.808 | [embeddinggemma2-onnx-fp32-cpu](embeddinggemma2-onnx-fp32-cpu/README.md) |
| EmbeddingGemma 2 `onnx-fp32-cuda` | passed | **not-run** (same; measured 9.8e-7) | passed: 0.8325 / 0.8136 | [embeddinggemma2-onnx-fp32-cuda](embeddinggemma2-onnx-fp32-cuda/README.md) |
| EmbeddingGemma 2 `image-fp32-cpu` | passed | passed (8.55e-7 / 1e-5; soft tokens 8.7e-5 / 1e-3), with `--max-image-pixels` for one 79 MP image | none | [embeddinggemma2-image-fp32-cpu](embeddinggemma2-image-fp32-cpu/README.md) |
| EmbeddingGemma 2 `image-fp32-cuda` | passed | passed (1.53e-6 / 1e-5; soft tokens 1.06e-4 / 1e-3) | none | [embeddinggemma2-image-fp32-cuda](embeddinggemma2-image-fp32-cuda/README.md) |
| EmbeddingGemma 2 `video-fp32-cpu` | **failed** (Ogg Theora frames of `commons-video-02`) | **failed** (8.9e-3 / 1e-5 on those rows) | none | [embeddinggemma2-video-fp32-cpu](embeddinggemma2-video-fp32-cpu/README.md) |
| EmbeddingGemma 2 `video-fp32-cuda` | **failed** (same) | **failed** (same) | none | [embeddinggemma2-video-fp32-cuda](embeddinggemma2-video-fp32-cuda/README.md) |
| GLiNER2 `onnx-fp32-cpu` | passed | passed (span logits 1.78e-5 / 1e-4) | passed: exact F1 0.3626 ≥ 0.352, partial 0.4134 ≥ 0.403 | [gliner2-onnx-fp32-cpu](gliner2-onnx-fp32-cpu/README.md) |
| GLiNER2 `onnx-fp32-cuda` | passed | passed (1.24e-5 / 1e-4) | passed: 0.3626 / 0.4134 | [gliner2-onnx-fp32-cuda](gliner2-onnx-fp32-cuda/README.md) |
| Qwen3.5 `ortgenai-fp32x-cpu` | passed | **not-run**: semantics keys exist only for the development prompts (greedy 32/32 exact) | passed: exact match 0.64 ≥ 0.60 | [qwen35-ortgenai-fp32x-cpu](qwen35-ortgenai-fp32x-cpu/README.md) |
| Qwen3.5 `ortgenai-fp32x-cuda` | passed | **not-run** (same); development-fixture rerun G2 passed | passed: 0.64 ≥ 0.60 | [qwen35-ortgenai-fp32x-cuda](qwen35-ortgenai-fp32x-cuda/README.md) |
| Qwen3.5 `ortgenai-fp16x-cuda` | passed | **not-run** (same; 2 allowed near-tie divergences) | passed: 0.64 ≥ 0.56 | [qwen35-ortgenai-fp16x-cuda](qwen35-ortgenai-fp16x-cuda/README.md) |

## Findings for the maintainer

1. **Video, Ogg Theora:** the C# FFmpeg adapter's frames for `commons-video-02` differ from torchcodec's seek-based decode. The set's `SOURCE.md` predicted this. G1 and G2 fail on CPU and CUDA, while every row without that file is within the bounds.
2. **EmbeddingGemma 2 text:** the accepted release set lacks `embeddings.jsonl` (`SOURCE.md` decision 2, still open). At 6 threads the reference regenerates byte-identical to the recorded digest.
3. **Qwen G2:** the runtime-semantics tolerance keys can be produced only on the development prompts. The full step-0 reference vectors did not reproduce bit for bit in the available environment (torch +cu130 wheel, 4 threads; the reference used +cpu with 6 threads).
4. **Image pixel limit:** one accepted image (79 MP) exceeds the package's 50 MP default. The coordinator's decision is to keep the default and evaluate that row with the option, recorded as an `experimental` capability.
5. **Reproducing outside-Git reference artifacts depends on the thread count.** The image tensors and the text embeddings matched only at the reference's 6 threads.
6. **G3 digest form:** fixed in #26.

Workstation: Intel i7-5820K (12 threads, AVX2), 94 GB RAM, Linux x64, .NET 10.0.12, ONNX Runtime 1.30.0 (GenAI 0.17.1). CUDA runs used the RTX 3090 (GPU 0 only, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0`, `use_tf32: 0`) with CUDA 13 and cuDNN 9 from the `encoder/requirements-cuda.txt` wheels. CPU runs used 4 threads while the one-hour soak measurements ran; the two 6-thread reference regenerations ran after the soaks finished. Wall-clock times are not benchmarks.
