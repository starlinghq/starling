# Per-release GPU regression, 2026-10-16: run 38857860186

The first run of the qualification workflow's `cuda (runpod)` job (#73): [run 38857860186](https://github.com/starlinghq/starling/actions/runs/38857860186), manual dispatch, artifact `qualification-cuda`. The summary, every gate report and every soak report are copied here under [`run-38857860186/`](run-38857860186/).

| Item | Value |
| --- | --- |
| Commit | `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8` (#73). Between it and the promotion change, #75 and #76 changed capability tables, READMEs, manifests and documentation only; no CUDA runtime, session or model code. |
| Pod | RunPod secure cloud EU-CZ-1, one on-demand pod, **NVIDIA GeForce RTX 4090** (Ada, sm_89), driver 580.159.03, CUDA 13 runtime and cuDNN 9 from the volume's pinned wheels. The job then accepted an RTX 3090 or 4090; it landed on a 4090. |
| Settings | `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES=0`, `NVIDIA_TF32_OVERRIDE=0`, TF32 off in every session; transparent huge pages `madvise`; bounds and budgets only from `model-pipeline/protected/`; the G2 references are the hosted, digest-verified ones (#71). |
| Soak | 10-minute smoke test per profile: wrong outputs, errors and crashes fail it; the growth verdict is recorded but not binding (growth is judged by the binding one-hour G4 soaks, [`../qualification-2026-10-16-cuda/`](../qualification-2026-10-16-cuda/README.md)). `MALLOC_MMAP_THRESHOLD_=65536` was set for GLiNER2 and EmbeddingGemma 2 video, where the package README requires it. |
| Wall time | 105 minutes on the pod. The job's `Result` step failed because one check failed (below). |

## Results per profile

| Profile | CUDA tests | G1 | G2 | G3 | Soak smoke test | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| `gliner2-base-v1/onnx-fp32-cuda` | 8/8 passed | passed: 232 processor rows exact | passed: span logits 1.4e-5 relative (bound 1e-4), scores 3.4e-6 (2e-5), entities 30/30 from inputs and from text | passed: WNUT-17 exact micro F1 0.3626 (budget 0.352) | passed: 151 cycles, 0 wrong, 0 errors; cancellation p50 12 ms, max 84 ms | **all passed** |
| `multilingual-e5-small/onnx-fp32-cuda` | 9 passed, 1 skipped (the missing-provider test, opt-in) | passed: 232 tokenizer rows exact | passed: embeddings 1.21e-7 (bound 1e-5); C# against Python ORT CUDA 8e-8 (1e-6) | passed: MIRACL nDCG@10 0.7978 (0.792), MRR@10 0.7717 (0.765) | passed: 264 cycles; cancellation p50 5 ms, max 19 ms | **all passed** |
| `embeddinggemma-2/onnx-fp32-cuda` | text tests passed (see below) | passed: 232 tokenizer rows exact | passed: 696 rows, embeddings 8.6e-7 (bound 5e-6); C# against Python ORT CUDA 2.5e-7 (1e-6) | passed: MIRACL nDCG@10 0.8325 (0.826), MRR@10 0.8136 (0.808) | passed: 138 cycles; cancellation p50 59 ms, max 212 ms | **all passed** |
| `embeddinggemma-2/image-fp32-cuda` | image tests passed (see below) | passed (with G2) | passed: 19 rows, embeddings 2.16e-6 (bound 1e-5), token embeddings 6.8e-5 row-relative (1e-3) | passed: `image-retrieval-v1` nDCG@10 0.9222 (0.916), MRR@10 0.9008 (0.893) | passed: 136 cycles; cancellation p50 33 ms, max 108 ms | **all passed** |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | 55/55 passed (`Fp32xCudaTests`, `Fp32xCudaChatClientTests`) | passed: 232 tokenizer rows, 32 renderings and prompt ids exact | passed: greedy ids 32/32 exact; step 0 full vocabulary 9.3e-4 (5e-3); cached vs uncached 4.6e-5 (2e-4) | passed: GSM8K 0.64 (0.60) | passed: 115 cycles; cancellation p50 104 ms, max 326 ms | **all passed** |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | 51/51 passed (`Fp16xCudaTests`, `Fp16xCudaChatClientTests`) | passed: as fp32x | passed: greedy ids 28/32 exact, 4 at allowed near ties (margin below 0.05); step 0 full vocabulary 0.046 (0.15); cached vs uncached 0.045 (0.1) | passed: GSM8K 0.60 (0.56) | passed: 152 cycles; cancellation p50 47 ms, max 130 ms | **all passed** |
| `embeddinggemma-2/video-fp32-cuda` | 6 video tests skipped (no video set on the pod) | passed | **failed**: embedding max abs **1.009e-5** against 1e-5; token embeddings **3.34e-3** against 1e-3 row-relative; worst row `commons-video-04` | passed: `video-retrieval-v1` nDCG@10 0.8126 (0.797), MRR@10 0.7714 (0.751) | passed: 92 cycles | **G2 failed** |

Also passed: models acquired and verified against their locks. The EmbeddingGemma 2 CUDA test project ran 25 tests: 18 passed (text parity, task prompts, cancellation, the queued-work probe, concurrency, over-8,192 rejection, placement, the text generator and ML.NET transform over CUDA, image parity, image batches, image retrieval, image budgets) and 7 were skipped because the pod has no video fixture set: the six `CudaVideoParityTests` and `CudaAdapterTests.MultimodalGeneratorOverCudaMeetsTheProtectedImageAndVideoTolerances`. The multimodal generator over CUDA was therefore not exercised for images either; it stays experimental on `image-fp32-cuda`, and its evidence remains G7 of 2026-10-14.

The soak cancellation figures are from the smoke test's deadline requests (5 to 150 ms deadlines on its own request sizes); they are not the worst case. The contracts state the worst case per chunk from [`../gliner2-cuda-wrong-output/`](../gliner2-cuda-wrong-output/README.md): about 50 ms for GLiNER2 (8 long texts), about 125 ms for e5 (32 x 512 tokens), up to about 1.2 s for EmbeddingGemma 2 text (16 rows of 2,048 tokens).

## The video G2 failure: GPU, not code

On the RTX 3090, the same release set passed G2 with embedding max abs 2.77e-6 and token embeddings 7.97e-4 ([release-set gates](../release-set-gates-2026-10-14/embeddinggemma2-video-fp32-cuda/README.md)), and the binding G4 and G5 round ran on RTX 3090s. On the RTX 4090 (Ada) the embedding difference is about 3.6 times larger and just over the protected bound; the token embeddings exceed theirs by a factor of 3.3. G1 (decoding, sampling, preprocessing, ids) and G3 passed on the 4090. The bound was not changed and no rerun was made to chase a pass.

Decisions taken from it:

1. **The CUDA contracts cover the NVIDIA GeForce RTX 3090 (Ampere) only.** Every CUDA gate ran there. The RTX 4090 and every other GPU are untested and outside the contracts. Each CUDA manifest records this in its compatibility comment and known issues (the schema has no GPU field).
2. **The `cuda (runpod)` job is pinned to the RTX 3090.** `tools/ci/runpod.py` requests only that GPU type, fails when none is in stock rather than falling back, and refuses a pod that reports another GPU. The workflow's start step and `tools/ci/runpod-cuda-pod.sh` check `nvidia-smi` again before any test runs.
3. **`embeddinggemma-2/video-fp32-cuda` stays Experimental** until a per-release run on an RTX 3090 is green for it.
4. **The six profiles whose checks all passed are promoted to Stable** on the RTX 3090 (`docs/kickoff.md`, "Stable CUDA profiles, first round"). They passed on the 4090 as well, but that is not part of their contracts.

## Open

- A confirming per-release run on an RTX 3090 for `embeddinggemma-2/video-fp32-cuda` (and the six promoted profiles). It was not dispatched: the RunPod balance was about $5.70, and the job refuses to start below $5, which leaves no margin for a 105-minute run. It needs a credit top-up.
- The video fixture set on the GPU pod's volume, so that the video CUDA tests and the multimodal generator test run per release instead of being skipped.
