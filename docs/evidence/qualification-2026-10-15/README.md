# Qualification round, 2026-10-14 to 2026-10-15: G4 soaks and G5 matched benchmarks

Commit `683457f`. Linux x64, Intel i7-5820K (12 threads, 94 GB), NVIDIA RTX 3090 (GPU 0, `CUDA_DEVICE_ORDER=PCI_BUS_ID`, TF32 off). GPU 1 ran an unrelated workload and was never used. One item at a time, no other Starling job running. The run log is [`run-log.txt`](run-log.txt).

Rules are the committed ones and were not changed for this round: the soak growth rule in `Starlings.Soak.Tool`, and the G5 budgets in `model-pipeline/protected/performance-budgets.yaml`, checked by `model-pipeline bench g5` (3 rounds, 6 threads, 10 cold processes per round; both sides of each comparison on this machine).

## G4: one-hour soaks

Yesterday's CPU soaks (`docs/evidence/g4-soak-1h-2026-10-14/`) covered e5, EmbeddingGemma 2 text and Qwen fp32x on CPU (all PASS).

| Profile | Verdict | Report |
| --- | --- | --- |
| gliner2-base-v1 `onnx-fp32-cpu` | **PASS** with `MALLOC_MMAP_THRESHOLD_=65536` (the documented Linux deployment setting; `docs/evidence/gliner2-soak-growth/`) | [soak/](soak/) `gliner2-*` |
| embeddinggemma-2 `image-fp32-cpu` | **PASS** (unloaded working set kept between loads is reused by the next load: allocator retention, reported) | `embeddinggemma2-image-*` |
| embeddinggemma-2 `video-fp32-cpu` | **PASS** | `embeddinggemma2-video-*` |
| embeddinggemma-2 `onnx-fp32-cuda` | **PASS** | `embeddinggemma2-*` (cuda) |
| qwen3.5-0.8b `ortgenai-fp32x-cuda` | **PASS** | `qwen35-*` (cuda) |
| qwen3.5-0.8b `ortgenai-fp16x-cuda` | **PASS** | `qwen35-fp16x-*` |
| multilingual-e5-small `onnx-fp32-cuda` | **FAIL**: loaded and unloaded memory grow about 190 to 209 MB/h | `e5-*` (cuda) |
| gliner2-base-v1 `onnx-fp32-cuda`, embeddinggemma-2 `image-fp32-cuda`, `video-fp32-cuda` | **not run**: the run environment lacked the cuDNN libraries (`libcudnn.so` not found), so each exited at its first cuDNN kernel. A test-setup error, rerun with the pinned wheels' CUDA libraries on `LD_LIBRARY_PATH` | run log |

All CUDA soaks of this round ran without cuDNN on the library path. The EmbeddingGemma 2 text and both Qwen CUDA runs completed because none of their kernels needs cuDNN; their sessions placed every node on the CUDA execution provider as in the fidelity runs, so those passes stand. The e5 CUDA run also completed, but in an environment the ONNX Runtime CUDA execution provider does not support (no cuDNN); it is therefore not a qualifying run. Rule fixed before the rerun: the rerun with the pinned wheels' CUDA and cuDNN libraries is the qualifying e5 CUDA soak, and its result stands whichever way it goes. If it fails, the growth is investigated as GLiNER2's was, not run a third time. The rerun records the GPU it used.

## G5: matched benchmarks (C# against the Python same-backend run)

| Profile | Verdict | Key numbers |
| --- | --- | --- |
| e5 `onnx-fp32-cpu` | **PASS**, qualifying | warm ratios within budget on every workload |
| e5 `onnx-fp32-cuda` | **PASS**, qualifying | warm 0.90 to 0.93 |
| EmbeddingGemma 2 `onnx-fp32-cpu` | **PASS**, qualifying | warm 1.02 to 1.08 (budget 1.10) |
| EmbeddingGemma 2 `onnx-fp32-cuda` | **PASS**, qualifying | warm 0.96 to 1.00 |
| EmbeddingGemma 2 `image-fp32-cpu` | **PASS**, qualifying | warm about 1.00 |
| EmbeddingGemma 2 `video-fp32-cpu` | **PASS**, qualifying | warm 1.01 to 1.02 |
| GLiNER2 `onnx-fp32-cpu` | **PASS**, qualifying | inference 0.94 to 0.96 |
| Qwen `ortgenai-fp32x-cuda` | **PASS**, qualifying | decode and time to first token within budget |
| Qwen `ortgenai-fp16x-cuda` | **PASS**, qualifying | decode and time to first token within budget |
| Qwen `ortgenai-fp32x-cpu` | **FAIL** | time to first token 1.119 [0.848, 1.447] on `short-decode64` against 1.10 |
| EmbeddingGemma 2 `image-fp32-cuda` | **FAIL** | preprocessing 8 to 48% of embed against 5%; photo warm ratio 1.118 against 1.10 |
| EmbeddingGemma 2 `video-fp32-cuda` | **FAIL** | preprocessing 11 to 36% of embed against 5%; 1080p warm ratio 1.229 |
| GLiNER2 `onnx-fp32-cuda` | **FAIL** | prepare + decode 8 to 13% of inference against 5% |

The failures are real: on a fast GPU the C# CPU-side work (image and video preprocessing, GLiNER2 preparation and decoding) becomes a large share of the request, and Qwen's CPU path spends more than Python before its first token. They are being fixed in code; budgets are unchanged, and a fixed profile is benchmarked again under this protocol.

Raw results and full verdicts: [`g5/`](g5/) (`<profile>.jsonl`, `<profile>.jsonl.verdict.txt`).

## G4 CUDA soaks on RunPod (binding reruns), 2026-10-15

The four CUDA soaks that could not qualify locally were run on RunPod pods (commit `26249631`; CUDA 13 and cuDNN 9 from the pinned wheels on `LD_LIBRARY_PATH`; TF32 off; one model per pod; one hour each). Reports are in [`runpod-soak/`](runpod-soak/). Under the rule fixed above, these results are binding.

| Profile | GPU, driver | Verdict |
| --- | --- | --- |
| gliner2-base-v1 `onnx-fp32-cuda` (`MALLOC_MMAP_THRESHOLD_=65536`) | RTX 3090, 580.126.20 | **FAIL**: 1 wrong output in 80,100 requests (cycle 388, request 14: an empty entity list instead of the reference). Memory plateaued. A correctness defect under investigation. |
| embeddinggemma-2 `image-fp32-cuda` | RTX 3090, 580.159.03 | **FAIL**: native memory growth, unloaded +85 MB/h, loaded +77 MB/h; 0 wrong outputs |
| embeddinggemma-2 `video-fp32-cuda` | RTX 4090, 580.159.03 | **FAIL**: native memory growth, unloaded +105 MB/h, loaded +103 MB/h; 0 wrong outputs |
| multilingual-e5-small `onnx-fp32-cuda` | RTX 3090, 580.126.20 | **FAIL**: unloaded +120 MB/h (82 MB in the first half, 12 MB in the second, limit 32 MB), loaded +110 MB/h; 1,492 cycles, 268,560 requests, 0 wrong outputs. The missing cuDNN did not cause the growth. |

The managed heap stayed flat in every run, so the growth is native. Its cause is being investigated per model, as the CPU GLiNER2 growth was, and is not settled by more runs.
