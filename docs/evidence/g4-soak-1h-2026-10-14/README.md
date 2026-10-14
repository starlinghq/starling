# G4 one-hour soaks, CPU, 2026-10-14

Plan 10.2 qualification soaks: `Starlings.Soak.Tool <model> --duration 1h`, Release build at `b98c9ce`, Linux x64, Intel i7-5820K, run one at a time. Each cycle loads the model, serves concurrent requests (including timed-out ones), checks every output against the first cycle's, disposes the model, and samples memory loaded and unloaded. The growth rule is the tool's own (fitted post-warmup growth above the limit and continuing in both halves); it was not changed for these runs. `private` is address space on Linux and does not gate.

| Profile | Cycles | Wrong outputs / errors | Gating memory | Verdict | Report |
| --- | ---: | --- | --- | --- | --- |
| multilingual-e5-small `onnx-fp32-cpu` | 1,368 | 0 / 0 | plateau (loaded rss_anon about 940 to 990 MB, unloaded about 450 to 490 MB) | **PASS** | [e5](e5-20261014T172556Z.md) |
| embeddinggemma-2 `onnx-fp32-cpu` | see report | 0 / 0 | plateau | **PASS** | [embeddinggemma2](embeddinggemma2-20261014T192616Z.md) |
| qwen3.5-0.8b `ortgenai-fp32x-cpu` | see report | 0 / 0 | plateau | **PASS** | [qwen35](qwen35-20261014T212546Z.md) |
| gliner2-base-v1 `onnx-fp32-cpu` | 810 | 0 / 0 | loaded and unloaded rss_anon grow about 82 MB/h | **FAIL** | [gliner2](gliner2-20261014T182601Z.md) |
| embeddinggemma-2 `image-fp32-cpu` | 133 | 0 / 0 | plateau at the last checkpoint | **INCOMPLETE** | [embeddinggemma2-image](embeddinggemma2-image-20261014T202621Z.md) |

The e5 result supersedes the 5-minute development screening of 2026-10-13 that flagged growth: over one hour the gating series plateau.

The GLiNER2 growth is under investigation (loaded memory grows too, so it is not only allocator retention between loads); GLiNER2's G4 stays open until it is fixed or explained and a one-hour rerun passes. The EmbeddingGemma 2 image process was killed with SIGKILL at 3,540 s, after its 3,491 s checkpoint (which shows a plateau and no wrong output); the kernel log holds no OOM kill for it, so the cause is not established and the run is repeated. Video profiles have no soak workload yet. CUDA soaks follow on GPU 0.
