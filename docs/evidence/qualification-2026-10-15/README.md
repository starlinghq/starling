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
| Qwen `ortgenai-fp32x-cpu` | **FAIL** (profile stays Preview) | time to first token 1.119 [0.848, 1.447] on `short-decode64` against 1.10; the gap is inside the native prefill call, and the two sides load different native ONNX Runtime 1.30.0 builds (NuGet against pip). Cause not established; see [Qwen CPU time to first token](#qwen-cpu-time-to-first-token) |
| EmbeddingGemma 2 `image-fp32-cuda` | **FAIL** | preprocessing 8 to 48% of embed against 5%; photo warm ratio 1.118 against 1.10 |
| EmbeddingGemma 2 `video-fp32-cuda` | **FAIL** | preprocessing 11 to 36% of embed against 5%; 1080p warm ratio 1.229 |
| GLiNER2 `onnx-fp32-cuda` | **FAIL** | prepare + decode 8 to 13% of inference against 5% |

The failures are real. On a fast GPU, the C# CPU-side work (image and video preprocessing, GLiNER2 preparation and decoding) becomes a large share of the request. Those three are being fixed in code. On the Qwen CPU path, C# takes longer than Python to produce its first token, but the time is spent inside the native prefill call, not in C# code; see the next section. Budgets are unchanged, and a fixed profile is benchmarked again under this protocol.

### Qwen CPU time to first token

Where the time goes: the C# time outside the native prefill call is 2.2 to 2.4 ms per request, against 2.1 ms in Python. On CPU, both sides prefill the whole prompt in one call. The gap is therefore inside that native call.

What differs inside it: both sides report ONNX Runtime 1.30.0 and ONNX Runtime GenAI 0.17.1. `libonnxruntime-genai.so` is byte-identical on both sides, but `libonnxruntime.so` is not:

| Side | Library loaded | SHA-256 |
| --- | --- | --- |
| C# (NuGet `Microsoft.ML.OnnxRuntime` 1.30.0) | `runtimes/linux-x64/native/libonnxruntime.so` | `245a6f8c38127551057a1cd1ffd59f0a186a227ade4f3492dea2494eb565542e` |
| Python (pip `onnxruntime` 1.30.0) | `onnxruntime/capi/libonnxruntime.so.1.30.0` | `c902c70b3003c0e99fada202f37478c515ae9bba7944c2b2abd0017bae0c82ed` |
| both (`onnxruntime-genai` 0.17.1) | `libonnxruntime-genai.so` | `0cdf8076ecf75eebe2bd1507c7904651e3d27d6f0b853ed361772ab25c1f2d40` |

The table below gives time to first token per prompt length on `short-decode64`, recomputed from [`g5/qwen35-cpu.jsonl`](g5/qwen35-cpu.jsonl). Each figure is the median of 6 warm samples per side (3 runs, 2 per run); the ratio is C# over Python.

| Prompt tokens | C# median (ms) | Python median (ms) | Ratio |
| ---: | ---: | ---: | ---: |
| 20 | 143.67 | 137.83 | 1.042 |
| 22 | 167.72 | 145.03 | 1.156 |
| 29 | 184.16 | 166.74 | 1.104 |
| 40 | 223.97 | 220.17 | 1.017 |
| 61 | 318.79 | 313.21 | 1.018 |
| pooled | 187.04 | 167.14 | 1.119 (the gated value) |

Most of the failing ratio comes from the 22-token prompt. The longer prompts are close to 1.0: tools at 286 tokens is 0.982, and long-prefill at 1,666 tokens is 1.011.

Is the native build the cause? This is not established. A follow-up swapped the pip `libonnxruntime.so` into the C# harness and ran C# with the NuGet build, C# with the pip build, and Python, interleaved, for two rounds of a one-token time-to-first-token probe. The results did not show the pip build to be faster under C#:

- Pooled medians: 207.5 ms (NuGet build), 229.2 ms (pip build), 205.3 ms (Python).
- At 22 tokens: 171.2, 204.4 and 151.0 ms.
- Per-length minima were nearly equal across all three (22 tokens: 138.8, 138.1 and 137.9 ms).
- Those rounds ran on a loaded machine (load average 6 to 10). A later C# run of the same binary had a pooled median of 296.7 ms, so the run-to-run spread there exceeds the effect.

The build difference is a documented fact. Its effect on prefill is an open question.

Conclusion: G5 stays **failed** for `ortgenai-fp32x-cpu`, and the profile stays **Preview**. The run is not repeated in search of a pass. The budget compares the shipped products, and C# users of the package get the NuGet build, so a pass obtained some other way would not describe what they run. From now on, every run records the path and SHA-256 of each native ONNX Runtime and GenAI library it loaded (`environment.native_library_digests`). `model-pipeline bench g5` prints whether the two sides' builds are identical, as context rather than as a verdict input. The raw files of this round predate that field, so `bench g5` reports their builds as "unavailable". The digests above were taken from the files the runs loaded, at the paths in their `native_libraries`. A draft upstream report, not filed, is [`upstream-ort-nuget-vs-pip-draft.md`](upstream-ort-nuget-vs-pip-draft.md).

Raw results and full verdicts: [`g5/`](g5/) (`<profile>.jsonl`, `<profile>.jsonl.verdict.txt`).
