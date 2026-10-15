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
| Qwen `ortgenai-fp32x-cpu` | **FAIL** (profile stays Preview) | time to first token 1.119 [0.848, 1.447] on `short-decode64` against 1.10; the gap is inside the native prefill call, and the two sides load different native ONNX Runtime 1.30.0 builds (NuGet against pip). Cause not established; see [Qwen CPU time to first token](#qwen-cpu-time-to-first-token). **Rerun on a quiet host: PASS, qualifying** (1.045 [0.823, 1.274]); the gap was host load, with no configuration or build effect. See [the diagnosis](#qwen-cpu-time-to-first-token-diagnosis-on-a-quiet-host) |
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
- The swapped library's digest was not recorded, because that run predates `native_library_digests`, and its build directory was not kept. The swap is therefore as the investigating run reported it, not independently verified.
- Those rounds ran on a loaded machine (load average 6 to 10). A later C# run of the same binary had a pooled median of 296.7 ms, so the run-to-run spread there exceeds the effect.

The build difference is a documented fact. Its effect on prefill is an open question.

Conclusion: G5 stays **failed** for `ortgenai-fp32x-cpu`, and the profile stays **Preview**. The run is not repeated in search of a pass. The budget compares the shipped products, and C# users of the package get the NuGet build, so a pass obtained some other way would not describe what they run. From now on, every run records the path and SHA-256 of each native ONNX Runtime and GenAI library it loaded (`environment.native_library_digests`). `model-pipeline bench g5` prints whether the two sides' builds are identical, as context rather than as a verdict input. The raw files of this round predate that field, so `bench g5` reports their builds as "unavailable". The digests above were taken from the files the runs loaded, at the paths in their `native_libraries`. A draft upstream report, not filed, is [`upstream-ort-nuget-vs-pip-draft.md`](upstream-ort-nuget-vs-pip-draft.md).

### Qwen CPU time to first token: diagnosis on a quiet host

This was a diagnosis, not a repeat in search of a pass. Its plan was fixed before measuring. If a configuration difference explained the gap, fix the C# defaults. If only the native build explained it, report the build difference and ship nothing. Then run the qualifying G5 once on the same host, and let that result stand whichever way it went. Raw files are in [`g5-rerun/`](g5-rerun/).

**Effective configuration: identical by construction.** Both sides hand the same inputs to the same GenAI binary (`libonnxruntime-genai.so` `0cdf8076…` on both), and that binary builds the ORT session:

- the same bundle `genai_config.json`, whose decoder `session_options` are only `{"log_id": "onnxruntime-genai", "provider_options": []}`;
- the same two overlays in the same order: `intra_op_num_threads` 6, then `pad_token_id` 248319;
- `clear_providers` (CPU);
- the same search options: `max_length` = prompt + max_new_tokens, `do_sample` false.

Neither side sets inter-op threads, spinning, affinity, execution mode, graph level, memory pattern or arena, so GenAI's defaults apply to both. On CPU, `GenAISequence.Invoke` calls the native function directly, with no lock and no thread hop. No configuration difference was found to align.

**Diagnosis.** The host was a RunPod pod: AMD EPYC 7H12, 256 logical CPUs, load average about 10, no cgroup CPU quota. Each run pinned itself to physical cores 16 to 21 (NUMA node 0). The workload was warm `short-decode64` with `--samples-scale 3`: 30 samples per run, 6 per prompt. The runs, interleaved:

- (a) C# as shipped (NuGet `libonnxruntime.so` `245a6f8c…`), and Python;
- (b) C# with the pip `libonnxruntime.so` swapped in; its recorded digest is `c902c70b…`;
- (c) C# with `DOTNET_TieredPGO=0`, with `DOTNET_TieredCompilation=0`, and with `DOTNET_gcConcurrent=0`. These are the only process-level differences between the sides that we control.

Each cell is the native prefill median / minimum, in ms. Raw files: [`g5-rerun/prefill-diagnosis/`](g5-rerun/prefill-diagnosis/).

| Run | 20 tokens | 22 tokens | 29 tokens | 40 tokens | 61 tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| (a) C# NuGet build | 144.5 / 143.1 | 142.6 / 141.0 | 160.7 / 159.5 | 201.7 / 200.9 | 273.1 / 272.9 |
| (a) Python | 144.5 / 143.2 | 142.4 / 141.3 | 161.8 / 157.3 | 201.7 / 200.1 | 275.3 / 273.9 |
| (b) C# pip build | 144.6 / 143.0 | 143.5 / 140.8 | 161.0 / 159.7 | 202.3 / 201.0 | 272.7 / 272.3 |
| (c) C# TieredPGO off | 146.3 / 145.5 | 140.3 / 139.6 | 158.0 / 156.3 | 198.7 / 196.9 | 277.1 / 274.7 |
| (c) C# tiered compilation off | 147.8 / 146.5 | 142.4 / 141.0 | 159.2 / 157.4 | 200.6 / 199.1 | 274.9 / 273.7 |
| (c) C# concurrent GC off | 143.3 / 141.9 | 142.0 / 141.4 | 157.7 / 155.4 | 199.3 / 198.1 | 276.2 / 274.4 |

On a quiet host, the shipped C# prefill matches Python's to within about 1% at every prompt length, the 22-token prompt included (142.6 ms against 142.4 ms). Neither the native build nor any runtime setting moves it. The 1.119 of the first round came from host load: that run was on the shared workstation at load average 5 to 12. No code or default was changed.

**Qualifying rerun.** One `run-g5-matched.sh qwen35 cpu … 3 6 10` on the same pod, at clean commit `12907ef`, with both sides pinned to the same cores. The load average was 6 to 59 on 256 logical CPUs (see `qwen35-cpu.load.tsv`). Result: **PASS**, qualifying. Full verdict: [`g5-rerun/qwen35-cpu.jsonl.verdict.txt`](g5-rerun/qwen35-cpu.jsonl.verdict.txt).

- Time to first token, `short-decode64`: 1.045 [0.823, 1.274]. The per-length warm prefill medians are equal (22 tokens: 142.1 ms against 142.6 ms).
- Other time-to-first-token ratios: 0.994 (long-prefill), 1.014 (tools), 1.026 (text).
- Decode: 0.992 to 1.037.

`bench g5` notes that the ORT native builds differ, as context only. Under the rule above, this G5 pass for `ortgenai-fp32x-cpu` stands. The rerun measured the shipped NuGet build, so it describes what C# users run. Promoting the profile's status is a separate change.

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
