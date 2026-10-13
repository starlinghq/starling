# M0: e5-small on CUDA, C# parity, CPU thread counts, exploratory timing

Run on 2026-10-05 on a local workstation. `intfloat/multilingual-e5-small` at `614241f622f53c4eeff9890bdc4f31cfecc418b3`; `onnx/model.onnx` SHA-256 `ca456c06...`, `model.safetensors` `1a55775f...`, `tokenizer.json` `0b44a9d7...`, equal to the manifest. The reference is the accepted `onnx-fp32-cpu` evidence (`model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/`), read and not modified. Every run uses the fixture `input_ids`; nothing is tokenized here.

## Environment

| Item | Value |
| --- | --- |
| CPU | Intel Core i7-5820K @ 3.30GHz (Haswell-E), 6 cores, 12 threads. ISA flags: `avx`, `avx2`, `fma`, `f16c`. No AVX-512, no AVX-VNNI, no AMX. |
| RAM, OS | 94 GB; Linux 7.2.5 (Omarchy), glibc 2.44, x86-64 |
| GPU | NVIDIA GeForce RTX 4060 Ti 16 GB (sm_89), PCI 04:00.0, selected with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`. The RTX 3090 (03:00.0) was not used. |
| Driver | 610.57.04 (CUDA 13.3 capable) |
| Python environment | Python 3.11.16; `model-pipeline/reference/encoder/requirements-cuda.txt` (hashed): torch 2.14.1 (CUDA 13.0 build from PyPI), onnxruntime-gpu 1.30.0 (CUDA 13.0, build `f2c39fe2f`), transformers 5.18.0, sentence-transformers 6.1.0, tokenizers 0.23.2, onnx 1.23.1, numpy 2.4.6, nvidia-cudnn-cu13 9.24.0.43, nvidia-cuda-runtime 13.0.96, nvidia-cublas 13.1.1.3 |
| C# probe | .NET SDK 10.0.401, `net10.0`, Release; `Microsoft.ML.OnnxRuntime` 1.30.0 (CPU build) and `Microsoft.ML.OnnxRuntime.Gpu` 1.30.0 (CUDA build) in two builds of one throwaway console project outside the repository. The CUDA build loads CUDA 13 and cuDNN 9 from the Python environment's `nvidia-*` wheels through `LD_LIBRARY_PATH`; the NuGet package does not ship them. |
| Contention | The GPU and CPU were shared with other agents' jobs during every run (a Kev-0.8B CUDA job used 4 to 10 GB and up to 100% of the 4060 Ti; CPU load average 20 to 30 on 12 threads). This does not affect the numerical results (all runs were repeatable to 0); it makes the timings unreliable. |

Runners: `model-pipeline/reference/encoder/run_e5_cuda.py` (parity, placement), `compare_e5_csharp.py` (C# against Python dumps), `bench_e5_ort.py` (timing). The C# probe, raw results, dumps and model snapshot are under `/home/mark/Projects/AI-models/e5/` (not in Git).

## 1. CUDA parity (Python)

Reference: the protected embeddings (sentence-transformers modules, PyTorch fp32 CPU). The fixtures do not store hidden states, so the runner recomputes the PyTorch fp32 CPU hidden states with the same modules and checks its pooled output against the protected embeddings first: max abs 1.86e-7 on this machine. Candidates pool with NumPy float32 mean over the mask, then L2 normalization. Cosine is computed in float64 (the CPU `parity.json` used float32 norms, which limits it to about 6e-8). 46 single-text cases. `parity.json` in `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cuda/` (proposed):

| Candidate | Hidden max abs | Hidden mean abs | Hidden p99 abs | Embedding max abs | Embedding mean abs | Embedding p99 abs | Min cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ORT CUDA EP, `use_tf32=0` | 2.11e-6 | 1.81e-7 | 6.78e-7 | 1.94e-7 | 1.87e-8 | 6.71e-8 | 0.99999999999956 |
| ORT CUDA EP, ORT defaults (`use_tf32=1`) | 1.54e-3 | 7.57e-5 | 2.74e-4 | 7.42e-5 | 9.79e-6 | 3.74e-5 | 0.99999987934 |
| PyTorch CUDA fp32 (TF32 off, deterministic) | 1.88e-6 | 1.90e-7 | 6.85e-7 | 2.07e-7 | 2.13e-8 | 7.82e-8 | 0.99999999999939 |

Padded batches, embedding max abs:

| Candidate | 6-row batch fixture vs protected | 6-row batch vs its own single rows | 46 cases in batches of 16 vs protected | Batches of 16 vs own single rows |
| --- | ---: | ---: | ---: | ---: |
| ORT CUDA, `use_tf32=0` | 1.01e-7 | 7.26e-8 | 1.42e-7 | 9.69e-8 |
| ORT CUDA, defaults | 1.37e-4 | 1.21e-4 | 1.46e-4 | 1.14e-4 |
| PyTorch CUDA fp32 | 1.02e-7 | 8.94e-8 | 1.97e-7 | 9.69e-8 |

- ORT 1.30.0's CUDA EP reports `use_tf32: 1` by default. TF32 rounds matmul inputs to a 10-bit mantissa; the hidden-state error grows by about 700× (worst case `special-lookalike-spaced`, 25 tokens). With `use_tf32=0` CUDA matches the reference as closely as ORT CPU does. The proposed profile pins `use_tf32: 0`.
- Worst hidden-state case with TF32 off: `long-english` (512 tokens), 2.11e-6.
- No NaN or Inf. Re-running the two longest cases on the same session gives identical hidden states (0) for every candidate; the GPU results are run-to-run deterministic on this setup.
- TensorRT EP: untested. `onnxruntime-gpu` lists `TensorrtExecutionProvider`, but the TensorRT libraries are not installed; session creation falls back to CUDA and CPU. Installing TensorRT was not attempted.

### Execution-provider placement

From an ORT profile of the first run of each session (optimized graph, `placement.json`) and ORT's placement log:

| Session | CUDA EP nodes | CPU EP nodes | CPU nodes |
| --- | ---: | ---: | --- |
| `model.onnx`, CUDA EP (either TF32 setting) | 349 | 2 | `/embeddings/Gather`, `/embeddings/Unsqueeze` |
| fp16 graph, CUDA EP | 350 | 2 | same |
| `model.onnx`, CPU EP | 0 | 351 | all |

ORT logs the two CPU nodes as "Force fallback to CPU execution ... because the CPU execution path is deemed faster", that is, shape-path nodes it assigns to CPU on purpose, and warns "Some nodes were not assigned to the preferred execution providers". The C# CUDA session logs the same warning. `MemcpyTransformer modified: 0`: no copy nodes were inserted. The CUDA EP runs the fused graph: 12 `FusedMatMul`, 12 `BiasGelu`, 25 `LayerNormalization`, 84 `MatMul`, 12 `Softmax`, 98 `Add`.

## 2. C# on CPU and CUDA

The C# probe feeds the fixture ids from `embeddings.jsonl` (and the `tokenizer-batch.json` row ids), mean-pools over the attention mask with double accumulation, L2-normalizes, and compares with the protected embeddings. It writes hidden states and embeddings for a byte comparison with the Python dumps (`compare_e5_csharp.py`).

| C# run | Embedding max abs | Mean abs | p99 abs | Min cosine | 6-row batch vs protected | Batch vs own single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `Microsoft.ML.OnnxRuntime`, CPU, 1 thread | 2.33e-7 | 1.95e-8 | 7.26e-8 | 0.99999999999945 | 2.09e-7 | 0 |
| `Microsoft.ML.OnnxRuntime`, CPU, 12 threads | 2.33e-7 | 1.95e-8 | 7.26e-8 | 0.99999999999945 | 2.09e-7 | 0 |
| `Microsoft.ML.OnnxRuntime.Gpu`, CPU EP, 12 threads | 2.33e-7 | 1.95e-8 | 7.26e-8 | 0.99999999999945 | 2.09e-7 | 0 |
| `Microsoft.ML.OnnxRuntime.Gpu`, CUDA, `use_tf32=0` | 1.94e-7 | 1.82e-8 | 6.71e-8 | 0.99999999999957 | 1.02e-7 | 7.17e-8 |
| `Microsoft.ML.OnnxRuntime.Gpu`, CUDA, `use_tf32=1` | 7.42e-5 | 9.79e-6 | 3.74e-5 | 0.99999987934 | 1.37e-4 | 1.21e-4 |

C# against Python ORT on the same provider and settings:

| Pair | Hidden states | Embedding (C# double pooling vs NumPy float32 pooling) max abs |
| --- | --- | ---: |
| CPU, 1 thread | bit-identical, 46/46 cases | 1.04e-7 |
| CPU, 12 threads | bit-identical, 46/46 | 1.04e-7 |
| GPU package on CPU EP, 12 threads, vs Python CPU 12 threads | bit-identical, 46/46 | 1.04e-7 |
| CUDA, `use_tf32=0` | bit-identical, 46/46 | 1.04e-7 |
| CUDA, `use_tf32=1` | bit-identical, 46/46 | 7.45e-8 |

The only C#/Python difference is pooling arithmetic, below one float32 ulp of the embedding values in the worst element.

## CPU thread-count robustness

ORT CPU EP, `onnx/model.onnx`, `intra_op_num_threads` 1 and 12 (`inter_op_num_threads` 1), against the protected PyTorch reference (hidden states recomputed as in section 1):

| Run | Hidden max abs | Hidden mean abs | Hidden p99 abs | Embedding max abs | Embedding mean abs | Embedding p99 abs | Min cosine | 6-row batch vs protected | Batch vs own single |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Python ORT, 1 thread | 2.89e-6 | 2.00e-7 | 7.60e-7 | 2.31e-7 | 1.99e-8 | 7.45e-8 | 0.99999999999945 | 2.09e-7 | 0 |
| Python ORT, 12 threads | 2.89e-6 | 2.00e-7 | 7.60e-7 | 2.31e-7 | 1.99e-8 | 7.45e-8 | 0.99999999999945 | 2.09e-7 | 0 |
| C# ORT, 1 thread | bit-identical to Python | | | 2.33e-7 | 1.95e-8 | 7.26e-8 | 0.99999999999945 | 2.09e-7 | 0 |
| C# ORT, 12 threads | bit-identical to Python | | | 2.33e-7 | 1.95e-8 | 7.26e-8 | 0.99999999999945 | 2.09e-7 | 0 |

- 1 and 12 threads give bit-identical hidden states on every case (max abs 0). Padded batches (the 6-row fixture and 46 cases in batches of 16) are bit-identical to single-row runs on CPU.
- On this AVX2 CPU the ORT-vs-PyTorch hidden-state maximum is 2.89e-6 (`special-lookalike-spaced`), against 2.09e-6 in the accepted cloud run on a Xeon (`docs/m0/reference-runs.md` section 1). ORT CPU selects kernels by ISA, so CPU results depend on the CPU model; thread count did not matter here. Tolerances set from the cloud run alone would be tighter than this machine's observed error.

## 3. Exploratory timing (toward P-008)

Exploratory, not a benchmark result: one machine, shared GPU and heavily loaded CPU (see Environment), two passes for CUDA and one for CPU. Same artifact (`onnx/model.onnx`), same ORT version and provider settings, same workload file (`bench_e5_ort.py workloads`, fixture ids, right padding with id 1) in both languages. Timed region: one `Run` call with host inputs and a host output, which includes the device-to-host copy and therefore GPU synchronization; tokenization excluded. Warm: one session per process, 10 warmup runs per workload, then n timed samples. CUDA used `use_tf32=0` and ORT's default thread settings; CPU used `intra_op_num_threads=6`, `inter_op_num_threads=1`. Workloads: `short-bN` cycles the 44 cases under 512 tokens in file order (batch 1 is the 2-token `empty` case; batches of 8 and 32 pad to 44 tokens); `long512-bN` repeats `long-english` (512 tokens).

Median ms (p5–p95):

| Workload | Python CUDA pass 1 | Python CUDA pass 2 | C# CUDA pass 1 | C# CUDA pass 2 | Python CPU 6 threads | C# CPU 6 threads |
| --- | --- | --- | --- | --- | --- | --- |
| short-b1 (1 × 2), n=300 | 4.19 (2.96–7.81) | 7.97 (4.64–13.83) | 4.25 (3.34–7.30) | 7.31 (2.99–11.96) | 16.6 (7.6–28.5) | 13.5 (8.4–23.0) |
| short-b8 (8 × 44), n=200 | 4.51 (3.60–10.02) | 7.77 (5.54–14.34) | 5.74 (4.29–10.70) | 13.74 (10.02–18.29) | 260 (186–341) | 298 (221–401) |
| short-b32 (32 × 44), n=100 | 11.2 (10.1–13.8) | 14.3 (9.6–20.2) | 14.9 (9.7–20.4) | 11.7 (10.0–17.0) | 707 (599–816) | 1,010 (869–1,171) |
| long512-b1 (1 × 512), n=50 | 6.56 (5.99–12.33) | 9.39 (7.21–11.76) | 8.75 (5.92–12.53) | 14.7 (9.5–20.1) | 470 (373–562) | 545 (432–734) |
| long512-b8 (8 × 512), n=20 | 65.1 (62.6–68.8) | 60.4 (58.2–65.4) | 73.6 (66.4–108.7) | 60.7 (57.2–67.1) | 2,456 (2,237–2,646) | 2,975 (2,741–3,186) |
| long512-b32 (32 × 512), n=10 | 275 (265–289) | 260 (254–325) | 286 (260–319) | 267 (262–287) | 9,547 (8,977–11,386) | 10,694 (10,220–11,583) |

- Pass-to-pass variation within one language (up to 2× for small CUDA batches) is larger than the C#/Python difference, so these runs cannot measure binding overhead. A usable P-008 comparison needs an idle GPU and CPU, interleaved repetitions, and more independent runs.
- CPU runs overlapped other jobs (load average 20 to 30 on 12 hardware threads, wide p5–p95 spreads); they are recorded, not interpreted.
- On CUDA, 512-token rows run at about 120 to 135 rows/s at batch 8 and 32; small batches are dominated by fixed per-call overhead.

## 4. fp16 drift (separate profile candidate, not the reference)

Same reference and inputs as section 1. `results/exploratory/exploratory.json` in the scratch directory.

| Candidate | Hidden max abs | Hidden mean abs | Hidden p99 abs | Embedding max abs | Embedding mean abs | Embedding p99 abs | Min cosine | Batch of 16 vs own single, max abs |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ORT CUDA, fp16 graph (`convert_float_to_float16(keep_io_types=True)`, SHA-256 `f49ade3f...`, 235 MB) | 2.39e-3 | 2.46e-4 | 9.05e-4 | 1.77e-4 | 3.09e-5 | 1.06e-4 | 0.99999945 | 2.25e-4 |
| PyTorch CUDA, `torch.autocast(float16)` | 9.15e-4 | 9.24e-5 | 3.33e-4 | 1.01e-4 | 1.50e-5 | 5.29e-5 | 0.99999979 | 1.37e-4 |
| PyTorch CUDA, `model.half()` | 3.75e-3 | 1.66e-4 | 6.41e-4 | 2.22e-4 | 2.34e-5 | 8.37e-5 | 0.99999936 | 1.87e-4 |
| PyTorch CUDA fp32 with TF32 matmul | 6.16e-4 | 5.86e-5 | 2.17e-4 | 5.90e-5 | 8.39e-6 | 3.00e-5 | 0.99999992 | 6.68e-5 |

No NaN or Inf in any fp16 run. fp16 is about 1,000× further from the reference than fp32 in hidden states; per plan 10.1 it would be a separate variant with a task-quality budget, not a numerically equivalent profile. Task-level effects (retrieval ranking) were not measured.

## 5. Proposed artifacts

- Manifest: `model-pipeline/registry/multilingual-e5-small/onnx-fp32-cuda/manifest.yaml`, maturity experimental, validated by `model-pipeline validate registry`. `execution.provider_options: {use_tf32: 0}`; Linux x64 CUDA experimental (verified 2026-10-05), Windows x64 CUDA untested; TF32, fp16 and TensorRT listed as unsupported in this profile. G0 checklist; G1, G2, G6 not run (no tolerances).
- Parity evidence: `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cuda/` (`parity.json`, `placement.json`, `environment.json`, `PROPOSED.md`). No new goldens, no tolerance file.
- The schema cannot express: the execution-provider name and its allowed CPU fallback set, native runtime requirements (CUDA 13 runtime, cuDNN 9, minimum driver), and GPU architecture. The manifest records them in comments and `known_issues`.
