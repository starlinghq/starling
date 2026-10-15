# Benchmark protocol (G5)

How the repository measures performance for gate G5 ("controlled benchmark with raw results, environment manifest, and stage timing"; plan section 10) and for the benchmark program of plan section 11. It covers the harnesses under `ai-dotnet/benchmarks/`: the e5 matched-backend harness (P-008, details in `ai-dotnet/benchmarks/README.md`) and the model harnesses for GLiNER2, EmbeddingGemma 2 and Qwen3.5.

**Budgets are maintainer decisions.** A harness measures and checks quality; it never decides whether a profile is fast enough. The G5 performance budget of each profile is chosen by the maintainer (production-readiness plan, action M2) and recorded in the manifest's gate entry with the evidence link. Until a budget exists, a G5 entry cannot pass, whatever the numbers. The figures in plan section 11 (p95 adapter overhead no more than the larger of 1 ms or 10%; at least 95% of matched native-backend decode throughput; alert on a repeatable regression above 5%) are proposals to calibrate, not budgets.

Development runs of each harness (uncontrolled, not G5 evidence): [GLiNER2](gliner2-2026-10-13.md), [EmbeddingGemma 2](embeddinggemma2-2026-10-13.md), [Qwen3.5](qwen35-2026-10-13.md).

## Harnesses

| Harness | Profile(s) | Paths and workloads (`ai-dotnet/benchmarks/workloads/`) | Stages timed | Quality checked on every series |
| --- | --- | --- | --- | --- |
| `Starlings.Benchmarks.E5Harness` (+ `.Cuda`, Python `bench_e5_matched.py`) | e5 `onnx-fp32-cpu`, `onnx-fp32-cuda` | `e5-v1.json`: ids and text, short and 512-token inputs, batch 1/8/32, concurrency 1 and 4 | library: `tokenize`, `encode`; direct and Python: `tokenize`, `prep`, `run`, `pool` | embeddings against the protected rows, max abs within the profile tolerance |
| `Starlings.Benchmarks.Gliner2Harness` (+ `.Cuda`) | GLiNER2 `onnx-fp32-cpu`, `onnx-fp32-cuda` | `gliner2-v1.json`: short texts with 1, 3, 5 and 12 labels; batch 8; first 100/250/400 words of a long fixture text | `prepare` (normalization, word split, tokenization, schema and index tensors), `inference` (padding and the ORT run), `decode` (spans to entities) | entities equal to the protected reference under the protected order and threshold-flip rules, confidence within `float.confidence_max_abs`, logits finite |
| `Starlings.Benchmarks.EmbeddingGemma2Harness` (+ `.Cuda`) | EmbeddingGemma 2 `onnx-fp32-{cpu,cuda}` (text), `image-fp32-{cpu,cuda}` | `embeddinggemma2-v1.json`: text at about 10–50 tokens (batch 1/8/16), first 100 and 400 words of a long text, 1,570–1,750 tokens (batch 1/4); images: photos, 1×1 and 8×6, 4000×3000, batch 4, two images interleaved with text, preprocessing alone | text: `tokenize`, `encode` (padding, ORT run, pooling); image: `decode` (SkiaSharp adapter from bytes in memory), `embed` (preprocessing, vision graph, multimodal graph: one public call); `preprocess-only`: `decode`, `preprocess` | embeddings against the protected rows (`float.embedding_max_abs`); preprocessing: pixel-values SHA-256 equal to the fixture digest |
| `Starlings.Benchmarks.EmbeddingGemma2Harness` (+ `.Cuda`) `--paths video` | EmbeddingGemma 2 `video-fp32-{cpu,cuda}` | `embeddinggemma2-v1.json` video workloads, on decoded frames (the reference's sampled RGB frames from `run_embeddinggemma2_video.py --tensors`, `STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS`, every array checked against the protected `video-fp32-cpu/tensors.json`; no video decoder in the process): 3–6 frames (batch 1 and 4), 4 frames of 1920×1080, 32 frames (4,226 tokens), text with one and two videos, `video-preprocess-only` | `embed` (per-frame preprocessing at 140 soft tokens, up to 8 frames in parallel ahead of the vision calls, the vision graph in calls of at most 2 frames, the multimodal text graph: one public call); `video-preprocess-only`: `preprocess` (the batch processor on up to 8 frames at a time, as the embedder runs it) | embeddings against the protected video rows (`float.embedding_max_abs`); preprocessing: per-video pixel-values SHA-256 equal to the fixture digest |
| `Starlings.Benchmarks.Qwen35Harness` (+ `.Cuda`) | Qwen3.5 `ortgenai-fp32x-cpu`, `ortgenai-fp32x-cuda`, `ortgenai-fp16x-cuda` (`--variant fp16x`) | `qwen35-v1.json`: five short prompts with 64 new tokens (ids and text path), the tools prompt (286 tokens, 32 new), the 1,666-token prompt with 8 new tokens | `template` (chat template and tokenizer; text path only), `prefill`, `decode`; per-sample metrics `ttft_ms`, `prompt_tokens`, `completion_tokens`, `prefill_tokens_per_s`, `decode_tokens_per_s`, `inter_token_ms` | greedy ids against the protected reference output up to `max_new_tokens` under the profile's `discrete.greedy_ids` rule: fp32x exact (a near-tie divergence also fails and reports its reference margin); fp16x equal up to the first divergence, which must fall where the reference top-1 margin is below 0.05; prompt ids equal to the reference |

Workloads that have no upstream reference (GLiNER2 and EmbeddingGemma 2 `synthetic` lengths) are checked against the same input processed alone at load in the same process. That catches a batch, padding or state defect, not a fidelity defect; the fidelity evidence for those lengths is the gate G1/G2 work.

Video runs on decoded frames, so the comparison times what follows decoding (preprocessing, the vision graph, the multimodal text graph) the same way on both sides; decoding itself (the FFmpeg 8 LGPL adapter against torchcodec) is not benchmarked. The decoded-frame fixtures are the spike's reference tensors (`run_embeddinggemma2_video.py --tensors` on the video set from `make_eg2_videos.py`, torchcodec 0.17.0 on FFmpeg n8.1.3, the LGPL shared build from `fetch_eg2_ffmpeg.sh`; `video-fp32-cpu/videos.json` and `decoders.json` record the builds), outside Git and pinned by `tensors.json`. See "Gaps".

**CUDA builds.** Every harness has a `.Cuda` build (`Starlings.Benchmarks.<Model>Harness.Cuda`): the same sources compiled with `ORT_CUDA` against the CUDA natives (`Microsoft.ML.OnnxRuntime.Gpu.Linux`; Qwen: `Microsoft.ML.OnnxRuntimeGenAI.Cuda`), with their own lock files and outside `AI.slnx`, like the `*.Tests.Cuda` projects, so CI never restores the GPU packages. A CUDA build runs only `--provider cuda` and a CPU build only `--provider cpu`, so a result never comes from the other native package; CUDA needs `CUDA_DEVICE_ORDER=PCI_BUS_ID` and an explicit `CUDA_VISIBLE_DEVICES`, uses device 0 of it and TF32 off (`use_tf32=0`, checked after configuration), and records the provider options the session reports. A CUDA profile is judged against the same protected reference rows as its CPU profile, with the CUDA profile's own `tolerances.yaml` (hashed into `fixture_sha256`).

## What every run records

The raw format is `starlings-bench-raw/1` (`model-pipeline/schemas/benchmark-raw.schema.json`), JSON lines, one `run` record per process and one `series` record per timed series. Summaries are computed from the raw records (`model-pipeline bench summarize`), never stored instead of them.

- **Raw per-sample results.** Each series holds every sample's total (`ms`), every sample's stage times (`stages_ms`) and, for generation, every sample's metrics (`metrics`). No pre-aggregation.
- **Environment manifest** (`run.environment`): CPU model and ISA flags, physical and logical cores, RAM, OS, kernel, .NET runtime and GC settings, ONNX Runtime version and providers, session providers, loaded native libraries (`native_libraries`), the path and SHA-256 of every ONNX Runtime and GenAI native library the process mapped (`native_library_digests`, read from `/proc/self/maps`; `"unavailable"` off Linux), CPU governor and turbo state, load average and GPU utilization before and after, and a free-text machine note (`--note`).
- **Artifacts** (`run.artifacts`): model id and revision, SHA-256 of the model graph(s) and `tokenizer.json`, workload file and fixture SHA-256, git commit and dirty flag, package versions including `Starlings.Tokenizers` (`tokenizer_version`), ONNX Runtime and, for Qwen, ONNX Runtime GenAI.
- **Settings** (`run.settings`): provider, intra- and inter-op threads, execution mode, optimization level, provider options, model-specific limits.
- **Warm/cold split.** `warm`: one process loads the model once (`run.load_ms`), runs a preliminary quality sweep over the fixture cases, then for each workload `warmup_calls` untimed calls followed by `n` timed calls cycling through the workload's requests. `cold`: the parent starts a fresh process per sample and passes its spawn time on CLOCK_MONOTONIC; the child records `spawn_to_main`, its load stages, the first request's stages and `spawn_to_result`; the parent adds `process_wall`. Model files are in the OS page cache (warm OS cache, cold process).
- **Quality.** The preliminary sweep (`run.quality`) and every series (`series.quality`, recomputed after the timed loop over every request of the workload) must pass, or the run exits 1 and its timings are excluded from every summary table ("fast but wrong" is flagged, never reported as a result).

## Procedure for a controlled (G5) run

A G5 result needs a controlled machine (plan 13.1; maintainer action M6). Development runs on a shared workstation are labelled "uncontrolled development numbers" and are not G5 evidence.

1. Quiet machine: no other workload, load average near zero, fixed CPU governor (`performance`), turbo state recorded, no GPU work for CPU runs. Record what shares the machine in `--note`.
2. Fixed settings per profile: `--threads` = physical cores (6 on the reference workstation), `--inter-threads 1`, Release build, the pinned packages. Never compare raw files with different model SHA-256, workload SHA-256 or package versions.
3. Several independent processes: at least 3 warm rounds (one process each) and at least 10 cold processes per harness, interleaved with any comparison implementation so drift and foreign load hit both sides.
4. Full sample counts (`--samples-scale 1`). The summary reports medians with 95% hierarchical bootstrap intervals (runs, then samples), p5/p95 only with at least 20 samples and p99 only with at least 1,000.
5. Publish the raw files (outside Git, pinned by SHA-256) with the generated summary, and every quality failure. A failing comparison is not hidden by changing workloads or dropping a slow configuration.
6. The maintainer compares the result with the profile's budget and records the G5 entry.

## Running

```sh
export DOTNET_ROOT=~/.dotnet STARLING_MODELS=/path/to/model-cache
dotnet build -c Release ai-dotnet/benchmarks/Starlings.Benchmarks.Gliner2Harness   # likewise the other two
H=ai-dotnet/benchmarks/Starlings.Benchmarks.Gliner2Harness/bin/Release/net10.0/Starlings.Benchmarks.Gliner2Harness
$H warm --threads 6 --out raw.jsonl --note "machine description"
$H cold --threads 6 --processes 10 --out raw.jsonl
# EmbeddingGemma 2 images: STARLING_EMBEDDINGGEMMA2_IMAGES=<make_eg2_images.py --out dir>; --paths text,image; --cold-path text|image
# Qwen3.5: STARLING_QWEN35_SNAPSHOT=<snapshot> or the acquisition cache under STARLING_MODELS
model-pipeline bench validate raw.jsonl && model-pipeline bench summarize raw.jsonl --json summary.json > summary.md
```

Options common to the model harnesses: `--only <workloads>`, `--samples-scale F`, `--hash-model true` (check the graph SHA-256 at load), `--workloads <file>`. Exit codes: 0 pass, 1 quality failed (timings invalid), 2 usage error, 3 assets missing (the message names what is missing and how to get it; nothing is downloaded).

## Python same-backend comparison (G5 budgets)

The G5 budgets (`model-pipeline/protected/performance-budgets.yaml`) are ratios against a Python run on the same backend, so every budgeted profile has a Python runner that matches its C# harness workload for workload: the same graph files, the same workload file and fixture rows, the same thread settings (`--threads`, `--inter-threads 1`, sequential execution, all optimizations; CUDA `device_id` 0, `use_tf32` 0), the same warm/cold protocol (warmup calls, `n` timed calls cycling the requests, every request judged after the loop; fresh `cold-child` processes started with their spawn time) and the same raw format. The runners share `model-pipeline/reference/bench/matched_core.py`.

| Profile(s) | C# harness (CUDA: its `.Cuda` build) | Python runner | What the Python side runs | Environment |
| --- | --- | --- | --- | --- |
| e5 `onnx-fp32-{cpu,cuda}` | `E5Harness` | `reference/encoder/bench_e5_matched.py` | see `ai-dotnet/benchmarks/README.md` | `encoder/requirements-bench.txt`, `-cuda.txt` |
| EmbeddingGemma 2 `onnx-fp32-{cpu,cuda}` | `EmbeddingGemma2Harness --paths text` | `reference/bench/bench_eg2_matched.py --paths text` | `tokenize` (prompt prepended, Hugging Face `tokenizers` on the same `tokenizer.json`), `encode` (the C# chunk plan of 16 rows and 8,192 cells, padding with 0, ORT run, float64 mean pooling, normalization) | `encoder/requirements-bench.txt`, `-cuda.txt` |
| EmbeddingGemma 2 `image-fp32-{cpu,cuda}` | `EmbeddingGemma2Harness --paths image` | `bench_eg2_matched.py --paths image` | `decode` (Pillow, RGB), `embed` (the pinned reference preprocessing: transformers 5.19.0 `Gemma4ImageProcessor`, torchvision backend, restated as in `run_embeddinggemma2_image.py`; rendering and tokenizing; the vision graph one image per call on its unpadded patches, as the C# defaults; soft tokens scattered into the multimodal text graph; pooling), also split into `embed_preprocess`, `embed_vision`, `embed_text`; `preprocess-only`: pixel-values digest | `embeddinggemma2/requirements.txt`, `-cuda.txt` |
| EmbeddingGemma 2 `video-fp32-{cpu,cuda}` | `EmbeddingGemma2Harness --paths video` | `bench_eg2_matched.py --paths video` | `embed` on the same decoded frames (read at load, checked against `tensors.json`): per frame the video processor at 140 soft tokens (transformers 5.19.0 size rule for 1,260 patches, uint8 bicubic antialiased resize, x 1/255, 16x16 patches, unpadded), rendering and tokenizing (`<|image>` + n x `<|video|>` + `<image|>` per frame), the vision graph on the chunk's frames in the C# call plan (at most 2 frames and 2,520 patch rows per call, padded to the call's largest frame), soft tokens at the `<|video|>` positions of the multimodal text graph, pooling; also split into `embed_preprocess` (it includes the frames' HWC-to-CHW tensor copy, which the C# side does not need), `embed_vision`, `embed_text`; `video-preprocess-only`: per-video pixel-values digest. The video cold start reads its frames inside `spawn_to_result` on both sides; no budget covers it | `embeddinggemma2/requirements.txt`, `-cuda.txt` |
| GLiNER2 `onnx-fp32-{cpu,cuda}` | `Gliner2Harness` | `reference/bench/bench_gliner2_matched.py --inputs <dump>` | `inference` only, on the exact tensors the C# `Prepare` built (`Gliner2Harness dump-inputs`): chunks of 8, right padding as `RunModel`, ORT run, outputs copied out; fixture items' tensors checked equal to the reference tensors, logits against the protected logits | `encoder/requirements-bench.txt`, `-cuda.txt` |
| Qwen3.5 `ortgenai-fp32x-cpu`, `ortgenai-{fp32x,fp16x}-cuda` | `Qwen35Harness` | `reference/bench/bench_qwen35_matched.py` | Python `onnxruntime-genai` 0.17.1 over `onnxruntime` 1.30.0 (the C# versions; 0.17.1 alone resolves onnxruntime 1.31.0, so both are pinned), the same bundle and overlays (threads, `pad_token_id` 248319), greedy, the prompt appended in one call on CPU and in 512-token chunks on CUDA (the C# defaults), stop ids 248044 and 248046; stages and metrics as the C# harness, TTFT to the first streamed text (incremental detokenizer) | `bench/requirements-genai.txt`, `-cuda.txt` (Python applies the same `discrete.greedy_ids` rule per profile) |

Python-side quality is checked on every series as on the C# side (embeddings against the protected rows; GLiNER2 logits, since Python has no decoder; Qwen greedy ids exact), and a failing run's timings are excluded.

**Budget check.** `model-pipeline bench g5 <raw files of both sides>` (`model-pipeline/src/model_pipeline/g5.py`) maps every series to its profile (model, provider and path), pairs C# library and Python series by mode, path, workload and concurrency, and applies each key of the profile; a key without an implemented rule is an error. Decisions use ratios of medians pooled over runs, reported with their 95% hierarchical bootstrap intervals; YAML comments (the e5 "upper 95% bound at most 1.15") are printed, not gated. Rules: `warm_median_ratio_max` and `cold_start_ratio_max` on sample totals (`warm_median_ratio_max` leaves out workloads that run no graph, EmbeddingGemma 2 `preprocess-only`, which compares SkiaSharp and the C# resampler with Pillow and torchvision rather than a backend); `inference_ratio_max` on the `inference` stage; `decode_tokens_per_second_ratio_min` and `time_to_first_token_ratio_max` on the `decode_tokens_per_s` and `ttft_ms` metrics; `adapter_overhead_max` as median library minus median direct at most max(1 ms, 10% of direct); `prepare_plus_decode_share_of_inference_max` per workload as the median of each sample's `prepare` + `decode` over the median `inference`; `preprocess_share_of_embed_max` per row of the path's preprocessing-only workload (image `preprocess-only`, video `video-preprocess-only`) as its median `preprocess` over the median `embed` of the samples that embedded that row alone (`photo-b1`, `huge-b1`; video `video-b1`, `video-1080p-b1`, `video-32-frames-b1`; samples map to requests as `i % requests`, from the workload file the run names). A workload one side lacks is "no data", never a pass. A profile is `pass`, `fail` (a rule fails, a series failed quality, or the sides differ in graph, workload, provider, threads or runtime version) or `incomplete`; it is G5 evidence only if it passes and qualifies (at least 3 warm runs and 10 cold processes per side, clean checkout). Exit code 0 only when every profile in the files passes and qualifies.

**Same version is not same build.** Both sides can report the same ONNX Runtime version and still load different native builds: the C# packages ship NuGet's `libonnxruntime.so`, the Python runners load pip's. `bench g5` therefore compares `native_library_digests` per library (`onnxruntime`, `onnxruntime-genai`, `onnxruntime_providers_*`, independent of version suffixes; a Python process whose only runtime is the `onnxruntime_pybind11_state` extension counts that as its `onnxruntime`) and prints, under each profile's header, `native builds: identical`, `NATIVE BUILDS DIFFER: same version, different native build`, or `native builds: unavailable` (a run without digests, such as a raw file written before they were recorded). This is context, never a verdict input: the budget compares the shipped products, and C# users get the NuGet builds, so a difference neither fails nor excuses a profile. When a ratio fails, read this line before attributing the gap to C# code (example: the 2026-10-15 Qwen CPU time to first token, `docs/evidence/qualification-2026-10-15/README.md`).

**Run commands (quiet machine, step 1 to 3 of the procedure).** Each runs 3 rounds of warm and cold processes, C# and Python interleaved, appending to one raw file:

```sh
export DOTNET_ROOT=~/.dotnet STARLING_MODELS=/path/to/model-cache BENCH_NOTE="machine description"
# venvs: uv venv <dir> --python 3.11 && uv pip install --python <dir>/bin/python -r <requirements file>
# multilingual-e5-small/onnx-fp32-cpu (and -cuda with requirements-bench-cuda.txt and CUDA_VISIBLE_DEVICES)
STARLING_E5_SNAPSHOT=<snapshot> BENCH_PYTHON=<venv: encoder/requirements-bench.txt>/bin/python \
  ai-dotnet/benchmarks/run-e5-matched.sh cpu e5-cpu.jsonl 3 6 10
# embeddinggemma-2/onnx-fp32-cpu
BENCH_PYTHON=<venv: encoder/requirements-bench.txt>/bin/python ai-dotnet/benchmarks/run-g5-matched.sh eg2-text cpu eg2-text-cpu.jsonl 3 6 10
# embeddinggemma-2/image-fp32-cpu
STARLING_EMBEDDINGGEMMA2_IMAGES=<make_eg2_images.py output> BENCH_PYTHON=<venv: embeddinggemma2/requirements.txt>/bin/python \
  ai-dotnet/benchmarks/run-g5-matched.sh eg2-image cpu eg2-image-cpu.jsonl 3 6 10
# embeddinggemma-2/video-fp32-cpu (decoded frames: run_embeddinggemma2_video.py --tensors output)
STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS=<tensors dir> BENCH_PYTHON=<venv: embeddinggemma2/requirements.txt>/bin/python \
  ai-dotnet/benchmarks/run-g5-matched.sh eg2-video cpu eg2-video-cpu.jsonl 3 6 10
# gliner2-base-v1/onnx-fp32-cpu (dumps the C# inputs to gliner2-cpu.gliner2-inputs.jsonl first)
BENCH_PYTHON=<venv: encoder/requirements-bench.txt>/bin/python ai-dotnet/benchmarks/run-g5-matched.sh gliner2 cpu gliner2-cpu.jsonl 3 6 10
# qwen3.5-0.8b/ortgenai-fp32x-cpu
BENCH_PYTHON=<venv: bench/requirements-genai.txt>/bin/python ai-dotnet/benchmarks/run-g5-matched.sh qwen35 cpu qwen35-cpu.jsonl 3 6 10
# CUDA profiles: the .Cuda builds of the C# harnesses and the -cuda requirements, on an explicit GPU
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>
STARLING_E5_SNAPSHOT=<snapshot> BENCH_PYTHON=<venv: encoder/requirements-bench-cuda.txt>/bin/python \
  ai-dotnet/benchmarks/run-e5-matched.sh cuda e5-cuda.jsonl 3 6 10
BENCH_PYTHON=<venv: encoder/requirements-bench-cuda.txt>/bin/python ai-dotnet/benchmarks/run-g5-matched.sh eg2-text cuda eg2-text-cuda.jsonl 3 6 10
STARLING_EMBEDDINGGEMMA2_IMAGES=<make_eg2_images.py output> BENCH_PYTHON=<venv: embeddinggemma2/requirements-cuda.txt>/bin/python \
  ai-dotnet/benchmarks/run-g5-matched.sh eg2-image cuda eg2-image-cuda.jsonl 3 6 10
STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS=<tensors dir> BENCH_PYTHON=<venv: embeddinggemma2/requirements-cuda.txt>/bin/python \
  ai-dotnet/benchmarks/run-g5-matched.sh eg2-video cuda eg2-video-cuda.jsonl 3 6 10
BENCH_PYTHON=<venv: encoder/requirements-bench-cuda.txt>/bin/python ai-dotnet/benchmarks/run-g5-matched.sh gliner2 cuda gliner2-cuda.jsonl 3 6 10
BENCH_PYTHON=<venv: bench/requirements-genai-cuda.txt>/bin/python ai-dotnet/benchmarks/run-g5-matched.sh qwen35 cuda qwen35-fp32x-cuda.jsonl 3 6 10
BENCH_PYTHON=<venv: bench/requirements-genai-cuda.txt>/bin/python ai-dotnet/benchmarks/run-g5-matched.sh qwen35-fp16x cuda qwen35-fp16x-cuda.jsonl 3 6 10
# then, per file (or all together):
model-pipeline bench validate <file> && model-pipeline bench g5 <file> --json g5.json
```

CUDA profiles run both sides like the CPU ones: `run-e5-matched.sh cuda ...` and `run-g5-matched.sh <model> cuda ...` (`qwen35-fp16x` for `ortgenai-fp16x-cuda`), with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<gpu>` and `BENCH_PYTHON` from the `-cuda` requirements. The script builds the `.Cuda` harness and puts the CUDA 13 and cuDNN 9 libraries of that environment's `nvidia-*` wheels on `LD_LIBRARY_PATH`, so both sides load the same CUDA libraries (`environment.native_libraries` records them). `BENCH_SIDES=python|csharp` runs one side; `BENCH_WARM_ARGS` passes extra warm options to both sides for development smokes (never for a qualifying run). The Python runners can also be run one command at a time; every runner takes `warm|cold --provider cpu|cuda --threads N --inter-threads 1 --out <file>` plus `--snapshot` (or its `STARLING_*_SNAPSHOT` variable), `--only`, `--samples-scale` and `--processes` as the C# harnesses.

## Gaps

- **Python same-backend comparison**: runners exist for every budgeted profile. They have not run on a controlled machine yet, so no profile has G5 evidence. GLiNER2's Python side covers the inference stage only (the budget's scope); prepare and decode stay C#-only. Qwen's Python TTFT uses Hugging Face `tokenizers` for the incremental detokenization (C#: the owned tokenizer).
- **Video** (EmbeddingGemma 2, `video-fp32-{cpu,cuda}` budgets): both sides run on the same decoded frames, so video decoding (FFmpeg adapter against torchcodec) is outside the comparison and has no performance evidence. Rows with a still image (`mm-image-and-video`) and text-only rows are not in the video workloads.
- **CUDA**: every CUDA profile has a matched C#/Python pair (the `.Cuda` builds); none has run on a controlled machine (M6), so no CUDA profile has G5 evidence.
- **Resources**: allocations, GC pauses, native memory and VRAM are not recorded by the workload harnesses (G4's soak covers memory trends); no latency-bound concurrency sweep beyond the e5 harness's concurrency 4.
- **Qwen TTFT** is the time from the call to the first streamed update, so a first token held back by the incremental detokenizer (an incomplete UTF-8 sequence) is counted at its emission.
