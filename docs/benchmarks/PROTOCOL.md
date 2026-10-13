# Benchmark protocol (G5)

How the repository measures performance for gate G5 ("controlled benchmark with raw results, environment manifest, and stage timing"; plan section 10) and for the benchmark program of plan section 11. It covers the harnesses under `ai-dotnet/benchmarks/`: the e5 matched-backend harness (P-008, details in `ai-dotnet/benchmarks/README.md`) and the model harnesses for GLiNER2, EmbeddingGemma 2 and Qwen3.5.

**Budgets are maintainer decisions.** A harness measures and checks quality; it never decides whether a profile is fast enough. The G5 performance budget of each profile is chosen by the maintainer (production-readiness plan, action M2) and recorded in the manifest's gate entry with the evidence link. Until a budget exists, a G5 entry cannot pass, whatever the numbers. The figures in plan section 11 (p95 adapter overhead no more than the larger of 1 ms or 10%; at least 95% of matched native-backend decode throughput; alert on a repeatable regression above 5%) are proposals to calibrate, not budgets.

Development runs of each harness (uncontrolled, not G5 evidence): [GLiNER2](gliner2-2026-10-13.md), [EmbeddingGemma 2](embeddinggemma2-2026-10-13.md), [Qwen3.5](qwen35-2026-10-13.md).

## Harnesses

| Harness | Profile(s) | Paths and workloads (`ai-dotnet/benchmarks/workloads/`) | Stages timed | Quality checked on every series |
| --- | --- | --- | --- | --- |
| `Starlings.Benchmarks.E5Harness` (+ `.Cuda`, Python `bench_e5_matched.py`) | e5 `onnx-fp32-cpu`, `onnx-fp32-cuda` | `e5-v1.json`: ids and text, short and 512-token inputs, batch 1/8/32, concurrency 1 and 4 | library: `tokenize`, `encode`; direct and Python: `tokenize`, `prep`, `run`, `pool` | embeddings against the protected rows, max abs within the profile tolerance |
| `Starlings.Benchmarks.Gliner2Harness` | GLiNER2 `onnx-fp32-cpu` | `gliner2-v1.json`: short texts with 1, 3, 5 and 12 labels; batch 8; first 100/250/400 words of a long fixture text | `prepare` (normalization, word split, tokenization, schema and index tensors), `inference` (padding and the ORT run), `decode` (spans to entities) | entities equal to the protected reference under the protected order and threshold-flip rules, confidence within `float.confidence_max_abs`, logits finite |
| `Starlings.Benchmarks.EmbeddingGemma2Harness` | EmbeddingGemma 2 `onnx-fp32-cpu` (text), `image-fp32-cpu` | `embeddinggemma2-v1.json`: text at about 10–50 tokens (batch 1/8/16), first 100 and 400 words of a long text, 1,570–1,750 tokens (batch 1/4); images: photos, 1×1 and 8×6, 4000×3000, batch 4, two images interleaved with text, preprocessing alone | text: `tokenize`, `encode` (padding, ORT run, pooling); image: `decode` (SkiaSharp adapter from bytes in memory), `embed` (preprocessing, vision graph, multimodal graph: one public call); `preprocess-only`: `decode`, `preprocess` | embeddings against the protected rows (`float.embedding_max_abs`); preprocessing: pixel-values SHA-256 equal to the fixture digest |
| `Starlings.Benchmarks.Qwen35Harness` | Qwen3.5 `ortgenai-fp32x-cpu` | `qwen35-v1.json`: five short prompts with 64 new tokens (ids and text path), the tools prompt (286 tokens, 32 new), the 1,666-token prompt with 8 new tokens | `template` (chat template and tokenizer; text path only), `prefill`, `decode`; per-sample metrics `ttft_ms`, `prompt_tokens`, `completion_tokens`, `prefill_tokens_per_s`, `decode_tokens_per_s`, `inter_token_ms` | greedy ids equal to the protected reference output up to `max_new_tokens` (exact; a near-tie divergence also fails and reports its reference margin), prompt ids equal to the reference |

Workloads that have no upstream reference (GLiNER2 and EmbeddingGemma 2 `synthetic` lengths) are checked against the same input processed alone at load in the same process. That catches a batch, padding or state defect, not a fidelity defect; the fidelity evidence for those lengths is the gate G1/G2 work.

Not covered: EmbeddingGemma 2 video (the video fixtures need FFmpeg to decode, and no decoded-frame fixtures exist); CUDA builds of the model harnesses (the e5 harness has one); Python same-backend runners for GLiNER2 and EmbeddingGemma 2 (see "Gaps").

## What every run records

The raw format is `starlings-bench-raw/1` (`model-pipeline/schemas/benchmark-raw.schema.json`), JSON lines, one `run` record per process and one `series` record per timed series. Summaries are computed from the raw records (`model-pipeline bench summarize`), never stored instead of them.

- **Raw per-sample results.** Each series holds every sample's total (`ms`), every sample's stage times (`stages_ms`) and, for generation, every sample's metrics (`metrics`). No pre-aggregation.
- **Environment manifest** (`run.environment`): CPU model and ISA flags, physical and logical cores, RAM, OS, kernel, .NET runtime and GC settings, ONNX Runtime version and providers, session providers, loaded native libraries, CPU governor and turbo state, load average and GPU utilization before and after, and a free-text machine note (`--note`).
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

## Gaps

- **Python same-backend comparison** (plan 11, second question) exists for e5 only. GLiNER2: the Python preprocessing is the upstream `gliner2` package on PyTorch, so a same-backend Python runner needs that environment plus a NumPy port of the decode; not straightforward from `model-pipeline/reference/gliner2/`. EmbeddingGemma 2 text: feasible (`tokenizers` + ONNX Runtime + NumPy pooling, as for e5), but `bench_e5_matched.py` is e5-specific and needs to be factored into a shared runner first.
- **Qwen native comparison**: no same-backend runner (ONNX Runtime GenAI C API or Python `onnxruntime-genai`) yet, so the plan's decode-throughput objective (at least 95% of matched native decode throughput) cannot be evaluated.
- **Video** (EmbeddingGemma 2): needs decoded-frame fixtures that do not depend on FFmpeg, or a run with the FFmpeg adapter.
- **CUDA**: no CUDA build of the model harnesses; GPU runs need M6.
- **Resources**: allocations, GC pauses, native memory and VRAM are not recorded by the workload harnesses (G4's soak covers memory trends); no latency-bound concurrency sweep beyond the e5 harness's concurrency 4.
- **Qwen TTFT** is the time from the call to the first streamed update, so a first token held back by the incremental detokenizer (an incomplete UTF-8 sequence) is counted at its emission.
