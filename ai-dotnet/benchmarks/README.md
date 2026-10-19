# Benchmarks (P-008)

The matched-backend benchmark program of plan section 11 for `multilingual-e5-small`, and the G5 harnesses for GLiNER2, EmbeddingGemma 2 and Qwen3.5 (protocol, recorded fields and gaps: [`docs/benchmarks/PROTOCOL.md`](../../docs/benchmarks/PROTOCOL.md); the sections below describe the e5 comparison). It answers the second question of that section: is the .NET integration efficient, measured as C# against Python on the same backend, artifact, provider, settings, hardware and workload. Results are evidence only together with their raw files, environment and quality.

| Project | What | In `AI.slnx` (CI builds, never runs) |
| --- | --- | --- |
| `Starlings.Benchmarks.Micro` | BenchmarkDotNet micro-benchmarks: owned tokenizer encode (per short text, 512-token text, batch of 44), e5 embedding from ids and from text per workload. `[MemoryDiagnoser]`. CPU. | yes |
| `Starlings.Benchmarks.Micro.Cuda` | Same sources with `Microsoft.ML.OnnxRuntime.Gpu.Linux`; CUDA cases only with `STARLING_BENCH_CUDA=1` and an explicit `CUDA_VISIBLE_DEVICES`. | no (local) |
| `Starlings.Benchmarks.E5Harness` | Realistic-workload harness: warm series with stage timings and concurrency, cold starts in fresh processes, quality per run. Writes `starlings-bench-raw/1`. CPU. | yes |
| `Starlings.Benchmarks.E5Harness.Cuda` | Same sources, CUDA native package. | no (local) |
| `../../model-pipeline/reference/encoder/bench_e5_matched.py` | The Python side (on the shared Python core `model-pipeline/reference/bench/matched_core.py`): ONNX Runtime 1.30.0, same model file, thread settings, workload file and fixtures; same raw format. Pinned environments: `requirements-bench.txt` (CPU), `requirements-bench-cuda.txt` (CUDA, same CUDA/cuDNN wheels as the qualified CUDA profile). | n/a |
| `workloads/e5-v1.json` | The shared workload file: requests as protected fixture case ids, sample counts per provider, warmup, the cold-start request. | n/a |
| `run-e5-matched.sh` | Runs the whole comparison, interleaving C# library, C# direct and Python over several rounds of fresh processes, then cold starts. | n/a |
| `Starlings.Benchmarks.Gliner2Harness` | G5 harness for GLiNER2 entity extraction: stages `prepare`, `inference`, `decode`; entities checked against the protected reference on every series. `dump-inputs --out <file>` writes the prepared model inputs of every workload request for the Python inference-stage runner. CPU. | yes |
| `Starlings.Benchmarks.Gliner2Harness.Cuda` | Same sources, CUDA native package (`--provider cuda`). | no (local) |
| `Starlings.Benchmarks.EmbeddingGemma2Harness` | G5 harness for EmbeddingGemma 2 text, image and video (video on decoded frames: the reference's sampled frames, `STARLING_EMBEDDINGGEMMA2_VIDEO_TENSORS`; no decoder). CPU. | yes |
| `Starlings.Benchmarks.EmbeddingGemma2Harness.Cuda` | Same sources, CUDA native package (`--provider cuda`). | no (local) |
| `Starlings.Benchmarks.Qwen35Harness` | G5 harness for Qwen3.5 greedy generation: prefill, decode tokens/s, time to first token, greedy ids checked against the reference under the profile's rule. CPU (fp32x). | yes |
| `Starlings.Benchmarks.Qwen35Harness.Cuda` | Same sources, GenAI CUDA natives (`--provider cuda`, `--variant fp32x|fp16x`). | no (local) |
| `Core/` | Shared harness core: timing loop, cold-start spawning, environment manifest, raw writer, quality gate, and the warm/cold/cold-child host the three G5 harnesses run on (`HarnessHost`, one `IBenchDriver` per model). The e5 harness uses the same timing, spawning, environment and writer code. | n/a |
| `Shared/` | e5 harness support (`BenchSupport.cs`): protected fixture cases and the e5 workload file, compiled into the e5 harnesses and the micro-benchmarks. | n/a |
| `Starlings.Benchmarks.Tests` | xunit tests of the shared core (`QualityGate`, `HarnessCore`, `NativeLibraryDigests`, workload parsing). | yes (a test project: CI runs it) |
| `workloads/{gliner2,embeddinggemma2,qwen35}-v1.json` | The G5 workload files (requests as protected fixture ids). | n/a |
| `../../model-pipeline/reference/bench/` | The Python same-backend runners of the G5 harnesses (`bench_eg2_matched.py`, `bench_gliner2_matched.py`, `bench_qwen35_matched.py`) and their shared core; same graphs, workload files, thread settings, warm/cold protocol and raw format. | n/a |
| `run-g5-matched.sh` | Runs one model's comparison (`eg2-text`, `eg2-image`, `eg2-video`, `gliner2`, `qwen35`, `qwen35-fp16x`) on `cpu` or `cuda` (the `.Cuda` builds), interleaving the C# harness and the Python runner over rounds of fresh processes, then cold starts. | n/a |
| `sweep` command of the e5, GLiNER2 and EmbeddingGemma 2 harnesses (+ `.Cuda`), `workloads/batching-sweep-v1.json`, `run-batching-sweep.sh`, `../../tools/ci/runpod-batching-bench.sh` | Batching sweep (Phase 0 of the batching investigation): batch size x concurrency x workload set, intra-op thread sweep, padding efficiency, peak RSS and device memory, every output against the same item run alone. Writes `starlings-bench-sweep/1`. See "Batching sweep" below. | harnesses yes; scripts n/a |

Summaries: `model-pipeline bench validate <raw.jsonl>`, `model-pipeline bench summarize <raw.jsonl> [--json summary.json]` and the G5 budget check `model-pipeline bench g5 <raw.jsonl>...` (`g5.py`) (`model-pipeline/src/model_pipeline/benchmarks.py`, schema `model-pipeline/schemas/benchmark-raw.schema.json`).

## What is timed

Implementations, all on one session per process:

- **C# library** (`--impl library`): the product API. Text path: `E5Tokenizer.Tokenize` per text (stage `tokenize`), then `E5Encoder.EmbedBatch` (stage `encode`: right padding into pooled buffers, ORT run, mean pooling, normalization). Ids path: `EmbedBatch` only.
- **C# direct** (`--impl direct`): the identical backend workflow without the `Starlings.Runtime.Onnx` and encoder layers: a raw `InferenceSession` with the same session options, the same padding into pooled buffers bound as `OrtValue`s, a preallocated host output and the same double-accumulation pooling. Stages `tokenize`, `prep`, `run`, `pool`. The plan's adapter-overhead budget (p95 overhead no greater than the larger of 1 ms or 10%) is measured as library against direct.
- **Python** (`python-ort`): `tokenizers.Tokenizer.encode` per text (sequential, like the C# tokenizer; `encode_batch` would use Rust threads), NumPy padding, `InferenceSession.run` with NumPy feeds and a NumPy output, NumPy float32 pooling. Stages `tokenize`, `prep`, `run`, `pool`.

Both languages: `intra_op_num_threads` as given (6 = physical cores of the reference workstation), `inter_op_num_threads=1`, sequential execution, all graph optimizations, right padding with id 1, `token_type_ids` zeros. CUDA: `device_id=0` after `CUDA_VISIBLE_DEVICES`, `use_tf32=0` (the qualified profile). The output is copied to host memory inside the timed `run` call in both languages, so GPU synchronization is inside every sample. The C# CUDA harness loads the CUDA and cuDNN libraries of the Python environment's `nvidia-*` wheels (`run-e5-matched.sh` sets `LD_LIBRARY_PATH`); each run records the native libraries actually mapped (`environment.native_libraries`).

Warm series: per workload, path (`ids`, `text`) and concurrency level (1 and 4 for batches 1 and 8; 1 for batch 32): `max(10, requests)` warmup calls, then `n` timed calls cycling through the workload's requests (worker k of a concurrent series starts at request k). Every sample records its total and its stages. After the timed loop every request is embedded once more and compared with the protected embeddings (`series.quality`).

Cold starts: the parent spawns a fresh process per sample and passes its spawn time (CLOCK_MONOTONIC). The child records `spawn_to_main`, Python's `runtime_init` (imports), `tokenizer_load`, `session_create` (model load), the first request's stages, and `spawn_to_result`; the parent adds `process_wall` (spawn to exit). The model file is in the page cache (warm OS cache, cold process). The model is not hashed at load in either language (`--hash-model true` measures the product's `ExpectedModelSha256` check, about 1 s for 470 MB).

Quality: every warm process first embeds the 46 protected cases from ids and from text, and in batches of 16, and compares them with `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/embeddings.jsonl` against the profile's `tolerances.yaml` (read only). Every timed series is checked the same way after its timed loop. The run passes only if the preliminary sweep and every series pass (`quality.preliminary_pass`, `series_pass`, `failed_series`; each series carries `quality.pass`); otherwise it exits non-zero and records `quality.pass = false`. `model-pipeline bench summarize` excludes failed series, and every series of a failed run, from all timing tables and lists them under "Quality failures".

## Raw result format `starlings-bench-raw/1`

JSON lines. A `run` record per process: `harness` (language, impl), `artifacts` (model and `tokenizer.json` SHA-256, workload and fixture SHA-256, git commit and dirty flag, package versions), `environment` (CPU model and ISA flags, physical and logical cores, RAM, OS, kernel, runtime, ORT version and providers, session providers, GPU name, PCI bus, driver, CUDA driver API version, loaded native libraries, governor, turbo, load average and all GPUs' utilization before and after, a machine note), `settings` (threads, execution mode, optimization level, provider options), `quality`. `series` records: mode (`warm`/`cold`), path, workload, batch, concurrency, warmup calls, `n`, `wall_ms`, raw per-sample `ms`, raw per-sample `stages_ms`, `quality`.

## Statistics

`model-pipeline bench summarize` computes, per group (provider, mode, path, workload, concurrency, implementation): median with a 95% hierarchical bootstrap interval (resample runs, then samples within each run, so run-to-run variation widens the interval), p5 and p95 only with at least 20 samples, p99 only with at least 1,000 samples (otherwise `n/a`), items per second from wall time, stage medians, worst quality. It pairs C# library and C# direct with Python (ratio of medians and difference, bootstrap intervals) and checks the adapter budget on p95.

## Running

```sh
export DOTNET_ROOT=~/.dotnet PATH=~/.dotnet:$PATH
export STARLING_E5_SNAPSHOT=/path/to/snapshot        # or the acquisition cache (ai-dotnet/tools/fetch-e5-model.sh)
uv venv -p 3.11 .venv-bench && uv pip install --python .venv-bench/bin/python --require-hashes \
    -r model-pipeline/reference/encoder/requirements-bench.txt

BENCH_PYTHON=.venv-bench/bin/python BENCH_NOTE="machine description" \
    ai-dotnet/benchmarks/run-e5-matched.sh cpu raw-cpu.jsonl 3 6 10
# CUDA: a Python environment from requirements-bench-cuda.txt, and an explicit device (never implicit):
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 BENCH_PYTHON=.venv-bench-cuda/bin/python \
    ai-dotnet/benchmarks/run-e5-matched.sh cuda raw-cuda.jsonl 3 6 10
model-pipeline bench summarize raw-cpu.jsonl raw-cuda.jsonl

# Micro-benchmarks (BenchmarkDotNet; results under BenchmarkDotNet.Artifacts/)
dotnet run -c Release --project ai-dotnet/benchmarks/Starlings.Benchmarks.Micro -- --filter '*'
```

Check that the machine is idle first and record what else shares it (`BENCH_NOTE`); every run records load average and GPU utilization before and after. Do not compare numbers across machines or across raw files with different `workload_sha256`, model SHA-256 or package versions. A failing comparison is not hidden by changing workloads or dropping a configuration (plan section 11).

## Batching sweep

Phase 0 of the batching investigation: measure before changing any default. Every harness with a sweep driver (e5, GLiNER2, EmbeddingGemma 2 text) has a `sweep` command; `sweep --help` lists its options.

```sh
H=ai-dotnet/benchmarks/Starlings.Benchmarks.E5Harness/bin/Release/net10.0/Starlings.Benchmarks.E5Harness   # likewise Gliner2Harness, EmbeddingGemma2Harness
$H sweep --threads 6 --out sweep.jsonl                         # full grid: sets short,long,mixed; batches 1,4,8,16,32,64; concurrency 1,4,16
$H sweep --threads 6 --out quick.jsonl --quick true            # batches 1,8,32; concurrency 1,4; 1 s per cell; pools of 32
model-pipeline bench sweep sweep.jsonl --out summary.md         # one table per model x provider x configuration
ai-dotnet/benchmarks/run-batching-sweep.sh cpu out/ [--quick]   # every model, then the thread sweep and the library defaults
```

- **Cells.** For each workload set, batch size b and concurrency c: c threads issue calls of b items each through the product API (tokenize, then one `EmbedBatch`, or GLiNER2 `Prepare`, `RunModel`, `Decode`), after `--warmup` untimed calls, until at least `--min-calls` calls and `--min-seconds` have passed (`--max-calls` caps it). Call i always carries the same items, whichever thread runs it. Concurrency 16 with batch 1 is the baseline a cross-request micro-batcher has to beat.
- **Rows per ORT run.** By default the library's `MaxBatchSize` is set to the largest swept batch (`--max-batch-size call`), so a call of b items is one ORT run and the sweep measures the batch size itself. `--max-batch-size default` keeps the library default (e5 32, GLiNER2 8, EmbeddingGemma 2 16 rows and 8,192 cells) to measure what callers get today; `--max-batch-size N` sets it. With micro-batching, `call` covers a whole dispatched batch (the larger of the swept batch and the micro-batch rows).
- **Workloads** (`workloads/batching-sweep-v1.json`, built at load from text in the repository, seeded, nothing downloaded). Embedding models: `short` = MIRACL queries in 8 languages (`release-shared/retrieval/miracl-dev-hn-8lang-v1`), `medium` = single Wikipedia passages, `long` = consecutive same-language passages joined up to 512 model tokens (at least 75% of it), each with the model's query or passage prefix or prompt. GLiNER2: WNUT-17 test sentences with their six labels (one label set per call): `short` one sentence, `medium` 2 to 4 consecutive sentences, `long` consecutive sentences up to 480 tokens. `mixed` (the realistic distribution) draws each item's category with probability 0.40 short, 0.45 medium, 0.15 long. The run record lists each pool's size and token percentiles.
- **Recorded per call:** latency, start offset, worker, stage times (`tokenize`/`encode`; GLiNER2 `prepare`/`inference`/`decode`), real tokens, padded cells (rows x longest row per ORT run, following the library's chunk plan) and runs. Per cell: items/s and calls/s over the cell's wall time, p50/p95, padding efficiency (real tokens / padded cells), peak RSS (`VmHWM`, reset through `/proc/self/clear_refs` before the cell), on CUDA the device's peak memory (`nvidia-smi -lms 100`, whole GPU), and the quality check. Without micro-batching a call runs on its caller's thread, so there is no queue time; GLiNER2's `inference` on CUDA includes the wait on its run gate. With micro-batching the cell's `micro_batching` record holds the batcher's own counters over the timed calls: dispatched batches, mean rows per batch, queue time per item and run time per batch, and the cell's padding efficiency is that of the dispatched batches.
- **Quality on every cell:** each output of every timed call is compared with the same item run alone at load, within the profile's protected padded-batch tolerance (e5 and EmbeddingGemma 2 `float.padded_batch_vs_single_embedding_max_abs`; GLiNER2 `float.padded_batch_vs_single_span_logits_relative` on the valid span logits, and identical entities). The fixture sweep of the harness runs first (`--preliminary false` skips it in thread-sweep runs). A failed cell fails the run (exit 1); the summary marks it and never ranks it.
- **Format** `starlings-bench-sweep/1` (`model-pipeline/schemas/benchmark-sweep.schema.json`), a sibling of `starlings-bench-raw/1`: the `run` record has the same artifacts, environment and settings blocks plus the sweep configuration; one `cell` record per cell with the raw per-call arrays, written as the cell finishes; an `end` record (a run without one did not finish). `--max-inflight-items N` skips cells with b x c above N (listed in the end record), a memory guard for CPU machines.
- **Threads.** ORT intra-op threads are fixed per process, so the thread sweep is separate processes (`run-batching-sweep.sh`: `SWEEP_THREADS` for the main grid, `SWEEP_THREAD_LIST` for the reduced grid on `mixed`); the summary adds an items/s table by thread count.
- **Batching configurations** (the option contract; every run records its label as `sweep.config`, and the summary keeps one table per configuration plus a comparison table): `--sort-by-length true` sets the library's `SortByLength` (length-sorted chunking within one call, results scattered back), `--max-batch-tokens N` its `MaxBatchTokens` (padded cells per ORT run; e5, GLiNER2 and EmbeddingGemma 2), and any of `--microbatch-wait-ms MS`, `--microbatch-max-rows N`, `--microbatch-max-tokens N` turns on cross-request micro-batching (`MicroBatchingOptions` MaxWait, MaxRows, default 8, and MaxTokens). A micro-batched run issues each item as its own call through the library's micro-batched entry point (e5 and EmbeddingGemma 2 `IEmbeddingGenerator.GenerateAsync` with one value, GLiNER2 `ExtractAsync`), so with `--batches 1` it is c single-item callers whose requests the library batches; compare it with the same grid without micro-batching (plain concurrency). GLiNER2's public `ExtractAsync` returns entities only, so its micro-batched quality check is identical entities and confidence within `float.confidence_max_abs`. `run-batching-sweep.sh` runs `SWEEP_CONFIGS` after the baseline (default `sorted tokens sorted-tokens micro`; `--quick`: `sorted tokens micro`). Sorting only changes anything when a call is split into several ORT runs, so `sorted` runs calls of 16 to 64 items at `--max-batch-size 8` next to the same grid unsorted (`mbs8`); `tokens` and `sorted-tokens` keep one run per call apart from the budget's own splits. On CPU, GLiNER2's in-flight guard defaults to 32 items (`SWEEP_MAX_INFLIGHT_GLINER2`): its long b32 c1 cell peaked at 10.2 GB RSS, and b32 c4 was killed for memory on the development machine. Micro-batching is Experimental and is adopted only if it beats plain concurrency 4 and 16.
- **On a GPU pod:** `tools/ci/runpod-batching-bench.sh <sha> <out> [--quick]` (RTX 3090 only) checks out the commit from origin, builds the CPU and `.Cuda` harnesses, re-verifies the models, runs the sweep on CUDA and then on CPU with thread counts sized to the container's CPU quota, and writes `status.txt`, `environment.txt`, the raw files, `summary.md`, `exit-code` and `DONE`.

Results from a shared or loaded machine are development numbers only; record the load in `SWEEP_NOTE` (every run also records its load average before and after).

## Not covered

Allocation and GC counts are measured only by the micro-benchmarks, not by the workload harness; VRAM and native memory are not recorded by the matched harnesses (the batching sweep records peak RSS and device memory); no latency-bound concurrency sweep beyond 1 and 4 in the matched comparison (the batching sweep goes to 16 callers); Windows, macOS and ARM64 are untested; TensorRT, fp16 and TF32 are outside the profile.
