# Benchmarks (P-008)

The matched-backend benchmark program of plan section 11 for `multilingual-e5-small`. It answers the second question of that section: is the .NET integration efficient, measured as C# against Python on the same backend, artifact, provider, settings, hardware and workload. Results are evidence only together with their raw files, environment and quality.

| Project | What | In `AI.slnx` (CI builds, never runs) |
| --- | --- | --- |
| `Starlings.Benchmarks.Micro` | BenchmarkDotNet micro-benchmarks: owned tokenizer encode (per short text, 512-token text, batch of 44), e5 embedding from ids and from text per workload. `[MemoryDiagnoser]`. CPU. | yes |
| `Starlings.Benchmarks.Micro.Cuda` | Same sources with `Microsoft.ML.OnnxRuntime.Gpu.Linux`; CUDA cases only with `STARLING_BENCH_CUDA=1` and an explicit `CUDA_VISIBLE_DEVICES`. | no (local) |
| `Starlings.Benchmarks.E5Harness` | Realistic-workload harness: warm series with stage timings and concurrency, cold starts in fresh processes, quality per run. Writes `starlings-bench-raw/1`. CPU. | yes |
| `Starlings.Benchmarks.E5Harness.Cuda` | Same sources, CUDA native package. | no (local) |
| `../../model-pipeline/reference/encoder/bench_e5_matched.py` | The Python side: ONNX Runtime 1.30.0, same model file, thread settings, workload file and fixtures; same raw format. Pinned environments: `requirements-bench.txt` (CPU), `requirements-bench-cuda.txt` (CUDA, same CUDA/cuDNN wheels as the qualified CUDA profile). | n/a |
| `workloads/e5-v1.json` | The shared workload file: requests as protected fixture case ids, sample counts per provider, warmup, the cold-start request. | n/a |
| `run-e5-matched.sh` | Runs the whole comparison, interleaving C# library, C# direct and Python over several rounds of fresh processes, then cold starts. | n/a |

Summaries: `model-pipeline bench validate <raw.jsonl>` and `model-pipeline bench summarize <raw.jsonl> [--json summary.json]` (`model-pipeline/src/model_pipeline/benchmarks.py`, schema `model-pipeline/schemas/benchmark-raw.schema.json`).

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

## Not covered

Allocation and GC counts are measured only by the micro-benchmarks, not by the workload harness; VRAM and native memory are not recorded; no latency-bound concurrency sweep beyond 1 and 4; Windows, macOS and ARM64 are untested; TensorRT, fp16 and TF32 are outside the profile.
