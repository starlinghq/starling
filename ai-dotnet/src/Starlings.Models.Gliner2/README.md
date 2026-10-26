# Starlings.Models.Gliner2

English entity extraction with [`fastino/gliner2-base-v1`](https://huggingface.co/fastino/gliner2-base-v1) at revision `f9634218e53580c56edf0de97ca1a7d3f1c2354e` (GLiNER2 2.0.0, source `55656fbf`), on ONNX Runtime 1.30.0, in-process and offline after an explicit acquisition. You name the entity labels; the model returns labelled spans with offsets and confidences. The package is versioned 0.x, so the API may still change between minor versions; maturity is per profile (below).

- GLiNER2's text handling is ported with Python semantics (normalization, the regular-expression word splitter, `str.lower()`), and the schema prefix is encoded with the owned tokenizer `Starlings.Tokenizers` (profile `gliner2-base-v1-unigram`). No Microsoft.ML.Tokenizers, no native tokenizer, no Python.
- One ONNX graph (token ids and routing positions to count and span logits); GLiNER2's count gate, threshold, overlap suppression and result order in C#.
- The package ships the asset lock (sizes and SHA-256), not the 838 MB of model files. Acquisition is an explicit call; inference never downloads.

## Maturity

<!-- catalog:maturity profiles=gliner2-base-v1/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `gliner2-base-v1/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; linux-x64 cuda: supported; windows-x64 cpu: experimental |
| `gliner2-base-v1/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Install

```sh
dotnet add package Starlings.Models.Gliner2
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0          # CPU natives
# or, for CUDA on Linux x64: Microsoft.ML.OnnxRuntime.Gpu.Linux 1.30.0 (reference exactly one)
```

## Use

```csharp
using Starlings.Models.Gliner2;

await Gliner2ModelAssets.DownloadAsync();                   // once, explicit, needs the network; verifies SHA-256
Gliner2ModelFiles files = Gliner2ModelAssets.OpenLocal();   // offline, verified, never downloads
using var extractor = Gliner2EntityExtractor.Create(files);

var result = extractor.Extract(
    "Apple CEO Tim Cook announced the iPhone 15 in Cupertino.",
    ["company", "person", "product", "location"],
    threshold: 0.5f);
foreach (var e in result.Entities)
{
    Console.WriteLine($"{e.Label}: {e.Text} [{e.Start}, {e.End}) {e.Confidence:F3}");
}
```

## Batching options

`ExtractBatch` and `RunModel` split a batch (one label set) into ONNX Runtime calls, each right-padded to its longest text. `Gliner2ExtractorOptions` controls the split:

| Option | Default | Effect |
| --- | --- | --- |
| `MaxBatchSize` | 8 | Texts per call at most (GLiNER2's default `batch_size`). Chunks are consecutive texts in input order. |
| `MaxBatchTokens` | `null` (no budget) | Padded tokens (texts x longest text, schema prefix included) per call at most. A chunk closes before it would exceed the budget; a text longer than the budget runs alone. Experimental (`batch.token_budget`). |
| `SortByLength` | `false` | Chunks texts in ascending token count (ties in input order) so each call pads less; results stay in input order. Experimental (`batch.sort_by_length`). |

The defaults are the qualified behavior, with outputs unchanged bit for bit. With `SortByLength`, which texts share a call depends on the whole batch, so span logits can differ from the default by up to the profile's padded-batch-vs-single bound (`padded_batch_vs_single_span_logits_relative`); the CPU tests saw no difference and identical entities on the 54 in-contract fixture cases in random orders. On CUDA, calls stay serialized (`SerializeNativeCalls`); the options only change how one request is chunked.

## Stable contract

This is the contract of profile `gliner2-base-v1/onnx-fp32-cpu`. Its Stable label commits to it under the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md). The statuses are the registry manifest's and `Gliner2Capabilities.ProfileCapabilities`, which a test keeps equal key for key.

| Item | Contract |
| --- | --- |
| Profile | `gliner2-base-v1/onnx-fp32-cpu` (registry manifest `model-pipeline/registry/gliner2-base-v1/onnx-fp32-cpu/manifest.yaml`). |
| Model | `fastino/gliner2-base-v1` at `f9634218e53580c56edf0de97ca1a7d3f1c2354e`; GLiNER2 source `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`. The ONNX graph is Starling's reproducible export (`onnx/gliner2-entities-fp32.onnx`, digest in `Gliner2ModelAssets`), downloaded with no token from the public repository `Starlinghq/gliner2-base-v1-onnx` at `c02c3350`. |
| Tokenizer | `Starlings.Tokenizers` `0.3.2`, profile `gliner2-base-v1-unigram` (fast-tokenizer semantics; byte-fallback pieces map to `[UNK]` as GLiNER2 does). |
| Runtime | ONNX Runtime 1.30.0 (the application adds `Microsoft.ML.OnnxRuntime` 1.30.0 CPU natives), fp32, .NET 10. |
| Supported platform | Linux x64, CPU, with the deployment setting below. Gates G1 to G5 ran on an Intel Core i7-5820K (Haswell-E, 6 cores and 12 threads, AVX2 and FMA, no AVX-512), .NET 10.0.12. The weekly qualification reruns G1 to G3 on the release sets and a short soak (with the setting) on GitHub-hosted `ubuntu-latest` x64. |
| Linux deployment requirement | Start the process with `MALLOC_MMAP_THRESHOLD_=65536`. The G4 qualification soak passed with it; see [Memory on Linux](#memory-on-linux). |
| Input | English text and an ordered set of entity labels, threshold (default 0.5). One label set per call (also per batch). |
| Output | Entities with label, text, confidence and offsets into the original text: `Start`/`End` in UTF-16 code units, and `ScalarStart`/`ScalarEnd` in Unicode scalar values. `Text == input[Start..End]` always. Order: labels in request order, then descending confidence. Per label, confidence-first suppression of overlapping spans; spans of different labels may overlap. |
| Numerical bound | The protected tolerances of the profile (span logits relative 1e-4; order and threshold ties within 2e-5 of the reference, per the maintainer decision of 2026-10-10). Release set: 1.78e-5. |
| Task quality | WNUT-17 test, 300 sentences: exact micro F1 at least 0.352 and partial micro F1 at least 0.403 (budget); measured 0.3626 and 0.4134, equal to the PyTorch reference. |
| Supported | Entity extraction with labels and threshold, the staged API, original-text offsets, batch, cancellation, concurrent calls, offline asset resolution and explicit download, Native AOT (CPU natives). |
| Limits | At most 512 tokens, schema prefix included; longer texts throw `Gliner2InputTooLongException`, and nothing is truncated. Spans up to 8 words. The model files (838 MB) are acquired explicitly; inference never downloads. |
| Outside the Stable contract | Label descriptions and the closed count gate: experimental (proposed capability cases, not yet accepted fixtures). CUDA is the separate profile `gliner2-base-v1/onnx-fp32-cuda`, with its own contract ([Stable contract: CUDA](#stable-contract-cuda)). Windows x64 CPU: experimental (CI builds, runs the weight-free tests and the clean-install drill without the model; no model run). macOS and ARM64: unsupported (no evidence). |
| Unsupported | inputs over 512 tokens, unserialized concurrent calls on CUDA (wrong entities in 6 of 8 test runs), truncation, the `max_len` word cap, `extract_long`, GLiNER2's plain de-duplicated output, mixed label sets in one batch, classification, relation extraction, structured (JSON) extraction, per-label thresholds, overlap policies, the character splitter, non-English text, TF32, fp16, quantized graphs, streaming, persistence, training, Microsoft.Extensions.AI, ML.NET (no adapter exists). |
| Known issues | Most entity scores saturate to exactly 1.0 in fp32, so near-ties make the order sensitive to backend noise. Upstream span offsets index the normalized text: a span that includes the appended `.` ends one past the original text (seen at threshold 0.0); the package maps such spans onto the input. With default glibc settings, memory retained by the allocator trips the soak's growth rule (below). |

## Stable contract: CUDA

This is the contract of profile `gliner2-base-v1/onnx-fp32-cuda` (Stable since 2026-10-16). It is the CPU contract above, with these differences. The statuses are the registry manifest's and `Gliner2Capabilities.ProfileCapabilities`, which a test keeps equal key for key.

| Item | Contract |
| --- | --- |
| Profile | `gliner2-base-v1/onnx-fp32-cuda` (registry manifest `model-pipeline/registry/gliner2-base-v1/onnx-fp32-cuda/manifest.yaml`). The same model, graph and tokenizer as the CPU profile. |
| Runtime | ONNX Runtime 1.30.0 with the CUDA execution provider (the application adds `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 instead of the CPU natives), `use_tf32=0`, fp32. The CUDA 13 runtime and cuDNN 9 must be on the loader path; the package does not ship them. |
| Supported platform | Linux x64 with an **NVIDIA GeForce RTX 3090** (Ampere), driver 580 or later. G1 to G3 on the release sets, the binding G4 one-hour soak and G5 ran on RTX 3090s. **Regression:** the per-release GPU run (the qualification workflow's `cuda (runpod)` job, for every `ai-v*` tag and on demand) reruns the CUDA tests, G1 to G3 on the release sets and a soak smoke test, on an RTX 3090 only; the job refuses any other GPU. Other GPUs, including the RTX 4090 (Ada), are untested and outside the contract. |
| Linux deployment requirement | Start the process with `MALLOC_MMAP_THRESHOLD_=65536`, as on CPU ([Memory on Linux](#memory-on-linux)). The G4 soak and the per-release soak smoke test run with it. |
| Concurrency | Calls on one extractor are serialized on CUDA (`Gliner2ExtractorOptions.SerializeNativeCalls`, the CUDA default). Unserialized concurrent calls on one CUDA session are unsupported (wrong entities in 6 of 8 test runs). |
| Cancellation | A CUDA run is never terminated mid-graph: it completes, then the call throws `OperationCanceledException` (#61). Latency is one chunk, **about 50 ms** for 8 long texts; a 4,096-text batch cancelled after 50 ms threw at 52 to 53 ms. |
| Numerical bound | The CUDA profile's protected tolerances (span logits relative 1e-4; order and threshold ties within 2e-5). Release set: span logits 1.4e-5 relative, entities 30/30 from inputs and from text. |
| Task quality | WNUT-17: exact micro F1 0.3626, partial micro F1 0.4134 (budget 0.352 and 0.403). |
| Performance | G5 on an RTX 3090: inference 0.959 to 0.996 of the Python ONNX Runtime CUDA run; prepare and decode 0.6 to 1.5% of inference (budget 5%). |
| Supported | Entity extraction with labels and threshold, the staged API, original-text offsets, batch, cancellation, serialized concurrent calls, offline asset resolution and explicit download. |
| Outside the Stable contract | Native AOT with the CUDA natives: experimental (one local run, `tools/aot-check-ai.sh --cuda`, not in the per-release run). Label descriptions and the closed count gate: experimental, as on CPU. Windows x64 CUDA: untested. |

Evidence: [binding CUDA G4 and G5 round](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-16-cuda/README.md), [per-release GPU regression, run 38857860186](https://github.com/starlinghq/starling/blob/main/docs/evidence/gpu-regression-2026-10-16/README.md), [release-set gates](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/gliner2-onnx-fp32-cuda/README.md), [CUDA cancellation](https://github.com/starlinghq/starling/blob/main/docs/evidence/gliner2-cuda-wrong-output/README.md).

## Micro-batching (experimental, opt-in)

`Gliner2ExtractorOptions.MicroBatching` (null by default: off) applies to `ExtractAsync` only; `Extract`, `ExtractBatch` and the staged methods never queue. Without it, `ExtractAsync` runs on the calling thread like `Extract` and returns a completed task. With it, each call's text is prepared and length-checked on the calling thread (an invalid text fails only its own call), then queued; one dispatcher thread runs the queued texts of concurrent callers that share a schema (the same labels in the same order and the same descriptions) in padded batches of at most `MaxRows` texts and `MaxTokens` padded token cells, and each caller decodes its spans with its own threshold. `MaxBatchSize` still splits each batch into graph calls, and the CUDA run gate still applies, so on CUDA serialized single calls become serialized batched calls. Outputs then depend on the concurrent traffic: equal to a single call within the profile's padded-batch tolerance, not bit for bit reproducible. Cancelling a queued call removes it; cancelling a dispatched one cancels only that caller's task (the batch keeps running for the others). A full queue (`QueueCapacity`) makes callers wait for a slot. `Dispose` faults queued calls with `ObjectDisposedException` and waits for a running batch. `MicroBatchingStatistics` reports batches, items, padding and queue and run time. Status: experimental (capability `micro_batching`): weight-free tests and one real-model test within the padded-batch tolerance on Linux x64 CPU; no benchmark adoption decision, no CUDA run and no soak yet.

## Execution scheduler (experimental, opt-in)

`Gliner2ExtractorOptions.ExecutionScheduler` (a `TaskScheduler`, null by default) chooses where `ExtractAsync` executes. Null keeps it inline: on the calling thread, a completed task without micro-batching. `TaskScheduler.Default` offloads to the thread pool; an application's own scheduler (for example one thread per device) can serialize or place the work. With micro-batching on, only the preparation and queueing move to the scheduler; batches still run on the dispatcher thread. The synchronous methods ignore it, and the entities do not change. Status: experimental (capability `execution_scheduler`; weight-free tests).

## Memory on Linux

**Deployment requirement: set `MALLOC_MMAP_THRESHOLD_=65536` in the environment of a Linux process that runs this profile (CPU or CUDA).**

With default glibc settings, a process that loads and unloads the extractor repeatedly keeps freed native memory resident: about 1.2 to 1.5 GiB between loads, rising by about 65 to 90 MiB over the first half hour of cycling and then holding. This is allocator retention (glibc malloc's resident high-water mark and fragmentation), not a leak: with the threshold set, every unload returns what the load and its requests allocated. But the soak's growth rule, which is not changed for this, judges the default run: the one-hour default soak fit 81 to 82 MiB/h over the whole run and failed the rule, although the curve flattens after about 30 minutes. With `MALLOC_MMAP_THRESHOLD_=65536` the footprint after unloading held at about 76 MiB, the loaded footprint was about 740 MiB lower, and the one-hour G4 soak passed.

Evidence: [GLiNER2 soak growth](https://github.com/starlinghq/starling/blob/main/docs/evidence/gliner2-soak-growth/README.md) (diagnosis, six controlled runs), [G4 qualification soak](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-15/README.md).

## Shared types

A snapshot verified with `CacheVerification.Sha256` (the default) carries its digests: `Gliner2ModelFiles.ModelSha256` is the graph's SHA-256 then, and `Gliner2EntityExtractor.Create(files)` passes it to the session as `OnnxSessionOptions.KnownModelSha256`, so the graph is read once at startup, by the verification, not again by the session. With `CacheVerification.Size` it is null and the session hashes the graph as before.

The package uses the backend-neutral types of `Starlings.Models.Abstractions`. `Gliner2InputTooLongException` derives from `InputTooLongException` (`InputIndex`, `TokenCount`, `MaxTokens`). `Gliner2ModelAssets.Assets` is the lock and acquisition as an `IModelAssets<Gliner2ModelFiles>`. `Gliner2Capabilities.Statuses` is the package-wide capability table in `Starlings.Models.CapabilityStatus`, and `Gliner2Capabilities.ProfileCapabilities` holds each profile's capability record (`ModelProfileCapabilities`), generated from its registry manifest. `Gliner2Capabilities.All`, `Gliner2Capabilities.Profiles` and `Gliner2CapabilityStatus` are obsolete and are removed in the next minor version.

## Status

Profiles `gliner2-base-v1/onnx-fp32-cpu` (Linux x64 CPU) and `gliner2-base-v1/onnx-fp32-cuda` (Linux x64, RTX 3090, Stable since 2026-10-16) have passed every qualification gate, G0 to G7; their maturity is in the table above. GPU regression runs for every release and on demand, on an RTX 3090 only (the qualification workflow's `cuda (runpod)` job); CPU regression runs weekly. Evidence and reproduction: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#gliner2-entity-extraction-p-015) and the [catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md). License of the weights and the converted graph: Apache-2.0 (model card); GLiNER2 by Fastino AI, backbone initialised from `microsoft/deberta-v3-base` (MIT).

## Qualification evidence

- G0 provenance: the [2026-10-14 provenance check](https://github.com/starlinghq/starling/blob/main/docs/evidence/g0-provenance-2026-10-14/README.md).
- G1, G2 and G3 on the P-030 release sets: [release-set gates](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/gliner2-onnx-fp32-cpu/README.md).
- G4 one-hour soak with `MALLOC_MMAP_THRESHOLD_=65536` and G5 matched benchmark: [2026-10-15](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-15/README.md).
- G6 distribution and G7 integration: [G6](https://github.com/starlinghq/starling/blob/main/docs/evidence/g6-distribution-2026-10-14/README.md), [G7](https://github.com/starlinghq/starling/blob/main/docs/evidence/g7-integration-2026-10-14/README.md).
- CUDA: G4 and G5 in the [binding CUDA round](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-16-cuda/README.md); per-release GPU regression [run 38857860186](https://github.com/starlinghq/starling/blob/main/docs/evidence/gpu-regression-2026-10-16/README.md).
- Regression support: the weekly [qualification workflow](https://github.com/starlinghq/starling/blob/main/.github/workflows/qualification.yml) (jobs `release-gates` and `soak`), and its per-release `cuda (runpod)` job for CUDA.
