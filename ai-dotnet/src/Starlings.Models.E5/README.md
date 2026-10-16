# Starlings.Models.E5

Sentence embeddings from [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) at revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, on ONNX Runtime 1.30.0, in-process and offline after an explicit download. The package is versioned 0.x, so the API may still change between minor versions; maturity is per profile (below).

- Text goes through the owned tokenizer `Starlings.Tokenizers` (exact against huggingface/tokenizers on the published fixtures). No Microsoft.ML.Tokenizers and no native tokenizer library.
- Batching with right padding, mean pooling over the attention mask, L2 normalization: 384-dimensional unit vectors.
- The package ships the model's asset lock (repository, commit, file sizes and SHA-256), not the 487 MB of model files. Acquisition is an explicit call; inference never downloads.

## Maturity

<!-- catalog:maturity profiles=multilingual-e5-small/* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `multilingual-e5-small/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `multilingual-e5-small/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
<!-- /catalog:maturity -->

## Install

```sh
dotnet add package Starlings.Models.E5 --prerelease
dotnet add package Microsoft.ML.OnnxRuntime --version 1.30.0          # CPU natives
# or, for CUDA on Linux x64: Microsoft.ML.OnnxRuntime.Gpu.Linux 1.30.0 (reference exactly one)
```

The library references only `Microsoft.ML.OnnxRuntime.Managed` 1.30.0; the application chooses the native runtime.

## Use

```csharp
using Starlings.Models.E5;
using Starlings.Runtime.Onnx;

await E5ModelAssets.DownloadAsync();                 // once, explicit, needs the network; verifies SHA-256
E5ModelFiles files = E5ModelAssets.OpenLocal();      // offline, verified, never downloads

using var encoder = E5Encoder.Create(files.ModelPath, new E5EncoderOptions
{
    Session = new OnnxSessionOptions { ExpectedModelSha256 = E5ModelAssets.ModelSha256 },
});
var embedder = new E5TextEmbedder(E5Tokenizer.FromFile(files.TokenizerPath, E5ModelAssets.TokenizerSha256), encoder);

float[] query = embedder.Embed(E5TextEmbedder.QueryPrefix + "how much protein should a female eat");
float[][] passages = embedder.EmbedBatch([E5TextEmbedder.PassagePrefix + "As a general guideline, ..."]);
```

e5 expects `"query: "` or `"passage: "` at the start of every text; nothing is added for you. Vectors are normalized, so cosine similarity is the dot product.

## Status

Profile `multilingual-e5-small/onnx-fp32-cpu` (Linux x64 CPU) is Stable since 2026-10-15: it passed every qualification gate, G0 to G7, and the weekly qualification workflow reruns G1 to G3 on the release sets and a one-hour soak, green. The hosted soak that failed on 2026-10-15 (about 50 MB/h) did so because of glibc retention after unloading and transparent huge pages set to `always` on the runner image, not a leak ([evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/scheduled-regression-fix/README.md)). Profile `multilingual-e5-small/onnx-fp32-cuda` (Linux x64, NVIDIA GeForce RTX 3090) is Stable since 2026-10-16 ([Stable contract: CUDA](#stable-contract-cuda)): every gate passed, and the per-release GPU regression run was green for it. GPU regression runs for every release and on demand, on an RTX 3090 only (the qualification workflow's `cuda (runpod)` job); CPU regression runs weekly. Gate evidence: the [catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md). Model license: MIT, stated only in the model card.

## Stable contract

This is the contract of profile `multilingual-e5-small/onnx-fp32-cpu`. Its Stable label (since 2026-10-15) commits to it under the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md). The statuses are the registry manifest's and `E5Capabilities.Profiles`, which a test keeps equal key for key.

| Item | Contract |
| --- | --- |
| Profile | `multilingual-e5-small/onnx-fp32-cpu` (registry manifest `model-pipeline/registry/multilingual-e5-small/onnx-fp32-cpu/manifest.yaml`). |
| Model | `intfloat/multilingual-e5-small` at `614241f622f53c4eeff9890bdc4f31cfecc418b3`; `onnx/model.onnx` and `tokenizer.json` verified by SHA-256 (`E5ModelAssets.Lock`). |
| Tokenizer | `Starlings.Tokenizers` `0.3.0-preview.1`, profile `xlmr-unigram-e5` (pinned `tokenizer.json`, not SentencePiece or transformers' rebuilt tokenizer). |
| Runtime | ONNX Runtime 1.30.0 (`Microsoft.ML.OnnxRuntime.Managed`; the application adds `Microsoft.ML.OnnxRuntime` 1.30.0 CPU natives), fp32, .NET 10. |
| Supported platform | Linux x64, CPU. Gates G1 to G5 ran on an Intel Core i7-5820K (Haswell-E, 6 cores and 12 threads, AVX2 and FMA, no AVX-512), .NET 10.0.12. The weekly qualification reruns G1 to G3 on the release sets and a one-hour soak on GitHub-hosted `ubuntu-latest` x64 (4 logical CPUs; transparent huge pages `madvise`). No `MALLOC_*` environment setting is required. |
| Memory on Linux | The runtime calls `malloc_trim(0)` when a session is released, on by default; see [Memory on Linux](#memory-on-linux). |
| Input | Text, or token ids (`EncoderInput`). The caller adds `"query: "` or `"passage: "`; nothing is inserted. |
| Output | 384-dimensional float32 vectors: mean pooling over the attention mask, then L2 normalization. Batches are right-padded; batch rows equal single calls within the protected tolerances. |
| Numerical bound | The protected tolerances of the profile (embedding max abs 1e-5 against the PyTorch reference). Release set: 1.46e-7. |
| Task quality | MIRACL dev, 8 languages: nDCG@10 at least 0.792 and MRR@10 at least 0.765 (budget); measured 0.7978 and 0.7717. |
| Supported | Embedding from text and from token ids; batch; cancellation (at ONNX Runtime node boundaries on CPU, after the current run on CUDA); concurrent calls; explicit acquisition (download, verify, offline reuse); Native AOT for the typed API and the Extensions.AI generator; `Starlings.Models.E5.Extensions.AI` (`IEmbeddingGenerator`); `Starlings.Models.E5.MLNet` (pipeline composition, row-wise and look-ahead batching). |
| Limits | 512 tokens including `<s>` and `</s>`. `E5Tokenizer` truncates on the right to 512 (the profile's `truncate-right`, as the reference); `E5Encoder` rejects id sequences longer than `MaxSequenceLength` and never truncates them. The model files (487 MB) are acquired explicitly; inference never downloads. |
| Outside the Stable contract | Windows x64 CPU: experimental (CI builds, runs the weight-free tests and the clean-install drill without the model; no model run). CUDA is the separate profile `onnx-fp32-cuda` with its own contract ([Stable contract: CUDA](#stable-contract-cuda)). macOS, ARM64, Windows CUDA: unsupported (no evidence). |
| Unsupported | streaming, pair input, non-zero `token_type_ids`, encoder-side truncation, prefix insertion, TF32, fp16, quantized graphs, TensorRT, persistence (ML.NET save), training, ML.NET cancellation, the ML.NET adapter under Native AOT, Extensions.AI `Dimensions` truncation. |
| Known issues | On hosts with transparent huge pages set to `always`, resident memory can grow over time with no leak behind it ([Memory on Linux](#memory-on-linux)). The pinned `tokenizer.json` emits a lone `▁` token before special-token literals and for trailing whitespace, where SentencePiece and transformers 5.18 emit none; the profile follows `tokenizer.json`. It also leaves decomposed Hangul jamo uncomposed, while SentencePiece composes them. Both are recorded, not fixed. |

## Stable contract: CUDA

This is the contract of profile `multilingual-e5-small/onnx-fp32-cuda` (Stable since 2026-10-16). It is the CPU contract above, with these differences. The statuses are the registry manifest's and `E5Capabilities.Profiles`, which a test keeps equal key for key.

| Item | Contract |
| --- | --- |
| Profile | `multilingual-e5-small/onnx-fp32-cuda` (registry manifest `model-pipeline/registry/multilingual-e5-small/onnx-fp32-cuda/manifest.yaml`). The same model and tokenizer as the CPU profile. |
| Runtime | ONNX Runtime 1.30.0 with the CUDA execution provider (the application adds `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 instead of the CPU natives), `use_tf32=0`, fp32. The CUDA 13 runtime and cuDNN 9 must be on the loader path; the package does not ship them. |
| Supported platform | Linux x64 with an **NVIDIA GeForce RTX 3090** (Ampere), driver 580 or later. G1 to G3 on the release sets, the binding G4 one-hour soak and G5 ran on RTX 3090s. **Regression:** the per-release GPU run (the qualification workflow's `cuda (runpod)` job, for every `ai-v*` tag and on demand) reruns the CUDA tests (the adapters over CUDA included), G1 to G3 on the release sets and a soak smoke test, on an RTX 3090 only; the job refuses any other GPU. Other GPUs, including the RTX 4090 (Ada), are untested and outside the contract. |
| Memory on Linux | No `MALLOC_*` setting is required; heap trim on release as on CPU. |
| Cancellation | A CUDA run is never terminated mid-graph: it completes, then the call throws `OperationCanceledException` (#61). Latency is one chunk, **about 125 ms** for 32 x 512 tokens. |
| Numerical bound | The CUDA profile's protected tolerances (embedding max abs 1e-5 against the PyTorch reference). Release set: 1.21e-7; C# against Python ONNX Runtime CUDA 8e-8. |
| Task quality | MIRACL: nDCG@10 0.7978, MRR@10 0.7717 (budget 0.792 and 0.765). |
| Performance | G5 on an RTX 3090: warm latency 0.915 to 1.028 of the Python ONNX Runtime CUDA run (budget 1.05). |
| Supported | Embedding from text and from token ids; batch; cancellation; concurrent calls; explicit acquisition; `Starlings.Models.E5.Extensions.AI` and `Starlings.Models.E5.MLNet` over a CUDA session. |
| Outside the Stable contract | Native AOT with the CUDA natives: experimental (one local run, `tools/aot-check-ai.sh --cuda`, not in the per-release run). Windows x64 CUDA: untested. |

Evidence: [binding CUDA G4 and G5 round](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-16-cuda/README.md), [per-release GPU regression, run 38857860186](https://github.com/starlinghq/starling/blob/main/docs/evidence/gpu-regression-2026-10-16/README.md), [release-set gates](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/e5-onnx-fp32-cuda/README.md), [CUDA cancellation](https://github.com/starlinghq/starling/blob/main/docs/evidence/gliner2-cuda-wrong-output/README.md).

## Memory on Linux

- **Heap trim on release (on by default).** When a session's native memory is released (after `Dispose` and the last in-flight run), `Starlings.Runtime.Onnx` calls glibc's `malloc_trim(0)`, so a process that loads and unloads the model returns the freed memory instead of keeping it resident. It never runs on the request path and changes no allocator setting. Opt out per session with `OnnxSessionOptions.TrimNativeHeapOnRelease = false`, or for the whole process with the `AppContext` switch `Starlings.Runtime.Onnx.DisableNativeHeapTrim` (details in the `Starlings.Runtime.Onnx` README).
- **Transparent huge pages.** The qualification soaks ran with `/sys/kernel/mm/transparent_hugepage/enabled` set to `madvise` (the Ubuntu default). On hosts set to `always`, the kernel's khugepaged fills in partly used 2 MiB heap ranges, so resident memory can look larger and rise over time without any leak.
- No `MALLOC_MMAP_THRESHOLD_` or other `MALLOC_*` setting is required for this profile.

Evidence: [scheduled regression fix](https://github.com/starlinghq/starling/blob/main/docs/evidence/scheduled-regression-fix/README.md).

## Qualification evidence

- G0 provenance: the [0.1.0-preview.1 record](https://github.com/starlinghq/starling/blob/main/docs/releases/e5-0.1.0-preview.1.md), confirmed anonymously in the [2026-10-14 provenance check](https://github.com/starlinghq/starling/blob/main/docs/evidence/g0-provenance-2026-10-14/README.md).
- G1, G2 and G3 on the P-030 release sets: [release-set gates](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/e5-onnx-fp32-cpu/README.md).
- G4 one-hour soak: [2026-10-14](https://github.com/starlinghq/starling/blob/main/docs/evidence/g4-soak-1h-2026-10-14/README.md). G5 matched benchmark: [2026-10-15](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-15/README.md).
- G6 distribution and G7 integration: [G6](https://github.com/starlinghq/starling/blob/main/docs/evidence/g6-distribution-2026-10-14/README.md), [G7](https://github.com/starlinghq/starling/blob/main/docs/evidence/g7-integration-2026-10-14/README.md).
- Regression support: the weekly [qualification workflow](https://github.com/starlinghq/starling/blob/main/.github/workflows/qualification.yml) (jobs `release-gates` and `soak`), green in [run 38822625245](https://github.com/starlinghq/starling/actions/runs/38822625245).
