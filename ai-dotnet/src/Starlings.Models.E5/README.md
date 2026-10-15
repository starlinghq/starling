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
| `multilingual-e5-small/onnx-fp32-cpu` | preview | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `multilingual-e5-small/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
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

Profile `multilingual-e5-small/onnx-fp32-cpu` (Linux x64 CPU) has passed every qualification gate, G0 to G7; its maturity is in the table above. Profile `multilingual-e5-small/onnx-fp32-cuda` (Linux x64 CUDA) is Experimental: its one-hour soak failed (memory growth, recorded in the catalog), and there are no scheduled GPU regression runs. Gate evidence: the [catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md). Model license: MIT, stated only in the model card.

## Stable contract

This is the contract of profile `multilingual-e5-small/onnx-fp32-cpu`. Its Stable label commits to it under the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md). The statuses are the registry manifest's and `E5Capabilities.Profiles`, which a test keeps equal key for key.

| Item | Contract |
| --- | --- |
| Profile | `multilingual-e5-small/onnx-fp32-cpu` (registry manifest `model-pipeline/registry/multilingual-e5-small/onnx-fp32-cpu/manifest.yaml`). |
| Model | `intfloat/multilingual-e5-small` at `614241f622f53c4eeff9890bdc4f31cfecc418b3`; `onnx/model.onnx` and `tokenizer.json` verified by SHA-256 (`E5ModelAssets.Lock`). |
| Tokenizer | `Starlings.Tokenizers` `0.3.0-preview.1`, profile `xlmr-unigram-e5` (pinned `tokenizer.json`, not SentencePiece or transformers' rebuilt tokenizer). |
| Runtime | ONNX Runtime 1.30.0 (`Microsoft.ML.OnnxRuntime.Managed`; the application adds `Microsoft.ML.OnnxRuntime` 1.30.0 CPU natives), fp32, .NET 10. |
| Supported platform | Linux x64, CPU. Gates G1 to G5 ran on an Intel Core i7-5820K (Haswell-E, 6 cores and 12 threads, AVX2 and FMA, no AVX-512), .NET 10.0.12. The weekly qualification reruns G1 to G3 on the release sets and a short soak on GitHub-hosted `ubuntu-latest` x64. |
| Input | Text, or token ids (`EncoderInput`). The caller adds `"query: "` or `"passage: "`; nothing is inserted. |
| Output | 384-dimensional float32 vectors: mean pooling over the attention mask, then L2 normalization. Batches are right-padded; batch rows equal single calls within the protected tolerances. |
| Numerical bound | The protected tolerances of the profile (embedding max abs 1e-5 against the PyTorch reference). Release set: 1.46e-7. |
| Task quality | MIRACL dev, 8 languages: nDCG@10 at least 0.792 and MRR@10 at least 0.765 (budget); measured 0.7978 and 0.7717. |
| Supported | Embedding from text and from token ids; batch; cancellation (at ONNX Runtime node boundaries); concurrent calls; explicit acquisition (download, verify, offline reuse); Native AOT for the typed API and the Extensions.AI generator; `Starlings.Models.E5.Extensions.AI` (`IEmbeddingGenerator`); `Starlings.Models.E5.MLNet` (pipeline composition, row-wise and look-ahead batching). |
| Limits | 512 tokens including `<s>` and `</s>`. `E5Tokenizer` truncates on the right to 512 (the profile's `truncate-right`, as the reference); `E5Encoder` rejects id sequences longer than `MaxSequenceLength` and never truncates them. The model files (487 MB) are acquired explicitly; inference never downloads. |
| Outside the Stable contract | Windows x64 CPU: experimental (CI builds, runs the weight-free tests and the clean-install drill without the model; no model run). The CUDA profile and the adapters over CUDA: experimental. macOS, ARM64, Windows CUDA: unsupported (no evidence). |
| Unsupported | streaming, pair input, non-zero `token_type_ids`, encoder-side truncation, prefix insertion, TF32, fp16, quantized graphs, TensorRT, persistence (ML.NET save), training, ML.NET cancellation, the ML.NET adapter under Native AOT, Extensions.AI `Dimensions` truncation. |
| Known issues | The pinned `tokenizer.json` emits a lone `▁` token before special-token literals and for trailing whitespace, where SentencePiece and transformers 5.18 emit none; the profile follows `tokenizer.json`. It also leaves decomposed Hangul jamo uncomposed, while SentencePiece composes them. Both are recorded, not fixed. |

## Qualification evidence

- G0 provenance: the [0.1.0-preview.1 record](https://github.com/starlinghq/starling/blob/main/docs/releases/e5-0.1.0-preview.1.md), confirmed anonymously in the [2026-10-14 provenance check](https://github.com/starlinghq/starling/blob/main/docs/evidence/g0-provenance-2026-10-14/README.md).
- G1, G2 and G3 on the P-030 release sets: [release-set gates](https://github.com/starlinghq/starling/blob/main/docs/evidence/release-set-gates-2026-10-14/e5-onnx-fp32-cpu/README.md).
- G4 one-hour soak: [2026-10-14](https://github.com/starlinghq/starling/blob/main/docs/evidence/g4-soak-1h-2026-10-14/README.md). G5 matched benchmark: [2026-10-15](https://github.com/starlinghq/starling/blob/main/docs/evidence/qualification-2026-10-15/README.md).
- G6 distribution and G7 integration: [G6](https://github.com/starlinghq/starling/blob/main/docs/evidence/g6-distribution-2026-10-14/README.md), [G7](https://github.com/starlinghq/starling/blob/main/docs/evidence/g7-integration-2026-10-14/README.md).
- Regression support: the weekly [qualification workflow](https://github.com/starlinghq/starling/blob/main/.github/workflows/qualification.yml) (jobs `release-gates` and `soak`).
