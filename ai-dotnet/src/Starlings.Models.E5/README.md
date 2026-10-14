# Starlings.Models.E5

**Preview.** Sentence embeddings from [`intfloat/multilingual-e5-small`](https://huggingface.co/intfloat/multilingual-e5-small) at revision `614241f622f53c4eeff9890bdc4f31cfecc418b3`, on ONNX Runtime 1.30.0, in-process and offline after an explicit download. The API and the supported platforms may still change. Not Stable yet: see [Path to Stable](#path-to-stable).

- Text goes through the owned tokenizer `Starlings.Tokenizers` (exact against huggingface/tokenizers on the published fixtures). No Microsoft.ML.Tokenizers and no native tokenizer library.
- Batching with right padding, mean pooling over the attention mask, L2 normalization: 384-dimensional unit vectors.
- The package ships the model's asset lock (repository, commit, file sizes and SHA-256), not the 487 MB of model files. Acquisition is an explicit call; inference never downloads.

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

Profile `multilingual-e5-small/onnx-fp32-cpu`: maturity **Preview**. Linux x64 CPU: experimental (46 fixture texts within 2.4e-7 of the PyTorch reference, bound 1e-5). Windows, macOS, ARM64, Native AOT: untested. Profile `multilingual-e5-small/onnx-fp32-cuda` (Linux x64 CUDA): Experimental. Evidence, known issues and reproduction commands: [release notes](https://github.com/starlinghq/starling/blob/main/docs/releases/ai-0.1.0-preview.6.md). Model license: MIT, stated only in the model card.

## Stable contract

This is the contract a Stable `onnx-fp32-cpu` profile would commit to. Today it is a Preview contract: the statuses below are the current ones, from the registry manifest and `E5Capabilities.All`. Support levels are defined in the [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md).

| Item | Contract |
| --- | --- |
| Model | `intfloat/multilingual-e5-small` at `614241f622f53c4eeff9890bdc4f31cfecc418b3`; `onnx/model.onnx` and `tokenizer.json` verified by SHA-256 (`E5ModelAssets.Lock`). |
| Tokenizer | `Starlings.Tokenizers` `0.3.0-preview.1`, profile `xlmr-unigram-e5` (pinned `tokenizer.json`, not SentencePiece or transformers' rebuilt tokenizer). |
| Runtime | ONNX Runtime 1.30.0 (`Microsoft.ML.OnnxRuntime.Managed`; the application adds exactly one native package), fp32, CPU. |
| Input | Text, or token ids (`EncoderInput`). The caller adds `"query: "` or `"passage: "`; nothing is inserted. |
| Limits | 512 tokens including `<s>` and `</s>`. `E5Tokenizer` truncates on the right to 512 (the profile's `truncate-right`, as the reference). `E5Encoder` rejects id sequences longer than `MaxSequenceLength` and never truncates them. |
| Output | 384-dimensional float32 vectors: mean pooling over the attention mask, then L2 normalization. Batches are right-padded; batch rows equal single calls within the protected tolerances. |
| Numerical bound | The protected tolerances of the profile (embedding max abs 1e-5 against the PyTorch reference). |
| Supported | None yet. Every capability below is at most `experimental`. |
| Experimental | text and pre-tokenized embedding, batch, cancellation (at ONNX Runtime node boundaries), concurrent calls, asset acquisition, `Starlings.Models.E5.Extensions.AI` (`IEmbeddingGenerator`), `Starlings.Models.E5.MLNet` (pipeline composition). Linux x64 CPU. |
| Untested | Native AOT, Windows x64, macOS, ARM64; the Extensions.AI and ML.NET adapters on CUDA. |
| Unsupported | streaming, pair input, non-zero `token_type_ids`, encoder-side truncation, prefix insertion, TF32, fp16, quantized graphs, TensorRT, persistence (ML.NET save), training, ML.NET cancellation, Extensions.AI `Dimensions` truncation. |

## Path to Stable

Profile `onnx-fp32-cpu` is the first Stable candidate (`docs/plan/production-readiness.md`, order of promotion). Gate evidence is in the [catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md) once draft PR #4 lands. Agents produce evidence; only the maintainer records a gate or changes maturity.

- [x] G0 provenance: passed ([preview.1 record](https://github.com/starlinghq/starling/blob/main/docs/releases/e5-0.1.0-preview.1.md)). The MIT license is stated only in the model card, and maintainer sign-off is pending (M4).
- [x] G1 and G2: passed ([fidelity report](https://github.com/starlinghq/starling/blob/main/docs/releases/e5-0.1.0-preview.2/fidelity-report.md)), on development fixtures at tokenizer `0.1.0-preview.4`. Rerun at `0.3.0-preview.1` passed, 2.32e-7 (draft PR #6, development fixtures).
- [ ] G1 and G2 on the protected P-030 release sets (M1).
- [ ] G3 task quality: the retrieval evaluator (nDCG@10, recall@k) is in draft PR #5. It needs a licensed dataset and a budget in `protected/` (M1, M2). Today: checklist.
- [ ] G4 resource correctness: one-hour soak and the plan 10.2 operational tests (R1). Today: checklist.
- [ ] G5 performance: a controlled benchmark against the profile budget (M2, M6). Today: one uncontrolled run ([2026-10-09](https://github.com/starlinghq/starling/blob/main/docs/benchmarks/e5-2026-10-09.md)), so checklist.
- [x] G6 distribution, Linux x64 CPU: passed ([rc drill](https://github.com/starlinghq/starling/blob/main/docs/releases/e5-0.1.0-preview.2/rc-drill.log)).
- [ ] G6 on Windows x64, and the ML.NET package in the release drill (R9).
- [ ] G7 integration: a representative downstream use by the first named consumer (M7, P-028). Today: checklist.
- [ ] `aot`: tested to `supported` or recorded `unsupported` (R9). The validator refuses Stable with any `untested` capability.
- [ ] Qualification CI (P-018, R6): a Windows x64 job and a scheduled qualification run. These give the compatibility matrix rows and regression support.
- [ ] A Linux x64 CPU compatibility row at `supported`, from qualification CI.
- [ ] Support policy accepted, and a second release owner named (M7; [support policy](https://github.com/starlinghq/starling/blob/main/docs/support-policy.md)).
- [ ] The maintained-update and rollback drill (P-022, [design](https://github.com/starlinghq/starling/blob/main/docs/plan/maintained-update-drill.md)).
- [ ] Packages published: the `release` environment and the NuGet prefix (M5).
