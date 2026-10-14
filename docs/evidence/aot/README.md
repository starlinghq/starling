# Native AOT evidence (R9, gate G6)

Date 2026-10-13. Commit `756ba5a`, a work-in-progress commit later squashed into this PR's single commit; `ai-dotnet/` and `tools/` are identical, and only the evidence and manifest comments were added after the run. Linux x64, CPU only, Intel i7-5820K, .NET SDK 10.0.100 in the clone (packing) and 10.0.401 outside it (consumers), runtime 10.0.12, clang as the native linker. The machine was shared with other heavy jobs, so timings are meaningless and none are reported.

## Method

`tools/aot-check-ai.sh` builds `tools/release-drill/aot-consumer` outside the repository against the **packed** packages from a local folder feed. It builds once per target, and each target references only one package and its dependencies, so a warning belongs to that package. For each target it does the following:

1. It does a normal JIT build (`dotnet build`, dynamic code on) and runs it on the cached model files. This is the reference.
2. It runs `dotnet publish -r linux-x64 -p:PublishAot=true` with `TrimmerSingleWarn=false`, so every trim/AOT warning is reported per call site. The warnings are saved in `<target>/warnings.txt`.
3. It runs the native binary on the same files with no network (`unshare -rn`).
4. It compares the two results files. Strings (token ids, entity spans, generated text and token ids, shapes, finish reasons and usage) must be exact. Float vectors must be within 1e-5. That bound belongs to this check, not to a protected tolerance, and every vector was in fact bit for bit equal (max abs 0). Diagnostics that may legitimately differ are reported but not compared.

`tools/release-drill-ai.sh --aot` runs the same check on the drill's own feed and model cache (step 12). The run here is [`aot-check-drill.log`](aot-check-drill.log), which is part of [`../release-drill-r9/rc-drill.log`](../release-drill-r9/rc-drill.log). The drill seeds only the EmbeddingGemma 2 text graph unless an `images/` directory is given, so it skipped the `eg2` target. That target ran standalone on the full cache: [`aot-check-eg2.log`](aot-check-eg2.log).

## Results

| Target | Packages exercised | Trim/AOT warnings in Starlings.* | Other warnings | AOT run | AOT vs JIT |
| --- | --- | --- | --- | --- | --- |
| `acquisition` | Acquisition | 0 | 0 | ok | Lock JSON round trip, offline verification, and a real tokenizer.json download from the public Hub with no token: exact |
| `runtime` | Runtime.Onnx | 0 | 0 | ok | Raw e5 session output 1x8x384: max abs 0 |
| `e5` | Models.E5 (+ Tokenizers) | 0 | 0 | ok | Ids exact (6 texts); 12 embeddings max abs 0 |
| `e5-meai` | Models.E5.Extensions.AI | 0 | 0 | ok | 6 embeddings max abs 0; usage and metadata exact |
| `e5-mlnet` | Models.E5.MLNet | 0 | 20 (Microsoft.ML) | **fails** | Not comparable: see below |
| `gliner2` | Models.Gliner2 | 0 | 0 | ok | Entity spans exact; confidences max abs 0 |
| `eg2` | Models.EmbeddingGemma2 | 0 | 0 | ok | Query and document text, synthetic image, interleaved text and image, and a 3-frame video from decoded frames: 9 vectors max abs 0 |
| `eg2-meai` | Models.EmbeddingGemma2.Extensions.AI | 0 | 0 | ok | 4 vectors (including Matryoshka 256) max abs 0 |
| `eg2-mlnet` | Models.EmbeddingGemma2.MLNet | 0 | 20 (Microsoft.ML) | **fails** | Not comparable: see below |
| `qwen35` | Models.Qwen35 + Generation.OnnxGenAI | 0 | 0 | ok | "The capital of France is Paris.": token ids, finish reason, usage 24+7 exact |
| `qwen35-meai` | Models.Qwen35.Extensions.AI | 0 | 0 | ok | Response and streamed text, finish reason, usage exact |

Each AOT binary is 3.2 to 7.2 MB. The ONNX Runtime and GenAI natives sit next to it (`<target>/publish-files.txt`).

## Findings

- **Library code is trim/AOT clean.** The trim, AOT and single-file analyzers report nothing in any Starlings library, and neither does ILC on the published binaries, except in Microsoft.ML. This PR sets `IsAotCompatible` on the ten packages whose AOT runs pass: Acquisition, Runtime.Onnx, Models.E5, Models.Gliner2, Models.EmbeddingGemma2, Generation.OnnxGenAI, Models.Qwen35, and the three Extensions.AI adapters. The analyzers now run in every build, and with `TreatWarningsAsErrors` a regression fails the build. The setting adds the SDK's `Microsoft.NET.ILLink.Tasks` (build-time only, `PrivateAssets=all`) to those packages' lock files. The drill confirmed it is not in any nuspec. The lock entry follows the SDK band: `10.0.0` for SDK 10.0.100, pinned by `global.json`.
- **ML.NET adapters: AOT unsupported.** Microsoft.ML 5.0.0 is not AOT-compatible, and ILC reports 20 warnings in it (IL2060 x5, IL2065, IL2070 x4, IL3050 x10). At run time, `MLContext.Data.LoadFromEnumerable` builds its row getters with `System.Reflection.Emit.DynamicMethod` (`Microsoft.ML.ApiUtils.GeneratePeek`) through `MakeGenericMethod`. Under Native AOT this throws `NotSupportedException` ("missing native code"). It throws `PlatformNotSupportedException` when dynamic code is merely switched off. The consumer roots its own row type, so the failure is ML.NET's and not a trimmed property (`<target>/run-aot.log`). `Starlings.Models.*.MLNet` themselves produce no warnings, but they cannot work without ML.NET's data views. The fix belongs upstream in ML.NET, so it is not made here. The check runs these targets anyway and reports a pass as "revisit".
- **Qwen 3.5 version diagnostics under AOT.** `GenAIRuntimeInfo` reads the GenAI package versions from the `.deps.json`, as its documentation states. A Native AOT binary has no `.deps.json`, so `GenAIVersion` reports "unknown (qualified with 0.17.1)" and the version-match check has nothing to compare. Generation is identical. The application's NuGet references are then the only guard against a mismatched GenAI native. This counts against the G6 "version diagnostics" item for AOT deployments.
- **Not covered.** CUDA under AOT (no GPU here). The SkiaSharp image decoder and the FFmpeg video adapter (EmbeddingGemma 2 images and video ran from already-decoded pixels). Windows, macOS and ARM64 Native AOT. `Starlings.Tokenizers` alone (it is exercised through every model target, and it lives in starlinghq/tokenizers).

## Proposed `aot` statuses

These manifests change from `untested` to `experimental`, with a comment pointing here: `multilingual-e5-small/onnx-fp32-cpu`, `gliner2-base-v1/onnx-fp32-cpu`, `embeddinggemma-2/onnx-fp32-cpu`, `image-fp32-cpu`, `video-fp32-cpu` and `qwen3.5-0.8b/ortgenai-fp32x-cpu`. The CUDA profiles stay `untested`. No profile is `unsupported`: the ML.NET limitation belongs to the adapter, and the comments record it. Promotion to `supported` needs this check in qualification CI (R6) and a Windows run. The maintainer decides.
