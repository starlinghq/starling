# Native AOT smoke check of every package (backlog E1)

Date 2026-10-26. Linux x64 (`ubuntu.24.04-x64`), 4 vCPU Intel Xeon at 2.80 GHz, CPU only, no GPU. .NET SDK 10.0.113 (the
container's; `global.json` pins 10.0.100), clang as the native linker. The packages were packed from the working tree
of this change on `a691d5c` (the change adds the build targets and this check; no library code that ILC compiles
changed except `Starlings.Models.Bert`'s contract table loader), not from a clean clone, so this is a development
record, not a release drill. Timings are not reported.

## Method

`tools/aot-smoke-ai.sh --feed <feed>` builds `tools/release-drill/aot-smoke` once per package outside the repository
against the packed package and the CPU natives (`Microsoft.ML.OnnxRuntime` 1.30.0, `Microsoft.ML.OnnxRuntimeGenAI`
0.17.1). The project roots the whole package assembly (`TrimmerRootAssembly`), so ILC compiles and analyzes every
member of the package, not only the code a sample reaches. `dotnet publish -r <rid> -p:PublishAot=true` with
`TrimmerSingleWarn=false` reports every trim and AOT warning per call site; the script counts those in `Starlings.*`
code and elsewhere. The binary then runs with no network (`unshare -rn`): it loads the package assembly and reports
`Native AOT` mode. No model file is used, and no inference runs: [`../aot/`](../aot/README.md) (`tools/aot-check-ai.sh`)
is the inference check for the packages it covers.

Raw output: [`aot-smoke.log`](aot-smoke.log) (paths replaced by `<feed>` and `<out>`), [`summary.md`](summary.md), and
the warnings of the two packages that have any.

## Results

| Package | Warnings in Starlings.* | Other warnings | AOT publish | AOT run | Verdict |
| --- | --- | --- | --- | --- | --- |
| Starlings.Acquisition | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Runtime.Onnx | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.E5 | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.E5.Extensions.AI | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.E5.MLNet | 0 | 8 (Microsoft.ML) | ok | ok | clean in Starlings.*; AOT unsupported (below) |
| Starlings.Models.Gliner2 | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.EmbeddingGemma2 | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.EmbeddingGemma2.Extensions.AI | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.EmbeddingGemma2.MLNet | 0 | 8 (Microsoft.ML) | ok | ok | clean in Starlings.*; AOT unsupported (below) |
| Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.EmbeddingGemma2.Video.FFmpeg | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.EmbeddingGemma2.Video.FFmpeg9 | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Generation.OnnxGenAI | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.Qwen35 | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.Qwen35.Extensions.AI | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.Kev | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.QwenChat | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.Bert | 0 | 0 | ok | ok | AOT-clean |
| Starlings.Models.Whisper | 0 | 0 | ok | ok | AOT-clean |

Not published: `Starlings.Runtime.OnnxWeb`, `Starlings.Runtime.OnnxWeb.WebGpu` and `Starlings.Models.E5.Browser`, which
target `browser-wasm` (trimmed and checked in `browser/conformance/`).

## Findings

- **No trim or AOT warning in any Starlings package**, with every member rooted. This includes seven packages that do
  not set `IsAotCompatible` (so their builds do not run the analyzers): `Starlings.Models.Bert`,
  `Starlings.Models.QwenChat`, `Starlings.Models.EmbeddingGemma2.Imaging.SkiaSharp`, `.Video.FFmpeg`, `.Video.FFmpeg9`
  and the two ML.NET adapters. Setting it on the first five is a candidate follow-up; it is not part of this change.
- **ML.NET adapters stay AOT unsupported.** ILC reports 8 warnings in Microsoft.ML 5.0.0 here (IL2060, IL2065, IL2070
  x4, IL3050 x2); the inference check found 20 because its consumer reaches more of ML.NET. Loading the adapter works;
  using it fails at run time because ML.NET builds its row getters with `Reflection.Emit` ([`../aot/`](../aot/README.md)).
  The smoke run does not exercise that path, so "ok" in its run column says nothing about the adapter's AOT support.
- **What "AOT-clean" covers.** The package publishes, its code produces no trim or AOT warning, and the binary starts
  with the assembly present. It does not cover behavior: results under AOT are compared with JIT only by
  `tools/aot-check-ai.sh`, on real model files. CUDA, Windows, macOS and ARM64 Native AOT are not covered.
- No profile's `aot` capability status changes on this record.
