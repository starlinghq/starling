# Native runtimes, CUDA builds and Native AOT

How an application picks the ONNX Runtime and ONNX Runtime GenAI native packages for the Starlings packages, how to
build one project for CPU and CUDA, what a Linux-only GPU deployment needs, and which packages publish under Native
AOT. The package READMEs and [ai-dotnet/README.md](../../ai-dotnet/README.md#onnx-runtime-packages) hold the
reference tables; this guide puts the steps in one place.

## The native packages

The Starlings libraries reference only the managed bindings: `Microsoft.ML.OnnxRuntime.Managed` 1.30.0
(`Starlings.Runtime.Onnx`, used by e5, GLiNER2, EmbeddingGemma 2, BERT, Kev and Whisper) and
`Microsoft.ML.OnnxRuntimeGenAI.Managed` 0.17.1 (`Starlings.Generation.OnnxGenAI`, used by Qwen 3.5 and QwenChat). The
application references exactly one native package per runtime, at the same version:

| Runtime | CPU | CUDA (Linux x64) |
| --- | --- | --- |
| ONNX Runtime | `Microsoft.ML.OnnxRuntime` 1.30.0 | `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 |
| ONNX Runtime GenAI | `Microsoft.ML.OnnxRuntimeGenAI` 0.17.1 (brings `Microsoft.ML.OnnxRuntime`) | `Microsoft.ML.OnnxRuntimeGenAI.Cuda` 0.17.1 (brings `Microsoft.ML.OnnxRuntime.Gpu`) |

The CPU and GPU packages of one runtime ship the native library under the same file name (`libonnxruntime`,
`libonnxruntime-genai`), so a project must never carry both. An application that uses an encoder and a generator on
CUDA references `Microsoft.ML.OnnxRuntimeGenAI.Cuda` only: it brings the GPU ONNX Runtime the encoder also uses. That
combination is untested (ai-dotnet/README.md, "ONNX Runtime GenAI packages").

## Build-time check

`Starlings.Runtime.Onnx` and `Starlings.Generation.OnnxGenAI` ship MSBuild targets in `buildTransitive/`, so they run
in every project that references a Starlings model runtime package (not the definition packages `Starlings.Models.Qwen35`,
`Starlings.Models.QwenChat` and `Starlings.Models.Kev`, which bring no runtime). After restore they read the native assets in the
project's package graph (direct or transitive) and report:

| Code | Level | Condition |
| --- | --- | --- |
| `STARLINGS001` | warning | `Starlings.Runtime.Onnx` is in the graph but no ONNX Runtime native package |
| `STARLINGS002` | error | both `Microsoft.ML.OnnxRuntime` and a `Microsoft.ML.OnnxRuntime.Gpu*` package are in the graph |
| `STARLINGS003` | warning | `Starlings.Generation.OnnxGenAI` is in the graph but no GenAI native package |
| `STARLINGS004` | error | both `Microsoft.ML.OnnxRuntimeGenAI` and `Microsoft.ML.OnnxRuntimeGenAI.Cuda` are in the graph |

The checks run in executables (`OutputType` `Exe` or `WinExe`) and test projects; a class library leaves the choice to
its application. Browser projects (`RuntimeIdentifier` `browser-*`, or `Starlings.Runtime.OnnxWeb` in the graph) are
skipped. An application that deploys its own native library turns a check off with
`<StarlingsOnnxRuntimeNativeCheck>false</StarlingsOnnxRuntimeNativeCheck>` or
`<StarlingsOnnxRuntimeGenAINativeCheck>false</StarlingsOnnxRuntimeGenAINativeCheck>`. Without the warning, a missing
native shows up only when the first session or model is created (`OnnxRuntimeUnavailableException`,
`GenAIRuntimeUnavailableException`). The release drill (`tools/release-drill-ai.sh`) builds a consumer for each of
the four codes from the packed packages, and checks that no consumer with one native package per runtime triggers any
of them.

## One project, CPU and CUDA builds

Starling's own tools (`ai-dotnet/tools/Starlings.Kev.Fidelity.Tool`, `Starlings.Qwen35.Fidelity`) build one project
for both providers with a `UseCuda` property. The same pattern in an application:

```xml
<PropertyGroup>
  <!-- The CUDA build restores other packages, so it keeps its own lock file. -->
  <NuGetLockFilePath Condition="'$(UseCuda)' == 'true'">packages.cuda.lock.json</NuGetLockFilePath>
</PropertyGroup>
<ItemGroup>
  <PackageReference Include="Starlings.Models.E5" Version="0.3.0" />
</ItemGroup>
<ItemGroup Condition="'$(UseCuda)' != 'true'">
  <PackageReference Include="Microsoft.ML.OnnxRuntime" Version="1.30.0" />
</ItemGroup>
<ItemGroup Condition="'$(UseCuda)' == 'true'">
  <PackageReference Include="Microsoft.ML.OnnxRuntime.Gpu.Linux" Version="1.30.0" />
</ItemGroup>
```

```sh
dotnet build -c Release                                   # CPU natives
dotnet publish -c Release -r linux-x64 -p:UseCuda=true    # CUDA natives, Linux x64 only
```

The provider is still chosen in code (`OnnxExecutionProvider.Cuda`, or the model package's provider option); the
build only decides which native library ships. A CPU session runs on either build, because the GPU package contains
the CPU provider too. A CUDA session on the CPU build fails with `OnnxProviderUnavailableException`; there is no
silent fallback. For a generator, swap `Microsoft.ML.OnnxRuntimeGenAI` for `Microsoft.ML.OnnxRuntimeGenAI.Cuda` the same
way.

## Linux-only GPU deployments

- **CUDA libraries are the machine's.** Neither GPU package ships the CUDA 13 runtime (`libcudart`), cuBLAS
  (`libcublas`, `libcublasLt`) or cuDNN 9. They must be on the loader path (`LD_LIBRARY_PATH`, or the system library
  directories) of the process. The release drill passes them with `--cuda-lib-path`.
- **Prefer `Microsoft.ML.OnnxRuntime.Gpu.Linux` over `Microsoft.ML.OnnxRuntime.Gpu`.** The latter is a metapackage
  that depends on `.Gpu.Linux` and `.Gpu.Windows`; on Linux the Windows natives are dead weight (the
  `.Gpu.Windows` 1.30.0 package is 153 MB compressed, nuget.org, 2026-10-26).
- **`Microsoft.ML.OnnxRuntimeGenAI.Cuda` brings `Microsoft.ML.OnnxRuntime.Gpu`, so both natives.** Restore downloads
  them; build and publish with a runtime identifier (`-r linux-x64`, or `<RuntimeIdentifier>linux-x64</RuntimeIdentifier>`)
  so that only the linux-x64 native assets are copied to the output. A build without a runtime identifier copies every
  RID's natives under `runtimes/`.
- **Pick the device explicitly.** Set `CUDA_VISIBLE_DEVICES` (with `CUDA_DEVICE_ORDER=PCI_BUS_ID` when indexes must match
  `nvidia-smi`) and the package's device id option; TF32 is always off.

## Native AOT

Two checks publish the packed packages with `-p:PublishAot=true` outside the repository:

- `tools/aot-check-ai.sh` runs inference on real model files under JIT and under AOT and compares the results
  ([evidence](../evidence/aot/README.md)). It covers the packages its targets list.
- `tools/aot-smoke-ai.sh` needs no model files. For every published package it roots the whole package assembly, so
  ILC analyzes every member, records each trim and AOT warning, runs the binary, and writes `summary.md` with a verdict
  per package: `AOT-clean`, `clean in Starlings.*` (warnings only in third-party code) or `not clean`. The release
  drill runs it with `--aot` or `--aot-smoke`. The latest recorded run is in
  [docs/evidence/aot-smoke/](../evidence/aot-smoke/README.md).

The ML.NET adapters (`Starlings.Models.E5.MLNet`, `Starlings.Models.EmbeddingGemma2.MLNet`) are AOT unsupported:
Microsoft.ML builds its row getters with `Reflection.Emit` ([evidence](../evidence/aot/README.md)). CUDA under Native
AOT is untested.

## Why there are no runtime convenience packages

Packages such as `Starlings.Runtime.Cpu` or `Starlings.Runtime.Cuda`, which would reference the qualified native
versions for the application, are not provided. CI forbids native ONNX Runtime and GenAI package references under
`ai-dotnet/src/`, directly or transitively, so the libraries never force a native choice or a GPU download on their
users. The build-time check above covers the failure such packages would prevent (a missing or duplicate native),
without a package per provider and platform. Umbrella packages per model family are not provided either: each package
already pins its Starlings dependencies to the exact version (`[0.3.0]`), so referencing a model package brings the
whole set at one version, and a project that mixes versions gets NuGet warning NU1608 at restore.
