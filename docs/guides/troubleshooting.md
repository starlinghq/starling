# Troubleshooting first runs

The failures people hit before any model call succeeds, with the exact error text and the fix. Nothing here is
invented: every message is quoted from the source in `ai-dotnet/src/`. For what each profile supports, see the
[model catalog](../catalog/README.md) and the [compatibility matrix](../catalog/compatibility-matrix.md); for the
native packages in depth, [native-runtimes-and-cuda.md](native-runtimes-and-cuda.md).

## The build fails before anything runs: two native ONNX Runtime packages

```
STARLINGS002 (error): Both the ONNX Runtime CPU package (Microsoft.ML.OnnxRuntime) and a GPU package (Microsoft.ML.OnnxRuntime.Gpu.Linux) are in this project's package graph. They ship the same native library, so only one may be referenced: keep Microsoft.ML.OnnxRuntime for CPU, or Microsoft.ML.OnnxRuntime.Gpu.Linux for CUDA (see the Starlings.Runtime.Onnx README, 'Native package check'). A package that brings one transitively (Microsoft.ML.OnnxRuntimeGenAI brings the CPU package, Microsoft.ML.OnnxRuntimeGenAI.Cuda the GPU ones) counts too.
```

Fix: reference exactly one native package per runtime. For CPU, `Microsoft.ML.OnnxRuntime`; for CUDA on Linux x64,
`Microsoft.ML.OnnxRuntime.Gpu.Linux`. A project that also runs a generator keeps only
`Microsoft.ML.OnnxRuntimeGenAI.Cuda` — it brings the GPU ONNX Runtime the encoders use. The CPU and GPU packages of
one runtime ship the same native library name (`libonnxruntime`), so both cannot load. The related checks:
`STARLINGS001` (warning: `Starlings.Runtime.Onnx` in the graph, no native package) and `STARLINGS004` (error: both
GenAI packages). Source: `ai-dotnet/src/Starlings.Runtime.Onnx/buildTransitive/Starlings.Runtime.Onnx.targets`.

A sibling failure after the build: `OnnxProviderUnavailableException` — the app was built against the CPU package but
asks for `OnnxExecutionProvider.Cuda`. There is no silent fallback; rebuild with the GPU package
(`dotnet publish -c Release -r linux-x64 -p:UseCuda=true` in a dual-provider project).

The opposite first-run failure: no native package at all. The build passes with only the `STARLINGS001` warning
("This project references Starlings.Runtime.Onnx (through a Starlings model package) but no ONNX Runtime native
package, so creating a session will throw OnnxRuntimeUnavailableException. Add exactly one: Microsoft.ML.OnnxRuntime
1.30.0 (CPU) or Microsoft.ML.OnnxRuntime.Gpu.Linux 1.30.0 (CUDA)."), and the first model creation throws
`OnnxRuntimeUnavailableException`. The fix is what the warning says: add the native package the graph is missing
(`Microsoft.ML.OnnxRuntime` 1.30.0 for CPU, `Microsoft.ML.OnnxRuntime.Gpu.Linux` 1.30.0 for CUDA on Linux x64), and
for the GenAI runtime the matching `Microsoft.ML.OnnxRuntimeGenAI`/`.Cuda` package — `STARLINGS003` warns when
`Starlings.Generation.OnnxGenAI` is in the graph without one.

## The first run fails with a download error

`DownloadAsync` on `XModelAssets` (for example `E5ModelAssets.DownloadAsync()`) is the only member that uses the
network. A network or server failure that persists after all retries (default 4, exponential backoff, HTTP range
resumption) throws `ModelAssetDownloadException`; a private or gated repository throws
`ModelAssetDownloadException`'s sibling `ModelAssetAccessDeniedException` on 401/403 from huggingface.co — set
`ModelAcquisitionOptions.HuggingFaceToken` or `$HF_TOKEN`.

What to know when it fails:

- A partial download is kept as `<file>.partial` under `<cache root>/.downloads/`, and the next attempt resumes it
  with an HTTP range request; cancellation and process kills keep it too. If a resumed file fails the digest, the
  store starts over once from zero.
- The cache root is `$STARLING_MODELS` if set, else the user cache directory (`$XDG_CACHE_HOME` or `~/.cache` +
  `/starlings/models` on Linux, `~/Library/Caches/starlings/models` on macOS, `%LOCALAPPDATA%\starlings\models` on
  Windows). A full disk fails the download like any other write error — clear space or point `STARLING_MODELS` at
  another volume; `.partial` files of abandoned downloads can be removed with
  `ModelAssetStore.RemovePartialDownloads` or `Starlings.Acquisition.Tool`'s `clean` command.
- Later runs need no network: `XModelAssets.OpenLocal(...)` verifies the cached files and returns the snapshot
  offline. But an empty cache is an error, not a download: `OpenLocal` throws `ModelAssetMissingException` and
  fetches nothing. Run `Starlings.Acquisition.Tool status` to list missing and failing files (or `Check`/
  `TryOpenLocal` for the same without exceptions), then call `DownloadAsync` — the quickstart's first line —
  before reopening locally.
- Acquisitions of the same file serialize on a lock file, so two processes (or two threads) each waiting on the
  same model is normal: the second finds the verified file and does not download again.

Source: `ai-dotnet/src/Starlings.Acquisition/ModelAssetStore.cs`, `Exceptions.cs`, and the acquisition section of
[ai-dotnet/README.md](../../ai-dotnet/README.md#model-acquisition-p-009).

## The call throws "input too long"

`Gliner2InputTooLongException` (GLiNER2, limit 512 tokens including the schema prefix by default,
`Gliner2ExtractorOptions.MaxSequenceLength`) and `EmbeddingGemma2InputTooLongException` (EmbeddingGemma 2, up to
8,192 including `<bos>`, `<eos>` and the prompt, `EmbeddingGemma2EncoderOptions.MaxSequenceLength`) both mean the
same thing: the input was **rejected, not truncated**. Nothing was embedded or extracted; the exception carries
`InputIndex` (which input in the call), `TokenCount` (what it needed) and `MaxTokens` (the limit). Split or shorten
the text.

The exception is the e5 contract: an input longer than `E5EncoderOptions.MaxSequenceLength` (512) throws
`InputTooLongException` with the message:

```
Input {index} has {n} tokens; the maximum is {options.MaxSequenceLength}. Truncate in the tokenizer.
```

e5 is the only model that truncates instead of throwing, and only on one of its two layers: `E5TextEmbedder` with
the default `E5Tokenizer` truncates text on the right to `MaxSequenceLength` (512, `<s>`/`</s>` included) before
embedding, as the reference does. The encoder itself never truncates: a token-id sequence over its limit throws
`InputTooLongException`, and any other `IEncoderTokenizer` must truncate to the same length itself.
`E5TextEmbedder.CountTokens(text)` counts without embedding and
`Validate(text)` reports whether a text would be truncated; `TextChunker` can split long documents into 512-token
windows (pass the `"passage: "` prefix as `TextChunkerOptions.Prefix`).

## Native memory grows across loads on Linux (GLiNER2, EG2 video on CUDA)

With default glibc settings, a process that loads and unloads these models repeatedly keeps freed native memory
resident, and the high-water mark creeps upward (allocator retention, not a leak: `malloc_trim` returns it). The
fix is one environment variable on the deployment process:

```
MALLOC_MMAP_THRESHOLD_=65536
```

Required for the EmbeddingGemma 2 CUDA video profile (`video-fp32-cuda`), recommended for `image-fp32-cuda`, and
required for GLiNER2 in deployment on CPU and CUDA. The e5 and text profiles need no `MALLOC_*` setting. The
systemd, Dockerfile and launcher-script forms are in the
[`Starlings.Runtime.Onnx` README](https://github.com/starlinghq/starling/blob/main/ai-dotnet/src/Starlings.Runtime.Onnx/README.md#setting-malloc_mmap_threshold_);
the measurements are in `docs/evidence/gliner2-soak-growth/` and `docs/evidence/cuda-soak-growth/`.

## Common first-run mistakes

- **e5 prefixes.** e5 expects `"query: "` or `"passage: "` at the start of every text; nothing is added for you.
  Similarity between an unprefixed query and prefixed passages is silently poor, not an error.
- **EmbeddingGemma 2 prompts.** Prompts are explicit and never added for you: `EmbeddingGemma2Prompt.Query`
  (`"task: search result | query: "`) for queries, `EmbeddingGemma2Prompt.Document` (`"title: none | text: "`) for
  documents, or `None`. Prefixes apply to text only — pass images, video and audio without one.
- **Matryoshka dimensions.** EmbeddingGemma 2 supports 768 (default), 512, 256 and 128; the vector is truncated and
  then normalized again, like sentence-transformers `truncate_dim`. Other sizes are rejected with
  `ArgumentOutOfRangeException` ("Supported dimensions are 768, 512, 256 and 128 (Matryoshka truncation).").
- **The CUDA package on non-Linux.** `Microsoft.ML.OnnxRuntime.Gpu.Linux` is Linux x64 only. There is no Windows or
  macOS GPU package in the supported matrix; on those platforms use the CPU package. macOS (Apple Silicon) and
  Windows x64 CPU are not yet covered by the contract at all — the run reports are issues
  [#77](https://github.com/starlinghq/starling/issues/77) and
  [#78](https://github.com/starlinghq/starling/issues/78).

## Where to look next

- [Model catalog](../catalog/README.md): every profile, its maturity and the evidence for each gate.
- [Compatibility matrix](../catalog/compatibility-matrix.md): OS, architecture and provider per profile.
- [Evidence index](../evidence/README.md): dated qualification, soak and regression records, including the CUDA
  qualification runs where the download-retry behaviour above was exercised.
- Package READMEs under `ai-dotnet/src/<package>/README.md` hold the per-model contracts, and
  [ai-dotnet/README.md](../../ai-dotnet/README.md) holds the acquisition and runtime reference sections.
- Bugs and platform reports: open an issue, or add evidence to #77 (macOS) / #78 (Windows) if that is your platform.
