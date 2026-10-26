# Starlings.Acquisition

Explicit, verified acquisition of model files. An asset lock names a Hub repository, a full 40-character commit SHA, and each file's size and SHA-256. `ModelAssetStore.DownloadAsync` downloads (with retries and resume) into a local cache and verifies every file; `OpenLocal` resolves the cache offline and never touches the network. No dependencies beyond the BCL. The API may still change.

## Maturity

<!-- catalog:maturity profiles=*-cpu,*-cuda -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/image-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/image-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `embeddinggemma-2/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `embeddinggemma-2/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `embeddinggemma-2/video-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/video-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `gliner2-base-v1/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; linux-x64 cuda: supported; windows-x64 cpu: experimental |
| `gliner2-base-v1/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `kev-0.8b/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: experimental; windows-x64 cpu: untested |
| `kev-0.8b/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `kev-4b/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: experimental; windows-x64 cpu: untested |
| `kev-4b/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `multilingual-e5-small/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `multilingual-e5-small/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `qwen2.5-0.5b-instruct/ortgenai-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `qwen3-0.6b/ortgenai-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `whisper-large-v3-turbo/onnx-fp16-cpu` | experimental | linux-x64 cpu: experimental; linux-x64 cuda: untested; windows-x64 cpu: untested |
<!-- /catalog:maturity -->

```csharp
using Starlings.Acquisition;

var assetLock = new ModelAssetLock("my-model", "owner/name", "<40-character commit sha>",
    [new ModelAssetFile("tokenizer.json", 17082730, "<sha256>")]);
using var store = new ModelAssetStore(new ModelAcquisitionOptions { CacheDirectory = "/var/cache/models" });
ModelSnapshot downloaded = await store.DownloadAsync(assetLock);     // network, explicit
ModelSnapshot local = store.OpenLocal(assetLock);                    // offline, verified
string path = local.GetFilePath("tokenizer.json");
```

Cache root: `ModelAcquisitionOptions.CacheDirectory`, else `$STARLING_MODELS`, else the user cache directory. Gated repositories: `HuggingFaceToken` or `$HF_TOKEN`, sent only over https, to the Hub endpoint and to explicit sources on `https://huggingface.co` (redirects are followed per hop; the token never goes to the Hub's storage or CDN hosts, to any other host, or over `http://`, so an `http://` endpoint is used without it). A file that is not in the Hub repository (for example a converted ONNX graph published as a release asset or in another Hugging Face repository) carries an explicit `https` source, `new ModelAssetFile(path, size, sha256, new Uri("https://..."))`: it is downloaded from that address (with the token only if the address is on huggingface.co) and verified in the same snapshot directory (lock JSON `schema_version` 2 with a per-file `url`; locks without explicit sources stay `schema_version` 1). Model packages ship their lock, for example `E5ModelAssets.Lock` in `Starlings.Models.E5`. Documentation: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#model-acquisition-p-009).

## API

- `ModelAssetStore(options, httpClient)`: no network or file access in the constructor. `DownloadAsync(lock, progress, cancellationToken)` is the only member that downloads; it reuses verified cached files, re-downloads invalid ones and returns a `ModelSnapshot`. `OpenLocal(lock)` verifies the cache and never downloads. `ModelAssetStore.VerifyDirectory(lock, directory, verification)` verifies any directory laid out like the repository (for example a copied bundle). `GetSnapshotDirectory(lock)` and `CacheDirectory` give the paths. Cache layout: `<cache>/<owner>/<name>/<commit SHA>/<path>`; partial downloads live under `<cache>/.downloads/` and resume on the next attempt.
- `ModelSnapshot`: `Directory`, `Lock`, `Verification`, `GetFilePath(path)` (throws for a path not in the lock).
- `ModelAssetLock`: `Name`, `Repository`, `Revision`, `Files`, `TotalSize`, `GetFile(path)`, `Parse(json)`, `Load(path)`, `ToJson()`. `ModelAssetFile`: `Path`, `Size`, `Sha256`, optional `Source`.
- `ModelAcquisitionOptions`: `CacheDirectory`, `Verification` (default `CacheVerification.Sha256`; `Size` checks sizes only), `HuggingFaceToken`, `UseEnvironmentToken` (default true), `Endpoint` (default `https://huggingface.co/`), `MaxAttempts` (default 4, for network errors, 408, 429 and 5xx), `RetryDelay` (default 1 s, doubled per attempt), `ResponseTimeout` (default 100 s, for the response headers and for each body read: a stalled transfer is a transient failure that keeps its partial file and resumes). Default cache root (`DefaultCacheDirectory`): `$XDG_CACHE_HOME/starlings/models` or `~/.cache/starlings/models` on Linux, `~/Library/Caches/starlings/models` on macOS, `%LOCALAPPDATA%\starlings\models` on Windows. Fresh downloads are always verified by size and SHA-256.
- Progress: `IProgress<ModelAssetProgress>` with `Path`, `Stage` (`Verifying`, `Downloading`, `Completed`), `BytesProcessed`, `TotalBytes`.
- Exceptions, all derived from `ModelAcquisitionException` (`Repository`, `Revision`, `AssetPath`; messages never contain the token): `ModelAssetMissingException` (offline resolution found files missing; `MissingPaths`, `SnapshotDirectory`), `ModelAssetIntegrityException` (size or SHA-256 differs; expected and actual values), `ModelAssetAccessDeniedException` (HTTP 401/403; `StatusCode`, `TokenProvided`), `ModelAssetNotFoundException` (HTTP 404), `ModelAssetDownloadException` (failure after all attempts; `StatusCode`, `Attempts`). A failure to write the partial file itself (disk full, permissions) is a `ModelAcquisitionException` naming the path; it is not retried.

Status: G6 passed on Linux x64 for the locks of all 13 Stable profiles: clean installs from the packed packages, caches seeded and verified or filled by anonymous downloads into empty caches, then offline runs ([evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/g6-distribution-2026-10-14/README.md)). Windows and macOS untested.
