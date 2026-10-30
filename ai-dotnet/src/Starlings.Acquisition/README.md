# Starlings.Acquisition

Explicit, verified acquisition of model files. An asset lock names a Hub repository, a full 40-character commit SHA, and each file's size and SHA-256. `ModelAssetStore.DownloadAsync` downloads (with retries and resume) into a local cache and verifies every file; `OpenLocal` resolves the cache offline and never touches the network. No dependencies beyond the BCL. The API may still change.

## Maturity

<!-- catalog:maturity profiles=*-cpu,*-cuda -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `all-minilm-l6-v2/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `bge-base-en-v1.5/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `bge-m3/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `bge-reranker-v2-m3/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `bge-small-en-v1.5/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
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
| `ms-marco-minilm-l6-v2/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `multilingual-e5-small/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `multilingual-e5-small/onnx-fp32-cuda` | stable | linux-x64 cuda: supported; windows-x64 cuda: untested |
| `nomic-embed-text-v1.5/onnx-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
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

Cache root: `ModelAcquisitionOptions.CacheDirectory`, else `$STARLING_MODELS`, else the user cache directory. Gated repositories: `HuggingFaceToken` or `$HF_TOKEN`, sent only over https, to the Hub endpoint and to explicit sources on `https://huggingface.co` (redirects are followed per hop; the token never goes to the Hub's storage or CDN hosts, to any other host, or over `http://`, so an `http://` endpoint is used without it). A file that is not in the Hub repository (for example a converted ONNX graph published as a release asset or in another Hugging Face repository) carries an explicit `https` source, `new ModelAssetFile(path, size, sha256, new Uri("https://..."))`: it is downloaded from that address (with the token only if the address is on huggingface.co) and verified in the same snapshot directory (lock JSON `schema_version` 2 with a per-file `url`; locks without explicit sources stay `schema_version` 1). A lock whose files live in another Hub repository than the model's own sets `Hosting = new ModelAssetHosting(repository, commit)`: files without a source are downloaded from there, while the cache directory and messages keep the lock's `Repository` and `Revision` (`schema_version` 3, which also carries an optional `DisplayName`). Model packages ship their lock, for example `E5ModelAssets.Lock` in `Starlings.Models.E5`. Documentation: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#model-acquisition-p-009).

## API

- `ModelAssetStore(options, httpClient)`: no network or file access in the constructor. `DownloadAsync(lock, progress, cancellationToken)` is the only member that downloads; it reuses verified cached files, re-downloads invalid ones and returns a `ModelSnapshot`. `OpenLocal(lock)` verifies the cache and never downloads. `ModelAssetStore.VerifyDirectory(lock, directory, verification)` verifies any directory laid out like the repository (for example a copied bundle). `GetSnapshotDirectory(lock)` and `CacheDirectory` give the paths. Cache layout: `<cache>/<owner>/<name>/<commit SHA>/<path>`; partial downloads live under `<cache>/.downloads/` and resume on the next attempt.
- Per-call settings: `OpenLocal(lock, verification)`, `TryOpenLocal(lock[, verification], out snapshot)`, and `DownloadAsync(lock, ModelDownloadOptions, cancellationToken)` with `Verification`, `Progress`, `LockWaitTimeout` and `MaxConcurrentFiles` (null members use the store's options). A store is thread-safe and holds no per-lock state; keep one for the life of the process rather than building one per call.
- Status without exceptions: `Check(lock[, verification])` and `ModelAssetStore.CheckDirectory(lock, directory, verification)` return a `ModelAssetStatus`: `State` (`Complete`, `Missing`, `Incomplete`, `Invalid`), `SnapshotDirectory`, `MissingPaths`, `Problems` (`ModelAssetProblem`: path, `Kind` `SizeMismatch`/`DigestMismatch`/`Unreadable`, expected and actual size and digest, message), `MissingBytes`, and `Snapshot` when complete. Never downloads and changes nothing.
- Download options (`ModelAcquisitionOptions`): `MaxConcurrentFiles` (default 1, one file after the other as before), `LockWaitTimeout` (default null, wait as before; then `ModelAssetLockTimeoutException` with `LockFilePath` and `Waited`), `FallbackEndpoints` (Hub-layout mirrors tried in order after `Endpoint` on a download failure after all attempts, 401/403, 404 or wrong bytes; a mirror gets the token only under the usual rule, so one on another host never does; when all fail, the `Endpoint` failure is thrown with every failure in `EndpointFailures`), `LocalSources` (directories laid out like the cache, copied from and verified before any request; never changed) and `AllowNetwork` (default true; false for air-gapped installs: only the cache and local sources, and a missing file raises `ModelAssetMissingException` before anything is copied). Explicit sources do not use the mirrors.
- Maintenance: `RemovePartialDownloads(lock)` removes the lock's partial downloads and lock files under `.downloads/`, skipping any file whose lock another downloader holds. `PlanPrune(keepLocks)` lists, without changing anything, the snapshot directories no kept lock names and, under `.downloads/`, partial downloads that are not of a kept lock's file and lock files; `Prune(plan)` removes them, checking each lock again, and `Prune(keepLocks)` does both. Only directories in the cache layout (`<owner>/<name>/<40-character SHA>`) are touched. Both return a `ModelCacheCleanup` (`Removed`, `Skipped`, `BytesRemoved`) of `ModelCacheEntry` values (`Path`, `Kind`, `Bytes`, `Repository`, `Revision`). Pass every lock that any user of the cache root still needs, and prune when no download of a revision being removed is starting.
- `ModelSnapshot`: `Directory`, `Lock`, `Verification`, `GetFilePath(path)` (throws for a path not in the lock), `GetVerifiedSha256(path)`: the lock's digest of a file when the snapshot hashed every file (`Verification` is `Sha256`, or the files were just downloaded and hashed in transit), otherwise null. Pass it as `OnnxSessionOptions.KnownModelSha256` so a session does not read the file a second time; the model packages' `XModelFiles` expose it (`ModelSha256`) and their factories pass it.
- `ModelAssetLock`: `Name`, `Repository`, `Revision`, `Files`, `TotalSize`, `GetFile(path)`, `Parse(json)`, `Load(path)`, `ToJson()`, and (format version 3) `DisplayName`, `Hosting` (`ModelAssetHosting`: `Repository`, `Revision`), `HostingRepository`, `HostingRevision`. `ModelAssetFile`: `Path`, `Size`, `Sha256`, optional `Source`.
- Lock format: the JSON of `ToJson`/`Parse` is the interchange format for locks with other tools. `ToJson` writes the lowest version that holds the lock (1: Hub-layout files only; 2: a per-file `url`; 3: `display_name` or `hosting`); `Parse` reads 1 to 3 and rejects a field of a later version than the one declared. The schema is [`model-pipeline/schemas/asset-lock.schema.json`](https://github.com/starlinghq/starling/blob/main/model-pipeline/schemas/asset-lock.schema.json) (`model-pipeline validate-lock <file>`). `ModelAssetLock.SchemaVersion` is now 3; `SourcesSchemaVersion` (2) and `HubOnlySchemaVersion` (1) name the others.
- `ModelAcquisitionOptions`: `CacheDirectory`, `Verification` (default `CacheVerification.Sha256`; `Size` checks sizes only), `HuggingFaceToken`, `UseEnvironmentToken` (default true), `Endpoint` (default `https://huggingface.co/`), `MaxAttempts` (default 4, for network errors, 408, 429 and 5xx), `RetryDelay` (default 1 s, doubled per attempt), `ResponseTimeout` (default 100 s, for the response headers and for each body read: a stalled transfer is a transient failure that keeps its partial file and resumes). Default cache root (`DefaultCacheDirectory`): `$XDG_CACHE_HOME/starlings/models` or `~/.cache/starlings/models` on Linux, `~/Library/Caches/starlings/models` on macOS, `%LOCALAPPDATA%\starlings\models` on Windows. Fresh downloads are always verified by size and SHA-256.
- Progress: `IProgress<ModelAssetProgress>` with `Path`, `Stage` (`Verifying`, `Downloading`, `Completed`, and `Waiting` once while another downloader holds the file's lock), `BytesProcessed`, `TotalBytes`, and the lock totals `FileIndex`, `FileCount`, `LockBytesProcessed`, `LockTotalBytes`. Values are reported one at a time, also with concurrent files.
- Exceptions, all derived from `ModelAcquisitionException` (`Repository`, `Revision`, `AssetPath`; messages never contain the token): `ModelAssetMissingException` (offline resolution found files missing; `MissingPaths`, `SnapshotDirectory`), `ModelAssetIntegrityException` (size or SHA-256 differs; expected and actual values), `ModelAssetAccessDeniedException` (HTTP 401/403; `StatusCode`, `TokenProvided`), `ModelAssetNotFoundException` (HTTP 404), `ModelAssetDownloadException` (failure after all attempts; `StatusCode`, `Attempts`), `ModelAssetLockTimeoutException` (`LockWaitTimeout` passed; `LockFilePath`, `Waited`). With fallback endpoints, `EndpointFailures` lists the failure of each address tried. A failure to write the partial file itself (disk full, permissions) is a `ModelAcquisitionException` naming the path; it is not retried.

Status of the members added after 0.3.0 (status, per-call settings, lock totals, lock wait, concurrent files, mirrors, local sources, hosting, cleanup and pruning): experimental, covered by tests against the in-process fake Hub on Linux x64; no real-Hub run.

Status: G6 passed on Linux x64 for the locks of all 13 Stable profiles: clean installs from the packed packages, caches seeded and verified or filled by anonymous downloads into empty caches, then offline runs ([evidence](https://github.com/starlinghq/starling/blob/main/docs/evidence/g6-distribution-2026-10-14/README.md)). Windows and macOS untested.
