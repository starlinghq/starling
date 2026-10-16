# Starlings.Acquisition

Explicit, verified acquisition of model files. An asset lock names a Hub repository, a full 40-character commit SHA, and each file's size and SHA-256. `ModelAssetStore.DownloadAsync` downloads (with retries and resume) into a local cache and verifies every file; `OpenLocal` resolves the cache offline and never touches the network. No dependencies beyond the BCL. The API may still change.

## Maturity

<!-- catalog:maturity profiles=* -->
Maturity is per profile, generated from the registry manifests by `model-pipeline catalog` (do not edit this table by hand). Gates, evidence and capabilities per profile: the [model catalog](https://github.com/starlinghq/starling/blob/main/docs/catalog/README.md).

| Profile | Maturity | Platforms |
| --- | --- | --- |
| `embeddinggemma-2/image-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `embeddinggemma-2/image-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `embeddinggemma-2/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `embeddinggemma-2/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `embeddinggemma-2/video-fp32-cpu` | experimental | linux-x64 cpu: experimental; windows-x64 cpu: untested |
| `embeddinggemma-2/video-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `gliner2-base-v1/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; linux-x64 cuda: experimental; windows-x64 cpu: experimental |
| `gliner2-base-v1/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `multilingual-e5-small/onnx-fp32-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: experimental |
| `multilingual-e5-small/onnx-fp32-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `qwen3.5-0.8b/ortgenai-fp16x-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cpu` | stable | linux-x64 cpu: supported; windows-x64 cpu: untested |
| `qwen3.5-0.8b/ortgenai-fp32x-cuda` | experimental | linux-x64 cuda: experimental; windows-x64 cuda: untested |
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

Cache root: `ModelAcquisitionOptions.CacheDirectory`, else `$STARLING_MODELS`, else the user cache directory. Gated repositories: `HuggingFaceToken` or `$HF_TOKEN`, sent only to the Hub endpoint and to explicit sources on `https://huggingface.co` (redirects are followed per hop; the token never goes to the Hub's storage or CDN hosts or to any other host). A file that is not in the Hub repository (for example a converted ONNX graph published as a release asset or in another Hugging Face repository) carries an explicit `https` source, `new ModelAssetFile(path, size, sha256, new Uri("https://..."))`: it is downloaded from that address (with the token only if the address is on huggingface.co) and verified in the same snapshot directory (lock JSON `schema_version` 2 with a per-file `url`; locks without explicit sources stay `schema_version` 1). Model packages ship their lock, for example `E5ModelAssets.Lock` in `Starlings.Models.E5`. Documentation: [ai-dotnet/README.md](https://github.com/starlinghq/starling/blob/main/ai-dotnet/README.md#model-acquisition-p-009).

Status: supported on Linux x64 for the GLiNER2 CPU profile's lock, experimental for the others (G6: anonymous downloads into empty caches, offline reuse from the packed packages); Windows and macOS untested.
