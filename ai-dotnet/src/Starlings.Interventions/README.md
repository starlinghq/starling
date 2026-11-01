# Starlings.Interventions

The contract of ONNX Runtime graph interventions: a profile that names one original ONNX graph and adds a low-rank directional projection at chosen residual-stream sites, the per-request strength, and the manifest a graph compiler writes for the derived graph. No ONNX Runtime or model dependency (it references only `Starlings.Models.Abstractions` for `CapabilityStatus`).

This library is **experimental** and source only. V1 evidence uses synthetic random orthonormal directions only; no behavioural profile ships or has been evaluated. The library ships the mechanism. Deciding which callers may apply which profile is the host application's job.

- `InterventionProfile`: `profile.json` (schema version 1). Id, `base_model_sha256` (the SHA-256 of the base `model.onnx` as the package's asset lock records it), operator, sites. Strict parsing: unknown, missing or duplicate properties, unknown schema versions, operators, targets and dtypes are rejected. `Serialize()` writes the canonical form.
- `InterventionSite`: layer, target (`residual_post`), hidden size, rank, dtype (`float32`), directions file and its SHA-256, per-direction scales.
- `InterventionProfileFile`: a profile read from disk with the SHA-256 of its bytes (its identity in a manifest) and the directory its directions files resolve against.
- `InterventionDirections`: the raw directions file (little-endian float32, row-major `[rank, hidden]`, no header), checked for size, SHA-256 and finite values.
- `InterventionSelection(profileId, strength)`: what a request asks for. Strength 0 leaves outputs unchanged.
- `InterventionManifest`: `compiler-manifest.json` (schema version 1). Base artifacts, profile id and hash, compiler and site-mapping versions, opset, precision, declared inputs, resolved site tensors, derived files and their hashes.
- `InterventionCacheKey`: the cache key of a derived artifact, recomputed from its manifest (the same text the Python cache writer hashes).
- `InterventionArtifact` and `InterventionArtifactCache`: the read side of the derived-artifact cache `<cache>/<model>/interventions/<key>/`. Open an entry, check its directory name against the key, verify the manifest, profile and derived files (`Fast` for every load: sizes, plus device, inode and modification time of the linked weights on Linux; `Full` to hash the linked weights and the base bundle). Entries are written by the Python compiler ([cache.md](https://github.com/starlinghq/starling/blob/main/docs/interventions/cache.md)).

The operator, for a site tensor `y` of shape `[..., hidden]`:

```
y' = y - alpha * ((y Q^T) ⊙ s) Q        Q: [rank, hidden], s: [rank], alpha: request strength
```

```csharp
using Starlings.Interventions;

InterventionProfileFile file = InterventionProfileFile.Load("profiles/synthetic/profile.json");
file.Profile.VerifyBaseModel(baseModelSha256);   // before anything loads: a profile for another graph is rejected
file.VerifyDirections();                         // size, SHA-256 and finite values of every directions file

InterventionManifest manifest = InterventionManifest.Load(Path.Combine(derivedDir, "compiler-manifest.json"));
manifest.VerifyProfile(file);                    // same profile bytes, same base graph, same sites
manifest.VerifyDerivedFiles(derivedDir);

// A cache entry written by model_pipeline.interventions.cache.get_or_compile:
InterventionArtifact artifact = InterventionArtifactCache.Open(cacheRoot, "lfm3", cacheKey);
artifact.Manifest.VerifyBaseModel(baseModelSha256);
// artifact.GraphPath loads with the ordinary ONNX Runtime session API.

var selection = new InterventionSelection(file.Profile.Id, strength: 0.5f);
```

Formats: [docs/interventions/profile-format.md](https://github.com/starlinghq/starling/blob/main/docs/interventions/profile-format.md). Design: [docs/interventions/architecture.md](https://github.com/starlinghq/starling/blob/main/docs/interventions/architecture.md).

## Capabilities

`InterventionCapabilities.All` lists every capability with an explicit status. In short:

- **Experimental:** profile and manifest schema version 1, the cache key and artifact verification (read side), the `directional_projection` operator, the `residual_post` target, raw float32 directions.
- **Unsupported (rejected):** other operators, targets and dtypes (float16 directions), several profiles per graph, hot swap, behavioural profiles, access control (the host's job), writing cache entries (the Python compiler does that).
- **Untested:** AOT (analyzers on, no AOT publish run).
