# ONNX Runtime interventions: architecture (V1)

Status: **experimental**. This page describes the design; [profile-format.md](profile-format.md) specifies the files.

## Goal

Apply a low-rank intervention to a model's residual stream at inference time, with:

- one original ONNX model whose weights are never changed, copied or downloaded twice;
- a derived graph compiled locally, once, at load time, that adds the intervention operators of one or more profiles and one strength input (a vector with one element per profile);
- per-request strengths passed as an ordinary graph input, so requests never recompile or reload anything.

## The operators

V1 had one operator, `directional_projection`; profile schema version 2 adds `directional_addition`, `y' = y + alpha * (s Q)` (a fixed direction `sum_i s_i Q_i` scaled by the strength; see [profile-format.md](profile-format.md)). For a site tensor `y` of shape `[..., hidden]`, the projection is:

```
y' = y - alpha * ((y Q^T) ⊙ s) Q
```

- `Q` (`[rank, hidden]`): the directions. Rows should be orthonormal; that is the profile author's job. The contract checks the shape and that every value is finite, and offers `InterventionDirections.MaxOrthonormalityError()` as a diagnostic.
- `s` (`[rank]`): per-direction scales, from the profile.
- `alpha` (scalar): the request-level strength, a graph input. `alpha = 0` gives `y' = y` exactly in real arithmetic. With orthonormal `Q`, `s = 1` and `alpha = 1`, the component of `y` in the span of `Q` is removed.

The compiler inserts it as `MatMul(y, Q^T)`, `Mul(s, alpha)`, `Mul(·, s·alpha)`, `MatMul(·, Q)`, `Sub(y, ·)`: the strength scales the `[rank]` vector rather than the `[..., hidden]` correction. Where the site tensor is float16, a `Cast` to float32 precedes the projection and the float32 correction is cast back before the `Sub`, so `alpha = 0` leaves the site value bit-identical. See [compiler.md](compiler.md).

## Pieces and where they live

| Piece | Where | What |
| --- | --- | --- |
| Contract | `ai-dotnet/src/Starlings.Interventions` | Profile, site, selection, manifest, directions file format; strict parsing, deterministic serialization, hash and dimension checks. No ONNX Runtime dependency. |
| Graph compiler | `model-pipeline/src/model_pipeline/interventions/` | Python, onnx API ([compiler.md](compiler.md)). Reads a profile and a model-specific site mapping, inserts the operator at the mapped tensors, adds the strength input, writes the derived graph and the compiler manifest. |
| Site mappings | `model-pipeline/src/model_pipeline/interventions/injection_sites.py` | Per model, versioned, keyed to the exact SHA-256 of the base graph asset. A graph hash with no mapping is unsupported; unknown graphs are never pattern-matched. |
| Derived artifact cache | `<cache>/<model>/interventions/<key>/`; writer `model_pipeline/interventions/cache.py`, reader `InterventionArtifactCache` | `model.onnx`, hard links to the base weights, `profile.json` (verbatim), its directions files and `compiler-manifest.json`. Keyed by a hash of every compile input, published by one directory rename under a file lock, reused on repeat, verified on every open; the source bundle is never written ([cache.md](cache.md)). |
| Session loading | `Starlings.Runtime.Onnx` | Derived graphs load through the ordinary `OnnxModelSession.Create(path, options)`; `OnnxSessionOptions` has no intervention setting. The companion `OnnxInterventionDiagnostics` reports the original and loaded graph identities, whether interventions are on, the profile ids and the intervention inputs (see "Loading a derived graph" below). |
| Capture graphs and extraction | `model-pipeline/src/model_pipeline/interventions/capture.py`, `extract.py` | A capture graph exposes the mapped site tensors as outputs (cached apart from intervention entries). `model-pipeline interventions extract` turns two paired prompt sets into a profile, choosing layers by held-out separability ([extraction.md](extraction.md)). |
| Evaluation | `model-pipeline/src/model_pipeline/interventions/evaluate.py` | `model-pipeline interventions evaluate` runs a profile over a strength sweep, with a magnitude-matched random control, and writes an evidence report with a recommended strength range or none ([evaluation.md](evaluation.md)). |
| Model integration | `Starlings.Models.Lfm3.Interventions` (source only) over `Starlings.Models.Lfm3` | Recognizes a verified derived graph through its manifest, accepts exactly the declared extra input, and passes the strength vector on every call ([lfm3.md](lfm3.md)). |

The graph compiler, the derived-artifact cache, the session diagnostics and the LFM3 integration are experimental. A packable model package cannot reference the source-only contract library, so a model's integration lives in a source-only companion that the model package grants internal access to; the model package's public surface does not change.

## Identities

- **Base graph identity**: the SHA-256 of the base `model.onnx` bytes, exactly as the model package's asset lock records it. A profile names it in `base_model_sha256`; applying a profile to another graph fails before any session is created.
- **External weights**: each external data file of the base graph is identified by path, SHA-256 and size in the manifest's `base_artifacts`. The derived graph references the original files; it never carries a copy.
- **Profile identity**: the SHA-256 of the `profile.json` bytes as stored. It is not a hash of a re-serialization, so no cross-language canonical JSON is needed. The cache stores `profile.json` byte for byte.
- **Derived graph identity**: SHA-256 of the derived `model.onnx` and of every other derived file, in the manifest.

The cache key of a derived graph combines the base graph and external weight identities, the profile hash, the compiler version, the site-mapping version, the opset, the precision and the declared inputs. All of them are recorded in the compiler manifest, so a reader recomputes the key from the manifest and checks it against the cache directory name ([cache.md](cache.md)).

## Loading a derived graph

A model package that supports interventions loads a derived graph like any other model file, then records what it loaded:

```csharp
InterventionArtifact artifact = InterventionArtifactCache.Open(cacheRoot, "lfm3", key);   // verify before loading
artifact.Manifest.VerifyBaseModel(expectedBaseGraphSha256);                                // wrong base graph: no session
using var session = OnnxModelSession.Create(artifact.GraphPath, new OnnxSessionOptions
{
    KnownModelSha256 = artifact.Manifest.DerivedGraph.Sha256,   // verified by Open; not re-read
});
var interventions = OnnxInterventionDiagnostics.ForDerivedGraph(
    session,
    artifact.Manifest.BaseModelSha256,
    artifact.Manifest.DerivedGraph.Sha256,
    artifact.Manifest.ProfileIds,          // every profile, in strength-vector order
    [.. artifact.Manifest.Inputs.Select(i => i.Name)]);
```

`Starlings.Runtime.Onnx` knows nothing about profiles or manifests: the diagnostics type takes plain values, and the model package (which references both libraries) fills it in. Without an intervention the package loads the original graph and reports `OnnxInterventionDiagnostics.ForOriginalGraph(session)`: same inputs, same cost as before.

## Requests

A request carries an `InterventionSelection`: a set of `(profileId, strength)` pairs, one pair in the simple case. The integration resolves it outside the graph to the strength vector of the loaded graph (each named profile gets its strength, every other profile 0; `InterventionManifest.ResolveStrengths`). So preloaded profiles are combined by strength, not selected: `0` disables one. The vector is fixed for the lifetime of a generation sequence and passed as a fresh tensor on every native call. When no intervention is configured, the original graph is loaded: no extra inputs, no extra cost.

## Several profiles in one graph

A graph compiled from profiles `[p0, ..., p(n-1)]` (compiler manifest schema version 2) has one float32 input of shape `[n]`. Each profile reads its own element through a `Gather`, so it sees a `[1]` strength exactly as in a single-profile graph. Where profiles share a site they are applied in strength-vector order, each reading the previous one's output; with every strength but one at 0, the graph computes what the single-profile graph of that profile computes (bit for bit on the synthetic graphs and on LFM3 fp32 CPU). The order is fixed when compiling and recorded in the manifest; a different order is a different graph and cache key. Single-profile graphs (manifest schema version 1) remain valid and load unchanged.

## Scope of V1

In scope: the `directional_projection` and (profile schema version 2) `directional_addition` operators at `residual_post` sites, float32 directions, one or more profiles per derived graph with a strength vector, synthetic directions as evidence.

Out of scope: training profiles, refusal-direction or other safety-reducing profiles, the ONNX Runtime GenAI path, CUDA kernels, hot swap.

Added after V1 (experimental): extracting a profile from two contrasting prompt sets with capture graphs ([extraction.md](extraction.md)). The first extracted profile is the benign style profile LFM3 `lfm3-1.7b-concise`. Evaluating a profile over a strength sweep ([evaluation.md](evaluation.md)) came next.

## Evidence and responsibility

V1 evidence uses synthetic random orthonormal directions only. It shows that the mechanism is exact (strength 0 matches the original graph; other strengths match a NumPy reference of the operators and of their composition), not that any profile has a useful or safe behavioural effect. The extracted `concise` profile comes with held-out separability evidence ([extraction.md](extraction.md)) and a strength-sweep evaluation that recommends no strength ([evaluation.md](evaluation.md), [evidence](../evidence/lfm3-concise-evaluation-2026-10-31/README.md)).

The library ships the mechanism. Which callers may use which profile, at what strength, is the host application's decision; the library performs no access control.
