# Intervention graph compiler

Status: **experimental**. Code: `model-pipeline/src/model_pipeline/interventions/`. Tests: `model-pipeline/tests/test_interventions.py`. Design: [architecture.md](architecture.md); file formats: [profile-format.md](profile-format.md).

The compiler turns one base ONNX graph and one `profile.json` into a derived `model.onnx` that applies the `directional_projection` operator at the profile's sites, plus a `compiler-manifest.json` that the C# `InterventionManifest` reads and verifies. It needs the `interventions` extra of `model-pipeline` (numpy, onnx 1.23.1, onnxruntime 1.30.0, the version the C# packages pin); the `dev` extra includes it.

The mechanism evidence uses synthetic random orthonormal directions (`model_pipeline.interventions.synthetic`). Profiles extracted from prompt sets ([extraction.md](extraction.md)) compile the same way. The compiler ships the mechanism; which callers get which profile is the host application's job.

## Use

```python
from model_pipeline.interventions import compile_intervention

result = compile_intervention(
    base_dir="bundle",            # base model directory: manifest base_artifacts paths are relative to it
    graph="onnx/model.onnx",      # the base graph, relative to base_dir
    profile="profile/profile.json",
    out_dir="derived",            # must not be inside base_dir, nor contain it
)
# derived/model.onnx, derived/model.onnx_data (hard link), derived/compiler-manifest.json
```

Run the derived graph with the ordinary inputs plus `intervention_strength`, a float32 tensor of shape `[1]`. `0` gives the base graph's outputs; the strength is a graph input, so changing it per request recompiles nothing.

## What it does

1. Reads the base graph bytes and hashes them. The profile's `base_model_sha256` must equal that hash, or the compile fails before anything is written.
2. Finds the site mapping for that hash (`injection_sites.SITE_MAPPINGS`, or the `mapping` argument). A hash with no mapping is **unsupported** (`UnsupportedGraphError`); unknown graphs are never pattern-matched.
3. Parses the graph with `onnx.load_model_from_string`: external-data references stay references, and no weight is loaded.
4. For every profile site, checks the mapped tensor: it exists, is produced by a node (or is a graph input), is not an initializer, is float32 or float16, and its last dimension (when known from the graph or shape inference) equals `hidden_size`. The profile's and the mapping's `hidden_size` must agree. Directions are read with the size, then SHA-256, then finite check.
5. Inserts, right after the producer of `y`:

   ```
   coefficients = MatMul(y, Q^T)            # [..., rank]
   scaled       = Mul(s, alpha)             # [rank]
   delta        = MatMul(coefficients * scaled, Q)
   y'           = Sub(y, delta)
   ```

   A float16 site gets `Cast(y)` to float32 first and `Cast(delta)` back to float16 before the `Sub`. `Q^T`, `Q` and `s` are inline float32 initializers named `intervention/layer<L>/<target>/...`; `alpha` is the graph input `intervention_strength`, shared by every site.
6. Rewires **only the consumer nodes the site mapping lists** to read `y'`. The inserted nodes read `y`, and so does every unlisted reader, graph outputs included: a graph output that exposes the site tensor keeps reporting the original value. Inserted names that collide with any existing name are rejected.
7. Validates the derived graph against the base graph: onnx.checker (on a view where external initializers are graph inputs, so no file is opened; skipped when the checker already rejects the base graph, as it does for graphs that use operators only ONNX Runtime registers in the default domain, such as LFM3's `SimplifiedLayerNormalization`; a derived graph of a checker-clean base must pass), topological order (no cycle), non-strict shape inference, inputs equal to the base inputs plus exactly the strength input, outputs unchanged, every base initializer present and byte-identical (for external tensors: same location, offset and length), same IR version and opset imports, every base node still present. When the checker is skipped, a warning is logged (`model_pipeline.interventions.validate` logger) and the manifest records `"validation": {"onnx_checker": "skipped_base_rejected", "detail": "<first line of the checker's message>"}`; otherwise `"onnx_checker": "passed"`. The real LFM3 fp32 graph records `skipped_base_rejected` ("No Op registered for SimplifiedLayerNormalization with domain_version of 21").
8. Writes `model.onnx`, hard-links every external-data file of the base graph into the output directory under its original location, and writes `compiler-manifest.json` (canonical form; the C# reader re-serializes it byte for byte). The manifest lists each of those external-data files in `derived_files` (location, SHA-256, size, and for each file it linked the `identity`: device, inode and modification time in ns), so `VerifyDerivedFiles` checks the bytes ONNX Runtime reads next to the derived graph (the identity is taken before the weights are hashed and must be unchanged after, so a file rewritten in place during the compile fails it), not only the originals, and the cache readers' Fast level can detect a replaced or rewritten file without hashing it.
9. If anything fails after writing started, the files, links and directories this call created are removed again (a file it overwrote in an existing directory is not restored); temporary files are named per process, thread and call.

The base bundle is only read. The derived graph is deterministic: the same inputs give the same bytes.

## External data

The derived graph keeps the base graph's external-data entries unchanged, so it must sit next to files with those names. ONNX Runtime 1.30 (C# and Python, measured) rejects every location that leaves the graph's directory: parent-relative (`../original/model.onnx_data`), absolute, and symlinked files or directories, which it canonicalizes before the check. A **hard link** next to the derived graph loads and shares the original bytes, so there is no second copy.

- A hard link cannot cross file systems. The compiler compares device numbers before writing anything and raises `ExternalDataLinkError` (`EXDEV`, with both mount points and the remedy in the message) instead of copying; see [cache.md](cache.md) for why there is no copy fallback. The derived-artifact cache must live on the bundle's file system. The other route the spike measured is loading the derived graph with the session option `session.model_external_initializers_file_folder_path` set to the bundle directory; that needs a session-level option and is not what the compiler sets up.
- A link shares the inode. Rewriting the original file in place changes what the derived graph reads, and replacing it by rename leaves the link on the old bytes. The linked files are in the manifest's `derived_files`, so `InterventionManifest.VerifyDerivedFiles` detects either case (and a file swapped in next to `model.onnx`); it reads the whole weights file, so the caller decides when to pay for it. `VerifyBaseArtifacts` checks the originals in the bundle.
- onnx 1.23's own path-based checks refuse hard-linked external data ("potential hardlink attack"): `onnx.checker.check_model(path)` and `onnx.load(path)` with external data loading fail on the derived graph, although ONNX Runtime loads it. Use `onnx.load(path, load_external_data=False)` to inspect a derived graph.
- `link_external=False` writes only the graph and manifest, for a caller that places the external files itself. The manifest still lists them in `derived_files`, so verification fails until exactly those files are in place. Those files carry no `identity`, so the cache readers' Fast level rejects such an entry; only Full opens it.
- The output directory must not be inside the base directory, and must not contain it (a link at `out_dir/<location>` could otherwise land in the bundle).

## Site mappings

`injection_sites.SiteMapping` records, for one exact base graph SHA-256: a model name, a version label (`site_mapping_version` in the manifest), the precision, and one `MappedSite` per supported `(layer, target)`: the tensor, its hidden size and the consumer node names to rewire. Mappings are written by hand from the exported graph and versioned; they are never inferred.

The registry holds one mapping, `lfm3-1.7b/fp32/1`, for `LiquidAI/LFM3-1.7B-ONNX` `onnx/model.onnx` (fp32) at `c680759a`, SHA-256 `add6653f…5ef77` (the fp32 asset lock's): `residual_post` after each of the 24 decoder layers, the tensor `/model/layers.L/Add_2/output_0` (float32, hidden 2048), rewiring both readers (layer `L+1`'s operator norm and first residual add; after layer 23, the `logits_to_keep` slice). A test checks that these are every reader of each tensor in the real graph. The fp16 graph has no mapping. See [lfm3.md](lfm3.md).

## Capture mode

`model_pipeline.interventions.capture` builds a *capture graph*: the base graph with the mapped site tensors of the requested layers appended as graph outputs, and no other change (no weight change, no extra input). It follows the same rules as above: lookup by exact graph hash, external data left as references and hard-linked, a structural check against the base, deterministic bytes. Entries are cached under `<cache>/<model>/intervention-captures/<key>/` with their own `capture-manifest.json`, apart from intervention entries. Profile extraction uses it to read the residual stream ([extraction.md](extraction.md)).

## Not covered

- Real models other than the LFM3 fp32 graph: unsupported (no mapping).
- Site tensors inside subgraphs (`If`, `Loop`, `Scan` bodies) and listed consumers inside subgraphs: not supported. An unlisted subgraph reader keeps the original value.
- Element types other than float32 and float16 at the site, and profile dtypes other than float32.
- Base graphs below IR version 4 (there every initializer must also be a graph input, which would add inputs) and below default-domain opset 7.
- Strength 0 is exact for finite activations. An infinite activation at a site gives `inf * 0 = NaN` in the correction.
- Atomic publication, locking and reuse of derived artifacts belong to the derived-artifact cache ([cache.md](cache.md)); `compile_intervention` writes in place.
- Windows, the CUDA execution provider and graphs over 2 GiB without external data were not tested.
