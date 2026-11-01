# Intervention file formats (schema versions 1 and 2)

Status: **experimental**. Normative for the C# contract (`Starlings.Interventions`) and for the Python graph compiler (`model-pipeline`). Design: [architecture.md](architecture.md).

The mechanism evidence uses synthetic random orthonormal directions; `model-pipeline interventions extract` writes profiles in this format from prompt sets ([extraction.md](extraction.md)). The library ships the mechanism; which callers may apply which profile is the host application's job.

## Common rules

These apply to `profile.json` and `compiler-manifest.json`.

- UTF-8 without a byte order mark; standard JSON (no comments, no trailing commas, no `NaN` or `Infinity`); at most 16 MiB.
- **Strict**: a reader rejects an unknown property, a missing required property and a property that appears twice, in every object. A writer that needs a new field must raise the schema version and update the C# parser in the same change.
- `schema_version` is checked first. A reader accepts the integers `1` and `2` (see "Schema version 2" below) and rejects any other value, so a newer document fails on its version, not on a field it added. Version 1 documents are read and written exactly as before; a document is always re-serialized in its own version.
- Patterns written as `^...$` below match the whole string (Python `re.fullmatch`): a trailing newline is not allowed.
- Integers are JSON numbers without a fraction or exponent (`1`, not `1.0`).
- **SHA-256** values are 64 lower-case hexadecimal characters over the exact bytes of a file.
- **Relative paths** use forward slashes, are not rooted, and have no empty, `.` or `..` segment, no backslash, no colon and no control character (at most 512 characters). A path resolved against its directory never leaves it.
- Closed vocabularies are case-sensitive. Unknown values are rejected:

| Field | Values |
| --- | --- |
| `operator` | schema version 1: `directional_projection`; schema version 2: `directional_projection`, `directional_addition` |
| `target` | `residual_post`: the residual stream after decoder layer `layer` (that layer's output, the next layer's input) |
| `dtype` | `float32` |
| input `role` | `strength` |
| base artifact `role` | `graph`, `external_data` |

### Canonical serialization

`InterventionProfile.Serialize()` and `InterventionManifest.Serialize()` write the canonical form: properties in the order of the tables below, two-space indentation, LF line ends, a final LF, `sites` sorted by `layer` (then `target`), other arrays in document order, float32 numbers in the shortest form that round-trips to the same float32 (`0.1`, `0.33333334`, `1`). Equal documents serialize to equal bytes; parsing and re-serializing a canonical document reproduces it byte for byte.

Identity never depends on re-serialization: the **profile hash is the SHA-256 of the `profile.json` bytes as stored**. Authors should store the canonical form, but a reader hashes whatever bytes it was given, and the derived-artifact cache copies `profile.json` verbatim.

## `profile.json`

| Key | Type | Required | Rule |
| --- | --- | --- | --- |
| `schema_version` | integer | yes | `1`, or `2` (needed for `directional_addition`) |
| `id` | string | yes | `^[a-z0-9][a-z0-9._-]{0,127}$` |
| `description` | string | no | free text for people; omitted when absent |
| `base_model_sha256` | SHA-256 | yes | SHA-256 of the base `model.onnx` bytes, as the model package's asset lock records it |
| `operator` | string | yes | `directional_projection`; with schema version 2 also `directional_addition` |
| `sites` | array of site | yes | 1 to 1,024 entries; no two with the same `layer` and `target` |

Site:

| Key | Type | Required | Rule |
| --- | --- | --- | --- |
| `layer` | integer | yes | zero-based decoder layer, `>= 0` |
| `target` | string | yes | `residual_post` |
| `hidden_size` | integer | yes | last dimension of the site tensor, 1 to 1,048,576 |
| `rank` | integer | yes | number of directions, 1 to `hidden_size` |
| `dtype` | string | yes | `float32`: dtype of the directions file and the scales |
| `directions_file` | relative path | yes | relative to the directory of `profile.json` |
| `directions_sha256` | SHA-256 | yes | of the directions file |
| `scales` | array of number | yes | exactly `rank` finite float32 values: the per-direction scales `s` |

Scales are float32 values written as JSON numbers. A Python reader takes `numpy.float32(value)` of each; a C# reader parses them as `float`.

Example (synthetic):

```json
{
  "schema_version": 1,
  "id": "synthetic-demo",
  "description": "Synthetic random directions for tests",
  "base_model_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "operator": "directional_projection",
  "sites": [
    {
      "layer": 1,
      "target": "residual_post",
      "hidden_size": 8,
      "rank": 2,
      "dtype": "float32",
      "directions_file": "directions/layer1.f32",
      "directions_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
      "scales": [
        1,
        0.5
      ]
    }
  ]
}
```

## Directions file

V1 (`dtype: float32`): raw IEEE 754 binary32 values, **little-endian**, **row-major** `[rank, hidden_size]`, no header, exactly `rank * hidden_size * 4` bytes. Row `i` is direction `i`.

```python
q = q.astype("<f4")                       # shape (rank, hidden_size)
path.write_bytes(q.tobytes(order="C"))
sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
```

A reader checks the size, then the SHA-256 against `directions_sha256`, then that every value is finite. Rows should be orthonormal; that is the profile author's job and is not checked (`InterventionDirections.MaxOrthonormalityError()` reports the deviation).

The format was chosen over `.npy` so that neither side needs a header parser: the profile already records the shape and dtype.

## `compiler-manifest.json`

Written by the graph compiler next to the derived graph; read and verified by `InterventionManifest`. This section is schema version 1 (one profile per graph); schema version 2 (several profiles, one strength vector) is below.

| Key | Type | Required | Rule |
| --- | --- | --- | --- |
| `schema_version` | integer | yes | `1` |
| `compiler_version` | string | yes | `^[A-Za-z0-9][A-Za-z0-9._+/-]{0,63}$` |
| `site_mapping_version` | string | yes | same pattern; the version of the model-specific site mapping |
| `base_artifacts` | array of base artifact | yes | exactly one with role `graph`; paths unique |
| `profile_id` | string | yes | the profile's `id` |
| `profile_sha256` | SHA-256 | yes | of the `profile.json` bytes the graph was compiled from |
| `operator` | string | yes | the profile's operator (`directional_projection`; a `directional_addition` profile needs a schema version 2 manifest) |
| `opset` | integer | yes | default-domain ONNX opset of the derived graph, 1 to 1,000 |
| `precision` | string | yes | `^[a-z0-9][a-z0-9_]{0,31}$`, the base graph's precision (for example `fp32`, `fp16`, `q4`) |
| `inputs` | array of input | yes | the derived graph's extra inputs beyond the base graph's; exactly one with role `strength`, holding one element (shape `[]` or `[1]`); names unique |
| `sites` | array of site binding | yes | one per profile site, 1 to 1,024; no two with the same `layer` and `target`, and no two with the same `tensor` |
| `derived_graph` | file | yes | the derived `model.onnx`, relative to the derived artifact directory |
| `derived_files` | array of derived file | yes | other derived files; paths unique and different from `derived_graph.path`; may be empty. In a derived-artifact cache entry these are exactly the hard links of the base graph's external data ([cache.md](cache.md), "Verification"); a compiler that writes a file of its own here needs a reader change and a compiler-version bump |
| `validation` | validation | no | how the compiler validated the derived graph (below); the compiler always writes it |

Base artifact: `role` (`graph` or `external_data`), `path` (relative to the base model directory), `sha256`, `size` (integer bytes, `>= 0`), in that order.

File: `path`, `sha256`, `size`.

Derived file: `path`, `sha256`, `size`, then optionally `identity`: an object with `device`, `inode` (integers, 0 to 2^63-1) and `mtime_ns` (integer, -2^63 to 2^63-1), in that order. It is the `stat` of the file (symlinks followed) when the compiler linked it: device number, inode number and modification time in nanoseconds since the Unix epoch. A hard link shares all three with its base file. The values are machine-local and are not part of the cache key; Fast verification compares them ([cache.md](cache.md)). The compiler writes `identity` for every file it links and omits it with `link_external=False`.

Validation: `onnx_checker`, then optionally `detail`. `onnx_checker` is `passed` (onnx.checker accepted the derived graph) or `skipped_base_rejected` (onnx.checker already rejects the base graph, for example because it uses operators only ONNX Runtime registers in the default domain, so it cannot judge the derived graph; the structural checks and the ONNX Runtime load stand in for it). `detail` is one line, 1 to 512 characters, no control characters: the first line of the checker's message about the base graph.

Input: `name` (ONNX input name, non-empty, at most 1,024 characters, no control characters), `role` (`strength`), `dtype` (`float32`), `shape` (array of non-negative integers, at most 8 dimensions and 2,147,483,647 elements; `[]` is a scalar; the strength input's shape is exactly `[]` or `[1]`). A model integration accepts exactly the base graph's inputs plus these.

Site binding: `layer`, `target`, `tensor` (the base graph tensor the site mapping resolved the site to).

A reader verifies, before loading the derived graph:

1. `VerifyProfile(profile, sha)`: same `profile_id`, same `profile_sha256`, same operator, the base graph (`base_artifacts` entry with role `graph`) has the profile's `base_model_sha256`, and the sites (layer, target) equal the profile's.
2. `VerifyBaseModel(sha)`: the base graph is the one the application's asset lock expects.
3. `VerifyDerivedFiles(dir)`: size, then SHA-256 of `derived_graph` and every `derived_files` entry (`identity` is not used here; the cache readers' Fast level checks it instead of hashing).
4. Optionally `VerifyBaseArtifacts(dir)`: size, then SHA-256 of every base artifact (this hashes the external weights).

How the derived graph reaches the original external data files is decided by the derived-artifact cache and recorded there; the manifest identifies the files it depends on.

Example (synthetic):

```json
{
  "schema_version": 1,
  "compiler_version": "0.1.0",
  "site_mapping_version": "tiny-mlp/fp32/1",
  "base_artifacts": [
    {
      "role": "graph",
      "path": "onnx/model.onnx",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size": 1234
    },
    {
      "role": "external_data",
      "path": "onnx/model.onnx_data",
      "sha256": "4444444444444444444444444444444444444444444444444444444444444444",
      "size": 5000000000
    }
  ],
  "profile_id": "synthetic-demo",
  "profile_sha256": "2222222222222222222222222222222222222222222222222222222222222222",
  "operator": "directional_projection",
  "opset": 17,
  "precision": "fp32",
  "inputs": [
    {
      "name": "intervention_strength",
      "role": "strength",
      "dtype": "float32",
      "shape": [
        1
      ]
    }
  ],
  "sites": [
    {
      "layer": 1,
      "target": "residual_post",
      "tensor": "/model/layers.1/Add_1_output_0"
    }
  ],
  "derived_graph": {
    "path": "model.onnx",
    "sha256": "3333333333333333333333333333333333333333333333333333333333333333",
    "size": 2048
  },
  "derived_files": [
    {
      "path": "model.onnx_data",
      "sha256": "4444444444444444444444444444444444444444444444444444444444444444",
      "size": 5000000000,
      "identity": {
        "device": 66307,
        "inode": 1234567,
        "mtime_ns": 1761000000123456789
      }
    }
  ],
  "validation": {
    "onnx_checker": "skipped_base_rejected",
    "detail": "No Op registered for SimplifiedLayerNormalization with domain_version of 17"
  }
}
```

## Schema version 2

### Profiles: the `directional_addition` operator

A `profile.json` with `"schema_version": 2` has the same keys and rules as version 1; the only difference is that `operator` may also be `directional_addition`. A version 1 document naming `directional_addition` is rejected (on `operator`), as is any other operator name. For a site tensor `y` of shape `[..., hidden]`:

```
directional_projection   y' = y - alpha * ((y Q^T) * s) Q
directional_addition     y' = y + alpha * (s Q)          s Q = sum_i s_i Q_i, shape [hidden]
```

Directions files, scales, sites and hashing are the same for both. The addition does not depend on `y`; with `alpha = 0` it adds an exact zero, so the site value is unchanged (bit for bit, except that `-0` may become `+0`). Rows of `Q` need not be orthonormal for the addition, but the same file format and checks apply.

Example (synthetic; directions file and hash as in the version 1 example):

```json
{
  "schema_version": 2,
  "id": "synthetic-shift",
  "base_model_sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
  "operator": "directional_addition",
  "sites": [
    {
      "layer": 3,
      "target": "residual_post",
      "hidden_size": 8,
      "rank": 1,
      "dtype": "float32",
      "directions_file": "directions/layer3.f32",
      "directions_sha256": "1111111111111111111111111111111111111111111111111111111111111111",
      "scales": [
        0.5
      ]
    }
  ]
}
```

### Compiler manifest: several profiles, one strength vector

A derived graph may carry several profiles. It then takes **one** float32 input of shape `[n]`, `n` the number of profiles: element `i` is the strength of profile `i`. It is a strength vector, not a selector: profiles combine freely, and `0` disables one. The manifest that describes such a graph has `"schema_version": 2`. Its keys, in order:

| Key | Type | Required | Rule |
| --- | --- | --- | --- |
| `schema_version` | integer | yes | `2` |
| `compiler_version`, `site_mapping_version`, `base_artifacts` | | yes | as in schema version 1 |
| `profiles` | array of profile entry | yes | 1 to 64, in strength-vector order; ids unique |
| `opset`, `precision` | | yes | as in schema version 1 |
| `inputs` | array of input | yes | as in schema version 1, except that the strength input's shape is exactly `[n]` (also `[1]` for one profile; never `[]`) |
| `sites` | array of site binding | yes | as in schema version 1, plus `profiles` on each binding (below); every profile is applied at one site at least |
| `derived_graph`, `derived_files`, `validation` | | as in schema version 1 | |

Profile entry: `id` (profile id), `sha256` (of that `profile.json` as stored), `operator` (`directional_projection` or `directional_addition`), in that order. Its position is its strength index.

Site binding: `layer`, `target`, `tensor`, then `profiles`: the strength indices of the profiles applied at this site, one or more, strictly ascending.

**Composition at shared sites.** Where several profiles have a site at the same `(layer, target)`, they are applied **in strength-vector order** (ascending index, the order of the binding's `profiles`): the first reads the base tensor, each later one reads the previous one's output, and the consumers the site mapping lists read the last output. For example, with a projection at index 0 and an addition at index 1 on one site, `y' = (y - a0 ((y Q0^T) * s0) Q0) + a1 (s1 Q1)`. The operators do not commute in general, so two orders of the same profiles are two different graphs (and two cache keys). With one profile's strength at 0 its link adds an exact zero, so the graph computes what a graph without that profile computes. The order of the profiles in a request's selection does not matter.

A reader verifies, in place of the schema 1 profile check, `VerifyProfiles(profiles)`: the profiles in strength-vector order, each with the recorded id, SHA-256 and operator, the manifest's base graph, and exactly the sites whose bindings list its index. The other checks are those of schema version 1.

Example (synthetic):

```json
{
  "schema_version": 2,
  "compiler_version": "0.1.0",
  "site_mapping_version": "tiny-mlp/fp32/1",
  "base_artifacts": [
    {
      "role": "graph",
      "path": "onnx/model.onnx",
      "sha256": "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
      "size": 1234
    },
    {
      "role": "external_data",
      "path": "onnx/model.onnx_data",
      "sha256": "4444444444444444444444444444444444444444444444444444444444444444",
      "size": 5000000000
    }
  ],
  "profiles": [
    {
      "id": "synthetic-demo",
      "sha256": "2222222222222222222222222222222222222222222222222222222222222222",
      "operator": "directional_projection"
    },
    {
      "id": "synthetic-shift",
      "sha256": "5555555555555555555555555555555555555555555555555555555555555555",
      "operator": "directional_addition"
    }
  ],
  "opset": 17,
  "precision": "fp32",
  "inputs": [
    {
      "name": "intervention_strength",
      "role": "strength",
      "dtype": "float32",
      "shape": [
        2
      ]
    }
  ],
  "sites": [
    {
      "layer": 1,
      "target": "residual_post",
      "tensor": "/model/layers.1/Add_1_output_0",
      "profiles": [
        0
      ]
    },
    {
      "layer": 3,
      "target": "residual_post",
      "tensor": "/model/layers.3/Add_1_output_0",
      "profiles": [
        0,
        1
      ]
    }
  ],
  "derived_graph": {
    "path": "model.onnx",
    "sha256": "3333333333333333333333333333333333333333333333333333333333333333",
    "size": 2048
  },
  "derived_files": [
    {
      "path": "model.onnx_data",
      "sha256": "4444444444444444444444444444444444444444444444444444444444444444",
      "size": 5000000000,
      "identity": {
        "device": 66307,
        "inode": 1234567,
        "mtime_ns": 1761000000123456789
      }
    }
  ],
  "validation": {
    "onnx_checker": "passed"
  }
}
```

## `InterventionSelection`

Not a file: the request-level choice, a set of `(profile_id, strength)` pairs. Each `profile_id` follows the profile id rule and appears once; each `strength` is any finite float32 (0 leaves outputs unchanged); the set is not empty (no selection at all means every strength 0). The one-pair form `(profile_id, strength)` is the schema version 1 case. A model integration resolves the selection to the strength vector outside the graph (`InterventionManifest.ResolveStrengths`): each named profile gets its strength, every other profile 0. A selection that names a profile the loaded graph does not carry is rejected.
