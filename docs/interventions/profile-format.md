# Intervention file formats (schema version 1)

Status: **experimental**. Normative for the C# contract (`Starlings.Interventions`) and for the Python graph compiler (`model-pipeline`). Design: [architecture.md](architecture.md).

V1 evidence uses synthetic random orthonormal directions only. The library ships the mechanism; which callers may apply which profile is the host application's job.

## Common rules

These apply to `profile.json` and `compiler-manifest.json`.

- UTF-8 without a byte order mark; standard JSON (no comments, no trailing commas, no `NaN` or `Infinity`); at most 16 MiB.
- **Strict**: a reader rejects an unknown property, a missing required property and a property that appears twice, in every object. A writer that needs a new field must raise the schema version and update the C# parser in the same change.
- `schema_version` is checked first. Any value other than the integer `1` is rejected, so a newer document fails on its version, not on a field it added.
- Patterns written as `^...$` below match the whole string (Python `re.fullmatch`): a trailing newline is not allowed.
- Integers are JSON numbers without a fraction or exponent (`1`, not `1.0`).
- **SHA-256** values are 64 lower-case hexadecimal characters over the exact bytes of a file.
- **Relative paths** use forward slashes, are not rooted, and have no empty, `.` or `..` segment, no backslash, no colon and no control character (at most 512 characters). A path resolved against its directory never leaves it.
- Closed vocabularies are case-sensitive. Unknown values are rejected:

| Field | V1 values |
| --- | --- |
| `operator` | `directional_projection` |
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
| `schema_version` | integer | yes | `1` |
| `id` | string | yes | `^[a-z0-9][a-z0-9._-]{0,127}$` |
| `description` | string | no | free text for people; omitted when absent |
| `base_model_sha256` | SHA-256 | yes | SHA-256 of the base `model.onnx` bytes, as the model package's asset lock records it |
| `operator` | string | yes | `directional_projection` |
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

Written by the graph compiler next to the derived graph; read and verified by `InterventionManifest`.

| Key | Type | Required | Rule |
| --- | --- | --- | --- |
| `schema_version` | integer | yes | `1` |
| `compiler_version` | string | yes | `^[A-Za-z0-9][A-Za-z0-9._+/-]{0,63}$` |
| `site_mapping_version` | string | yes | same pattern; the version of the model-specific site mapping |
| `base_artifacts` | array of base artifact | yes | exactly one with role `graph`; paths unique |
| `profile_id` | string | yes | the profile's `id` |
| `profile_sha256` | SHA-256 | yes | of the `profile.json` bytes the graph was compiled from |
| `operator` | string | yes | the profile's operator |
| `opset` | integer | yes | default-domain ONNX opset of the derived graph, 1 to 1,000 |
| `precision` | string | yes | `^[a-z0-9][a-z0-9_]{0,31}$`, the base graph's precision (for example `fp32`, `fp16`, `q4`) |
| `inputs` | array of input | yes | the derived graph's extra inputs beyond the base graph's; exactly one with role `strength`, holding one element (shape `[]` or `[1]`); names unique |
| `sites` | array of site binding | yes | one per profile site, 1 to 1,024; no two with the same `layer` and `target` |
| `derived_graph` | file | yes | the derived `model.onnx`, relative to the derived artifact directory |
| `derived_files` | array of file | yes | other derived files (for example a small external data file of the derived graph); paths unique and different from `derived_graph.path`; may be empty |

Base artifact: `role` (`graph` or `external_data`), `path` (relative to the base model directory), `sha256`, `size` (integer bytes, `>= 0`), in that order.

File: `path`, `sha256`, `size`.

Input: `name` (ONNX input name, non-empty, at most 1,024 characters, no control characters), `role` (`strength`), `dtype` (`float32`), `shape` (array of non-negative integers; `[]` is a scalar). A model integration accepts exactly the base graph's inputs plus these.

Site binding: `layer`, `target`, `tensor` (the base graph tensor the site mapping resolved the site to).

A reader verifies, before loading the derived graph:

1. `VerifyProfile(profile, sha)`: same `profile_id`, same `profile_sha256`, same operator, the base graph (`base_artifacts` entry with role `graph`) has the profile's `base_model_sha256`, and the sites (layer, target) equal the profile's.
2. `VerifyBaseModel(sha)`: the base graph is the one the application's asset lock expects.
3. `VerifyDerivedFiles(dir)`: size, then SHA-256 of `derived_graph` and every `derived_files` entry.
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
  "derived_files": []
}
```

## `InterventionSelection`

Not a file: the request-level choice `(profile_id, strength)`. `profile_id` follows the profile id rule; `strength` is any finite float32 (0 leaves outputs unchanged). A model integration rejects a selection whose profile id differs from the loaded derived graph's.
