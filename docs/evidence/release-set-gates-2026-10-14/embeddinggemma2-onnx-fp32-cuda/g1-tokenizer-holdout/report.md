# Tokenizer fixtures report: embeddinggemma-2/onnx-fp32-cuda

Date 2026-10-15. Evaluator model-pipeline evaluate --tokenizer-fixtures at `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023`. Exit code 0.

Fixtures: `model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release/tokenizer.jsonl` (232 rows, format `encoding`, SHA-256 `25a991f55885ce1fc686e7daf23bb62e727f6cad0a0f3b2a7546de0b4abe2954`). Candidate tokenizer SHA-256 `4d777ef5bdc1aa36227abdfb77c3e49e7b9c892d16e1b6bda41c393504828be4`; producer Starlings.Fidelity.Tool tokenizer `0eaaf9f943412c96f738c9e9c7e59192f2910845`. Manifest offsets unit: `none`.

## Gate

| Gate | Status | Reason |
| --- | --- | --- |
| G1 | passed | every bound field is exact on every fixture row |

## Checks

Exact comparison per row. Binding fields are the protected tolerances' `discrete` keys; the others are measured and never gate.

| Check | Tolerance key | Binding | Rows compared | Mismatches | Status |
| --- | --- | --- | ---: | ---: | --- |
| `identity` |  | yes | 0 | 0 | passed |
| `cases` |  | yes | 232 | 0 | passed |
| `ids` | discrete.token_ids | yes | 232 | 0 | passed |
| `attention_mask` | discrete.attention_mask | yes | 232 | 0 | passed |
| `tokens` |  | no | 232 | 0 | passed |
| `type_ids` |  | no | 232 | 0 | passed |
| `special_tokens_mask` |  | no | 232 | 0 | passed |
| `decode.skip_special_tokens` |  | no | 232 | 0 | passed |
| `decode.keep_special_tokens` |  | no | 232 | 0 | passed |
| `offsets.unicode-scalars` |  | no | 232 | 0 | passed |
| `offsets.utf16-code-units` |  | no | 232 | 0 | passed |
| `offsets.utf8-bytes` |  | no | 232 | 0 | passed |

Details:

- `offsets.unicode-scalars`: behavior.offsets_unit is none: the profile exposes no offsets; discrete.offsets does not apply
- `offsets.utf16-code-units`: behavior.offsets_unit is none: the profile exposes no offsets; discrete.offsets does not apply
- `offsets.utf8-bytes`: behavior.offsets_unit is none: the profile exposes no offsets; discrete.offsets does not apply

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `75574190e77f5528a416e87c6e12ea17e3b85d33`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/embeddinggemma-2/onnx-fp32-cuda/tolerances.yaml` | `1bd9c65a257d3ff992fabce9c820ff634dc88f2506bd68bb5c50d3017d7beba0` |
| `model-pipeline/protected/embeddinggemma-2/release/onnx-fp32-release/tokenizer.jsonl` | `25a991f55885ce1fc686e7daf23bb62e727f6cad0a0f3b2a7546de0b4abe2954` |
