# Tokenizer fixtures report: multilingual-e5-small/onnx-fp32-cpu

Date 2026-10-14. Evaluator model-pipeline evaluate --tokenizer-fixtures at `b469654b17262920457e810af410063b13cce8ea`. Exit code 0.

Fixtures: `model-pipeline/protected/multilingual-e5-small/release/onnx-fp32-release/tokenizer.jsonl` (232 rows, format `encoding`, SHA-256 `0469daa84f5225c4b11b63f78ba85ba0b5f7d2f3b48f96c124fafe24a868af1b`). Candidate tokenizer SHA-256 `0b44a9d7b51c3c62626640cda0e2c2f70fdacdc25bbbd68038369d14ebdf4c39`; producer Starlings.Fidelity.Tool tokenizer `b469654b17262920457e810af410063b13cce8ea`. Manifest offsets unit: `none`.

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

Status **verified** (merge base of HEAD and origin/main; base `b469654b17262920457e810af410063b13cce8ea`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/multilingual-e5-small/onnx-fp32-cpu/tolerances.yaml` | `d6a18d85059f0401b2449d7a01d91b9f8b93ce5098639aee211e03ddd6358e12` |
| `model-pipeline/protected/multilingual-e5-small/release/onnx-fp32-release/tokenizer.jsonl` | `0469daa84f5225c4b11b63f78ba85ba0b5f7d2f3b48f96c124fafe24a868af1b` |
