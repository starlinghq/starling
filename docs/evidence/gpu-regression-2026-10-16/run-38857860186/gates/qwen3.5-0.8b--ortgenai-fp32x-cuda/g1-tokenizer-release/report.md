# Tokenizer fixtures report: qwen3.5-0.8b/ortgenai-fp32x-cuda

Date 2026-10-16. Evaluator model-pipeline evaluate --tokenizer-fixtures at `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8`. Exit code 0.

Fixtures: `model-pipeline/protected/qwen3.5-0.8b/release/tokenizer-release/tokenizer.jsonl` (232 rows, format `encoding`, SHA-256 `2dd97c3887638e3f1b1197f73a45d4a0d196b0fc4bde82b146f86b113646b0ff`). Candidate tokenizer SHA-256 `a5cd9732badce41de57e6efce8302930ded1c1188c5f81feb2bd6c24c4a1941f`; producer Starlings.Fidelity.Tool tokenizer `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8`. Manifest offsets unit: `none`.

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
| `ids` | discrete.prompt_ids | yes | 232 | 0 | passed |
| `attention_mask` |  | no | 232 | 0 | passed |
| `tokens` |  | no | 232 | 0 | passed |
| `type_ids` |  | no | 232 | 0 | passed |
| `special_tokens_mask` |  | no | 232 | 0 | passed |
| `decode.skip_special_tokens` |  | no | 232 | 0 | passed |
| `decode.keep_special_tokens` |  | no | 232 | 0 | passed |
| `offsets.unicode-scalars` |  | no | 232 | 0 | passed |
| `offsets.utf16-code-units` |  | no | 232 | 0 | passed |
| `offsets.utf8-bytes` |  | no | 232 | 0 | passed |

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `9a18b019a4d8b280ced291e4b7cd5dc4bee73fe8`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/qwen3.5-0.8b/ortgenai-fp32x-cuda/tolerances.yaml` | `56c6025091e85914d8de84c80cf11957c6ead4bad5697db1e3e25448c30f6d04` |
| `model-pipeline/protected/qwen3.5-0.8b/release/tokenizer-release/tokenizer.jsonl` | `2dd97c3887638e3f1b1197f73a45d4a0d196b0fc4bde82b146f86b113646b0ff` |
