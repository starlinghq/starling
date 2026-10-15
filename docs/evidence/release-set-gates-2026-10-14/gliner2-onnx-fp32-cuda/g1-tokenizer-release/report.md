# Tokenizer fixtures report: gliner2-base-v1/onnx-fp32-cuda

Date 2026-10-15. Evaluator model-pipeline evaluate --tokenizer-fixtures at `472e1876ba69b2c4c0f2a7956adfb5e6c9e0b023`. Exit code 0.

Fixtures: `model-pipeline/protected/gliner2-base-v1/release/tokenizer-release/gliner2-tokenizer.jsonl` (232 rows, format `whitespace_splitter`, SHA-256 `90eb38666294ac12074583934a88c71597229770f177d5a4f2db1eb3efaa666e`). Candidate tokenizer SHA-256 `1b7fbabfb4c690bed84c6793bfecae9b8dfe205751b04f9ffd1e76a1e7df9c16`; producer Starlings.Fidelity.Tool tokenizer `10c016c1e8724983306d79a5db02b84326109c32`. Manifest offsets unit: `unicode-scalars`.

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
| `normalized_text` | discrete.normalized_text | yes | 232 | 0 | passed |
| `words` | discrete.words | yes | 232 | 0 | passed |
| `word_offsets` | discrete.character_maps | yes | 232 | 0 | passed |
| `schema_tokens` |  | no | 232 | 0 | passed |
| `input_ids` | discrete.token_ids | yes | 232 | 0 | passed |
| `word_positions` | discrete.word_positions | yes | 232 | 0 | passed |
| `schema_positions` | discrete.schema_positions | yes | 232 | 0 | passed |

## Protected evidence

Status **verified** (merge base of HEAD and origin/main; base `75574190e77f5528a416e87c6e12ea17e3b85d33`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cuda/tolerances.yaml` | `6735d00f44f07ad7739a5e921f4496ded0921e515a3423e704c039b12c683957` |
| `model-pipeline/protected/gliner2-base-v1/release/tokenizer-release/gliner2-tokenizer.jsonl` | `90eb38666294ac12074583934a88c71597229770f177d5a4f2db1eb3efaa666e` |
