# Tokenizer fixtures report: gliner2-base-v1/onnx-fp32-cpu

Date 2026-10-14. Evaluator model-pipeline evaluate --tokenizer-fixtures at `10c016c1e8724983306d79a5db02b84326109c32`. Exit code 0.

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

Status **verified** (merge base of HEAD and origin/main; base `10c016c1e8724983306d79a5db02b84326109c32`).

| File | SHA-256 |
| --- | --- |
| `model-pipeline/protected/gliner2-base-v1/onnx-fp32-cpu/tolerances.yaml` | `522ffd8c4945d3b4d7d09368b8d6cab18e87044c4e6f8009c0a40aec91c67f85` |
| `model-pipeline/protected/gliner2-base-v1/release/tokenizer-release/gliner2-tokenizer.jsonl` | `90eb38666294ac12074583934a88c71597229770f177d5a4f2db1eb3efaa666e` |
