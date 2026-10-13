# Staged fidelity evaluation: embeddinggemma-2/onnx-fp32-cpu, stages 1 to 3

`report.md` and `report.json` are the unedited output of

```sh
model-pipeline evaluate model-pipeline/registry/embeddinggemma-2/onnx-fp32-cpu/manifest.yaml --results <dir> --out <dir> --require-gates G1,G2
```

run on 2026-10-11 at `3901763` (evaluator from #38, tolerance comments reworded in #39; protected evidence verified against Git), re-run on the unchanged stage results from `48e9a18`, with the binding tolerances (`model-pipeline/protected/embeddinggemma-2/onnx-fp32-cpu/tolerances.yaml`) and the accepted development fixtures (PR #29: 46 corpus texts x no prompt, `query`, `document` = 138 cases; prompt rows are cases `<id>@<prompt>`). One workstation (Intel i7-5820K, Linux x64, ORT 1.30.0 CPU, default threads). Development evidence; the manifest's gate entries are left to the maintainer.

**Result: exit 0, first divergence none, G1 passed, G2 passed. Every .NET check passed.** The first run (evaluator at `48e9a18`) failed G2 on `export.token_embeddings` because it judged the bound per value; see [Token embeddings: per-case bound](#token-embeddings-per-case-bound).

## Stage results

| Stage | Producer |
| --- | --- |
| 1 tokenizer | `ai-dotnet/tools/Starlings.Fidelity.Tool -- embeddinggemma2` at `48e9a18`: `EmbeddingGemma2Tokenizer` (owned profile `gemma-bpe-embeddinggemma2`, Starlings.Tokenizers 0.1.0-preview.4) on prompt text + text |
| 2 export | `model-pipeline/reference/embeddinggemma2/stage_embeddinggemma2_export.py` at `8420e30` (spike `venv`, `PYTHONPATH=model-pipeline/src`): Python ORT on the release graph (`02cc7ecc...`), token embeddings and hidden states against `eg2_text.TextPath` on the same ids |
| 3 .NET | the same tool run: embeddings from fixture ids and from text with each row's prompt, the protected padded batch, all rows in batches of 8, one padded text batch per prompt |

## Numbers (138 cases)

| Check | Bound | Stage 2 (Python ORT) | Stage 3 (C# ids) | Stage 3 (C# text) |
| --- | ---: | ---: | ---: | ---: |
| Token ids, attention mask | exact | | stage 1: 138/138 exact | |
| Embedding max abs vs reference | 5e-6 | 4.2e-7 | 3.7e-7 | 3.7e-7 |
| Minimum cosine vs reference | 0.9999999 | 0.99999999999473 | 0.99999999999473 | 0.99999999999473 |
| Padded batches vs single rows | 5e-6 | 1.8e-7 | 1.8e-7 | 1.8e-7 |
| C# vs Python ORT, same machine and provider | 1e-6 | | 4.2e-7 | |
| Token embeddings, per case `max |d| / max(1, max |ref|)` | 1e-4 | 6.2e-6 | | |
| Token embeddings, per value `|d| / max(1, |ref|)` (report only) | | 1.7e-2 | | |
| Hidden state (512) max abs (report only: no protected key) | | 1.0e-3 | | |

## Token embeddings: per-case bound

The tolerance file's comment read "|d| <= 1e-4 * max(1, |ref|)", and the first evaluator (PR #37) judged that per value, which failed at 1.7e-2 (worst case `long-english@document`: token-embedding components near zero in a long row carry the fp32 error of neighbours near 3,900). The value 6.2e-6 the bound was set from (admission spike, `parity.json` `token_embeddings_max_abs_relative_to_max_1_ref`) was computed per case. On 2026-10-11 Mark decided the binding meaning is per case: for each case, `max_i |d_i| / max(1, max_i |ref_i|) <= 1e-4`. The evaluator now judges it that way (worst case 6.17e-6, `long-english@document`; max abs 0.0225) and keeps the per-value figure (0.0166) in the report as report only. The bound value is unchanged; the tolerance comments were reworded to the per-case definition in #39. The pooled embeddings, which the package returns, are within 4.2e-7 either way.
