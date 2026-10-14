# Proposed: GLiNER2 count-gate and label-description cases

Status: **proposed** (2026-10-14). Not accepted and not under `model-pipeline/protected/`. If the maintainer accepts it, the intended destination is `model-pipeline/protected/gliner2-base-v1/capability-cases/`, byte-identical. `CpuParityTests.CountGateAndDescriptionCasesMatchTheReference` (and the same test in `Starlings.Models.Gliner2.Tests.Cuda`) reads the protected copy when it exists and this proposal otherwise.

These cases cover two capabilities the accepted 58-case corpus leaves untested: `count_gate_closed` (the count gate never closed on that corpus) and `label_descriptions` (one development case).

## Files

| File | Contents |
| --- | --- |
| `corpus.jsonl` | 22 cases in the corpus format of `gliner2-extraction-corpus-v1.jsonl`, written by `model-pipeline/reference/gliner2/make_gliner2_capability_cases.py` |
| `extraction.jsonl` | The reference rows (format of the protected `extraction.jsonl`) from `run_gliner2_extraction.py` |
| `batch.json` | Upstream padded batches vs single calls: entities identical on all 22, confidence max abs 1.1e-6 |
| `environment.json` | Packages (gliner2 2.0.0 at `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`, torch 2.14.1 CPU, transformers 4.57.6), threads 1, asset digests |

## Cases

- **Count gate closed (12):** a paragraph with labels it does not contain (for example the Constitution's Article IV section 2 with `spacecraft`). GLiNER2's count head predicts 0, so upstream returns no entities. Margins between count logit 0 and the next logit run from 1.13 to 8.06. One of the 12 has a label description.
- **Label descriptions (11):** 10 paragraphs with descriptions for every label (one with descriptions for two of three labels), plus the gate-closed case above. All have open gates and between 1 and 13 entities.

Texts are paragraphs of two public-domain Project Gutenberg files pinned by SHA-256: Alice's Adventures in Wonderland (eBook #11) and the United States Constitution (eBook #5). They are the files of `model-pipeline/proposed/embeddinggemma-2/long-input/texts.json`, with whitespace collapsed. Labels and descriptions were written for this repository. The gate-closed pairs come from a scan of 230 paragraphs x 8 label sets with the pinned model, in which 586 of 1,840 gates closed. The reference run records whether each gate closes; the case file does not assume it.

## Result at the time of the proposal

C# (`Gliner2EntityExtractor`, ORT 1.30.0 CPU), judged with the protected `onnx-fp32-cpu` tolerances. Inputs were exact on 22 of 22 cases, the gate matched on 22 of 22, and entities were exact on 22 of 22. Count logits max abs was 4.2e-5, span-logit max relative difference 1.7e-5, confidence max abs 2.3e-6. Padded batches matched single runs. Details are in `docs/evidence/capability-closure-2026-10-14/`.
