# Proposed: G3 NER sets, reference scores and budgets, GLiNER2

Status: **proposed**. Not accepted and not under `model-pipeline/protected/`. If accepted, the chosen gold set moves under `model-pipeline/protected/` and a candidate from `budgets/` becomes `model-pipeline/protected/gliner2-base-v1/<profile>/task-quality.yaml`. `extraction-release-v1.jsonl` is a candidate input for a held-out G1/G2 extraction release fixture (P-030).

## Sets

R5 NER gold format (one JSON line per sentence: `id`, `text`, `labels`, `entities` with code-point offsets, end exclusive). Each directory has `gold.jsonl` and `dataset.json` (provenance, license, selection); the evaluator digest covers both.

| Set | Source and revision | License (evidence) | Sentences | Gold entities | Digest |
| --- | --- | --- | ---: | ---: | --- |
| `wnut17-test-v1/` (**primary**) | WNUT-17 `emerging.test.annotated`, github.com/leondz/emerging_entities_17 at `e52a2d2a71ac1a051ca2d532eef28653c5c02603` | CC-BY-4.0 (upstream README: "distributed under the CC-BY 4.0 license"; Hub card `leondz/wnut_17` at `be8fbd57…`: `cc-by-4.0`) | 300 | 262 | `c4322bab…9a55` |
| `fewnerd-test-v1/` (alternative) | Few-NERD supervised test, `DFKI-SLT/few-nerd` at `205f3e9c9f3577ea2561d43f2f62dc249ab92d5b` | CC-BY-SA-4.0 (card front matter and "Licensing Information"); **share-alike**: the committed `gold.jsonl` stays CC BY-SA 4.0 | 300 | 694 | `ff30dce2…1fd9e` |

WNUT-17 is user-generated text (YouTube comments, Stack Overflow responses, Twitter, Reddit comments, per its README) published by its authors under CC BY 4.0; the maintainer may still want to note that platform terms applied to the original posts. Few-NERD is Wikipedia text.

Selection (`reference/stable/select_ner.py`, seed 20261014): sentences of 5 to 60 tokens, one seeded shuffle, first 300, kept in source order; sentences without entities kept. Text is the source tokens joined by single spaces. WNUT-17 BIO tags; Few-NERD IO tags (a maximal run of one type is one entity, the dataset's convention), coarse type `other` dropped from gold and not queried. Labels are the class names verbatim (`creative-work` keeps its hyphen); they were not paraphrased for the model.

Rejected NER sources (licenses checked 2026-10-14): CoNLL-2003 (`other`, Reuters research agreement), OntoNotes 5 (LDC), WikiNEuRal and MultiNERD (CC-BY-NC-SA-4.0), CrossNER (no license stated on the Hub card).

## Reference scores (`reference-scores/`)

`reference/stable/reference_ner.py`: the upstream GLiNER2 package (source `55656fbf`, `reference/gliner2/requirements-export.txt`) on PyTorch CPU fp32, `GLiNER2.extract(text, Schema().entities(labels), threshold=0.5, include_confidence=True, include_spans=True)`, the default operating point that `Starlings.Fidelity.Tool ner` mirrors. Scored with the evaluator's own `ner_metrics`. Every predicted span's offsets index its text (0 mismatches). About 45 s per set.

| Set | exact micro P / R / F1 | partial micro F1 | exact macro F1 |
| --- | --- | ---: | ---: |
| WNUT-17 | 0.260 / 0.599 / **0.3626** | 0.4134 | 0.3206 |
| Few-NERD | 0.431 / 0.710 / **0.5365** | 0.6137 | 0.4061 |

Per label (exact F1), WNUT-17: person 0.494, location 0.512, corporation 0.321, group 0.288, creative-work 0.203, product 0.105. Few-NERD: person 0.677, location 0.658, organization 0.561, building 0.348, product 0.278, event 0.195, art 0.125. These are zero-shot scores at the default threshold with the datasets' own label names (low precision, high recall); they measure faithfulness of the port, not the best attainable NER quality.

```sh
<gliner2 venv>/bin/python model-pipeline/reference/stable/reference_ner.py --model-dir <fastino/gliner2-base-v1@f9634218 snapshot> \
  --gold <set>/gold.jsonl --predictions <out>.jsonl --scores <out>.json --threads 6
```

## Budgets (`budgets/`)

GLiNER2 `onnx-fp32-cpu` and `onnx-fp32-cuda`: WNUT-17 `exact.micro.f1 >= 0.352`, `partial.micro.f1 >= 0.403` (reference minus 0.01). The protected entity rule already allows threshold flips within 2e-5 of the threshold and order swaps; on this set one extra false positive moves exact micro F1 by about 0.0004 and one lost true positive by about 0.0023, so 0.01 allows a handful of flips, and anything larger is a real fidelity defect. The Few-NERD alternative is in the same file as comments (`exact >= 0.526`, `partial >= 0.603`).

## Extraction release corpus

`extraction-release-v1.jsonl`: the first 30 rows of each set (60 rows) in the format of `reference/corpus/gliner2-extraction-corpus-v1.jsonl` (threshold 0.5). Fixture commands (not run here):

```sh
cd model-pipeline/reference/gliner2
<gliner2-export venv>/bin/python run_gliner2_extraction.py --model-dir <snapshot> --revision f9634218e53580c56edf0de97ca1a7d3f1c2354e \
  --corpus ../../proposed/stable/gliner2-base-v1/ner/extraction-release-v1.jsonl --out <out> --tensors-out <tensors>
```

## Decisions for the reviewer

1. Primary set: WNUT-17 (CC BY) as proposed, or Few-NERD (CC BY-SA, cleaner text, more entities).
2. The 0.01 margin.
3. Accept the extraction release corpus as P-030 input for GLiNER2 G1/G2.
