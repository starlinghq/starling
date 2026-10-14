# Proposed: GLiNER2 extraction release fixture (P-030), WNUT-17 sentences

Status: **proposed**. This fixture is not accepted and is not under `model-pipeline/protected/`. Intended use: the held-out G1/G2 extraction release fixture for `gliner2-base-v1/onnx-fp32-cpu` and `onnx-fp32-cuda`. If accepted, it moves byte-identical next to the development `extraction.jsonl` in a `protected-change` pull request. No .NET run has used these files.

## Input

These are the 30 WNUT-17 rows (`category: wnut17`) of the Stable proposal's extraction release corpus `gliner2-base-v1/ner/extraction-release-v1.jsonl` (PR #20, SHA-256 `53858dceec30b4c0f66915c8b9d50843420e0956d15f3065843ee6cbcf95a3ea`). Every sentence is verbatim WNUT-17 test text (github.com/leondz/emerging_entities_17 at `e52a2d2a71ac1a051ca2d532eef28653c5c02603`, CC BY 4.0, Derczynski et al.), selected by `reference/stable/select_ner.py`. No agent wrote any of them. The labels are the six WNUT-17 class names, at threshold 0.5. The 30 Few-NERD rows of that corpus are left out because Few-NERD is CC BY-SA 4.0.

## Files

| File | SHA-256 | Content |
| --- | --- | --- |
| `extraction.jsonl` | `dac63a49413cd0fe382b7cecebaa51733e5027ce2589e1329f7735e6974389ec` | 30 rows: public output, model inputs, count and span logits (valid region), threshold margins, stage tensor digests |
| `batch.json` | `065d6810cc48394b755274e6f8b2a5775187786b0d170d931af593cd710103ed` | padded batches of 8 vs single calls |
| `environment.json` | `825d23a456b768a7e7fbd1dc64670e7956eab4696969d95d97326bf7b01ea8a3` | runner commit `b98c9ce`, packages, asset and corpus digests |

The stage tensors (`.npz`, one per row) are not committed. They regenerate from the command below.

## Reference

`reference/gliner2/run_gliner2_extraction.py` ran with the upstream GLiNER2 package (source `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`, 2.0.0), torch 2.14.1 CPU fp32, transformers 4.57.6 and 6 threads. It used `fastino/gliner2-base-v1` at `f9634218e53580c56edf0de97ca1a7d3f1c2354e`; each asset matched the SHA-256 recorded in the protected development `environment.json`.

```sh
cd model-pipeline/reference/gliner2
<gliner2 venv (requirements-export.txt)>/bin/python run_gliner2_extraction.py --model-dir <snapshot> \
  --revision f9634218e53580c56edf0de97ca1a7d3f1c2354e \
  --corpus ../../proposed/stable/gliner2-base-v1/ner/extraction-release-v1.jsonl --categories wnut17 \
  --out <out> --tensors-out <tensors> --threads 6
```

Findings:

- 30 rows, 56 entities. The count gate is open on every row.
- The pure-Python decoder equals the public output on all 30 rows, and the exportable wrapper gives the same entities.
- Wrapper vs upstream span logits: max abs 4.6e-5. Count logits are identical.
- Smallest threshold margin: 0.027.
- Batch of 8 vs single calls: same entities on every row; confidence max abs 1.3e-6.

None of these suggests a tolerance change.

## Decisions for the reviewer

1. Accept the WNUT-17 rows as the GLiNER2 extraction release set (it depends on accepting WNUT-17 in PR #20).
2. Move the three files to the protected profile directory.
