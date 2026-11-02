# Proposed: GLiNER2 classification cases

Status: **proposed** (2026-11-01, issue #183 step 1). Not accepted and not under `model-pipeline/protected/`. If the maintainer accepts it, the intended destination is `model-pipeline/protected/gliner2-base-v1/classification/`, byte-identical. `ClassificationParityTests.ClassificationMatchesTheReference` (`ai-dotnet/tests/Starlings.Models.Gliner2.Tests/ClassificationTests.cs`) reads this proposal.

These rows are parity fixtures for the capability `classification`. They are not a task-quality set (G3). A G3 set for classification belongs to the maintainer.

## Files

| File | Contents |
| --- | --- |
| `corpus.jsonl` | 26 cases: id, task, labels, multi_label, cls_threshold, text |
| `classification.jsonl` | The reference rows from `model-pipeline/reference/gliner2/run_gliner2_classification.py`: input ids, schema positions (`[P]`, then one `[L]` per label), classifier logits, probabilities and GLiNER2's public output. `ort_logits` is the same head applied to the locked ONNX graph's `last_hidden_state` with ONNX Runtime CPU (diagnostic; the C# test does not read it) |
| `environment.json` | Packages (gliner2 2.0.0 at `55656fbfa01d3d4a77485e1a1eeeaf682990ccdf`, torch 2.14.1 CPU, transformers 4.57.6, onnxruntime 1.30.0), threads 1, asset digests, and the ORT-vs-PyTorch summary |

## Cases

The texts, task names and labels were written for this repository; no third-party text is used.

- **Single-label (19):** sentiment with three labels (positive, negative, neutral) and with two, including a mixed review; topic with five labels; support intent with five multi-word labels; language, urgency (with `°C` and an em dash) and spam detection with two labels.
- **Multi-label (7):** topics and emotions with five labels. One case selects nothing above 0.5, so GLiNER2 falls back to the best label (`multi-topic-3`). One case uses `cls_threshold` 0.8.

The reference is GLiNER2's public API, `GLiNER2.classify_text(text, {task: {"labels": labels, "multi_label": multi_label, "cls_threshold": cls_threshold}}, include_confidence=True)`, on CPU in fp32. The input ids are captured at the encoder and the logits at the `classifier` module.

## Reproduce

```sh
python model-pipeline/reference/gliner2/run_gliner2_classification.py \
    --model-dir <fastino/gliner2-base-v1 snapshot at f9634218> \
    --corpus model-pipeline/proposed/gliner2-base-v1/classification/corpus.jsonl \
    --out <dir> --onnx <onnx/gliner2-entities-fp32.onnx>
```

Run it in the environment of `model-pipeline/reference/gliner2/requirements.txt`, with `model-pipeline/src` on `PYTHONPATH`.

## Result at the time of the proposal

The CPU run was on an Intel i7-5820K with ONNX Runtime 1.30.0. C# `Gliner2TextClassifier` matched on every case. Input ids and schema positions were exact on 26 of 26 cases, and decisions (the selected labels) were exact on 26 of 26. Logits had a maximum relative difference of 2.74e-5, against the profile's span-logit bound of 1e-4. Scores had a maximum absolute difference of 8.46e-6, against the profile's score bound of 2e-5. The Python ORT route gave the same figures (2.75e-5 and 8.46e-6). Details are in `docs/evidence/gliner2-classification-2026-11-01/`.
